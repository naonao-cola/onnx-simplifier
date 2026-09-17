"""Tests for ``onnxsim.apply_llm_fp4_activation_quantization_cpp`` and
``onnxsim.apply_llm_fp4_activation_quantization_per_tensor_cpp`` -- the
C++-backed ports of ``onnxsim.apply_llm_fp4_activation_quantization``
(data-free, per-token) and
``onnxsim.apply_llm_fp4_activation_quantization_per_tensor`` (calibrated,
per-tensor), see ``onnxsim/llm_fp4_activation_entry.h``. Both act only on
layers already weight-quantized by ``onnxsim.quantize_weight_only_llm_fp4``
(itself already a verified C++ port, see ``tests/test_llm_fp4_cpp.py``).

Neither pass has any RNG in its own quantize/dequantize construction (a
closed-form node splice reusing a fixed, already-baked codebook; the
per-tensor variant's own scale search is likewise closed-form for
calibration data under this file's own small test sizes, well under the
2**18-element subsampling cap the header documents), so exact agreement
against the pure-Python reference is expected and checked directly via
onnxruntime.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.llm_fp4 import (
    apply_llm_fp4_activation_quantization,
    apply_llm_fp4_activation_quantization_per_tensor,
    quantize_weight_only_llm_fp4,
)

ort = pytest.importorskip("onnxruntime")


def _model(body, initializer=(), opset=18, ir_version=9):
    model = parser.parse_model(
        f"""
        <
          ir_version: {ir_version},
          opset_import: ["": {opset}]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model


def _f32(array, name):
    return onnx.numpy_helper.from_array(np.asarray(array, dtype=np.float32), name)


def _weight_quantized_matmul(K=64, N=16, seed=0, opset=18):
    rng = np.random.default_rng(seed)
    w = (rng.standard_normal((K, N)) * 0.5).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f32(w, "W")],
        opset=opset,
    )
    return quantize_weight_only_llm_fp4(model)


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _calibration(K, num_samples=8, seed=1):
    rng = np.random.default_rng(seed)
    return [{"X": rng.standard_normal((4, K)).astype(np.float32)}]


# --- Data-free, per-token variant -------------------------------------


def test_per_token_cpp_matches_python_exactly():
    K, N = 64, 16
    q = _weight_quantized_matmul(K, N, seed=0)
    py = apply_llm_fp4_activation_quantization(q)
    cpp = onnxsim.apply_llm_fp4_activation_quantization_cpp(q)
    onnx.checker.check_model(cpp)

    rng = np.random.default_rng(9)
    x = rng.standard_normal((4, K)).astype(np.float32)
    (y_py,) = _run(py, {"X": x})
    (y_cpp,) = _run(cpp, {"X": x})
    np.testing.assert_allclose(y_cpp, y_py, rtol=1e-5, atol=1e-6)


def test_per_token_cpp_inserts_expected_node_chain():
    K, N = 64, 16
    q = _weight_quantized_matmul(K, N, seed=2)
    cpp = onnxsim.apply_llm_fp4_activation_quantization_cpp(q)
    onnx.checker.check_model(cpp)

    op_types = [n.op_type for n in cpp.graph.node]
    # 11 inserted nodes (Abs, ReduceMax, Max, Div, Div, Unsqueeze, Sub,
    # Abs, ArgMin, Gather, Mul) plus the original MatMul and the 6 nodes
    # the weight-quantization pass itself inserted (Cast, Gather, Reshape
    # x3, Mul).
    assert op_types.count("Abs") == 2
    assert op_types.count("ArgMin") == 1
    assert op_types.count("MatMul") == 1
    assert "Unsqueeze" in op_types


def test_per_token_cpp_epsilon_floors_all_zero_token():
    K, N = 32, 8
    q = _weight_quantized_matmul(K, N, seed=3)
    cpp = onnxsim.apply_llm_fp4_activation_quantization_cpp(q, epsilon=1e-6)
    onnx.checker.check_model(cpp)
    x = np.zeros((2, K), dtype=np.float32)
    (y,) = _run(cpp, {"X": x})
    assert np.all(np.isfinite(y))


def test_per_token_cpp_noop_below_min_opset():
    K, N = 32, 8
    q = _weight_quantized_matmul(K, N, seed=4, opset=13)
    result = onnxsim.apply_llm_fp4_activation_quantization_cpp(q)
    assert result.SerializeToString() == q.SerializeToString()


def test_per_token_cpp_noop_without_weight_quantization():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_llm_fp4_activation_quantization_cpp(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_per_token_cpp_gemm_transb():
    K, N = 48, 8
    rng = np.random.default_rng(6)
    w = (rng.standard_normal((N, K)) * 0.5).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm<transB = 1>(X, W)
        }}
        """,
        initializer=[_f32(w, "W")],
    )
    q = quantize_weight_only_llm_fp4(model)
    py = apply_llm_fp4_activation_quantization(q)
    cpp = onnxsim.apply_llm_fp4_activation_quantization_cpp(q)
    onnx.checker.check_model(cpp)

    x = rng.standard_normal((4, K)).astype(np.float32)
    (y_py,) = _run(py, {"X": x})
    (y_cpp,) = _run(cpp, {"X": x})
    np.testing.assert_allclose(y_cpp, y_py, rtol=1e-5, atol=1e-6)


# --- Calibrated, per-tensor variant -------------------------------------


def test_per_tensor_cpp_matches_python_exactly():
    K, N = 64, 16
    q = _weight_quantized_matmul(K, N, seed=10)
    cal = _calibration(K, seed=11)
    py = apply_llm_fp4_activation_quantization_per_tensor(q, calibration_data=cal)
    cpp = onnxsim.apply_llm_fp4_activation_quantization_per_tensor_cpp(
        q, calibration_data=cal
    )
    onnx.checker.check_model(cpp)

    rng = np.random.default_rng(12)
    x = rng.standard_normal((4, K)).astype(np.float32)
    (y_py,) = _run(py, {"X": x})
    (y_cpp,) = _run(cpp, {"X": x})
    np.testing.assert_allclose(y_cpp, y_py, rtol=1e-5, atol=1e-6)


def test_per_tensor_cpp_inserts_fewer_nodes_than_per_token():
    K, N = 64, 16
    q = _weight_quantized_matmul(K, N, seed=13)
    cal = _calibration(K, seed=14)
    cpp_pt = onnxsim.apply_llm_fp4_activation_quantization_per_tensor_cpp(
        q, calibration_data=cal
    )
    cpp_tok = onnxsim.apply_llm_fp4_activation_quantization_cpp(q)
    onnx.checker.check_model(cpp_pt)
    # Per-tensor has no Abs/ReduceMax/Max range-reduction node at all
    # (constant scale) -- strictly fewer nodes than the per-token pass.
    assert len(cpp_pt.graph.node) < len(cpp_tok.graph.node)
    assert "ReduceMax" not in [n.op_type for n in cpp_pt.graph.node]

    # The scale is baked in as a plain scalar float32 initializer.
    scale_inits = [
        t
        for t in cpp_pt.graph.initializer
        if t.name.endswith("_scale") and list(t.dims) == []
    ]
    assert len(scale_inits) >= 1


def test_per_tensor_cpp_custom_clip_ratios_matches_python():
    K, N = 64, 8
    q = _weight_quantized_matmul(K, N, seed=15)
    cal = _calibration(K, seed=16)
    ratios = [0.6, 0.75, 0.9, 1.0]
    py = apply_llm_fp4_activation_quantization_per_tensor(
        q, calibration_data=cal, clip_ratios=ratios
    )
    cpp = onnxsim.apply_llm_fp4_activation_quantization_per_tensor_cpp(
        q, calibration_data=cal, clip_ratios=ratios
    )
    onnx.checker.check_model(cpp)

    rng = np.random.default_rng(17)
    x = rng.standard_normal((4, K)).astype(np.float32)
    (y_py,) = _run(py, {"X": x})
    (y_cpp,) = _run(cpp, {"X": x})
    np.testing.assert_allclose(y_cpp, y_py, rtol=1e-5, atol=1e-6)


def test_per_tensor_cpp_missing_calibration_input_raises():
    K, N = 32, 8
    q = _weight_quantized_matmul(K, N, seed=18)
    bad_cal = [{"wrong_name": np.zeros((2, K), dtype=np.float32)}]
    with pytest.raises(ValueError):
        onnxsim.apply_llm_fp4_activation_quantization_per_tensor_cpp(
            q, calibration_data=bad_cal
        )


def test_per_tensor_cpp_noop_below_min_opset():
    K, N = 32, 8
    q = _weight_quantized_matmul(K, N, seed=19, opset=12)
    cal = _calibration(K, seed=20)
    result = onnxsim.apply_llm_fp4_activation_quantization_per_tensor_cpp(
        q, calibration_data=cal
    )
    assert result.SerializeToString() == q.SerializeToString()


def test_per_tensor_cpp_noop_without_weight_quantization():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    cal = [{"X": np.zeros((2, 4), dtype=np.float32)}]
    result = onnxsim.apply_llm_fp4_activation_quantization_per_tensor_cpp(
        model, calibration_data=cal
    )
    assert result.SerializeToString() == model.SerializeToString()
