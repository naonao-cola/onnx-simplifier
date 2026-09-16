"""Tests for ``onnxsim.quantize_weight_only_nvfp4_cpp`` -- the C++-backed
port of ``onnxsim.quantize_weight_only_nvfp4`` (see
``onnxsim/passes/nvfp4_quantization.h``). Cross-checks the C++ port
against the pure-Python reference implementation on the same input, in
addition to structural checks on the two-level (global scale, per-block
E4M3-rounded scale) scheme this format uses -- see
``tests/test_mx_quantization_cpp.py`` for the sibling MXFP4 port's own
version of this same test shape.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.nvfp4_quantization import (
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    NVFP4_BLOCK_SIZE,
)

ort = pytest.importorskip("onnxruntime")


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _model(body, initializer=(), opset=13, ir_version=8):
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


def _matmul_model(K=64, N=16, weight=None, seed=0):
    if weight is None:
        rng = np.random.default_rng(seed)
        weight = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    return _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f32(weight, "W")],
    )


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-6)


def test_cpp_nvfp4_quantizes_matmul_with_standard_ops_only():
    model = _matmul_model(K=64, N=16, seed=0)
    q = onnxsim.quantize_weight_only_nvfp4_cpp(model)
    onnx.checker.check_model(q)

    op_types = {n.op_type for n in q.graph.node}
    assert op_types <= {"MatMul", "Cast", "Gather", "Reshape", "Mul"}
    assert all(n.domain in ("", "ai.onnx") for n in q.graph.node)


def test_cpp_nvfp4_block_scale_never_exceeds_e4m3_max_and_is_e4m3_representable():
    # The per-block *effective* scale is block_scale * global_scale, so it
    # isn't itself required to be an exact E4M3 grid point -- but
    # dividing back out global_scale must recover a value E4M3 can
    # represent exactly (round-tripping through the grid is a no-op).
    from onnxsim.nvfp4_quantization import _round_to_e4m3

    rng = np.random.default_rng(1)
    K, N = 64, 16
    weight = rng.standard_normal((K, N)).astype(np.float32) * 3.7
    model = _matmul_model(weight=weight)
    q = onnxsim.quantize_weight_only_nvfp4_cpp(model)

    scale_init = next(
        t
        for t in q.graph.initializer
        if t.data_type == onnx.TensorProto.FLOAT
        and list(t.dims) == [K // NVFP4_BLOCK_SIZE, N]
    )
    effective_scale = onnx.numpy_helper.to_array(scale_init).astype(np.float64).ravel()

    tensor_amax = max(float(np.abs(weight).max()), 1e-30)
    global_scale = tensor_amax / (FLOAT4_E2M1_MAX * FLOAT8_E4M3_MAX)
    block_scale = effective_scale / global_scale
    assert np.all(block_scale <= FLOAT8_E4M3_MAX + 1e-6)
    np.testing.assert_allclose(_round_to_e4m3(block_scale), block_scale, rtol=1e-5)


def test_cpp_nvfp4_output_stays_close_to_float_via_onnxruntime():
    model = _matmul_model(K=64, N=16, seed=3)
    q = onnxsim.quantize_weight_only_nvfp4_cpp(model)
    onnx.checker.check_model(q)

    rng = np.random.default_rng(4)
    x = rng.standard_normal((8, 64)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.3


def test_cpp_nvfp4_beats_mxfp4_on_the_same_weight():
    # NVFP4's finer per-block E4M3 scale (vs MXFP4's power-of-two-only
    # E8M0 scale) should reconstruct at least as well on the same input.
    rng = np.random.default_rng(8)
    weight = rng.standard_normal((64, 16)).astype(np.float32) * 0.9
    model = _matmul_model(weight=weight)

    q_nvfp4 = onnxsim.quantize_weight_only_nvfp4_cpp(model)
    q_mxfp4 = onnxsim.quantize_weight_only_mxfp4_cpp(model)

    rng2 = np.random.default_rng(9)
    x = rng2.standard_normal((4, 64)).astype(np.float32)
    (y_nvfp4,) = _run(q_nvfp4, {"X": x})
    (y_mxfp4,) = _run(q_mxfp4, {"X": x})
    (y_float,) = _run(model, {"X": x})
    assert _rel_l2(y_float, y_nvfp4) <= _rel_l2(y_float, y_mxfp4) + 1e-6


def test_cpp_nvfp4_gemm_transb():
    rng = np.random.default_rng(5)
    K, N = 128, 12
    weight = rng.standard_normal((N, K)).astype(np.float32) * 0.5
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm<transB = 1>(X, W)
        }}
        """,
        initializer=[_f32(weight, "W")],
    )
    q = onnxsim.quantize_weight_only_nvfp4_cpp(model)
    onnx.checker.check_model(q)

    x = rng.standard_normal((4, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert _rel_l2(float_y, q_y) < 0.3


def test_cpp_nvfp4_skips_non_block_divisible_k():
    model = _matmul_model(K=24, N=8, seed=7)  # 24 is not a multiple of 16
    q = onnxsim.quantize_weight_only_nvfp4_cpp(model)
    assert q.SerializeToString() == model.SerializeToString()


def test_cpp_nvfp4_skips_non_constant_weight():
    model = _model(
        """
        g (float[4,64] X, float[64,4] W) => (float[4,4] Y)
        {
          Y = MatMul(X, W)
        }
        """
    )
    q = onnxsim.quantize_weight_only_nvfp4_cpp(model)
    assert q.SerializeToString() == model.SerializeToString()


def test_cpp_nvfp4_matches_python_reference_output():
    # The C++ port and the pure-Python reference implement the same
    # two-level-scale NVFP4 algorithm (round-to-nearest onto the same
    # fixed E2M1 codebook, the same E4M3-rounded per-block scale relative
    # to the same per-tensor global scale) -- on the same weight they
    # should produce numerically equivalent dequantized outputs, not just
    # separately "close to float32".
    rng = np.random.default_rng(9)
    weight = rng.standard_normal((64, 16)).astype(np.float32) * 0.7
    model = _matmul_model(weight=weight)

    q_py = onnxsim.quantize_weight_only_nvfp4(model, block_size=NVFP4_BLOCK_SIZE)
    q_cpp = onnxsim.quantize_weight_only_nvfp4_cpp(model)
    onnx.checker.check_model(q_py)
    onnx.checker.check_model(q_cpp)

    x = rng.standard_normal((4, 64)).astype(np.float32)
    (y_py,) = _run(q_py, {"X": x})
    (y_cpp,) = _run(q_cpp, {"X": x})
    assert np.allclose(y_py, y_cpp, rtol=1e-4, atol=1e-4)
