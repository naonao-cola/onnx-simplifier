"""Tests for ``onnxsim.quantize_weight_only_pb_llm_cpp`` -- the C++-backed
port of ``onnxsim.quantize_weight_only_pb_llm`` (PB-LLM's structured
mixed-precision binarizer, see ``onnxsim/pb_llm_entry.h``). Like
``test_gptq_cpp.py``/``test_llm_int8_cpp.py``, this runs the model over
real calibration data through a real ``onnxruntime``-backed executor --
never a fake/mock executor -- and checks exact (bit-for-bit) agreement
against the pure-Python reference: unlike GPTQ, this technique has no
Hessian *inversion* at all (just a diagonal sum-of-squares statistic), so
there is even less floating-point-order risk than GPTQ's own already-
exact-in-practice Cholesky solve; any divergence here is a bug, not an
accepted tolerance.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

from onnxsim.onnx_simplifier import quantize_weight_only_pb_llm_cpp
from onnxsim.pb_llm import quantize_weight_only_pb_llm

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


def _matmul_model(K=32, N=8, seed=0):
    rng = np.random.default_rng(seed)
    weight = (rng.standard_normal((K, N)) * 0.5).astype(np.float32)
    return _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f32(weight, "W")],
    )


def _calibration(K=32, num_samples=24, seed=1):
    # Deliberately uneven per-channel energy (a few channels scaled way
    # up) so diag(H) -- and therefore the salient/non-salient split --
    # isn't a coin flip between near-identical columns.
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((num_samples, K)).astype(np.float32)
    x[:, :3] *= 8.0
    return [{"X": x}]


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)[0]


def _assert_exact_parity(model, calibration_data, **kwargs):
    py = quantize_weight_only_pb_llm(model, calibration_data, **kwargs)
    cpp = quantize_weight_only_pb_llm_cpp(model, calibration_data, **kwargs)
    py_nodes = sorted(
        (n.op_type, tuple(n.input), tuple(n.output)) for n in py.graph.node
    )
    cpp_nodes = sorted(
        (n.op_type, tuple(n.input), tuple(n.output)) for n in cpp.graph.node
    )
    assert py_nodes == cpp_nodes
    py_inits = sorted(py.graph.initializer, key=lambda t: t.name)
    cpp_inits = sorted(cpp.graph.initializer, key=lambda t: t.name)
    assert [t.name for t in py_inits] == [t.name for t in cpp_inits]
    for a, b in zip(py_inits, cpp_inits):
        assert a.data_type == b.data_type, a.name
        assert list(a.dims) == list(b.dims), a.name
        assert a.raw_data == b.raw_data, a.name
    return cpp


def _codes_and_scale(model):
    code_t = next(t for t in model.graph.initializer if t.name.endswith("_code"))
    scale_t = next(t for t in model.graph.initializer if t.name.endswith("_scale"))
    return (
        onnx.numpy_helper.to_array(code_t),
        onnx.numpy_helper.to_array(scale_t),
    )


def test_cpp_matches_python_exactly():
    _assert_exact_parity(_matmul_model(), _calibration())


def test_cpp_matches_python_across_salient_ratios():
    model = _matmul_model(K=40, N=6, seed=2)
    cals = _calibration(K=40, seed=3)
    for ratio in (0.0, 0.15, 0.5, 1.0):
        _assert_exact_parity(model, cals, salient_ratio=ratio)


def test_cpp_gemm_transb():
    K, N = 24, 5
    rng = np.random.default_rng(4)
    weight = (rng.standard_normal((N, K)) * 0.5).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm<transB = 1>(X, W)
        }}
        """,
        initializer=[_f32(weight, "W")],
    )
    _assert_exact_parity(model, _calibration(K=K, seed=5))


def test_cpp_biased_gemm():
    K, N = 24, 6
    rng = np.random.default_rng(6)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm(X, W, B)
        }}
        """,
        initializer=[
            _f32(rng.standard_normal((K, N)).astype(np.float32), "W"),
            _f32(rng.standard_normal((N,)).astype(np.float32), "B"),
        ],
    )
    cpp = _assert_exact_parity(model, _calibration(K=K, seed=7))
    # Bias is untouched -- still feeds the (now-dequantized-weight)
    # MatMul/Gemm node directly, not folded into anything new.
    assert any(n.op_type == "Gemm" and "B" in n.input for n in cpp.graph.node)


def test_cpp_multi_batch_and_3d_activation():
    K, N = 20, 4
    rng = np.random.default_rng(8)
    model3d = _model(
        f"""
        g (float[batch,seq,{K}] X) => (float[batch,seq,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f32(rng.standard_normal((K, N)).astype(np.float32), "W")],
    )
    cals3d = [
        {"X": rng.standard_normal((2, 5, K)).astype(np.float32)},
        {"X": rng.standard_normal((3, 4, K)).astype(np.float32)},
    ]
    _assert_exact_parity(model3d, cals3d)


def test_cpp_skips_non_float_weight():
    rng = np.random.default_rng(9)
    K, N = 16, 4
    weight = rng.standard_normal((K, N)).astype(np.float16)
    model = _model(
        f"""
        g (float16[batch,{K}] X) => (float16[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[onnx.numpy_helper.from_array(weight, "W")],
    )
    result = quantize_weight_only_pb_llm_cpp(
        model, [{"X": np.zeros((2, K), np.float16)}]
    )
    assert result.SerializeToString() == model.SerializeToString()


def test_cpp_skips_non_2d_weight():
    rng = np.random.default_rng(10)
    w = rng.standard_normal((2, 4, 4, 4)).astype(np.float32)
    model = _model(
        """
        g (float[1,2,8,8] X) => (float[1,2,5,5] Y)
        {
          Y = Conv(X, W)
        }
        """,
        [_f32(w, "W")],
    )
    result = quantize_weight_only_pb_llm_cpp(
        model, [{"X": rng.standard_normal((1, 2, 8, 8)).astype(np.float32)}]
    )
    assert result.SerializeToString() == model.SerializeToString()


def test_cpp_noop_when_no_matmul_gemm_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = quantize_weight_only_pb_llm_cpp(
        model, [{"X": np.zeros((4, 4), np.float32)}]
    )
    assert result.SerializeToString() == model.SerializeToString()


def test_cpp_salient_ratio_zero_binarizes_every_column():
    model = _matmul_model(K=24, N=4, seed=11)
    cpp = quantize_weight_only_pb_llm_cpp(
        model, _calibration(K=24, seed=12), salient_ratio=0.0
    )
    code, _scale = _codes_and_scale(cpp)
    assert set(np.unique(code).tolist()) <= {-1, 1}


def test_cpp_salient_ratio_one_int8_quantizes_every_column():
    model = _matmul_model(K=24, N=4, seed=13)
    cpp = quantize_weight_only_pb_llm_cpp(
        model, _calibration(K=24, seed=14), salient_ratio=1.0
    )
    code, _scale = _codes_and_scale(cpp)
    # A genuine INT8 code range in active use, not just {-1, 1} -- would
    # be a near-impossible coincidence for random float32 data quantized
    # to a 255-level grid.
    assert np.any(np.abs(code) > 1)


def test_cpp_quantized_model_runs_and_stays_finite():
    model = _matmul_model(K=32, N=8, seed=15)
    cpp = quantize_weight_only_pb_llm_cpp(model, _calibration(K=32, seed=16))
    onnx.checker.check_model(cpp)

    rng = np.random.default_rng(17)
    x = rng.standard_normal((4, 32)).astype(np.float32)
    float_y = _run(model, {"X": x})
    quant_y = _run(cpp, {"X": x})
    assert np.all(np.isfinite(quant_y))
    assert quant_y.shape == float_y.shape
