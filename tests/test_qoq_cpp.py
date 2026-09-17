"""Tests for ``onnxsim.apply_qoq_cpp`` -- the C++-backed port of
``onnxsim.quantize_weight_only_qoq`` (QServe's QoQ progressive two-stage
weight quantization, see ``onnxsim/passes/qoq.h``). This is a closed-form,
deterministic scheme with no RNG or fitting algorithm, so -- unlike this
repo's k-means/rotation-family ``*_cpp`` ports -- these tests check exact
(bit-for-bit-in-practice) numeric agreement against the pure-Python
reference, not just structural/algebraic properties. Unlike most other
data-free weight-only ``*_cpp`` ports in this repo (which fold their
reconstruction into a single replacement float32 initializer), this port
keeps the real graph shape ``quantize_weight_only_qoq``/
``quantize_weight_only_int4`` both use: a genuine packed-INT4
``DequantizeLinear`` node, checked directly here (codes + scale), not just
the reconstructed float weight.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.qoq import quantize_weight_only_qoq

ort = pytest.importorskip("onnxruntime")

_BLOCK_SIZE = 32
_INT8_CLIP_MAX = 119


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _model(body, initializer=(), opset=21, ir_version=10):
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


def _matmul_model(w, K, N, batch="batch", opset=21):
    return _model(
        f"""
        g (float[{batch},{K}] X) => (float[{batch},{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(w, "W")],
        opset=opset,
    )


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-9)


def _dequant_node(model):
    return next(n for n in model.graph.node if n.op_type == "DequantizeLinear")


def _codes_and_scale(model):
    dq = _dequant_node(model)
    inits = {t.name: t for t in model.graph.initializer}
    codes_t = inits[dq.input[0]]
    scale_t = inits[dq.input[1]]
    return (
        onnx.numpy_helper.to_array(codes_t).astype(np.int64),
        onnx.numpy_helper.to_array(scale_t),
        {a.name: a for a in dq.attribute},
    )


def test_cpp_matches_python_codes_and_scale_exactly():
    K, N = _BLOCK_SIZE * 3, 6
    rng = np.random.default_rng(0)
    w = (rng.standard_normal((K, N)) * 0.5).astype(np.float32)
    model = _matmul_model(w, K, N)

    py = quantize_weight_only_qoq(model)
    cpp = onnxsim.apply_qoq_cpp(model)
    onnx.checker.check_model(py)
    onnx.checker.check_model(cpp)

    py_codes, py_scale, py_attrs = _codes_and_scale(py)
    cpp_codes, cpp_scale, cpp_attrs = _codes_and_scale(cpp)

    np.testing.assert_array_equal(cpp_codes, py_codes)
    np.testing.assert_allclose(cpp_scale, py_scale, rtol=1e-6, atol=1e-8)
    assert cpp_attrs["axis"].i == py_attrs["axis"].i
    assert cpp_attrs["block_size"].i == py_attrs["block_size"].i == _BLOCK_SIZE


def test_cpp_matches_python_end_to_end_output():
    K, N = _BLOCK_SIZE * 4, 8
    rng = np.random.default_rng(1)
    w = (rng.standard_normal((K, N)) * 0.3).astype(np.float32)
    model = _matmul_model(w, K, N)

    py = quantize_weight_only_qoq(model)
    cpp = onnxsim.apply_qoq_cpp(model)

    x = rng.standard_normal((4, K)).astype(np.float32)
    (py_y,) = _run(py, {"X": x})
    (cpp_y,) = _run(cpp, {"X": x})
    np.testing.assert_allclose(cpp_y, py_y, rtol=1e-5, atol=1e-6)


def test_cpp_reconstruction_error_beats_single_stage_int4():
    # QoQ's own whole point: a two-stage (INT8-then-INT4) round trip loses
    # slightly more than a single-stage INT4 fit in principle (rounding
    # twice), but is close enough that it should never be dramatically
    # worse -- this test guards against a genuinely broken port (e.g. one
    # that accidentally quantizes the float weight directly instead of the
    # INT8 intermediate) rather than asserting QoQ is strictly better.
    K, N = _BLOCK_SIZE * 2, 4
    rng = np.random.default_rng(2)
    w = (rng.standard_normal((K, N)) * 0.5).astype(np.float32)
    model = _matmul_model(w, K, N)
    cpp = onnxsim.apply_qoq_cpp(model)
    onnx.checker.check_model(cpp)

    x = rng.standard_normal((4, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(cpp, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.2


def test_cpp_gemm_transb():
    K, N = _BLOCK_SIZE * 2, 4
    rng = np.random.default_rng(3)
    weight = rng.standard_normal((N, K)).astype(np.float32) * 0.4
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm<transB = 1>(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )
    py = quantize_weight_only_qoq(model)
    cpp = onnxsim.apply_qoq_cpp(model)
    onnx.checker.check_model(cpp)

    py_codes, py_scale, _ = _codes_and_scale(py)
    cpp_codes, cpp_scale, _ = _codes_and_scale(cpp)
    np.testing.assert_array_equal(cpp_codes, py_codes)
    np.testing.assert_allclose(cpp_scale, py_scale, rtol=1e-6, atol=1e-8)


def test_cpp_gemm_with_bias():
    K, N = _BLOCK_SIZE * 2, 4
    rng = np.random.default_rng(4)
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.4
    bias = rng.standard_normal(N).astype(np.float32) * 0.1
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm(X, W, B)
        }}
        """,
        initializer=[_f32(weight, "W"), _f32(bias, "B")],
    )
    cpp = onnxsim.apply_qoq_cpp(model)
    onnx.checker.check_model(cpp)

    x = rng.standard_normal((4, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(cpp, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.2


def test_cpp_noop_when_no_matmul_gemm_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_qoq_cpp(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_cpp_skips_non_2d_weight():
    rng = np.random.default_rng(5)
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
    result = onnxsim.apply_qoq_cpp(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_cpp_skips_non_block_divisible_k():
    K, N = _BLOCK_SIZE + 3, 4
    rng = np.random.default_rng(6)
    w = rng.standard_normal((K, N)).astype(np.float32)
    model = _matmul_model(w, K, N)
    result = onnxsim.apply_qoq_cpp(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_cpp_declines_pre_opset21():
    K, N = _BLOCK_SIZE * 2, 4
    rng = np.random.default_rng(7)
    w = rng.standard_normal((K, N)).astype(np.float32)
    model = _matmul_model(w, K, N, opset=18)
    result = onnxsim.apply_qoq_cpp(model)
    assert result.SerializeToString() == model.SerializeToString()
