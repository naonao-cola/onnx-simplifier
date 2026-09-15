"""Tests for ``onnxsim.apply_awq_cpp`` -- the C++-backed port of
``onnxsim.apply_awq`` (AWQ's grid-searched per-channel rescaling, see
``onnxsim/awq.py``). Like ``test_gptq_cpp.py``, this runs the float
model over real calibration data through a real ``onnxruntime``-backed
executor -- never a fake/mock executor -- and checks exact (bit-for-bit)
agreement against the pure-Python reference: both sides search the same
alpha grid, keep the same winner, and emit the same codes, scales, and
compensating Mul nodes, so any divergence is a bug, not an accepted
tolerance.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.awq import apply_awq
from onnxsim.onnx_simplifier import apply_awq_cpp

ort = pytest.importorskip("onnxruntime")


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


def _f32(array, name):
    return onnx.numpy_helper.from_array(np.asarray(array, dtype=np.float32), name)


def _matmul_model(K=64, N=16, seed=0):
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


def _salient_calibration(K=64, num_samples=32, seed=1):
    # AWQ's own motivating scenario: one input channel consistently much
    # larger than the rest, so rescaling its weight column upward (and
    # compensating the activation downward) buys real error reduction.
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((num_samples, K)).astype(np.float32)
    x[:, 3] *= 15.0
    return [{"X": x}]


def _assert_exact_parity(float_model, calibration_data, **kwargs):
    quant = onnxsim.quantize_weight_only_int4(float_model)
    py = apply_awq(float_model, quant, calibration_data, **kwargs)
    cpp = apply_awq_cpp(float_model, quant, calibration_data, **kwargs)
    py_nodes = sorted(
        (n.op_type, tuple(n.input), tuple(n.output), n.name) for n in py.graph.node
    )
    cpp_nodes = sorted(
        (n.op_type, tuple(n.input), tuple(n.output), n.name) for n in cpp.graph.node
    )
    assert py_nodes == cpp_nodes
    py_inits = sorted(py.graph.initializer, key=lambda t: t.name)
    cpp_inits = sorted(cpp.graph.initializer, key=lambda t: t.name)
    assert [t.name for t in py_inits] == [t.name for t in cpp_inits]
    for a, b in zip(py_inits, cpp_inits):
        assert a.data_type == b.data_type, a.name
        ta, tb = onnx.numpy_helper.to_array(a), onnx.numpy_helper.to_array(b)
        assert ta.shape == tb.shape, a.name
        assert np.array_equal(ta, tb), a.name
    return cpp


def test_awq_cpp_matches_python_exactly():
    cpp = _assert_exact_parity(_matmul_model(), _salient_calibration())
    # The salient channel must actually win a nonzero alpha here (a Mul
    # compensating the activation), or this test proves nothing about
    # the rewrite path.
    assert any(n.op_type == "Mul" for n in cpp.graph.node)


def test_awq_cpp_matches_python_across_shapes_and_steps():
    for K, N, seed, steps in [
        (32, 8, 5, 8),
        (128, 32, 7, 20),
        (64, 16, 9, 5),
        (96, 24, 11, 12),
    ]:
        model = _matmul_model(K=K, N=N, seed=seed)
        cals = _salient_calibration(K=K, seed=seed + 100)
        _assert_exact_parity(model, cals, num_alpha_steps=steps)


def test_awq_cpp_gemm_transb():
    rng = np.random.default_rng(13)
    K, N = 64, 16
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
    _assert_exact_parity(model, _salient_calibration(K=K, seed=14))


def test_awq_cpp_alpha_zero_wins_cleanly():
    # Flat, outlier-free calibration: no rescaling can beat plain
    # round-to-nearest, so alpha == 0 must win -- codes rewritten, but
    # no Mul inserted and the original scale kept, on both sides.
    K, N = 32, 8
    model = _matmul_model(K=K, N=N, seed=15)
    rng = np.random.default_rng(16)
    cals = [{"X": (rng.standard_normal((16, K)) * 0.2).astype(np.float32)}]
    quant = onnxsim.quantize_weight_only_int4(model)
    py = apply_awq(model, quant, cals)
    cpp = apply_awq_cpp(model, quant, cals)
    assert not any(n.op_type == "Mul" for n in py.graph.node)
    assert not any(n.op_type == "Mul" for n in cpp.graph.node)
    _assert_exact_parity(model, cals)


def test_awq_cpp_multi_batch_and_3d_activation():
    K, N = 32, 8
    model = _matmul_model(K=K, N=N, seed=17)
    rng = np.random.default_rng(18)
    x1 = rng.standard_normal((8, K)).astype(np.float32)
    x1[:, 2] *= 12.0
    x2 = rng.standard_normal((12, K)).astype(np.float32)
    x2[:, 2] *= 12.0
    _assert_exact_parity(model, [{"X": x1}, {"X": x2}])

    model3d = _model(
        f"""
        g (float[batch,seq,{K}] X) => (float[batch,seq,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f32(rng.standard_normal((K, N)).astype(np.float32), "W")],
    )
    cals3d = [{"X": (rng.standard_normal((2, 5, K)) * 2).astype(np.float32)}]
    cpp = _assert_exact_parity(model3d, cals3d)
    onnx.checker.check_model(cpp)


def test_awq_cpp_skips():
    # Empty calibration data returns the quantized model structurally
    # unchanged on both sides.
    model = _matmul_model()
    quant = onnxsim.quantize_weight_only_int4(model)
    for fn in (apply_awq, apply_awq_cpp):
        out = fn(model, quant, [])
        assert [n.op_type for n in out.graph.node] == [
            n.op_type for n in quant.graph.node
        ]


def test_awq_cpp_beats_round_to_nearest_via_onnxruntime():
    # End to end through onnxruntime: the AWQ-optimized layer
    # reconstructs no worse than the plain round-to-nearest INT4
    # baseline -- and the C++ port matches the reference's own error
    # exactly here.
    model = _matmul_model(K=64, N=16, seed=19)
    x = _salient_calibration(K=64, num_samples=32, seed=20)[0]["X"]
    quant = onnxsim.quantize_weight_only_int4(model)
    cpp = apply_awq_cpp(model, quant, [{"X": x}])
    py = apply_awq(model, quant, [{"X": x}])
    onnx.checker.check_model(cpp)
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    q_sess = ort.InferenceSession(
        quant.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    c_sess = ort.InferenceSession(
        cpp.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    p_sess = ort.InferenceSession(
        py.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    ref = sess.run(["Y"], {"X": x})[0].astype(np.float64).ravel()
    norm = max(np.linalg.norm(ref), 1e-6)
    rtn_err = (
        np.linalg.norm(ref - q_sess.run(["Y"], {"X": x})[0].astype(np.float64).ravel())
        / norm
    )
    cpp_err = (
        np.linalg.norm(ref - c_sess.run(["Y"], {"X": x})[0].astype(np.float64).ravel())
        / norm
    )
    py_err = (
        np.linalg.norm(ref - p_sess.run(["Y"], {"X": x})[0].astype(np.float64).ravel())
        / norm
    )
    assert np.all(np.isfinite(c_sess.run(["Y"], {"X": x})[0]))
    assert cpp_err == py_err
    assert cpp_err < rtn_err
