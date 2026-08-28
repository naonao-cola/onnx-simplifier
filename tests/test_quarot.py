"""Tests for ``onnxsim.apply_quarot`` -- see ``onnxsim/quarot.py`` for the
technique (random-rotation preprocessing, reused from
``onnxsim.quip_sharp``, plus INT4 round-to-nearest quantization of *both*
the weight and the activation -- no calibration data needed).
"""

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper
import pytest

import onnxsim

ort = pytest.importorskip("onnxruntime")


def _vi(name, shape):
    return onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _model(nodes, inputs, outputs, initializer, opset=21):
    graph = onnx.helper.make_graph(nodes, "g", inputs, outputs, initializer)
    return onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", opset)], ir_version=10
    )


def _matmul_model(K=32, N=8, weight=None, seed=0, opset=21):
    if weight is None:
        rng = np.random.default_rng(seed)
        weight = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    return _model(
        nodes,
        [_vi("X", ["batch", K])],
        [_vi("Y", ["batch", N])],
        [_f32(weight, "W")],
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
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-6)


def test_quarot_output_stays_close_to_float_via_onnxruntime():
    model = _matmul_model(K=32, N=8, seed=0)
    q = onnxsim.apply_quarot(model, block_size=8, seed=1)
    onnx.checker.check_model(q)

    op_types = [n.op_type for n in q.graph.node]
    assert op_types.count("MatMul") == 2  # X rotation, core
    assert "DequantizeLinear" in op_types
    assert "ReduceMax" in op_types  # data-free per-token activation scale

    rng = np.random.default_rng(2)
    x = rng.standard_normal((8, 32)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    # A generous bound: both operands are INT4 here (unlike every other
    # onnxsim weight-only scheme, which leaves the activation in float),
    # so this is a strictly harder target than e.g. SpinQuant's own 0.3.
    assert _rel_l2(float_y, q_y) < 0.5


def test_quarot_needs_no_calibration_data():
    # The whole point of a random (vs. fit) rotation: no calibration_data
    # kwarg exists at all, unlike apply_spinquant/apply_smoothquant/apply_awq.
    import inspect

    sig = inspect.signature(onnxsim.apply_quarot)
    assert "calibration_data" not in sig.parameters


def test_quarot_rotation_matrix_is_orthogonal():
    model = _matmul_model(K=16, N=4, seed=3)
    q = onnxsim.apply_quarot(model, block_size=4, seed=4)
    u_init = next(t for t in q.graph.initializer if t.name.endswith("_quarot_u"))
    u = onnx.numpy_helper.to_array(u_init).astype(np.float64)
    assert np.allclose(u @ u.T, np.eye(u.shape[0]), atol=1e-4)


def test_quarot_gemm_transb_with_bias():
    rng = np.random.default_rng(5)
    K, N = 32, 8
    weight = rng.standard_normal((N, K)).astype(np.float32) * 0.5
    bias = rng.standard_normal((N,)).astype(np.float32) * 0.1
    nodes = [onnx.helper.make_node("Gemm", ["X", "W", "B"], ["Y"], transB=1)]
    model = _model(
        nodes,
        [_vi("X", ["batch", K])],
        [_vi("Y", ["batch", N])],
        [_f32(weight, "W"), _f32(bias, "B")],
    )
    q = onnxsim.apply_quarot(model, block_size=8, seed=6)
    onnx.checker.check_model(q)
    assert "Add" in [n.op_type for n in q.graph.node]

    x = rng.standard_normal((4, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert _rel_l2(float_y, q_y) < 0.5


def test_quarot_declines_when_k_not_divisible_by_block_size():
    model = _matmul_model(K=20, N=4, seed=7)  # 20 is not a multiple of 8
    q = onnxsim.apply_quarot(model, block_size=8)
    assert q.SerializeToString() == model.SerializeToString()


def test_quarot_declines_non_constant_weight():
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    model = _model(
        nodes, [_vi("X", [4, 32]), _vi("W", [32, 4])], [_vi("Y", [4, 4])], []
    )
    q = onnxsim.apply_quarot(model)
    assert q.SerializeToString() == model.SerializeToString()


def test_quarot_noop_when_no_matmul_present():
    nodes = [onnx.helper.make_node("Relu", ["X"], ["Y"])]
    model = _model(nodes, [_vi("X", [4, 4])], [_vi("Y", [4, 4])], [])
    result = onnxsim.apply_quarot(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_quarot_declines_below_opset21():
    model = _matmul_model(K=32, N=8, opset=13)
    result = onnxsim.apply_quarot(model)
    assert result.SerializeToString() == model.SerializeToString()
