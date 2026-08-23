"""Tests for ``onnxsim.quantize_qoperator_activation`` (the
``qoperator_quantize_activation`` C++ pass) -- the unary-activation analogue
of ``test_qoperator_quantize_elementwise.py``'s ``QLinearAdd``/``QLinearMul``
coverage, using ONNX Runtime's "com.microsoft" contrib ops
``QLinearSigmoid``/``QLinearLeakyRelu`` instead.
"""

import collections

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper
import pytest

import onnxsim

# A bare ``import onnxruntime`` would fail collection (not skip the test) on
# platforms onnxruntime doesn't ship wheels for (e.g. s390x).
ort = pytest.importorskip("onnxruntime")


def _model(nodes, inputs, outputs, initializer, opset=13):
    graph = onnx.helper.make_graph(nodes, "g", inputs, outputs, initializer)
    return onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", opset)], ir_version=10
    )


def _vi(name, shape):
    return onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _op_counts(model):
    return collections.Counter(n.op_type for n in model.graph.node)


def _assert_close(float_outputs, quant_outputs, rel_l2_tol=0.1):
    for f, q in zip(float_outputs, quant_outputs):
        f = np.asarray(f, dtype=np.float64).ravel()
        q = np.asarray(q, dtype=np.float64).ravel()
        assert np.all(np.isfinite(q))
        rel_l2 = np.linalg.norm(f - q) / max(np.linalg.norm(f), 1e-6)
        assert rel_l2 < rel_l2_tol, f"relative L2 error too large: {rel_l2:.4f}"


def test_quantize_sigmoid():
    rng = np.random.default_rng(0)
    nodes = [onnx.helper.make_node("Sigmoid", ["X"], ["Y"])]
    model = _model(nodes, [_vi("X", [4, 8])], [_vi("Y", [4, 8])], [])

    quant = onnxsim.quantize_qoperator_activation(
        model, num_calibration_samples=16, seed=0
    )
    onnx.checker.check_model(quant)
    ops = _op_counts(quant)
    assert ops["Sigmoid"] == 0
    assert ops["QLinearSigmoid"] == 1
    assert ops["QuantizeLinear"] == 1
    assert ops["DequantizeLinear"] == 1
    domains = {o.domain for o in quant.opset_import}
    assert "com.microsoft" in domains

    x = rng.standard_normal((4, 8)).astype(np.float32)
    _assert_close(_run(model, {"X": x}), _run(quant, {"X": x}))


def test_quantize_leaky_relu():
    rng = np.random.default_rng(1)
    nodes = [onnx.helper.make_node("LeakyRelu", ["X"], ["Y"], alpha=0.2)]
    model = _model(nodes, [_vi("X", [4, 8])], [_vi("Y", [4, 8])], [])

    quant = onnxsim.quantize_qoperator_activation(
        model, num_calibration_samples=16, seed=1
    )
    onnx.checker.check_model(quant)
    ops = _op_counts(quant)
    assert ops["LeakyRelu"] == 0
    assert ops["QLinearLeakyRelu"] == 1
    assert ops["QuantizeLinear"] == 1
    assert ops["DequantizeLinear"] == 1

    qlop = next(n for n in quant.graph.node if n.op_type == "QLinearLeakyRelu")
    alpha = next(a.f for a in qlop.attribute if a.name == "alpha")
    assert alpha == pytest.approx(0.2)

    x = rng.standard_normal((4, 8)).astype(np.float32)
    _assert_close(_run(model, {"X": x}), _run(quant, {"X": x}))


def test_quantize_leaky_relu_default_alpha():
    # LeakyRelu's alpha defaults to 0.01 when omitted -- the rewrite must
    # carry that default over, not silently drop it to 0.
    rng = np.random.default_rng(2)
    nodes = [onnx.helper.make_node("LeakyRelu", ["X"], ["Y"])]
    model = _model(nodes, [_vi("X", [4, 8])], [_vi("Y", [4, 8])], [])

    quant = onnxsim.quantize_qoperator_activation(
        model, num_calibration_samples=16, seed=2
    )
    onnx.checker.check_model(quant)
    qlop = next(n for n in quant.graph.node if n.op_type == "QLinearLeakyRelu")
    alpha = next(a.f for a in qlop.attribute if a.name == "alpha")
    assert alpha == pytest.approx(0.01)

    x = rng.standard_normal((4, 8)).astype(np.float32)
    _assert_close(_run(model, {"X": x}), _run(quant, {"X": x}))


def test_quantize_multiple_independent_nodes():
    rng = np.random.default_rng(3)
    nodes = [
        onnx.helper.make_node("Sigmoid", ["A"], ["T1"]),
        onnx.helper.make_node("LeakyRelu", ["B"], ["T2"], alpha=0.1),
        onnx.helper.make_node("Concat", ["T1", "T2"], ["C"], axis=0),
    ]
    model = _model(nodes, [_vi("A", [4, 8]), _vi("B", [4, 8])], [_vi("C", [8, 8])], [])

    quant = onnxsim.quantize_qoperator_activation(
        model, num_calibration_samples=16, seed=3
    )
    onnx.checker.check_model(quant)
    ops = _op_counts(quant)
    assert ops["QLinearSigmoid"] == 1
    assert ops["QLinearLeakyRelu"] == 1

    a = rng.standard_normal((4, 8)).astype(np.float32)
    b = rng.standard_normal((4, 8)).astype(np.float32)
    feeds = {"A": a, "B": b}
    _assert_close(_run(model, feeds), _run(quant, feeds))


def test_quantize_skips_non_float():
    nodes = [onnx.helper.make_node("Sigmoid", ["X"], ["Y"])]
    graph = onnx.helper.make_graph(
        nodes,
        "g",
        [onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT16, [4])],
        [onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT16, [4])],
        [],
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 13)], ir_version=10
    )
    quant = onnxsim.quantize_qoperator_activation(model)
    assert _op_counts(quant)["Sigmoid"] == 1
    assert _op_counts(quant)["QLinearSigmoid"] == 0


def test_list_qoperator_activation_quantizable_tensors():
    nodes = [onnx.helper.make_node("Sigmoid", ["X"], ["Y"])]
    model = _model(nodes, [_vi("X", [4, 8])], [_vi("Y", [4, 8])], [])
    import onnxsim.onnxsim_cpp2py_export as C

    names = C.list_qoperator_activation_quantizable_tensors(model.SerializeToString())
    assert set(names) == {"X", "Y"}
