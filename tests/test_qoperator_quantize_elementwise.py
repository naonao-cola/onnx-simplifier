"""Tests for ``onnxsim.quantize_qoperator_elementwise`` (the
``qoperator_quantize_elementwise`` C++ pass) -- the elementwise Add/Mul
analogue of ``test_qoperator_quantize_matmul.py``'s ``QLinearMatMul``
coverage, using ONNX Runtime's "com.microsoft" contrib ops ``QLinearAdd``/
``QLinearMul`` instead.
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


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


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


def test_quantize_add():
    rng = np.random.default_rng(0)
    nodes = [onnx.helper.make_node("Add", ["A", "B"], ["C"])]
    model = _model(nodes, [_vi("A", [4, 8]), _vi("B", [4, 8])], [_vi("C", [4, 8])], [])

    quant = onnxsim.quantize_qoperator_elementwise(
        model, num_calibration_samples=16, seed=0
    )
    onnx.checker.check_model(quant)
    ops = _op_counts(quant)
    assert ops["Add"] == 0
    assert ops["QLinearAdd"] == 1
    assert ops["QuantizeLinear"] == 2  # one for A, one for B
    assert ops["DequantizeLinear"] == 1  # one for the output
    domains = {o.domain for o in quant.opset_import}
    assert "com.microsoft" in domains

    a = rng.standard_normal((4, 8)).astype(np.float32)
    b = rng.standard_normal((4, 8)).astype(np.float32)
    _assert_close(_run(model, {"A": a, "B": b}), _run(quant, {"A": a, "B": b}))


def test_quantize_mul():
    rng = np.random.default_rng(1)
    nodes = [onnx.helper.make_node("Mul", ["A", "B"], ["C"])]
    model = _model(nodes, [_vi("A", [4, 8]), _vi("B", [4, 8])], [_vi("C", [4, 8])], [])

    quant = onnxsim.quantize_qoperator_elementwise(
        model, num_calibration_samples=16, seed=1
    )
    onnx.checker.check_model(quant)
    ops = _op_counts(quant)
    assert ops["Mul"] == 0
    assert ops["QLinearMul"] == 1
    assert ops["QuantizeLinear"] == 2
    assert ops["DequantizeLinear"] == 1

    a = rng.standard_normal((4, 8)).astype(np.float32)
    b = rng.standard_normal((4, 8)).astype(np.float32)
    _assert_close(_run(model, {"A": a, "B": b}), _run(quant, {"A": a, "B": b}))


def test_quantize_broadcast():
    # QLinearAdd/QLinearMul support the same bidirectional broadcasting as
    # plain Add/Mul (see QLinearBinaryShapeInference in contrib_schemas.cpp).
    rng = np.random.default_rng(2)
    nodes = [onnx.helper.make_node("Add", ["A", "B"], ["C"])]
    model = _model(nodes, [_vi("A", [4, 8]), _vi("B", [1, 8])], [_vi("C", [4, 8])], [])

    quant = onnxsim.quantize_qoperator_elementwise(
        model, num_calibration_samples=16, seed=2
    )
    onnx.checker.check_model(quant)
    assert _op_counts(quant)["QLinearAdd"] == 1

    a = rng.standard_normal((4, 8)).astype(np.float32)
    b = rng.standard_normal((1, 8)).astype(np.float32)
    _assert_close(_run(model, {"A": a, "B": b}), _run(quant, {"A": a, "B": b}))


def test_quantize_multiple_independent_nodes():
    # Two unrelated elementwise Adds in the same graph (distinct operands,
    # not chained through each other) should both be quantized. A node
    # consuming *another* rewritten node's output is a separate, pre-existing
    # QOperator-family limitation (see qoperator_quantize_matmul.h/_conv.h,
    # which have the same characteristic): the rewrite replaces the node
    # with a fresh Value carrying an auto-generated name, so a downstream
    # node's activation-range lookup (keyed by the *original* tensor name)
    # no longer finds an entry -- not exercised here.
    rng = np.random.default_rng(3)
    nodes = [
        onnx.helper.make_node("Add", ["A", "B"], ["T1"]),
        onnx.helper.make_node("Add", ["C", "D"], ["T2"]),
        onnx.helper.make_node("Concat", ["T1", "T2"], ["E"], axis=0),
    ]
    model = _model(
        nodes,
        [_vi("A", [4, 8]), _vi("B", [4, 8]), _vi("C", [4, 8]), _vi("D", [4, 8])],
        [_vi("E", [8, 8])],
        [],
    )

    quant = onnxsim.quantize_qoperator_elementwise(
        model, num_calibration_samples=16, seed=3
    )
    onnx.checker.check_model(quant)
    assert _op_counts(quant)["QLinearAdd"] == 2

    a = rng.standard_normal((4, 8)).astype(np.float32)
    b = rng.standard_normal((4, 8)).astype(np.float32)
    c = rng.standard_normal((4, 8)).astype(np.float32)
    d = rng.standard_normal((4, 8)).astype(np.float32)
    feeds = {"A": a, "B": b, "C": c, "D": d}
    _assert_close(_run(model, feeds), _run(quant, feeds))


def test_quantize_skips_constant_operand():
    # A constant operand (e.g. a per-channel bias/embedding) is left alone --
    # it should be quantized from its own static values, not force-fed
    # through the calibration harness.
    bias = _f32(np.random.default_rng(4).standard_normal(8), "B")
    nodes = [onnx.helper.make_node("Add", ["A", "B"], ["C"])]
    model = _model(nodes, [_vi("A", [4, 8])], [_vi("C", [4, 8])], [bias])
    import onnxsim.onnxsim_cpp2py_export as C

    names = C.list_qoperator_elementwise_quantizable_tensors(model.SerializeToString())
    assert names == []

    quant = onnxsim.quantize_qoperator_elementwise(model)
    assert _op_counts(quant)["Add"] == 1
    assert _op_counts(quant)["QLinearAdd"] == 0


def test_quantize_skips_non_float():
    nodes = [onnx.helper.make_node("Add", ["A", "B"], ["C"])]
    graph = onnx.helper.make_graph(
        nodes,
        "g",
        [
            onnx.helper.make_tensor_value_info("A", onnx.TensorProto.INT64, [4]),
            onnx.helper.make_tensor_value_info("B", onnx.TensorProto.INT64, [4]),
        ],
        [onnx.helper.make_tensor_value_info("C", onnx.TensorProto.INT64, [4])],
        [],
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 13)], ir_version=10
    )
    quant = onnxsim.quantize_qoperator_elementwise(model)
    assert _op_counts(quant)["Add"] == 1
    assert _op_counts(quant)["QLinearAdd"] == 0


def test_list_qoperator_elementwise_quantizable_tensors():
    nodes = [onnx.helper.make_node("Add", ["A", "B"], ["C"])]
    model = _model(nodes, [_vi("A", [4, 8]), _vi("B", [4, 8])], [_vi("C", [4, 8])], [])
    import onnxsim.onnxsim_cpp2py_export as C

    names = C.list_qoperator_elementwise_quantizable_tensors(model.SerializeToString())
    assert set(names) == {"A", "B", "C"}
