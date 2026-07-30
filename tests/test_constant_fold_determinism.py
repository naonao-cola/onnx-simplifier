"""Constant folding of deterministic ops that ONNX defines via function bodies.

Regression tests for the folding-eligibility rule in ``IsDeterministic``
(onnxsim/onnxsim.cpp). Ops such as ``Range`` are perfectly deterministic but
ONNX describes them through a function body that contains a subgraph-carrying op
(``Loop``) or a context-dependent function, so the schema's inferred determinism
is ``NonDeterministic``/``Unknown``. onnxsim must still fold them when their
inputs are constant; otherwise a single unfolded ``Range`` poisons whole static
subgraphs (e.g. the attention-mask construction in Swin-style models). At the
same time, genuinely non-deterministic ops (the random generators) must never be
folded.

These tests use only the ``onnx`` Python API, so they run without torch.
"""

import numpy as np
import onnx
from onnx import TensorProto, helper

import onnxsim


def _make_model(nodes, outputs, initializers, inputs=None, opset=18):
    graph = helper.make_graph(nodes, "g", inputs or [], outputs, initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 8
    return model


def _op_types(model):
    return [n.op_type for n in model.graph.node]


def test_range_on_constants_is_folded():
    # Range is a deterministic op that ONNX defines through a function body
    # (a Loop for opset < 27, a context-dependent function for opset >= 27), so
    # its schema determinism is not reported as ``Deterministic``. With constant
    # inputs it must still fold away.
    inits = [
        helper.make_tensor("start", TensorProto.INT64, [], [0]),
        helper.make_tensor("limit", TensorProto.INT64, [], [8]),
        helper.make_tensor("delta", TensorProto.INT64, [], [1]),
    ]
    nodes = [helper.make_node("Range", ["start", "limit", "delta"], ["out"])]
    outputs = [helper.make_tensor_value_info("out", TensorProto.INT64, [8])]
    model = _make_model(nodes, outputs, inits)

    sim, ok = onnxsim.simplify(model)

    assert ok
    assert "Range" not in _op_types(sim), _op_types(sim)


def test_range_rooted_subgraph_is_fully_folded():
    # Range -> Add(range, bias): the whole chain is constant and should collapse
    # to a single initializer. Before the fix the unfoldable Range left both
    # nodes in the graph.
    inits = [
        helper.make_tensor("start", TensorProto.INT64, [], [0]),
        helper.make_tensor("limit", TensorProto.INT64, [], [6]),
        helper.make_tensor("delta", TensorProto.INT64, [], [1]),
        helper.make_tensor("bias", TensorProto.INT64, [6], [10, 20, 30, 40, 50, 60]),
    ]
    nodes = [
        helper.make_node("Range", ["start", "limit", "delta"], ["r"]),
        helper.make_node("Add", ["r", "bias"], ["out"]),
    ]
    outputs = [helper.make_tensor_value_info("out", TensorProto.INT64, [6])]
    model = _make_model(nodes, outputs, inits)

    sim, ok = onnxsim.simplify(model)

    assert ok
    assert len(sim.graph.node) == 0, _op_types(sim)
    # The baked value is correct.
    (folded,) = [t for t in sim.graph.initializer if t.name == "out"]
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(folded),
        np.arange(0, 6) + np.array([10, 20, 30, 40, 50, 60]),
    )


def test_random_ops_are_not_folded():
    # Genuinely non-deterministic generators must be preserved even though all
    # their (attribute) inputs are "constant".
    for op, extra in [
        ("RandomUniform", {"shape": [4]}),
        ("RandomNormal", {"shape": [4]}),
    ]:
        nodes = [helper.make_node(op, [], ["out"], **extra)]
        outputs = [helper.make_tensor_value_info("out", TensorProto.FLOAT, [4])]
        model = _make_model(nodes, outputs, [])
        sim, ok = onnxsim.simplify(model)
        assert ok
        assert op in _op_types(sim), (op, _op_types(sim))
