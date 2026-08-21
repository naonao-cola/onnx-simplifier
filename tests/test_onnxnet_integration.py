"""Integration test guarding onnxsim as a preprocessor for ONNX-Net-style tools.

ONNX-Net (arXiv:2510.04938) turns an ONNX graph into a text encoding for an
LLM-based architecture performance predictor. Its pipeline explicitly runs a
model through onnxsim first, then relies on two structural properties of the
*simplified* graph before it can serialize it cheaply:

  * "node removal" -- no-op scaffolding (e.g. ``Identity``) must already be
    gone, since such nodes carry no architectural information but would still
    consume a line of text.
  * "subgraph merging" -- common multi-node idioms with a known high-level
    interpretation, e.g. ``MatMul`` + bias ``Add``, must already be collapsed
    into the single node that names them (``Gemm``, i.e. a "linear layer" in
    the paper's own example), rather than left as separate primitive ops.

Those two properties in turn make the graph a plain, unbranching chain, which
is exactly what ONNX-Net's chain-condensation step needs: it merges any run
of nodes with no branching onto a single text line, and a graph that is
*already* one straight chain condenses to a single line for free.

This test does not depend on ONNX-Net's code (it lives in a separate
repository); it only pins down, from this repo's side, that onnxsim's output
still has the shape that pipeline assumes.
"""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

import onnxsim


def _vi(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _f32(array, name):
    return numpy_helper.from_array(array.astype(np.float32), name)


def _build_mlp_with_scaffolding():
    # A tiny two-layer MLP (Linear -> ReLU -> Linear), but expressed the way an
    # exporter might emit it before simplification: each Linear as a separate
    # MatMul + bias Add, and a redundant Identity ahead of each consumer. This
    # mirrors ONNX-Net's own "linear layer" merge example while also exercising
    # the identity-removal step it depends on.
    rng = np.random.RandomState(0)
    w1 = _f32(rng.randn(16, 8), "W1")
    b1 = _f32(rng.randn(8), "B1")
    w2 = _f32(rng.randn(8, 8), "W2")
    b2 = _f32(rng.randn(8), "B2")

    nodes = [
        helper.make_node("Identity", ["X"], ["x_id"]),
        helper.make_node("MatMul", ["x_id", "W1"], ["mm1"]),
        helper.make_node("Add", ["mm1", "B1"], ["lin1"]),
        helper.make_node("Relu", ["lin1"], ["act1"]),
        helper.make_node("Identity", ["act1"], ["act1_id"]),
        helper.make_node("MatMul", ["act1_id", "W2"], ["mm2"]),
        helper.make_node("Add", ["mm2", "B2"], ["Y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "mlp_with_scaffolding",
        [_vi("X", [4, 16])],
        [_vi("Y", [4, 8])],
        [w1, b1, w2, b2],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=10
    )
    onnx.checker.check_model(model)
    return model


def _assert_single_unbranching_chain(model):
    # ONNX-Net's chain-condensation step only collapses runs of nodes that
    # have no branching: each node must have exactly one node-produced input
    # (besides initializers / the graph input) and its output must feed at
    # most one downstream node (besides the graph output). Verify the whole
    # graph satisfies that, i.e. it condenses to a single text line.
    graph = model.graph
    initializer_names = {i.name for i in graph.initializer}
    graph_input_names = {i.name for i in graph.input}
    graph_output_names = {o.name for o in graph.output}

    producer_of = {}
    for node in graph.node:
        for out in node.output:
            producer_of[out] = node

    consumer_count = {out: 0 for node in graph.node for out in node.output}
    for node in graph.node:
        for inp in node.input:
            if inp in consumer_count:
                consumer_count[inp] += 1

    for node in graph.node:
        node_producer_inputs = [
            inp
            for inp in node.input
            if inp not in initializer_names and inp not in graph_input_names
        ]
        assert len(node_producer_inputs) <= 1, (
            f"{node.op_type} node {node.name!r} has {len(node_producer_inputs)} "
            "node-produced inputs, which is a branch/merge point ONNX-Net's "
            "chain condensation cannot collapse into a single line"
        )
        for out in node.output:
            if out in graph_output_names:
                continue
            assert consumer_count[out] <= 1, (
                f"output {out!r} of {node.op_type} node {node.name!r} fans out to "
                f"{consumer_count[out]} consumers, breaking the single-chain shape"
            )


def test_onnxnet_preprocessing_yields_clean_chain_graph():
    model = _build_mlp_with_scaffolding()
    sim_model, check_ok = onnxsim.simplify(model, check_n=3)
    assert check_ok
    onnx.checker.check_model(sim_model)

    op_types = [n.op_type for n in sim_model.graph.node]
    # Node removal: the scaffolding Identity nodes carry no architectural
    # information and must be gone.
    assert "Identity" not in op_types
    # Subgraph merging: each MatMul+Add bias pair collapses into the single
    # node ONNX-Net's own example names a "linear layer".
    assert "MatMul" not in op_types
    assert "Add" not in op_types
    assert op_types.count("Gemm") == 2
    assert op_types.count("Relu") == 1
    assert len(op_types) == 3

    # The simplified graph must be exactly the shape ONNX-Net's serializer
    # needs: a single unbranching chain of (Gemm -> Relu -> Gemm).
    _assert_single_unbranching_chain(sim_model)

    # No initializer may be left dangling for the same reason as the wider
    # simplifier suite: an unreferenced weight tensor would still show up as
    # graph clutter for a downstream graph-to-text pass.
    used = {i for n in sim_model.graph.node for i in n.input}
    for init in sim_model.graph.initializer:
        assert init.name in used, f"unused initializer left behind: {init.name}"
