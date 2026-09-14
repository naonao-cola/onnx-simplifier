"""Golden-string tests for :mod:`onnxsim.onnxnet_encoder`, a port of ONNX-Net's
``chain_slim``/``chain_slim_base`` text encodings (arXiv:2510.04938, see that
module's own docstring for exactly what is and isn't ported, and why two
upstream behaviors that look like bugs are preserved rather than fixed).

Shares ``test_onnxnet_integration.py``'s MLP-with-scaffolding fixture (same
model, same simplification step) so both files are pinned to the same
starting point: that file checks onnxsim's *structural* output (op types,
no branching); this one checks the encoder's exact *text* output on top of
that structure.
"""

import numpy as np
import onnx
import onnx.numpy_helper
from onnx import parser

import onnxsim
from onnxsim import onnxnet_encoder


def _model(body, initializer=(), opset=13, ir_version=10):
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
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _simplify(model):
    onnx.checker.check_model(model)
    sim_model, ok = onnxsim.simplify(model, check_n=0)
    assert ok
    onnx.checker.check_model(sim_model)
    return sim_model


def test_chain_slim_on_simplified_mlp():
    # Same model as test_onnxnet_integration.py's _build_mlp_with_scaffolding:
    # Identity-scaffolded MatMul+bias-Add, which onnxsim collapses to two
    # Gemms around a Relu.
    rng = np.random.RandomState(0)
    model = _model(
        """
        mlp_with_scaffolding (float[4,16] X) => (float[4,8] Y)
        {
          x_id = Identity(X)
          mm1 = MatMul(x_id, W1)
          lin1 = Add(mm1, B1)
          act1 = Relu(lin1)
          act1_id = Identity(act1)
          mm2 = MatMul(act1_id, W2)
          Y = Add(mm2, B2)
        }
        """,
        initializer=[
            _f32(rng.randn(16, 8), "W1"),
            _f32(rng.randn(8), "B1"),
            _f32(rng.randn(8, 8), "W2"),
            _f32(rng.randn(8), "B2"),
        ],
    )
    sim_model = _simplify(model)
    assert [n.op_type for n in sim_model.graph.node] == ["Gemm", "Relu", "Gemm"]

    # The very first Gemm's only non-constant input is the graph input X
    # itself, not a node-produced value -- node_inputs excludes graph
    # inputs, so this node is never "eligible for chaining" and is printed
    # via the branch-point path instead, with every input named explicitly
    # (X resolved to its own shape string via name_map, the two weights to
    # "Param[...]"). Gemm's alpha/beta/transA/transB never appear (all
    # scalar attributes -- see onnxnet_encoder's own docstring for why).
    # Relu then starts a fresh chain reading that closed value by name, and
    # the second Gemm continues it ("prev") straight through to "Out".
    assert onnxnet_encoder.chain_slim(sim_model) == (
        "Gemm(4x16, Param[16,8], Param[8]) --> Value1:4x8\n"
        "Relu(Value1) --> Gemm(prev, Param[8,8], Param[8]) --> Out"
    )


def test_chain_slim_base_on_simplified_mlp():
    rng = np.random.RandomState(0)
    model = _model(
        """
        mlp_with_scaffolding (float[4,16] X) => (float[4,8] Y)
        {
          x_id = Identity(X)
          mm1 = MatMul(x_id, W1)
          lin1 = Add(mm1, B1)
          act1 = Relu(lin1)
          act1_id = Identity(act1)
          mm2 = MatMul(act1_id, W2)
          Y = Add(mm2, B2)
        }
        """,
        initializer=[
            _f32(rng.randn(16, 8), "W1"),
            _f32(rng.randn(8), "B1"),
            _f32(rng.randn(8, 8), "W2"),
            _f32(rng.randn(8), "B2"),
        ],
    )
    sim_model = _simplify(model)

    assert onnxnet_encoder.chain_slim_base(sim_model) == (
        "Gemm --> Value\nRelu --> Gemm --> Out"
    )


def test_chain_slim_breaks_on_a_two_input_branch_point():
    # Two independent chains (Relu(X), Sigmoid(X)) merged by an Add whose
    # both inputs are node-produced -- exercises the branch-point path via
    # an actual multi-node-input merge, distinct from the "only input is
    # the graph input" case the MLP fixture above exercises.
    model = _model(
        """
        residual (float[2,4] X) => (float[2,4] Y)
        {
          a = Relu(X)
          b = Sigmoid(X)
          Y = Add(a, b)
        }
        """
    )
    sim_model = _simplify(model)
    assert [n.op_type for n in sim_model.graph.node] == ["Relu", "Sigmoid", "Add"]

    assert onnxnet_encoder.chain_slim(sim_model) == (
        "Relu(2x4) --> Value1:2x4\n"
        "Sigmoid(2x4) --> Value2:2x4\n"
        "Add(Value1, Value2) --> Out:2x4\n"
    )
    assert onnxnet_encoder.chain_slim_base(sim_model) == (
        "Relu --> Value\nSigmoid --> Value\nAdd --> Out\n"
    )


def test_chain_slim_collapses_an_all_equal_ints_attribute_to_a_scalar():
    # Conv's kernel_shape=[3,3] is an INTS attribute with every entry equal,
    # which _attribute_to_str collapses to the single value "3" rather than
    # "[3,3]" -- unlike the Gemm nodes above (whose alpha/beta/transA/transB
    # are scalar FLOAT/INT attributes and never appear at all).
    model = _model(
        """
        convnet (float[1,3,8,8] X) => (float[1,4,6,6] Y)
        {
          Y = Conv<kernel_shape=[3,3]>(X, W)
        }
        """,
        initializer=[_f32(np.random.RandomState(0).randn(4, 3, 3, 3), "W")],
    )
    sim_model = _simplify(model)
    assert onnxnet_encoder.chain_slim(sim_model) == (
        "Conv(1x3x8x8, Param[4,3,3,3])(kernel_shape=3) --> Out:1x4x6x6\n"
    )


def test_scalar_float_and_int_attributes_never_render():
    # Direct unit check of the documented upstream quirk _attribute_to_str
    # reproduces: a scalar FLOAT/INT attribute's value lives in
    # AttributeProto.f/.i, but the string form here reads .floats/.ints (the
    # *list* fields, always empty for a scalar attribute) -- so it always
    # serializes to "[]" and gets filtered out by _format_attrs, regardless
    # of the attribute's actual value.
    node = onnx.helper.make_node(
        "Gemm", ["A", "B"], ["C"], alpha=2.0, beta=0.5, transA=0, transB=1
    )
    assert onnxnet_encoder._format_attrs(node) == ""


def test_list_valued_attributes_do_render():
    node = onnx.helper.make_node("Pad", ["A"], ["B"], pads=[1, 2, 1, 2])
    assert onnxnet_encoder._format_attrs(node) == "pads=[1,2,1,2]"
