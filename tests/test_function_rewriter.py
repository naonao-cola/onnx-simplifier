"""Tests for the ``function_rewrite_rules`` hook of ``onnxsim.simplify``.

Unlike ``custom_rewriter`` (a Python callable, e.g. an ``onnxscript.rewriter``
rule set), ``function_rewrite_rules`` is *pure data*: a list of
``(pattern, replacement)`` pairs of ``onnx.FunctionProto`` matched and applied
natively by onnxsim's C++ core, so the exact same rules also work from the C
and Rust bindings. These tests build the FunctionProtos with
``onnx.parser.parse_function`` -- the intended authoring path.
"""

import collections

import numpy as np
import onnx
import pytest
from onnx import parser

import onnxsim


def _model(nodes, inputs, outputs, initializer, opset=18):
    graph = onnx.helper.make_graph(nodes, "g", inputs, outputs, initializer)
    return onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", opset)], ir_version=10
    )


def _vi(name, shape):
    return onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _op_types(model):
    return collections.Counter(n.op_type for n in model.graph.node)


def _matmul_add_model(swap_add_operands=False):
    """x @ W + B, i.e. a MatMul feeding an Add -- the classic Gemm fusion."""
    add_inputs = ["B", "mm"] if swap_add_operands else ["mm", "B"]
    nodes = [
        onnx.helper.make_node("MatMul", ["x", "W"], ["mm"], name="mm"),
        onnx.helper.make_node("Add", add_inputs, ["y"], name="add"),
    ]
    inits = [_f32(np.random.randn(4, 5), "W"), _f32(np.random.randn(5), "B")]
    return _model(nodes, [_vi("x", [3, 4])], [_vi("y", [3, 5])], inits)


# The canonical rule: y = MatMul(x, w) + b  ==>  y = Gemm(x, w, b).
_GEMM_PATTERN = parser.parse_function(
    """
<domain: "com.onnxsim.test", opset_import: ["" : 18]>
matmul_add_pattern (x, w, b) => (y)
{
    t = MatMul(x, w)
    y = Add(t, b)
}
"""
)
_GEMM_REPLACEMENT = parser.parse_function(
    """
<domain: "com.onnxsim.test", opset_import: ["" : 18]>
gemm_replacement (x, w, b) => (y)
{
    y = Gemm(x, w, b)
}
"""
)


def test_function_rewriter_matmul_add_to_gemm():
    """The headline case: MatMul + Add fuse into a single Gemm."""
    model = _matmul_add_model()
    sim_model, check_ok = onnxsim.simplify(
        model, check_n=1, function_rewrite_rules=[(_GEMM_PATTERN, _GEMM_REPLACEMENT)]
    )
    counts = _op_types(sim_model)
    assert counts["Gemm"] == 1, counts
    assert counts["MatMul"] == 0 and counts["Add"] == 0, counts
    assert check_ok


def test_function_rewriter_commutative_add():
    """The matcher tries both operand orders for commutative ops, so an
    ``Add(b, MatMul(x, w))`` (operands swapped) still fuses."""
    model = _matmul_add_model(swap_add_operands=True)
    sim_model, _ = onnxsim.simplify(
        model, check_n=0, function_rewrite_rules=[(_GEMM_PATTERN, _GEMM_REPLACEMENT)]
    )
    assert _op_types(sim_model)["Gemm"] == 1


def test_function_rewriter_attribute_wildcard():
    """A ``@name`` ref-attribute is an attribute wildcard: it binds the matched
    node's attribute and substitutes it into the replacement. Here ``Relu``
    feeding a ``LeakyRelu<alpha>`` collapses to a single ``LeakyRelu`` that keeps
    the original ``alpha``."""
    pattern = parser.parse_function(
        """
<domain: "com.onnxsim.test", opset_import: ["" : 18]>
relu_leaky (x) => (y)
{
    t = Relu(x)
    y = LeakyRelu <alpha = @a> (t)
}
"""
    )
    replacement = parser.parse_function(
        """
<domain: "com.onnxsim.test", opset_import: ["" : 18]>
leaky (x) => (y)
{
    y = LeakyRelu <alpha = @a> (x)
}
"""
    )
    model = _model(
        [
            onnx.helper.make_node("Relu", ["x"], ["t"], name="relu"),
            onnx.helper.make_node("LeakyRelu", ["t"], ["y"], alpha=0.2, name="lr"),
        ],
        [_vi("x", [2, 2])],
        [_vi("y", [2, 2])],
        [],
    )
    sim_model, _ = onnxsim.simplify(
        model, check_n=0, function_rewrite_rules=[(pattern, replacement)]
    )
    counts = _op_types(sim_model)
    assert counts["Relu"] == 0 and counts["LeakyRelu"] == 1, counts
    leaky = next(n for n in sim_model.graph.node if n.op_type == "LeakyRelu")
    alpha = next(a.f for a in leaky.attribute if a.name == "alpha")
    assert abs(alpha - 0.2) < 1e-6, alpha


def test_function_rewriter_constant_value_match():
    """A pattern ``Constant`` matches a host initializer with a byte-equal
    tensor: ``Mul(x, ones)`` rewrites to ``Identity(x)``."""
    ones = np.ones([4], dtype=np.float32)
    pattern = onnx.helper.make_function(
        "com.onnxsim.test",
        "mul_ones",
        ["x"],
        ["y"],
        [
            onnx.helper.make_node(
                "Constant", [], ["c"], value=onnx.numpy_helper.from_array(ones, "")
            ),
            onnx.helper.make_node("Mul", ["x", "c"], ["y"]),
        ],
        [onnx.helper.make_opsetid("", 18)],
    )
    replacement = onnx.helper.make_function(
        "com.onnxsim.test",
        "identity",
        ["x"],
        ["y"],
        [onnx.helper.make_node("Identity", ["x"], ["y"])],
        [onnx.helper.make_opsetid("", 18)],
    )
    model = _model(
        [onnx.helper.make_node("Mul", ["x", "ones"], ["y"], name="mul")],
        [_vi("x", [4])],
        [_vi("y", [4])],
        [onnx.numpy_helper.from_array(ones, "ones")],
    )
    sim_model, _ = onnxsim.simplify(
        model, check_n=0, function_rewrite_rules=[(pattern, replacement)]
    )
    counts = _op_types(sim_model)
    assert counts["Mul"] == 0 and counts["Identity"] >= 1, counts


def test_function_rewriter_no_match_is_noop():
    """A rule whose pattern never matches leaves the model's ops unchanged."""
    model = _model(
        [onnx.helper.make_node("MatMul", ["x", "W"], ["y"], name="mm")],
        [_vi("x", [3, 4])],
        [_vi("y", [3, 5])],
        [_f32(np.random.randn(4, 5), "W")],
    )
    baseline, _ = onnxsim.simplify(model, check_n=0)
    sim_model, _ = onnxsim.simplify(
        model, check_n=0, function_rewrite_rules=[(_GEMM_PATTERN, _GEMM_REPLACEMENT)]
    )
    assert _op_types(sim_model) == _op_types(baseline)
    assert _op_types(sim_model)["Gemm"] == 0


def test_function_rewriter_safety_shared_intermediate():
    """The matched interior value (the MatMul output) is also consumed by a Relu
    outside the match, so fusing would break the Relu -- the rewrite must be
    skipped and MatMul/Add kept."""
    nodes = [
        onnx.helper.make_node("MatMul", ["x", "W"], ["mm"], name="mm"),
        onnx.helper.make_node("Add", ["mm", "B"], ["y"], name="add"),
        onnx.helper.make_node("Relu", ["mm"], ["z"], name="relu"),
    ]
    inits = [_f32(np.random.randn(4, 5), "W"), _f32(np.random.randn(5), "B")]
    model = _model(
        nodes, [_vi("x", [3, 4])], [_vi("y", [3, 5]), _vi("z", [3, 5])], inits
    )
    sim_model, _ = onnxsim.simplify(
        model, check_n=0, function_rewrite_rules=[(_GEMM_PATTERN, _GEMM_REPLACEMENT)]
    )
    counts = _op_types(sim_model)
    assert counts["Gemm"] == 0 and counts["MatMul"] == 1 and counts["Add"] == 1, counts


def test_function_rewriter_mutually_exclusive_with_custom_rewriter():
    """Passing both a callable and data rules is a usage error."""
    model = _matmul_add_model()
    with pytest.raises(ValueError):
        onnxsim.simplify(
            model,
            check_n=0,
            custom_rewriter=lambda m: m,
            function_rewrite_rules=[(_GEMM_PATTERN, _GEMM_REPLACEMENT)],
        )


def test_function_rewriter_replaces_builtin_pass():
    """A FunctionProto rule can stand in for a hand-written onnxoptimizer pass.

    With the built-in ``fuse_matmul_add_bias_into_gemm`` pass skipped, onnxsim no
    longer fuses on its own; the rule reproduces exactly that fusion. (The
    imperative pass additionally guards on a constant, rank-compatible bias and
    sets alpha/transA -- guards a pure structural rule does not express, which is
    why the two are equivalent here but not in general.)"""
    model = _matmul_add_model()

    skipped_no_rule, _ = onnxsim.simplify(
        model, check_n=0, skipped_optimizers=["fuse_matmul_add_bias_into_gemm"]
    )
    assert _op_types(skipped_no_rule)["Gemm"] == 0, "sanity: builtin fusion was skipped"

    with_rule, check_ok = onnxsim.simplify(
        model,
        check_n=1,
        skipped_optimizers=["fuse_matmul_add_bias_into_gemm"],
        function_rewrite_rules=[(_GEMM_PATTERN, _GEMM_REPLACEMENT)],
    )
    counts = _op_types(with_rule)
    assert counts["Gemm"] == 1 and counts["MatMul"] == 0 and counts["Add"] == 0, counts
    assert check_ok
