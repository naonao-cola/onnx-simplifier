"""Differential test: drive onnxsim's native FunctionProto matcher and
``onnxscript.rewriter`` from the *same* FunctionProtos and compare.

A single source of truth -- one ``(pattern, replacement)`` pair of
``onnx.FunctionProto`` -- is fed to both engines: onnxsim via
``simplify(function_rewrite_rules=...)``, and onnxscript via a small interpreter
that rebuilds the pattern/replacement as an ``onnxscript.rewriter`` rule. This
shows both where the two agree (parity) and where onnxsim's first-version
matcher is deliberately narrower (limitations).
"""

import collections

import numpy as np
import onnx
import pytest
from onnx import helper, parser

import onnxsim

pattern_mod = pytest.importorskip("onnxscript.rewriter").pattern
rewrite = pytest.importorskip("onnxscript.rewriter").rewrite


# --------------------------------------------------------------------------
# FunctionProto -> onnxscript rule interpreter (the shared source of truth).
# Supports the subset these tests use: single-output nodes, value inputs, and
# concrete attributes. That is enough for the MatMul+Add -> Gemm rule below.
# --------------------------------------------------------------------------
def _interp(op, fp, env):
    env = dict(env)
    for node in fp.node:
        assert len(node.output) == 1, "interpreter supports single-output nodes only"
        inputs = [env[name] for name in node.input]
        attrs = {}
        for attr in node.attribute:
            assert not attr.ref_attr_name, "interpreter supports concrete attrs only"
            attrs[attr.name] = helper.get_attribute_value(attr)
        env[node.output[0]] = getattr(op, node.op_type)(*inputs, **attrs)
    outs = tuple(env[o] for o in fp.output)
    return outs[0] if len(outs) == 1 else outs


def _to_callable(fp):
    ns = {"_interp": _interp, "_fp": fp}
    params = ", ".join(fp.input)
    env_lit = ", ".join(f"{name!r}: {name}" for name in fp.input)
    src = f"def _rule(op, {params}):\n    return _interp(op, _fp, {{{env_lit}}})\n"
    exec(src, ns)  # noqa: S102 - trusted, generated from our own FunctionProto
    return ns["_rule"]


def _onnxscript_rules(pattern_fp, replacement_fp):
    rule = pattern_mod.RewriteRule(
        _to_callable(pattern_fp), _to_callable(replacement_fp)
    )
    return pattern_mod.RewriteRuleSet([rule])


def _all_op_types(model):
    """Op-type counts across the main graph AND every nested subgraph body."""
    counts = collections.Counter()

    def walk(graph):
        for node in graph.node:
            counts[node.op_type] += 1
            for attr in node.attribute:
                if attr.HasField("g"):
                    walk(attr.g)
                for subgraph in attr.graphs:
                    walk(subgraph)

    walk(model.graph)
    return counts


_PATTERN = parser.parse_function(
    """
<domain: "com.onnxsim.test", opset_import: ["" : 18]>
matmul_add_pattern (x, w, b) => (y)
{
    t = MatMul(x, w)
    y = Add(t, b)
}
"""
)
_REPLACEMENT = parser.parse_function(
    """
<domain: "com.onnxsim.test", opset_import: ["" : 18]>
gemm_replacement (x, w, b) => (y)
{
    y = Gemm(x, w, b)
}
"""
)


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _vi(name, shape):
    return helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)


# ==========================================================================
# Parity: on a plain main-graph pattern, both engines produce the same fusion.
# ==========================================================================
def test_parity_matmul_add_to_gemm():
    def build():
        nodes = [
            helper.make_node("MatMul", ["x", "W"], ["mm"]),
            helper.make_node("Add", ["mm", "B"], ["y"]),
        ]
        inits = [_f32(np.random.randn(4, 5), "W"), _f32(np.random.randn(5), "B")]
        graph = helper.make_graph(
            nodes, "g", [_vi("x", [3, 4])], [_vi("y", [3, 5])], inits
        )
        return helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10
        )

    ours, _ = onnxsim.simplify(
        build(), check_n=0, function_rewrite_rules=[(_PATTERN, _REPLACEMENT)]
    )
    theirs = rewrite(
        build(), pattern_rewrite_rules=_onnxscript_rules(_PATTERN, _REPLACEMENT)
    )

    assert _all_op_types(ours)["Gemm"] == 1
    assert _all_op_types(theirs)["Gemm"] == 1
    # Same structural outcome from the same FunctionProto.
    assert _all_op_types(ours)["MatMul"] == _all_op_types(theirs)["MatMul"] == 0
    assert _all_op_types(ours)["Add"] == _all_op_types(theirs)["Add"] == 0


# ==========================================================================
# Limitation: onnxsim's matcher only traverses the main graph, so a pattern
# living inside an If-branch subgraph is rewritten by onnxscript but NOT by
# onnxsim. This is asserted (not xfail) so the boundary is explicit: if a future
# version starts traversing subgraphs, this test flips and must be updated.
# ==========================================================================
def _model_with_pattern_in_subgraph():
    then_nodes = [
        helper.make_node("MatMul", ["x", "W"], ["mm"]),
        helper.make_node("Add", ["mm", "B"], ["tb"]),
    ]
    then_g = helper.make_graph(then_nodes, "then", [], [_vi("tb", [3, 5])])
    else_g = helper.make_graph(
        [helper.make_node("Identity", ["B2"], ["eb"])], "else", [], [_vi("eb", [3, 5])]
    )
    nodes = [
        helper.make_node("If", ["cond"], ["y"], then_branch=then_g, else_branch=else_g)
    ]
    inits = [
        _f32(np.random.randn(4, 5), "W"),
        _f32(np.random.randn(5), "B"),
        _f32(np.random.randn(3, 5), "B2"),
        _f32(np.random.randn(3, 4), "x"),
    ]
    graph = helper.make_graph(
        nodes,
        "g",
        [helper.make_tensor_value_info("cond", onnx.TensorProto.BOOL, [])],
        [_vi("y", [3, 5])],
        inits,
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10
    )


def test_limitation_subgraph_body_not_traversed():
    # onnxscript rewrites inside the If branch...
    theirs = rewrite(
        _model_with_pattern_in_subgraph(),
        pattern_rewrite_rules=_onnxscript_rules(_PATTERN, _REPLACEMENT),
    )
    theirs_counts = _all_op_types(theirs)
    assert theirs_counts["Gemm"] == 1, theirs_counts
    assert theirs_counts["MatMul"] == 0 and theirs_counts["Add"] == 0, theirs_counts

    # ...but onnxsim's FunctionProto matcher only looks at the main graph, so the
    # MatMul+Add inside the branch is left untouched. (Constant folding / shape
    # inference do not fuse this, so the ops survive.)
    ours, _ = onnxsim.simplify(
        _model_with_pattern_in_subgraph(),
        check_n=0,
        skip_constant_folding=True,
        function_rewrite_rules=[(_PATTERN, _REPLACEMENT)],
    )
    ours_counts = _all_op_types(ours)
    assert ours_counts["Gemm"] == 0, ours_counts
    assert ours_counts["MatMul"] == 1 and ours_counts["Add"] == 1, ours_counts
