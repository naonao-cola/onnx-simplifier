"""Legalize ONNX ops nncase's importer doesn't support into ones it does.

nncase 1.9.0's ONNX importer (see its `opcode.def`) covers roughly half of
the ai.onnx domain -- perfectly fine for the CNN/RNN-shaped models this repo
targets, but Hugging Face exports increasingly use newer elementwise/norm
ops (`Gelu`, `Swish`, `Mish`, `MeanVarianceNormalization`, ...) that aren't
in that list and fail import with "Not supported ONNX opcode: ...". Most of
these have no special hardware meaning -- they're just a named shorthand for
a small arithmetic expression built entirely out of ops nncase *does*
support (`Add`, `Mul`, `Sigmoid`, `Tanh`, `Erf`, `Equal`, `And`, `Not`, ...).
This module rewrites the graph to spell those expressions out before nncase
ever sees them, node-for-node, so the compiled kmodel is mathematically
identical to the original graph -- not an approximation of it.

Each decomposition replaces exactly one node with a small chain of nodes
that produces the *same output tensor name* as the original, so anything
downstream stays wired up automatically. Constants the expression needs
(0.5, sqrt(2), an op's own attribute values, ...) are added as `Constant`
nodes scoped to that one node's replacement, not as graph-level
initializers, so repeated legalization of the same op never collides.

Left out on purpose:
- `GroupNormalization`, `Mod`: real decompositions exist, but they're
  bigger (reshape-heavy, or semantics-attribute-dependent) and no model
  this repo has actually needed yet has hit them. Add them here the same
  way if one does.
- Control-flow (`If`, `Loop`, `Scan`), sequence ops, string ops: nncase's
  importer -- and its whole static-shape compilation model -- has no
  equivalent to decompose these *into*; they're not a "missing shorthand"
  the way the ops below are.
"""

from __future__ import annotations

import itertools
import math

import onnx
from onnx import TensorProto, helper

_counter = itertools.count()


def _uid(prefix: str) -> str:
    return f"{prefix}__legalized_{next(_counter)}"


def _const(prefix: str, value: float) -> tuple[onnx.NodeProto, str]:
    """A scalar float32 Constant node (broadcasts against any input shape)."""
    name = _uid(prefix)
    node = helper.make_node(
        "Constant", [], [name], name=name,
        value=helper.make_tensor(name + "_value", TensorProto.FLOAT, [], [value]),
    )
    return node, name


def _get_attr(node: onnx.NodeProto, name: str, default):
    for attr in node.attribute:
        if attr.name == name:
            return helper.get_attribute_value(attr)
    return default


def _legalize_reciprocal(node: onnx.NodeProto) -> list[onnx.NodeProto]:
    # Reciprocal(x) = Pow(x, -1)
    minus_one, minus_one_name = _const(node.name or "Reciprocal", -1.0)
    pow_node = helper.make_node("Pow", [node.input[0], minus_one_name], [node.output[0]], name=_uid("Pow"))
    return [minus_one, pow_node]


def _legalize_gelu(node: onnx.NodeProto) -> list[onnx.NodeProto]:
    x = node.input[0]
    approximate = _get_attr(node, "approximate", b"none")
    if isinstance(approximate, bytes):
        approximate = approximate.decode()
    prefix = node.name or "Gelu"

    if approximate == "tanh":
        # 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        c_a, c_a_n = _const(prefix, math.sqrt(2.0 / math.pi))
        c_b, c_b_n = _const(prefix, 0.044715)
        c_half, c_half_n = _const(prefix, 0.5)
        c_one, c_one_n = _const(prefix, 1.0)
        c_three, c_three_n = _const(prefix, 3.0)
        cube = helper.make_node("Pow", [x, c_three_n], [_uid("x3")], name=_uid("Pow"))
        scaled_cube = helper.make_node("Mul", [cube.output[0], c_b_n], [_uid("bx3")], name=_uid("Mul"))
        inner_sum = helper.make_node("Add", [x, scaled_cube.output[0]], [_uid("sum")], name=_uid("Add"))
        scaled = helper.make_node("Mul", [inner_sum.output[0], c_a_n], [_uid("scaled")], name=_uid("Mul"))
        tanh = helper.make_node("Tanh", [scaled.output[0]], [_uid("tanh")], name=_uid("Tanh"))
        plus_one = helper.make_node("Add", [tanh.output[0], c_one_n], [_uid("plus1")], name=_uid("Add"))
        times_x = helper.make_node("Mul", [x, plus_one.output[0]], [_uid("xtimes")], name=_uid("Mul"))
        half_times = helper.make_node("Mul", [times_x.output[0], c_half_n], [node.output[0]], name=_uid("Mul"))
        return [c_three, c_b, c_half, c_one, c_a, cube, scaled_cube, inner_sum, scaled, tanh, plus_one, times_x, half_times]

    # exact: 0.5 * x * (1 + erf(x / sqrt(2)))
    c_inv_sqrt2, c_inv_sqrt2_n = _const(prefix, 1.0 / math.sqrt(2.0))
    c_half, c_half_n = _const(prefix, 0.5)
    c_one, c_one_n = _const(prefix, 1.0)
    scaled = helper.make_node("Mul", [x, c_inv_sqrt2_n], [_uid("scaled")], name=_uid("Mul"))
    erf = helper.make_node("Erf", [scaled.output[0]], [_uid("erf")], name=_uid("Erf"))
    plus_one = helper.make_node("Add", [erf.output[0], c_one_n], [_uid("plus1")], name=_uid("Add"))
    times_x = helper.make_node("Mul", [x, plus_one.output[0]], [_uid("xtimes")], name=_uid("Mul"))
    half_times = helper.make_node("Mul", [times_x.output[0], c_half_n], [node.output[0]], name=_uid("Mul"))
    return [c_inv_sqrt2, c_half, c_one, scaled, erf, plus_one, times_x, half_times]


def _legalize_swish(node: onnx.NodeProto) -> list[onnx.NodeProto]:
    # Swish(x) = x * Sigmoid(alpha * x)
    x = node.input[0]
    alpha = _get_attr(node, "alpha", 1.0)
    prefix = node.name or "Swish"
    c_alpha, c_alpha_n = _const(prefix, alpha)
    scaled = helper.make_node("Mul", [x, c_alpha_n], [_uid("scaled")], name=_uid("Mul"))
    sig = helper.make_node("Sigmoid", [scaled.output[0]], [_uid("sig")], name=_uid("Sigmoid"))
    out = helper.make_node("Mul", [x, sig.output[0]], [node.output[0]], name=_uid("Mul"))
    return [c_alpha, scaled, sig, out]


def _legalize_mish(node: onnx.NodeProto) -> list[onnx.NodeProto]:
    # Mish(x) = x * Tanh(Softplus(x))
    x = node.input[0]
    softplus = helper.make_node("Softplus", [x], [_uid("softplus")], name=_uid("Softplus"))
    tanh = helper.make_node("Tanh", [softplus.output[0]], [_uid("tanh")], name=_uid("Tanh"))
    out = helper.make_node("Mul", [x, tanh.output[0]], [node.output[0]], name=_uid("Mul"))
    return [softplus, tanh, out]


def _legalize_or(node: onnx.NodeProto) -> list[onnx.NodeProto]:
    # Or(a, b) = Not(And(Not(a), Not(b)))
    a, b = node.input
    not_a = helper.make_node("Not", [a], [_uid("not_a")], name=_uid("Not"))
    not_b = helper.make_node("Not", [b], [_uid("not_b")], name=_uid("Not"))
    both_not = helper.make_node("And", [not_a.output[0], not_b.output[0]], [_uid("nor")], name=_uid("And"))
    out = helper.make_node("Not", [both_not.output[0]], [node.output[0]], name=_uid("Not"))
    return [not_a, not_b, both_not, out]


def _legalize_xor(node: onnx.NodeProto) -> list[onnx.NodeProto]:
    # Xor(a, b) = And(Or(a, b), Not(And(a, b)))
    a, b = node.input
    prefix = node.name or "Xor"
    or_nodes = _legalize_or(helper.make_node("Or", [a, b], [_uid(prefix + "_or")], name=_uid("Or")))
    and_ab = helper.make_node("And", [a, b], [_uid("and_ab")], name=_uid("And"))
    not_and_ab = helper.make_node("Not", [and_ab.output[0]], [_uid("not_and_ab")], name=_uid("Not"))
    out = helper.make_node("And", [or_nodes[-1].output[0], not_and_ab.output[0]], [node.output[0]], name=_uid("And"))
    return [*or_nodes, and_ab, not_and_ab, out]


def _legalize_isnan(node: onnx.NodeProto) -> list[onnx.NodeProto]:
    # IsNaN(x) = Not(Equal(x, x))  -- NaN is the only float that isn't equal to itself
    x = node.input[0]
    eq = helper.make_node("Equal", [x, x], [_uid("eq_self")], name=_uid("Equal"))
    out = helper.make_node("Not", [eq.output[0]], [node.output[0]], name=_uid("Not"))
    return [eq, out]


def _legalize_isinf(node: onnx.NodeProto) -> list[onnx.NodeProto]:
    # IsInf(x) = (detect_positive and x == +inf) or (detect_negative and x == -inf)
    x = node.input[0]
    prefix = node.name or "IsInf"
    detect_negative = _get_attr(node, "detect_negative", 1)
    detect_positive = _get_attr(node, "detect_positive", 1)

    nodes: list[onnx.NodeProto] = []
    checks: list[str] = []
    if detect_positive:
        c_pos, c_pos_n = _const(prefix, math.inf)
        eq_pos = helper.make_node("Equal", [x, c_pos_n], [_uid("eq_pos")], name=_uid("Equal"))
        nodes += [c_pos, eq_pos]
        checks.append(eq_pos.output[0])
    if detect_negative:
        c_neg, c_neg_n = _const(prefix, -math.inf)
        eq_neg = helper.make_node("Equal", [x, c_neg_n], [_uid("eq_neg")], name=_uid("Equal"))
        nodes += [c_neg, eq_neg]
        checks.append(eq_neg.output[0])

    if not checks:
        # Neither direction requested: always false.
        c_zero, c_zero_n = _const(prefix, 0.0)
        c_one, c_one_n = _const(prefix, 1.0)
        always_false = helper.make_node("Equal", [c_zero_n, c_one_n], [node.output[0]], name=_uid("Equal"))
        return [c_zero, c_one, always_false]
    if len(checks) == 1:
        # Re-point the one check's output at this node's real output name instead of
        # its throwaway intermediate one.
        last = nodes[-1]
        nodes[-1] = helper.make_node(last.op_type, list(last.input), [node.output[0]], name=last.name)
        return nodes

    or_nodes = _legalize_or(helper.make_node("Or", checks, [node.output[0]], name=_uid("Or")))
    return [*nodes, *or_nodes]


def _legalize_meanvariancenormalization(node: onnx.NodeProto) -> list[onnx.NodeProto]:
    # MVN(x) = (x - mean(x, axes)) / sqrt(mean((x - mean(x, axes))^2, axes) + eps)
    #
    # ReduceMean's `axes` is deliberately emitted as an attribute (pre-opset-18 form),
    # not opset 18's second input tensor: nncase's own importer
    # (src/importer/onnx/ops/reduce.cpp) only ever reads `axes` as an attribute and
    # silently reduces over every axis otherwise. That makes this decomposition itself
    # invalid per onnx's checker under a declared opset >= 18 -- harmless for nncase,
    # which has no opset-version check here, but worth knowing if this ever needs to
    # pass a generic ONNX runtime instead.
    x = node.input[0]
    axes = list(_get_attr(node, "axes", [0, 2, 3]))
    prefix = node.name or "MeanVarianceNormalization"
    eps = 1e-9  # matches onnxruntime's own MVN implementation

    mean = helper.make_node("ReduceMean", [x], [_uid("mean")], name=_uid("ReduceMean"), axes=axes, keepdims=1)
    centered = helper.make_node("Sub", [x, mean.output[0]], [_uid("centered")], name=_uid("Sub"))
    squared = helper.make_node("Mul", [centered.output[0], centered.output[0]], [_uid("sq")], name=_uid("Mul"))
    var = helper.make_node("ReduceMean", [squared.output[0]], [_uid("var")], name=_uid("ReduceMean"), axes=axes, keepdims=1)
    c_eps, c_eps_n = _const(prefix, eps)
    var_eps = helper.make_node("Add", [var.output[0], c_eps_n], [_uid("var_eps")], name=_uid("Add"))
    std = helper.make_node("Sqrt", [var_eps.output[0]], [_uid("std")], name=_uid("Sqrt"))
    out = helper.make_node("Div", [centered.output[0], std.output[0]], [node.output[0]], name=_uid("Div"))
    return [mean, centered, squared, var, c_eps, var_eps, std, out]


_LEGALIZERS = {
    "Reciprocal": _legalize_reciprocal,
    "Gelu": _legalize_gelu,
    "Swish": _legalize_swish,
    "Mish": _legalize_mish,
    "Or": _legalize_or,
    "Xor": _legalize_xor,
    "IsNaN": _legalize_isnan,
    "IsInf": _legalize_isinf,
    "MeanVarianceNormalization": _legalize_meanvariancenormalization,
}

#: ops this module knows how to legalize, for callers that want to check
#: ahead of time (e.g. to decide whether legalization is even needed).
SUPPORTED_OPS = frozenset(_LEGALIZERS)


def legalize(model: onnx.ModelProto) -> onnx.ModelProto:
    """Rewrite every node whose op_type is in `SUPPORTED_OPS` into an
    equivalent chain of nncase-supported ops. Nodes nncase already
    supports are left untouched. Does not run shape inference itself --
    callers that need intermediate value_info (nncase's importer does,
    see onnx_to_kmodel.py) should re-run it afterward.
    """
    graph = model.graph
    new_nodes: list[onnx.NodeProto] = []
    changed = False
    for node in graph.node:
        legalizer = _LEGALIZERS.get(node.op_type)
        if legalizer is None:
            new_nodes.append(node)
            continue
        new_nodes.extend(legalizer(node))
        changed = True

    if changed:
        del graph.node[:]
        graph.node.extend(new_nodes)
    return model
