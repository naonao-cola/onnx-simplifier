#!/usr/bin/env python3
"""Rewrites that replace three ONNX ops absent from TVM v0.8's ONNX-import
convert map (`drp_ai_tvm_ops.DRP_AI_TVM_IMPORTABLE_OPS` -- see that
module's docstring) with the exact primitive-op decomposition each op's own
ONNX spec defines it as. Unlike `scripts/axelera/legalize.py`'s rules (a
documented op used *outside* its accelerator's supported attribute/shape
range), the problem these three solve is different: the op_type itself
isn't in TVM v0.8's frontend at all (`GraphProto.from_onnx()` raises
`tvm.error.OpNotImplemented` for the whole import the moment one appears --
see `drp_ai_tvm_simulator.py`'s docstring), regardless of its attributes.
Rewriting it into ops that *are* in the map is the fix; there is no
attribute value that would make `HardSwish`/`Mish`/`LayerNormalization`
importable as-is.

- **`HardSwish(x)` -> `Mul(x, HardSigmoid(x, alpha=1/6, beta=0.5))`.** The
  ONNX `HardSwish` spec defines it exactly this way: `x * max(0, min(1,
  alpha*x + beta))`, and `HardSigmoid(x, alpha, beta)` is defined as `max(0,
  min(1, alpha*x + beta))` -- the same expression, already its own op.
- **`Mish(x)` -> `Mul(x, Tanh(Softplus(x)))`.** The ONNX `Mish` spec defines
  it as `x * tanh(softplus(x))`, again already three ops that exist
  independently.
- **`LayerNormalization` -> `ReduceMean`/`Sub`/`Mul`/`ReduceMean`/`Add`/
  `Sqrt`/`Div`, plus a trailing `Mul`/`Add` for `Scale`/`B`.** The ONNX
  `LayerNormalization` spec (opset 17) itself defines the op via exactly
  this formula (mean and biased variance over the trailing `rank(X) - axis`
  axes, then `(X - Mean) / Sqrt(Var + epsilon) * Scale + B`) -- this is the
  same decomposition ONNX exporters emitted before opset 17 introduced the
  fused op (TVM v0.8 predates opset 17: TVM v0.8 shipped in 2021,
  `LayerNormalization` in opset 17/2022, so the op simply didn't exist yet
  when TVM v0.8's ONNX frontend was written).

All three are *exact* rewrites (the replacement ops are literally how each
spec defines the original one), not approximations -- verified both
structurally (against the ONNX operator spec's own formula) and numerically
(`onnx.reference.ReferenceEvaluator`, before vs. after) in
`tests/test_renesas_legalize.py`, which also confirms
`drp_ai_tvm_simulator.would_import_succeed()` flips from `False` to `True`
for a graph using each op. None of this has been checked against a real
DRP-AI TVM/TVM v0.8 import (see `tvm_v08_frontend_backend.py` for that);
these three ops simply are not in TVM v0.8's convert map by op_type alone,
so the "would the real frontend accept this op_type" question doesn't need
the real compiler to answer -- see `drp_ai_tvm_ops.py`'s docstring for how
firmly that specific fact is established (a literal scrape of, and now
also a live diff against, TVM v0.8's own `_get_convert_map()`).

`layer_normalization_to_primitives` needs `X`'s rank statically known (to
turn `axis` into `ReduceMean`'s `axes`) and its element type known (to
build an epsilon constant of the matching dtype); a node either lacks
suffices to skip rather than guess. It also skips a node whose optional
`Mean`/`InvStdDev` outputs (opset 17's 2nd/3rd outputs) are actually
consumed -- reproducing those (including `stash_type`'s dtype-upcast
behavior) exactly is out of scope here.

Usage::

    legalize.py in.onnx out.onnx
    legalize.py --rules mish_to_primitives in.onnx out.onnx

Or, inside onnxsim's own simplification fixed point, with no rebuild:
``onnxsim.simplify(model, custom_rewriter=legalize.as_custom_rewriter())``
-- see `as_custom_rewriter()`'s docstring, same contract as
`scripts/axelera/legalize.py`'s.
"""

from __future__ import annotations

import argparse
import collections

import numpy as np
import onnx
import onnx.shape_inference
from onnx import helper, numpy_helper


def _attr(node, name):
    for a in node.attribute:
        if a.name == name:
            return a
    return None


def _attr_int(node, name, default):
    a = _attr(node, name)
    return a.i if a is not None else default


def _attr_float(node, name, default):
    a = _attr(node, name)
    return a.f if a is not None else default


def _unique_name(model, stem):
    taken = (
        {i.name for i in model.graph.initializer}
        | {n.name for n in model.graph.node if n.name}
        | {o for n in model.graph.node for o in n.output}
    )
    name, k = stem, 0
    while name in taken:
        k += 1
        name = f"{stem}_{k}"
    return name


def _value_shapes_and_types(model):
    """`({tensor name: [dims]}, {tensor name: onnx elem_type int})` for
    every value with a fully static shape and a known element type --
    graph inputs/outputs/value_info with every dim resolved, plus
    initializers. Runs shape inference itself, same as
    `scripts/axelera/legalize.py`'s `_value_shapes()`.
    """
    inferred = model
    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    except Exception:
        pass
    shapes, types = {}, {}
    for value in (
        list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    ):
        # `HasField("shape")` first: an *absent* shape (rank not statically
        # known -- e.g. after a deliberate `ClearField("shape")`) and a
        # true rank-0 scalar (`shape.dim` present but empty) both iterate
        # as an empty `dim` list, so skipping the HasField check would read
        # "unknown rank" as "known rank 0" (`all()` over an empty sequence
        # is vacuously True).
        if value.type.tensor_type.HasField("shape"):
            dims = value.type.tensor_type.shape.dim
            if all(d.HasField("dim_value") for d in dims):
                shapes[value.name] = [d.dim_value for d in dims]
        if value.type.tensor_type.elem_type:
            types[value.name] = value.type.tensor_type.elem_type
    for init in model.graph.initializer:
        shapes[init.name] = list(init.dims)
        types[init.name] = init.data_type
    return shapes, types


def _default_domain_opset(model):
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx"):
            return opset.version
    return 1


def _make_reduce_mean(model, opset, data, output, axes, name_stem):
    """`ReduceMean` changed `axes` from an attribute (opset <18) to an
    optional second input tensor (opset >=18, ONNX's axes-as-input
    migration shared with `ReduceSum`/`ReduceMax`/etc.) -- build whichever
    form the model's own opset expects, rather than always emitting the
    older attribute form (which opset >=18's checker rejects outright, not
    just deprecates).
    """
    if opset >= 18:
        axes_name = _unique_name(model, f"{name_stem}_axes")
        model.graph.initializer.append(
            numpy_helper.from_array(np.array(axes, dtype=np.int64), name=axes_name)
        )
        return helper.make_node(
            "ReduceMean", [data, axes_name], [output], keepdims=1, name=name_stem
        )
    return helper.make_node(
        "ReduceMean", [data], [output], axes=axes, keepdims=1, name=name_stem
    )


def hardswish_to_primitives(model):
    """`HardSwish(x)` -> `Mul(x, HardSigmoid(x, alpha=1/6, beta=0.5))` --
    see this module's docstring for why these are the exact same function.
    """
    out, changed = [], 0
    for node in model.graph.node:
        if node.op_type != "HardSwish":
            out.append(node)
            continue
        stem = node.name or node.output[0]
        gated = _unique_name(model, f"{stem}_hsigmoid")
        out.append(
            helper.make_node(
                "HardSigmoid",
                [node.input[0]],
                [gated],
                alpha=1.0 / 6.0,
                beta=0.5,
                name=_unique_name(model, f"{stem}_hardsigmoid"),
            )
        )
        out.append(
            helper.make_node(
                "Mul",
                [node.input[0], gated],
                [node.output[0]],
                name=_unique_name(model, f"{stem}_mul"),
            )
        )
        changed += 1
    if changed:
        del model.graph.node[:]
        model.graph.node.extend(out)
    return changed


def mish_to_primitives(model):
    """`Mish(x)` -> `Mul(x, Tanh(Softplus(x)))` -- see this module's
    docstring for why these are the exact same function.
    """
    out, changed = [], 0
    for node in model.graph.node:
        if node.op_type != "Mish":
            out.append(node)
            continue
        stem = node.name or node.output[0]
        sp = _unique_name(model, f"{stem}_softplus")
        th = _unique_name(model, f"{stem}_tanh")
        out.append(
            helper.make_node(
                "Softplus",
                [node.input[0]],
                [sp],
                name=_unique_name(model, f"{stem}_sp"),
            )
        )
        out.append(
            helper.make_node("Tanh", [sp], [th], name=_unique_name(model, f"{stem}_th"))
        )
        out.append(
            helper.make_node(
                "Mul",
                [node.input[0], th],
                [node.output[0]],
                name=_unique_name(model, f"{stem}_mul"),
            )
        )
        changed += 1
    if changed:
        del model.graph.node[:]
        model.graph.node.extend(out)
    return changed


def layer_normalization_to_primitives(model):
    """`LayerNormalization` -> `ReduceMean`/`Sub`/`Mul`/`ReduceMean`/`Add`/
    `Sqrt`/`Div` (+ trailing `Mul`/`Add` for `Scale`/`B`) -- see this
    module's docstring for the formula this reproduces and the two
    conditions (`X`'s rank and element type both statically known, optional
    `Mean`/`InvStdDev` outputs unused) a node needs to qualify.
    """
    shapes, types = _value_shapes_and_types(model)
    out, changed = [], 0
    for node in model.graph.node:
        if node.op_type != "LayerNormalization":
            out.append(node)
            continue
        if len(node.output) > 1 and any(node.output[1:]):
            out.append(node)  # Mean/InvStdDev consumed downstream -- skip
            continue
        x = node.input[0]
        xshape, elem_type = shapes.get(x), types.get(x)
        if xshape is None or not elem_type:
            out.append(node)
            continue
        rank = len(xshape)
        axis = _attr_int(node, "axis", -1)
        if axis < 0:
            axis += rank
        axes = list(range(axis, rank))
        epsilon = _attr_float(node, "epsilon", 1e-05)

        stem = node.name or node.output[0]

        def n(suffix):
            return _unique_name(model, f"{stem}_{suffix}")

        np_dtype = helper.tensor_dtype_to_np_dtype(elem_type)
        eps_name = n("epsilon")
        model.graph.initializer.append(
            numpy_helper.from_array(np.array(epsilon, dtype=np_dtype), name=eps_name)
        )

        mean, centered, sq, var, var_eps, std, normed = (
            n("mean"),
            n("centered"),
            n("sq"),
            n("var"),
            n("var_eps"),
            n("std"),
            n("normed"),
        )
        opset = _default_domain_opset(model)
        out.append(_make_reduce_mean(model, opset, x, mean, axes, n("reduce_mean")))
        out.append(helper.make_node("Sub", [x, mean], [centered], name=n("sub")))
        out.append(
            helper.make_node("Mul", [centered, centered], [sq], name=n("square"))
        )
        out.append(_make_reduce_mean(model, opset, sq, var, axes, n("reduce_var")))
        out.append(
            helper.make_node("Add", [var, eps_name], [var_eps], name=n("add_eps"))
        )
        out.append(helper.make_node("Sqrt", [var_eps], [std], name=n("sqrt")))

        scale = node.input[1] if len(node.input) > 1 and node.input[1] else None
        bias = node.input[2] if len(node.input) > 2 and node.input[2] else None
        y = node.output[0]

        if scale is None and bias is None:
            out.append(helper.make_node("Div", [centered, std], [y], name=n("div")))
        elif bias is None:
            out.append(
                helper.make_node("Div", [centered, std], [normed], name=n("div"))
            )
            out.append(
                helper.make_node("Mul", [normed, scale], [y], name=n("mul_scale"))
            )
        elif scale is None:
            out.append(
                helper.make_node("Div", [centered, std], [normed], name=n("div"))
            )
            out.append(helper.make_node("Add", [normed, bias], [y], name=n("add_bias")))
        else:
            scaled = n("scaled")
            out.append(
                helper.make_node("Div", [centered, std], [normed], name=n("div"))
            )
            out.append(
                helper.make_node("Mul", [normed, scale], [scaled], name=n("mul_scale"))
            )
            out.append(helper.make_node("Add", [scaled, bias], [y], name=n("add_bias")))
        changed += 1
    if changed:
        del model.graph.node[:]
        model.graph.node.extend(out)
    return changed


#: Order doesn't matter between these three -- each only touches its own
#: op_type, and none produces a node another consumes.
RULES = {
    "hardswish_to_primitives": hardswish_to_primitives,
    "mish_to_primitives": mish_to_primitives,
    "layer_normalization_to_primitives": layer_normalization_to_primitives,
}


def legalize(model, rules=None):
    """Apply the named rules in order; returns `{rule: sites changed}`."""
    applied = collections.OrderedDict()
    for name in rules or RULES:
        applied[name] = RULES[name](model)
    return applied


def as_custom_rewriter(rules=None):
    """A callable usable as ``onnxsim.simplify(model, custom_rewriter=...)``
    -- see `scripts/axelera/legalize.py`'s `as_custom_rewriter()` for the
    full contract this matches (identical adapter, different `RULES`).
    """

    def rewriter(model):
        applied = legalize(model, rules)
        return None if any(applied.values()) else False

    return rewriter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--rules", nargs="*", choices=sorted(RULES))
    args = parser.parse_args(argv)

    model = onnx.load(args.input)
    for name, count in legalize(model, args.rules).items():
        print(f"  {name}: {count} sites")
    onnx.save(
        model,
        args.output,
        save_as_external_data=True,
        location=args.output.rsplit("/", 1)[-1] + ".data",
        size_threshold=1024,
    )
    print("wrote", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
