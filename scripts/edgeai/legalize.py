#!/usr/bin/env python3
"""Rewrites that steer a graph toward the forms TIDL's real importer wants.

`tidl_ops.has_decomposed_normalization()` (see that module's docstring)
flags a graph that spells LayerNorm out by hand instead of using the fused
`LayerNormalization` op. This module is how you act on that flag -- and,
having fetched edgeai-tidl-tools' actual `docs/operators.md` and
`docs/vision_transformers.md` (raw.githubusercontent.com is reachable even
though the interactive `github.com` repo page and `software-dl.ti.com`'s
binary downloads are not, see `scripts/edgeai/README.md`), the two rules
here point in *different* directions, not the same one:

- `fuse_decomposed_layernorm` fuses *toward* a single op:
  `docs/operators.md` lists `LayerNormalization` as its own directly
  supported layer (`TIDL_LayerNormLayer`), so collapsing the decomposed
  `ReduceMean`/`Sub`/`Pow`/`Sqrt`/`Div` chain into that op is what the real
  importer wants.
- `unfuse_gelu_to_erf` fuses *away* from a single op, in the opposite
  direction of what an earlier version of this module did. `docs/
  operators.md` has no `Gelu` entry at all -- only `Erf`/`Identity`,
  explicitly "not supported as an individual operator... only supported as
  part of the fused combination of GELU" (`docs/vision_transformers.md`'s
  own GELU section: the importer pattern-matches the decomposed
  `Div`/`Erf`/`Add`/`Mul`/`Mul` sequence itself and maps it to TIDL's
  internal BatchNorm-with-activation layer). A literal ONNX `Gelu` node
  (opset 20+) is exactly what this rule un-fuses back into that sequence,
  since the real importer has nothing else to do with it otherwise.

Neither rule's target has been confirmed against a real compile -- there is
no real TIDL toolchain reachable from this repository to run one (see
`scripts/edgeai/README.md`) -- but both are now checked against the actual
published operator-support doc rather than reconstructed from memory, which
is what caught the GELU direction being backwards in the first place.

Each rule only rewrites the *default* graph -- it does not recurse into
subgraph attributes (`If`/`Loop` bodies), and only matches the specific
node-for-node shapes real exporters produce (documented per rule below);
anything spelled slightly differently is conservatively left alone rather
than guessed at.

Usage::

    legalize.py in.onnx out.onnx                    # apply every rule
    legalize.py --rules unfuse_gelu_to_erf in.onnx out.onnx
"""

from __future__ import annotations

import argparse
import collections
import itertools
import math

import numpy as np
import onnx
import onnx.shape_inference
from onnx import TensorProto, helper, numpy_helper


def _producer_map(nodes):
    m = {}
    for n in nodes:
        for o in n.output:
            if o:
                m[o] = n
    return m


def _consumer_map(nodes):
    m = collections.defaultdict(list)
    for n in nodes:
        for i in n.input:
            if i:
                m[i].append(n)
    return m


def _initializer(model, name):
    for init in model.graph.initializer:
        if init.name == name:
            return init
    return None


def _scalar_constant(model, name):
    """The scalar value of `name` if it is a constant, else None."""
    init = _initializer(model, name)
    if init is not None:
        arr = numpy_helper.to_array(init)
        return float(arr.reshape(-1)[0]) if arr.size == 1 else None
    for node in model.graph.node:
        if node.op_type == "Constant" and node.output and node.output[0] == name:
            for attr in node.attribute:
                if attr.name == "value":
                    arr = numpy_helper.to_array(attr.t)
                    return float(arr.reshape(-1)[0]) if arr.size == 1 else None
    return None


def _reduce_mean_axes(node):
    """`ReduceMean`'s reduced axes, or None if expressed as an input (opset
    18+'s `axes` moved from an attribute to an optional second input --
    unsupported here, see this module's docstring)."""
    for attr in node.attribute:
        if attr.name == "axes":
            return list(attr.ints)
    return None


def _keepdims(node, default=1):
    for attr in node.attribute:
        if attr.name == "keepdims":
            return attr.i
    return default


def _unique_name(model, stem):
    existing = {n for node in model.graph.node for n in (*node.input, *node.output)}
    existing |= {init.name for init in model.graph.initializer}
    for i in itertools.count():
        candidate = f"{stem}_{i}"
        if candidate not in existing:
            return candidate


def _last_dims(model):
    """`{tensor_name: static_size_of_its_last_dim}` for every tensor whose
    last dimension is statically known.

    Runs shape inference on a *copy* -- not `model` itself, since a rule
    below snapshots `model.graph.node` by object identity before rewriting;
    inferring in place would replace those nodes with new objects and break
    that identity tracking (confirmed: silently left the old nodes in place
    alongside the new fused one, duplicating an output name).
    """
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
    except Exception:
        inferred = model
    dims = {}
    for value in (
        *inferred.graph.input,
        *inferred.graph.value_info,
        *inferred.graph.output,
    ):
        if not value.type.HasField("tensor_type"):
            continue
        shape = value.type.tensor_type.shape.dim
        if shape and shape[-1].HasField("dim_value"):
            dims[value.name] = shape[-1].dim_value
    return dims


def _ensure_min_opset(model, min_version, domain=""):
    for opset in model.opset_import:
        if opset.domain == domain:
            if opset.version < min_version:
                opset.version = min_version
            return
    model.opset_import.append(helper.make_opsetid(domain, min_version))


def _replace_nodes(model, to_remove, to_add):
    kept = [n for n in model.graph.node if id(n) not in to_remove]
    del model.graph.node[:]
    model.graph.node.extend(kept)
    model.graph.node.extend(to_add)


def fuse_decomposed_layernorm(model: onnx.ModelProto) -> int:
    """`ReduceMean`/`Sub`/`Pow(2)`/`ReduceMean`/`Add`/`Sqrt`/`Div` -> `LayerNormalization`.

    Matches the standard hand-written LayerNorm export over the last axis
    only (`ReduceMean`'s `axes == [-1]`, `keepdims=1`), the shape
    `tests/test_edgeai_tidl_compat.py::
    test_decomposed_layer_norm_flagged_as_normalization_risk`'s fixture
    uses and `tidl_ops.has_decomposed_normalization()` flags. Also folds a
    trailing `Mul(scale)`/`Add(bias)` pair into `LayerNormalization`'s own
    scale/bias inputs when present (the full affine form); otherwise
    synthesizes ones/zeros so the rewritten op is still well-formed. Exact,
    not approximate: `LayerNormalization`'s own formula is this same
    `(x - mean) / sqrt(var + eps) * scale + bias`.
    """
    last_dims = _last_dims(model)
    nodes = list(model.graph.node)
    producer = _producer_map(nodes)
    consumers = _consumer_map(nodes)
    to_remove: set = set()
    to_add = []
    changed = 0

    for sub_node in nodes:
        if sub_node.op_type != "Sub" or id(sub_node) in to_remove:
            continue
        if len(sub_node.input) != 2:
            continue
        x_name, mean_name = sub_node.input
        mean_node = producer.get(mean_name)
        if (
            mean_node is None
            or mean_node.op_type != "ReduceMean"
            or id(mean_node) in to_remove
        ):
            continue
        if mean_node.input[0] != x_name:
            continue
        if _reduce_mean_axes(mean_node) != [-1] or _keepdims(mean_node) != 1:
            continue

        centered = sub_node.output[0]
        cons = consumers.get(centered, [])
        pow_node = next((n for n in cons if n.op_type == "Pow"), None)
        div_node = next((n for n in cons if n.op_type == "Div"), None)
        if pow_node is None or div_node is None:
            continue
        if _scalar_constant(model, pow_node.input[1]) != 2.0:
            continue

        var_node = next(
            (
                n
                for n in consumers.get(pow_node.output[0], [])
                if n.op_type == "ReduceMean"
            ),
            None,
        )
        if (
            var_node is None
            or _reduce_mean_axes(var_node) != [-1]
            or _keepdims(var_node) != 1
        ):
            continue

        add_node = next(
            (n for n in consumers.get(var_node.output[0], []) if n.op_type == "Add"),
            None,
        )
        if add_node is None:
            continue
        eps_name = next((i for i in add_node.input if i != var_node.output[0]), None)
        eps = _scalar_constant(model, eps_name) if eps_name else None
        if eps is None:
            continue

        sqrt_node = next(
            (n for n in consumers.get(add_node.output[0], []) if n.op_type == "Sqrt"),
            None,
        )
        if sqrt_node is None:
            continue
        if set(div_node.input) != {centered, sqrt_node.output[0]}:
            continue

        chain = [mean_node, sub_node, pow_node, var_node, add_node, sqrt_node, div_node]
        y_name = div_node.output[0]

        scale_name = None
        bias_name = None
        mul_node = next(
            (n for n in consumers.get(y_name, []) if n.op_type == "Mul"), None
        )
        if mul_node is not None:
            candidate_scale = next((i for i in mul_node.input if i != y_name), None)
            if candidate_scale and _initializer(model, candidate_scale) is not None:
                add2_node = next(
                    (
                        n
                        for n in consumers.get(mul_node.output[0], [])
                        if n.op_type == "Add"
                    ),
                    None,
                )
                if add2_node is not None:
                    candidate_bias = next(
                        (i for i in add2_node.input if i != mul_node.output[0]), None
                    )
                    if (
                        candidate_bias
                        and _initializer(model, candidate_bias) is not None
                    ):
                        chain += [mul_node, add2_node]
                        scale_name, bias_name = candidate_scale, candidate_bias
                        y_name = add2_node.output[0]

        if scale_name is None:
            dim = last_dims.get(x_name)
            if dim is None:
                continue
            scale_name = _unique_name(model, "ln_scale")
            model.graph.initializer.append(
                numpy_helper.from_array(np.ones(dim, np.float32), scale_name)
            )
            bias_name = _unique_name(model, "ln_bias")
            model.graph.initializer.append(
                numpy_helper.from_array(np.zeros(dim, np.float32), bias_name)
            )

        ln_node = helper.make_node(
            "LayerNormalization",
            [x_name, scale_name, bias_name],
            [y_name],
            name=_unique_name(model, "fused_layer_norm"),
            axis=-1,
            epsilon=eps,
        )
        to_add.append(ln_node)
        to_remove.update(id(n) for n in chain)
        changed += 1

    if changed:
        _replace_nodes(model, to_remove, to_add)
        _ensure_min_opset(model, 17)
    return changed


def _elem_types(model):
    """`{tensor_name: onnx.TensorProto element type}` for every tensor whose
    type is statically known -- same non-mutating-copy approach as
    `_last_dims`, for the same reason."""
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
    except Exception:
        inferred = model
    types = {}
    for value in (
        *inferred.graph.input,
        *inferred.graph.value_info,
        *inferred.graph.output,
    ):
        if value.type.HasField("tensor_type"):
            types[value.name] = value.type.tensor_type.elem_type
    for init in model.graph.initializer:
        types[init.name] = init.data_type
    return types


def _attr_str(node, name, default=""):
    for attr in node.attribute:
        if attr.name == name:
            return attr.s.decode() if isinstance(attr.s, bytes) else attr.s
    return default


def unfuse_gelu_to_erf(model: onnx.ModelProto) -> int:
    """The fused, exact `Gelu` op (`approximate="none"`) -> `0.5 * x * (1 +
    Erf(x / sqrt(2)))`.

    The opposite direction of what an earlier version of this rule did --
    see this module's docstring for why: `docs/operators.md` has no `Gelu`
    entry at all, only `Erf`/`Identity`, "not supported as an individual
    operator... only supported as part of the fused combination of GELU" --
    the real importer pattern-matches this exact decomposed sequence itself
    and maps it to TIDL's internal BatchNorm-with-activation layer, so a
    literal `Gelu` node has nothing to match. Only the exact/erf-based
    approximation mode is unfused: `approximate="tanh"` computes a
    different formula, not this one, so it is conservatively left alone
    (ONNX's own default for `approximate`, when the attribute is absent, is
    `"none"`).

    This formula is *also* independently confirmed against ONNX's own
    schema-defined `Gelu` decomposition (`onnx.defs.get_schema("Gelu",
    ...).get_context_dependent_function_with_opset_version(...)`, inlined
    via `onnx.inliner.inline_local_functions()` -- the same mechanism
    `scripts/renesas/legalize.py::legalize_via_onnx_function` uses) in
    `tests/test_edgeai_legalize.py::
    test_unfuse_gelu_to_erf_matches_onnx_schema_function_decomposition`.
    That schema-derived form is deliberately *not* what this rule emits,
    though: it's a structurally different, `Constant`/`CastLike`/`Sqrt`/
    `Sum`-heavy 12-node sequence (ONNX's own formal spec artifact, opset
    20+), not the 5-node `Div`/`Erf`/`Add`/`Mul`/`Mul` shape real exporters
    (and, per `docs/vision_transformers.md`'s image-only GELU diagram --
    not literal text -- most likely TIDL's real importer) actually expect.
    Producing the schema-derived shape here would risk silently defeating
    this rule's whole purpose; extracting it is useful only as an
    independent check that this hand-written formula computes what ONNX's
    own spec says `Gelu` means, not as the rewrite's output.
    """
    elem_types = _elem_types(model)
    nodes = list(model.graph.node)
    to_remove: set = set()
    to_add = []
    changed = 0
    half_sqrt2 = math.sqrt(2.0)

    for node in nodes:
        if node.op_type != "Gelu" or _attr_str(node, "approximate", "none") != "none":
            continue

        x_name = node.input[0]
        y_name = node.output[0]
        np_dtype = helper.tensor_dtype_to_np_dtype(
            elem_types.get(x_name, TensorProto.FLOAT)
        )

        sqrt2_name = _unique_name(model, "gelu_sqrt2")
        one_name = _unique_name(model, "gelu_one")
        half_name = _unique_name(model, "gelu_half")
        model.graph.initializer.extend(
            [
                numpy_helper.from_array(np.array(half_sqrt2, np_dtype), sqrt2_name),
                numpy_helper.from_array(np.array(1.0, np_dtype), one_name),
                numpy_helper.from_array(np.array(0.5, np_dtype), half_name),
            ]
        )
        stem = _unique_name(model, "gelu_unfused")
        t0, t1, t2, t3 = (f"{stem}_t{i}" for i in range(4))
        to_add.extend(
            [
                helper.make_node("Div", [x_name, sqrt2_name], [t0], name=f"{stem}_div"),
                helper.make_node("Erf", [t0], [t1], name=f"{stem}_erf"),
                helper.make_node("Add", [t1, one_name], [t2], name=f"{stem}_add"),
                helper.make_node("Mul", [x_name, t2], [t3], name=f"{stem}_mul1"),
                helper.make_node("Mul", [t3, half_name], [y_name], name=f"{stem}_mul2"),
            ]
        )
        to_remove.add(id(node))
        changed += 1

    if changed:
        _replace_nodes(model, to_remove, to_add)
    return changed


RULES = {
    "fuse_decomposed_layernorm": fuse_decomposed_layernorm,
    "unfuse_gelu_to_erf": unfuse_gelu_to_erf,
}


def legalize(model: onnx.ModelProto, rules=None) -> dict:
    """Apply each named rule (default: all of them) in `RULES`'s order.

    Returns `{rule_name: rewrite_count}`. Mutates `model` in place, same
    convention as each rule function.
    """
    selected = rules if rules else list(RULES)
    return {name: RULES[name](model) for name in selected}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument(
        "--rules",
        nargs="*",
        default=None,
        choices=list(RULES),
        help="subset of rules to apply (default: all of them)",
    )
    args = ap.parse_args(argv)

    model = onnx.load(args.input)
    counts = legalize(model, args.rules)
    onnx.checker.check_model(model)
    onnx.save(model, args.output)
    for name, count in counts.items():
        print(f"{name}: {count} rewrite(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
