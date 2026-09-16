#!/usr/bin/env python3
"""Rewrites that fuse decomposed ops into the forms TIDL's docs prefer.

`tidl_ops.has_decomposed_normalization()` (see that module's docstring)
flags a graph that spells LayerNorm out by hand instead of using the fused
`LayerNormalization` op -- edgeai-tidl-tools' transformer-support notes
recommend the fused form. This module is how you act on that flag: each
rule here recognizes one standard decomposed-op export pattern and rewrites
it into the single fused op, exactly (not approximately) preserving the
graph's numerics.

Unlike `scripts/axera/legalize.py`, none of these rules are motivated by a
real compiler run -- there is no real TIDL toolchain reachable from this
repository to have refused a model and prompted one (see
`scripts/edgeai/README.md`). They exist purely because TI's own published
guidance names the fused ops as preferred; running the rewritten graph
through the real `edgeai-tidl-tools` importer, if one is ever available, is
the only way to confirm it actually changes what the accelerator schedules.

Each rule only rewrites the *default* graph -- it does not recurse into
subgraph attributes (`If`/`Loop` bodies), and only matches the specific
node-for-node shapes real exporters produce (documented per rule below);
anything spelled slightly differently is conservatively left alone rather
than guessed at.

Usage::

    legalize.py in.onnx out.onnx                    # apply every rule
    legalize.py --rules fuse_erf_gelu in.onnx out.onnx
"""

from __future__ import annotations

import argparse
import collections
import itertools
import math

import numpy as np
import onnx
import onnx.shape_inference
from onnx import helper, numpy_helper


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


def fuse_erf_gelu(model: onnx.ModelProto) -> int:
    """`0.5 * x * (1 + Erf(x / sqrt(2)))` -> the fused, exact `Gelu` op.

    This is the graph `torch.onnx.export` gives `nn.GELU()` (its default,
    exact/erf-based mode) before ONNX opset 20 added a native `Gelu` op --
    exactly what `Gelu`'s own default (`approximate="none"`) computes, so
    this is an exact fusion, not the tanh approximation.
    """
    nodes = list(model.graph.node)
    producer = _producer_map(nodes)
    consumers = _consumer_map(nodes)
    to_remove: set = set()
    to_add = []
    changed = 0
    half_sqrt2 = math.sqrt(2.0)

    for erf_node in nodes:
        if erf_node.op_type != "Erf" or id(erf_node) in to_remove:
            continue
        div_node = producer.get(erf_node.input[0])
        if div_node is None or div_node.op_type != "Div":
            continue
        if len(div_node.input) != 2:
            continue
        x_name, sqrt2_name = div_node.input
        sqrt2 = _scalar_constant(model, sqrt2_name)
        if sqrt2 is None or abs(sqrt2 - half_sqrt2) > 1e-4:
            continue

        add_node = next(
            (n for n in consumers.get(erf_node.output[0], []) if n.op_type == "Add"),
            None,
        )
        if add_node is None:
            continue
        one_name = next((i for i in add_node.input if i != erf_node.output[0]), None)
        if _scalar_constant(model, one_name) != 1.0:
            continue

        mul1_node = next(
            (
                n
                for n in consumers.get(add_node.output[0], [])
                if n.op_type == "Mul" and x_name in n.input
            ),
            None,
        )
        if mul1_node is None:
            continue

        mul2_node = next(
            (n for n in consumers.get(mul1_node.output[0], []) if n.op_type == "Mul"),
            None,
        )
        if mul2_node is None:
            continue
        half_name = next((i for i in mul2_node.input if i != mul1_node.output[0]), None)
        if _scalar_constant(model, half_name) != 0.5:
            continue

        chain = [div_node, erf_node, add_node, mul1_node, mul2_node]
        gelu_node = helper.make_node(
            "Gelu",
            [x_name],
            [mul2_node.output[0]],
            name=_unique_name(model, "fused_gelu"),
            approximate="none",
        )
        to_add.append(gelu_node)
        to_remove.update(id(n) for n in chain)
        changed += 1

    if changed:
        _replace_nodes(model, to_remove, to_add)
        _ensure_min_opset(model, 20)
    return changed


RULES = {
    "fuse_decomposed_layernorm": fuse_decomposed_layernorm,
    "fuse_erf_gelu": fuse_erf_gelu,
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
