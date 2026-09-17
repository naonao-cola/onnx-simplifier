#!/usr/bin/env python3
"""Rewrites an unimportable ONNX op (any op_type absent from TVM v0.8's
ONNX-import convert map, `drp_ai_tvm_ops.DRP_AI_TVM_IMPORTABLE_OPS`) into
the primitive-op decomposition ONNX's *own operator schema* already defines
for it, when one exists -- extracted straight from `onnx.defs.get_schema()`
and inlined with ONNX's own `onnx.inliner.inline_local_functions()`. This
used to be three hand-transcribed rewrites (one Python function per op,
each manually reproducing that op's spec formula); this file no longer
transcribes anything -- ONNX ships the decomposition as data
(`OpSchema.function_body` / `get_context_dependent_function(...)`, a real
`FunctionProto`), and `onnx.inliner` is the same tool ONNX's own reference
evaluator and backend test suite use to expand a function-defined op into
primitives (see `onnx/reference/ops/_op_list.py` and `onnx/backend/test/
case/node/__init__.py`, which call the identical two schema methods this
file does). There is no hand-derived math left to get wrong.

Two kinds of schema-level function exist (`OpSchema.has_function` and
`OpSchema.has_context_dependent_function`):

- **`has_function`**: a single, fixed `FunctionProto` per opset version,
  independent of any particular call site -- e.g. `HardSwish`
  (`x * HardSigmoid(x, alpha=1/6, beta=0.5)`) and `Mish`
  (`x * Tanh(Softplus(x))`). `get_function_with_opset_version(version)`
  returns it directly.
- **`has_context_dependent_function`**: the body depends on the actual
  node -- its attributes and its inputs' element types -- so it's built via
  `get_context_dependent_function_with_opset_version(version, node_bytes,
  input_type_bytes)`, given the real `NodeProto` and a `TypeProto` per
  input. `LayerNormalization` is this kind: the returned function already
  bakes in the node's actual `axis`/`epsilon` and handles `stash_type`'s
  dtype-upcast-then-downcast behavior and arbitrary/dynamic rank via its
  own `Shape`/`Slice`/`Reshape` machinery -- genuinely more robust than a
  hand-rolled version that would need `X`'s rank statically known to build
  `ReduceMean`'s `axes` (confirmed empirically while replacing this file's
  old version: the schema-derived function round-trips correctly through
  `onnx.reference.ReferenceEvaluator` for float32 inputs, with and without
  `B`, with a non-default `axis`, and with `Mean`/`InvStdDev` requested).

`legalize_via_onnx_function()` is the one general rule this reduces to: for
each node whose op_type isn't already importable, try extracting its
schema function (skip if the schema has neither kind, or if a
context-dependent one needs an input dtype this model doesn't statically
know), inline it, and **only commit the replacement if every resulting
op_type is itself importable** -- fails closed rather than trading one
unimportable op for a decomposition that still contains one (ONNX's own
function bodies bottom out in ops like `Constant`/`Shape`/`Slice`/
`ConstantOfShape`/`Concat`/`Flatten`/`Neg`/`Reciprocal`, all confirmed
present in `TVM_V08_ONNX_CONVERT_MAP_OPS` for `LayerNormalization`'s case,
but nothing guarantees that for an arbitrary future op or a future ONNX
version's rewrite of an existing one). This one rule already covers
`HardSwish`/`Mish`/`LayerNormalization` with no per-op code, and picks up
any other function-defined op (e.g. `GroupNormalization`, `Gelu`,
`MeanVarianceNormalization`) the same way, with no changes needed here.

`hardswish_to_primitives`/`mish_to_primitives`/`layer_normalization_to_
primitives` remain as thin, `op_types`-filtered wrappers around the one
general rule -- kept only so `RULES`/`--rules`/existing callers keep their
names; none of them contain op-specific logic anymore.

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

import onnx
import onnx.inliner
import onnx.shape_inference

# Same convention as scripts/axelera/voyager_simulator.py's `from
# voyager_ops import ...`: this only resolves when scripts/renesas is on
# sys.path (documented in this directory's README.md), which every caller
# in this codebase (tests/test_renesas_legalize.py, this file's own `python
# legalize.py` CLI usage from within the directory) already arranges.
from drp_ai_tvm_ops import DRP_AI_TVM_IMPORTABLE_OPS
from onnx import TypeProto, helper


def _unique_name(model, stem, reserved):
    """Like `scripts/axelera/legalize.py`'s `_unique_name()`, plus
    `reserved`: names already claimed *within the current `legalize()`
    call* but not yet written back into `model.graph.node` (multiple
    unimportable nodes get namespaced before any of their replacements are
    spliced in -- without this, two different nodes' decompositions could
    independently pick the same fresh name, each unaware of the other).
    """
    taken = (
        {i.name for i in model.graph.initializer}
        | {n.name for n in model.graph.node if n.name}
        | {o for n in model.graph.node for o in n.output}
        | reserved
    )
    name, k = stem, 0
    while name in taken:
        k += 1
        name = f"{stem}_{k}"
    reserved.add(name)
    return name


def _value_types(model):
    """{tensor name: onnx elem_type int}, for every graph input/output/
    value_info/initializer with a known element type. Runs shape inference
    itself (only the element type is used here, not shape -- the
    schema-derived functions this file extracts handle rank dynamically).
    """
    inferred = model
    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    except Exception:
        pass
    types = {}
    for value in (
        list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    ):
        if value.type.tensor_type.elem_type:
            types[value.name] = value.type.tensor_type.elem_type
    for init in model.graph.initializer:
        types[init.name] = init.data_type
    return types


def _default_domain_opset(model):
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx"):
            return opset.version
    return 1


def _pick_function_opset(available_versions, target_opset):
    """The schema function version to use for a model importing
    `target_opset`: the newest one at or below it, same resolution rule
    ONNX itself uses for an op's own version. `None` if the schema defines
    no function version this model's opset can use at all.
    """
    usable = [v for v in available_versions if v <= target_opset]
    return max(usable) if usable else None


def _extract_schema_function_nodes(node, target_opset, types):
    """The primitive-op decomposition ONNX's own schema defines for `node`,
    inlined via `onnx.inliner.inline_local_functions()` -- a list of
    `NodeProto`s whose boundary input/output names already match `node`'s
    real tensor names (the wrapper graph built around it uses them
    directly). Internal/intermediate names are namespaced by the caller,
    not here (see `_namespace_and_collect`) -- this function only extracts
    and inlines one node's call in isolation.

    Returns `None` -- meaning "nothing to extract, leave the node alone" --
    when the schema has neither `has_function` nor
    `has_context_dependent_function` at a version this model's opset can
    use, or (context-dependent case only) when an input's element type
    isn't statically known.
    """
    try:
        schema = onnx.defs.get_schema(node.op_type, target_opset, domain=node.domain)
    except Exception:
        return None

    if schema.has_function:
        fn_opset = _pick_function_opset(schema.function_opset_versions, target_opset)
        if fn_opset is None:
            return None
        function_bytes = schema.get_function_with_opset_version(fn_opset)
    elif schema.has_context_dependent_function:
        fn_opset = _pick_function_opset(
            schema.context_dependent_function_opset_versions, target_opset
        )
        if fn_opset is None:
            return None
        input_type_protos = []
        for inp in node.input:
            if not inp:
                continue
            elem_type = types.get(inp)
            if not elem_type:
                return None  # dtype not statically known -- fail closed
            tp = TypeProto()
            tp.tensor_type.elem_type = elem_type
            input_type_protos.append(tp)
        function_bytes = schema.get_context_dependent_function_with_opset_version(
            fn_opset,
            node.SerializeToString(),
            [t.SerializeToString() for t in input_type_protos],
        )
    else:
        return None

    function_proto = onnx.FunctionProto()
    function_proto.ParseFromString(function_bytes)

    def value_info(name):
        elem_type = types.get(name, onnx.TensorProto.FLOAT)
        return helper.make_tensor_value_info(name, elem_type, None)

    wrapper_graph = helper.make_graph(
        [node],
        "wrapper",
        [value_info(i) for i in node.input if i],
        [value_info(o) for o in node.output if o],
    )
    wrapper_model = helper.make_model(
        wrapper_graph, opset_imports=[helper.make_opsetid(node.domain, fn_opset)]
    )
    wrapper_model.functions.append(function_proto)
    try:
        inlined = onnx.inliner.inline_local_functions(wrapper_model)
    except Exception:
        return None
    return list(inlined.graph.node)


def _namespace_and_collect(model, nodes, original_node, reserved):
    """Renames every value the extracted `nodes` introduce beyond
    `original_node`'s own real input/output names to a fresh name unique
    across `model` *and* every other node processed in this same
    `legalize()` call (via `reserved` -- see `_unique_name()`), so
    multiple replacements can be spliced into one graph without colliding.
    """
    boundary = {n for n in list(original_node.input) + list(original_node.output) if n}
    rename = {}
    for node in nodes:
        for name in list(node.input) + list(node.output):
            if name and name not in boundary and name not in rename:
                rename[name] = _unique_name(model, name, reserved)

    renamed = []
    for node in nodes:
        copy = onnx.NodeProto()
        copy.CopyFrom(node)
        copy.input[:] = [rename.get(i, i) for i in copy.input]
        copy.output[:] = [rename.get(o, o) for o in copy.output]
        if copy.name:
            copy.name = _unique_name(model, copy.name, reserved)
        renamed.append(copy)
    return renamed


def legalize_via_onnx_function(model, op_types=None):
    """Replaces every node whose op_type is not in
    `DRP_AI_TVM_IMPORTABLE_OPS` with its ONNX schema's own function-body
    decomposition, when one is extractable and every op_type it bottoms
    out in *is* importable -- see this module's docstring. Restricted to
    `op_types` if given (a set/frozenset of op_type strings); default: any
    op_type with such a schema function.
    """
    target_opset = _default_domain_opset(model)
    types = _value_types(model)
    reserved = set()
    out, changed = [], 0
    for node in model.graph.node:
        if node.op_type in DRP_AI_TVM_IMPORTABLE_OPS:
            out.append(node)
            continue
        if op_types is not None and node.op_type not in op_types:
            out.append(node)
            continue
        replacement = _extract_schema_function_nodes(node, target_opset, types)
        if replacement is None or not all(
            n.op_type in DRP_AI_TVM_IMPORTABLE_OPS for n in replacement
        ):
            out.append(node)
            continue
        out.extend(_namespace_and_collect(model, replacement, node, reserved))
        changed += 1
    if changed:
        del model.graph.node[:]
        model.graph.node.extend(out)
    return changed


def hardswish_to_primitives(model):
    """`legalize_via_onnx_function()` restricted to `HardSwish` -- see this
    module's docstring."""
    return legalize_via_onnx_function(model, op_types={"HardSwish"})


def mish_to_primitives(model):
    """`legalize_via_onnx_function()` restricted to `Mish` -- see this
    module's docstring."""
    return legalize_via_onnx_function(model, op_types={"Mish"})


def layer_normalization_to_primitives(model):
    """`legalize_via_onnx_function()` restricted to `LayerNormalization` --
    see this module's docstring."""
    return legalize_via_onnx_function(model, op_types={"LayerNormalization"})


#: Order doesn't matter: `legalize_via_onnx_function()` only ever touches
#: an op_type still outside `DRP_AI_TVM_IMPORTABLE_OPS` at the time it
#: runs, and the other three are each a strict subset of it.
RULES = {
    "legalize_via_onnx_function": legalize_via_onnx_function,
    "hardswish_to_primitives": hardswish_to_primitives,
    "mish_to_primitives": mish_to_primitives,
    "layer_normalization_to_primitives": layer_normalization_to_primitives,
}


def legalize(model, rules=None):
    """Apply the named rules in order; returns `{rule: sites changed}`."""
    applied = collections.OrderedDict()
    for name in rules or ["legalize_via_onnx_function"]:
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
