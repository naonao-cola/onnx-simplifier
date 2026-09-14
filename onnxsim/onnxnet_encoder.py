"""Encodes a simplified ONNX graph as ONNX-Net's ``chain_slim`` text format.

ONNX-Net (Qin et al., arXiv:2510.04938, https://github.com/shiwenqin/ONNX-Net)
turns an ONNX graph into a compact text encoding fed to an LLM-based
architecture-performance predictor. Its own preprocessing runs a model
through onnxsim first (see ``tests/test_onnxnet_integration.py``, which
pins the two structural properties -- no dead scaffolding nodes, and
MatMul+bias-Add collapsed into Gemm -- that make the *simplified* graph a
plain, unbranching chain) before their own encoder ever sees it.

This module is a faithful, line-for-line port of that encoder's two
simplest modes -- ``chain_slim`` (the mode used throughout the paper) and
``chain_slim_base`` (their own ablation baseline) -- from
``src/onnxnet/process/utils.py``'s ``ONNXConverter.get_onnx_infos_chain_slim``
/ ``get_onnx_infos_chain_slim_base`` (fetched from the ``main`` branch,
2026-09-14; there is no tagged release to pin to). The five other modes
that file defines (``chain_slim_param``/``_outshape``/``_input``, the
non-chain ``full``/``slim``/``oponly`` family, and the "template" op-class
abstraction) are not ported -- they follow the same walk with different
fields included, and can be added the same way if a caller needs one.
**Not ported at all**: their ``get_onnx_str`` wrapper's token-counting
(needs a HuggingFace tokenizer) and accuracy lookup (needs an
ONNX-Bench-style accuracy value baked into ``model.metadata_props``) --
both are ONNX-Net's own benchmarking concerns, not part of the text
encoding itself, and pulling in a tokenizer dependency for them would be
out of place here (see the repo's own stance on optional heavy
dependencies).

**Two upstream behaviors preserved deliberately, not "fixed"**, because
the entire point of a port is producing byte-identical output to the
original for whatever downstream (their own predictor, or a compatible
one) expects it:

- Upstream's attribute-to-string helper reads a scalar attribute's value
  off the wrong protobuf field (``AttributeProto.floats``/``.ints``, the
  *list* fields, instead of ``.f``/``.i``, the scalar ones). A scalar
  ``FLOAT``/``INT`` attribute's list field is always empty, so its string
  form is always ``"[]"`` -- which every mode then filters out as "no
  attribute to show". Net effect: a node's *scalar* attributes (Gemm's
  ``alpha``/``beta``/``transA``/``transB``, for instance) never appear in
  the text, no matter their value; only *list*-valued attributes (Conv's
  ``kernel_shape``, say) do. :func:`_attribute_to_str` reproduces this
  exactly, via the same field access.
- ``chain_slim_base`` names every chain-breaking intermediate value the
  literal string ``"Value"`` (no index), while ``chain_slim`` numbers them
  ``Value1``, ``Value2``, ... -- not an oversight to reconcile between the
  two modes, just how the original distinguishes its ablation baseline
  from its main encoding.

See ``tests/test_onnxnet_encoder.py`` for golden-string tests against the
same simplified MLP fixture ``test_onnxnet_integration.py`` builds.
"""

from __future__ import annotations

from typing import Dict, List

import onnx
from onnx import shape_inference


def _attribute_to_str(attr: onnx.AttributeProto) -> str:
    """Upstream's ``attrtype_to_str`` closure, verbatim -- see this module's
    own docstring for why a scalar ``FLOAT``/``INT`` attribute always comes
    out as ``"[]"`` here (reading ``.floats``/``.ints`` instead of
    ``.f``/``.i`` is upstream's own choice, reproduced rather than corrected).
    """
    t = onnx.AttributeProto
    if attr.type == t.FLOAT:
        return str(attr.floats)
    if attr.type == t.FLOATS:
        return str(list(attr.floats)).replace(" ", "")
    if attr.type == t.INT:
        return str(attr.ints)
    if attr.type == t.INTS:
        ints = list(attr.ints)
        # "all entries equal" collapses to that one value, e.g. a symmetric
        # kernel_shape/strides/pads reads as a single int rather than a list.
        if len(set(ints)) == 1:
            return str(ints[0])
        return str(ints).replace(" ", "")
    if attr.type == t.STRING:
        return str(attr.strings)
    if attr.type == t.STRINGS:
        return str(list(attr.strings)).replace(" ", "")
    if attr.type in (t.TENSOR, t.GRAPH, t.SPARSE_TENSOR):
        return "<BLOB>"
    if attr.type in (t.TENSORS, t.GRAPHS, t.SPARSE_TENSORS):
        return "<BLOB,...>"
    if attr.type == t.TYPE_PROTO:
        return "<TYPE>"
    if attr.type == t.TYPE_PROTOS:
        return "<TYPE,...>"
    if attr.type == t.UNDEFINED:
        return "undefined"
    raise ValueError(f"Unknown attribute type: {attr.type}")


def _format_attrs(node: onnx.NodeProto) -> str:
    """``",".join(name=value)`` over every attribute whose string form isn't
    the empty-list sentinel ``"[]"`` -- see this module's own docstring for
    which attributes that filters out.
    """
    return ",".join(
        f"{a.name}={_attribute_to_str(a)}"
        for a in node.attribute
        if _attribute_to_str(a) != "[]"
    )


def _output_shapes(model: onnx.ModelProto) -> Dict[str, str]:
    """``{tensor name: "x"-joined shape}`` for every intermediate value and
    graph output ``model`` has static shape info for (needs
    ``shape_inference.infer_shapes`` to have populated ``graph.value_info``
    first -- see :func:`chain_slim`/:func:`chain_slim_base`, which both call
    it before this). A symbolic (``dim_param``) dimension reads as ``"0"``
    here, same as upstream: neither this port nor the original special-cases
    a dynamic shape in this particular string.
    """
    shapes: Dict[str, str] = {}
    for value_info in model.graph.value_info:
        dims = [d.dim_value for d in value_info.type.tensor_type.shape.dim]
        shapes[value_info.name] = "x".join(str(d) for d in dims)
    for output in model.graph.output:
        dims = [d.dim_value for d in output.type.tensor_type.shape.dim]
        shapes[output.name] = "x".join(str(d) for d in dims)
    return shapes


def chain_slim(model: onnx.ModelProto) -> str:
    """ONNX-Net's primary text encoding -- the one the paper's own predictor
    is trained on. A faithful port of
    ``ONNXConverter.get_onnx_infos_chain_slim``; see this module's own
    docstring for exactly what that means and doesn't mean.

    Best fed a model that has already been through :func:`onnxsim.simplify`:
    a run of dead scaffolding (``Identity``) and un-fused patterns
    (``MatMul``+bias ``Add``) reads as clutter here just as it would to
    ONNX-Net's own preprocessing, which expects the same (see
    ``tests/test_onnxnet_integration.py``). Every node with exactly one
    node-produced input and one output chains onto the previous one as
    ``Op1(...) --> Op2(...) --> ...``; a node that branches (more than one
    node-produced input, e.g. a residual ``Add``, or more than one output)
    ends the current chain and is printed with its own inputs named
    explicitly instead.
    """
    model = shape_inference.infer_shapes(model)
    shapes = _output_shapes(model)
    graph_inputs = {i.name for i in model.graph.input}
    params = {i.name for i in model.graph.initializer}

    name_map: Dict[str, str] = {}
    for input_ in model.graph.input:
        dims = [d.dim_value for d in input_.type.tensor_type.shape.dim]
        name_map[input_.name] = "x".join(str(d) for d in dims)
    for init in model.graph.initializer:
        name_map[init.name] = f"Param{list(init.dims)!s}".replace(" ", "")
    for out in model.graph.output:
        name_map[out.name] = "Out"

    value_index = 1
    text = ""
    open_output: str | None = None
    for node in model.graph.node:
        node_inputs: List[str] = [
            i for i in node.input if i not in graph_inputs and i not in params
        ]
        other_inputs = [
            name_map[i] for i in node.input if i in graph_inputs or i in params
        ]
        attrs = _format_attrs(node)
        op_type = node.op_type

        if len(node_inputs) == 1 and len(node.output) == 1:
            if open_output is None:
                # Starts a fresh chain from nothing open yet: name the one
                # real input if we have a name for it.
                text += f"{op_type}("
                if other_inputs:
                    head = name_map.get(node_inputs[0], "prev")
                    text += f"{head}, {', '.join(other_inputs)})"
                else:
                    text += (
                        f"{name_map[node_inputs[0]]})"
                        if node_inputs[0] in name_map
                        else "prev)"
                    )
            elif open_output in node_inputs:
                # Continues the open chain.
                text += f"{op_type}("
                text += f"prev, {', '.join(other_inputs)})" if other_inputs else "prev)"
            else:
                # A single-input/output node, but not a continuation of the
                # currently open chain (its one input comes from somewhere
                # else entirely) -- close the open chain first, then start a
                # new one. Deliberately asymmetric with the "nothing open
                # yet" case above when there are no other_inputs: upstream's
                # own fresh-chain-after-a-close case omits the input name
                # entirely rather than falling back to "prev" the way the
                # very first chain of the graph does -- reproduced here
                # rather than smoothed over, since the two are byte-different
                # in the original.
                name_map[open_output] = f"Value{value_index}"
                value_index += 1
                text += (
                    f"{name_map[open_output]}:{shapes[open_output]}\n"
                    if open_output in shapes
                    else f"{name_map[open_output]}\n"
                )
                text += f"{op_type}("
                if other_inputs:
                    head = name_map.get(node_inputs[0], "prev")
                    text += f"{head}, {', '.join(other_inputs)})"
                else:
                    text += ")"
            text += f"({attrs}) --> " if attrs else " --> "
            open_output = node.output[0]
        else:
            # A branch point: more than one node-produced input (e.g. a
            # residual Add) or more than one output. Close whatever chain
            # was open, then print this node with every input named
            # explicitly rather than folding it into a chain. Upstream
            # indexes `shapes[...]` unguarded at both spots below, assuming
            # shape inference always reached these tensors; this port uses
            # `.get(..., "")` instead so a tensor shape inference could not
            # resolve renders as an empty shape rather than raising.
            if open_output is not None:
                name_map[open_output] = f"Value{value_index}"
                value_index += 1
                text += f"{name_map[open_output]}:{shapes.get(open_output, '')}\n"

            text += f"{op_type}("
            text += ", ".join(name_map.get(i, "prev") for i in node.input)
            text += f")({attrs}) --> " if attrs else ") --> "

            for out in node.output:
                if out not in name_map:
                    name_map[out] = f"Value{value_index}"
                    value_index += 1
            text += ", ".join(name_map[o] for o in node.output)
            text += f":{shapes.get(node.output[0], '')}\n"
            open_output = None

    if open_output is not None:
        text += name_map[open_output]
    return text


def chain_slim_base(model: onnx.ModelProto) -> str:
    """ONNX-Net's bare-bones ablation baseline: the same chain walk as
    :func:`chain_slim`, but with every attribute, shape and parameter
    reference dropped -- just the sequence of op-type names. A faithful
    port of ``ONNXConverter.get_onnx_infos_chain_slim_base``; see this
    module's own docstring for why every chain-breaking intermediate value
    is named the literal ``"Value"`` here (no index), unlike
    :func:`chain_slim`'s numbered ``Value1``, ``Value2``, ....
    """
    model = shape_inference.infer_shapes(model)
    graph_inputs = {i.name for i in model.graph.input}
    params = {i.name for i in model.graph.initializer}

    # Upstream also seeds name_map with an input-shape string per graph
    # input and a "Param[...]" string per initializer here, same as
    # chain_slim -- but this mode's bare "op_type --> op_type" text never
    # looks a node's *input* up in name_map (only a branch point's own
    # *outputs*, and the closing tensor at the very end), so those entries
    # are dead weight in the original and are not reproduced here.
    name_map: Dict[str, str] = {}
    for out in model.graph.output:
        name_map[out.name] = "Out"

    text = ""
    open_output: str | None = None
    for node in model.graph.node:
        node_inputs = [
            i for i in node.input if i not in graph_inputs and i not in params
        ]
        op_type = node.op_type

        if len(node_inputs) == 1 and len(node.output) == 1:
            if open_output is not None and open_output not in node_inputs:
                name_map[open_output] = "Value"
                text += f"{name_map[open_output]}\n"
            text += f"{op_type} --> "
            open_output = node.output[0]
        else:
            if open_output is not None:
                name_map[open_output] = "Value"
                text += f"{name_map[open_output]}\n"

            text += f"{op_type} --> "
            for out in node.output:
                if out not in name_map:
                    name_map[out] = "Value"
            text += ", ".join(name_map[o] for o in node.output) + "\n"
            open_output = None

    if open_output is not None:
        text += name_map[open_output]
    return text
