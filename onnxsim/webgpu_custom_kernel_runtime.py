"""Splits an ONNX model around one node carrying a custom WebGPU program
(:mod:`onnxsim.webgpu_kernel_metadata`) into a ``(pre, post)`` model pair, so
a browser runtime can run ``pre`` and ``post`` as ordinary
``onnxruntime-web`` WebGPU-EP sessions and splice the custom program in
between them for the one node ORT-web itself can't run -- the "runtime on
top of ort-web" piece :mod:`onnxsim.webgpu_kernel_metadata`'s own docstring
describes as not built yet. See
``scripts/convertmodel/webgpu_custom_kernel_runtime.mjs`` for the JS half
that actually runs the two sessions and dispatches the program between them
via onnxruntime-web's GPU-buffer IO binding (``Tensor.fromGpuBuffer`` /
``preferredOutputLocation: 'gpu-buffer'``), so no tensor ever round-trips
through the CPU at the splice point.

**Why physically remove the node, not just let ORT try and fail:**
``scripts/convertmodel/test/webgpu_attention_placement.test.mjs`` found
(empirically, against a real onnxruntime-web build) that ORT-web's WebGPU EP
partitioner assigns a node to WebGPU by op type alone -- ``GetCapability``
can't inspect which inputs are wired up -- so an unsupported *variant* of a
supported op (Attention with ``mask_index``, say) is committed to WebGPU at
partition time and only fails once its kernel actually runs, throwing and
failing the *whole* session, not just that node. A CPU/wasm fallback
provider does not help: it only catches nodes ``GetCapability`` declined
outright, never ones accepted and failed at ``Compute()``-time. Physically
excising the node from both ``pre`` and ``post``'s own ``GraphProto`` (as
this module does) sidesteps that failure mode entirely -- the flagged node
is never in a graph ORT-web's partitioner ever sees.

## What this covers

Splitting around exactly **one** node with a fully static shape, matching
the same scope :mod:`onnxsim.webgpu_tinygrad_codegen` generates kernels
for. Not covered (both explicitly, not silently):

- More than one flagged node in the same model -- call
  :func:`split_around_node` once per node from the *innermost* outward
  yourself if you need that; there is no multi-node orchestration here.
- A dynamic dispatch size -- :class:`~onnxsim.webgpu_kernel_metadata.WebgpuKernelStep`'s
  ``dispatch`` is a fixed ``[x, y, z]`` triple (see that module's own
  docstring), so this only ever targets a node whose shapes (and therefore
  whose generated dispatch size) are fixed at kernel-generation time.

## What building this needed from ``onnxsim`` that already existed

The actual graph surgery is :func:`onnxsim.vitisai_target.split_model` --
built for a different EP-placement problem (Vitis AI's NPU can't compile an
``If``, so split around it and run the two halves on different EPs) but
exactly the right primitive here too: "cut a graph in two at a named
tensor boundary, reconnecting side inputs automatically." This module calls
it *twice* (once at the node's own inputs, once at its outputs) to carve the
one node out from the middle rather than cutting the graph in half, and
adds the ONE-node sanity check and boundary-collection this specific use
needs.
"""

from __future__ import annotations

from typing import List, NamedTuple, Optional, Union

import onnx

from onnxsim.vitisai_target import split_model


class SplitAroundNode(NamedTuple):
    """The result of :func:`split_around_node`.

    :param pre: the sub-model producing every input ``node`` needs that some
            other node in the original graph produced -- ``None`` if
            ``node`` consumes only the original graph's own inputs/
            initializers directly (nothing to run before it).
    :param post: the sub-model consuming ``node``'s outputs and producing
            the original graph's own outputs. Always present, even if
            ``node`` were the very last computation (then ``post`` is just
            an identity pass-through of its outputs).
    :param node: the excised node itself (a copy, detached from either
            sub-model) -- read its own :func:`onnxsim.webgpu_kernel_metadata.read_webgpu_kernel`
            attachment (on the *original* model, by ``node.name`` -- this
            copy no longer lives in a model :func:`read_webgpu_kernel` can
            search) to get the program a runtime should dispatch in its
            place.
    """

    pre: Optional[onnx.ModelProto]
    post: onnx.ModelProto
    node: onnx.NodeProto


def _find_node(graph: onnx.GraphProto, node_name: str) -> onnx.NodeProto:
    for node in graph.node:
        if node.name == node_name:
            return node
    raise ValueError(f"no node named {node_name!r} in the graph")


def split_around_node(
    model: Union[str, onnx.ModelProto], node_name: str
) -> SplitAroundNode:
    """Splits ``model`` into a ``(pre, post)`` pair with the node named
    ``node_name`` excised from both, so a caller can run ``pre`` and
    ``post`` as ordinary ONNX Runtime sessions and substitute its own
    computation (e.g. a custom WebGPU program) for the removed node.

    :param model: the onnx ModelProto to split, or a file path.
    :param node_name: the ``NodeProto.name`` to excise -- not an output
            name or op type.
    :raises ValueError: no node named ``node_name`` exists, or (should not
            happen for a well-formed graph with a unique ``node_name``) the
            excised middle segment ended up with other than exactly that
            one node.
    :returns: a :class:`SplitAroundNode`. Neither the input model nor its
            names are mutated; every returned model is a fresh object
            sharing no state with it or each other.
    """
    source = model if isinstance(model, onnx.ModelProto) else onnx.load(model)
    node = _find_node(source.graph, node_name)

    produced = {o for n in source.graph.node for o in n.output if o}
    produced_inputs = [i for i in node.input if i and i in produced]

    if produced_inputs:
        pre, rest = split_model(source, produced_inputs)
    else:
        # Every input node consumes is already a top-level graph input or
        # an initializer -- nothing upstream to run first.
        pre = None
        rest = onnx.ModelProto()
        rest.CopyFrom(source)

    node_outputs: List[str] = [o for o in node.output if o]
    mid, post = split_model(rest, node_outputs)

    if len(mid.graph.node) != 1 or mid.graph.node[0].name != node_name:
        found = [n.name for n in mid.graph.node]
        raise ValueError(
            f"expected exactly one node named {node_name!r} between the two cuts, "
            f"got {found!r} -- the node's inputs/outputs may be shared with another "
            "node in a way split_around_node does not expect"
        )

    return SplitAroundNode(pre=pre, post=post, node=mid.graph.node[0])
