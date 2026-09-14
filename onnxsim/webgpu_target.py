"""Flags ``com.microsoft::Attention`` nodes that crash onnxruntime-web's
WebGPU execution provider instead of running on the GPU.

onnxruntime-web's WebGPU backend lists ``Attention`` as a supported op, but
its own operator table (``js/web/docs/webgpu-operators.md``) annotates it
with "need implementing mask and past/present": the WebGPU kernel does not
yet handle the ``mask_index`` or ``past``/``present`` KV-cache inputs. A node
using either one is still a *valid* graph -- ``onnx.checker`` has nothing to
say about it.

**This is not a graceful fallback.** It was originally documented here as
one (matching how ORT's docs describe most operator-coverage gaps: the
unsupported node quietly falls back to another execution provider), but
running the actual check against onnxruntime-web 1.29 found otherwise: ONNX
Runtime's partitioner assigns a node to WebGPU by op type alone --
``GetCapability`` has no way to inspect *which optional inputs are wired
up* -- so a ``mask_index``-bearing ``Attention`` node is still committed to
WebGPU at partition time, and only fails once its kernel actually runs::

    Error: [WebGPU] Kernel "[Attention] " failed. Error: Mask not supported

This happens even with ``wasm`` listed as a fallback provider (verified in
``scripts/convertmodel/test/webgpu_attention_placement.test.mjs``, which
runs this exact scenario in a real browser): a fallback provider only
catches nodes ``GetCapability`` declined outright, not ones that were
accepted and then failed at ``Compute()``-time. So the real, measured
consequence of this gap is ``session.run()`` throwing and the **whole
session failing** -- not "just this node loses acceleration". This was only
confirmed for ``mask_index`` (building a ``past``/``present`` KV-cache tensor
for the same test needs more setup); ``past`` is grouped with the same "need
implementing" note upstream, but is not separately, empirically re-verified
here.

:func:`onnxsim.fuse_attention <onnxsim.onnx_simplifier>`'s own fusion
(``onnxsim/passes/fuse_attention.h``) never produces this shape itself -- it
only matches self-attention with no mask and no past/present to begin with --
so this only ever fires on an ``Attention`` node that was already in the
input model (e.g. exported by another tool) before reaching onnxsim, and
survives simplification unchanged. ``MultiHeadAttention`` is not covered:
onnxsim has no pass that produces or consumes that op, so there is nothing in
this codebase to inspect it for.

Meant to be called on the *output* of
``onnxsim.simplify(model, gemm_fusion_backend="webgpu")`` -- the point where
the model is about to be shipped to a WebGPU-targeting caller -- not on the
input model, since simplification could in principle still add or remove
such a node.

:func:`estimate_webgpu_islands` estimates how many separate contiguous
"WebGPU islands" a flagged node would split the rest of the graph into, and
how many device-copy boundaries would result, *if* the node gracefully fell
back the way ``onnxsim._ep_fragmentation`` assumes -- worth having for
whatever future gap actually does fall back gracefully, but given the
finding above, that model **understates** what actually happens for this
specific, currently-checked gap: a crash, not a fallback with copy overhead.

## Two more gaps, from onnxruntime-web's own operator table

``js/web/docs/webgpu-operators.md`` documents several more rows with a
correctness (not just performance) caveat. Two are checked here, each
statically decidable from the node alone (no shape inference needed) the
same way the Attention check is:

- **``Conv``/``ConvTranspose`` with spatial rank 3.** The table's note reads
  "conv3d is not supported" / "ConvTranspose3d is not supported". Spatial
  rank comes from the ``kernel_shape`` attribute when present, else the ``W``
  initializer's shape. *Unlike* the Attention gap, a 3-D kernel shape **is**
  visible to ``GetCapability`` from static graph info alone, so this one more
  plausibly falls back gracefully instead of crashing -- but that has not
  been independently re-verified in a real browser the way the Attention
  finding was, so :func:`check_webgpu_support` says so rather than asserting
  it.
- **``Resize`` with ``coordinate_transformation_mode="align_corners"`` and a
  constant ``scales`` input that downsamples.** The table's note reads
  "CoordinateTransformMode align_corners is not supported with
  downsampling" -- upsampling is implied to be fine, so this only fires when
  downsampling can actually be confirmed from a constant ``scales``
  initializer; a non-constant ``scales``, or a ``sizes``-based call (which
  would need static shape inference to compare against the input's shape),
  is left unflagged rather than guessed at. Like the Attention gap, the scale
  *values* are a runtime tensor input rather than static graph shape, so
  ``GetCapability`` plausibly can't see this either -- again, not
  independently re-verified, so worded as a plausible mechanism, not a
  confirmed one.

:func:`check_webgpu_support` runs all three checks (Attention, conv/deconv
3-D, Resize) and :func:`estimate_webgpu_islands` folds all three into one
fragmentation estimate -- still a lower bound, per
``onnxsim._ep_fragmentation``'s own caveat, since neither the perf-only rows
(``AveragePool``/``Conv``/``ConvTranspose``/``MaxPool``/``Transpose``'s "need
perf optimization" notes) nor the two blanket "no GPU kernel" rows
(``Reshape``/``Shape``, which would fire on nearly every node in nearly every
graph and add noise without a node-specific condition to check) are checked
here.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import onnx
import onnx.numpy_helper

from onnxsim._ep_fragmentation import IslandReport, estimate_fragmentation

# com.microsoft::Attention's positional input order (ContribOperators.md):
# 0 input, 1 weights, 2 bias, 3 mask_index, 4 past, 5 attention_bias,
# 6 past_sequence_length. Only mask_index/past are checked here: they are the
# two onnxruntime-web's WebGPU Attention kernel documents as unimplemented.
_MASK_INDEX_INPUT_POSITION = 3
_PAST_INPUT_POSITION = 4


def _flagged_attention_nodes(graph: onnx.GraphProto) -> List[Tuple[int, str]]:
    """Every ``com.microsoft::Attention`` node using ``mask_index``/``past``,
    as ``(node index in graph.node, message)`` pairs -- the single source of
    truth both :func:`check_webgpu_attention_support` and
    :func:`estimate_webgpu_islands` build on.
    """
    flagged = []
    for i, node in enumerate(graph.node):
        if node.domain != "com.microsoft" or node.op_type != "Attention":
            continue
        unsupported = []
        if (
            len(node.input) > _MASK_INDEX_INPUT_POSITION
            and node.input[_MASK_INDEX_INPUT_POSITION]
        ):
            unsupported.append(f"mask_index={node.input[_MASK_INDEX_INPUT_POSITION]!r}")
        if len(node.input) > _PAST_INPUT_POSITION and node.input[_PAST_INPUT_POSITION]:
            unsupported.append(f"past={node.input[_PAST_INPUT_POSITION]!r}")
        if not unsupported:
            continue
        node_label = node.name or (node.output[0] if node.output else "<unnamed>")
        flagged.append(
            (
                i,
                f"Attention node {node_label!r} has {' and '.join(unsupported)} "
                "wired up; onnxruntime-web's WebGPU execution provider does not "
                "yet implement mask/past-present support for Attention, and "
                "(verified for mask_index, see this module's docstring) this "
                "is not a graceful fallback to another execution provider -- "
                "the WebGPU kernel throws at runtime and the whole "
                "session.run() call fails, even with wasm listed as a "
                "fallback provider.",
            )
        )
    return flagged


_CONV_LIKE_OPS = ("Conv", "ConvTranspose")
_DEFAULT_DOMAINS = ("", "ai.onnx")  # onnx.parser/onnx.helper both emit "".


def _conv_spatial_rank(
    node: onnx.NodeProto, initializer_map: dict
) -> Optional[int]:
    """The convolution's spatial rank (2 for a "normal" Conv, 3 for conv3d),
    or ``None`` if it can't be determined from the node alone: prefers the
    ``kernel_shape`` attribute (most exporters set it), else falls back to
    the ``W`` initializer's shape (rank - 2 spatial dims), else gives up
    rather than guessing -- this module never runs shape inference.
    """
    for attr in node.attribute:
        if attr.name == "kernel_shape":
            return len(attr.ints)
    if len(node.input) > 1:
        w_init = initializer_map.get(node.input[1])
        if w_init is not None:
            return max(len(w_init.dims) - 2, 0)
    return None


def _flagged_conv3d_nodes(graph: onnx.GraphProto) -> List[Tuple[int, str]]:
    """Every ``Conv``/``ConvTranspose`` node with spatial rank 3, as
    ``(node index in graph.node, message)`` pairs -- see this module's
    docstring for why conv3d/ConvTranspose3d specifically, and why this is
    less certain to crash than the Attention gap.
    """
    initializer_map = {init.name: init for init in graph.initializer}
    flagged = []
    for i, node in enumerate(graph.node):
        if node.domain not in _DEFAULT_DOMAINS or node.op_type not in _CONV_LIKE_OPS:
            continue
        rank = _conv_spatial_rank(node, initializer_map)
        if rank != 3:
            continue
        node_label = node.name or (node.output[0] if node.output else "<unnamed>")
        flagged.append(
            (
                i,
                f"{node.op_type} node {node_label!r} has spatial rank 3 (a "
                f"3-D convolution); onnxruntime-web's WebGPU operator table "
                f"documents {node.op_type}3d as \"not supported\". This "
                "rank is visible to ONNX Runtime's GetCapability from "
                "static graph shape alone (unlike the Attention gap above), "
                "so this more plausibly falls back gracefully to another "
                "execution provider instead of crashing -- but that has not "
                "been independently re-verified against a real "
                "onnxruntime-web build the way the Attention finding was.",
            )
        )
    return flagged


def _resize_coordinate_transformation_mode(node: onnx.NodeProto) -> str:
    for attr in node.attribute:
        if attr.name == "coordinate_transformation_mode":
            value = onnx.helper.get_attribute_value(attr)
            return value.decode() if isinstance(value, bytes) else value
    return "half_pixel"  # ONNX Resize's documented default.


def _flagged_resize_align_corners_downsample_nodes(
    graph: onnx.GraphProto,
) -> List[Tuple[int, str]]:
    """Every ``Resize`` node using ``coordinate_transformation_mode=
    "align_corners"`` with a constant ``scales`` input that downsamples at
    least one axis, as ``(node index in graph.node, message)`` pairs -- see
    this module's docstring for why only the constant-``scales`` case is
    checked.
    """
    initializer_map = {init.name: init for init in graph.initializer}
    flagged = []
    for i, node in enumerate(graph.node):
        if node.domain not in _DEFAULT_DOMAINS or node.op_type != "Resize":
            continue
        if _resize_coordinate_transformation_mode(node) != "align_corners":
            continue
        scales_name = node.input[2] if len(node.input) > 2 else ""
        if not scales_name or scales_name not in initializer_map:
            continue
        scales = onnx.numpy_helper.to_array(initializer_map[scales_name])
        if not (scales < 1.0).any():
            continue
        node_label = node.name or (node.output[0] if node.output else "<unnamed>")
        flagged.append(
            (
                i,
                f"Resize node {node_label!r} uses "
                'coordinate_transformation_mode="align_corners" with a '
                f"constant scales input that downsamples at least one axis "
                f"({scales.tolist()!r}); onnxruntime-web's WebGPU operator "
                "table documents align_corners as \"not supported with "
                "downsampling\". Like the Attention gap above, the scale "
                "values are a runtime tensor input rather than static graph "
                "shape, so GetCapability plausibly can't see this either -- "
                "but that mechanism has not been independently re-verified "
                "for this specific gap the way it was for Attention.",
            )
        )
    return flagged


def check_webgpu_attention_support(
    model: Union[str, onnx.ModelProto],
) -> List[str]:
    """Scans for ``com.microsoft::Attention`` nodes wired up with a
    ``mask_index`` and/or ``past`` input -- the configuration that crashes
    onnxruntime-web's WebGPU execution provider at runtime rather than
    running on the GPU (see this module's docstring for the mechanism, and
    why "does not accelerate" understates it).

    :param model: the onnx ModelProto to inspect, or a file path
    :returns: one human-readable message per offending node (empty if none);
            each message names the node and which unsupported input(s) it
            uses. This is advisory only -- it does not modify ``model`` and it
            does not itself raise (the graph is still perfectly valid ONNX),
            but running the flagged node's model on WebGPU will.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return [msg for _, msg in _flagged_attention_nodes(model.graph)]


def check_webgpu_conv3d_support(model: Union[str, onnx.ModelProto]) -> List[str]:
    """Scans for ``Conv``/``ConvTranspose`` nodes with spatial rank 3 (see
    this module's docstring for the "conv3d is not supported" upstream note,
    and why this is less certain to crash than the Attention gap).

    :param model: the onnx ModelProto to inspect, or a file path
    :returns: one human-readable message per offending node (empty if none).
            Advisory only, same caveats as :func:`check_webgpu_attention_support`.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return [msg for _, msg in _flagged_conv3d_nodes(model.graph)]


def check_webgpu_resize_support(model: Union[str, onnx.ModelProto]) -> List[str]:
    """Scans for ``Resize`` nodes using ``align_corners`` with a constant,
    downsampling ``scales`` input (see this module's docstring for the
    upstream note and why only the constant-``scales`` case is checked).

    :param model: the onnx ModelProto to inspect, or a file path
    :returns: one human-readable message per offending node (empty if none).
            Advisory only, same caveats as :func:`check_webgpu_attention_support`.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return [msg for _, msg in _flagged_resize_align_corners_downsample_nodes(model.graph)]


def check_webgpu_support(model: Union[str, onnx.ModelProto]) -> List[str]:
    """Runs every check this module has -- :func:`check_webgpu_attention_support`,
    :func:`check_webgpu_conv3d_support`, and :func:`check_webgpu_resize_support`
    -- and concatenates their messages (attention first, then conv/deconv
    3-D, then Resize). See this module's docstring for what each one covers
    and, just as importantly, what it doesn't.

    :param model: the onnx ModelProto to inspect, or a file path
    :returns: one human-readable message per offending node (empty if none).
            Advisory only, same caveats as :func:`check_webgpu_attention_support`.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    graph = model.graph
    return (
        [msg for _, msg in _flagged_attention_nodes(graph)]
        + [msg for _, msg in _flagged_conv3d_nodes(graph)]
        + [msg for _, msg in _flagged_resize_align_corners_downsample_nodes(graph)]
    )


def estimate_webgpu_islands(model: Union[str, onnx.ModelProto]) -> IslandReport:
    """Estimates how much the nodes :func:`check_webgpu_support` flags
    (Attention ``mask_index``/``past``, conv/deconv 3-D, ``Resize``
    ``align_corners`` downsampling) fragment the rest of the graph into
    separate WebGPU-accelerated islands (see this module's docstring and
    ``onnxsim._ep_fragmentation`` for the method and its limits: this only
    accounts for the specific gaps this module checks for, so it is a lower
    bound on real fragmentation, not a full simulation of ONNX Runtime's
    partitioner).

    :param model: the onnx ModelProto to inspect, or a file path
    :returns: an :class:`onnxsim._ep_fragmentation.IslandReport`
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    graph = model.graph
    flagged = (
        _flagged_attention_nodes(graph)
        + _flagged_conv3d_nodes(graph)
        + _flagged_resize_align_corners_downsample_nodes(graph)
    )
    return estimate_fragmentation(graph, {i for i, _ in flagged})
