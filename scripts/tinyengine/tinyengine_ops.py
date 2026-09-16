#!/usr/bin/env python3
"""Static op-support heuristic for MIT HAN Lab's TinyEngine code generator.

`mit-han-lab/tinyengine <https://github.com/mit-han-lab/tinyengine>`_ is a
source-code generator that compiles a quantized model straight to C for
ARMv7E-M (Cortex-M) microcontrollers. **It has no NPU and nothing to do with
TI's edgeai/TIDL** -- an earlier request in this project's history conflated
a "~2.5 GOPS NPU" figure with TI's accelerator when it actually referred to
this project instead, and TinyEngine has no accelerator of any kind; see
``scripts/tinyengine/README.md`` for the correction. It is a pure-CPU
inference *compiler*: the C it emits for each op *is* the runtime
implementation, generated ahead of time, not dispatched to a library at
run time.

Two structural differences from every other ``scripts/<vendor>/*_ops.py`` in
this repo matter enough to shape this whole module:

1. **It ingests TFLite, not ONNX.** ``code_generator/TfliteConvertor.py``
   reads a ``.tflite`` flatbuffer and walks its op list; there is no ONNX
   importer at all. This repository has no ONNX->TFLite conversion step, so
   this heuristic works the way ``scripts/edgeai/tidl_ops.py``'s QOperator
   check does for a *different* format mismatch: it maps each ONNX op type
   to the TFLite builtin op(s) it would plausibly become after conversion,
   and checks *that* against TinyEngine's real dispatch table. This is an
   approximation of a conversion step that never actually runs here -- a
   "would this survive conversion and match the whitelist", not a
   guarantee, exactly the caveat ``tidl_ops.py`` gives for its own
   heuristic.
2. **There is no fallback path.** TIDL, QNN, OpenVINO and friends all fall
   an unsupported op back to a host CPU implementation and keep going.
   TinyEngine cannot: it has no separate "for anything else" runtime to
   fall back to, because the generated C is the only implementation that
   will ever exist for that graph. ``TfliteConvertor.py``'s
   ``_handleOperator`` method (the dispatch every op in the graph passes
   through) ends its if/elif chain with a plain
   ``else: raise NotImplementedError(f"Unsupported {op_code_str}")`` --
   confirmed by reading that method directly, not inferred. One
   unrecognized op anywhere in the graph fails the whole compile. There is
   no such thing as "partial coverage, rest on CPU fallback" here the way
   ``tidl_backend.coverage()`` means it -- "partial" for TinyEngine means
   "will not compile at all".

What ``_handleOperator`` actually dispatches on (read directly from
``code_generator/TfliteConvertor.py`` and ``code_generator/constant.py`` in
the real repo, not reconstructed from the TFLite schema's op enum -- see
point 3 below for why that distinction matters):

- ``CONV_2D`` / ``DEPTHWISE_CONV_2D`` -- both routed through the same
  ``TF_Parser.parse_conv2d``, keyed only on which BuiltinOperator each node
  actually carries; not distinguished by depthwise-ness at this dispatch
  level.
- ``ADD``, ``AVERAGE_POOL_2D``, ``MAX_POOL_2D``, ``FULLY_CONNECTED``,
  ``TRANSPOSE`` (via ``_convert_TRANSPOSE``), ``PAD`` (via ``_convert_PAD``),
  ``RESIZE_NEAREST_NEIGHBOR``, and ``MEAN`` (via ``parse_mead1dto2d`` --
  yes, that is the real function's name in the source, a 1D-to-2D mean
  reshape, not a typo introduced here).
- ``op_code_str in SKIP_OPs: pass`` -- ``constant.py`` defines
  ``SKIP_OPs = {"QUANTIZE", "DEQUANTIZE", "RESHAPE"}``. These ops are not
  rejected, but nothing is generated for them either: the node is silently
  dropped from the emitted C. That is exactly right for `QUANTIZE`/
  `DEQUANTIZE` (TinyEngine's whole point is to run natively quantized, so
  the fixed-point representation these ops describe is what every other
  op already assumes) but ``constant.py`` itself carries a
  ``# TODO: Handle RESHAPE during codegen`` comment right above the
  ``RESHAPE`` entry -- i.e. TinyEngine's own authors flag that skipping a
  reshape's actual data movement is not always safe, only convenient. This
  module treats ``Reshape`` as supported-but-flagged, not silently clean --
  see ``RESHAPE_SKIP_CAVEAT`` below.
- **``MUL`` has no case of its own.** It only appears indirectly, in a
  *lookahead* the convertor runs when it sees a ``DEPTHWISE_CONV_2D``:
  ``checkIfRequireSEelementmult`` scans the next few ops for the exact
  three-op sequence ``ADD -> MUL -> MUL`` and, if found, fuses that triple
  via ``TF_Parser.parse_SEelement`` as a squeeze-and-excite gate (the
  ``Add``/``Clip``-as-hard-sigmoid-numerator, ``Mul``-by-1/6, ``Mul``-onto-
  the-main-branch pattern real SE blocks use). A ``Mul`` anywhere else --
  including the extremely common Sigmoid/HardSigmoid-gated Swish/SiLU
  activation (``scripts/common/synthetic_models.sigmoid_mul_swish``) --
  has no dispatch case and falls into the same hard
  ``NotImplementedError`` as any other unrecognized op.
- Every other builtin op code, and anything ``_handleOperator`` cannot even
  attempt to look up, hits that terminal ``else`` and fails the whole
  conversion.

3. **A schema entry is not support.** ``code_generator/tflite/
   BuiltinOperator.py`` (TinyEngine's own vendored, schema-generated
   bindings) defines an enum value for ``PRELU``, ``LEAKY_RELU``, ``MUL``,
   ``RELU``, and dozens of other ops that appear nowhere in
   ``_handleOperator``'s dispatch above. The generated bindings describe
   what the *TFLite format* can represent, not what *this specific code
   generator* implements -- having a real, valid opcode is necessary but
   nowhere near sufficient. ``UNSUPPORTED_WITH_REAL_TFLITE_OPCODE`` below
   names a few of these deliberately, specifically because a schema lookup
   alone would wrongly suggest they're fine.

Fused activations are the other real subtlety. TFLite's own ``Conv2D``/
``DepthwiseConv2D`` op carries its activation as a
``FusedActivationFunction`` field on the *same* op (``Conv2DOptions.py``'s
``Conv2DOptionsAddFusedActivationFunction``) -- a `Relu`/`Relu6` (`Clip(0,
6)` in ONNX terms) immediately following a conv is not a separate TFLite
node at all once real conversion tooling fuses it in, so
``_handleOperator`` never sees it as its own op. A `Relu`/`Clip` that is
*not* the sole, immediate consumer of a `Conv` output is not eligible for
that fusion and, like `Mul`, has no standalone dispatch case -- so it would
surface as a real op in the `.tflite` graph and hit the same
`NotImplementedError`. ``is_fusable_activation`` below approximates that
one condition (immediate, sole consumer of a `Conv`) since this repository
never runs a real converter to confirm the fusion actually happens.

See ``scripts/tinyengine/README.md`` for the full survey (real published
latency numbers included) and exactly which of this module's checks are
exact ports of something read in the source versus a necessary
approximation because no ONNX->TFLite conversion runs here.
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Set

import onnx

# ONNX op types that map directly onto a TFLite builtin op
# `_handleOperator` actually dispatches on, regardless of graph position --
# no fusion window, no lookahead. `Conv` covers both `CONV_2D` and
# `DEPTHWISE_CONV_2D`: the real dispatch routes both through the same
# `parse_conv2d` call, keyed off the node's own opcode rather than a
# `group` attribute, so this module does not need to (and cannot, without
# shape inference) tell them apart either.
DIRECTLY_SUPPORTED_OPS: frozenset = frozenset(
    {
        "Conv",
        "Add",
        "AveragePool",
        "GlobalAveragePool",
        "MaxPool",
        "GlobalMaxPool",
        "Pad",
        "ReduceMean",
        "Transpose",
        "Gemm",
        "MatMul",
    }
)

# `constant.py`'s `SKIP_OPs`. `QuantizeLinear`/`DequantizeLinear` are the
# ONNX-side equivalent of TFLite `QUANTIZE`/`DEQUANTIZE` and are genuinely
# fine to drop -- TinyEngine already runs natively quantized. `Reshape`
# (`RESHAPE`) is also dropped by the real generator, but with that repo's own
# "TODO: Handle RESHAPE during codegen" comment attached -- see this module's
# docstring. Kept in its own set, not merged into `DIRECTLY_SUPPORTED_OPS`,
# so callers can tell "recognized and code-generated" from "recognized and
# silently skipped" apart.
SKIP_OPS: frozenset = frozenset({"QuantizeLinear", "DequantizeLinear", "Reshape"})

RESHAPE_SKIP_CAVEAT = (
    "TinyEngine's own constant.py lists RESHAPE in SKIP_OPs with a "
    '"TODO: Handle RESHAPE during codegen" comment attached -- the node is '
    "dropped, not code-generated, which is only safe if the reshape is a "
    "true no-op in the generated buffer's own memory layout"
)

# ONNX ops that have a real, valid TFLite BuiltinOperator enum value in
# TinyEngine's own vendored schema bindings (`tflite/BuiltinOperator.py`)
# but no case anywhere in `TfliteConvertor._handleOperator`'s dispatch --
# named individually because a schema-only check would wrongly call these
# supported. Not exhaustive; anything not otherwise recognized by this
# module already falls through to the same generic "no dispatch case"
# reason via `_reason`.
UNSUPPORTED_WITH_REAL_TFLITE_OPCODE: frozenset = frozenset(
    {"PRelu", "LeakyRelu", "Sigmoid", "HardSigmoid", "Softmax"}
)

# Activations TFLite's own Conv2D/DepthwiseConv2D op can fuse in as a
# `FusedActivationFunction` field on the conv node itself, so they never
# reach `_handleOperator` as a standalone op -- but only when the conv is
# their sole, immediate producer (see `is_fusable_activation`). `Clip` here
# stands in for ONNX's spelling of `Relu6` (`Clip(0, 6)`), the activation
# MobileNet-style networks (and TinyEngine's own mcunet/proxyless model zoo)
# actually use.
FUSABLE_ACTIVATION_OPS: frozenset = frozenset({"Relu", "Clip"})

# The one op TinyEngine's real code generator handles through a lookahead
# rather than a direct dispatch case: `checkIfRequireSEelementmult`, run when
# the convertor is looking at a DEPTHWISE_CONV_2D node, scans forward for the
# exact ADD -> MUL -> MUL triple a squeeze-and-excite gate produces and fuses
# the whole triple via `parse_SEelement`. `se_window_mul_names` below walks
# the graph's own edges forward from every `Add` node to find the same
# three-op chain (an `Add` consumed by a `Mul`, whose output is itself
# consumed by another `Mul`) and marks *both* Muls as part of a matched
# window, since the real fusion consumes the whole triple rather than just
# its last op. That is a real, checkable approximation of the same shape,
# not the identical algorithm -- in particular, the real check additionally
# requires the chain to originate from a `DEPTHWISE_CONV_2D`'s own output,
# which this module does not verify.
SE_MUL_OP = "Mul"


class BlockingOp(NamedTuple):
    node_name: str
    op_type: str
    reason: str


def _iter_all_nodes(graph: onnx.GraphProto):
    """Yield every node in ``graph``, recursing into subgraph attributes."""
    for node in graph.node:
        yield node
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                yield from _iter_all_nodes(attr.g)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for g in attr.graphs:
                    yield from _iter_all_nodes(g)


def _producer_map(graph: onnx.GraphProto) -> Dict[str, onnx.NodeProto]:
    """{output tensor name: producing node}, across the whole graph incl. subgraphs."""
    producers: Dict[str, onnx.NodeProto] = {}
    for node in _iter_all_nodes(graph):
        for out in node.output:
            if out:
                producers[out] = node
    return producers


def _consumer_counts(graph: onnx.GraphProto) -> Dict[str, int]:
    """{tensor name: number of nodes that consume it as an input}."""
    counts: Dict[str, int] = {}
    for node in _iter_all_nodes(graph):
        for inp in node.input:
            if inp:
                counts[inp] = counts.get(inp, 0) + 1
    return counts


def is_fusable_activation(
    node: onnx.NodeProto,
    producer_of: Dict[str, onnx.NodeProto],
    consumer_count_of: Dict[str, int],
) -> bool:
    """True if `node` (a Relu/Clip) is immediately, solely fed by a `Conv`.

    Approximates TFLite's `FusedActivationFunction` fusion -- see this
    module's docstring. Requires the conv's output to have exactly one
    consumer (this node): a conv output also used elsewhere (e.g. a
    residual branch) would keep the raw conv output around anyway, which
    is exactly the shape real fusion passes decline to fuse through.
    """
    if not node.input:
        return False
    producer = producer_of.get(node.input[0])
    if producer is None or producer.op_type != "Conv":
        return False
    return consumer_count_of.get(node.input[0], 0) == 1


def se_window_mul_names(model: onnx.ModelProto) -> Set[str]:
    """Names of every `Mul` node that is part of some `Add -> Mul -> Mul`
    chain -- both the middle and the trailing `Mul`, not just the trailing
    one. TinyEngine's real `parse_SEelement` fuses the whole three-op
    window into one unit, so from the compat heuristic's point of view
    *neither* `Mul` in a matched window is a standalone, unsupported `Mul`
    -- only the trailing node of the pair is what a purely backward,
    per-node walk would otherwise reach; this scans every `Mul` forward
    from each `Add` instead, so both members of a matched window clear.

    See `SE_MUL_OP`'s comment above for exactly what this does and does not
    verify relative to TinyEngine's real `checkIfRequireSEelementmult`.
    """
    consumers_of: Dict[str, List[onnx.NodeProto]] = {}
    for node in _iter_all_nodes(model.graph):
        for inp in node.input:
            if inp:
                consumers_of.setdefault(inp, []).append(node)

    matched: Set[str] = set()
    for node in _iter_all_nodes(model.graph):
        if node.op_type != "Add":
            continue
        for out in node.output:
            for mid in consumers_of.get(out, []):
                if mid.op_type != "Mul":
                    continue
                for mid_out in mid.output:
                    for trailing in consumers_of.get(mid_out, []):
                        if trailing.op_type == "Mul":
                            matched.add(mid.name)
                            matched.add(trailing.name)
    return matched


def _resize_reason_if_blocked(node: onnx.NodeProto) -> Optional[str]:
    """`Resize` only maps onto `RESIZE_NEAREST_NEIGHBOR`; any other mode has
    no TFLite/TinyEngine equivalent in this dispatch at all."""
    mode = "nearest"
    for attr in node.attribute:
        if attr.name == "mode":
            mode = attr.s.decode() if isinstance(attr.s, bytes) else attr.s
    if mode != "nearest":
        return (
            f"Resize(mode={mode!r}) has no TFLite equivalent dispatched by "
            "TfliteConvertor -- only nearest-neighbor resize "
            "(RESIZE_NEAREST_NEIGHBOR) is handled"
        )
    return None


def _reason(
    node: onnx.NodeProto,
    producer_of: Dict[str, onnx.NodeProto],
    consumer_count_of: Dict[str, int],
    se_window_names: Set[str],
) -> Optional[str]:
    """None if `node` is not a blocker; else a human-readable reason."""
    op_type = node.op_type

    if op_type == "Resize":
        return _resize_reason_if_blocked(node)
    if op_type in DIRECTLY_SUPPORTED_OPS:
        return None
    if op_type in SKIP_OPS:
        return None
    if op_type == SE_MUL_OP:
        if node.name in se_window_names:
            return None
        return (
            "Mul has no standalone case in TfliteConvertor._handleOperator -- "
            "it is only recognized as part of the exact Add->Mul->Mul "
            "squeeze-and-excite window checkIfRequireSEelementmult looks "
            "for after a depthwise conv"
        )
    if op_type in FUSABLE_ACTIVATION_OPS:
        if is_fusable_activation(node, producer_of, consumer_count_of):
            return None
        return (
            f"{op_type} has no standalone dispatch case -- it is only "
            "representable as a Conv2D/DepthwiseConv2D's own fused "
            "activation, which requires being that conv's sole, immediate "
            "consumer"
        )
    if op_type in UNSUPPORTED_WITH_REAL_TFLITE_OPCODE:
        return (
            f"{op_type} has a real TFLite BuiltinOperator opcode but no case "
            "in TfliteConvertor._handleOperator's dispatch"
        )
    return (
        f"{op_type} has no case in TfliteConvertor._handleOperator's "
        "dispatch and no fallback exists -- would raise NotImplementedError"
    )


def blocking_ops(model: onnx.ModelProto) -> List[BlockingOp]:
    """Every node this heuristic believes TinyEngine's real dispatch would reject."""
    producer_of = _producer_map(model.graph)
    consumer_count_of = _consumer_counts(model.graph)
    se_window_names = se_window_mul_names(model)
    out = []
    for node in _iter_all_nodes(model.graph):
        reason = _reason(node, producer_of, consumer_count_of, se_window_names)
        if reason is not None:
            out.append(BlockingOp(node.name, node.op_type, reason))
    return out


def blocking_op_types(model: onnx.ModelProto) -> Set[str]:
    return {op.op_type for op in blocking_ops(model)}


def skip_op_nodes(model: onnx.ModelProto) -> List[str]:
    """Names of nodes TinyEngine would silently drop rather than code-generate.

    Currently only `Reshape` carries a real caveat (see `RESHAPE_SKIP_CAVEAT`);
    `QuantizeLinear`/`DequantizeLinear` are included for completeness but are
    not themselves a risk.
    """
    return [
        node.name for node in _iter_all_nodes(model.graph) if node.op_type in SKIP_OPS
    ]


def skip_op_types(model: onnx.ModelProto) -> Set[str]:
    """Distinct SKIP_OPS op types actually present in `model`."""
    return {
        node.op_type
        for node in _iter_all_nodes(model.graph)
        if node.op_type in SKIP_OPS
    }


def has_dynamic_shape(model: onnx.ModelProto) -> bool:
    """True if any graph input has a symbolic/unknown dimension.

    A source-code generator emits fixed-size C buffers and loop bounds; it
    cannot compile a graph whose shapes aren't known ahead of time, the same
    hard requirement `scripts/edgeai/tidl_ops.has_dynamic_shape` documents
    for a different reason (a fixed-shape accelerator partitioner there,
    fixed-size generated buffers here).
    """
    initializer_names = {init.name for init in model.graph.initializer}
    for inp in model.graph.input:
        if inp.name in initializer_names:
            continue
        if not inp.type.HasField("tensor_type"):
            continue
        shape = inp.type.tensor_type.shape
        if not shape.dim:
            continue
        for dim in shape.dim:
            if dim.HasField("dim_param") or not dim.HasField("dim_value"):
                return True
    return False
