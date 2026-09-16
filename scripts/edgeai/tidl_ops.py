#!/usr/bin/env python3
"""Static op-support heuristic for TI's TIDL (edgeai) accelerator offload.

`TexasInstruments/edgeai <https://github.com/TexasInstruments/edgeai>`_ is the
umbrella repo for TI's edge AI SDK -- model training/export/quantization
tooling (``edgeai-modeloptimization``, ``edgeai-tensorlab``,
``edgeai-tidl-tools``, ...) that targets **TIDL** (TI Deep Learning), the
inference engine for the C7x-MMA deep-learning accelerator on TI's
Jacinto/Sitara SoCs (TDA4x, AM62A/68A, ...).

Like Axera's Pulsar2 (see ``scripts/axera/pulsar2_ops.py``), TIDL has no PyPI
package and no ONNX Runtime execution provider installable in a plain CI
container: the TIDL-enabled ``onnxruntime`` build ships as part of TI's
PSDK/edgeai-tidl-tools SDK and needs either the target device or a matching
x86 "PC emulation" build, neither of which this repository provisions. So,
same as ``pulsar2_ops.py``, **this module wraps no real compiler and makes no
hardware-confirmed claims** -- everything here is a static heuristic derived
from TIDL's own published operator-support documentation (edgeai-tidl-tools'
"Supported operators" tables), not a real device, SDK, or simulator run. If a
runner with the real SDK is ever provisioned, replace this with an actual
model-import/compile check the way ``scripts/qualcomm``/``scripts/intel``/
``scripts/amd`` wrap a real execution provider.

Two things TIDL's public documentation states plainly enough to check for
without hardware:

1. **No dynamic shapes.** TIDL compiles a fixed-shape subgraph ahead of time;
   every graph input must have a fully static shape (including batch size).
   A symbolic/unknown dimension anywhere is a hard blocker, not a "runs on
   CPU fallback" case.
2. **Some ops just don't map onto the accelerator.** Control flow
   (``If``/``Loop``/``Scan``), the ``Sequence``/``Optional``/``Map`` types,
   and string tensors have no fixed-function NPU/DSP equivalent on *any*
   accelerator of this class -- this is the same generic complement
   ``pulsar2_ops.py`` and the sibling QNN/OpenVINO/MIGraphX backends use, not
   a TIDL-specific list. ``NonMaxSuppression`` is included too: TIDL's
   detection post-processing runs NMS on the host ARM core, not as an
   in-graph accelerator op, per edgeai-tidl-tools' own detection-model
   documentation.

Presence of one of these is a strong signal the graph (or region of it) will
not run on TIDL's accelerator as-is; *absence* is not proof the rest offloads
cleanly -- this harness only checks op *type* and shape-staticness, not the
per-op attribute-level constraints (e.g. supported ``Resize`` modes,
``Conv`` group/dilation limits) TIDL's docs also list.
"""

from __future__ import annotations

from typing import List, NamedTuple, Set

import onnx
from onnx import TensorProto

# Control flow: no fixed-function accelerator of this class runs a
# data-dependent loop or branch on-chip.
CONTROL_FLOW_OPS: frozenset = frozenset({"If", "Loop", "Scan"})

# Sequence/Optional/Map container types: TIDL's graph partitioner (like every
# fixed-shape NPU compiler in this repo's other vendor checks) works over
# plain tensors, not these ONNX-ML container ops.
SEQUENCE_OPTIONAL_OPS: frozenset = frozenset(
    {
        "SequenceConstruct",
        "SequenceAt",
        "SequenceEmpty",
        "SequenceErase",
        "SequenceInsert",
        "SequenceLength",
        "SequenceMap",
        "SplitToSequence",
        "ConcatFromSequence",
        "Optional",
        "OptionalGetElement",
        "OptionalHasElement",
    }
)

# Ops whose *output shape* is a function of runtime data, not just the input
# shape -- a fixed-shape compiler needs to know every tensor's shape ahead of
# time, so these can't be part of an offloaded subgraph.
DATA_DEPENDENT_SHAPE_OPS: frozenset = frozenset({"NonZero", "Unique", "Compress"})

# Detection-model NMS: edgeai-tidl-tools' own detection post-processing runs
# this on the host ARM core, not on the accelerator -- see this module's
# docstring.
HOST_ONLY_OPS: frozenset = frozenset({"NonMaxSuppression"})

BLOCKING_OP_TYPES: frozenset = frozenset(
    CONTROL_FLOW_OPS | SEQUENCE_OPTIONAL_OPS | DATA_DEPENDENT_SHAPE_OPS | HOST_ONLY_OPS
)


class BlockingOp(NamedTuple):
    node_name: str
    op_type: str
    reason: str


def _reason(op_type: str) -> str:
    if op_type in CONTROL_FLOW_OPS:
        return "control flow has no fixed-function TIDL accelerator equivalent"
    if op_type in SEQUENCE_OPTIONAL_OPS:
        return "Sequence/Optional container ops are not accelerator-schedulable"
    if op_type in DATA_DEPENDENT_SHAPE_OPS:
        return "output shape depends on runtime data, not just input shape"
    if op_type in HOST_ONLY_OPS:
        return "documented as running on the host ARM core, not the accelerator"
    return "unrecognized blocker category"


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


def blocking_ops(model: onnx.ModelProto) -> List[BlockingOp]:
    """Every node whose op type is a known TIDL-accelerator blocker."""
    return [
        BlockingOp(node.name, node.op_type, _reason(node.op_type))
        for node in _iter_all_nodes(model.graph)
        if node.op_type in BLOCKING_OP_TYPES
    ]


def blocking_op_types(model: onnx.ModelProto) -> Set[str]:
    return {op.op_type for op in blocking_ops(model)}


def has_dynamic_shape(model: onnx.ModelProto) -> bool:
    """True if any graph input has a symbolic/unknown dimension.

    TIDL compiles a fixed-shape subgraph; a ``dim_param`` (symbolic dim) or a
    missing ``dim_value`` anywhere in a graph input's shape means the model
    cannot be compiled as-is, regardless of which ops it uses.
    """
    initializer_names = {init.name for init in model.graph.initializer}
    for inp in model.graph.input:
        if inp.name in initializer_names:
            continue
        if not inp.type.HasField("tensor_type"):
            continue
        shape = inp.type.tensor_type.shape
        if not shape.dim:
            # No shape at all is unranked, which is even less static than a
            # dynamic dim -- also unsupported.
            continue
        for dim in shape.dim:
            if dim.HasField("dim_param") or not dim.HasField("dim_value"):
                return True
    return False


def has_string_tensor(model: onnx.ModelProto) -> bool:
    """True if any graph input/output/initializer is a STRING tensor."""
    initializer_names = {init.name for init in model.graph.initializer}
    for init in model.graph.initializer:
        if init.data_type == TensorProto.STRING:
            return True
    for value_info in (*model.graph.input, *model.graph.output):
        if value_info.name in initializer_names:
            continue
        if (
            value_info.type.HasField("tensor_type")
            and value_info.type.tensor_type.elem_type == TensorProto.STRING
        ):
            return True
    return False
