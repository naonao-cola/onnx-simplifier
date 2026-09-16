#!/usr/bin/env python3
"""Static TIDL (TI edgeai accelerator) coverage check -- no compiler, no device.

Unlike ``scripts/qualcomm/qnn_backend.py``, ``scripts/intel/openvino_backend.py``
and ``scripts/amd/migraphx_backend.py``, this module wraps no real compiler:
TIDL has no PyPI package and no plain-pip ONNX Runtime execution provider to
invoke here (see ``tidl_ops.py``'s docstring for why), so there is nothing to
install or emulate. ``TIDL_AVAILABLE`` is always True and this always runs on
a plain CPU host with only ``onnx`` installed -- it exists for interface
symmetry with the sibling EP backends, not because availability can vary.

What this *can* do without a real compiler: flag when onnxsim turns a graph
region that had no known TIDL blocker into one that does, and flag inputs
whose shape isn't fully static (TIDL requires static shapes end to end --
see ``tidl_ops.has_dynamic_shape``'s docstring).
"""

from __future__ import annotations

from typing import List, Set

import onnx
from tidl_ops import (
    BlockingOp,
    blocking_op_types,
    blocking_ops,
    has_decomposed_normalization,
    has_dynamic_shape,
    has_string_tensor,
)

TIDL_AVAILABLE = True


def unavailable_reason() -> None:
    """Always None: this check has no external dependency to be missing."""
    return None


def coverage(model: onnx.ModelProto) -> str:
    """ "full" if no known TIDL blocker was found, else "partial".

    "full" here means "this harness found no reason TIDL's accelerator
    partitioner would reject part of the graph" -- a heuristic based on
    published documentation, not a guarantee the whole graph maps onto the
    accelerator (see ``tidl_ops.py``'s docstring).
    """
    return (
        "partial"
        if (
            blocking_ops(model)
            or dynamic_shape_risks(model)
            or normalization_risks(model)
        )
        else "full"
    )


def blockers(model: onnx.ModelProto) -> List[BlockingOp]:
    return blocking_ops(model)


def new_blocking_op_types(orig: onnx.ModelProto, simp: onnx.ModelProto) -> Set[str]:
    """Blocking op types present after simplification but not before.

    An empty result does not mean simplification is TIDL-safe overall --
    only that it did not *introduce* a new known-blocking op type relative to
    the original graph.
    """
    return blocking_op_types(simp) - blocking_op_types(orig)


def dynamic_shape_risks(model: onnx.ModelProto) -> List[str]:
    """Human-readable reasons TIDL's static-shape compiler would reject `model`."""
    risks = []
    if has_dynamic_shape(model):
        risks.append(
            "one or more graph inputs have a symbolic or unranked dimension; "
            "TIDL requires fully static input shapes"
        )
    if has_string_tensor(model):
        risks.append("model uses a STRING tensor; TIDL has no string op support")
    return risks


def normalization_risks(model: onnx.ModelProto) -> List[str]:
    """Human-readable reasons `model` may not offload as a single fused unit.

    Only checks for the decomposed-LayerNorm signature (see
    `tidl_ops.has_decomposed_normalization`'s docstring) -- absence doesn't
    guarantee every op fuses cleanly, only that this one known pattern wasn't
    found.
    """
    if has_decomposed_normalization(model):
        return [
            "graph spells LayerNorm out as separate ReduceMean/Sub/Pow/Sqrt/"
            "Div-style ops instead of using the fused LayerNormalization op; "
            "edgeai-tidl-tools' transformer-support notes recommend the fused "
            "form"
        ]
    return []
