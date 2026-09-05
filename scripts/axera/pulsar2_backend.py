#!/usr/bin/env python3
"""Static Pulsar2 (Axera AXCL NPU) coverage check -- no compiler, no device.

Unlike `scripts/qualcomm/qnn_backend.py`, `scripts/intel/openvino_backend.py`
and `scripts/amd/migraphx_backend.py`, this module wraps no real compiler:
Pulsar2 has no pip package and no ONNX Runtime execution provider to invoke,
so there is nothing to install or emulate here (see `pulsar2_ops.py`'s
docstring for why). ``PULSAR2_AVAILABLE`` is always True and this always runs
on a plain CPU host with only ``onnx`` installed -- it exists for interface
symmetry with the sibling EP backends, not because availability can vary.

What this *can* do without a real compiler: flag when onnxsim turns a graph
region that had no known Pulsar2-NPU blocker into one that does (see
`pulsar2_ops.blocking_ops`). That is the concrete risk §4(b) of the handoff
notes calls out -- a simplification could fold something into a form the NPU
partitioner then refuses, silently pushing more of the graph onto Pulsar2's
CPU fallback path.

It can also catch something worse, confirmed against a real AX650N and a
real compiled `.axmodel`: `stripped_npu_data()` detects when onnxsim has
dropped a `neu mode` node's NPU weight/command blob because Axera references
it only by name inside a JSON attribute, not as a declared graph input (see
`pulsar2_ops.missing_npu_data`). That isn't a coverage regression, it's
outright file corruption -- the resulting `.axmodel` fails to even load, and
no combination of `simplify()`'s public parameters was found to avoid it
(see `pulsar2_ops.has_out_of_band_npu_data`, meant to be checked *before*
calling `simplify()` at all -- right now the only confirmed-safe answer for a
model that already has NPU subgraph nodes is not to run it through onnxsim).

`ax650_build_risks()` uses Axera's own published AX650 op list plus a real
`pulsar2:6.0-lite` + AX650N run to flag likely hard build failures ahead of
time (see `pulsar2_ops.py`'s docstring for the real `resnet18d`/`googlenet-6`
conversions this is based on).
"""

from __future__ import annotations

from typing import List

import onnx
from pulsar2_ops import (
    AX650_MIN_OPSET,
    BlockingOp,
    below_ax650_min_opset,
    blocking_op_types,
    blocking_ops,
    confirmed_broken_on_ax650,
    has_out_of_band_npu_data,
    missing_npu_data,
    opset_version,
    unsupported_on_ax650,
)

PULSAR2_AVAILABLE = True


def unavailable_reason() -> None:
    """Always None: this check has no external dependency to be missing."""
    return None


def coverage(model: onnx.ModelProto) -> str:
    """ "full" if no known Pulsar2-NPU blocker was found, else "partial".

    "full" here means "this harness found no reason Pulsar2 would reject
    part of the graph" -- a heuristic, not a guarantee the whole graph maps
    onto the NPU (see this module's docstring).
    """
    return "partial" if blocking_ops(model) else "full"


def blockers(model: onnx.ModelProto) -> List[BlockingOp]:
    return blocking_ops(model)


def new_blocking_op_types(orig: onnx.ModelProto, simp: onnx.ModelProto) -> set:
    """Blocking op types present after simplification but not before.

    An empty result does not mean simplification is Pulsar2-safe overall --
    only that it did not *introduce* a new known-blocking op type relative to
    the original graph.
    """
    return blocking_op_types(simp) - blocking_op_types(orig)


def stripped_npu_data(simp: onnx.ModelProto) -> set:
    """NPU weight/command initializer names a `neu mode` node needs but lost.

    Non-empty means the simplified model is broken, not just less
    NPU-friendly -- see this module's docstring.
    """
    return missing_npu_data(simp)


def unsafe_for_simplify(model: onnx.ModelProto) -> bool:
    """True if `model` should not be passed to `onnxsim.simplify()` at all.

    Pre-flight version of `stripped_npu_data()` -- call this first and skip
    simplification entirely rather than simplifying and checking after.
    """
    return has_out_of_band_npu_data(model)


def ax650_build_risks(model: onnx.ModelProto) -> List[str]:
    """Human-readable reasons `pulsar2 build --target_hardware AX650` might
    reject `model` outright, confirmed against the real toolchain + device:

    - an op type outside `pulsar2_ops.AX650_SUPPORTED_OPS` (confirmed for
      `LRN`: a hard frontend parse failure, not a graceful CPU fallback);
    - an op type confirmed broken *despite* being listed in
      `pulsar2_ops.AX650_SUPPORTED_OPS` (`pulsar2_ops.AX650_CONFIRMED_BROKEN_OPS`
      -- 7 ops confirmed via a real single-node-per-op hardware sweep, see
      that module's docstring);
    - an opset below `pulsar2_ops.AX650_MIN_OPSET` (11), which Pulsar2's own
      docs state as a hard requirement.

    Empty does not guarantee a successful build -- only that this harness
    found none of the specific risks it currently knows to check for.
    """
    risks = []
    unsupported = sorted(unsupported_on_ax650(model))
    if unsupported:
        note = (
            " -- LRN specifically is a confirmed hard build failure, not a "
            "CPU fallback; the others here are untested"
            if "LRN" in unsupported
            else " (untested here whether pulsar2 build hard-fails on these "
            "or falls back to CPU -- only LRN has been confirmed to hard-fail)"
        )
        risks.append(
            f"op type(s) not on the confirmed AX650 op list: {unsupported}{note}"
        )
    broken = confirmed_broken_on_ax650(model)
    for op_type, reason in sorted(broken.items()):
        risks.append(
            f"op type {op_type!r} is listed in AX650_SUPPORTED_OPS but "
            f"confirmed to hard-fail a real build: {reason}"
        )
    if below_ax650_min_opset(model):
        risks.append(
            f"opset {opset_version(model)} is below Pulsar2's documented "
            f"minimum of {AX650_MIN_OPSET} for AX650"
        )
    return risks
