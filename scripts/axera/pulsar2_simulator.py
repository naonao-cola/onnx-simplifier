#!/usr/bin/env python3
"""An axmodel *simulator*: NPU/CPU partition + numeric estimate, no Docker/device.

Shaped like the sibling `*_backend.py` modules (`coverage()`, `run_with_cpu()`,
a run-the-other-path function, `compare()`), but where those wrap a *real*
execution provider, this wraps two things that only need `onnx` + `onnxruntime`:

1. **Partitioning**, from `pulsar2_ops.AX650_SUPPORTED_OPS` -- the real AX650
   NPU operator-coverage list read out of Pulsar2's own docs and confirmed
   against real hardware (see `pulsar2_ops.py`'s docstring: it correctly
   predicted a real `resnet18d_Opset18` build succeeding fully-on-NPU and a
   real `googlenet-6` build hard-failing on `LRN`). `partition()` classifies
   every node the same way and reports what fraction would run on NPU.
2. **Numeric estimation**, via `pulsar2_quantizer.quantize_like_pulsar2()`
   run through onnxruntime's CPU EP -- an approximation of what the
   INT8-quantized NPU path would produce, since there is no real EP for
   Pulsar2 to run through (see `pulsar2_ops.py`'s docstring for why).

**This is an estimate, not an emulator** -- two things it does NOT do:

- It does not reproduce Pulsar2's actual quantized IR (`AxQuantizedConv` and
  friends, confirmed proprietary and non-ONNX-Runtime-executable -- see
  `pulsar2_quantizer.py`) or its exact fusion/rounding behavior.
- Partitioning here is purely per-node op-type membership in
  `AX650_SUPPORTED_OPS` (minus `AX650_CONFIRMED_BROKEN_OPS`, see below); it
  does not model attribute-level limits (e.g. Conv's `auto_pad` must be
  `NOTSET`) or how Pulsar2 actually groups/merges eligible nodes into
  subgraphs.

**Update, from the 91/92-op real-hardware coverage sweep:** 7 ops listed in
`AX650_SUPPORTED_OPS` (`ConvTranspose`, `Xor`, `Squeeze`, `LpNormalization`,
`RotaryEmbedding`, `Swish`, `InverseSigmoid`) are confirmed via real
`pulsar2 build` to hard-fail anyway -- see
`pulsar2_ops.AX650_CONFIRMED_BROKEN_OPS`. `partition()` now places these on
the CPU side (tracked in `Partition.confirmed_broken_op_types`) instead of
treating docs-list membership alone as NPU-eligible.

Use it to get a fast first read (does this graph look NPU-friendly? is
onnxsim's simplification likely to change that? roughly how much does INT8
hurt this model's outputs?) before spending time on `pulsar2 build` --
always confirm anything that matters on the real toolchain and hardware.

**Validated against real hardware** (`resnet18d_Opset18`, real AX650N, same
input image both times): `coverage()` correctly reported "full" (matching
the real build's single fused NPU subgraph) and "partial" with
`cpu_op_types={"LRN": 2, "Dropout": 1}` for `googlenet-6` (matching its real
hard build failure on `LRN`). For `simulate()`'s numeric side: cosine
similarity between this simulator's INT8 output (via `onnxsim.
quantize_static`, see `pulsar2_quantizer.py`) and the real device's actual
INT8 output was **0.938**, close to fp32-vs-real's own **0.949** -- i.e. this
simulator's quantization noise is roughly the same *magnitude* as Pulsar2's
real quantization noise, relative to the fp32 baseline. It is **not**
rank-accurate, though: top-5 class rankings did not match between fp32, this
simulator, and the real device on that (random-noise, no real semantic
content) test input -- small logit perturbations from any INT8 path are
enough to reorder closely-spaced logits. Treat `simulate()`'s output as "is
the quantization noise roughly sane," not "will this be the same top-1
prediction as real hardware."
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import onnx

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import pulsar2_quantizer  # noqa: E402
from pulsar2_ops import (  # noqa: E402
    AX650_CONFIRMED_BROKEN_OPS,
    AX650_SUPPORTED_OPS,
    AXERA_NPU_OP_TYPE,
)

_SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Only keep scripts/ on sys.path for the duration of this import: scripts/
# also holds directories like rfdetr/ with no __init__.py, which Python 3
# treats as importable namespace packages. Leaving scripts/ on sys.path for
# the rest of the process would make `import rfdetr` "succeed" as that empty
# namespace package instead of skipping via pytest.importorskip, and shadow
# the real one everywhere else it's checked for.
_inserted = _SCRIPTS_DIR not in sys.path
if _inserted:
    sys.path.insert(0, _SCRIPTS_DIR)
try:
    from common.ep_numerics import compare, random_feeds  # noqa: E402,F401
finally:
    if _inserted:
        sys.path.remove(_SCRIPTS_DIR)

SIMULATOR_AVAILABLE = pulsar2_quantizer.PULSAR2_QUANTIZER_AVAILABLE


def unavailable_reason() -> Optional[str]:
    """Why the numeric side of the simulator is unusable, or None if usable.

    `partition()`/`coverage()` need only `onnx` and work regardless; only the
    onnxruntime-based `simulate()` depends on this.
    """
    return pulsar2_quantizer.unavailable_reason()


@dataclass
class Partition:
    npu_nodes: List[str]
    cpu_nodes: List[str]
    cpu_op_types: Dict[str, int] = field(default_factory=dict)
    # Subset of `cpu_op_types` that ARE in AX650_SUPPORTED_OPS but were
    # placed on the CPU side anyway because a real single-node-per-op
    # hardware sweep confirmed they hard-fail a real build -- see
    # `pulsar2_ops.AX650_CONFIRMED_BROKEN_OPS`. Surfaced separately from the
    # rest of `cpu_op_types` (ops never claimed to be supported at all)
    # since these represent a confirmed gap in the docs-scraped list, not
    # just an ordinary CPU-fallback op.
    confirmed_broken_op_types: Dict[str, int] = field(default_factory=dict)

    @property
    def npu_node_fraction(self) -> float:
        total = len(self.npu_nodes) + len(self.cpu_nodes)
        return len(self.npu_nodes) / total if total else 1.0


def partition(model: onnx.ModelProto) -> Partition:
    """Classify every node as NPU-eligible or not, by `AX650_SUPPORTED_OPS`.

    A node whose op_type is `AXERA_NPU_OP_TYPE` (an already-compiled `neu
    mode` node, e.g. from re-loading a real `.axmodel`) counts as NPU, not
    CPU -- it's already placed, not something left over for the partitioner.

    An op type in `AX650_CONFIRMED_BROKEN_OPS` is placed on the CPU side
    even though it's also in `AX650_SUPPORTED_OPS`: a real hardware sweep
    confirmed these hard-fail a real build despite being docs-listed as
    supported (see `pulsar2_ops.py`'s docstring). Tracked separately in
    `Partition.confirmed_broken_op_types` so callers can distinguish "not
    docs-supported" from "docs-supported but confirmed broken."
    """
    npu: List[str] = []
    cpu: List[str] = []
    cpu_types: Dict[str, int] = {}
    broken_types: Dict[str, int] = {}
    for node in model.graph.node:
        label = node.name or f"<{node.op_type}>"
        if node.op_type == AXERA_NPU_OP_TYPE:
            npu.append(label)
        elif node.op_type in AX650_CONFIRMED_BROKEN_OPS:
            cpu.append(label)
            cpu_types[node.op_type] = cpu_types.get(node.op_type, 0) + 1
            broken_types[node.op_type] = broken_types.get(node.op_type, 0) + 1
        elif node.op_type in AX650_SUPPORTED_OPS:
            npu.append(label)
        else:
            cpu.append(label)
            cpu_types[node.op_type] = cpu_types.get(node.op_type, 0) + 1
    return Partition(npu, cpu, cpu_types, broken_types)


def coverage(model: onnx.ModelProto) -> str:
    """ "full" (all nodes NPU-eligible), "none", or "partial"."""
    p = partition(model)
    if not p.cpu_nodes:
        return "full"
    if not p.npu_nodes:
        return "none"
    return "partial"


def run_with_cpu(
    model: onnx.ModelProto, feeds: Dict[str, np.ndarray]
) -> List[np.ndarray]:
    """Run `model` on ONNX Runtime's CPU provider (the fp32 reference)."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3  # ERROR
    sess = ort.InferenceSession(
        model.SerializeToString(), sess_options=so, providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def run_npu_simulated(
    model: onnx.ModelProto,
    feeds: Dict[str, np.ndarray],
    calibration_feeds: Optional[List[Dict[str, np.ndarray]]] = None,
    seed: int = 0,
) -> List[np.ndarray]:
    """Run the Pulsar2-style-quantized graph (see `pulsar2_quantizer.py`).

    Without `calibration_feeds`, synthesizes 16 random calibration inputs
    (matching the real build's calibration set size) via `random_feeds`.
    """
    if calibration_feeds is None:
        calibration_feeds = [random_feeds(model, seed=seed + i) for i in range(16)]
    quantized = pulsar2_quantizer.quantize_like_pulsar2(model, calibration_feeds)
    return run_with_cpu(quantized, feeds)


def simulate(
    model: onnx.ModelProto,
    feeds: Optional[Dict[str, np.ndarray]] = None,
    seed: int = 0,
    rtol: float = 0.2,
    atol: float = 0.1,
) -> dict:
    """One-shot: partition + fp32-vs-simulated-INT8 comparison.

    The default `rtol`/`atol` are loose relative to `common.ep_numerics`'s
    defaults -- real INT8 PTQ is expected to move outputs measurably; this
    is meant to catch "wildly different" (a real problem), not "not
    bit-exact" (expected). Tighten for a model where you have a real-hardware
    reference to calibrate against -- see `pulsar2_ops.py`'s docstring for a
    worked example (real AX650N vs this simulator, on `resnet18d_Opset18`).
    """
    feeds = feeds if feeds is not None else random_feeds(model, seed=seed)
    fp32 = run_with_cpu(model, feeds)
    npu_sim = run_npu_simulated(model, feeds, seed=seed)
    close, max_diff = compare(fp32, npu_sim, rtol=rtol, atol=atol)
    return {
        "partition": partition(model),
        "fp32": fp32,
        "npu_simulated": npu_sim,
        "close": close,
        "max_abs_diff": max_diff,
    }
