#!/usr/bin/env python3
"""Heuristics for Axera's Pulsar2/AXCL NPU compiler stack, without a real compiler.

Axera's `pulsar2 build` toolchain (the compiler behind AXCL / `.axmodel` for
the AX6xx/AX8xx NPU line) is a Docker-only download with no PyPI package and
no ONNX Runtime execution provider -- unlike the Qualcomm QNN
(`scripts/qualcomm`), Intel OpenVINO (`scripts/intel`), and AMD MIGraphX
(`scripts/amd`) stacks, there is no way to actually invoke it here to measure
real op coverage. See `../../../junk/axcl-axmodel-onnxsim-notes.md` (the
handoff notes this harness is based on) for the fuller picture.

**Update, verified against a real AX650N + a real compiled `.axmodel`
(`AXERA-TECH/YOLOv8`'s `AX650/yolov8n_640x640_npu1.axmodel`):** the notes'
§4 guess that an embedded NPU subgraph would show up as a node with a
*non-standard ONNX `domain`* turned out to be **wrong**. The real file has
exactly one node, `op_type="neu mode"` in the plain default domain (`""`),
whose declared graph input is only the model's real input (`images`) -- the
NPU weight/command blobs are ordinary UINT8 `graph.initializer` tensors
(`npu_params`, `<node_name>_b<N>_neu`) that are **not** referenced as a node
input at all. Instead, the node's `npu_graph_info` attribute is a JSON string
naming them by key (`const_data_key`, `neu_key`); a sibling `outputs_info`
attribute JSON-describes the outputs' dtype/shape. See `AXERA_NPU_OP_TYPE`,
`referenced_const_data_keys()` and `missing_npu_data()` below, and
`inspect_axmodel.py`, which is what produced this.

This out-of-band reference is exactly the fragility §4(a) of the handoff
notes worried about, and it is not hypothetical: `missing_npu_data()` exists
because running plain `onnxsim.simplify()` on this real file reproducibly
**strips `npu_params`/`npu_dyn_params`/`*_neu` from the output** -- both
onnxsim's own constant-folding cleanup and onnx-optimizer's
`eliminate_unused_initializer`/`eliminate_deadend` passes independently treat
them as dead, since neither is a declared node *input* -- shrinking the file
from 3.4MB to 1.7KB. The result fails to even load via `axcl_run_model` on
the real device ("Create model handle failed").

**There is no working flag-only avoidance**, confirmed by exhausting the
combinations: `skip_constant_folding=True` alone still strips them (the
onnx-optimizer passes get them); `skipped_optimizers=["eliminate_unused_
initializer", "eliminate_deadend"]` alone still strips them (constant folding
gets them); doing *both* keeps the 3 initializers byte-identical, but the
result *still* fails to load, because onnxsim's shape-inference/graph-rebuild
also unconditionally drops the `graph.value_info` entries describing those
tensors (10 -> 0, even with `skip_shape_inference=True`) and Axera's runtime
needs those too. In short: as of this onnxsim version, **feeding it an
already-compiled `.axmodel` is not safe with any public `simplify()`
parameter combination** -- see `has_out_of_band_npu_data()` below, meant to
be checked *before* calling `simplify()` at all, and `worker.py`/`missing_
npu_data()` to catch it *after* if that guard is skipped.

For everything Pulsar2 itself hasn't confirmed (the general NPU-vs-CPU op
split, since only one node's contents were ever visible in this one file),
this module still falls back to the safer complement it started with: op
types extremely unlikely to be NPU-schedulable on *any* fixed-function NPU
toolchain (control flow, sequence/optional types, string ops, ops with
data-dependent output shape), plus the original (now known to be
Axera-specific-case-of, not general-case) non-standard-`domain` check, kept
as a heuristic for vendor blobs that don't follow Axera's exact convention.
Presence of one of these is a strong signal the graph (or region of it) will
not run on Pulsar2's NPU as-is; *absence* is not proof the rest is
NPU-schedulable, only that this harness found no known blocker.

**Update: the real NPU op-support list is no longer an open question.**
Axera publishes it -- `AX650_SUPPORTED_OPS` below is the 92 op names scraped
from Pulsar2 V7.0's docs (`appendix/op_support_list_ax650.html`), which also
states **the whole model must be ONNX opset >= 11** (`AX650_MIN_OPSET`).
`unsupported_on_ax650()` flags op types outside that list.

This was exercised end-to-end with the real `pulsar2:6.0-lite` Docker
toolchain + a real AX650N, converting two real `onnxmodelzoo` models:

- `resnet18d_Opset18` (opset 18, all ops in `AX650_SUPPORTED_OPS`): both the
  original ONNX and its onnxsim-simplified twin compiled to a single NPU
  subgraph with **identical compiler-reported `max_cycle`
  (1,318,764)**, and running both `.axmodel`s on the real device with the
  same input produced **bit-identical output** (`np.array_equal` `True`,
  max abs diff `0.0`) -- the strongest evidence so far that onnxsim's
  approach-(b) simplification (§4(b) of the handoff notes) is safe for
  Pulsar2.
- `googlenet-6` (opset 9, uses `LRN` -- not in `AX650_SUPPORTED_OPS`):
  `pulsar2 build` did **not** gracefully fall LRN back to CPU the way the
  handoff notes' §2 description of mixed CPU/NPU graphs implied it might --
  it hard-failed the whole build at the frontend parse stage with
  `KeyError('dont support LRN opr in AXOPS/ONNXOPS/CUSTOM_OPS')` before any
  CPU/NPU partitioning happened. So an op outside `AX650_SUPPORTED_OPS` is
  not just "less NPU-friendly," it can make `pulsar2 build` refuse the model
  outright -- confirmed for `LRN`, not verified for every other absent op.

**Update: LLMs, resolving the handoff notes' §3.** `pulsar2 build` (this
module's focus so far) is not how Axera compiles LLMs at all -- that's a
completely separate subcommand, `pulsar2 llm_build` (the real `pulsar2:
6.0-lite` toolchain's name; Pulsar2's own newer docs describe a
`llm_build2` with a slightly different flag set, e.g.
`--max_context`/`--prefill_step_size` instead of `--kv_cache_len` -- v6.0
only has `llm_build`). Confirmed by actually running it, on a real
`Qwen/Qwen3-0.6B` checkpoint through the real toolchain, then on the real
AX650N (see `pulsar2_docker.llm_build()`'s docstring for the full
command/timing):

- **`--input_path` is a raw HuggingFace checkpoint directory**
  (`*.safetensors`/`pytorch_model.bin` + `config.json`) -- there is no ONNX
  step anywhere in this pipeline. The public `ax-llm-build` project Pulsar2's
  docs point to for this workflow (github.com/AXERA-TECH/ax-llm-build)
  contains no model-tracing/export code at all, only per-architecture config
  JSONs and small helper scripts around the actual (closed-source)
  `pulsar2 llm_build` call. **onnxsim has no direct integration point in
  this ingestion path** -- there is no ONNX graph for it to simplify or
  quantize before Pulsar2 ever sees the model. `-w`/`--weight_type` (default
  `s8`) is Pulsar2's *own* built-in weight quantization, entirely separate
  from `pulsar2_quantizer.py`.
- Output is **one compiled `.axmodel` per transformer layer** (28 for this
  0.6B model) plus one `_post.axmodel` (the LM head) -- confirming the
  handoff notes' guess that LLMs are "a directory of small, structurally
  similar single-block graphs," not one big graph.
- Each per-layer file has **two** `AXERA_NPU_OP_TYPE` nodes, not one: a
  decode subgraph (batch-1 shapes) and a prefill subgraph
  (`prefill_len`-batch shapes), both sharing one `npu_params` initializer.
  Both also carry explicit `K_cache`/`V_cache` graph inputs *and*
  `K_cache_out`/`V_cache_out` outputs -- the KV cache is ordinary graph
  tensors the host runtime persists between calls, not something opaque
  inside the compiled blob. `npu_subgraph_nodes()`/`has_out_of_band_npu_data()`/
  `missing_npu_data()` below already handle multiple NPU nodes per graph
  correctly with no changes needed -- verified: `onnxsim.simplify()`
  corrupts a real per-layer `.axmodel` the same way as the CNN case (3
  initializers -> 0).
- Both a per-layer file and the post model ran successfully on the real
  AX650N via `axcl_run_model` (~1.5ms and ~9ms respectively).
"""

from __future__ import annotations

import json
from typing import Dict, List, NamedTuple, Set

import onnx

# The real AX650 NPU op-support list, from Pulsar2 V7.0's own docs
# (appendix/op_support_list_ax650.html, "NPU Operators support list (AX650)").
# That page also states the whole model must be ONNX opset >= 11
# (AX650_MIN_OPSET) and that this covers AX650A/AX650N/AX8850/M76H alike.
# Each op's docs page also lists attribute-level limits (e.g. Conv's auto_pad
# must be NOTSET) not captured here -- this module only checks op *type*.
AX650_MIN_OPSET = 11
AX650_SUPPORTED_OPS: frozenset = frozenset(
    {
        "Abs",
        "Add",
        "And",
        "ArgMax",
        "ArgMin",
        "AveragePool",
        "BatchNormalization",
        "Cast",
        "Ceil",
        "Clip",
        "Concat",
        "Constant",
        "ConstantOfShape",
        "Conv",
        "ConvTranspose",
        "Cos",
        "DepthToSpace",
        "Div",
        "Elu",
        "Equal",
        "Erf",
        "Exp",
        "Expand",
        "Flatten",
        "Floor",
        "Gather",
        "GatherElements",
        "GatherND",
        "Gelu",
        "Gemm",
        "GlobalAveragePool",
        "GlobalMaxPool",
        "Greater",
        "GreaterOrEqual",
        "GridSample",
        "GroupNormalization",
        "HardSigmoid",
        "HardSwish",
        "Identity",
        "InstanceNormalization",
        "InverseSigmoid",
        "LSTM",
        "LayerNormalization",
        "LeakyRelu",
        "Less",
        "LessOrEqual",
        "LogSoftmax",
        "LpNormalization",
        "MatMul",
        "Max",
        "MaxPool",
        "Min",
        "Mish",
        "Mul",
        "Not",
        "PRelu",
        "Pad",
        "Pow",
        "RMSNormalization",
        "ReduceL2",
        "ReduceMax",
        "ReduceMean",
        "ReduceMin",
        "ReduceSum",
        "Relu",
        "Reshape",
        "Resize",
        "RoiAlign",
        "RotaryEmbedding",
        "Round",
        "ScatterElements",
        "ScatterND",
        "Sigmoid",
        "Silu",
        "Sin",
        "Slice",
        "Softmax",
        "Softplus",
        "SpaceToDepth",
        "SpatialTransformer",
        "Split",
        "Sqrt",
        "Squeeze",
        "Sub",
        "Swish",
        "Tanh",
        "Tile",
        "Topk",
        "Transpose",
        "Unsqueeze",
        "Where",
        "Xor",
    }
)

# Confirmed (not guessed) against a real compiled `.axmodel` run on an AX650N:
# the op_type Pulsar2 gives a node whose contents are an opaque, already
# NPU-compiled subgraph. Domain stays the plain default ("") -- see this
# module's docstring for why the domain-based heuristic below doesn't catch
# this on its own.
AXERA_NPU_OP_TYPE = "neu mode"

# Standard ONNX domains a compliant model may use without it being a vendor
# extension. Anything else on a node's `domain` field is exactly the
# mechanism ONNX documents for vendor-specific ops -- see §4 of the handoff
# notes for why an embedded NPU blob is expected to show up this way.
STANDARD_DOMAINS = frozenset({"", "ai.onnx", "ai.onnx.ml", "ai.onnx.training"})

# Op types that no mainstream fixed-function NPU compiler (Pulsar2 included,
# on priors -- this is not Pulsar2-specific data) schedules on-device: control
# flow needs a host to drive it, Sequence/Optional are container types with no
# NPU tensor representation, string ops have no NPU numeric kernel, and
# NonZero/Unique have data-dependent output shapes an ahead-of-time NPU
# compile can't size. Each maps to a short reason for `coverage()` diagnostics.
CPU_ONLY_OPS: Dict[str, str] = {
    "If": "control flow",
    "Loop": "control flow",
    "Scan": "control flow",
    "SequenceMap": "control flow",
    "SequenceConstruct": "sequence type, no NPU tensor representation",
    "SequenceEmpty": "sequence type, no NPU tensor representation",
    "SequenceInsert": "sequence type, no NPU tensor representation",
    "SequenceErase": "sequence type, no NPU tensor representation",
    "SequenceAt": "sequence type, no NPU tensor representation",
    "SequenceLength": "sequence type, no NPU tensor representation",
    "SplitToSequence": "sequence type, no NPU tensor representation",
    "ConcatFromSequence": "sequence type, no NPU tensor representation",
    "Optional": "optional type, no NPU tensor representation",
    "OptionalGetElement": "optional type, no NPU tensor representation",
    "OptionalHasElement": "optional type, no NPU tensor representation",
    "StringNormalizer": "string op, no NPU numeric kernel",
    "StringConcat": "string op, no NPU numeric kernel",
    "StringSplit": "string op, no NPU numeric kernel",
    "RegexFullMatch": "string op, no NPU numeric kernel",
    "NonZero": "data-dependent output shape",
    "Unique": "data-dependent output shape",
}


class BlockingOp(NamedTuple):
    node_name: str
    op_type: str
    domain: str
    reason: str


def blocking_ops(model: onnx.ModelProto) -> List[BlockingOp]:
    """Nodes (incl. subgraphs) unlikely to be Pulsar2-NPU-schedulable.

    Walks `If`/`Loop`/`Scan` subgraphs too, since a blocking op nested inside
    a branch/body is just as real a CPU-fallback trigger as one at top level.
    """
    found: List[BlockingOp] = []

    def visit(graph: onnx.GraphProto) -> None:
        for node in graph.node:
            if node.domain not in STANDARD_DOMAINS:
                found.append(
                    BlockingOp(
                        node.name or f"<{node.op_type}>",
                        node.op_type,
                        node.domain,
                        f"non-standard domain {node.domain!r} "
                        "(likely a vendor/compiled blob, see notes §4)",
                    )
                )
            elif node.op_type in CPU_ONLY_OPS:
                found.append(
                    BlockingOp(
                        node.name or f"<{node.op_type}>",
                        node.op_type,
                        node.domain,
                        CPU_ONLY_OPS[node.op_type],
                    )
                )
            for attr in node.attribute:
                if attr.HasField("g"):
                    visit(attr.g)
                for sub_g in attr.graphs:
                    visit(sub_g)

    visit(model.graph)
    return found


def blocking_op_types(model: onnx.ModelProto) -> set:
    """Just the distinct op types from `blocking_ops`, for before/after diffing."""
    return {b.op_type for b in blocking_ops(model)}


def unsupported_on_ax650(model: onnx.ModelProto) -> Set[str]:
    """Op types in `model` that are not on the confirmed AX650 op list.

    Unlike `blocking_ops` (a generic, cross-vendor guess), this is Axera's
    own published AX650 op list -- but "not on the list" is not uniformly
    "falls back to CPU": confirmed for `LRN`, it instead makes `pulsar2
    build` hard-fail at the frontend parse stage before any CPU/NPU
    partitioning (see this module's docstring). Treat any hit here as
    "verify empirically," not "safe to ignore."
    """
    return {
        node.op_type
        for node in model.graph.node
        if node.op_type not in AX650_SUPPORTED_OPS and node.op_type != AXERA_NPU_OP_TYPE
    }


def opset_version(model: onnx.ModelProto) -> int:
    """The model's default-domain opset version, or 0 if it has none."""
    return max(
        (o.version for o in model.opset_import if o.domain in ("", "ai.onnx")),
        default=0,
    )


def below_ax650_min_opset(model: onnx.ModelProto) -> bool:
    """True if `model`'s opset is below what AX650_MIN_OPSET requires.

    A model at 0 (no default-domain opset import at all) is not flagged --
    that is a malformed-model question, not an opset-too-old one.
    """
    v = opset_version(model)
    return 0 < v < AX650_MIN_OPSET


def npu_subgraph_nodes(model: onnx.ModelProto) -> List[onnx.NodeProto]:
    """Nodes matching Axera's confirmed compiled-NPU-subgraph marker."""
    return [
        node
        for node in model.graph.node
        if node.op_type == AXERA_NPU_OP_TYPE and node.domain in STANDARD_DOMAINS
    ]


def referenced_const_data_keys(node: onnx.NodeProto) -> Set[str]:
    """Initializer names a `neu mode` node depends on out-of-band.

    Parses the `npu_graph_info` attribute's JSON (confirmed format -- see this
    module's docstring): `{"dotneus": [{"neu_key": ..., "extra_inputs": [
    {"const_data_key": ...}, ...]}, ...]}`. These names are the node's *real*
    data dependencies even though none of them appear in `node.input` --
    onnx-optimizer's dead-initializer elimination has no way to see that.
    """
    keys: Set[str] = set()
    for attr in node.attribute:
        if attr.name != "npu_graph_info" or not attr.s:
            continue
        try:
            info = json.loads(attr.s)
        except (ValueError, UnicodeDecodeError):
            continue
        for neu in info.get("dotneus", []):
            if "neu_key" in neu:
                keys.add(neu["neu_key"])
            for extra in neu.get("extra_inputs", []):
                if "const_data_key" in extra:
                    keys.add(extra["const_data_key"])
    return keys


def missing_npu_data(model: onnx.ModelProto) -> Set[str]:
    """Initializer names a `neu mode` node's metadata references but which are gone.

    A non-empty result means the file is broken: the node's NPU weight/command
    blob(s) were dropped from `graph.initializer` (typically by generic
    dead-initializer elimination that only understands declared node inputs --
    see this module's docstring for a reproduced, real-hardware-confirmed
    case of exactly that).
    """
    initializer_names = {init.name for init in model.graph.initializer}
    missing: Set[str] = set()
    for node in npu_subgraph_nodes(model):
        for key in referenced_const_data_keys(node):
            if key not in initializer_names:
                missing.add(key)
    return missing


def has_out_of_band_npu_data(model: onnx.ModelProto) -> bool:
    """True if `model` has data `onnxsim.simplify()` cannot currently round-trip.

    Call this *before* `simplify()`, not after: no combination of its public
    parameters (`skip_constant_folding`, `skipped_optimizers=[
    "eliminate_unused_initializer", "eliminate_deadend"]`, even
    `skip_shape_inference`) has been found to preserve both the referenced
    initializers and their `value_info` entries at once -- see this module's
    docstring. Right now the only confirmed-safe thing to do with a model
    that has NPU subgraph nodes is **not run it through onnxsim at all**, per
    the handoff notes' own recommendation to only simplify pre-`pulsar2
    build` ONNX (approach (b)), never an already-compiled `.axmodel`
    (approach (a)).
    """
    return any(referenced_const_data_keys(node) for node in npu_subgraph_nodes(model))
