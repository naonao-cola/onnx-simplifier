# onnxsim vs. Intel NNCF: comparison and future work

**Status: research note.** This records a comparison of onnxsim's quantization
series against Intel's [NNCF](https://github.com/openvinotoolkit/nncf)
(Neural Network Compression Framework), done to answer "how do we stack up,"
and lists concrete follow-up work the comparison surfaced. It is not a
commitment or a roadmap -- it exists so the comparison doesn't need to be
re-run from scratch next time the question comes up. A repo-wide search found
exactly one prior NNCF mention (`scripts/intel/README.md`, which explicitly
scopes an OpenVINO execution-provider compatibility check to fp32-only and
disclaims any overlap with NNCF's quantization pipeline) -- there is no
existing interop, comparison benchmark, or shared code path between the two
projects.

## Where the two tools actually differ

They aren't solving the same problem. NNCF is a general compression
*framework* -- PTQ **and** QAT (training-time, PyTorch), an accuracy-aware
search loop, and deep integration with OpenVINO's deployment pipeline and
Hugging Face Optimum Intel. onnxsim's quantization layer is pure
post-training graph rewriting: ~35 independent passes/algorithms
(`onnxsim/*.py`, `onnxsim/passes/*.h`) operating directly on ONNX protobufs,
with no training loop, targeting ONNX Runtime rather than OpenVINO IR.

| Dimension | onnxsim | NNCF |
|---|---|---|
| Paradigm | Pure PTQ, ONNX graph-to-graph | PTQ *and* QAT |
| Target runtime | ONNX Runtime | OpenVINO IR (also exports ONNX) |
| Weight-compression API | ~20 independent functions (`apply_gptq`, `apply_awq`, ...), unified only via `QuantizationConfig` | One composable `compress_weights()` with `awq=`/`gptq=`/`scale_estimation=`/`lora_correction=` flags |
| Advanced PTQ algorithm count | ~20 ported papers (GPTQ, AWQ, SmoothQuant, HQQ, AdaRound, AutoRound, SqueezeLLM, AQLM, QuIP#, QuaRot, SpinQuant, DuQuant, OmniQuant, SpQR, LLM.int8(), LoRC, k-means, double quant) | GPTQ, AWQ, Scale Estimation, LoRA Correction |
| Accuracy tooling | Data-free `precision_estimator` risk scoring + data-driven `accuracy-drop` measurement (both report-only) | Accuracy-aware quantization loop that auto-reverts sensitive layers to hit a target validation metric |
| Pruning | Magnitude + Wanda, N:M sparsity | Movement/magnitude pruning, JPQD (joint pruning + quantization + distillation) |
| Ecosystem | ONNX/ORT-only, single integrated CLI/Python surface, younger | Intel-backed, deeply wired into OpenVINO + Optimum Intel, production track record |

Full writeup of the comparison (per-scheme detail, file-level citations) is
in the session that produced this note; this file keeps only what's needed
to drive future decisions.

## Candidate future work

### 1. A composable weight-compression entry point
onnxsim already has a unified dispatcher (`QuantizationConfig` /
`onnxsim.quantize()`, see `docs/quantization-config.md`) but each advanced
algorithm (GPTQ, AWQ, Scale-Estimation-equivalent, LoRA-style correction) is
still called as its own function. NNCF's `compress_weights()` treats these as
composable *flags on one int4 pipeline* rather than mutually exclusive
schemes. Worth evaluating: can `QuantizationConfig` grow flags that chain
e.g. AWQ's rescale into GPTQ's Hessian pass into double-quantization of the
resulting scales, rather than requiring the caller to hand-sequence separate
function calls? This is additive to the existing dispatcher, not a rewrite.

### 2. An automated accuracy-aware search loop
`precision_estimator.py` and `accuracy.py` currently *report* risk/drop; they
don't *act* on it. `mixed_precision.py` does have a per-layer sensitivity
dispatcher, which is the closest existing building block. NNCF's
accuracy-aware quantization (iteratively reverting the most-sensitive layers
against a validation metric until a target is met) is a natural extension of
that dispatcher rather than a new concept -- the missing piece is the
iterate-until-target-met control loop and a formal target-metric API, not new
math.

### 3. A real onnxsim-vs-NNCF benchmark
The comparison above is architectural, not empirical -- no benchmark in this
repo actually runs the same model through both onnxsim and NNCF and compares
accuracy/latency/size. Given the algorithm surfaces overlap most directly on
int4 weight-only compression (onnxsim's `quantize_weight_only_int4` + GPTQ/AWQ
vs. NNCF's `compress_weights(mode=INT4_*, awq=True, gptq=True)`), that's the
most tractable starting point for a head-to-head benchmark script under
`bench/`, producing numbers instead of a qualitative comparison.

### 4. Structured pruning parity (lower priority)
onnxsim's pruning (`pruning.py`) covers magnitude + Wanda + N:M sparsity but
not movement pruning or joint pruning-quantization-distillation (JPQD).
Movement pruning would fit the existing pruning module's shape; JPQD requires
a training loop and so inherits the QAT question below -- treat it as blocked
on that decision, not as independent work.

## Explicitly out of scope (recorded so it isn't re-litigated)

- **Quantization-aware training (QAT).** onnxsim's whole architecture is
  stateless graph-rewriting on an existing ONNX protobuf; QAT needs a training
  loop with a framework-native model (NNCF's is PyTorch-based). Porting QAT
  would mean maintaining a second, fundamentally different code path rather
  than adding another pass, which is a much bigger scope change than anything
  else in this list. Not pursued unless there's a specific driving use case.
- **OpenVINO IR export/interop.** onnxsim targets ONNX Runtime; adding an
  OpenVINO IR export path duplicates what NNCF already does well and doesn't
  play to onnxsim's ONNX/ORT-centric strength.

## Sources

- [openvinotoolkit/nncf](https://github.com/openvinotoolkit/nncf)
- [NNCF weight compression usage docs](https://github.com/openvinotoolkit/nncf/blob/develop/docs/usage/post_training_compression/weights_compression/Usage.md)
- [NNCF post-training quantization usage docs](https://github.com/openvinotoolkit/nncf/blob/develop/docs/usage/post_training_compression/post_training_quantization/Usage.md)
- [OpenVINO weight compression guide](https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/weight-compression.html)
