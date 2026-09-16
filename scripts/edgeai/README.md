# TI edgeai / TIDL static compatibility check

Verifies that `onnxsim`'s output stays friendly to **TIDL** (TI Deep
Learning), the inference engine TI's [edgeai](https://github.com/TexasInstruments/edgeai)
SDK (`edgeai-modeloptimization`, `edgeai-tensorlab`, `edgeai-tidl-tools`, ...)
compiles ONNX models for, targeting the C7x-MMA deep-learning accelerator on
Jacinto/Sitara SoCs (TDA4x, AM62A/68A, ...).

## Two checks, not one: a static heuristic, and now a real compile

`tidl_ops.py`/`tidl_backend.py` are a static heuristic that needs nothing
beyond `onnx` -- it runs on every PR. `real_compile.py` is a *genuinely
real* compile/import via TI's own `onnxruntime_tidl` (`TIDLCompilationProvider`)
and `tidl_tools` binaries, in x86 "PC emulation"/compile-only mode -- see
"Running a real compile" below for how that became possible and what it
does and doesn't confirm. Keep both: the static check runs everywhere with
no setup and no network dependency, the real one is the actual ground
truth when it's available.

Everything the static heuristic checks for is verified directly against
edgeai-tidl-tools' own published docs -- `docs/operators.md` ("Supported
Operators") and `docs/vision_transformers.md` ("Vision Transformers") --
fetched from `raw.githubusercontent.com` (see "Reaching edgeai-tidl-tools
from here" below) rather than reconstructed from memory. That distinction
mattered in practice: an earlier version of this harness's `legalize.py`
had a GELU rule backwards until the actual doc was checked (see that
section below).

1. **No dynamic shapes.** Every graph input must have a fully static shape
   (including batch size) for TIDL to compile it at all.
2. **Some ops have no accelerator equivalent on any hardware of this
   class**: control flow (`If`/`Loop`/`Scan`), the `Sequence`/`Optional`
   container ops, and string tensors -- the same generic complement
   `scripts/axera/pulsar2_ops.py` and the sibling QNN/OpenVINO/MIGraphX
   backends already use, not a TIDL-specific op list. `NonMaxSuppression` is
   also flagged: it has no entry in `docs/operators.md`'s supported-op
   table, and detection post-processing runs it on the host ARM core per
   `docs/od_meta_arch.md`.
3. **The decomposed LayerNorm chain should be fused, but the decomposed
   GELU chain should *not* be.** `docs/operators.md` lists
   `LayerNormalization` as its own directly supported layer
   (`TIDL_LayerNormLayer`), so `has_decomposed_normalization()` flags the
   hand-spelled `ReduceMean`/`Sub`/`Pow`/`Sqrt`/`Div` chain. GELU is the
   opposite: that same doc has *no* `Gelu` entry at all -- only
   `Erf`/`Identity`, "not supported as an individual operator... only
   supported as part of the fused combination of GELU"
   (`docs/vision_transformers.md`'s GELU section: the real importer
   pattern-matches the decomposed `Div`/`Erf`/`Add`/`Mul`/`Mul` sequence
   itself and maps it to TIDL's internal BatchNorm-with-activation layer).
   So a literal ONNX `Gelu` node is the thing to flag/unfuse, not the
   decomposed form -- see "`legalize.py`" below.

What the static check does, per model:

- Compute the blocker set (op-type blockers + the static-shape check) before
  and after `onnxsim.simplify()`.
- Fail (`tidl_regression`) if simplification introduced a *new* blocking op
  type that wasn't already present in the original graph -- the concrete
  risk this harness exists to catch: a simplification could fold something
  into a form TIDL's partitioner then refuses, silently pushing more of the
  graph onto a CPU fallback (or off the accelerator path entirely).

What it deliberately does **not** claim: that a model with zero flagged
blockers actually compiles on a real TIDL toolchain, or that its per-op
attribute-level limits (e.g. supported `Resize` modes, `Conv` group/dilation
ranges) are satisfied -- it only checks op *type* and shape-staticness.
`real_compile.py` (below) is what actually confirms the rest, when it's
available.

## Reaching edgeai-tidl-tools from here

Three different hosts, three different (and, for two of them, since-changed)
answers:

- **`raw.githubusercontent.com` is reachable.** `docs/operators.md`,
  `docs/vision_transformers.md`, the top-level `README.md`, `docs/
  model_compilation.md`, `scripts/setup/setup.sh`, and
  `runtimes/examples/python/basic_example/config.yaml` were all fetched
  directly from there (`master` branch) and used to write and correct
  this harness. The interactive `github.com` repo page and
  `api.github.com` both 403 (looks like GitHub's normal anti-automation
  response to a plain unauthenticated request, not something specific to
  this repository), but the raw file server does not.
- **`software-dl.ti.com` and `downloads.ti.com` are now both reachable.**
  `scripts/setup/setup.sh` downloads everything real from
  `software-dl.ti.com` -- the TIDL-patched `onnxruntime_tidl` wheel,
  `tidl_tools` itself, the TFLite/TVM runtime wheels, even the out-of-box
  example data -- but every one of those links 302-redirects to a
  *second* host, `downloads.ti.com`, that actually serves the file. Both
  were confirmed 403 (at the CONNECT level) at an earlier point in this
  project's history; both are unblocked now, which is what made
  `real_compile.py` (below) possible. If they are blocked again,
  `find_tidl_python()`/`tidl_tools_path()` return `None` and
  `tests/test_edgeai_tidl_real_compile.py` skips cleanly rather than
  failing -- the static heuristic keeps working regardless either way.

One thing worth citing directly rather than the general "onnxsim is used
as post-export cleanup" framing this repo's top-level README already
gives other projects: edgeai-tidl-tools' own `docs/vision_transformers.md`
DeiT walkthrough runs `onnxsim` as one of its own documented steps --
`pip install timm onnx onnxsim` then `!onnxsim deit_tiny.onnx
deit_tiny_1.onnx`, right before the resulting model is handed to TIDL.

## Running a real compile

`real_compile.py` wraps TI's actual `onnxruntime_tidl` (`TIDLCompilationProvider`)
and `tidl_tools` binaries -- see that module's docstring for the two things
this does and doesn't confirm (a real compile/import stage, not on-device
inference; genuinely real, not a mock). To set it up locally:

```bash
# 1. Download the real tidl_tools + onnxruntime_tidl wheel (needs network
#    access to software-dl.ti.com and downloads.ti.com).
python3 scripts/edgeai/real_compile.py setup /tmp/tidl --soc J721S2

# 2. onnxruntime_tidl ships as a cp310-only wheel, so it needs its own
#    Python 3.10 venv, pinned to numpy<2 (the wheel predates NumPy 2's ABI).
python3.10 -m venv /tmp/tidl_venv
/tmp/tidl_venv/bin/pip install /tmp/tidl/onnxruntime_tidl-*.whl "numpy<2"

# 3. Point the real-compile tests at both.
export TIDL_PYTHON=/tmp/tidl_venv/bin/python
export TIDL_TOOLS_PATH=/tmp/tidl/tidl_tools
pytest -v tests/test_edgeai_tidl_real_compile.py
```

`--soc` can be any device from the top-level README's supported-devices
table (normalized per `scripts/setup/setup_env.sh`, e.g. `AM68A`/`TDA4VL`
both map to `J721S2`) -- the compile stage needs no device, so which one
you pick only changes which C7x firmware profile it compiles against.
`.github/workflows/edgeai-integration.yml`'s `tidl-real-compile` job
automates exactly these steps in CI (not yet verified there directly --
only reproduced in a development sandbox with the same network access).

## `legalize.py`: acting on what the heuristic flags

A static check can flag a graph; it can't fix it. `legalize.py` holds
semantics-preserving rewrites that steer a graph toward what the real
importer wants -- and, per the point above, that is not always "fuse
everything":

- `fuse_decomposed_layernorm` -- the hand-written `ReduceMean`/`Sub`/
  `Pow(2)`/`ReduceMean`/`Add`/`Sqrt`/`Div` chain `has_decomposed_normalization()`
  flags, replaced with a single `LayerNormalization` node (folding a
  trailing `Mul(scale)`/`Add(bias)` pair into its scale/bias inputs when
  present) -- fusing *toward* the op `docs/operators.md` lists as directly
  supported.
- `unfuse_gelu_to_erf` -- the reverse direction: a literal `Gelu` node
  (`approximate="none"`, opset 20+) expanded back into
  `Div`/`Erf`/`Add`/`Mul`/`Mul`, since `docs/operators.md` has no `Gelu`
  entry at all and the real importer only recognizes the decomposed form
  (see the note above). An earlier version of this module had a
  `fuse_erf_gelu` rule that went the *other* way -- fusing toward a
  `Gelu` node -- based on a wrong assumption that GELU worked the same way
  as LayerNorm. It didn't; this is the correction, made after actually
  reading `docs/operators.md` instead of assuming.

Both are exact, not approximate, and only fire on the specific node
wiring real exporters produce -- see each rule's own docstring for exactly
what is matched and what is conservatively left alone. Run standalone as
`legalize.py in.onnx out.onnx`, or call `legalize(model)` directly; see
`../../tests/test_edgeai_legalize.py` for the correctness checks (each
rewrite is compared against `onnx.reference.ReferenceEvaluator` on the
original graph).

## Quantization: what TIDL supports, and using onnxsim's own quantizer

Surveyed directly from `docs/quantization.md`, `docs/quantization_proto.md`,
and `docs/model_compilation.md`'s "Quantization Specific Options" table
(all fetched the same way as the operator docs above).

**Precision TIDL's C7x-MMA actually runs in: 8-bit and 16-bit fixed-point,
and mixtures of the two -- nothing lower.**

- **8-bit is the default and "recommended for optimal performance"** on
  every supported SoC.
- **16-bit** is available "for cases requiring higher precision", at a
  performance cost.
- **Mixed precision** (per-layer 8-bit *or* 16-bit, not both at once on
  the same tensor) is TIDL's main accuracy/performance dial -- manual
  (`advanced_options:output_feature_16bit_names_list`/
  `params_16bit_names_list`) or automatic
  (`advanced_options:mixed_precision_factor`, framed as a latency budget:
  `T_mixed_precision / T_8bit`).
- **32-bit is floating point and explicitly "not supported for target
  execution"** (`model_compilation.md`'s `tensor_bits` option) -- it's the
  CPU-fallback path, not something the accelerator runs.
- No 4-bit (or lower) mode is documented anywhere in these three files.

Other real constraints worth knowing before assuming "any QDQ model
works": **asymmetric quantization is unsupported on one SoC**
(`J721E`/`TDA4VM` -- symmetric-only there; every other SoC supports both);
weights are quantized **per-channel** and activations **per-tensor**
(`docs/quantization.md`'s per-layer table, e.g.
`TIDL_ConvolutionLayer`: "Weights: Symmetric, Per-channel" /
"Activations: Asymmetric, Per-tensor" -- the exact scheme
`quantize_for_tidl.py` below matches); and QAT (embedding ranges into the
model at training time, near-zero accuracy drop) is documented as
**`J721E`/`TDA4VM`-specific**, via a separate project
([edgeai-modeloptimization](https://github.com/TexasInstruments/edgeai-tensorlab/blob/main/edgeai-modeloptimization/torchmodelopt/README.md#quantization)),
not this repo.

**`quantize_for_tidl.py` uses `onnxsim.calibration.quantize_static`/
`quantize_static_int16`** (already in this package) as exactly the "use
your own quantization algorithm" case `docs/quantization.md`'s
pre-quantized-models section names as a first-class use case for feeding
TIDL an ONNX QDQ model directly
(`advanced_options:prequantized_model=1`). Structurally, the match to
`docs/quantization.md`'s per-layer table is exact --
`check_tidl_qdq_scheme()` checks this on any model.

**But actually compiling that QDQ output through the real x86 PC
compiler segfaults it**, confirmed by trying rather than assumed working
just because the structure matches: `advanced_options:prequantized_model=1`
crashes `tidl_tools` release `11_02_20_00` on both a `Conv`-only and a
separate `MatMul`-only model, always at the same point
(`TIDL_runtimesOptimizeNet`, right after optimization for the subgraph
starts). The same exact models compile fine through TIDL's *own*
calibration on the plain float model. `tests/test_edgeai_tidl_real_compile.py::
test_prequantized_qdq_import_still_crashes` reproduces this as a
regression guard (documenting the bug, not working around it) -- see
`quantize_for_tidl.py`'s docstring for the full record, including a
parallel attempt at the third documented path (`quant_params_proto_path`
"write mode") that didn't produce output in testing either, so isn't
wired up.

**So, for now**: use `quantize_for_tidl.py` to produce and structurally
validate a QDQ model (for inspection, another QDQ-aware runtime, or a
future/patched `tidl_tools` release), but compile via TIDL's own
calibration on the *float* `onnxsim.simplify()` output for an actual
compile today -- plain `tensor_bits: 8`/`16` provider options, per
`real_compile.py`'s already-confirmed-working path.

## Files

- `tidl_ops.py` -- the op-type blocker lists (control flow, Sequence/
  Optional, data-dependent-shape ops, host-only ops, QOperator-format
  quantized ops), the static-shape check, and the decomposed-LayerNorm
  signature check, plus the functions that walk a `ModelProto` (including
  subgraphs) to apply them.
- `tidl_backend.py` -- the small `coverage()`/`blockers()`/
  `new_blocking_op_types()`/`dynamic_shape_risks()`/`normalization_risks()`
  API `worker.py` and the tests use, kept separate from `tidl_ops.py` for
  the same interface-symmetry reason `scripts/axera/pulsar2_backend.py` is
  split from `pulsar2_ops.py`.
- `models.py` -- re-exports `scripts/common/synthetic_models.py`'s shared
  suite and adds three fixtures: `edgeai_dynamic_batch_leaf` (a symbolic
  batch dimension, since none of the shared suite's models are
  dynamic-shaped), `mobilenet_block` (MobileNetV2's inverted-residual
  bottleneck -- verified as the real backbone in edgeai-tidl-tools' own
  object-detection/segmentation example configs, not "the quickstart
  model" as an earlier version of this comment claimed without checking),
  and `vision_transformer_block` (a pre-LN ViT encoder block using the
  fused `LayerNormalization` op but the *decomposed* GELU sequence -- see
  the note on GELU above).
- `worker.py` -- checks one model in its own subprocess; see its docstring
  for the exact steps and status values.
- `run_tidl_compat.py` -- drives `worker.py` over the whole suite (or a
  `--models` subset) and writes a CSV report.
- `legalize.py` -- the fusion rewrites described above.
- `real_compile.py` -- the real `onnxruntime_tidl`/`tidl_tools` compile
  wrapper described above ("Running a real compile").
- `quantize_for_tidl.py` -- the QDQ quantization flow and scheme checker
  described above ("Quantization").
- `../../tests/test_edgeai_tidl_compat.py` -- the pytest suite CI runs.
- `../../tests/test_edgeai_legalize.py` -- correctness checks for
  `legalize.py`'s rewrites.
- `../../tests/test_edgeai_quantize_for_tidl.py` -- structural checks for
  `quantize_for_tidl.py`'s QDQ output and the QOperator-format blocker.
- `../../tests/test_edgeai_tidl_real_compile.py` -- the real-compile
  regression check (including the confirmed prequantized-import crash),
  skip-guarded on `TIDL_PYTHON`/`TIDL_TOOLS_PATH`.
