# TI edgeai / TIDL static compatibility check

Verifies that `onnxsim`'s output stays friendly to **TIDL** (TI Deep
Learning), the inference engine TI's [edgeai](https://github.com/TexasInstruments/edgeai)
SDK (`edgeai-modeloptimization`, `edgeai-tensorlab`, `edgeai-tidl-tools`, ...)
compiles ONNX models for, targeting the C7x-MMA deep-learning accelerator on
Jacinto/Sitara SoCs (TDA4x, AM62A/68A, ...).

## This one is not like `scripts/qualcomm`/`scripts/intel`/`scripts/amd`

Those wrap a **real** compiler via a pip-installable ONNX Runtime execution
provider, so they measure actual compile/run behavior. TIDL, like Axera's
Pulsar2 (`scripts/axera`), has no PyPI package and no plain-pip execution
provider -- the TIDL-enabled `onnxruntime` build ships as part of TI's own
PSDK/edgeai-tidl-tools SDK and needs either the target device or a matching
x86 "PC emulation" build, neither of which this repository provisions.

So, **unlike `scripts/axera`, this check makes no hardware- or
compiler-confirmed claims at all** -- there is no equivalent here of a real
device, a real compiled artifact, or a real toolchain run to cite. It *is*,
however, checked directly against edgeai-tidl-tools' own published docs --
`docs/operators.md` ("Supported Operators") and `docs/vision_transformers.md`
("Vision Transformers") -- fetched from `raw.githubusercontent.com` (see
"Reaching edgeai-tidl-tools from here" below) rather than reconstructed from
memory. That distinction mattered in practice: an earlier version of this
harness's `legalize.py` had a GELU rule backwards until the actual doc was
checked (see that section).

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

What this check does, per model:

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
ranges) are satisfied -- this harness only checks op *type* and
shape-staticness. If a runner with the real SDK (or a device) is ever
provisioned, replace this with an actual model-import/compile check, the way
`scripts/qualcomm`/`scripts/intel`/`scripts/amd` wrap a real execution
provider.

## Reaching edgeai-tidl-tools from here

Two different hosts, two different answers:

- **`raw.githubusercontent.com` is reachable.** `docs/operators.md`,
  `docs/vision_transformers.md`, the top-level `README.md`, `docs/
  model_compilation.md`, `scripts/setup/setup.sh`, and
  `runtimes/examples/python/basic_example/config.yaml` were all fetched
  directly from there (`master` branch) and used to write and correct
  this harness. The interactive `github.com` repo page and
  `api.github.com` both 403 (looks like GitHub's normal anti-automation
  response to a plain unauthenticated request, not something specific to
  this repository), but the raw file server does not.
- **`software-dl.ti.com` is not.** `scripts/setup/setup.sh` downloads
  *everything* real from there -- the TIDL-patched `onnxruntime_tidl`
  wheel, `tidl_tools` itself, the TFLite/TVM runtime wheels, even the
  out-of-box example data -- and this repository's own network policy
  403s that host at the CONNECT level. So there is currently no way to
  replace this static heuristic with a real compile/import check from
  here, even though the documentation describing what that check should
  look for is readable. If a runner with the real SDK is ever
  provisioned, wire it in as a `workflow_dispatch`-only job (like
  `axera-integration.yml`'s `pulsar2-docker-convert`), which stays
  dormant until such a runner exists.

One thing worth citing directly rather than the general "onnxsim is used
as post-export cleanup" framing this repo's top-level README already
gives other projects: edgeai-tidl-tools' own `docs/vision_transformers.md`
DeiT walkthrough runs `onnxsim` as one of its own documented steps --
`pip install timm onnx onnxsim` then `!onnxsim deit_tiny.onnx
deit_tiny_1.onnx`, right before the resulting model is handed to TIDL.

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

## Files

- `tidl_ops.py` -- the op-type blocker lists (control flow, Sequence/
  Optional, data-dependent-shape ops, host-only ops), the static-shape
  check, and the decomposed-LayerNorm signature check, plus the functions
  that walk a `ModelProto` (including subgraphs) to apply them.
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
- `../../tests/test_edgeai_tidl_compat.py` -- the pytest suite CI runs.
- `../../tests/test_edgeai_legalize.py` -- correctness checks for
  `legalize.py`'s rewrites.
