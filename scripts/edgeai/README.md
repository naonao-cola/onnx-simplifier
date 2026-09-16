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

So, **unlike `scripts/axera`, this check makes no hardware-confirmed
claims at all** -- there is no equivalent here of a real device, a real
compiled artifact, or a real toolchain run to cite. Everything in
`tidl_ops.py`/`tidl_backend.py` is a static heuristic built from two things
TIDL's own published documentation states plainly:

1. **No dynamic shapes.** Every graph input must have a fully static shape
   (including batch size) for TIDL to compile it at all.
2. **Some ops have no accelerator equivalent on any hardware of this
   class**: control flow (`If`/`Loop`/`Scan`), the `Sequence`/`Optional`
   container ops, and string tensors -- the same generic complement
   `scripts/axera/pulsar2_ops.py` and the sibling QNN/OpenVINO/MIGraphX
   backends already use, not a TIDL-specific op list. `NonMaxSuppression` is
   also flagged: edgeai-tidl-tools' own detection-model documentation
   describes NMS running as host (ARM-core) post-processing, not an
   in-graph accelerator op.
3. **Transformer blocks should use the fused `LayerNormalization`/`Gelu`
   ops, not their decomposed equivalents.** edgeai-tidl-tools'
   transformer-support notes call this out; `has_decomposed_normalization()`
   flags the classic hand-spelled LayerNorm signature
   (`ReduceMean`/`Sub`/`Pow`/`Sqrt`/`Div`) so a graph exported in that form
   gets surfaced rather than silently offloading worse than it needs to.

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
  bottleneck -- the structure of edgeai-tidl-tools' own quickstart example
  model), and `vision_transformer_block` (a pre-LN ViT encoder block built
  from the doc-preferred fused `LayerNormalization`/`Gelu` ops).
- `worker.py` -- checks one model in its own subprocess; see its docstring
  for the exact steps and status values.
- `run_tidl_compat.py` -- drives `worker.py` over the whole suite (or a
  `--models` subset) and writes a CSV report.
- `../../tests/test_edgeai_tidl_compat.py` -- the pytest suite CI runs.
