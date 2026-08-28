# AMD MIGraphX integration check

Verifies that `onnxsim`'s output still works with
[**MIGraphX**](https://github.com/ROCm/AMDMIGraphX) — AMD's ROCm graph
compiler for GPU inference (the AMD analog to NVIDIA's TensorRT). The goal is
to catch the failure mode the unit tests and the large-model regression
don't: a simplification that produces a graph MIGraphX can no longer
**compile**, or that **changes the result** on AMD's stack.

It uses the pip-installable [`onnxruntime-migraphx`](https://pypi.org/project/onnxruntime-migraphx/)
wheel (a separate build of ONNX Runtime that bundles the MIGraphX EP — it
replaces, rather than supplements, plain `onnxruntime`, the same relationship
`onnxruntime-openvino` has).

## This one needs real AMD GPU hardware

Unlike the Apple Core ML (`scripts/apple`) and Intel OpenVINO (`scripts/intel`)
checks, **MIGraphX has no CPU fallback or host emulator** — Core ML runs on
the Mac that's building it, OpenVINO's `CPU` device target needs no
accelerator, and even Qualcomm's QNN check (`scripts/qualcomm`) gets an x86
emulation path from the HTP backend. MIGraphX compiles and executes directly
on a ROCm-capable AMD GPU; there is no equivalent path on a stock CPU-only CI
runner. That's why this check, unlike its siblings, is **not** wired into a
scheduled or PR-triggered workflow: `amd-integration.yml` is
`workflow_dispatch`-only and targets a `[self-hosted, rocm]` runner label,
which this repository does not currently provision. Point it at your own
ROCm-equipped self-hosted runner to use it; until then it's dormant.

## What it checks

For each model the harness runs **original vs. simplified through the same
MIGraphX backend**, so backend quirks cancel and only an onnxsim-introduced
change can fail the run:

1. `simplify` the model with onnxsim.
2. Compile + run the **original** graph on the MIGraphX EP.
   If that already fails, the backend just doesn't support the graph →
   reported as `unsupported`, **not** a failure.
3. Compile + run the **simplified** graph on the MIGraphX EP.
   If the original compiled but the simplified doesn't → `migraphx_regression`
   (a failure): simplification broke MIGraphX compatibility.
4. Compare the two MIGraphX outputs. Divergence beyond tolerance →
   `migraphx_regression`: simplification changed the on-device result.
5. Record the ONNX Runtime CPU-reference diff and the MIGraphX **coverage**
   (does the whole graph map onto MIGraphX, or do some nodes fall back to
   ORT's CPU provider) as information.

Partial coverage and `unsupported` are reported, never failed — plenty of
valid graphs are not 100% MIGraphX-mappable, and that is a backend property,
not an onnxsim bug.

## Files

| file | purpose |
| --- | --- |
| `migraphx_backend.py` | wraps the MIGraphX EP: builds/runs a model on MIGraphX (fp16 by default) and on the plain ORT CPU reference, measures coverage. Because the provider's shared library is bundled into the wheel regardless of hardware, availability is checked by actually building a session on a trivial graph, not just by checking `get_available_providers()`. Degrades gracefully (`MIGRAPHX_AVAILABLE`) when no ROCm device answers. |
| `models.py` | alias for `scripts/common/synthetic_models.py`, the small synthetic-graph suite shared with the other EP harnesses. |
| `worker.py` | runs the check for one model in an isolated subprocess, printing one `__RESULT__<json>` line. |
| `run_migraphx_compat.py` | drives the suite, writes a CSV, and exits non-zero on any regression. Entry point for the (dormant) CI workflow. |

## Running locally

Requires a ROCm-capable AMD GPU with the ROCm stack installed.

```bash
pip install onnxruntime-migraphx   # NOT alongside plain onnxruntime
pip install .                      # or install an onnxsim wheel

python scripts/amd/run_migraphx_compat.py --output migraphx-compat.csv
```

The in-tree smoke test `tests/test_migraphx_compat.py` reuses this harness
and is skipped automatically when the MIGraphX EP isn't usable (no
`onnxruntime-migraphx`, or no ROCm device answers).

## Fidelity tiers (what this does and doesn't cover)

This check runs the real MIGraphX compiler/runtime — real numerics, no
emulation — but only what it's pointed at:

- **fp16 by default.** `MIGRAPHX_FP16=0` compiles/runs in fp32 instead;
  set `rtol`/`atol` in `common/ep_numerics.compare` accordingly if you tighten
  tolerances for fp32-only runs.
- **Single GPU, default device.** Multi-GPU selection isn't wired up; set
  `device_id` via `migraphx_backend._migraphx_provider_options()` if needed.
- **Quantized (int8) paths.** This suite is fp16/fp32 only.

## Extending

`models.py` is intentionally small and self-contained so the harness needs no
downloads. Real models can be layered on by passing an on-disk path as
`worker.py`'s second argument, the same way `scripts/qualcomm` and
`scripts/regression` do.
