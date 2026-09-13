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
| `run_training_compat.py` | validates on-device *training* (the compiled step-graph loop behind `compile_training_loop` and every `step_providers=` loop) on each GPU provider the host offers (`ROCMExecutionProvider`, `MIGraphXExecutionProvider`): loss must fall, state must stay device-resident. Run it on ROCm hardware by hand; needs no CI wiring. |

## Installing the stack from scratch

A working MIGraphX EP is three layers, each with its own install. The
execution-provider wheel alone is never enough: onnxruntime ships the EP's
shared library inside the wheel, but that library loads the system MIGraphX
runtime at session-creation time, and *that* runtime needs the ROCm driver
underneath. Missing any layer reads as "unavailable", never as an error --
onnxruntime logs an `EP Error` and silently falls back to the CPU provider,
so always verify with a session build (the probe every harness here uses),
not just `get_available_providers()`.

1. **ROCm** (the driver + userspace). AMD's official path for Ubuntu is the
   `amdgpu-install` script from the driver download page for your distro and
   ROCm release (see [Install AMD ROCm](https://rocm.docs.amd.com/en/latest/deploy/linux/install.html)),
   then:

   ```bash
   sudo ./amdgpu-install --usecase=rocm
   sudo reboot
   ```

   For a GPU the release's prebuilt kernels don't cover -- e.g. Strix Halo
   (gfx1151) against the `onnxruntime-rocm` wheel's kernel set -- export
   `HSA_OVERRIDE_GFX_VERSION` before any run (see "Hardware notes" below).

2. **MIGraphX** (the graph compiler itself, system-wide). Once ROCm's apt
   repository is configured by the step above:

   ```bash
   sudo apt update && sudo apt install -y migraphx
   ```

   This is what provides `libmigraphx_c.so.3` -- the exact library onnxruntime
   looks for and the reason a bare `pip install onnxruntime-migraphx` alone
   falls back to CPU with `libmigraphx_c.so.3: cannot open shared object
   file`. A pip-installed MIGraphX (`python -m pip install --index-url
   https://stable.repo.amd.com/rocm/migraphx/whl-next/ migraphx==2.17.0+rocm10.0.0`)
   also works, but then the `migraphx_libs` directory must be on
   `LD_LIBRARY_PATH`.

3. **ONNX Runtime with the MIGraphX EP.** Either the monolithic wheel or, on
   newer ROCm stacks, the EP plugin:

   ```bash
   pip install onnxruntime-migraphx          # replaces plain onnxruntime
   ```

   (or pinned to a ROCm release:
   `pip3 install onnxruntime-migraphx -f https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/`).

   Newer ROCm stacks split the EP into a plugin wheel,
   [`onnxruntime-ep-migraphx`](https://pypi.org/project/onnxruntime-ep-migraphx/),
   which must be explicitly registered before querying providers:

   ```bash
   pip install "onnxruntime-ep-migraphx==1.0.0+rocm10.0.0"
   python -c "import migraphx, onnxruntime as ort, onnxruntime_ep_migraphx as m; \
   [ort.register_execution_provider_library(n,p) for n,p in zip(m.get_ep_names(), m.get_library_paths())]; \
   print(ort.get_available_providers())"
   ```

4. **Verify.** The wheel bundles the EP library regardless of hardware, so
   presence in `get_available_providers()` is not proof. A trivial session
   that *keeps* the provider is:

   ```bash
   python - <<'EOF'
   import onnxruntime as ort
   from onnx import parser
   m = parser.parse_model('<ir_version: 8, opset_import: ["": 17]>\n'
                          'agraph (float[1] x) => (float[1] y) { y = Identity(x) }')
   s = ort.InferenceSession(m.SerializeToString(), providers=['MIGraphXExecutionProvider'])
   print(s.get_providers())   # ['MIGraphXExecutionProvider'] on a working stack
   EOF
   ```

   If that prints `['CPUExecutionProvider']` instead, onnxruntime fell back:
   check for the `EP Error ... Failed to load library` line in the output and
   that `libmigraphx_c.so.3` is on the linker path. This is the same probe
   the in-tree tests and `run_training_compat.py` use to decide
   "unavailable" vs. "GPU present".

## Running locally

Requires a ROCm-capable AMD GPU with the stack above installed.

```bash
pip install onnxruntime-migraphx   # NOT alongside plain onnxruntime
pip install .                      # or install an onnxsim wheel

python scripts/amd/run_migraphx_compat.py --output migraphx-compat.csv
```

## Validating on-device training

`run_training_compat.py` exercises the other half of the ROCm story --
training, not inference. It runs `onnxsim.compile_training_loop`'s compiled
step graph (the same loop every `step_providers=` argument in `apply_qat`,
`apply_block_finetune`, `apply_adaround`/`apply_adaquant`/`apply_autoround`
and `compile_torch_training_loop` runs) on each GPU provider the host offers,
and reports per provider whether the loss falls, where the trained state
lives between steps, and whether `IOBinding` binds:

```bash
python scripts/amd/run_training_compat.py
python scripts/amd/run_training_compat.py --require-gpu --steps 200 --lr 0.05
```

A provider with no answering device is `skipped`, never failed; a training
run whose loss does not fall is `failed`.

## Hardware notes (Strix Halo / gfx1151)

Validated on an AMD Ryzen AI MAX+ 395 (Radeon 8060S, gfx1151) with
`onnxruntime-rocm`:

- The prebuilt wheel has no gfx1151 kernels: without anything else every
  device kernel fails with `hipErrorInvalidDeviceFunction`.
  `HSA_OVERRIDE_GFX_VERSION=11.0.0` (spoof as gfx1100) gets elementwise and
  session-management kernels running -- enough to prove the training step
  graph executes on-device -- but rocBLAS-backed ops (`Transpose`, `MatMul`)
  still fail inside rocBLAS (`HIPBLAS_STATUS_INTERNAL_ERROR`), so a full
  training loop cannot converge there. On gfx942/gfx950 or RDNA3 discrete
  GPUs the wheel has native kernels and no spoof is needed.
- `onnxruntime-migraphx` additionally needs the system `libmigraphx_c`
  (`apt install migraphx`); without it onnxruntime silently falls back to
  CPU. The validation script and the provider-gated tests
  (`tests/test_compile_training.py`, `tests/test_torch_training.py`) check
  `get_providers()` after the session build so that fallback reads as
  "unavailable", not as a passing GPU run.

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
