# tinygrad Hexagon / Metal integration check

Verifies that `onnxsim`'s output still runs through **tinygrad's device
backends** -- the `DSP` (Qualcomm Hexagon, `tinygrad/runtime/ops_dsp.py`) and
`METAL` (Apple GPU, `tinygrad/runtime/ops_metal.py`) devices -- the same way
`scripts/apple` (Core ML) and `scripts/qualcomm` (QNN) verify their own
execution providers. The goal is to catch the failure mode the unit tests and
the large-model regression don't: a simplification that produces a graph
tinygrad can no longer **import/run on the device**, or that **changes the
result** on that device.

It uses tinygrad's own eager `OnnxRunner` (`tinygrad.nn.onnx`), which imports
and executes an ONNX `ModelProto` directly -- no separate compile step, no
vendor SDK, no phone. Two execution tiers:

* **Metal**: runs natively on macOS (the only platform tinygrad's Metal
  device exists on). No extra package, no special hardware beyond the Mac
  itself -- the direct analogue of the Core ML check.
* **Hexagon DSP**: runs on any host via `MOCKDSP=1` -- tinygrad compiles the
  generated Hexagon code with a Hexagon-capable clang and executes it under
  `qemu-hexagon`. Needs `qemu-hexagon(-static)` on `PATH` plus an LLVM clang
  with the Hexagon target and `ld.lld` (the same prerequisites as
  `tests/test_hexagon_tinygrad.py`'s codegen checks; CI installs
  `clang-19`/`lld-19`/`qemu-user-static`).

Sibling checks follow the same pattern: [`scripts/apple`](../apple) (Core
ML), [`scripts/qualcomm`](../qualcomm) (QNN), [`scripts/intel`](../intel)
(OpenVINO), [`scripts/amd`](../amd) (MIGraphX). This one differs in one
respect: those wrap an ONNX Runtime execution provider, while this drives
tinygrad's `OnnxRunner` directly, since Hexagon has no ORT EP and Metal needs
none.

## What it checks

For each model the harness runs **original vs. simplified through the same
tinygrad device**, so backend quirks cancel and only an onnxsim-introduced
change can fail the run:

1. `simplify` the model with onnxsim.
2. Run the **original** graph on the device.
   If that already fails, the backend just doesn't support the graph →
   reported as `unsupported`, **not** a failure.
3. Run the **simplified** graph on the device.
   If the original ran but the simplified doesn't → `<dev>_regression`
   (a failure): simplification broke device compatibility.
4. Compare the two on-device outputs. Divergence beyond tolerance →
   `<dev>_regression`: simplification changed the on-device result.
5. Record the tinygrad-default-device reference diff as information.

`unsupported` is reported, never failed -- plenty of valid graphs are not
runnable through `OnnxRunner` on a given device, and that is a backend
property, not an onnxsim bug.

## Files

| file | purpose |
| --- | --- |
| `tinygrad_hexagon_backend.py` | wraps tinygrad's devices: probes availability (a real canary run, not just device listing), runs a model on DSP / Metal / the default-device reference, re-exports `compare`/`random_feeds` from `scripts/common/ep_numerics.py`. Degrades gracefully (`HEXAGON_AVAILABLE` / `METAL_AVAILABLE`) when a device is absent. |
| `models.py` | alias for `scripts/common/synthetic_models.py`, the small synthetic-graph suite shared with the other EP harnesses. |
| `worker.py` | runs the check for one model in an isolated subprocess, printing one `__RESULT__<json>` line. |
| `run_hexagon_compat.py` | drives the suite, writes a CSV, and exits non-zero on any regression. Entry point for CI. |

## Running locally

Metal (macOS only):

```bash
pip install tinygrad
pip install .                # or install an onnxsim wheel

python scripts/hexagon/run_hexagon_compat.py --device METAL --output metal-compat.csv
```

Hexagon DSP via the mock (any host, needs qemu + a Hexagon-capable clang):

```bash
pip install tinygrad
sudo apt-get install -y clang-19 lld-19 qemu-user-static   # Debian/Ubuntu

python scripts/hexagon/run_hexagon_compat.py --output hexagon-compat.csv
```

The in-tree smoke test `tests/test_tinygrad_hexagon_compat.py` reuses this
harness and is skipped automatically per device when that device isn't
available (no qemu/clang on the host, non-macOS for Metal, or no tinygrad
installed).

## Fidelity tiers (what this does and doesn't cover)

* **Mock DSP, not silicon.** `MOCKDSP=1` runs the real generated Hexagon code
  under `qemu-hexagon` -- genuine codegen + functional numerics, but an
  instruction-count proxy for timing, not cycle-accurate (see
  `scripts/android/tinygrad_hexagon_bridge/README.md`'s `HEXSIM=1` mode for
  the timing-grade alternative) and not a substitute for on-device
  validation. Set `MOCKDSP=0` (or unset `MOCKDSP`) with real FastRPC hardware
  to run on silicon instead; the harness never overrides an explicit setup.
* **One known tinygrad gap, worked around in the harness.** `OnnxRunner`
  parses plain numpy feeds onto tinygrad's *default* device even after
  `.to(device)` moves the graph's constants, so any graph with weights fails
  on-device with `all buffers must be on the same device`. The harness wraps
  feeds as `Tensor(..., device=device)` first -- the check validates
  onnxsim's output, not tinygrad's feed handling.
* **Real-device Hexagon transport** (FastRPC from Linux, TVM's Hexagon RPC,
  the `native_transport/` path) is covered by
  `scripts/android/tinygrad_hexagon_bridge/` and
  `tests/test_hexagon_tinygrad.py`, not here.

## Extending

`models.py` is intentionally small and self-contained so the CI job needs no
downloads. Real models can be layered on by passing an on-disk path as
`worker.py`'s second argument, the same way `scripts/qualcomm` and
`scripts/regression` do.
