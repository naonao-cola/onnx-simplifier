# Intel OpenVINO integration check

Verifies that `onnxsim`'s output still works with **OpenVINO** — Intel's
inference toolkit. The goal is to catch the failure mode the unit tests and
the large-model regression don't: a simplification that produces a graph
OpenVINO can no longer **compile**, or that **changes the result** on Intel's
stack.

It uses the pip-installable [`onnxruntime-openvino`](https://pypi.org/project/onnxruntime-openvino/)
wheel (a separate build of ONNX Runtime that bundles the OpenVINO EP — it
replaces, rather than supplements, plain `onnxruntime`). The default `CPU`
device target needs no discrete Intel hardware or driver, so the whole check
runs on an ordinary x86-64 CI runner with nothing but
`pip install onnxruntime-openvino`.

## What it checks

For each model the harness runs **original vs. simplified through the same
OpenVINO backend**, so backend quirks cancel and only an onnxsim-introduced
change can fail the run:

1. `simplify` the model with onnxsim.
2. Compile + run the **original** graph on the OpenVINO EP.
   If that already fails, the backend just doesn't support the graph →
   reported as `unsupported`, **not** a failure.
3. Compile + run the **simplified** graph on the OpenVINO EP.
   If the original compiled but the simplified doesn't → `openvino_regression`
   (a failure): simplification broke OpenVINO compatibility.
4. Compare the two OpenVINO outputs. Divergence beyond tolerance →
   `openvino_regression`: simplification changed the on-device result.
5. Record the ONNX Runtime CPU-reference diff and the OpenVINO **coverage**
   (does the whole graph map onto OpenVINO, or do some nodes fall back to
   ORT's CPU provider) as information.

Partial coverage and `unsupported` are reported, never failed — plenty of
valid graphs are not 100% OpenVINO-mappable, and that is a backend property,
not an onnxsim bug.

## Files

| file | purpose |
| --- | --- |
| `openvino_backend.py` | wraps the OpenVINO EP: builds/runs a model on OpenVINO (`CPU` device target by default) and on the plain ORT CPU reference, measures coverage. Degrades gracefully (`OPENVINO_AVAILABLE`) when the EP is absent. |
| `models.py` | alias for `scripts/common/synthetic_models.py`, the small synthetic-graph suite shared with the other EP harnesses. |
| `worker.py` | runs the check for one model in an isolated subprocess, printing one `__RESULT__<json>` line. |
| `run_openvino_compat.py` | drives the suite, writes a CSV, and exits non-zero on any regression. Entry point for CI. |

## Running locally

```bash
pip install onnxruntime-openvino   # NOT alongside plain onnxruntime
pip install .                      # or install an onnxsim wheel

python scripts/intel/run_openvino_compat.py --output openvino-compat.csv
```

The in-tree smoke test `tests/test_openvino_compat.py` reuses this harness
and is skipped automatically when `onnxruntime-openvino` isn't installed.

## Fidelity tiers (what this does and doesn't cover)

This check runs the real OpenVINO compiler/runtime targeting the `CPU`
device — real numerics, no emulation, but only one of OpenVINO's device
targets.

- **GPU / NPU targets.** Set `OPENVINO_DEVICE_TYPE=GPU` (or `NPU`) on a host
  with the matching Intel hardware and drivers to exercise those instead; the
  harness picks it up unchanged. CI here only covers `CPU` since that's the
  only target free-tier runners have.
- **Quantized / INT8 paths.** This suite is fp32 only; OpenVINO's NNCF
  quantization pipeline is a separate concern from graph compatibility.

## Extending

`models.py` is intentionally small and self-contained so the CI job needs no
downloads. Real models can be layered on by passing an on-disk path as
`worker.py`'s second argument, the same way `scripts/qualcomm` and
`scripts/regression` do.
