# Apple Core ML integration check

Verifies that `onnxsim`'s output still works with **Core ML** — the runtime
behind `coremltools` model deployment on macOS/iOS. The goal is to catch the
failure mode the unit tests and the large-model regression don't: a
simplification that produces a graph Core ML can no longer **compile**, or
that **changes the result** on Apple's stack.

It uses the [`CoreMLExecutionProvider`](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)
built into the standard `onnxruntime` PyPI wheel — no extra package, but it
only exists on the **macOS** build (`get_available_providers()` omits it on
Linux/Windows). So the whole check runs on a stock macOS GitHub-hosted
runner with nothing but `pip install onnxruntime`.

## What it checks

For each model the harness runs **original vs. simplified through the same
Core ML backend**, so backend quirks cancel and only an onnxsim-introduced
change can fail the run:

1. `simplify` the model with onnxsim.
2. Compile + run the **original** graph on the Core ML EP.
   If that already fails, the backend just doesn't support the graph →
   reported as `unsupported`, **not** a failure.
3. Compile + run the **simplified** graph on the Core ML EP.
   If the original compiled but the simplified doesn't → `coreml_regression`
   (a failure): simplification broke Core ML compatibility.
4. Compare the two Core ML outputs. Divergence beyond tolerance →
   `coreml_regression`: simplification changed the on-device result.
5. Record the ONNX Runtime CPU-reference diff and the Core ML **coverage**
   (does the whole graph map onto Core ML, or do some nodes fall back to
   ORT's CPU provider) as information.

Partial coverage and `unsupported` are reported, never failed — plenty of
valid graphs are not 100% Core ML-mappable, and that is a backend property,
not an onnxsim bug.

## Files

| file | purpose |
| --- | --- |
| `coreml_backend.py` | wraps the Core ML EP: builds/runs a model on Core ML and on the ORT CPU reference, measures coverage. Degrades gracefully (`COREML_AVAILABLE`) when the EP is absent (non-macOS). |
| `models.py` | alias for `scripts/common/synthetic_models.py`, the small synthetic-graph suite shared with the other EP harnesses. |
| `worker.py` | runs the check for one model in an isolated subprocess, printing one `__RESULT__<json>` line. |
| `run_coreml_compat.py` | drives the suite, writes a CSV, and exits non-zero on any regression. Entry point for CI. |

## Running locally

Requires macOS (the platform Core ML itself runs on).

```bash
pip install onnxruntime      # the macOS wheel bundles the Core ML EP
pip install .                # or install an onnxsim wheel

python scripts/apple/run_coreml_compat.py --output coreml-compat.csv
```

The in-tree smoke test `tests/test_coreml_compat.py` reuses this harness and
is skipped automatically when the Core ML EP isn't available (e.g. running on
Linux/Windows).

## Fidelity tiers (what this does and doesn't cover)

This check runs the **real** Core ML compiler and runtime — there is no
emulation step the way QNN's HTP backend needs one, since the check already
runs on real Apple hardware (the macOS CI runner itself). What it leaves
uncovered:

- **Compute unit selection.** `MLComputeUnits=ALL` (the default here) lets
  Core ML place ops on CPU, GPU, or the Neural Engine as it judges best. Set
  `COREML_COMPUTE_UNITS=CPUOnly` (or `CPUAndGPU`, `CPUAndNeuralEngine`) to
  pin a specific target if you need to isolate one.
- **iOS-specific behavior.** This runs the macOS Core ML stack; iOS devices
  share the same compiler but can differ in available ops per OS version.

## Extending

`models.py` is intentionally small and self-contained so the CI job needs no
downloads. Real models can be layered on by passing an on-disk path as
`worker.py`'s second argument, the same way `scripts/qualcomm` and
`scripts/regression` do.
