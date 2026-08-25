# Measuring accuracy drop (`measure_accuracy_drop`)

## What this is

`onnxsim.measure_accuracy_drop` (`onnxsim/accuracy.py`) runs a float model
and a quantized version of it on the same input data and reports how far
the quantized model's outputs actually drift from the float model's — an
**empirical measurement**, not an estimate. It complements
[precision-estimation.md](precision-estimation.md#whole-model-rollup-estimate_model_quantization_drop)'s
`estimate_model_quantization_drop`, which is fast and needs no data or
execution but is only a heuristic; this function is slower (it runs both
models, once per calibration sample) but gives ground truth for the
specific model and data you measure it on.

```python
import onnx
import onnxsim

model = onnx.load("model.onnx")
quantized = onnxsim.quantize_dynamic(model)  # or any quantize_*/quantize() call

report = onnxsim.measure_accuracy_drop(model, quantized)
print("worst relative L2 error:", report.worst_relative_l2)
for name, stats in report.per_output.items():
    print(name, stats.relative_l2, stats.cosine_similarity)
```

Both models are executed through `onnxsim.backend.run_model` — onnxruntime
when it's installed, the pure-Python reference evaluator otherwise (see
`onnxsim/backend.py`); no separate onnxruntime dependency is required to use
this function, though it is a much faster backend when present.

## What it reports

`AccuracyDropReport`:

| Field | Meaning |
| --- | --- |
| `num_samples` | how many calibration batches were run |
| `per_output` | `{output_name: OutputAccuracyStats}` |
| `worst_relative_l2` | max `relative_l2` over every output and sample |
| `worst_cosine_distance` | `1 - min(cosine_similarity)` over every output and sample |
| `all_finite` | `False` if the quantized model ever produced NaN/Inf where the float model didn't |

`OutputAccuracyStats` (one per graph output, **worst case across samples**,
not an average — the point of measuring accuracy drop is to know the worst
a deployment might see, not to average it away):

| Field | Meaning |
| --- | --- |
| `relative_l2` | `\|\|float - quantized\|\| / \|\|float\|\|` |
| `max_abs_error` | `max(\|float - quantized\|)` |
| `cosine_similarity` | `dot(float, quantized) / (\|\|float\|\| * \|\|quantized\|\|)` |

## Calibration / input data

Same convention as `quantize_static` and friends: pass `calibration_data`
(a list of `{input_name: np.ndarray}` dicts) for real, representative data —
see `onnxsim.load_huggingface_calibration_data` — or leave it unset to fall
back to `onnxsim.generate_random_calibration_data` (`num_samples`, `seed`).
Random data is a reasonable smoke test that the quantization pipeline
produces sane, finite output, but is a poor proxy for a real model's actual
accuracy drop on real inputs — use real calibration data before trusting
this measurement for a deployment decision.

## `keep_io_types=False` and non-float32 inputs

A `quantize_fp16`/`quantize_bf16`/`quantize_fp8` call made with
`keep_io_types=False` redeclares the quantized model's own graph inputs in
the narrow target format directly (see
[fp16-quantization.md](fp16-quantization.md)), so the same float32
calibration data used against the original model can't be fed to it
unmodified. `measure_accuracy_drop` auto-casts each calibration batch's
arrays to the quantized model's own declared input dtype before running it,
for float16 (numpy has a native dtype). bfloat16 and float8 have no native
numpy dtype (they need `ml_dtypes`), so those two are **not** auto-cast --
pre-cast `calibration_data` yourself with `ml_dtypes` before calling
`measure_accuracy_drop` on a `keep_io_types=False` bfloat16/float8 model, or
the run will fail with a type-mismatch error.

## Relationship to `estimate_model_quantization_drop`

|  | `estimate_model_quantization_drop` | `measure_accuracy_drop` |
| --- | --- | --- |
| Needs execution/data? | No | Yes |
| Speed | Fast (static analysis) | Slower (runs both models) |
| What it produces | A heuristic *estimate* from weights/shapes alone | An actual *measurement* on real output data |
| Best for | A quick pre-check before quantizing, or screening many candidate models | Verifying a specific quantized model's real-world accuracy before shipping it |

Use the static estimate first to catch outright-unsafe nodes (accumulator
overflow) cheaply, then measure the actual drop on the quantized model you
intend to ship, with real calibration data, before deploying it.
