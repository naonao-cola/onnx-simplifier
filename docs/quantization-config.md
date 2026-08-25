# Unified quantization config (`QuantizationConfig` / `quantize`)

## What this is

onnxsim ships more than a dozen `quantize_*` functions
(`quantize_dynamic`, `quantize_static`, `quantize_weight_only_int4`, ...),
each documenting and exposing its own scheme's specific parameters — see the
other docs in this directory for each one directly. `onnxsim.QuantizationConfig`
and `onnxsim.quantize` (`onnxsim/accuracy.py`) add a second, unified way to
reach all of them: describe *what* quantization you want as one typed
config object, and let `quantize` dispatch to the right underlying function.

This doesn't replace any existing `quantize_*` function — call those
directly when you already know which scheme you want. `quantize` is for code
that picks a scheme *programmatically*: a sweep over configs to compare
accuracy/size tradeoffs, a scheme read from a config file or CLI flag, or
any caller that wants one call site regardless of which scheme ends up
selected.

```python
import onnx
import onnxsim

model = onnx.load("model.onnx")
config = onnxsim.QuantizationConfig(scheme="weight_only", dtype="int4")
quantized = onnxsim.quantize(model, config)
onnx.save(quantized, "model.int4.onnx")
```

## The scheme / dtype / granularity matrix

| `scheme` | `dtype` | `granularity` | Calls |
| --- | --- | --- | --- |
| `"dynamic"` | `"int8"` | `"per_channel"` | `quantize_dynamic` |
| `"dynamic_fused"` | `"int8"` | `"per_channel"` | `quantize_dynamic_matmul_integer_to_float` |
| `"ternary"` | `"int8"` | `"per_channel"` | `quantize_ternary` |
| `"weight_only"` | `"int8"` | `"per_channel"` | `quantize_weight_only` |
| `"weight_only"` | `"int8"` | `"per_block"` | `quantize_weight_only_int8_block` |
| `"weight_only"` | `"int16"` | `"per_channel"` | `quantize_weight_only_int16` |
| `"weight_only"` | `"int4"` | `"per_block"` (only option) | `quantize_weight_only_int4` |
| `"static"` | `"int8"` | `"per_channel"` | `quantize_static` |
| `"static_int16"` | `"int16"` | `"per_channel"` | `quantize_static_int16` |
| `"qoperator"` | `"int8"` | `"per_channel"` | `quantize_qoperator` |
| `"float"` | `"float16"` | n/a | `quantize_fp16` |
| `"float"` | `"bfloat16"` | n/a | `quantize_bf16` |
| `"float"` | `"float8_e4m3"` | n/a | `quantize_fp8(format="e4m3")` |
| `"float"` | `"float8_e5m2"` | n/a | `quantize_fp8(format="e5m2")` |

`granularity` is only consulted for `scheme="weight_only", dtype="int8"` —
every other row has exactly one granularity onnxsim implements today, and
the field is ignored for those. Follow the linked docs above (or each
function's own docstring) for what each scheme actually does numerically;
this page only covers the dispatch layer.

An unknown `scheme`, a `dtype` not valid for that `scheme`, or an invalid
`granularity` for `weight_only`/`int8` all raise `ValueError` with a message
naming the valid options — `quantize` never silently guesses at a
misconfigured request.

## Other `QuantizationConfig` fields

- `calibration_data`, `num_calibration_samples`, `seed`, `providers`,
  `calibration_method` — forwarded as-is to `quantize_static`/
  `quantize_static_int16`/`quantize_qoperator` for the three calibration-based
  schemes; ignored otherwise. See
  [dynamic-quantization.md](dynamic-quantization.md) and
  [qoperator-quantization.md](qoperator-quantization.md) for what calibration
  data is and how to supply real (not random) data via
  `onnxsim.load_huggingface_calibration_data`.
- `keep_io_types` — forwarded to `quantize_fp16`/`quantize_bf16`/`quantize_fp8`
  for `scheme="float"`; ignored otherwise. See
  [fp16-quantization.md](fp16-quantization.md).

## Measuring what a config actually costs

Picking a `QuantizationConfig` is only half the question — how much accuracy
it costs is the other half. See
[precision-estimation.md](precision-estimation.md#whole-model-rollup-estimate_model_quantization_drop)
for a fast, data-free estimate, and [accuracy-drop.md](accuracy-drop.md) for
an actual, data-driven measurement of a specific quantized model.
