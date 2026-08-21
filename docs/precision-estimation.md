# Quantization precision estimation (`estimate_quantization_precision`)

## What this is

`onnxsim.estimate_quantization_precision` is a pure-Python, read-only
analysis (`onnxsim/precision_estimator.py`) that answers a narrower version of
"is INT8 safe here?" than actually running the model: given only a node's
**constant weight values** and its **shape hyperparameters** (the reduction
depth for MatMul/Gemm, `Cin/groups * kernel-volume` for Conv, `num_heads` /
`head_dim` for Attention), it estimates whether the INT8 scheme
`onnxsim.quantize_dynamic`/`quantize_static` apply (see
[dynamic-quantization.md](dynamic-quantization.md)) is numerically safe and
well-resolved for that node — without executing the model or needing
calibration data.

It never modifies the model, has no C++/pybind component, and does not
require building onnxsim's C++ extension.

```python
import onnx
import onnxsim

model = onnx.load("model.onnx")
for est in onnxsim.estimate_quantization_precision(model):
    print(est)
```

## The three things it checks

1. **Accumulator overflow** (MatMul/Gemm/Conv). onnxsim's INT8 weight
   quantization is symmetric and per-channel, scaled so a channel's
   largest-magnitude element always quantizes to exactly ±127 — the
   worst-case *quantized* value is fixed by the scheme, not by the actual
   weight data. Paired with a uint8 activation (the full range
   `DynamicQuantizeLinear`/`QuantizeLinear` can produce), the worst-case
   accumulated value is bounded by `reduction_depth * 127 * 255`; past
   `INT32_MAX` an int32 accumulator can wrap around. This is an exact bound —
   `int32_accumulator_safe` — and is the same check
   `onnxsim.quantize_dynamic` itself enforces before quantizing a node (see
   [dynamic-quantization.md](dynamic-quantization.md#accumulator-overflow-guard)).

2. **Effective resolution / outlier risk** (MatMul/Gemm/Conv). *Within* the
   safe range, actual weight data matters: since a channel's scale is set by
   its single largest-magnitude element, a channel with a few extreme
   outliers wastes most of its 8 bits on values the bulk of the channel's
   weights never approach. `max_outlier_ratio` is each channel's
   `max(|w|) / median(|w|)` (excluding exact zeros, so pruned/sparse channels
   aren't misread as outlier-dominated); `outlier_risk` is set once that
   ratio passes 127 — an 8-bit symmetric quantizer has 127 positive levels,
   so past that ratio the channel's *typical* weight rounds to within one
   quantization step of zero.

3. **float32-cast exactness** (MatMul/Gemm/Conv) — a distinct, much smaller
   effect from (1), not a correctness bound. Even a node that clears the
   int32-overflow check still has its accumulator go through
   `Cast<float>(Acc)` before dequantization; float32's 24-bit mantissa
   represents integers exactly only up to 2²⁴, so past about 518 reduction
   terms (a much lower bar than the ~66,311 int32 bound) the cast rounds to
   the nearest representable float32. `float32_cast_exact` reports this. It's
   ordinary floating-point rounding at ~2⁻²⁴ relative — orders of magnitude
   below INT8's own ~1/127 (~0.8%) quantization error — so it does not change
   any recommendation; it exists only so "int32-safe" is never read as "exact
   end-to-end".

**Attention** has no constant weight in the MatMul sense — Q/K/V are runtime
activations, so none of the above applies. Instead it reports an
advisory-only check: the pre-softmax `Q·Kᵀ` dot product's magnitude grows
with `head_dim`, which is exactly why attention scales scores by
`1 / sqrt(head_dim)` (Vaswani et al., 2017). The estimate compares that
canonical default against the node's actual `scale` attribute (or ai.onnx
opset 23 Attention's own default when absent); a mismatch means pre-softmax
logits grow unnormalized with `head_dim`, risking saturation/overflow in a
low-precision (e.g. fp16) softmax.

## What it returns

One dataclass per recognized node, in graph order:

| Op type | Estimate type | Key fields |
| --- | --- | --- |
| `MatMul`, `Gemm` | `MatMulGemmPrecisionEstimate` | `reduction_depth`, `num_channels`, `int32_accumulator_safe`, `float32_cast_exact`, `max_outlier_ratio`, `outlier_risk`, `recommendation` |
| `Conv` | `ConvPrecisionEstimate` | same shape as above |
| `Attention` | `AttentionPrecisionEstimate` | `num_query_heads`, `num_kv_heads`, `head_dim`, `default_scale`, `actual_scale`, `scale_matches_default`, `recommendation` |

Every estimate carries a `recommendation` string summarizing the verdict in
prose, e.g.:

```
int32-safe, but a channel's outliers dominate its scale: per-group
quantization or INT16 would preserve more resolution for this node's
typical-magnitude weights
```

## Scope and limitations

- Only the top-level graph is walked — nodes inside `If`/`Loop`/`Scan`
  subgraphs are not visited.
- A weight must be a top-level graph initializer; a weight produced by a
  `Constant` node, or coming from an external-data reference this process
  can't resolve, is skipped (no estimate is returned for that node).
- Attention shapes are resolved via `onnx.shape_inference`; when a dimension
  is symbolic/dynamic and no `q_num_heads`/`kv_num_heads` attribute supplies
  it, `head_dim` is `None` and the estimate says so rather than guessing.
- This module makes no graph-rewriting decisions on Conv or Attention — no
  such quantization pass exists in onnxsim today (`quantize_dynamic` only
  covers MatMul/Gemm; see [dynamic-quantization.md](dynamic-quantization.md)).
  Its estimates for those op types are informational.
