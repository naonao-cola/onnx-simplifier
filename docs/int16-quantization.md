# Weight-only INT16 quantization (`quantize_weight_only_int16`)

## What this is

`onnxsim.quantize_weight_only_int16` is a pair of single, self-contained C++
graph rewrites (`onnxsim/passes/weight_only_quantize_int16_matmul.h`,
`onnxsim/passes/weight_only_quantize_int16_conv.h`) that quantize every
`MatMul`, every "vanilla" `Gemm` (`transA=0`, `alpha=1`, `beta=1`), and every
`Conv`, whose weight is a constant float32 tensor -- the exact same
whole-weight, per-output-channel, symmetric scheme `quantize_weight_only`
uses, just to INT16 instead of INT8:

```
Before:
  Y = MatMul(X, W)                                   # W constant, [K, N], float32

After:
  Wq  = <int16, per-column symmetric>                # computed once, here
  Ws  = <float32, one scale per column of W>
  Wdq = DequantizeLinear(Wq, Ws, axis=1)              # float32
  Y   = MatMul(X, Wdq)                                # X is exactly as it was
```

Like `quantize_weight_only`, the **activation is never touched** (no
calibration data, no runtime quantize/dequantize cost on the activation
path), and a `Gemm`/`Conv` bias, if present, is left in float and untouched.

## Why this exists alongside `quantize_weight_only`'s INT8

INT16's scale is `max(|w|) / 32767` per channel -- about 8x finer than INT8's
`max(|w|) / 127`. That extra resolution matters for a specific, identifiable
failure mode of INT8: a channel whose scale is set by a single
largest-magnitude outlier wastes most of INT8's 8 bits on values the bulk of
the channel's weights never approach. Once a channel's
`max(|w|) / median(|w|)` ratio exceeds 127, its *typical* (median-magnitude)
weight rounds to within one INT8 quantization step of zero -- effectively
lost.

This is not a hypothetical: `onnxsim.estimate_quantization_precision` (see
`precision_estimator.py`) computes exactly this ratio for every
MatMul/Gemm/Conv weight and, once it crosses the threshold, its
recommendation names INT16 (or per-group quantization) as the fix. This pass
is that fix -- a drop-in, per-node alternative for the specific weights
`quantize_weight_only`'s INT8 handles poorly, not a blanket replacement for
it: INT16 is only ~2x smaller than float32 (INT8 is ~4x), so use it where the
extra resolution actually earns back the smaller size win.

```python
import onnx
import onnxsim

model = onnx.load("model.onnx")
report = onnxsim.estimate_quantization_precision(model)
outlier_heavy = [
    r.node_name for r in report
    if hasattr(r, "outlier_risk") and r.outlier_risk
]
print("consider INT16 for:", outlier_heavy)
```

## Scope

Handled:
- `MatMul(X, W)` with `W` a constant 2-D float32 tensor.
- `Gemm(X, W[, B])` with `transA=0`, `alpha=1`, `beta=1` (when `B` is
  present), `W` a constant 2-D float32 tensor. `transB` may be 0 or 1.
- `Conv(X, W[, B])` with `W` a constant float32 tensor, rank >= 3
  (`[Cout, Cin/groups, k...]`).
- **Opsets >= 21** -- INT16 is a `QuantizeLinear`/`DequantizeLinear` type
  only from opset 21 onward (unlike `quantize_weight_only`'s INT8, which
  only needs opset >= 13). An opset in `[13, 21)` is old for this pass even
  though it would be fine for the INT8 one.

Left untouched (safe no-op, node passes through as-is):
- Non-constant weights (e.g. two activations multiplied together).
- Non-default Gemm attributes (`alpha != 1`, `transA != 0`, or `beta != 1`
  when a bias is present).
- Non-float32 activations or weights, or an opset older than 21.

## End-to-end: simplify -> quantize -> deploy

### CLI

```bash
onnxsim model.onnx model.quant.onnx --weight-only-quantize-int16
```

### Python API

```python
import onnx
import onnxruntime as ort
import onnxsim

model = onnx.load("model.onnx")
model, check_ok = onnxsim.simplify(model)
assert check_ok

model = onnxsim.quantize_weight_only_int16(model)
onnx.save(model, "model.quant.onnx")

sess = ort.InferenceSession("model.quant.onnx", providers=["CPUExecutionProvider"])
outputs = sess.run(None, {"input": your_input_array})
```

`tests/test_weight_only_quantize_int16.py` runs this simplify -> quantize ->
deploy sequence on small `MatMul`/`Gemm`/`Conv` models, executing both the
float and quantized graphs through `onnxruntime.InferenceSession`, plus a
direct INT8-vs-INT16 comparison confirming INT16 tracks the float baseline
more closely for the same weight.

## Relationship to onnxsim's other quantization methods

See `docs/weight-only-quantization.md`'s comparison table for how
`quantize_weight_only`'s INT8 relates to `quantize_dynamic`/`quantize_static`
-- this pass sits at the same point in that table (activation untouched, no
calibration data), just with INT16 in place of INT8:

| | Bit width | Typical size vs. float32 | Best for |
|---|---|---|---|
| `quantize_weight_only` | INT8 | ~4x smaller | most weights |
| `quantize_weight_only_int16` | INT16 | ~2x smaller | outlier-heavy channels INT8 resolves poorly |
| `quantize_weight_only_int4` | INT4 (block-wise) | ~8x smaller (weights) | maximum compression, GPTQ/AWQ-style |

A model can freely mix nodes quantized by different passes -- for example,
running `quantize_weight_only` first and then re-quantizing just the
specific nodes `estimate_quantization_precision` flags with
`quantize_weight_only_int16` (both target the same `MatMul`/`Gemm`/`Conv`
node shapes, so the later pass's `DequantizeLinear` simply replaces the
earlier one's weight input).
