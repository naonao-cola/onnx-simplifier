# Weight-only INT8 quantization (`quantize_weight_only`)

## What this is

`onnxsim.quantize_weight_only` is a pair of single, self-contained C++ graph
rewrites (`onnxsim/passes/weight_only_quantize_matmul.h`,
`onnxsim/passes/weight_only_quantize_conv.h`) that quantize every `MatMul`,
every "vanilla" `Gemm` (`transA=0`, `alpha=1`, `beta=1`), and every `Conv`,
whose weight is a constant float32 tensor:

- The **weight** is quantized to INT8 ahead of time, per output channel,
  symmetric (`zero_point = 0`), from its static values alone -- the same
  scheme `quantize_dynamic`/`quantize_static` use.
- The **activation is never touched.** No `DynamicQuantizeLinear`, no
  QuantizeLinear/DequantizeLinear pair, no calibration data of any kind.

```
Before:
  Y = MatMul(X, W)                                   # W constant, [K, N], float32

After:
  Wq  = <int8, per-column symmetric>                 # computed once, here
  Ws  = <float32, one scale per column of W>
  Wdq = DequantizeLinear(Wq, Ws, axis=1)              # float32
  Y   = MatMul(X, Wdq)                                # X is exactly as it was
```

A `Gemm`/`Conv` bias, if present, is left in float and untouched.

This is the scheme real-world weight-heavy ONNX deployments most often ship
in practice -- large linear/embedding layers in transformer-style ASR/TTS
decoders, for example -- rather than full activation quantization: the
weights are static and dominate model size, so compressing them is nearly
free, while quantizing activations adds a runtime quantize/dequantize cost
for comparatively little size benefit. It differs from `quantize_dynamic`
(which additionally quantizes the activation to uint8 at runtime via
`DynamicQuantizeLinear`) and `quantize_static` (which additionally quantizes
the activation via a calibrated QuantizeLinear/DequantizeLinear pair) in
that **nothing about the activation path changes** -- this only shrinks the
serialized/loaded model's weight storage (~4x for the quantized tensors).

Because the weight-side math is entirely static, `quantize_weight_only` does
not need a `ModelExecutor` or any calibration data -- it runs directly on the
model's protobuf bytes, just like `quantize_dynamic`.

## Scope

Handled:
- `MatMul(X, W)` with `W` a constant 2-D float32 tensor.
- `Gemm(X, W[, B])` with `transA=0`, `alpha=1`, `beta=1` (when `B` is
  present), `W` a constant 2-D float32 tensor. `transB` may be 0 or 1.
- `Conv(X, W[, B])` with `W` a constant float32 tensor, rank >= 3
  (`[Cout, Cin/groups, k...]`).
- Opsets >= 13 (`DequantizeLinear`'s per-channel `axis` attribute needs
  opset 13).

Left untouched (safe no-op, node passes through as-is):
- Non-constant weights (e.g. two activations multiplied together).
- Non-default Gemm attributes (`alpha != 1`, `transA != 0`, or `beta != 1`
  when a bias is present).
- Non-float32 activations or weights, or an opset older than 13.

## End-to-end: simplify -> quantize -> deploy

### CLI

```bash
onnxsim model.onnx model.quant.onnx --weight-only-quantize
```

### Python API

```python
import onnx
import onnxruntime as ort
import onnxsim

model = onnx.load("model.onnx")
model, check_ok = onnxsim.simplify(model)
assert check_ok

model = onnxsim.quantize_weight_only(model)
onnx.save(model, "model.quant.onnx")

sess = ort.InferenceSession("model.quant.onnx", providers=["CPUExecutionProvider"])
outputs = sess.run(None, {"input": your_input_array})
```

`tests/test_weight_only_quantize.py` runs this simplify -> quantize -> deploy
sequence on small `MatMul`/`Gemm`/`Conv` models, executing both the float and
quantized graphs through `onnxruntime.InferenceSession`.

## Relationship to `quantize_dynamic`/`quantize_static`

All three passes share the same per-output-channel symmetric INT8 weight
quantizer; they differ only in what (if anything) happens to the activation:

| | Activation | Calibration data | Runtime activation cost |
|---|---|---|---|
| `quantize_weight_only` | untouched | none | none |
| `quantize_dynamic` | uint8, computed at runtime | none | `DynamicQuantizeLinear` per call |
| `quantize_static` | uint8, fixed QDQ range | required | `QuantizeLinear`/`DequantizeLinear` (fusable by a QDQ-aware runtime) |

Combining a weight-only pass with `quantize_dynamic`/`quantize_static` on the
same node is not meaningful (the weight input is already quantized) --
`quantize_weight_only` is a distinct, standalone entry point, not a flag on
the other two.
