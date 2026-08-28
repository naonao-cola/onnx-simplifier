# Static "W8A16" quantization (`quantize_static_int16`)

## What this is

`onnxsim.quantize_static_int16` is a pair of single, self-contained C++
graph rewrites (`onnxsim/passes/static_quantize_int16_matmul.h`,
`onnxsim/passes/static_quantize_int16_conv.h`) that statically
(calibration-based) quantize every `MatMul`, every "vanilla" `Gemm`
(`transA=0`, `alpha=1`, `beta=1`), and every `Conv`, whose weight is a
constant float32 tensor -- the exact same QDQ scheme `onnxsim.quantize_static`
uses, with one change: the **activation** is quantized to uint16 instead of
uint8, while the **weight stays INT8**. This is sometimes called a "W8A16"
scheme (8-bit weight, 16-bit activation).

```
Before:
  Y = MatMul(X, W)                                   # W constant, [K, N], float32

After (QDQ format -- the MatMul/Gemm/Conv node itself is untouched, only its
inputs change):
  Xq  = QuantizeLinear(X, Xs, Xzp)     -- Xs/Xzp: CALIBRATED, fixed, uint16
  Xdq = DequantizeLinear(Xq, Xs, Xzp)
  Wdq = DequantizeLinear(Wq, Ws, axis=1)              -- Wq: int8, per-channel
  Y   = MatMul(Xdq, Wdq)
```

## Why only the activation gets the finer type

The graph still computes in float32 either way -- QDQ format never replaces
`MatMul`/`Gemm`/`Conv` with an integer kernel directly, it just brackets the
float op with a `Cast`-like `QuantizeLinear`/`DequantizeLinear` round trip
that a QDQ-aware runtime can later fuse. So the only thing a wider
activation type changes is how much rounding error that round trip
introduces on `X` before it reaches the (still-float32) op -- the weight's
own INT8 precision is completely unaffected by the activation's type. Widening
the weight too would cost real storage for no matching accuracy benefit,
since the weight's precision was never the bottleneck this pass targets.

This is aimed at activations a QDQ round trip is unusually sensitive to:

- A tensor whose calibrated range is wide relative to its typical
  value -- uint8's coarser step (1/255 relative) rounds most of that
  tensor's actual values to a small handful of representable levels.
- Post-softmax attention scores, or any other activation feeding a
  downstream computation where small per-element errors compound across
  many terms.

uint16's ~8x finer step (1/65535 relative vs uint8's 1/255) resolves both
cases while keeping the weight's INT8 compression exactly as-is.

## Scope

Handled: identical to `quantize_static` -- see its own scope description
(mirrored by `onnxsim.list_quantizable_activations`) -- with one difference:

- **Opsets >= 21** -- uint16 is a `QuantizeLinear`/`DequantizeLinear` type
  only from opset 21 onward, unlike `quantize_static`'s uint8 scheme, which
  only needs opset 13. An opset in `[13, 21)` is old for this pass even
  though it would be fine for the uint8 one.

Like `quantize_static`, a node is only rewritten when its activation's
tensor name has a calibrated `(min, max)` entry -- this pass reads the same
calibration-ranges data `quantize_static` does, so any calibration you
already ran for `quantize_static` works unchanged here.

## End-to-end: calibrate -> quantize -> deploy

### CLI

```bash
onnxsim model.onnx model.quant.onnx --static-quantize-int16 \
  --calibration-dataset mnist --calibration-samples 32
```

(Omit `--calibration-dataset` to calibrate from random data instead.)

### Python API

```python
import onnx
import onnxruntime as ort
import onnxsim

model = onnx.load("model.onnx")
model, check_ok = onnxsim.simplify(model)
assert check_ok

# calibration_data defaults to onnxsim.generate_random_calibration_data;
# pass real representative data (e.g. via onnxsim.load_huggingface_calibration_data)
# for a tighter, more accurate calibration.
model = onnxsim.quantize_static_int16(model)
onnx.save(model, "model.quant.onnx")

sess = ort.InferenceSession("model.quant.onnx", providers=["CPUExecutionProvider"])
outputs = sess.run(None, {"input": your_input_array})
```

`tests/test_static_quantize_int16.py` runs this calibrate -> quantize ->
deploy sequence on small `MatMul`/`Gemm`/`Conv` models, executing both the
float and quantized graphs through `onnxruntime.InferenceSession`, plus a
direct comparison against `quantize_static`'s uint8 scheme on the same
calibrated activation confirming the uint16 round trip tracks the float
baseline more closely.

## Relationship to onnxsim's other quantization methods

| | Weight | Activation | Calibration data | Opset floor |
|---|---|---|---|---|
| `quantize_static` | INT8, per-channel | uint8, calibrated QDQ | required | 13 |
| `quantize_static_int16` | INT8, per-channel | uint16, calibrated QDQ | required | 21 |
| `quantize_qoperator` | INT8, per-channel | uint8, calibrated, `QLinearMatMul` | required | 13 |
| `quantize_dynamic` | INT8, per-channel | uint8, computed at runtime | none | 13 |
| `quantize_weight_only` | INT8, per-channel | untouched | none | 13 |

Pick `quantize_static_int16` over plain `quantize_static` when calibration
shows a specific activation's range is unusually wide, or when the model's
overall accuracy is more sensitive to activation rounding than its weight
size -- otherwise `quantize_static`'s uint8 scheme is the better default,
since it works on any opset >= 13 runtime and both formats need the same
calibration data either way.
