# QOperator activation quantization (`quantize_qoperator_activation`)

## What this is

`onnxsim.quantize_qoperator_activation` is a self-contained C++ graph rewrite
(`onnxsim/passes/qoperator_quantize_activation.h`) that statically
(calibration-based) quantizes every standalone `Sigmoid` or `LeakyRelu` node
whose input is a float32 tensor into ONNX Runtime's **`com.microsoft`**
contrib ops `QLinearSigmoid`/`QLinearLeakyRelu` -- the unary-activation
analogue of `quantize_qoperator_elementwise`'s `QLinearAdd`/`QLinearMul`
rewrite (see `docs/qoperator-quantize-elementwise.md` for why these are
contrib, not standard, ONNX ops).

```
Before (illustrated for Sigmoid; LeakyRelu is identical but for the
op/QLinear* name and its `alpha` attribute, carried over unchanged):
  Y = Sigmoid(X)                                    # X: runtime float32 tensor

After:
  Xq = QuantizeLinear(X, Xs, Xzp)                    # Xs/Xzp: CALIBRATED
  Yq = QLinearSigmoid(Xq, Xs, Xzp, Ys, Yzp)          # true int8 compute
  Y  = DequantizeLinear(Yq, Ys, Yzp)                 # Ys/Yzp: CALIBRATED
```

## Why the output needs a calibrated range too

Like every other QOperator-format pass in onnxsim, `QLinearSigmoid`/
`QLinearLeakyRelu` compute directly in int8 -- there is no float
intermediate -- so the node's **output**, not just its input, needs a
calibrated range. `list_qoperator_activation_quantizable_tensors` reports
both tensor names for every qualifying node; `calibrate()`'s
`extra_tensor_names` parameter is how they get folded into the same
calibration run -- `quantize_qoperator_activation` (the Python wrapper in
`onnxsim/calibration.py`) does this automatically.

## `alpha` is carried over unchanged

`LeakyRelu`'s `alpha` attribute (the negative-slope coefficient, ONNX
default `0.01`) is read from the matched node -- whether explicit or
defaulted -- and written onto the replacement `QLinearLeakyRelu` node
unchanged. `Sigmoid` has no such attribute.

## Why this is a contrib op, not standard ONNX

Standard ONNX has no quantized unary-activation operator either (only
`QLinearMatMul`/`QLinearConv`, both weight-shaped). ONNX Runtime fills the
gap with `com.microsoft` contrib ops, the same way it does for
`QLinearAdd`/`QLinearMul`. The quantized model needs a `com.microsoft`-aware
runtime (ONNX Runtime itself, or another runtime importing the same contrib
schemas) to execute; `quantize_qoperator_activation` adds `com.microsoft`
(version 1) to the model's opset imports the first time it rewrites a node.

## Scope

Handled:
- A standalone `Sigmoid(X)` or `LeakyRelu(X)`, `X` float32.

Left untouched (safe no-op, node passes through as-is):
- A non-float32 input, or a node whose input and/or output tensor has no
  calibrated range.
- A node consuming *another* rewritten node's output in the same
  quantization call -- the same pre-existing QOperator-family
  characteristic `qoperator_quantize_elementwise.h` documents (the rewrite
  replaces the matched node with a fresh output Value, so a downstream
  node's calibrated-range lookup, keyed by the *original* tensor name, won't
  find an entry for that edge afterwards).

## End-to-end: simplify -> quantize -> deploy

### CLI

```bash
onnxsim model.onnx model.quant.onnx --qoperator-quantize-activation
```

### Python API

```python
import onnx
import onnxruntime as ort
import onnxsim

model = onnx.load("model.onnx")
model, check_ok = onnxsim.simplify(model)
assert check_ok

model = onnxsim.quantize_qoperator_activation(model)
onnx.save(model, "model.quant.onnx")

sess = ort.InferenceSession("model.quant.onnx", providers=["CPUExecutionProvider"])
outputs = sess.run(None, {"input": your_input_array})
```

`tests/test_qoperator_quantize_activation.py` runs this simplify -> quantize
-> deploy sequence on small `Sigmoid`/`LeakyRelu` models (including a
non-default `alpha`), executing both the float and quantized graphs through
`onnxruntime.InferenceSession`.
