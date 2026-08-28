# QOperator concat quantization (`quantize_qoperator_concat`)

## What this is

`onnxsim.quantize_qoperator_concat` is a self-contained C++ graph rewrite
(`onnxsim/passes/qoperator_quantize_concat.h`) that statically
(calibration-based) quantizes every `Concat` node whose inputs are all
non-constant float32 tensors into ONNX Runtime's **`com.microsoft`** contrib
op `QLinearConcat` -- the variadic analogue of
`quantize_qoperator_elementwise`'s `QLinearAdd`/`QLinearMul` rewrite (see
`docs/qoperator-quantize-elementwise.md` for why these are contrib, not
standard, ONNX ops).

```
Before (illustrated for 2 inputs; QLinearConcat is variadic, so this
generalizes to any input count):
  Z = Concat(A, B, axis=ax)   -- A, B: both runtime float32 tensors

After:
  Aq = QuantizeLinear(A, As, Azp)   -- As/Azp: CALIBRATED
  Bq = QuantizeLinear(B, Bs, Bzp)   -- Bs/Bzp: CALIBRATED
  Zq = QLinearConcat(Zs, Zzp, Aq, As, Azp, Bq, Bs, Bzp, axis=ax)
       -- true int8 compute
  Z  = DequantizeLinear(Zq, Zs, Zzp)   -- Zs/Zzp: CALIBRATED
```

## The output scale/zero-point does double duty

Unlike `QLinearAdd`/`QLinearMul`, `QLinearConcat`'s schema takes
`Y_scale`/`Y_zero_point` as its **first two inputs**, not trailing outputs of
some separate computation -- the node produces its result directly at that
calibrated scale/zero-point. This pass reuses the same `(Ys, Yzp)` initializer
pair for the trailing `DequantizeLinear`, so the output is calibrated once and
that single pair does double duty.

Every input still needs its own calibrated range too, same as `QLinearAdd`/
`QLinearMul`'s `A` and `B` -- `QLinearConcat` has no "weight" role either.
`list_qoperator_concat_quantizable_tensors` reports every input's tensor name
plus the node's output name for each qualifying node; `calibrate()`'s
`extra_tensor_names` parameter is how they get folded into the same
calibration run -- `quantize_qoperator_concat` (the Python wrapper in
`onnxsim/calibration.py`) does this automatically.

## Why a constant operand is left alone

Same reasoning as `quantize_qoperator_elementwise`: a `Concat` input that is
a constant tensor is better quantized from its own static values than
force-fed through the runtime calibration harness as if it varied at
inference time. A `Concat` with *any* constant input is left untouched
entirely, rather than partially rewritten.

## Scope

Handled:
- A `Concat` whose every input is a non-constant float32 tensor (any input
  count -- `QLinearConcat` is variadic).

Left untouched (safe no-op, node passes through as-is):
- Any constant or non-float32 input, or a node whose inputs and/or output
  tensor has no calibrated range.
- A node consuming *another* rewritten node's output in the same
  quantization call -- the same pre-existing QOperator-family
  characteristic `qoperator_quantize_elementwise.h` documents.

## End-to-end: simplify -> quantize -> deploy

### CLI

```bash
onnxsim model.onnx model.quant.onnx --qoperator-quantize-concat
```

### Python API

```python
import onnx
import onnxruntime as ort
import onnxsim

model = onnx.load("model.onnx")
model, check_ok = onnxsim.simplify(model)
assert check_ok

model = onnxsim.quantize_qoperator_concat(model)
onnx.save(model, "model.quant.onnx")

sess = ort.InferenceSession("model.quant.onnx", providers=["CPUExecutionProvider"])
outputs = sess.run(None, {"input": your_input_array})
```

`tests/test_qoperator_quantize_concat.py` runs this simplify -> quantize ->
deploy sequence on small `Concat` models (2 and 3 inputs), executing both the
float and quantized graphs through `onnxruntime.InferenceSession`.
