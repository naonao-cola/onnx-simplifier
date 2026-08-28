# QOperator elementwise quantization (`quantize_qoperator_elementwise`)

## What this is

`onnxsim.quantize_qoperator_elementwise` is a self-contained C++ graph
rewrite (`onnxsim/passes/qoperator_quantize_elementwise.h`) that statically
(calibration-based) quantizes every elementwise `Add`/`Mul` node whose two
inputs are both **non-constant** float32 tensors -- a residual connection, an
elementwise gate between two activations, and similar shapes -- into ONNX
Runtime's **`com.microsoft`** contrib ops `QLinearAdd`/`QLinearMul`: the
elementwise analogue of `quantize_qoperator`'s `QLinearMatMul`/`QLinearConv`
rewrite.

```
Before (illustrated for Add; Mul is identical but for the op/QLinear* name):
  Z = Add(A, B)                                          # A, B: both runtime float32 tensors

After:
  Aq = QuantizeLinear(A, As, Azp)                         # As/Azp: CALIBRATED
  Bq = QuantizeLinear(B, Bs, Bzp)                         # Bs/Bzp: CALIBRATED
  Zq = QLinearAdd(Aq, As, Azp, Bq, Bs, Bzp, Zs, Zzp)      # true int8 compute
  Z  = DequantizeLinear(Zq, Zs, Zzp)                      # Zs/Zzp: CALIBRATED
```

## Why this is a contrib op, not standard ONNX

Standard ONNX has no quantized elementwise-binary operator at all: only
`QLinearMatMul`/`QLinearConv`, and both are constrained to a
weight-times-activation shape, not two arbitrary runtime tensors. ONNX
Runtime fills that gap with its own `com.microsoft` domain contrib ops
`QLinearAdd`/`QLinearMul`. This is the only `quantize_*` function in onnxsim
whose output is not portable standard ONNX: the quantized model needs a
`com.microsoft`-aware runtime (ONNX Runtime itself, or another runtime
importing the same contrib schemas) to execute. `quantize_qoperator_elementwise`
adds `com.microsoft` (version 1) to the model's opset imports the first time
it rewrites a node.

## Why both operands need a calibrated range

Unlike `quantize_qoperator`'s `QLinearMatMul`/`QLinearConv` (one calibrated
activation, one weight quantized ahead of time from its own static values),
`QLinearAdd`/`QLinearMul` treat both operands identically -- there is no
"weight" role at all. So **both** `A` and `B` need a calibrated range, on top
of the output `Z`'s (QOperator format computes directly in int8, so the
output must be quantized too -- the same reason `QLinearMatMul`'s `Y` needs
one; see `docs/qoperator-quantization.md`).

`list_qoperator_elementwise_quantizable_tensors` reports all three tensor
names (both operands plus the output) for every qualifying node;
`calibrate()`'s `extra_tensor_names` parameter is how they get folded into
the same calibration run -- `quantize_qoperator_elementwise` (the Python
wrapper in `onnxsim/calibration.py`) does this automatically.

## Why a constant operand is left alone

A node with one constant operand -- e.g. a per-channel bias or a learned
embedding added elementwise -- is **not** rewritten by this pass, even
though `QLinearAdd`/`QLinearMul`'s own schema would accept it. A constant is
better quantized from its own static values (the same per-channel scheme
`quantize_static`'s weight handling uses) than force-fed through the runtime
calibration harness as if it varied at inference time. This pass only
targets the genuinely dynamic case: both operands are runtime activations.

## Scope

Handled:
- `Add(A, B)` or `Mul(A, B)`, both inputs float32, **neither a constant**.

Left untouched (safe no-op, node passes through as-is):
- A constant operand, non-float32 operands, or a node whose operands and/or
  output tensor has no calibrated range.
- A node consuming *another* rewritten node's output in the same
  quantization call: this pass (like `qoperator_quantize_matmul.h`/
  `qoperator_quantize_conv.h`) replaces the matched node with a fresh output
  Value, so a downstream node's calibrated-range lookup (keyed by the
  *original* tensor name) won't find an entry for that edge afterwards --
  a pre-existing characteristic of the whole QOperator rewrite family, not
  specific to elementwise ops.

## End-to-end: simplify -> quantize -> deploy

### CLI

```bash
onnxsim model.onnx model.quant.onnx --qoperator-quantize-elementwise
```

### Python API

```python
import onnx
import onnxruntime as ort
import onnxsim

model = onnx.load("model.onnx")
model, check_ok = onnxsim.simplify(model)
assert check_ok

model = onnxsim.quantize_qoperator_elementwise(model)
onnx.save(model, "model.quant.onnx")

sess = ort.InferenceSession("model.quant.onnx", providers=["CPUExecutionProvider"])
outputs = sess.run(None, {"input": your_input_array})
```

`tests/test_qoperator_quantize_elementwise.py` runs this simplify -> quantize
-> deploy sequence on small `Add`/`Mul` models (including a broadcasting
case), executing both the float and quantized graphs through
`onnxruntime.InferenceSession`.
