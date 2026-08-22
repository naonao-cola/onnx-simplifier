# QOperator format quantization (`quantize_qoperator`)

## What this is

`onnxsim.quantize_qoperator` is a single, self-contained C++ graph rewrite
(`onnxsim/passes/qoperator_quantize_matmul.h`) that statically
(calibration-based) quantizes every `MatMul` and every "vanilla" `Gemm`
(`transA=0`, `alpha=1`, `beta=1`), whose weight is a constant 2-D float32
tensor, into the **"QOperator" format** -- `QLinearMatMul`, ONNX's
directly-quantized matmul op -- rather than `quantize_static`'s **QDQ**
format (`QuantizeLinear`/`DequantizeLinear` wrapping a float `MatMul`).

Both are standard ONNX operators, and both need the same kind of calibrated
activation range `quantize_static` does. The difference is what the rewrite
produces:

```
Before:
  Y = MatMul(X, W)                                       # W constant, [K, N], float32

After (QOperator format):
  Xq = QuantizeLinear(X, Xs, Xzp)                         # Xs/Xzp: CALIBRATED
  Yq = QLinearMatMul(Xq, Xs, Xzp, Wq, Ws, Wzp, Ys, Yzp)   # true int8 compute
  Y  = DequantizeLinear(Yq, Ys, Yzp)                      # Ys/Yzp: CALIBRATED
```

versus QDQ format's:

```
After (QDQ format, quantize_static):
  Xdq = DequantizeLinear(QuantizeLinear(X, Xs, Xzp), Xs, Xzp)
  Wdq = DequantizeLinear(Wq, Ws)
  Y   = MatMul(Xdq, Wdq)          # MatMul itself is untouched
```

`Wq`/`Ws` are the same per-output-channel symmetric INT8 weight
quantization every other onnxsim quantization pass uses.

## Why this needs an extra calibration step

QDQ format's `MatMul` still computes in float32 -- only a QDQ-aware runtime
recognizes the bracketing pattern and fuses it into an integer kernel at
load time, so the graph's *declared* dtypes stay float until that fusion
happens. `QLinearMatMul` has no such deferral: it computes directly in int8,
so its output `Y` is quantized *in the graph itself* -- there is no float
`MatMul` left anywhere. That means the rewrite needs a calibrated range for
the node's **output**, not just its activation input, which QDQ format never
required (its `DequantizeLinear` can hand back float and let whatever
consumes it worry about its own range later).

`list_qoperator_quantizable_outputs` (used internally by
`quantize_qoperator`) reports exactly those output tensor names, on top of
`list_quantizable_activations`' input names; `calibrate()`'s
`extra_tensor_names` parameter is how they get folded into the same
calibration run.

## Why QDQ is usually preferred, and when QOperator still matters

QDQ is ONNX Runtime's (and most modern runtimes') preferred format today
because it composes: when several quantized ops chain together, a QDQ-aware
optimizer can fuse the whole chain into integer kernels and drop every
intermediate float round-trip, whereas QOperator format commits to int8 (or
back to float, via an explicit `DequantizeLinear`) at every single node
individually. For one isolated `MatMul` in an otherwise-float graph -- what
this pass handles -- both formats end up inserting a comparable number of
nodes.

QOperator format still matters for runtimes/toolchains whose int8 kernels
key off `QLinearMatMul` specifically rather than recognizing the QDQ
pattern, or for older deployment pipelines that predate QDQ's adoption as
the default.

## Scope

Handled:
- `MatMul(X, W)` with `W` a constant 2-D float32 tensor.
- `Gemm(X, W[, B])` with `transA=0`, `alpha=1`, `beta=1` (when `B` is
  present), same weight constraint. `transB` may be 0 or 1. `B`, if present,
  is added back in float after dequantization (`QLinearMatMul` has no bias
  input).
- Opsets >= 10 (`QLinearMatMul`'s own minimum).

Left untouched (safe no-op, node passes through as-is):
- Non-constant or non-2-D weights, non-default Gemm attributes, non-float32
  operands, an opset older than 10, or a node whose activation and/or output
  tensor has no calibrated range.
- `Conv` -- not yet covered (`QLinearConv` is a natural, still-open
  follow-up with the same shape as this pass).

## End-to-end: simplify -> quantize -> deploy

### CLI

```bash
onnxsim model.onnx model.quant.onnx --qoperator-quantize
```

### Python API

```python
import onnx
import onnxruntime as ort
import onnxsim

model = onnx.load("model.onnx")
model, check_ok = onnxsim.simplify(model)
assert check_ok

model = onnxsim.quantize_qoperator(model)
onnx.save(model, "model.quant.onnx")

sess = ort.InferenceSession("model.quant.onnx", providers=["CPUExecutionProvider"])
outputs = sess.run(None, {"input": your_input_array})
```

`tests/test_qoperator_quantize_matmul.py` runs this simplify -> quantize ->
deploy sequence on small `MatMul`/`Gemm` models, executing both the float and
quantized graphs through `onnxruntime.InferenceSession`.
