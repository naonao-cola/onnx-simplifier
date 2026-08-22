# QOperator format quantization (`quantize_qoperator`)

## What this is

`onnxsim.quantize_qoperator` is two self-contained C++ graph rewrites
(`onnxsim/passes/qoperator_quantize_matmul.h` and
`onnxsim/passes/qoperator_quantize_conv.h`) that statically
(calibration-based) quantize every `MatMul`, every "vanilla" `Gemm`
(`transA=0`, `alpha=1`, `beta=1`), and every `Conv`, whose weight is a
constant float32 tensor (2-D for `MatMul`/`Gemm`, rank >= 3 for `Conv`), into
the **"QOperator" format** -- `QLinearMatMul`/`QLinearConv`, ONNX's
directly-quantized matmul/convolution ops -- rather than `quantize_static`'s
**QDQ** format (`QuantizeLinear`/`DequantizeLinear` wrapping a float op).

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

`Conv` follows the identical shape with `QLinearConv` in place of
`QLinearMatMul`; its optional bias, unlike Gemm's, is pre-quantized to INT32
(`QLinearConv`'s own bias input) rather than added back in float, since
`QLinearConv` accepts one directly -- see "Scope" below for the constraint
that puts on which Conv nodes get rewritten.

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
- `Conv(X, W[, B])` with `W` a constant float32 tensor, rank >= 3
  (`[Cout, Cin/groups, k...]`). `B`, if present, must be a constant float32
  `[Cout]` tensor -- `QLinearConv` takes its bias pre-quantized to INT32
  (scale = `x_scale * w_scale[c]`, zero_point 0), so unlike Gemm's bias there
  is no float fallback for a non-constant one; such a Conv is left untouched
  instead (see `passes/qoperator_quantize_conv.h`).
- Opsets >= 10 (`QLinearMatMul`/`QLinearConv`'s own minimum).

Left untouched (safe no-op, node passes through as-is):
- Non-constant or wrong-rank weights, non-default Gemm attributes,
  non-float32 operands, an opset older than 10, a node whose activation
  and/or output tensor has no calibrated range, or a Conv whose bias is
  present but not a constant float32 `[Cout]` tensor.

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
