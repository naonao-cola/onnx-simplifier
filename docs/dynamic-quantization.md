# Dynamic INT8 quantization (`quantize_dynamic`)

## What this is

`onnxsim.quantize_dynamic` is a single, self-contained C++ graph rewrite
(`onnxsim/passes/dynamic_quantize_matmul.h`) that dynamically quantizes every
`MatMul`, and every "vanilla" `Gemm` (`transA=0`, `alpha=1`, `beta=1`), whose
weight is a constant 2-D float32 tensor:

- The **weight** is quantized to INT8 ahead of time, per output channel,
  symmetric (`zero_point = 0`), from its static values alone. No calibration
  dataset is needed.
- The **activation** is quantized to uint8 *inside the graph* by
  `DynamicQuantizeLinear`, which computes its own scale/zero-point from each
  run's actual input range.

```
Before:
  Y = MatMul(X, W)                                   # W constant, [K, N], float32

After:
  Xq, Xs, Xzp = DynamicQuantizeLinear(X)              # uint8, computed at runtime
  Wq          = <int8, per-column symmetric>          # computed once, here
  Ws          = <float32, one scale per column of W>
  Acc         = MatMulInteger(Xq, Wq, Xzp)            # int32
  Y           = Cast<float>(Acc) * (Xs * Ws)          # dequantize
```

A `Gemm` bias (`C`), if present, is left in float and added back after
dequantization.

This mirrors the "dynamic quantization" scheme [ONNX Runtime's
`quantize_dynamic`](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
applies to `MatMul`/`Gemm` — it is not a from-scratch reimplementation of every
quantization framework in the ecosystem (ONNX Runtime's full quantization
toolkit, Intel Neural Compressor, TensorRT, AIMET, Brevitas, Vitis AI,
AutoRound, ...); those are large, independent projects, several tied to
vendor-specific hardware SDKs. What onnxsim adds is one well-scoped, C++,
calibration-free PTQ pass that fits its existing graph-rewrite architecture
(see `onnxsim/passes/*.h`) and needs no external runtime to *apply* — the
rewrite itself is pure graph surgery, only *running* the result needs an
executor (see below).

Because the weight-side math is entirely static, `quantize_dynamic` does not
need a `ModelExecutor` (unlike `simplify`'s constant folding) — it runs
directly on the model's protobuf bytes.

## Scope

Handled:
- `MatMul(X, W)` with `W` a constant 2-D float32 tensor.
- `Gemm(X, W[, B])` with `transA=0`, `alpha=1`, `beta=1` (when `B` is present),
  `W` a constant 2-D float32 tensor. `transB` may be 0 or 1 — `transB=1` is
  what PyTorch's ONNX exporter emits for `nn.Linear`, so it is the common case
  in practice, not the exception.
- Opsets >= 11 (`DynamicQuantizeLinear` was introduced in opset 11).

Left untouched (safe no-op, node passes through as-is):
- Non-constant or non-2-D weights (e.g. two activations multiplied together).
- Non-default Gemm attributes (`alpha != 1`, `transA != 0`, or `beta != 1`
  when a bias is present).
- Non-float32 activations, or an opset older than 11.

## End-to-end: simplify -> quantize -> deploy

### CLI

```bash
onnxsim model.onnx model.quant.onnx --dynamic-quantize
```

This runs onnxsim's usual simplification first, then applies
`quantize_dynamic` to the result before saving. Combine with the normal
simplify flags as needed (`--skip-fuse-bn`, `--overwrite-input-shape`, ...).

### Python API

```python
import onnx
import onnxruntime as ort
import onnxsim

# 1. Simplify.
model = onnx.load("model.onnx")
model, check_ok = onnxsim.simplify(model)
assert check_ok

# 2. Quantize (INT8 weights, dynamic uint8 activations).
model = onnxsim.quantize_dynamic(model)
onnx.save(model, "model.quant.onnx")

# 3. Deploy: run it like any other ONNX model. DynamicQuantizeLinear and
#    MatMulInteger are standard ONNX ops with CPU (and, on most builds, other
#    execution provider) kernels in ONNX Runtime -- no special runtime support
#    needed.
sess = ort.InferenceSession("model.quant.onnx", providers=["CPUExecutionProvider"])
outputs = sess.run(None, {"input": your_input_array})
```

`tests/test_dynamic_quantize_matmul.py` runs exactly this simplify → quantize
→ deploy sequence on small `MatMul`/`Gemm` models, executing both the float
and quantized graphs through `onnxruntime.InferenceSession` and checking the
outputs stay close, as a minimal end-to-end regression test.

## Why this shape

- **No calibration data required.** Picking activation ranges from sample data
  ("static" PTQ, what ONNX Runtime's `quantize_static` and most calibration
  based tools do) needs a representative dataset and a run of the model before
  you can even produce the quantized graph. Dynamic quantization instead
  computes the activation's scale/zero-point *at inference time*, on the
  actual input, via `DynamicQuantizeLinear` — so the rewrite here is a pure,
  data-independent graph transform, consistent with the rest of onnxsim's
  passes (which all run without needing sample inputs).
- **Per-channel weight scale.** A single scale for the whole weight matrix
  would be dominated by the largest-magnitude output channel, wasting
  precision on every other channel. Scaling per output column keeps each
  channel's full INT8 range.
- **Symmetric weight quantization (`zero_point = 0`).** `MatMulInteger`'s
  `b_zero_point` does support a per-column zero point, but a symmetric range
  needs none at all, which keeps the inserted graph smaller.
