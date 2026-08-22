# Block-wise INT4 weight-only quantization (`quantize_weight_only_int4`)

## What this is

`onnxsim.quantize_weight_only_int4` is a single, self-contained C++ graph
rewrite (`onnxsim/passes/weight_only_quantize_int4_matmul.h`) that
block-wise quantizes every `MatMul`, and every "vanilla" `Gemm` (`transA=0`,
`alpha=1`, `beta=1`), whose weight is a constant 2-D float32 tensor whose
reduction dimension `K` is evenly divisible by 32.

It extends `quantize_weight_only` (INT8, one scale per output channel) with
a smaller data type and finer granularity:

- The **weight** is quantized to INT4 (values in `[-7, 7]`), with a
  **separate symmetric scale per 32-element block of `K`, per output
  channel** -- the GPTQ/AWQ-style block quantization real weight-heavy LLM
  and ASR deployments increasingly ship (the [Audio8
  TTS-Preview-0.6B-ONNX-INT4](https://huggingface.co/Audio8/Audio8-TTS-Preview-0.6B-ONNX-INT4)
  model is one example).
- The **activation is never touched** -- same as `quantize_weight_only`: no
  calibration data, no runtime quantize/dequantize cost on the activation
  path.

```
Before:
  Y = MatMul(X, W)                                    # W constant, [K, N], float32

After:
  Wq  = <int4, per-(block, column) symmetric>          # computed once, here
  Ws  = <float32, [K/32, N]>                            # one scale per block per column
  Wdq = DequantizeLinear(Wq, Ws, axis=0, block_size=32) # float32
  Y   = MatMul(X, Wdq)                                  # X is exactly as it was
```

This uses **ONNX opset 21's INT4 tensor type and `DequantizeLinear`'s
`block_size` attribute** -- standard ONNX, not a contrib op like
`com.microsoft::MatMulNBits` -- so, like every other onnxsim quantization
pass, the output loads on any conformant opset-21+ runtime rather than only
onnxruntime builds new enough to ship a particular contrib kernel.

## Why block-wise, and why 32

A single per-channel scale (`quantize_weight_only`'s INT8 scheme) is
dominated by the block's largest-magnitude element; INT4 only has 16 levels
to work with, so that domination costs noticeably more precision than it
does at INT8's 255 levels. Splitting the reduction dimension into blocks and
scaling each one independently keeps every block's own dynamic range tight,
which is what makes INT4 weight-only quantization viable in practice at all
-- this is the same reasoning GPTQ, AWQ, and bitsandbytes' NF4/FP4 block
quantization schemes share, just applied to a plain round-to-nearest
quantizer instead of their calibration-aware ones.

32 is a common default in that literature: small enough to keep quantization
error low, without so many blocks that the scale tensor's own storage
overhead erodes the compression this pass exists to provide (a fixed
constant for now -- see Scope).

## Scope

Handled:
- `MatMul(X, W)` with `W` a constant 2-D float32 tensor whose reduction
  dimension (`K`) is a multiple of 32.
- `Gemm(X, W[, B])` with `transA=0`, `alpha=1`, `beta=1` (when `B` is
  present), same weight constraint. `transB` may be 0 or 1.
- Opsets >= 21.

Left untouched (safe no-op, node passes through as-is):
- A `K` that is not a multiple of 32 -- a ragged last block is left to a
  future extension rather than approximated.
- Non-constant or non-2-D weights, non-default Gemm attributes, non-float32
  operands, or an opset older than 21.
- `Conv` -- not yet covered by this pass (unlike `quantize_weight_only`,
  which does cover it); a natural, still-open follow-up.

## End-to-end: simplify -> quantize -> deploy

### CLI

```bash
onnxsim model.onnx model.quant.onnx --weight-only-quantize-int4
```

### Python API

```python
import onnx
import onnxruntime as ort
import onnxsim

model = onnx.load("model.onnx")
model, check_ok = onnxsim.simplify(model)
assert check_ok

model = onnxsim.quantize_weight_only_int4(model)
onnx.save(model, "model.quant.onnx")

sess = ort.InferenceSession("model.quant.onnx", providers=["CPUExecutionProvider"])
outputs = sess.run(None, {"input": your_input_array})
```

`tests/test_weight_only_quantize_int4.py` runs this simplify -> quantize ->
deploy sequence on small `MatMul`/`Gemm` models, executing both the float and
quantized graphs through `onnxruntime.InferenceSession`.

## Relationship to `quantize_weight_only`

| | Bits | Scale granularity | Typical compression |
|---|---|---|---|
| `quantize_weight_only` | INT8 | one scale per output channel | ~4x |
| `quantize_weight_only_int4` | INT4 | one scale per 32-element block, per output channel | ~7-8x (before the finer scale tensor's own overhead) |

Both leave the activation untouched and need no calibration data; pick INT8
when accuracy headroom matters more than size, and INT4 when the reverse is
true -- exactly the tradeoff real-world weight-heavy deployments (large
embedding/linear layers in transformer-style decoders) make explicitly by
shipping both as separate published checkpoints.
