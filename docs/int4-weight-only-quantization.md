# Block-wise INT4 weight-only quantization (`quantize_weight_only_int4`)

## What this is

`onnxsim.quantize_weight_only_int4` is a pair of single, self-contained C++
graph rewrites (`onnxsim/passes/weight_only_quantize_int4_matmul.h`,
`onnxsim/passes/weight_only_quantize_int4_conv.h`) that block-wise quantize
every `MatMul`, every "vanilla" `Gemm` (`transA=0`, `alpha=1`, `beta=1`), and
every `Conv`, whose weight is a constant float32 tensor whose flattened
reduction size is evenly divisible by 32 -- `K` for MatMul/Gemm; `Cin/groups
* prod(kernel dims)` for Conv, since Conv has no single reduction axis the
way MatMul does (see "Conv's flattened reduction" below).

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

## Conv's flattened reduction

Conv's weight (`[Cout, Cin/groups, k...]`) has no single axis that plays
`K`'s role the way MatMul/Gemm's weight does -- every one of `Cin/groups`
and the kernel's spatial dims contributes to one output pixel's sum.
Blocking one of Conv's own axes directly (say, `Cin/groups`) would put an
*independent* scale on every kernel spatial position, which for a 3x3 kernel
means 9x the scale-tensor overhead for little accuracy benefit, since each
of those 9 positions would get its own scale regardless of block size.

Instead, `weight_only_quantize_int4_conv.h` flattens everything but the
output channel into one sequence (`inner = Cin/groups * prod(k...)`) and
blocks *that*, exactly as MatMul's `K` is blocked:

```
Before:
  Y = Conv(X, W)              # W constant, [Cout, Cin/groups, k...], float32

After:
  Wq_flat  = <int4, [Cout, inner]>                        # inner = Cin/groups * prod(k...)
  Ws_flat  = <float32, [Cout, inner / 32]>
  Wdq_flat = DequantizeLinear(Wq_flat, Ws_flat, axis=1, block_size=32)
  Wdq      = Reshape(Wdq_flat, [Cout, Cin/groups, k...])   # restore Conv's expected shape
  Y        = Conv(X, Wdq)
```

The extra `Reshape` is the only structural difference from the MatMul/Gemm
rewrite; the quantization scheme itself (symmetric INT4, block-local scale)
is identical.

## Scope

Handled:
- `MatMul(X, W)` with `W` a constant 2-D float32 tensor whose reduction
  dimension (`K`) is a multiple of 32.
- `Gemm(X, W[, B])` with `transA=0`, `alpha=1`, `beta=1` (when `B` is
  present), same weight constraint. `transB` may be 0 or 1.
- `Conv(X, W[, B])` with `W` a constant float32 tensor, rank >= 3, whose
  flattened `Cin/groups * prod(k...)` is a multiple of 32.
- Opsets >= 21.

Left untouched (safe no-op, node passes through as-is):
- A reduction size that is not a multiple of 32 -- a ragged last block is
  left to a future extension rather than approximated.
- Non-constant or unsupported-rank weights, non-default Gemm attributes,
  non-float32 operands, or an opset older than 21.

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
deploy sequence on small `MatMul`/`Gemm`/`Conv` models, executing both the
float and quantized graphs through `onnxruntime.InferenceSession`.

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
