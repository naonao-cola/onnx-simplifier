# Block-wise INT8 weight-only quantization (`quantize_weight_only_int8_block`)

## What this is

`onnxsim.quantize_weight_only_int8_block` is a pair of single, self-contained
C++ graph rewrites (`onnxsim/passes/weight_only_quantize_int8_block_matmul.h`,
`onnxsim/passes/weight_only_quantize_int8_block_conv.h`) that block-wise
quantize every `MatMul`, every "vanilla" `Gemm` (`transA=0`, `alpha=1`,
`beta=1`), and every `Conv`, whose weight is a constant float32 tensor whose
flattened reduction size is evenly divisible by 32 -- `K` for MatMul/Gemm;
`Cin/groups * prod(kernel dims)` for Conv, exactly the same "flatten Conv's
non-output-channel axes into one sequence" scheme
`quantize_weight_only_int4` uses (see that doc's "Conv's flattened
reduction" section -- it applies identically here).

It sits between `quantize_weight_only` (INT8, one scale per output channel)
and `quantize_weight_only_int4` (INT4, one scale per 32-element block): the
**same INT8 bit width** as the former, at the **same block granularity** as
the latter.

- The **weight** is quantized to INT8 (values in `[-127, 127]`), with a
  **separate symmetric scale per 32-element block of the reduction, per
  output channel** -- identical granularity to `quantize_weight_only_int4`,
  just with INT8's much wider 255-level code range instead of INT4's 15.
- The **activation is never touched** -- same as every other weight-only
  pass here: no calibration data, no runtime quantize/dequantize cost on
  the activation path.

```
Before:
  Y = MatMul(X, W)                                    # W constant, [K, N], float32

After:
  Wq  = <int8, per-(block, column) symmetric>          # computed once, here
  Ws  = <float32, [K/32, N]>                            # one scale per block per column
  Wdq = DequantizeLinear(Wq, Ws, axis=0, block_size=32) # float32
  Y   = MatMul(X, Wdq)                                  # X is exactly as it was
```

This uses **ONNX opset 21's `DequantizeLinear` `block_size` attribute** --
standard ONNX, not a contrib op -- the same requirement
`quantize_weight_only_int4` has, even though plain INT8 itself is much
older (opset 13 is enough for `quantize_weight_only`'s per-channel scheme).

## Why this exists alongside the per-channel and INT4 schemes

`quantize_weight_only`'s single per-channel scale is set by that channel's
single largest-magnitude element; a channel with a wide dynamic range (a few
much-larger-magnitude weights than the rest) under-resolves its
smaller-magnitude weights as a result -- the same effective-resolution
problem `docs/int16-quantization.md` describes for `quantize_weight_only_int16`.
Block-wise scales fix this a different way: instead of widening the *type*
(INT16), they narrow the *scope* each scale has to cover, so an outlier in
one block no longer drags down every other block's resolution.

Doing this at INT8 rather than INT4 keeps the per-element code range wide
(255 levels vs. 15), so the *only* change from `quantize_weight_only` is
where the resolution loss can hide -- not how much of it there is to begin
with. The cost is almost entirely in the scale tensor: INT8 codes are still
1 byte each either way, so `quantize_weight_only_int8_block`'s total size is
only marginally larger than `quantize_weight_only`'s (one extra float per
block instead of per channel) for meaningfully better resolution on
outlier-heavy channels.

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
  left to a future extension rather than approximated, matching
  `quantize_weight_only_int4`.
- Non-constant or unsupported-rank weights, non-default Gemm attributes,
  non-float32 operands, or an opset older than 21.

## End-to-end: simplify -> quantize -> deploy

### CLI

```bash
onnxsim model.onnx model.quant.onnx --weight-only-quantize-int8-block
```

### Python API

```python
import onnx
import onnxruntime as ort
import onnxsim

model = onnx.load("model.onnx")
model, check_ok = onnxsim.simplify(model)
assert check_ok

model = onnxsim.quantize_weight_only_int8_block(model)
onnx.save(model, "model.quant.onnx")

sess = ort.InferenceSession("model.quant.onnx", providers=["CPUExecutionProvider"])
outputs = sess.run(None, {"input": your_input_array})
```

`tests/test_weight_only_quantize_int8_block.py` runs this simplify ->
quantize -> deploy sequence on small `MatMul`/`Gemm`/`Conv` models, executing
both the float and quantized graphs through `onnxruntime.InferenceSession`,
plus a direct comparison against `quantize_weight_only`'s per-channel scheme
on an outlier-heavy weight confirming the block-wise scheme resolves it more
closely.

## Relationship to onnxsim's other weight-only schemes

| | Bits | Scale granularity | Typical compression |
|---|---|---|---|
| `quantize_weight_only` | INT8 | one scale per output channel | ~4x |
| `quantize_weight_only_int8_block` | INT8 | one scale per 32-element block, per output channel | ~4x (marginally more scale-tensor overhead) |
| `quantize_weight_only_int16` | INT16 | one scale per output channel | ~2x |
| `quantize_weight_only_int4` | INT4 | one scale per 32-element block, per output channel | ~7-8x |

All four leave the activation untouched and need no calibration data. Pick
`quantize_weight_only_int8_block` over plain `quantize_weight_only` when a
weight has outlier-heavy channels but INT4's narrower code range (or
INT16's larger footprint) isn't the right tradeoff -- it keeps INT8's
storage and accuracy profile while fixing the specific resolution problem a
single per-channel scale has on that kind of weight.
