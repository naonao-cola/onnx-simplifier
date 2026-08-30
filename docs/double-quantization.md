# Double quantization (`apply_double_quantization`)

## What this is

`onnxsim.apply_double_quantization` is a second pass over an
**already-quantized** onnx model: it finds every `DequantizeLinear` node
whose scale is a sizable constant tensor (a per-block or per-channel
scale, as every onnxsim block-wise INT4 scheme produces) and quantizes
*that scale tensor itself* to UINT8, reconstructing it in-graph via a
second, nested `DequantizeLinear`.

```
Before:
  Whatever_hat = DequantizeLinear(Codes, Scale, ...)  -- Scale: float32, [blocks, N]

After:
  ScaleCodes: initializer, uint8, [blocks, N]
  MetaScale: initializer, float32 scalar
  ScaleHat = DequantizeLinear(ScaleCodes, MetaScale)
  Whatever_hat = DequantizeLinear(Codes, ScaleHat, ...)  -- same attributes as before
```

A block of 32 INT4 codes needs 16 bytes of codes plus 4 bytes of its own
float32 scale -- a 25% overhead on top of the codes themselves. Since
those scale values are already absmax-normalized magnitudes (a much
milder dynamic range than raw weights), quantizing them too trades a
little more precision for real additional memory savings, without
touching the original low-bit codes at all.

## Where this comes from

[QLoRA](https://arxiv.org/abs/2305.14314) (Dettmers et al., 2023, Section
3.2) introduces double quantization as part of its own 4-bit NormalFloat
scheme: after block-wise quantizing the weight, it quantizes the
per-block scale factors themselves with an 8-bit quantizer (with its own
observation that this saves roughly 0.37 bits/parameter on average across
a whole model). This module reproduces that second-level idea directly,
generically, as a standalone pass over any model's existing
`DequantizeLinear` nodes -- it doesn't know or care which onnxsim module
produced the outer node (or its `axis`/`block_size` attributes, left
untouched), only that its scale input is a sufficiently large constant
tensor. That makes it compose with every block-wise scheme already in
onnxsim (`quantize_weight_only_int4`, `onnxsim.nf4`, `onnxsim.spqr`,
`onnxsim.spinquant`, `onnxsim.quarot`, `onnxsim.quip_sharp`) unchanged,
and with future ones with no new code -- unlike every other module in
this package, `apply_double_quantization` has no live weights of its own
to quantize; it only ever post-processes another quantizer's own output.

Since scale values are always non-negative (absmax-derived magnitudes),
the inner quantizer uses a plain unsigned `0..255` range with no
zero-point offset (`codes = round(scale / meta_scale)`,
`meta_scale = max(scale) / 255`) rather than the signed, symmetric ranges
every other onnxsim INT4 scheme uses for weights -- the two quantization
problems (weights, which are signed and roughly zero-centered; scales,
which are strictly positive) call for different ranges.

## Scope

Handled:

- Any `DequantizeLinear` node whose scale input (`node.input[1]`) is a
  constant float32 initializer with at least `min_elements` values
  (default 64) -- regardless of which module produced it, or what its
  `axis`/`block_size` attributes are.

Left untouched (safe no-op, node passes through as-is):

- A scale smaller than `min_elements` (e.g. a single per-tensor scalar):
  a second quantizer's own overhead (a meta-scale initializer plus a new
  node) would cost more than it saves.
- A scale that is not a constant initializer -- e.g. a dynamically
  computed scale, like `quantize_kv_cache`'s Value-style per-token scale
  stream, which is a graph input/output, not a constant.
- A model with no `DequantizeLinear` node at all.

The original float32 scale initializer is left in the graph, unreferenced
(matching every other onnxsim rewrite, which leaves its own now-unused
original tensor in place) -- pair this with `onnxsim.simplify()` (or any
generic dead-initializer elimination) afterward to actually reclaim the
storage the original float32 scale used.

## Usage

```python
import onnx
import onnxsim

model = onnx.load("model.onnx")
quantized = onnxsim.apply_quarot(model)          # or apply_spinquant,
                                                   # quantize_weight_only_spqr,
                                                   # quantize_weight_only_int4, ...
doubly_quantized = onnxsim.apply_double_quantization(quantized)
onnx.save(doubly_quantized, "model.doubleq.onnx")
```

Run this as the *last* step after any other quantizer -- it operates on
`DequantizeLinear` nodes that must already exist in the graph.
