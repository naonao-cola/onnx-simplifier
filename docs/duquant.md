# DuQuant: calibrated permutation W4A4 quantization (`apply_duquant`)

## What this is

`onnxsim.apply_duquant` extends `onnxsim.apply_quarot`'s own random
rotation with a calibrated **permutation**: rather than trusting a single
random rotation draw to spread every outlier evenly, it uses calibration
data to find a layer's own worst outlier channels and explicitly
redistributes them one-per-block across the quantization grouping, then
applies an independent random rotation *within* each block. Like
`apply_quarot`, both the weight and the activation end up INT4.

```
Before:
  Y = MatMul(X, W) [+ bias]                 -- W constant, [K, N], float32

After:
  U: initializer, float32 [K, K]      -- permutation + block-local rotation
  Xrot = MatMul(X, U)                 -- runtime activation rotation
  Xq   = round_to_nearest_int4_per_token(Xrot)   -- data-free, no calibration
  Wtilde_hat = DequantizeLinear(Wtilde_q, Wtilde_s, axis=0, block_size=32)
                                              -- INT4 codes, [K, N]
  Y = MatMul(Xq, Wtilde_hat) [+ bias]
```

This is the exact same "after" graph shape `apply_quarot` produces -- only
how `U` is *built* differs.

## Where this comes from

[DuQuant](https://arxiv.org/abs/2406.01721) (Lin et al., 2024, NeurIPS
2024) starts from the same observation `apply_quarot` already acts on: a
random rotation removes outlier directions with high probability (the
concentration-of-measure argument `apply_quip_sharp` also relies on).
DuQuant's own motivation is a specific failure mode of that approach:
some LLMs have a handful of **massive-activation channels**, concentrated
so heavily that a *single* random rotation draw isn't guaranteed to
spread them out (the concentration argument is a high-probability
statement over the *choice* of rotation, not a guarantee for any one
specific draw) -- so quantizing right after a random rotation can still
leave those specific channels dominating whichever block they land in.

DuQuant's own fix has two stages: **rotate**, then **permute** using the
calibration data's own per-channel activation magnitude to redistribute
whichever channels are still the worst offenders, one-per-block, so no
single block bears more than its fair share of surviving outlier energy.
DuQuant's own reference implementation builds this via a greedy, iterative
algorithm pairing each outlier channel with a partner via a 2-D Givens
rotation -- a bespoke optimization procedure that isn't independently
verifiable the way a closed-form construction is (the same reason
`apply_spinquant` substitutes a closed-form eigenbasis for SpinQuant's own
learned rotation). This module instead builds the same two-stage effect
from two classical, verifiable pieces:

- **Permutation**: rank a layer's input channels by their own calibration
  abs-max magnitude, then greedily assign the highest-magnitude channels,
  one at a time, to whichever quantization block currently holds the
  least outlier magnitude -- an ordinary permutation matrix (itself
  orthogonal by construction).
- **Block-local random rotation**: after permutation, an independent
  Haar-random orthogonal rotation (`apply_quip_sharp`'s own
  `_random_orthogonal_matrix`) *within* each block -- the same spreading
  `apply_quarot` relies on globally, but applied locally, after the worst
  channels have already been separated from each other.

The composition of a permutation matrix and a block-diagonal orthogonal
matrix is itself orthogonal, so this module reuses `apply_quarot`'s own
graph-construction machinery verbatim (the weight rotated and block-INT4
-quantized offline via `onnxsim.omniquant`'s
`_quantize_blockwise_int4_with_clip`; the activation rotated and
INT4-quantized per token at graph-run time) -- only `U`'s construction
differs. Unlike `apply_quarot`, this needs calibration data: the whole
point is to target the *specific* channels the real activation
distribution concentrates outliers in.

## Scope

Handled:

- `MatMul(X, W)` with `W` a constant 2-D float32 tensor whose reduction
  dimension `K` is divisible by `block_size`, or `Gemm(X, W[, B])` with
  `transA=0`, `alpha=1`, `beta=1` (when `B` is present) under the same
  weight constraint. `transB` may be 0 or 1.
- Opsets >= 21 (INT4's tensor type and `DequantizeLinear`'s `block_size`
  attribute both need opset 21, matching `quantize_weight_only_int4`).

Left untouched (safe no-op, node passes through as-is):

- Non-constant weights, non-2-D weights, or a reduction dimension not
  divisible by `block_size`.
- A layer whose activation never appeared in any calibration batch (no
  data to rank channels by).
- A model with no matching layer, or an opset older than 21.

## Usage

```python
import onnx
import onnxsim

model = onnx.load("model.onnx")
quantized = onnxsim.apply_duquant(
    model, block_size=32, outlier_fraction=0.05, num_samples=32
)
onnx.save(quantized, "model.duquant.onnx")
```

`calibration_data` defaults to `onnxsim.generate_random_calibration_data`;
pass real representative batches (e.g. via
`onnxsim.load_huggingface_calibration_data`) for a permutation that
actually targets the model's own real outlier channels, since the whole
benefit over `apply_quarot`'s random rotation depends on the calibration
data being representative.
