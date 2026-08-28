# SpQR: outlier-aware block INT4 quantization (`quantize_weight_only_spqr`)

## What this is

`onnxsim.quantize_weight_only_spqr` quantizes a layer's weight to INT4
block-wise like `onnxsim.quantize_weight_only_int4` does, but first pulls
a small fraction of the weight's own most sensitive elements out of each
block's scale computation, storing an exact correction for just those
positions instead of letting them drag the whole block's precision down.

```
Before:
  Y = MatMul(X, W) [+ bias]                  -- W constant, [K, N], float32

After:
  Wq  = <int4, per-(block, column) symmetric, outlier positions
         excluded from each block's own scale>
  Ws  = <float32, [K/block_size, N]>
  Wdq = DequantizeLinear(Wq, Ws, axis=0, block_size=block_size)  -- float32
  zeros = ConstantOfShape([K, N], value=0.0)
  correction = ScatterND(zeros, outlier_indices, outlier_values)  -- float32
  Wreconstructed = Wdq + correction
  Y = MatMul(X, Wreconstructed) [+ bias]
```

A single unusually large weight inside an otherwise ordinary block forces
that block's whole scale up, wasting resolution on every other element
sharing it. Excluding the outlier from the scale computation lets the rest
of the block quantize tighter; the excluded position itself is
reconstructed exactly via a sparse correction, not quantized at all.

## Where this comes from

[SpQR](https://arxiv.org/abs/2306.03078) (Dettmers et al., 2023) observes
that outliers in LLM weights are scattered individual elements throughout
the matrix, not confined to a few channels the way activation outliers
are (the problem `onnxsim.apply_llm_int8`/`onnxsim.apply_smoothquant`
address) -- so excluding them channel-wise doesn't fit; they need to be
found and excluded element-by-element.

SpQR's own reference implementation ranks each weight's importance using
its true contribution to the OBQ/GPTQ objective, computed from the full
inverse-Hessian of the layer's calibration data -- expensive, and (like
GPTQ's own column-by-column update order) not independently verifiable
without re-deriving the same numerically delicate procedure. This module
uses the classical **diagonal-Hessian approximation** to that same
objective instead: for a squared-error objective with Hessian
`H = 2 X^T X`, OBQ's per-weight error contribution
`w_k^2 / [H^-1]_kk` reduces, when `H` is approximated as diagonal, to
`w_k^2 * H_kk = w_k^2 * mean(X[:, k]^2)` -- an ordinary, closed-form
sensitivity score computed directly from the weight and calibration
activations, no matrix inversion involved. The elements with the largest
score (by default the top 1%, tunable via `outlier_fraction`) become the
sparse correction; every other element is quantized normally.

The sparse correction itself is stored efficiently: only the
`num_outliers` outlier `(row, col)` positions and their correction values
are kept (an initializer of shape `[num_outliers, 2]` plus one of shape
`[num_outliers]`), reconstructed at runtime via `ScatterND` into a
`ConstantOfShape`-produced zero tensor -- ordinary ONNX ops, no custom
sparse tensor type or contrib op needed. Every outlier position
reconstructs exactly, since the correction is defined as
`W - block_quantized(W)` at exactly that position, independent of how it
happened to round.

## Scope

Handled:

- `MatMul(X, W)` with `W` a constant 2-D float32 tensor whose reduction
  dimension `K` is divisible by `block_size`, or `Gemm(X, W[, B])` with
  `transA=0`, `alpha=1`, `beta=1` (when `B` is present) under the same
  weight constraint. `transB` may be 0 or 1.
- Opsets >= 21 (INT4's tensor type and `DequantizeLinear`'s `block_size`
  attribute both need opset 21, matching `quantize_weight_only_int4`).
- `outlier_fraction=0.0` (or rounding to zero outliers for a given layer)
  skips the `ScatterND` machinery entirely for that layer, falling back
  to plain block-wise RTN.

Left untouched (safe no-op, node passes through as-is):

- Non-constant weights, non-2-D weights, or a reduction dimension not
  divisible by `block_size`.
- A layer whose activation never appeared in any calibration batch (no
  data to compute a sensitivity score from).
- A model with no matching layer, or an opset older than 21.

## Usage

```python
import onnx
import onnxsim

model = onnx.load("model.onnx")
quantized = onnxsim.quantize_weight_only_spqr(
    model, block_size=16, outlier_fraction=0.01, num_samples=32
)
onnx.save(quantized, "model.spqr.onnx")
```

`calibration_data` defaults to `onnxsim.generate_random_calibration_data`;
pass real representative batches (e.g. via
`onnxsim.load_huggingface_calibration_data`) for a sensitivity ranking
that reflects the model's own real activation statistics, since which
elements count as outliers depends on `mean(X[:, k]^2)`, not just the
weight itself.
