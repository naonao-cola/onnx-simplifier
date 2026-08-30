# Mixed-precision quantization (`apply_mixed_precision_quantization`)

## What this is

`onnxsim.apply_mixed_precision_quantization` assigns **different bit
widths to different layers** within one model, chosen from a
calibration-driven sensitivity score, rather than applying one uniform
scheme everywhere the way every other onnxsim quantizer does.

```
Before:
  Y = MatMul(X, W) [+ bias]    -- one of many layers, W constant, float32

After (per layer, chosen independently):
  Whigh_hat = DequantizeLinear(Wq_int8, Ws, axis=0, block_size=32)  -- sensitive layers
  Wlow_hat  = DequantizeLinear(Wq_int4, Ws, axis=0, block_size=32)  -- everything else
  Y = MatMul(X, W*_hat) [+ bias]
```

## Where this comes from

`onnxsim.quantize_weight_only_int4` (and everything built on it) applies
block-wise INT4 to *every* matched layer; `onnxsim.recommend_quantization`
searches across *global* schemes (try INT4-everywhere, then
INT8-everywhere, ...) and returns whichever single one meets an accuracy
budget. Neither assigns different bit widths to different layers within
the *same* model -- which leaves real compression on the table: in any
real network, some layers are far more sensitive to quantization error
than others (the premise behind the mixed-precision/bit-width-search
literature -- e.g. HAQ, and the per-layer outlier analysis behind
`onnxsim.llm_int8`/`onnxsim.spqr`), so spending the same number of bits on
every layer either wastes precision on layers that tolerate INT4 fine, or
loses too much on the few layers that don't.

This module is deliberately not a new *algorithm* the way
`apply_spinquant`/`apply_duquant` are -- it is a dispatcher over two
schemes onnxsim already has (block-wise INT4 and, for the most sensitive
layers, block-wise INT8), choosing which one each layer gets from a
data-driven sensitivity score:

```
mse = mean((W - INT4_dequant(W))^2)     -- INT4's own reconstruction error
sensitivity = mse * mean(X^2)           -- scaled by typical input magnitude
```

`mean(X^2)` is the same per-layer activation-energy signal
`apply_duquant`'s own sensitivity ranking is built on (there, per-channel;
here, a single per-layer scalar, since the decision being made -- INT4 vs
INT8 -- is per-layer, not per-channel). Layers are ranked by this score;
the top `high_bits_fraction` (by count, most sensitive first) get
block-wise INT8; every other layer gets ordinary block-wise INT4.

## Scope

Handled:

- `MatMul(X, W)` with `W` a constant 2-D float32 tensor whose reduction
  dimension `K` is divisible by `block_size`, or `Gemm(X, W[, B])` with
  `transA=0`, `alpha=1`, `beta=1` (when `B` is present) under the same
  weight constraint. `transB` may be 0 or 1.
- Opsets >= 21 (INT4's tensor type and `DequantizeLinear`'s `block_size`
  attribute both need opset 21).
- `high_bits_fraction=0.0` quantizes every layer to INT4 (matching
  `quantize_weight_only_int4`'s own behavior); `1.0` quantizes every layer
  to INT8.

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
quantized = onnxsim.apply_mixed_precision_quantization(
    model, high_bits_fraction=0.2, num_samples=32
)
onnx.save(quantized, "model.mixedprec.onnx")
```

`calibration_data` defaults to `onnxsim.generate_random_calibration_data`;
pass real representative batches (e.g. via
`onnxsim.load_huggingface_calibration_data`) for a sensitivity ranking
that reflects the model's own real activation statistics, since the
INT8/INT4 split depends on `mean(X^2)`, not just the weight itself.
