# AdaQuant: joint weight-rounding + activation-clip calibration (`apply_adaquant`)

## What this is

`onnxsim.apply_adaquant` extends `onnxsim.apply_adaround`'s per-layer,
gradient-descent weight-rounding optimization to also optimize the
activation's own quantization range (its scale, and its zero-point, for
asymmetric quantization) at the same time, against the same reconstruction
loss. AdaRound only ever touches the weight side, leaving whatever
calibration (`onnxsim.calibrate`) picked for the activation fixed; AdaQuant
jointly gradient-descends both, on real calibration data, one layer at a
time.

```
Before (produced by onnxsim.quantize_static):
  Xq  = QuantizeLinear(X, Xs, Xzp)        -- Xs/Xzp: CALIBRATED, fixed
  Xdq = DequantizeLinear(Xq, Xs, Xzp)
  Wdq = DequantizeLinear(Wq, Ws, axis=<W's output-channel axis>)
  Y   = MatMul(Xdq, Wdq)                  -- or Gemm

After:
  Xq  = QuantizeLinear(X, Xs', Xzp')      -- Xs'/Xzp': AdaQuant-OPTIMIZED
  Xdq = DequantizeLinear(Xq, Xs', Xzp')
  Wdq = DequantizeLinear(Wq', Ws, axis=<W's output-channel axis>)
  Y   = MatMul(Xdq, Wdq)                  -- or Gemm
```

Only `Xs`, `Xzp`, and `Wq`'s integer codes change -- the graph structure,
`Ws` (the weight's own per-channel scale), and every other tensor are left
exactly as `quantize_static` produced them.

## Where this comes from

[AdaQuant](https://arxiv.org/abs/2006.10518) (Hubara, Nahshan, Hanani,
Banner, Soudry, 2020/2021, "Improving Post Training Neural Quantization:
Layer-wise Calibration and Integer Programming") is the paper that
introduces `onnxsim.apply_adaround`'s closest relative. Nagel et al.'s
AdaRound (ported first, and cited by this paper as related work) optimizes
only the weight's own rounding, via a continuous "rectified sigmoid"
relaxation of each element's floor/ceil choice, against a layer's
reconstruction error (`||W_float @ X - W_quant @ X||^2`) -- leaving the
activation's quantization range exactly what calibration picked. AdaQuant's
own contribution is recognizing that this leaves error on the table for any
scheme that *also* quantizes the activation (W8A8, unlike AdaRound's own
weight-only INT4 target): the loss is a *joint* function of the weight's
rounding and the activation's clip range
(`Xdq = dequantize(quantize(X, scale, zero_point))`), and a rounding choice
optimal against one activation range is not, in general, optimal against a
different one -- a cross-term neither side's isolated optimization can see.
AdaQuant optimizes both by gradient descent on the same loss, still
layer-wise (each layer independently and sequentially, no cross-layer
interaction -- the same scope AdaRound itself uses, and the "layer-wise
calibration" half of the paper's own title).

The paper's title bundles a second, unrelated contribution: an integer-
programming solver that allocates a mixed per-layer bit-width budget across
a whole network. **That half is not ported here** -- it is a distinct,
much larger-scope global optimization problem (a budget-constrained search
over per-layer precision choices for a whole model) with no relationship to
the rounding/clip-range gradient descent this module implements. See
`docs/exl3-quantization-survey.md` for this repository's own precedent of
documenting an explicit "not ported" scope decision rather than silently
under-delivering on a paper's title.

The activation branch's gradients are the standard "straight-through"
quantized-affine gradients (Bengio et al.'s STE, applied to a *learnable*
clip range the way Esser et al. 2019's LSQ does): `round()`'s local
gradient is 1 everywhere it isn't saturated by the surrounding clip, and
the clip itself passes gradient through unchanged inside its range and
blocks it outside -- composed step by step through the actual
quantize-then-dequantize computation, not algebraically simplified first
(naively collapsing `dequantize(quantize(x, s)) ≈ x` under a literal
round-is-identity substitution would erase the very rounding-error signal
the scale's gradient needs). Everything here is plain numpy with
hand-derived gradients, matching `onnxsim.apply_adaround`'s own
"post-hoc adjustment from real activations" style -- no autodiff framework.

## Scope

Handled:

- `MatMul(X, W)` with `W` a constant 2-D float32 tensor, or `Gemm(X, W[, B])`
  with `transA=0`, `alpha=1`, `beta=1` (when `B` is present) under the same
  weight constraint -- `transB` may be 0 or 1 -- quantized by
  `onnxsim.quantize_static` specifically: the QDQ shape shown above, with
  the weight quantized per output channel, symmetric INT8, and the
  activation quantized asymmetric uint8.
- Both the weight's rounding (every element, via the same rectified-sigmoid
  relaxation `apply_adaround` uses) and the activation's `(scale,
  zero_point)` (kept continuous during optimization; the zero-point is
  projected to the nearest integer in `[0, 255]` only once, at the end) are
  optimized jointly.

Left untouched (safe no-op, layer passes through as-is):

- Any layer quantized by a scheme other than `quantize_static`'s QDQ format
  -- including `onnxsim.quantize_qoperator_gemm`'s QGemm (`com.microsoft`
  contrib op) format, which fuses activation quantization into a single
  op instead of leaving it as an addressable `QuantizeLinear`/
  `DequantizeLinear` pair, or any of the weight-only INT4 schemes
  `apply_adaround`/`apply_awq`/`apply_gptq` already target.
  Conv is not handled (`quantize_static` itself only quantizes MatMul and
  "vanilla" Gemm into this QDQ shape; there is no Conv variant of it to
  target).
- A layer whose calibration activation was never captured (no data to
  optimize against), or whose activation isn't a plain 2-D tensor.
- The paper's own integer-programming bit-width allocation -- out of scope
  for this module entirely, see above.
- A model with no matching layer -- returned unchanged.

## Usage

```python
import onnx
import onnxsim

model = onnx.load("model.onnx")
calibration_data = [{"input_name": my_representative_batch}]

quantized = onnxsim.quantize_static(model, calibration_data=calibration_data)
adaquant_model = onnxsim.apply_adaquant(
    model, quantized, calibration_data=calibration_data
)
onnx.save(adaquant_model, "model.adaquant.onnx")
```

`calibration_data` defaults to `onnxsim.generate_random_calibration_data`
for both calls; pass the same real, representative batches to both (e.g.
via `onnxsim.load_huggingface_calibration_data`) so the activation range
`quantize_static` calibrates and the range `apply_adaquant` then refines
are optimized against the same real distribution.
