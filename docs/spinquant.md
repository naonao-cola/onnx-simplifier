# SpinQuant: learned-rotation weight quantization (`apply_spinquant`)

## What this is

`onnxsim.apply_spinquant` applies a *calibrated* rotation to a layer's
weight before quantizing it to INT4, then reconstructs it in-graph via the
same block-wise scheme `onnxsim.quantize_weight_only_int4` already uses.

```
Before:
  Y = MatMul(X, W) [+ bias]                 -- W constant, [K, N], float32

After:
  U: initializer, float32 [K, K]            -- the fitted rotation
  X_rotated = MatMul(X, U)
  Wtilde_hat = DequantizeLinear(Wtilde_q, Wtilde_s, axis=0, block_size=32)
                                              -- INT4 codes, [K, N]
  Y = MatMul(X_rotated, Wtilde_hat) [+ bias]
```

Rotating by an orthogonal matrix before quantizing is lossless on its own
(`U @ U.T == I`) -- only the block-wise INT4 step after it loses any
precision, exactly as much as `quantize_weight_only_int4` already does on
an unrotated weight. The point of the rotation is that it changes *which*
weight gets quantized: one whose values are spread more evenly across
directions, rather than concentrated in a few outlier ones, quantizes with
less error at the same bit width.

## Where this comes from

[SpinQuant](https://arxiv.org/abs/2405.16406) (Liu et al., 2024) shares its
core idea with `onnxsim.apply_quip_sharp` -- conjugating a weight by an
orthogonal rotation before quantizing it removes outlier directions a
uniform grid would otherwise waste precision on. QuIP#'s own rotation is
*random* (a concentration-of-measure argument: any fixed vector, rotated
by a uniformly random orthogonal matrix, spreads out evenly with high
probability). SpinQuant's contribution is that the rotation doesn't have
to be random -- a rotation *fit to the data* can target the real
distribution's own outlier directions directly, rather than relying on a
probabilistic argument that ignores that structure.

SpinQuant's own reference implementation learns (typically four) rotation
matrices per layer via gradient descent against the quantized model's own
end-to-end loss, constrained to the orthogonal group via a Cayley-manifold
optimizer -- calibration-aware, differentiable-quantization machinery with
no ONNX export path, and not independently verifiable the way a
closed-form procedure is. `apply_spinquant` instead reproduces SpinQuant's
own "R1-only" ablation (its own simplified configuration, reported to
capture most of the improvement over no rotation at all) via a classical
substitute: fit a single input-side rotation per layer as the eigenvector
basis of that layer's own calibration-activation covariance matrix (an
ordinary symmetric eigendecomposition, `numpy.linalg.eigh`) -- the
closed-form answer to "which rotation makes this data's second-moment
structure as close to isotropic as possible," the same effect a learned
rotation is chasing, without an unverifiable optimization loop.

Unlike QuIP#'s rotation, this needs calibration data (the whole point is
to target the real distribution's own structure); unlike QuIP#, it needs
no second, output-side rotation or non-uniform lattice codebook --
SpinQuant's contribution is specifically about the rotation being
*learned* rather than random, not a different quantization backend, so
this module pairs it with the same block-wise RTN backend
`quantize_weight_only_int4` already uses.

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
  data to fit a rotation from).
- A model with no matching layer, or an opset older than 21.

## Usage

```python
import onnx
import onnxsim

model = onnx.load("model.onnx")
quantized = onnxsim.apply_spinquant(model, num_samples=32, block_size=32)
onnx.save(quantized, "model.spinquant.onnx")
```

`calibration_data` defaults to `onnxsim.generate_random_calibration_data`;
pass real representative batches (e.g. via
`onnxsim.load_huggingface_calibration_data`) for a rotation that actually
targets the model's own real activation structure, since the whole benefit
over QuIP#'s random rotation depends on the calibration data being
representative.
