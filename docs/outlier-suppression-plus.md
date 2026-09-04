# Outlier Suppression+: channel-wise shifting and scaling (`apply_outlier_suppression_plus`)

## What this is

`onnxsim.apply_outlier_suppression_plus` extends `onnxsim.apply_smoothquant`
with a **channel-wise shift**: before migrating quantization difficulty via
SmoothQuant's own scale, it re-centers each activation channel around zero,
using the channel's own calibration min/max midpoint, and folds the shift's
constant contribution back into the layer's output. Like `apply_smoothquant`,
this performs only the migration -- the result is still a float model,
provably equivalent to the input, meant to be fed to a separate W8A8
quantizer (e.g. `quantize_static`/`quantize_qoperator_gemm`) afterwards.

```
Before:
  Y = MatMul(X, W) [+ bias]        -- W constant, [K, N], float32

After:
  Xshift = Sub(X, Z)                -- Z: initializer, float32 [K]
  Xscaled = Mul(Xshift, 1/S)        -- S: initializer, float32 [K]
  Ypre = MatMul(Xscaled, W * S) [+ bias]
  Y = Add(Ypre, Z @ W)              -- Z @ W: initializer, float32 [N]
```

## Where this comes from

[Outlier Suppression+](https://arxiv.org/abs/2304.09145) (Wei et al., 2023,
EMNLP 2023) starts from `apply_smoothquant`'s own migration idea (scaling an
activation channel down and the matching weight row up by the same factor,
an exact no-op on the unquantized function) and observes a gap it leaves
open: SmoothQuant's scale is a single positive multiplier per channel, so it
can shrink a channel's *magnitude* but can never fix a channel that sits
mostly on one side of zero -- an **asymmetric** outlier, common in
post-LayerNorm transformer activations. Scaling a lopsided range by any
positive constant leaves it exactly as lopsided; only a *shift* can recenter
it.

Outlier Suppression+ therefore adds a channel-wise shift ahead of the scale:
for each input channel `j`, `z_j = (max(X_j) + min(X_j)) / 2` -- the
midpoint of that channel's own observed calibration range -- is subtracted
before scaling. This module then reuses `apply_smoothquant`'s own
closed-form scale formula (`s_j = max(|X_j - z_j|) ** alpha / max(|W_j|) **
(1 - alpha)`) on the now-shifted channel. The paper additionally proposes an
iterative grid refinement on top of that closed form to search for a better
scale than the formula alone gives; that refinement is not ported here --
what this module ports is the paper's headline structural contribution over
SmoothQuant (the shift), not its secondary scale-search refinement, matching
`apply_smoothquant`'s own documented practice of using the closed-form scale
rather than searching it.

Shifting an affine layer's input is not a free transformation the way
scaling is: `(X - z) @ W` differs from `X @ W` by exactly `z @ W`, a
*constant* vector (the same `z` every calibration batch converges to,
independent of the specific input at inference time). This module folds that
constant back in via a new `Add` node inserted right after the layer,
reusing the same "rename the producer's output, insert an op that reproduces
the original name" mechanics `onnxsim.correct_bias`'s own `_apply_correction`
helper already uses -- so every existing downstream consumer keeps working
unmodified. Unlike `correct_bias`, what's being folded back in here is an
*exact* algebraic identity from the shift, not an empirically-measured
quantization error: composing the shift, the scale, and this correction
reproduces the original float function exactly (up to floating-point
rounding), for any choice of `z`/`s` at all.

## Scope

Handled:

- `MatMul(X, W)` (2 inputs), or `Gemm(X, W[, B])` with `transA=0`, `alpha=1`,
  and (when `B` is present) `beta=1`. `transB` may be 0 or 1.
- `W` must be a constant 2-D float32 tensor.
- `X` must, when probed on the supplied calibration data, be a plain 2-D
  tensor (`[rows, K]`) whose `K` matches `W`'s reduction dimension.

Left untouched (safe no-op, node passes through as-is):

- Non-constant or non-2-D weights.
- A layer whose activation input never appeared as a plain 2-D tensor across
  the calibration batches (e.g. a 3-D `[batch, seq, hidden]` activation, or
  one that never ran at all).
- A model with no matching layer.

## Usage

```python
import onnx
import onnxsim

model = onnx.load("model.onnx")
migrated = onnxsim.apply_outlier_suppression_plus(model, alpha=0.5)
quantized, check_ok = onnxsim.quantize_static(migrated), True
onnx.save(quantized, "model.osplus.quant.onnx")
```

`calibration_data` defaults to `onnxsim.generate_random_calibration_data`;
pass real representative batches (e.g. via
`onnxsim.load_huggingface_calibration_data`) for a shift and scale that
actually target the model's own real activation distribution, the same
caveat `apply_smoothquant` documents.

## Relationship to `apply_smoothquant`

`apply_outlier_suppression_plus` is a strict superset of what
`apply_smoothquant` does: setting every channel's shift to `z = 0` (e.g. a
symmetric calibration distribution, where `max(X_j) == -min(X_j)`) makes the
two migrations numerically identical except for the extra (all-zero, hence
no-op) `Add` node this module always inserts. The two are not meant to be
composed -- each is a complete, self-contained migration to run ahead of a
W8A8 quantizer; pick whichever best matches the target model's own
activation distribution (a model with mostly symmetric outliers gains
nothing from the extra shift and its `Sub`/`Add` node overhead; one with
pronounced asymmetric outliers benefits from it).
