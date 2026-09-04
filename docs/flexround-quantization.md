# FlexRound: learnable division-based rounding (`apply_flexround`)

## What this is

`onnxsim.apply_flexround` optimizes, per weight element, *which* integer a
`quantize_weight_only_int4`-quantized `MatMul`/`Gemm` layer's weight rounds
to -- not by perturbing the rounding additively (as
`onnxsim.apply_adaround` does), but by reparametrizing the quantization
*divisor* itself and optimizing it by gradient descent against real
calibration activations.

```
Before (round-to-nearest, what quantize_weight_only_int4 already produced):
  code = round(W / scale)                    -- scale: fixed per (block, output channel)

After (FlexRound):
  S    = scale * S2 * s3                     -- S2: learned, one per weight element
                                                  s3: learned, one per output channel
  code = round(W / S)                        -- same scale multiplies the stored code at
                                                  deploy time; only which integer changes
```

`S2` and `s3` are both initialized to 1, so before any optimization this is
exactly the round-to-nearest `quantize_weight_only_int4` already produced;
gradient descent (Adam, straight-through through `round()`) then moves them
to minimize `||W X - W_hat X||^2` on real activations captured from the
float model -- the identical reconstruction-error objective
`onnxsim.apply_adaround`/`onnxsim.apply_gptq` use, just reached via a
different reparametrization of the rounding decision.

## Where this comes from

[FlexRound](https://arxiv.org/abs/2306.00317) (Lee et al., 2023, ICML 2023)
observes that every prior learnable-rounding PTQ method -- including
AdaRound (Nagel et al., 2020, `onnxsim.apply_adaround`'s own source) --
reparametrizes rounding via **element-wise addition**: a learned
perturbation `delta`, squashed into a fixed range (roughly `[-0.5, 1.5]`),
is added before rounding. That fixed range costs a large-magnitude weight
the same absolute nudge as a small one. FlexRound proposes **element-wise
division** instead: dividing `W` by a *learned* divisor `S` rather than by
the original fixed `scale`. Because `d(W/S)/dS` is proportional to `W`
itself, a straight-through gradient through this reparametrization
naturally gives large-magnitude weights a proportionally larger nudge and
small-magnitude ones a proportionally smaller one -- the paper's own
argument for why this scales better to weight distributions with heavy
tails or large outliers than a fixed-range additive perturbation does.

The paper's own decomposition of the divisor (Eq. 2, linear-layer case) is
`S = s1 (x) S2 (x) s3`, where `s1` is the quantization grid size itself
(jointly learned in the paper), `S2` is an element-wise correction the same
shape as `W`, and `s3` is a per-output-channel correction. This module
implements `S2` and `s3` exactly as the paper does, but -- like every other
onnxsim PTQ pass targeting `quantize_weight_only_int4`'s output -- keeps
`s1` fixed at the block scale the quantized model already committed to the
graph, rather than optimizing it: this pass only ever rewrites *which
integer* each element rounds to, never the scale tensor itself. See
`onnxsim/flexround.py`'s own module docstring for the full list of what's
ported versus simplified relative to the paper (positivity constraint,
no `s4` term, no activation quantization, no annealing schedule).

## Relationship to the other three PTQ techniques

Four onnxsim-native PTQ techniques all target the exact same scheme
(`quantize_weight_only_int4`'s block-wise symmetric INT4), each pulling a
different lever:

| Technique | Lever | Mechanism |
|---|---|---|
| `apply_adaround` | additive, per-element | learns a per-element perturbation `delta`, squashed through a rectified sigmoid, annealed toward a hard floor/ceil decision |
| `apply_gptq` | sequential, Hessian-based | quantizes input channels one at a time, propagating each one's rounding error into every not-yet-quantized channel |
| `apply_awq` | per-input-channel rescale | rescales whole input channels by their own activation saliency before quantizing, with a compensating `Mul` on the activation |
| `apply_flexround` | multiplicative, per-element | learns a per-element (and per-output-channel) *divisor* correction, optimized by gradient descent |

All four leave `quantize_weight_only_int4`'s scale tensor and graph
structure untouched (`apply_awq`'s inserted `Mul` node is the only
exception, and even then only when it measurably helps) and only rewrite
the INT4 codes -- so they can be applied in any order, or compared directly
against each other on the same calibration data, without needing to
re-quantize from scratch.

## Scope

Handled:

- `MatMul(X, W)` / `Gemm(X, W[, B])` (`transA=0`, `alpha=1`, `beta=1` when
  `B` is present; `transB` may be 0 or 1) already quantized by
  `quantize_weight_only_int4` (`DequantizeLinear(Wq, Ws, axis=..., block_size=...)`
  feeding the same node, matched by node output name between `float_model`
  and `quantized_model`).
- A 2-D, `float32` activation input captured from real calibration data.

Left untouched (safe no-op):

- Layers quantized by any other scheme, or left unquantized.
- A layer whose activation input isn't a plain 2-D tensor (batched or
  broadcast MatMul).
- A model with no matching layer at all -- `quantized_model` is returned
  unchanged.

## Usage

```python
import onnx
import onnxsim

model = onnx.load("model.onnx")
quant = onnxsim.quantize_weight_only_int4(model)

flexround_model = onnxsim.apply_flexround(
    model, quant, calibration_data=None,  # random calibration data by default
)
onnx.save(flexround_model, "model.flexround.onnx")
```

`calibration_data` defaults to `onnxsim.generate_random_calibration_data`;
pass real representative batches (e.g. via
`onnxsim.load_huggingface_calibration_data`) for a reconstruction target
that reflects the model's own real activation distribution -- as with
`apply_adaround`/`apply_gptq`/`apply_awq`, the whole benefit over plain
round-to-nearest depends on the calibration data being representative.

`tests/test_flexround.py` runs this simplify -> quantize -> optimize ->
deploy sequence end-to-end through `onnxruntime.InferenceSession`, and
confirms FlexRound measurably reduces reconstruction error over plain
round-to-nearest on a calibration set with real cross-element structure --
like AdaRound, FlexRound only ever changes *which* integer an element
rounds to at an already-fixed block scale, so it cannot rescue a block
whose scale is itself dominated by a single outlier (no per-element
divisor correction changes which integer a value many orders of magnitude
below the block's own scale rounds to); the gain instead comes from
exploiting genuine cross-element correlation in the calibration batch, the
same source AdaRound's own reconstruction-error test relies on.
