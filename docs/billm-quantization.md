# BiLLM: 1-bit-average weight binarization (`quantize_weight_only_billm`)

## What this is

`onnxsim.quantize_weight_only_billm` is a genuine **lossy PTQ quantizer**,
not a rounding-refinement lever like `onnxsim.apply_gptq`/`apply_awq`/
`apply_adaround`/`apply_flexround` (each of those takes a model that's
*already* been `quantize_weight_only_int4`-quantized and only changes which
grid point each element rounds to). This module instead takes an ordinary
dense float32 MatMul/Gemm layer straight out of the float model and pushes
it down to close to 1 bit/element on average -- the same family as
`quantize_weight_only_int4`/`quantize_weight_only_nf4`/
`quantize_weight_only_kmeans`, at the most extreme end of that family's
bit-width range.

It is also a different problem from `onnxsim.quantize_ternary` (see
`docs/ternary-quantization.md`): that pass only **detects** a weight that is
*already*, structurally, exactly `{-s, 0, +s}` -- a lossless rewrite of a
BitNet-family model someone else already trained ternary -- and leaves an
ordinary dense float32 weight completely untouched. BiLLM does the opposite:
it **binarizes** an ordinary dense float32 weight, lossily, on purpose,
trading real accuracy for close to 16x the storage reduction of INT4.

```
Before:
  Y = MatMul(X, W) [+ bias]        -- W constant, [K, N], float32

After:
  Code1: initializer, int8, [K, N]  -- sign(W), values in {-1, +1}
  Code2: initializer, int8, [K, N]  -- sign of the salient residual,
                                        values in {-1, 0, +1} (0 for every
                                        non-salient column)
  Scale1: initializer, float32, [K, 1]   -- per-column level-1 scale
  Scale2: initializer, float32, [K, 1]   -- per-column level-2 scale,
                                             0 for every non-salient column
  What_hat = Add(Mul(Cast(Code1, float), Scale1),
                 Mul(Cast(Code2, float), Scale2))
  Y = MatMul(X, What_hat) [+ bias]
```

No `Gather`/codebook lookup, unlike `quantize_weight_only_nf4`/
`quantize_weight_only_kmeans`: the "codebook" here is just `{-1, +1}`, so a
code cast to float already *is* the sign -- multiplying by the per-column
scale directly reconstructs the weight. Ordinary ONNX ops only
(`Cast`/`Mul`/`Add`), opset 11+.

## Where this comes from

[BiLLM](https://arxiv.org/abs/2402.04291) (Huang et al., 2024, ICML 2024)
observes that a pretrained LLM's per-weight Hessian sensitivity is
long-tailed (a small fraction of weights, concentrated in whole columns of
attention-projection-style layers, dominate a layer's output) while the
weight *magnitudes* themselves are bell-shaped and mostly redundant. It
exploits this with two different binarization strategies for two groups of
weights, both derived per calibration-block from real activations:

1. **Structured salient-column selection** (paper Section 3.1): a
   Hessian-based sensitivity metric, `s_i = w_i^2 / [H_c]_ii^2` (`H_c` the
   damped-Hessian-inverse Cholesky factor GPTQ's own reformulation already
   computes -- this module reuses `onnxsim.gptq._inverse_hessian_cholesky`
   directly), ranks a block's columns; a bounded search over how many
   leading columns to call "salient" picks the count that minimizes
   plain-binary reconstruction error for that block.
2. **Binary residual approximation for salient columns**: rather than
   keeping salient columns at full precision (which would blow the ~1-bit
   budget), BiLLM binarizes them *twice* -- `B1 = sign(W) * mean(|W|)`,
   then binarizes the residual `R = W - B1` the same way
   (`B2 = sign(R) * mean(|R|)`) -- giving `W ~= B1 + B2` at ~2 bits for a
   small fraction of columns, which the paper proves strictly reduces
   error versus stopping at `B1` alone.
3. **Plain flat binary for non-salient columns**: `sign(W) * mean(|W|)`,
   one scalar scale per block.
4. **Block-wise OBC/GPTQ-style error compensation**: whatever a block's
   chosen binarization couldn't represent is charged forward into
   not-yet-processed columns via `H_c`'s own off-diagonal structure --
   the same second-order argument `onnxsim.gptq` already implements.

This module ports steps 1, 2, and 4 **faithfully** (per-column Hessian
sensitivity, the bounded salient-count search over the paper's own stated
3-30 range, the exact two-level residual-approximation formula, and
GPTQ-style block-wise error compensation reusing `onnxsim.gptq`'s own
Cholesky helper). Step 3 is a **deliberate, documented simplification**:
the paper's own non-salient path (Section 3.2, "Bell-shaped Distribution
Splitting") further splits non-salient weights *elementwise* by magnitude
into a "concentrated" and "sparse" region, each with its own scale chosen
by a percentile search. This module instead applies one flat scale to the
whole non-salient sub-block. Two reasons: the paper's own results (Table 1)
show this second split contributes far less than the salient/residual
mechanism to overall accuracy -- it is not the headline result -- and an
elementwise split has no clean per-column broadcast shape the way every
other quantity in this module's encoding does (it would force a
per-*element* rather than a compact per-*column* scale, breaking the
uniform decode graph above). See `onnxsim/billm.py`'s own module docstring
for the full derivation and every other implementation decision.

## Scope

Handled:
- `MatMul(X, W)` / `Gemm(X, W[, B], transA=0, alpha=1, beta=1)` with `W` a
  constant 2-D float32 tensor and `X` a plain 2-D activation that appears
  (as a 2-D tensor) in the supplied calibration data.
- `transB` may be 0 or 1.
- Opsets >= 11 (only `Cast`/`Mul`/`Add` are needed).

Left untouched (safe no-op, node passes through as-is):
- Non-constant, non-2-D, or non-float32 weights.
- A layer whose activation input never produced a plain 2-D tensor in the
  calibration data (no Hessian to compute).
- Weight names listed in `skip_names`.

Needs calibration data (unlike `quantize_weight_only_nf4`/
`quantize_weight_only_kmeans`, which need none): BiLLM's salient-column
selection is inherently Hessian-based, exactly like `onnxsim.apply_gptq`/
`onnxsim.apply_awq`. `calibration_data` defaults to
`onnxsim.generate_random_calibration_data`; pass real representative
batches (e.g. via `onnxsim.load_huggingface_calibration_data`) for salient
columns that actually reflect deployment-time activation statistics.

This is an aggressive, lossy quantizer by construction -- close to 1
bit/element on average is BiLLM's entire point. Expect real accuracy loss
relative to INT4-family schemes; it targets the extreme end of the
size/accuracy trade-off, not a drop-in replacement for them.

## Usage

```python
import onnx
import onnxsim

model = onnx.load("model.onnx")
quantized = onnxsim.quantize_weight_only_billm(
    model,
    calibration_data=onnxsim.load_huggingface_calibration_data(model, ...),
    block_size=128,
)
onnx.save(quantized, "model.billm.onnx")
```

`tests/test_billm.py` covers: reconstruction error improving measurably
over a plain per-block binary baseline with no salient-column handling, on
a weight/calibration scenario with genuine outlier structure; codes staying
in their declared discrete sets (`{-1, +1}` / `{-1, 0, +1}`); end-to-end
float closeness (a loose tolerance, matching how lossy this scheme is by
design); `Gemm transB=1`; and a no-op on a non-matching layer.
