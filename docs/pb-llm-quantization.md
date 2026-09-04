# PB-LLM: partial binarization (`quantize_weight_only_pb_llm`)

## What this is

`onnxsim.quantize_weight_only_pb_llm` is a genuine **lossy PTQ quantizer**
that splits each matched layer's weight columns into two precisions: a
small, salience-selected fraction stays INT8, and every other column is
pushed to ~1 bit/element. It reuses `onnxsim.billm._sign` for the
binarization primitive and reads like a structured cousin of
`onnxsim.quantize_weight_only_billm`, but the two solve genuinely different
problems -- read `onnxsim/pb_llm.py`'s own module docstring for the full
comparison; the short version:

- `onnxsim.quantize_weight_only_billm` binarizes **the entire weight**:
  every column ends up represented by one or two binary (`{-1, +1}`)
  levels. Its own salient/non-salient split only decides *how many* binary
  levels a column gets (two, via a residual, for salient columns; one for
  everything else) -- the whole layer stays close-to-1-bit-average
  regardless of the split.
- `onnxsim.quantize_weight_only_pb_llm`'s salient fraction is **not
  binarized at all** -- it is quantized to genuine INT8, the same
  precision `onnxsim.quantize_weight_only_int8_block` produces for a whole
  layer. Only the non-salient remainder is binarized. `salient_ratio=0.0`
  degenerates to plain per-column binarization; `salient_ratio=1.0`
  degenerates to plain per-column INT8 -- BiLLM has no comparable "all
  INT8" limit, since every one of its own output columns is always binary.

It's also a different axis from `onnxsim.apply_mixed_precision_quantization`,
which dispatches a bit-width **per whole layer** (INT8 vs. INT4, via an
INT4-reconstruction-MSE-times-activation-energy score) and never splits
precision within one layer's own weight matrix. PB-LLM's split is per
**column**, inside a single layer, driven by a direct
Hessian-diagonal-weighted-magnitude salience score --
`salience_j = mean_n(|W[n, j]|) * diag(H)_j` (`H = X^T X`, the same
calibration-derived Hessian `onnxsim.gptq`/`onnxsim.billm`/`onnxsim.owq`
already build) -- with **no Hessian inversion**, unlike `onnxsim.owq`'s own
Optimal-Brain-Surgeon-style score.

```
Before:
  Y = MatMul(X, W) [+ bias]        -- W constant, [K, N], float32

After:
  Code: initializer, int8, [K, N]  -- full INT8 value for a salient
                                       column, {-1, +1} for a non-salient
                                       one
  Scale: initializer, float32, [K, 1]  -- per-column scale
  What_hat = Mul(Cast(Code, float), Scale)
  Y = MatMul(X, What_hat) [+ bias]
```

Because an INT8 code and a `{-1, +1}` code both reconstruct the same way
(`value = code * scale`), one matched layer needs only a single code
tensor and a single per-column scale tensor -- no `Add` of a second term
the way `onnxsim.quantize_weight_only_billm`'s two-level salient path
needs. Ordinary ONNX ops only (`Cast`/`Mul`), opset 11+.

## Where this comes from

[PB-LLM](https://arxiv.org/abs/2310.00034) (Shang, Yuan, Wu and Dong, 2024,
ICLR 2024) observes that pushing every weight of an LLM to 1 bit collapses
accuracy, because a small fraction of weights carry disproportionate
importance to the layer's output (the same premise GPTQ/SparseGPT/OWQ-style
salience metrics are built on). Its fix: identify that salient fraction (the
paper's own metric is a Hessian-diagonal-weighted magnitude, in the same
family as GPTQ/SparseGPT/OWQ's own pruning/quantization salience metrics),
keep it at a higher precision, and binarize the rest. This module implements
that structure -- the salience score, the salient/non-salient split, INT8
for the salient side, `sign(W) * mean(|W|)` binarization (`onnxsim.billm`'s
own single-level `_binary` primitive, applied per column) for the
non-salient side. Not ported: the paper's own "Optimal"-mode residual
refinement (Section 4.2), which adds a second binarization level on top of
the non-salient columns' own residual for extra accuracy -- a reasonable
future addition (it would reuse `onnxsim.billm._binary`'s residual pattern
directly, the same way `onnxsim.billm` itself applies it to *its own*
salient columns), not attempted here since the paper's simpler "Naive" mode
(what this module implements) already recovers most of PB-LLM's accuracy
gain over full binarization.

## Scope

Handled:
- `MatMul(X, W)` / `Gemm(X, W[, B], transA=0, alpha=1, beta=1)` with `W` a
  constant 2-D float32 tensor and `X` a plain 2-D activation that appears
  (as a 2-D tensor) in the supplied calibration data.
- `transB` may be 0 or 1.
- Opsets >= 11 (only `Cast`/`Mul` are needed).

Left untouched (safe no-op, node passes through as-is):
- Non-constant, non-2-D, or non-float32 weights.
- A layer whose activation input never produced a plain 2-D tensor in the
  calibration data (no Hessian diagonal to compute).
- Weight names listed in `skip_names`.

Needs calibration data (unlike `quantize_weight_only_nf4`/
`quantize_weight_only_kmeans`, which need none): PB-LLM's salience score is
inherently Hessian-based. `calibration_data` defaults to
`onnxsim.generate_random_calibration_data`; pass real representative
batches (e.g. via `onnxsim.load_huggingface_calibration_data`) for a
salience ranking that actually reflects deployment-time activation
statistics.

## Usage

```python
import onnx
import onnxsim

model = onnx.load("model.onnx")
quantized = onnxsim.quantize_weight_only_pb_llm(
    model,
    calibration_data=onnxsim.load_huggingface_calibration_data(model, ...),
    salient_ratio=0.15,
)
onnx.save(quantized, "model.pb_llm.onnx")
```

`tests/test_pb_llm.py` covers: reconstruction error improving over a fully
binary (`salient_ratio=0.0`) baseline on a weight/calibration scenario with
genuine outlier structure; codes spanning both the full INT8 range (salient
columns) and the plain `{-1, +1}` range (non-salient columns); the
`salient_ratio=0.0`/`1.0` degenerate cases (fully binary / fully INT8);
end-to-end float closeness (a loose tolerance, matching how lossy the
binarized fraction is by design); `Gemm transB=1`; and a no-op on a
non-matching layer.
