# K-means weight-codebook quantization (`quantize_weight_only_kmeans`)

## What this is

`onnxsim.quantize_weight_only_kmeans` fits a small codebook to each
layer's own weight values via ordinary k-means clustering, then stores
one codebook index per weight instead of the weight itself.

```
Before:
  Y = MatMul(X, W) [+ bias]                 -- W constant, [K, N], float32

After:
  Codebook: initializer, float32, [2**bits]  -- this LAYER's own fitted centroids
  Codes: initializer, uint8, [K, N]           -- codebook index per element
  Whatever_hat = Gather(Codebook, Cast(Codes, INT64), axis=0)
  Y = MatMul(X, Whatever_hat) [+ bias]
```

Unlike every other codebook-based scheme in onnxsim, no per-block scale
or `Reshape`/`Mul` is needed at all: since k-means clusters the weight's
own values directly (not a normalized `[-1, 1]` range), the codebook
entry gathered *is* the reconstructed weight.

## Where this comes from

[Deep Compression](https://arxiv.org/abs/1510.00149) (Han et al., 2015,
Section 3) introduces "trained quantization": cluster a layer's weight
values, share one codebook entry across every weight assigned to a
cluster, then fine-tune the codebook's own values against the network's
training loss. This module ports the *clustering* half via an ordinary,
from-scratch Lloyd's-algorithm k-means fit (alternate assigning each
weight to its nearest centroid, then recomputing each centroid as the
mean of its assigned weights, until convergence) -- not the paper's own
gradient-based codebook fine-tuning afterward, a training loop with no
ONNX export path (the same reason `apply_gptq`/`apply_awq` port their own
papers' *algorithms*, not any framework's live-training code). The
pruning half of the same paper is already covered separately by
`onnxsim.apply_magnitude_pruning`.

Every codebook-based scheme already in onnxsim (`onnxsim.nf4`,
`onnxsim.mx_quantization`) uses a **fixed** codebook -- chosen once by
the format's own definition, identical for every tensor. This module's
codebook is the opposite: fit *per layer*, directly to that layer's own
weight distribution -- the classical, verifiable procedure for fitting a
codebook to actual data, as opposed to NF4/MXFP4's fixed choices or
`apply_spinquant`'s eigenbasis (a *rotation*, not a scalar codebook).

## Scope

Handled:

- `MatMul(X, W)` with `W` a constant 2-D float32 tensor, or `Gemm(X, W[, B])`
  with `transA=0`, `alpha=1`, `beta=1` (when `B` is present). `transB`
  may be 0 or 1. No block-size or reduction-dimension divisibility
  constraint -- clustering works on the flattened weight regardless of
  shape.
- `skip_names` to leave specific matched weights unquantized.

Left untouched (safe no-op, node passes through as-is):

- Non-constant or non-2-D weights.

## Usage

```python
import onnx
import onnxsim

model = onnx.load("model.onnx")
quantized = onnxsim.quantize_weight_only_kmeans(model, bits=4, iters=20)
onnx.save(quantized, "model.kmeans.onnx")
```

Needs no calibration data: the codebook is fit entirely from the weight
tensor's own values.
