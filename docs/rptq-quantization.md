# RPTQ: cluster-and-reorder pre-quantization (`apply_rptq_reorder`)

## What this is

`onnxsim.apply_rptq_reorder` clusters a MatMul/vanilla-Gemm layer's input
channels by their own calibration range, then **reorders** the activation
(via a runtime `Gather`) and the weight's matching rows so same-cluster
channels sit contiguously. Unlike `onnxsim.apply_smoothquant`/
`onnxsim.apply_outlier_suppression_plus` (which *scale*/*shift* difficulty
from the activation into the weight) or `onnxsim.apply_quarot`/
`onnxsim.apply_duquant` (which *rotate*), this only *permutes* -- an exact
algebraic identity for any permutation, not an approximation. Like those
other migration passes, this returns a float model meant to be fed to a
separate W8A8 quantizer afterwards; unlike them, the actual benefit RPTQ's
paper reports needs a *per-cluster* quantization range, which is not (yet)
wired through onnxsim's existing calibration/quantization API -- see
"Scope" below.

```
Before:
  Y = MatMul(X, W) [+ bias]           -- W constant, [K, N], float32

After:
  perm: initializer, int64 [K]        -- sorts channels by calibration cluster
  Xr = Gather(X, perm, axis=-1)       -- reorders the activation's K axis
  Y = MatMul(Xr, Wr) [+ bias]         -- Wr: W's K-axis rows permuted by perm
```

## Where this comes from

[RPTQ](https://arxiv.org/abs/2304.01089) (Yuan et al., 2023) starts from the
same observation `apply_smoothquant` does: a handful of transformer
activation channels sit at a much larger scale than the rest, so a single
per-tensor (or per-token) quantization range wastes most of its resolution
on the ordinary channels. RPTQ's own fix is structurally different from
scaling: a per-tensor range is only forced on the quantizer because *all*
channels share one quantization group. If channels with similar ranges are
grouped together and each group gets its own tight, independently-computed
`[min, max]`-derived scale/zero-point (a **per-cluster** quantization
scheme), no single channel's outlier-ness drags down the resolution
available to unrelated channels. RPTQ finds those groups by **clustering**
the `K` input channels by their own calibration range statistics, then
**permuting** the channels so that same-cluster channels sit contiguously.

Reordering the reduction axis of an activation and reordering the matching
axis (the input-channel rows) of the weight the same way is an exact
identity for any permutation `perm`:

```
Gather(X, perm, axis=-1) @ Gather(W, perm, axis=0) == X @ W
```

so, like `apply_smoothquant`, this module performs only the reorder: no
quantization happens here. Per matched layer: compute each input channel's
own calibration abs-max, cluster those per-channel statistics into
`num_clusters` groups with a plain Lloyd's-algorithm k-means (no `scipy`
dependency -- a from-scratch implementation, seeded at evenly spaced
percentiles of the per-channel statistics rather than at random points, so a
small `num_clusters` reliably separates distinct scales), derive the
permutation that sorts channels by cluster, and apply it to both operands.

RPTQ's own paper goes considerably further than this module does. Not
ported: (1) a bespoke integer-programming search over the *number* of
clusters trading off latency against accuracy -- `num_clusters` is instead a
plain, fixed parameter, the same simplification `apply_smoothquant` makes
for its own `alpha`; (2) reordering and per-cluster-quantizing the attention
mechanism's own K/V cache and softmax input specifically -- RPTQ's paper
targets those tensors above all (see `onnxsim.quantize_kv_cache` for
onnxsim's own, separate KV-cache quantization support); and (3) a "reorder
back" step fusing the inverse permutation into LayerNorm scale parameters so
a reordered layer's output can feed the next, unreordered layer without a
runtime `Gather` -- this module always materializes the reorder as an actual
`Gather` node instead. This is the same "headline structural contribution,
not every secondary refinement" scope line `apply_outlier_suppression_plus`
draws relative to its own paper.

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

**Not wired up**: `apply_rptq_reorder` returns per-layer cluster boundary
metadata (an `RptqLayerInfo` per matched layer, mapping the layer's
original activation name to the permutation applied and the resulting
`[start, end)` slices of the *permuted* channel axis each cluster occupies)
alongside the reordered model -- exactly the information a per-cluster
quantizer needs. Wiring that all the way through `onnxsim.calibrate`,
`onnxsim.quantize_static`, or `onnxsim.quantize_qoperator_gemm` would need
each of those to grow a new per-slice-of-a-tensor calibration mode (today
they all calibrate exactly one range per whole tensor) -- real, independent
scope beyond porting RPTQ's own permutation construction. Until that
quantizer exists, running `apply_rptq_reorder` ahead of
`quantize_static`/`quantize_qoperator_gemm` is still a free, exact
transformation, but those quantizers will still calibrate one range across
the whole (now-reordered) tensor rather than one per cluster, so the
practical accuracy benefit RPTQ's own paper reports will not fully
materialize until a per-cluster quantizer consumes `RptqLayerInfo` too.

## Usage

```python
import onnx
import onnxsim

model = onnx.load("model.onnx")
reordered, layer_info = onnxsim.apply_rptq_reorder(model, num_clusters=4)
onnx.save(reordered, "model.rptq.onnx")

# layer_info maps each matched layer's original activation name to an
# RptqLayerInfo(x_name, w_name, gather_output, permutation, cluster_bounds)
for x_name, info in layer_info.items():
    print(x_name, "->", info.cluster_bounds)
```

`calibration_data` defaults to `onnxsim.generate_random_calibration_data`;
pass real representative batches (e.g. via
`onnxsim.load_huggingface_calibration_data`) for a clustering that actually
reflects the model's own real per-channel ranges, the same caveat
`apply_smoothquant`/`apply_duquant` document.
