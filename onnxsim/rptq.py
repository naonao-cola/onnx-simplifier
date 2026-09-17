"""RPTQ (Yuan et al., 2023, "RPTQ: Reorder-based Post-training Quantization
for Large Language Models", https://arxiv.org/abs/2304.01089). onnxsim ports
the *algorithm*, not any framework's code, per the same rationale as
:mod:`onnxsim.smoothquant`/:mod:`onnxsim.duquant` (RPTQ's own reference
implementation reorders and quantizes live PyTorch activations with no ONNX
export path).

:mod:`onnxsim.smoothquant`/:mod:`onnxsim.outlier_suppression_plus` address
the same root problem RPTQ does -- a handful of transformer activation
channels sit at a much larger scale than the rest, so a single per-tensor
(or per-token) quantization range wastes most of its resolution on the
ordinary channels -- but via *scaling* (and, for Outlier Suppression+, an
added shift): move difficulty from the activation into the weight with an
elementwise multiply. RPTQ's own fix is structurally different: it never
rescales anything. Instead, it observes that a per-tensor range is only
forced on the quantizer because *all* channels share one quantization
group; if channels with similar ranges are grouped together and each group
gets its own tight, independently-computed `[min, max]`-derived scale/zero
-point (a **per-cluster** quantization scheme), a channel's outlier-ness no
longer drags down the resolution available to unrelated channels. RPTQ finds
those groups by **clustering** the `K` input channels -- by each channel's
own calibration range statistics -- into a small number of clusters, then
**permuting** the channels so that same-cluster channels sit contiguously.

Reordering the reduction axis (`K`) of an activation and reordering the
matching axis (the input-channel rows) of the weight the same way is an
exact algebraic identity for any permutation `perm`, not an approximation:

    Gather(X, perm, axis=-1) @ Gather(W, perm, axis=0) == X @ W

(`Gather(W, perm, axis=0)` is `W[perm, :]` for a plain, un-transposed
`MatMul`/Gemm weight, or `W[:, perm]` when `transB=1` stores the weight as
`[N, K]` instead of `[K, N]`.) So, like :mod:`onnxsim.smoothquant`, this
module only performs the *reorder*: :func:`apply_rptq_reorder` returns a
float model, provably equivalent to the input up to floating-point
rounding -- no quantization happens here. Concretely, per matched layer:
compute each input channel's own calibration `[min, max]` (equivalently
here, its abs-max -- see below), cluster those per-channel statistics into
`num_clusters` groups with a plain Lloyd's-algorithm k-means (no `scipy`
dependency), derive the permutation that sorts channels by cluster, insert
a `Gather(X, perm_indices, axis=-1)` before the layer, and permute the
weight's `K`-axis rows in place by that same permutation. This composition
is exact regardless of how good the clustering turns out to be -- a bad
cluster assignment only costs quantization quality downstream, never
correctness of the permuted float graph produced here.

RPTQ's own paper goes considerably further than this module does, most
notably: (1) a bespoke integer-programming search over the *number* of
clusters trading off latency against accuracy (this module instead takes
`num_clusters` as a plain, fixed parameter, the same simplification
:mod:`onnxsim.smoothquant` makes for its own `alpha`); (2) reordering and
per-cluster-quantizing the attention mechanism's own K/V cache and
softmax input specifically (RPTQ's main target, since those tensors are
particularly outlier-heavy in the models the paper studies); and (3) a
"reorder back" step fusing the inverse permutation into LayerNorm scale
parameters so a reordered layer's *output* can feed directly into the next
unreordered layer, avoiding a runtime `Gather` entirely. None of these are
ported: this module targets the same general MatMul/Gemm layers
:mod:`onnxsim.smoothquant` does (not attention-specific KV-cache/softmax
tensors -- see :mod:`onnxsim.kv_cache_quantization` for onnxsim's existing,
separate KV-cache quantization support), always materializes the reorder as
a runtime `Gather` rather than fusing it away, and always quantizes the
result of clustering with the number of clusters given, exactly the same
"headline structural contribution, not every secondary refinement" scope
line :mod:`onnxsim.outlier_suppression_plus` draws relative to its own
paper.

Because the whole point of the reorder is to let a *separate* quantizer
apply a tight per-cluster range instead of one per-tensor range, this
module also returns, alongside the permuted model, a small
:class:`RptqLayerInfo` record per matched layer carrying the cluster
boundaries in the *permuted* channel order. Wiring per-cluster ranges all
the way through onnxsim's existing calibration/quantization API
(:func:`onnxsim.calibrate`, :func:`onnxsim.quantize_static`,
:func:`onnxsim.quantize_qoperator_gemm`) would need each of those to grow a
new per-slice-of-a-tensor calibration mode -- none support anything finer
than "one range per tensor" today -- which is real, independent scope
beyond porting RPTQ's own permutation-construction contribution; this
module stops at handing back the information a future per-cluster
quantizer would need, the same honest boundary
:mod:`onnxsim.outlier_suppression_plus`'s own docstring draws around OS+'s
secondary scale-search refinement. Until that quantizer exists, the
practical benefit of running this ahead of
:func:`onnxsim.quantize_static`/:func:`onnxsim.quantize_qoperator_gemm` is
smaller than RPTQ's own paper reports (those still calibrate one range per
whole tensor) -- but the reorder is still free (an exact identity) and
puts a model in the right shape for a per-cluster quantizer later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import onnx

from onnxsim.calibration import Tensors


@dataclass
class RptqLayerInfo:
    """Per-cluster metadata for one RPTQ-reordered layer, as returned by
    :func:`apply_rptq_reorder`. ``cluster_bounds`` names, for each cluster,
    the half-open ``[start, end)`` slice of the *permuted* channel axis
    (i.e. of the ``Gather`` node's output, and of the weight's permuted
    ``K``-axis rows) that cluster occupies, in increasing order -- exactly
    the slices a per-cluster quantizer would compute an independent
    ``[min, max]``-derived scale/zero-point over.
    """

    x_name: str
    w_name: str
    gather_output: str
    permutation: np.ndarray
    cluster_bounds: List[Tuple[int, int]] = field(default_factory=list)


def apply_rptq_reorder(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    num_clusters: int = 4,
    providers: Optional[Sequence[str]] = None,
) -> Tuple[onnx.ModelProto, Dict[str, RptqLayerInfo]]:
    """Clusters and reorders every MatMul/vanilla-Gemm layer's input
    channels (RPTQ's own contribution -- see this module's docstring),
    using real calibration activations. Returns a float model -- an exact
    reordering, not a quantization -- meant to be fed to a downstream
    per-cluster-aware W8A8 quantizer; see this module's own docstring for
    how far that composition is (and isn't) wired up today.

    Thin wrapper delegating to the verified C++ port
    (:func:`onnxsim.apply_rptq_reorder_cpp`) -- full parameter parity, no
    functionality gap (see that function's own docstring, and
    ``onnxsim/rptq_entry.h``, for the one documented, immaterial-to-
    -correctness numerical divergence: the k-means fit's own RNG does not
    reproduce numpy's PCG64 bit-for-bit).

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches to measure each
            input channel's calibration range on. Each batch is a
            ``{input_name: np.ndarray}`` dict matching ``model``'s graph
            inputs -- see :func:`onnxsim.generate_random_calibration_data`
            (the default when omitted) and
            :func:`onnxsim.load_huggingface_calibration_data` (real data, a
            much more representative clustering than random input).
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied) and for the k-means centroid
            tie-breaking
    :param num_clusters: number of channel clusters to reorder each
            matched layer's input into (the paper's own default range is a
            handful of clusters per layer; RPTQ's bespoke integer-
            -programming search over this count is not ported -- see this
            module's own docstring)
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations
    :returns: a ``(model, layer_info)`` pair. ``model`` has every matched
            layer's weight rows permuted in place by ``K``-axis cluster and
            a new ``Gather`` node inserted before it applying the same
            permutation to its activation input. ``layer_info`` maps each
            matched layer's original activation input name to an
            :class:`RptqLayerInfo` record describing the permutation and
            resulting cluster boundaries. Layers with a non-constant,
            non-2-D weight, or whose activation input isn't a plain 2-D
            tensor matching the weight's reduction dimension, are left
            untouched and have no entry in ``layer_info``.
    """
    from onnxsim.onnx_simplifier import apply_rptq_reorder_cpp

    return apply_rptq_reorder_cpp(
        model,
        calibration_data=calibration_data,
        num_samples=num_samples,
        seed=seed,
        num_clusters=num_clusters,
        providers=providers,
    )
