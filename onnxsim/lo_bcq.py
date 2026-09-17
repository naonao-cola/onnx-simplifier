"""LO-BCQ -- Block Clustered Quantization (Elangovan, Sakr, Raghunathan,
Khailany, 2025, "LO-BCQ: Block Clustered Quantization for 4-bit (W4A4) LLM
Inference", https://arxiv.org/abs/2502.05376). onnxsim ports the weight-side
half of the technique -- the paper's own dual contribution also covers
activation quantization (A4), out of scope here for the same reason every
other onnxsim weight-only scheme leaves activations in float: this repo's
graphs quantize weights statically and let the runtime handle activations.

:mod:`onnxsim.kmeans_quantization` already ports the idea of a
data-fitted (non-uniform) codebook: cluster a layer's own weight *values*
via Lloyd's algorithm and share one codebook across the whole tensor. Every
group-wise scheme already in onnxsim (:func:`onnxsim.quantize_weight_only_int4`,
:mod:`onnxsim.owq`, :mod:`onnxsim.slim_llm`) instead partitions a layer into
fixed-size contiguous groups along the reduction axis, but applies the exact
same scheme (one shared codebook, or one affine scale) to every group --
grouping only changes *which elements share a scale/codebook*, never how
that scale/codebook is chosen. LO-BCQ's own contribution sits strictly
between those two: **kmeans_quantization.py fits one codebook for the whole
tensor; this module fits several small codebooks, one per data-driven
cluster of blocks, chosen to minimize each cluster's own reconstruction
error.** Concretely:

1. Decompose the weight into fixed-size ``block_size``-element contiguous
   blocks along the reduction axis (the same ``group_size``-along-K
   convention as :mod:`onnxsim.nf4`'s own blocking).
2. Cluster the *blocks themselves* -- not their position, their own
   summary statistics (each block's mean and standard deviation) -- into
   ``num_clusters`` groups via ordinary multi-dimensional Lloyd's k-means.
   Blocks from anywhere in the tensor land in the same cluster purely
   because their own value distributions look alike.
3. Fit one small, dedicated non-uniform (Lloyd-max/k-means, reusing
   :func:`onnxsim.kmeans_quantization._kmeans_1d` unchanged) codebook per
   cluster, from only that cluster's own currently-assigned blocks' values.
4. Alternate: re-assign every block to whichever cluster's *current*
   codebook reconstructs it with the lowest mean-squared error, then
   re-fit each cluster's codebook from its newly-assigned blocks -- the
   paper's own greedy MSE-minimizing scheme -- for a fixed number of
   rounds (or until assignments stop changing).

The result is a modest generalization of :mod:`onnxsim.kmeans_quantization`:
``num_clusters`` small per-cluster codebooks instead of one, selected
per-block rather than per-tensor. Reconstruction needs one extra indexing
step versus that module's single ``Gather``:

    Before:
      Y = MatMul(X, W) [+ bias]                    -- W constant, [K, N], float32

    After:
      Codebooks: initializer, float32, [num_clusters, 2**bits]
      ClusterIds: initializer, int64, [num_blocks]      -- per-block cluster index
      Codes: initializer, uint8, [num_blocks, block_size] -- per-element code,
             indexing into that block's OWN cluster's codebook
      SelectedCodebooks = Gather(Codebooks, ClusterIds, axis=0)   -- [num_blocks, 2**bits]
      Gathered = GatherElements(SelectedCodebooks, Cast(Codes, INT64), axis=1)
      W_hat = Reshape(Gathered, W's own shape)
      Y = MatMul(X, W_hat) [+ bias]

``GatherElements`` (opset 11+, same as ``Gather``) is what makes the extra
per-block codebook selection expressible without a per-element ``If`` or a
custom op: ``SelectedCodebooks[b]`` is already the right ``2**bits``-entry
table for block ``b``, so indexing it along axis 1 with that block's own
per-element codes reconstructs every element in one shot -- no scale
``Mul`` needed at all, exactly like :mod:`onnxsim.kmeans_quantization`
(codebooks are fit directly in the weight's own units).

Deliberately not ported: the paper's own mixed-precision bit allocation
across layers and its activation-side (A4) quantization -- both live outside
this module's single-layer, weight-only scope, the same boundary every
other onnxsim weight-only module (e.g. :mod:`onnxsim.nf4`,
:mod:`onnxsim.gptvq`) already draws.
"""

from __future__ import annotations

from typing import Optional, Union

import onnx

from onnxsim.onnx_simplifier import apply_lo_bcq_cpp


def quantize_weight_only_lo_bcq(
    model: Union[str, onnx.ModelProto],
    bits: int = 4,
    block_size: int = 32,
    num_clusters: int = 4,
    outer_iters: int = 10,
    seed: int = 0,
    skip_names: Optional["set[str]"] = None,
) -> onnx.ModelProto:
    """Quantizes every MatMul/vanilla-Gemm layer with a constant 2-D
    float32 weight (whose reduction dimension ``K`` is evenly divisible by
    ``block_size``) via LO-BCQ's block-clustered codebook scheme -- see
    this module's own docstring for the technique. Needs no calibration
    data: every quantization decision comes from the weight tensor's own
    values.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param bits: each cluster's own codebook has ``2**bits`` entries
            (default 4, matching every other onnxsim codebook scheme's own
            INT4-equivalent storage)
    :param block_size: elements per block along the reduction dimension
            ``K`` (same convention as :mod:`onnxsim.nf4`'s own blocking)
    :param num_clusters: number of distinct per-cluster codebooks fit per
            layer (small by design -- LO-BCQ is a modest generalization of
            :mod:`onnxsim.kmeans_quantization`'s single shared codebook,
            not a full mixture model)
    :param outer_iters: maximum alternating rounds of (re-fit each
            cluster's codebook, then re-assign every block to whichever
            cluster's codebook now reconstructs it best); stops early once
            no block changes cluster
    :param seed: seed for every k-means step this module runs (the initial
            block-feature clustering and every per-cluster codebook fit)
    :param skip_names: weight initializer names to leave unquantized even
            if otherwise eligible
    :returns: ``model`` with every matched layer's weight replaced by
            ``Reshape(GatherElements(Gather(Codebooks, ClusterIds, axis=0),
            Cast(Codes, INT64), axis=1), original_shape)`` feeding the
            original MatMul/Gemm node -- ordinary ONNX ops only, no contrib
            op and no minimum opset beyond what ``Gather``/``GatherElements``
            themselves need (opset 11+). Layers with a non-constant,
            non-2-D, or non-block-divisible weight are left untouched.

    Delegates to the verified C++ port (:func:`onnxsim.apply_lo_bcq_cpp`)
    when called with the default ``bits=4``/``block_size=32``/
    ``num_clusters=4``/``outer_iters=10``/``seed=0``/``skip_names=None`` --
    the C++ port's own hardcoded values. A non-default value raises
    ``ValueError`` rather than being silently ignored (no real caller in
    this codebase needs a non-default value). The C++ port's own block-
    clustering step also uses a genuinely different (deterministic,
    feature-norm-sorted) initialization than this function's own former
    seeded-random-sample one -- an accepted, permanent divergence already
    documented in ``passes/lo_bcq.h``, not a bug.
    """
    if (
        bits != 4
        or block_size != 32
        or num_clusters != 4
        or outer_iters != 10
        or seed != 0
        or skip_names
    ):
        raise ValueError(
            "quantize_weight_only_lo_bcq now delegates to apply_lo_bcq_cpp, "
            "which hardcodes bits=4, block_size=32, num_clusters=4, "
            "outer_iters=10, seed=0 and has no skip_names knob; call with "
            "the defaults, or use apply_lo_bcq_cpp directly"
        )
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return apply_lo_bcq_cpp(model)
