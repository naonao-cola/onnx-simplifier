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

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from onnxsim.bias_correction import _all_names, _unique_name
from onnxsim.kmeans_quantization import _kmeans_1d
from onnxsim.quip_sharp import _match_matmul_like


def _kmeans_blocks_by_features(
    features: np.ndarray, k: int, iters: int, seed: int
) -> np.ndarray:
    """Ordinary multi-dimensional Lloyd's k-means over each block's own
    feature vector (here, ``[mean, std]`` -- see :func:`_block_features`),
    clustering blocks by data-driven similarity in that feature space
    rather than by fixed position. Returns the per-block cluster
    assignment, shape ``[num_blocks]``, values in ``[0, k)``.

    This is the multi-dimensional analogue of
    :func:`onnxsim.kmeans_quantization._kmeans_1d` (same Lloyd's-algorithm
    structure), generalized to vector-valued points since a block's
    clustering signal is its own ``(mean, std)`` pair, not a single scalar.
    """
    rng = np.random.default_rng(seed)
    num_blocks = features.shape[0]
    if num_blocks <= k:
        return (np.arange(num_blocks) % k).astype(np.int64)

    init_idx = rng.choice(num_blocks, size=k, replace=False)
    centroids = features[init_idx].astype(np.float64).copy()

    assignments = np.zeros(num_blocks, dtype=np.int64)
    for _ in range(iters):
        distances = np.sum(
            (features[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2
        )
        assignments = np.argmin(distances, axis=1)
        new_centroids = centroids.copy()
        for c in range(k):
            mask = assignments == c
            if mask.any():
                new_centroids[c] = features[mask].mean(axis=0)
        if np.allclose(new_centroids, centroids):
            centroids = new_centroids
            break
        centroids = new_centroids

    distances = np.sum(
        (features[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2
    )
    return np.argmin(distances, axis=1).astype(np.int64)


def _block_features(blocks: np.ndarray) -> np.ndarray:
    """Per-block ``[mean, std]`` summary statistics, shape
    ``[num_blocks, 2]`` -- the signal LO-BCQ clusters blocks by (as opposed
    to their position in the tensor)."""
    return np.stack([blocks.mean(axis=1), blocks.std(axis=1)], axis=1)


def _fit_lo_bcq(
    blocks: np.ndarray,
    num_clusters: int,
    num_codes: int,
    outer_iters: int,
    seed: int,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Runs LO-BCQ's own alternating block-clustering / per-cluster-codebook
    fitting loop over ``blocks`` (``[num_blocks, block_size]``, already
    partitioned along the reduction axis). Returns
    ``(codebooks, cluster_ids, codes)``:

    - ``codebooks``: ``[num_clusters, num_codes]`` float64, one Lloyd-max
      codebook per cluster, fit only from that cluster's own blocks.
    - ``cluster_ids``: ``[num_blocks]`` int64, the final cluster each block
      is assigned to.
    - ``codes``: ``[num_blocks, block_size]`` uint8, each element's index
      into its own block's cluster's codebook.
    """
    num_blocks, _block_size = blocks.shape
    flat = blocks.reshape(-1)

    # A single global fallback codebook seeds every cluster, so a cluster
    # that (transiently, or permanently for a small/unlucky num_blocks)
    # never gets any block assigned still reconstructs reasonably instead
    # of falling back to an all-zero codebook that could spuriously look
    # attractive to unrelated blocks.
    fallback_centroids, _ = _kmeans_1d(flat, num_codes, 20, seed)
    codebooks = np.tile(np.sort(fallback_centroids), (num_clusters, 1))

    features = _block_features(blocks)
    cluster_ids = _kmeans_blocks_by_features(features, num_clusters, 20, seed)

    for _outer in range(outer_iters):
        for c in range(num_clusters):
            mask = cluster_ids == c
            if not mask.any():
                continue
            values = blocks[mask].reshape(-1)
            centroids, _ = _kmeans_1d(values, num_codes, 20, seed + c)
            codebooks[c] = np.sort(centroids)

        errors = np.empty((num_blocks, num_clusters), dtype=np.float64)
        for c in range(num_clusters):
            diffs = np.abs(
                blocks[:, :, np.newaxis] - codebooks[c][np.newaxis, np.newaxis, :]
            )
            nearest = np.argmin(diffs, axis=2)
            recon = codebooks[c][nearest]
            errors[:, c] = np.mean((blocks - recon) ** 2, axis=1)
        new_cluster_ids = np.argmin(errors, axis=1).astype(np.int64)

        if np.array_equal(new_cluster_ids, cluster_ids):
            cluster_ids = new_cluster_ids
            break
        cluster_ids = new_cluster_ids

    codes = np.zeros((num_blocks, _block_size), dtype=np.uint8)
    for c in range(num_clusters):
        mask = cluster_ids == c
        if not mask.any():
            continue
        diffs = np.abs(
            blocks[mask][:, :, np.newaxis] - codebooks[c][np.newaxis, np.newaxis, :]
        )
        codes[mask] = np.argmin(diffs, axis=2).astype(np.uint8)

    return codebooks, cluster_ids, codes


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
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    skip_set: "set[str]" = set(skip_names) if skip_names is not None else set()

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph
    initializer_map = {t.name: t for t in graph.initializer}
    taken_names = _all_names(graph)
    num_codes = 2**bits

    nodes = list(graph.node)
    for node in nodes:
        match = _match_matmul_like(node)
        if match is None:
            continue
        _x_name, w_name, _bias_name, weight_transposed = match
        if w_name in skip_set:
            continue
        w_init = initializer_map.get(w_name)
        if (
            w_init is None
            or w_init.data_type != onnx.TensorProto.FLOAT
            or len(w_init.dims) != 2
        ):
            continue

        w = onnx.numpy_helper.to_array(w_init).astype(np.float64)
        w_nk = w if weight_transposed else w.T  # [N, K], output channel first
        n, k = w_nk.shape
        if k % block_size != 0:
            continue

        num_blocks_per_row = k // block_size
        num_blocks = n * num_blocks_per_row
        blocks = w_nk.reshape(num_blocks, block_size)

        codebooks, cluster_ids, codes = _fit_lo_bcq(
            blocks, num_clusters, num_codes, outer_iters, seed
        )

        prefix = f"{w_name}_lo_bcq"
        codebooks_name = _unique_name(f"{prefix}_codebooks", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(
                codebooks.astype(np.float32), name=codebooks_name
            )
        )
        cluster_ids_name = _unique_name(f"{prefix}_cluster_ids", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(cluster_ids, name=cluster_ids_name)
        )
        codes_name = _unique_name(f"{prefix}_codes", taken_names)
        graph.initializer.append(onnx.numpy_helper.from_array(codes, name=codes_name))

        selected_out = _unique_name(f"{prefix}_selected", taken_names)
        select_node = onnx.helper.make_node(
            "Gather",
            [codebooks_name, cluster_ids_name],
            [selected_out],
            axis=0,
            name=_unique_name(f"{prefix}_select_node", taken_names),
        )

        cast_out = _unique_name(f"{prefix}_codes_i64", taken_names)
        cast_node = onnx.helper.make_node(
            "Cast", [codes_name], [cast_out], to=onnx.TensorProto.INT64
        )

        gathered_out = _unique_name(f"{prefix}_gathered", taken_names)
        gather_elements_node = onnx.helper.make_node(
            "GatherElements",
            [selected_out, cast_out],
            [gathered_out],
            axis=1,
            name=_unique_name(f"{prefix}_gather_elements_node", taken_names),
        )

        nk_shape_name = _unique_name(f"{prefix}_nk_shape", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(
                np.array([n, k], dtype=np.int64), name=nk_shape_name
            )
        )
        unblocked_name = _unique_name(f"{prefix}_unblocked", taken_names)
        reshape_node = onnx.helper.make_node(
            "Reshape",
            [gathered_out, nk_shape_name],
            [unblocked_name],
            name=_unique_name(f"{prefix}_reshape_node", taken_names),
        )

        new_nodes = [select_node, cast_node, gather_elements_node, reshape_node]

        final_name = unblocked_name
        if not weight_transposed:
            final_name = _unique_name(f"{prefix}_transposed", taken_names)
            new_nodes.append(
                onnx.helper.make_node(
                    "Transpose",
                    [unblocked_name],
                    [final_name],
                    name=_unique_name(f"{prefix}_transpose_node", taken_names),
                    perm=[1, 0],
                )
            )

        node_idx = next(i for i, nd in enumerate(graph.node) if nd is node)
        for offset, new_node in enumerate(new_nodes):
            graph.node.insert(node_idx + offset, new_node)

        for i, inp in enumerate(node.input):
            if inp == w_name:
                node.input[i] = final_name

    return out
