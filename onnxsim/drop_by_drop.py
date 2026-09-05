"""Drop-by-Drop (Babaoglu, Chen, Khisti, University of Toronto, 2026,
"Drop-by-Drop: Multi-Bitwidth Quantization for LLMs Using Additive
Codebooks", https://arxiv.org/abs/2606.12876).

:mod:`onnxsim.aqlm` already ports the additive/residual codebook idea this
module builds on: represent each group of a weight row as the *sum* of
``M`` lookups, one from each of ``M`` codebooks fit greedily to the
residual left over by the previous stages --
``ŵ_group = C_1[i_1] + C_2[i_2] + ... + C_M[i_M]``. That module (and every
other single-artifact codebook scheme in onnxsim --
:mod:`onnxsim.kmeans_quantization`, :mod:`onnxsim.lo_bcq`,
:mod:`onnxsim.gptvq`) produces exactly *one* fixed-bit-width artifact per
call: getting a different bit-width means re-running the whole
quantization from scratch on different settings. AQLM's own ``M``
codebooks are fit (independently of this module's own choice) to
together minimize the *final*, full-``M``-term reconstruction error --
nothing about the fitting objective favours any particular prefix of
those ``M`` terms being independently good on its own, it just happens to
be a side effect of greedy residual fitting that a prefix is *usable* (if
software chooses to stop early) and *no worse* than not stopping early.

Drop-by-Drop's own distinguishing contribution is a **successive
refinement ("Matryoshka") structure**: quantize once into ``K`` additive
codebook terms, but fit and order them so that using only the *first*
``k <= K`` terms (simply dropping the remaining ``K - k``, no
re-quantization, no separate run) is *itself* the deployment artifact for
bit-width ``k`` -- one quantization pass yields ``K`` usable
bit-widths simultaneously, each a valid, complete, well-formed
reconstruction of the same weight, rather than one bit-width per run.
Restated precisely: **AQLM's own terms are fit to together minimize
reconstruction error at one fixed ``K``; this module's own terms are fit
so that every prefix of them is independently a good reconstruction on
its own, not just the full set.**

The plain greedy-residual order (fit codebook 1 to ``W``, fit codebook 2
to the residual ``W - C_1[i_1]``, and so on) already gives the *weak*
form of this property for free, entirely mechanically: since every stage
targets exactly the error the previous stages left over, each additive
prefix's reconstruction error is monotonically non-increasing in the
number of terms included (this module's own tests check that directly,
the same way :mod:`onnxsim.aqlm`'s tests do). What AQLM's own fitting
does *not* do is bias *which* residual gets corrected first towards what
matters most for a short prefix -- an unweighted greedy fit spends its
first (and therefore most bit-width-constrained) codebook equally across
every group, whether or not that group's own weights are large enough to
matter for the layer's output. This module's own simplification of the
paper's information-theoretic ("Gaussian weights, successive refinement")
framing: fix, once, a per-group importance weight from each group's own
RMS magnitude in the *original* (unquantized) weight, and use that same
fixed weighting in *every* stage's k-means fit (a weighted generalization
of Lloyd's algorithm -- weighted mean for the centroid update, ordinary
nearest-centroid assignment). Every stage, not just the first, therefore
spends its limited codebook capacity preferentially on the
higher-magnitude (higher-impact) groups, so a short prefix -- the ``k=1``
or ``k=2`` case in particular, where a plain unweighted fit would spread
its accuracy thinnest -- is a *better* reconstruction than the unweighted
scheme's own same-length prefix would be, not merely a *valid* one. This
is a pragmatic, directly verifiable approximation of the paper's own
theory, not a reproduction of its calibrated optimizer -- consistent with
:mod:`onnxsim.aqlm`'s own choice of classical greedy residual k-means
over AQLM's bespoke beam search.

Reconstruction, and the "usable at any prefix" property, is made
concrete in the emitted graph itself: each stage's codes/codebook are
separate initializers (exactly :mod:`onnxsim.aqlm`'s own layout), and the
dequantization is a running ``Add`` chain whose *intermediate* sums are
named, real graph values -- ``{prefix}_partial1 = stage_0``,
``{prefix}_partial2 = partial1 + stage_1``, ..., up to
``{prefix}_partial{K}``, the last of which is what the emitted graph
actually feeds into the original MatMul/Gemm by default (the
full-``K``-term, highest-fidelity option). :func:`select_drop_by_drop_prefix`
then rewires that same graph to instead feed ``partial{k}`` for any
``1 <= k <= K``, in place, with no re-quantization -- turning "any prefix
is a usable, complete reconstruction" from a claim about the fitting
algorithm into an operation this module's own tests exercise directly.
"""

from __future__ import annotations

from typing import List, Union

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from onnxsim.aqlm import _match_matmul_like
from onnxsim.bias_correction import _all_names, _unique_name

_PREFIX_SUFFIX = "_dbd"


def _fit_weighted_kmeans_codebook(
    data: np.ndarray,
    weights: np.ndarray,
    codebook_size: int,
    num_iterations: int,
    rng: np.random.Generator,
) -> "tuple[np.ndarray, np.ndarray]":
    """Weighted Lloyd's-algorithm k-means: same structure as
    :func:`onnxsim.aqlm._fit_kmeans_codebook` (nearest-centroid assignment
    each round, centroids seeded from a random sample of ``data``'s own
    rows), except each centroid's update is the ``weights``-weighted mean
    of its currently-assigned points rather than the plain mean.
    ``weights`` (``[num_points]``) is fixed across every call this module
    makes for a given layer, so every additive stage's fit is biased the
    same way -- see this module's own docstring for why. An empty cluster
    keeps its previous centroid.
    """
    num_points = data.shape[0]
    k = min(codebook_size, num_points)
    init_idx = rng.choice(num_points, size=k, replace=False)
    centroids = data[init_idx].copy()
    if k < codebook_size:
        pad = np.tile(centroids[-1:], (codebook_size - k, 1))
        centroids = np.vstack([centroids, pad])

    assignment = np.zeros(num_points, dtype=np.int64)
    for _ in range(num_iterations):
        dist = np.sum(
            (data[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2
        )
        assignment = np.argmin(dist, axis=1)
        new_centroids = centroids.copy()
        for c in range(codebook_size):
            mask = assignment == c
            if np.any(mask):
                w = weights[mask]
                new_centroids[c] = (data[mask] * w[:, np.newaxis]).sum(axis=0) / w.sum()
        centroids = new_centroids

    dist = np.sum((data[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
    assignment = np.argmin(dist, axis=1)
    return centroids, assignment


def quantize_weight_only_drop_by_drop(
    model: Union[str, onnx.ModelProto],
    group_dim: int = 8,
    num_codebooks: int = 4,
    codebook_size: int = 256,
    num_iterations: int = 10,
    seed: int = 0,
) -> onnx.ModelProto:
    """Quantizes every MatMul/vanilla-Gemm layer with a constant 2-D
    float32 weight (whose reduction dimension ``K`` is evenly divisible by
    ``group_dim``) into Drop-by-Drop's own successive-refinement additive
    codebook scheme -- see this module's own docstring for the technique
    and how it differs from :func:`onnxsim.aqlm.quantize_weight_only_aqlm`.
    Needs no calibration data: every codebook, and the per-group
    importance weighting used to fit it, comes directly from the weight
    tensor's own values.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param group_dim: elements per group (each ``group_dim``-element chunk
            of a row gets its own set of ``num_codebooks`` indices, one
            per codebook)
    :param num_codebooks: number of additive codebook stages ``K`` --
            unlike :mod:`onnxsim.aqlm`, every prefix length
            ``1 <= k <= K`` is independently a usable deployment
            bit-width (see :func:`select_drop_by_drop_prefix`), so this
            is naturally larger than AQLM's own typical 1-2
    :param codebook_size: entries per codebook (2^8 = 256 is a typical
            choice, matching one byte per stored index)
    :param num_iterations: weighted-Lloyd's-algorithm iterations refining
            each stage's codebook
    :param seed: seed for the k-means centroid initialization (a fresh
            ``numpy.random.Generator`` is derived per matched layer, in
            graph node order, so results are deterministic and
            reproducible for a given model and seed)
    :returns: ``model`` with every matched layer's weight replaced by a
            running ``Add`` chain of ``K`` ``Gather`` lookups (one per
            codebook stage), reshaped back to the weight's own shape and
            feeding the original MatMul/Gemm node; the chain's own
            intermediate sums are named ``{weight_name}_dbd_partial1`` ..
            ``{weight_name}_dbd_partial{K}`` (the last is what the graph
            actually uses) so :func:`select_drop_by_drop_prefix` can
            later retarget any layer to any shorter prefix in place.
            Layers with a non-constant, non-2-D, or non-group-divisible
            weight are left untouched.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph
    initializer_map = {t.name: t for t in graph.initializer}
    taken_names = _all_names(graph)

    nodes = list(graph.node)
    candidates = []
    for node in nodes:
        match = _match_matmul_like(node)
        if match is None:
            continue
        x_name, w_name, weight_transposed = match
        w_init = initializer_map.get(w_name)
        if (
            w_init is None
            or w_init.data_type != onnx.TensorProto.FLOAT
            or len(w_init.dims) != 2
        ):
            continue
        candidates.append((node, w_name, weight_transposed))

    if not candidates:
        return out

    rng = np.random.default_rng(seed)

    for node, w_name, weight_transposed in candidates:
        w_init = initializer_map[w_name]
        w = onnx.numpy_helper.to_array(w_init).astype(np.float64)
        dim0, dim1 = w.shape
        w_nk = w if weight_transposed else w.T  # [N, K], output channel first
        n, k = w_nk.shape
        if k % group_dim != 0:
            continue

        num_groups = n * (k // group_dim)
        groups = w_nk.reshape(num_groups, group_dim)

        # Fixed, per-group importance from the *original* weight's own
        # magnitude, reused unchanged at every stage -- see this module's
        # own docstring for why this is what makes short prefixes better,
        # not just valid.
        importance = np.sqrt(np.mean(groups**2, axis=1))
        importance = np.maximum(importance, importance.max() * 1e-6 + 1e-12)

        residual = groups.copy()
        codebooks: List[np.ndarray] = []
        codes: List[np.ndarray] = []
        for _ in range(num_codebooks):
            centroids, assignment = _fit_weighted_kmeans_codebook(
                residual, importance, codebook_size, num_iterations, rng
            )
            codebooks.append(centroids)
            codes.append(assignment)
            residual = residual - centroids[assignment]

        prefix = f"{w_name}{_PREFIX_SUFFIX}"
        stage_outputs = []
        new_nodes: List[onnx.NodeProto] = []
        for m in range(num_codebooks):
            codebook_name = _unique_name(f"{prefix}_codebook{m}", taken_names)
            graph.initializer.append(
                onnx.numpy_helper.from_array(
                    codebooks[m].astype(np.float32), name=codebook_name
                )
            )
            codes_name = _unique_name(f"{prefix}_codes{m}", taken_names)
            graph.initializer.append(
                onnx.numpy_helper.from_array(codes[m].astype(np.int64), name=codes_name)
            )
            stage_out = _unique_name(f"{prefix}_stage{m}", taken_names)
            gather_node = onnx.helper.make_node(
                "Gather",
                [codebook_name, codes_name],
                [stage_out],
                name=_unique_name(f"{prefix}_gather{m}_node", taken_names),
                axis=0,
            )
            new_nodes.append(gather_node)
            stage_outputs.append(stage_out)

        # Running-sum ("Matryoshka") chain: partial{m} is exactly the
        # reconstruction a caller gets by keeping only the first m stages.
        # partial1 is an Identity (not just stage_outputs[0] reused
        # directly) so it is always its own named node output, giving
        # select_drop_by_drop_prefix one uniform node to look for however
        # many stages a layer has, including num_codebooks == 1.
        partial1_name = _unique_name(f"{prefix}_partial1", taken_names)
        new_nodes.append(
            onnx.helper.make_node(
                "Identity",
                [stage_outputs[0]],
                [partial1_name],
                name=_unique_name(f"{prefix}_partial1_node", taken_names),
            )
        )
        combined = partial1_name
        for m in range(1, num_codebooks):
            partial_name = _unique_name(f"{prefix}_partial{m + 1}", taken_names)
            add_node = onnx.helper.make_node(
                "Add",
                [combined, stage_outputs[m]],
                [partial_name],
                name=_unique_name(f"{prefix}_add{m}_node", taken_names),
            )
            new_nodes.append(add_node)
            combined = partial_name

        shape_name = _unique_name(f"{prefix}_shape", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(
                np.array([n, k], dtype=np.int64), name=shape_name
            )
        )
        unblocked_name = _unique_name(f"{prefix}_unblocked", taken_names)
        reshape_node = onnx.helper.make_node(
            "Reshape",
            [combined, shape_name],
            [unblocked_name],
            name=_unique_name(f"{prefix}_reshape_node", taken_names),
        )
        new_nodes.append(reshape_node)

        final_name = unblocked_name
        if not weight_transposed:
            final_name = _unique_name(f"{prefix}_transposed", taken_names)
            transpose_node = onnx.helper.make_node(
                "Transpose",
                [unblocked_name],
                [final_name],
                name=_unique_name(f"{prefix}_transpose_node", taken_names),
                perm=[1, 0],
            )
            new_nodes.append(transpose_node)

        node_idx = next(i for i, n in enumerate(graph.node) if n is node)
        for offset, new_node in enumerate(new_nodes):
            graph.node.insert(node_idx + offset, new_node)
        for i, inp in enumerate(node.input):
            if inp == w_name:
                node.input[i] = final_name

    return out


def select_drop_by_drop_prefix(model: onnx.ModelProto, k: int) -> onnx.ModelProto:
    """Retargets every Drop-by-Drop-quantized layer in ``model`` to use
    only its own first ``k`` additive codebook terms, in place, with no
    re-quantization -- the concrete, exercisable form of this module's own
    "usable at any prefix length" property (see this module's docstring).

    Finds each layer's own ``Reshape`` node (named
    ``{weight_name}_dbd_reshape_node`` by
    :func:`quantize_weight_only_drop_by_drop`) and rewires its first input
    from that layer's own full-``K``-term sum to its own
    ``{weight_name}_dbd_partial{k}`` sum instead -- the exact same named
    intermediate value the full-precision graph already computed on the
    way to its own final sum, just consumed ``K - k`` ``Add`` nodes
    earlier in the chain. Nodes computing dropped stages become dead code
    (left in place; a follow-up :func:`onnx.utils.extract_model`-style
    cleanup or shape-inference pass can remove them, exactly as for any
    other ONNX graph edit that leaves unreachable nodes behind).

    :param model: a model previously returned by
            :func:`quantize_weight_only_drop_by_drop`
    :param k: how many additive terms to keep, ``1 <= k <= K`` for
            whatever ``K`` (``num_codebooks``) each individual layer was
            quantized with
    :raises ValueError: if ``k < 1``, or if ``k`` exceeds some quantized
            layer's own number of stages
    :returns: a new ``ModelProto`` with every Drop-by-Drop layer's
            dequantization truncated to its first ``k`` terms; a model
            with no Drop-by-Drop layers is returned unchanged
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph

    output_producer = {}
    for node in graph.node:
        for out_name in node.output:
            output_producer[out_name] = node

    suffix = "_reshape_node"
    for node in graph.node:
        if node.op_type != "Reshape" or not node.name.endswith(suffix):
            continue
        marker = f"{_PREFIX_SUFFIX}_reshape_node"
        if not node.name.endswith(marker):
            continue
        prefix = node.name[: -len(suffix)]

        num_stages = 0
        while f"{prefix}_partial{num_stages + 1}" in output_producer:
            num_stages += 1
        if num_stages == 0:
            continue

        if k > num_stages:
            raise ValueError(
                f"k={k} exceeds {prefix}'s own {num_stages} quantized stages"
            )

        node.input[0] = f"{prefix}_partial{k}"

    return out
