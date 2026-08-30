"""K-means weight-codebook quantization (Han et al., 2015, "Deep
Compression: Compressing Deep Neural Networks with Pruning, Trained
Quantization and Huffman Coding", https://arxiv.org/abs/1510.00149,
Section 3's own "trained quantization" -- the weight-sharing half of that
paper; the pruning half is already covered by
:mod:`onnxsim.pruning`/:mod:`onnxsim.apply_magnitude_pruning`). onnxsim
ports the *idea* (cluster a layer's own weight values, share one codebook
entry across every weight assigned to a cluster) via an ordinary,
from-scratch Lloyd's-algorithm k-means fit -- not the paper's own
gradient-based fine-tuning of the codebook after clustering (a training
loop with no ONNX export path, the same reason :mod:`onnxsim.gptq`/
:mod:`onnxsim.awq` port their own papers' *algorithms* rather than any
framework's live-training code).

Every codebook-based scheme already in onnxsim (:mod:`onnxsim.nf4`,
:mod:`onnxsim.mx_quantization`) uses a **fixed** codebook -- 16 values
chosen once, by the format's own definition (a standard normal
distribution's quantile points for NF4; a 4-bit float format's own bit
patterns for MXFP4), identical for every tensor. This module's codebook
is the opposite: it is fit *per layer*, directly to that layer's own
weight distribution, via k-means (Lloyd's algorithm: alternate assigning
each weight to its nearest centroid, then recomputing each centroid as
the mean of the weights assigned to it, until convergence or a fixed
iteration budget) -- the classical, closed-form-per-iteration, verifiable
procedure for fitting a codebook to actual data, as opposed to NF4/MXFP4's
"the codebook needs no fitting because it isn't data-derived at all" or
:mod:`onnxsim.spinquant`'s eigenbasis (a *rotation*, not a scalar
codebook).

Because the codebook is fit directly to real weight values (not to a
*normalized* [-1, 1] range the way NF4's is), no per-block scale is
needed at all -- the graph is simply:

    Before:
      Y = MatMul(X, W) [+ bias]                 -- W constant, [K, N], float32

    After:
      Codebook: initializer, float32, [2**bits]  -- this LAYER's own fitted centroids
      Codes: initializer, uint8, [K, N]           -- codebook index per element
      Whatever_hat = Gather(Codebook, Cast(Codes, INT64), axis=0)
      Y = MatMul(X, Whatever_hat) [+ bias]

-- two ordinary ops, no ``Reshape``/``Mul`` needed (unlike
:mod:`onnxsim.nf4`, which reshapes to broadcast a *separate* per-block
scale against its codebook lookup): here the gathered codebook value
already *is* the reconstructed weight, since clustering was already done
directly in the weight's own units.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from onnxsim.bias_correction import _all_names, _unique_name
from onnxsim.quip_sharp import _match_matmul_like


def _kmeans_1d(
    values: np.ndarray, k: int, iters: int, seed: int
) -> "tuple[np.ndarray, np.ndarray]":
    """Ordinary Lloyd's-algorithm k-means over a flat array of scalar
    values. Returns ``(centroids, assignments)``: ``k`` fitted cluster
    centers and, for each element of ``values``, its nearest centroid's
    index.

    Centroids are initialized at ``k`` evenly-spaced percentiles of
    ``values`` itself (not uniformly random) -- a deterministic choice
    that already roughly matches the data's own distribution shape,
    converging in far fewer iterations than random initialization would
    for the skewed, near-zero-centered distributions real weight tensors
    have.
    """
    rng = np.random.default_rng(seed)
    percentiles = np.linspace(0, 100, k)
    centroids = np.unique(np.percentile(values, percentiles))
    if centroids.shape[0] < k:
        extra = rng.choice(values, size=k - centroids.shape[0], replace=True)
        centroids = np.concatenate([centroids, extra])
    centroids = centroids.astype(np.float64)

    assignments = np.zeros(values.shape[0], dtype=np.int64)
    for _ in range(iters):
        distances = np.abs(values[:, np.newaxis] - centroids[np.newaxis, :])
        assignments = np.argmin(distances, axis=1)
        new_centroids = centroids.copy()
        for c in range(k):
            mask = assignments == c
            if mask.any():
                new_centroids[c] = values[mask].mean()
        if np.allclose(new_centroids, centroids):
            centroids = new_centroids
            break
        centroids = new_centroids

    distances = np.abs(values[:, np.newaxis] - centroids[np.newaxis, :])
    assignments = np.argmin(distances, axis=1)
    return centroids, assignments


def quantize_weight_only_kmeans(
    model: Union[str, onnx.ModelProto],
    bits: int = 4,
    iters: int = 20,
    seed: int = 0,
    skip_names: Optional["set[str]"] = None,
) -> onnx.ModelProto:
    """Quantizes every MatMul/vanilla-Gemm layer with a constant 2-D
    float32 weight into a per-layer k-means-fitted codebook -- see this
    module's own docstring for the technique. Needs no calibration data:
    the codebook is fit directly to the weight tensor's own values.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param bits: codebook size is ``2**bits`` (default 4, matching every
            other onnxsim scheme's own INT4-equivalent storage: 1 byte
            per weight for the code plus one small, shared codebook, same
            as :func:`onnxsim.quantize_weight_only_nf4`)
    :param iters: maximum Lloyd's-algorithm iterations per layer (stops
            early once no weight changes cluster assignment)
    :param seed: seed for the random fallback samples used only if a
            layer's own percentile-based centroid initialization yields
            fewer than ``2**bits`` distinct starting points (e.g. a
            weight with many repeated/zero values)
    :param skip_names: weight initializer names to leave unquantized even
            if otherwise eligible
    :returns: ``model`` with every matched layer's weight replaced by
            ``Gather(Codebook, Cast(Codes, INT64), axis=0)`` feeding the
            original MatMul/Gemm node -- ordinary ONNX ops only, no
            contrib op and no minimum opset beyond what
            ``Gather``/``Cast`` themselves need (opset 11+). Layers with
            a non-constant or non-2-D weight are left untouched.
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
        _x_name, w_name, _bias_name, _weight_transposed = match
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
        dim0, dim1 = w.shape
        flat = w.reshape(-1)

        centroids, assignments = _kmeans_1d(flat, num_codes, iters, seed)
        codes = assignments.reshape(dim0, dim1).astype(np.uint8)

        prefix = f"{w_name}_kmeans"
        codebook_name = _unique_name(f"{prefix}_codebook", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(
                centroids.astype(np.float32), name=codebook_name
            )
        )
        codes_name = _unique_name(f"{prefix}_codes", taken_names)
        graph.initializer.append(onnx.numpy_helper.from_array(codes, name=codes_name))

        cast_out = _unique_name(f"{prefix}_codes_i64", taken_names)
        cast_node = onnx.helper.make_node(
            "Cast", [codes_name], [cast_out], to=onnx.TensorProto.INT64
        )
        dq_out = _unique_name(f"{prefix}_dq", taken_names)
        gather_node = onnx.helper.make_node(
            "Gather",
            [codebook_name, cast_out],
            [dq_out],
            axis=0,
            name=_unique_name(f"{prefix}_gather_node", taken_names),
        )

        node_idx = next(i for i, n in enumerate(graph.node) if n is node)
        graph.node.insert(node_idx, cast_node)
        graph.node.insert(node_idx + 1, gather_node)

        for i, inp in enumerate(node.input):
            if inp == w_name:
                node.input[i] = dq_out

    return out
