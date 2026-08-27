"""SqueezeLLM (Kim et al., 2023, "SqueezeLLM: Dense-and-Sparse
Quantization", https://arxiv.org/abs/2306.07629). onnxsim ports the
algorithm, not any framework's code, per the same rationale as
:mod:`onnxsim.awq`/:mod:`onnxsim.gptq`/:mod:`onnxsim.hqq` (SqueezeLLM's own
reference implementation quantizes live PyTorch ``nn.Linear`` weights, with
no ONNX export path).

Every weight-only INT4 scheme already in onnxsim (``quantize_weight_only_int4``
and everything built on it, plus :mod:`onnxsim.hqq`) quantizes onto a
*uniform* integer grid -- one scale (and, for HQQ, one zero-point) per
group, evenly spaced codes. SqueezeLLM instead lets each group pick its own
small set of arbitrary values (a per-group codebook, not an arithmetic
sequence), fit directly to that group's own weight distribution -- so a
group whose values cluster tightly around a couple of modes gets levels
concentrated there, rather than spread evenly across the group's full
range as a uniform grid would. Two ideas combine to decide *where* those
per-group levels land:

- **Sensitivity-weighted k-means.** A weight element's effect on the
  layer's output scales with its input activation's own second moment (the
  same diagonal-Hessian/Fisher approximation :mod:`onnxsim.gptq` computes
  in full -- here only the diagonal, ``mean(x_k ** 2)`` per input channel
  ``k``, is needed). Each group's codebook is fit by ordinary Lloyd's-
  algorithm k-means, except each element's contribution to a centroid's
  update is weighted by that sensitivity -- so a centroid drifts to sit
  closer to elements the layer's output is more sensitive to, and the
  overall codebook is a weighted-least-squares-optimal fit to the group's
  actual distribution, not a fixed shape guessed in advance (e.g. NF4's
  fixed Gaussian-quantile codebook).
- **Dense-and-sparse decomposition.** A small fraction of weight elements
  (the paper's own default, ``0.45%``, by magnitude across the whole
  tensor) are excluded from the k-means fit entirely (so a handful of huge
  outliers can't drag a group's codebook toward them at the expense of
  every other element) and instead corrected back to their *exact* original
  value by a separate additive term. Unlike :mod:`onnxsim.llm_int8`
  (which excludes activation outlier *channels* from an INT8 matmul and
  computes the excluded part in float), this decomposition is on the
  *weight*, and the correction here is represented as an ordinary dense
  float32 initializer that is zero everywhere except at outlier positions
  -- exact, and expressible with a plain ``Add``, at the cost of not
  getting genuine sparse storage/compute savings (a real sparse-matrix
  deployment format is a separate, downstream concern this module does not
  address, matching :mod:`onnxsim.nf4`'s own choice to trade deployment
  compactness for a graph expressible in ordinary ONNX ops).

Dequantization is expressed with ``GatherND(codebook, codes, batch_dims=1)``
-- a per-group codebook lookup, unlike :mod:`onnxsim.nf4`'s plain ``Gather``
against one *global* fixed codebook -- followed by ``Reshape`` to unblock,
an ``Add`` of the sparse correction, and (when the weight was not already
stored transposed) a ``Transpose`` back to the node's own layout. No custom
op or contrib domain is needed, only ``GatherND``'s ``batch_dims`` support
(opset 12+).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from onnxsim import backend
from onnxsim.bias_correction import _add_probe_outputs, _all_names, _unique_name
from onnxsim.calibration import Tensors, generate_random_calibration_data


def _match_matmul_like(node: onnx.NodeProto):
    """Mirrors ``MatchMatMulLike`` (``passes/quantize_matmul_common.h``):
    a MatMul, or a Gemm with ``transA=0``, ``alpha=1`` and (when it has a
    bias) ``beta=1``. Returns ``(x_name, w_name, weight_transposed)`` or
    ``None``.
    """
    attrs = {a.name: a for a in node.attribute}
    if node.op_type == "MatMul":
        if len(node.input) != 2:
            return None
        return node.input[0], node.input[1], False
    if node.op_type == "Gemm":
        num_inputs = len(node.input)
        if num_inputs not in (2, 3):
            return None
        trans_a = attrs.get("transA")
        if trans_a is not None and trans_a.i != 0:
            return None
        alpha = attrs.get("alpha")
        if alpha is not None and alpha.f != 1.0:
            return None
        if num_inputs == 3:
            beta = attrs.get("beta")
            if beta is not None and beta.f != 1.0:
                return None
        trans_b = attrs.get("transB")
        weight_transposed = bool(trans_b is not None and trans_b.i)
        return node.input[0], node.input[1], weight_transposed
    return None


def _weighted_kmeans_quantize_groups(
    values: np.ndarray, weights: np.ndarray, num_levels: int, num_iterations: int
) -> "tuple[np.ndarray, np.ndarray]":
    """Returns ``(codes, centroids)`` for ``values``/``weights`` (each
    ``[num_groups, group_size]``): ``codes`` in ``[0, num_levels)`` and one
    ``num_levels``-entry codebook per group, fit by sensitivity-weighted
    Lloyd's-algorithm k-means (see this module's own docstring). A group's
    centroids are initialized from evenly-spaced quantiles of its own
    (unweighted) sorted values -- a reasonable per-group spread with no
    randomness needed beyond the calibration data itself.
    """
    num_groups, group_size = values.shape
    sorted_vals = np.sort(values, axis=1)
    init_idx = np.linspace(0, group_size - 1, num_levels).round().astype(np.int64)
    centroids = sorted_vals[:, init_idx]  # [G, L]

    for _ in range(num_iterations):
        dist = (values[:, :, np.newaxis] - centroids[:, np.newaxis, :]) ** 2
        codes = np.argmin(dist, axis=2)  # [G, group_size]
        new_centroids = centroids.copy()
        for level in range(num_levels):
            mask = codes == level
            wsum = np.sum(weights * mask, axis=1)
            vsum = np.sum(weights * values * mask, axis=1)
            safe = wsum > 1e-12
            new_centroids[safe, level] = vsum[safe] / wsum[safe]
        centroids = new_centroids

    dist = (values[:, :, np.newaxis] - centroids[:, np.newaxis, :]) ** 2
    codes = np.argmin(dist, axis=2)
    return codes.astype(np.int64), centroids.astype(np.float32)


def quantize_weight_only_squeezellm(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    block_size: int = 32,
    bits: int = 4,
    outlier_fraction: float = 0.0045,
    num_kmeans_iterations: int = 20,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Quantizes every MatMul/vanilla-Gemm layer with a constant 2-D
    float32 weight (whose reduction dimension ``K`` is evenly divisible by
    ``block_size``) and a plain 2-D activation input into SqueezeLLM-style
    dense-and-sparse non-uniform quantization -- see this module's own
    docstring for the technique.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches to measure each
            input channel's sensitivity (``mean(x_k ** 2)``) on. Each batch
            is a ``{input_name: np.ndarray}`` dict matching ``model``'s
            graph inputs -- see :func:`onnxsim.generate_random_calibration_data`
            (the default when omitted) and
            :func:`onnxsim.load_huggingface_calibration_data` (real data, a
            much more representative sensitivity estimate than random
            input).
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param block_size: elements per ``(output channel, block)`` codebook
            group along the reduction dimension
    :param bits: codebook size is ``2 ** bits`` centroids per group (the
            paper's own default, ``4``, i.e. 16 centroids)
    :param outlier_fraction: fraction of weight elements (by magnitude,
            across the whole tensor) excluded from the k-means fit and
            corrected back to their exact original value instead (the
            paper's own default, ``0.0045``, i.e. 0.45%)
    :param num_kmeans_iterations: weighted Lloyd's-algorithm iterations
            refining each group's codebook
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations
    :returns: ``model`` with every matched layer's weight replaced by
            ``GatherND(codebook, codes, batch_dims=1)`` (per-group
            codebook lookup) followed by ``Reshape``, an ``Add`` of the
            sparse outlier correction, and (when needed) a ``Transpose``,
            feeding the original MatMul/Gemm node; layers with a
            non-constant, non-2-D, non-block-divisible, or activation-less
            weight are left untouched
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph

    opset_ge_12 = any(
        o.domain in ("", "ai.onnx") and o.version >= 12 for o in out.opset_import
    )
    if not opset_ge_12:
        return out  # GatherND's batch_dims needs opset >= 12

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
        candidates.append((node, x_name, w_name, weight_transposed))

    if not candidates:
        return out

    probe_names = sorted({x_name for _, x_name, _, _ in candidates})
    probe_model = _add_probe_outputs(out, probe_names)

    act_sensitivity: Dict[str, "tuple[np.ndarray, int]"] = {}
    for batch in calibration_data:
        result = backend.run_model(probe_model, batch, providers=providers)
        for name in probe_names:
            x = np.asarray(result[name], dtype=np.float64)
            if x.ndim != 2:
                continue
            sq_sum = np.sum(x**2, axis=0)
            count = x.shape[0]
            if name in act_sensitivity:
                prev_sum, prev_count = act_sensitivity[name]
                act_sensitivity[name] = (prev_sum + sq_sum, prev_count + count)
            else:
                act_sensitivity[name] = (sq_sum, count)

    num_levels = 2**bits

    for node, x_name, w_name, weight_transposed in candidates:
        entry = act_sensitivity.get(x_name)
        if entry is None:
            continue  # never observed as a plain 2-D tensor; skip
        sq_sum, count = entry
        sensitivity_k = sq_sum / count  # [K], mean(x_k ** 2)

        w_init = initializer_map[w_name]
        w = onnx.numpy_helper.to_array(w_init).astype(np.float64)
        dim0, dim1 = w.shape
        w_nk = w if weight_transposed else w.T  # [N, K], output channel first
        n, k = w_nk.shape
        if sensitivity_k.shape[0] != k:
            continue  # activation's feature dim doesn't match K; skip
        if k % block_size != 0:
            continue

        threshold = np.quantile(np.abs(w_nk), 1.0 - outlier_fraction)
        outlier_mask_nk = np.abs(w_nk) > threshold

        num_blocks = k // block_size
        num_groups = n * num_blocks
        values = w_nk.reshape(num_groups, block_size)
        weights = (
            np.broadcast_to(
                sensitivity_k.reshape(num_blocks, block_size),
                (n, num_blocks, block_size),
            )
            .reshape(num_groups, block_size)
            .copy()
        )
        outlier_mask_flat = outlier_mask_nk.reshape(num_groups, block_size)
        weights[outlier_mask_flat] = 0.0  # outliers don't skew the codebook fit

        codes, codebook = _weighted_kmeans_quantize_groups(
            values, weights, num_levels, num_kmeans_iterations
        )
        dequant_nk = codebook[np.arange(num_groups)[:, np.newaxis], codes].reshape(n, k)
        sparse_diff_nk = np.where(outlier_mask_nk, w_nk - dequant_nk, 0.0).astype(
            np.float32
        )

        prefix = f"{w_name}_squeezellm"
        codebook_name = _unique_name(f"{prefix}_codebook", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(codebook, name=codebook_name)
        )
        codes_name = _unique_name(f"{prefix}_codes", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(
                codes.reshape(num_groups, block_size, 1), name=codes_name
            )
        )
        sparse_diff_name = _unique_name(f"{prefix}_sparse_diff", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(sparse_diff_nk, name=sparse_diff_name)
        )

        gathered_name = _unique_name(f"{prefix}_gathered", taken_names)
        gather_node = onnx.helper.make_node(
            "GatherND",
            [codebook_name, codes_name],
            [gathered_name],
            name=_unique_name(f"{prefix}_gathernd_node", taken_names),
            batch_dims=1,
        )

        unblocked_name = _unique_name(f"{prefix}_unblocked", taken_names)
        shape_name = _unique_name(f"{prefix}_shape", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(
                np.array([n, k], dtype=np.int64), name=shape_name
            )
        )
        reshape_node = onnx.helper.make_node(
            "Reshape",
            [gathered_name, shape_name],
            [unblocked_name],
            name=_unique_name(f"{prefix}_reshape_node", taken_names),
        )

        corrected_name = _unique_name(f"{prefix}_corrected", taken_names)
        add_node = onnx.helper.make_node(
            "Add",
            [unblocked_name, sparse_diff_name],
            [corrected_name],
            name=_unique_name(f"{prefix}_add_node", taken_names),
        )

        new_nodes = [gather_node, reshape_node, add_node]
        final_name = corrected_name
        if not weight_transposed:
            final_name = _unique_name(f"{prefix}_transposed", taken_names)
            transpose_node = onnx.helper.make_node(
                "Transpose",
                [corrected_name],
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
