"""DuQuant (Lin et al., 2024, "DuQuant: Distributing Outliers via Dual
Transformation Makes Stronger Quantized LLMs",
https://arxiv.org/abs/2406.01721, NeurIPS 2024). onnxsim ports the
algorithm, not any framework's code, per the same rationale as
:mod:`onnxsim.awq`/:mod:`onnxsim.gptq`/:mod:`onnxsim.quarot` (DuQuant's
own reference implementation rotates and quantizes live PyTorch weights
with no ONNX export path).

:mod:`onnxsim.quarot` already ports the core idea that a *random*
orthogonal rotation removes outlier directions from both a weight and its
activation with high probability (the same concentration-of-measure
argument :mod:`onnxsim.quip_sharp` relies on), letting both operands drop
to INT4. DuQuant's own motivation is a specific failure mode of that
approach: some LLMs have a handful of **massive-activation channels** --
not spread thinly across many directions the way an "ordinary" outlier
distribution is, but concentrated so heavily in just a few channels that
a *single* random rotation draw isn't guaranteed to spread them out
evenly (the concentration-of-measure argument is a high-probability
statement over the *choice* of rotation, not a guarantee for any one
specific draw) -- so quantizing right after a random rotation can still
leave those specific channels dominating whichever block they land in.

DuQuant's own fix has two stages, applied to the weight and activation the
same way :mod:`onnxsim.quarot` applies its single random rotation:

1. **Rotate** to spread out ordinary outlier structure (same idea as
   :mod:`onnxsim.quarot`).
2. **Permute**, using the calibration data's own per-channel activation
   magnitude to find which channels are still the worst offenders, and
   redistribute them one-per-block across the quantization grouping --
   so no single block ends up bearing more than its fair share of
   whatever outlier energy survived the rotation.

DuQuant's own reference implementation constructs its rotation via a
greedy, iterative algorithm that pairs each identified outlier channel
with a partner channel via a 2-D Givens rotation, repeated in "blocks"
across the hidden dimension -- calibration-driven, but a bespoke
optimization procedure that is not independently verifiable the way a
closed-form construction is (the same reason :mod:`onnxsim.spinquant`
substitutes a closed-form eigenbasis for SpinQuant's own learned Cayley
rotation). This module instead builds the same two-stage effect from two
classical, verifiable pieces:

- **Permutation** (a genuine permutation matrix, an orthogonal matrix by
  construction): rank channels by their own calibration abs-max
  magnitude, then greedily assign the highest-magnitude channels
  round-robin, one at a time, to whichever quantization block currently
  holds the least outlier magnitude -- so the surviving outlier channels
  end up spread as evenly as possible across blocks, rather than
  clustered whichever way they originally fell in the reduction
  dimension.
- **Block-local random rotation**: after permutation, apply an
  independent Haar-random orthogonal rotation (:mod:`onnxsim.quip_sharp`'s
  own ``_random_orthogonal_matrix``) *within* each block -- the same
  concentration-of-measure spreading :mod:`onnxsim.quarot` relies on
  globally, but applied locally, after the worst channels have already
  been separated from each other by the permutation, so each block's own
  rotation only ever has to spread out at most its own fair share of
  outlier energy.

The composition of a permutation matrix and a block-diagonal orthogonal
matrix is itself orthogonal (:math:`(PR)(PR)^T = P R R^T P^T = P P^T = I`
since each block of ``R`` is itself orthogonal), so this module reuses
:mod:`onnxsim.quarot`'s own graph-construction machinery verbatim -- the
weight rotated and block-INT4-quantized offline
(:mod:`onnxsim.omniquant`'s ``_quantize_blockwise_int4_with_clip``), the
activation rotated and INT4-quantized per token at graph-run time (the
same data-free pattern :mod:`onnxsim.kv_cache_quantization`'s Value-style
rewrite uses) -- with only the *construction* of ``U`` differing from
:mod:`onnxsim.quarot`'s fully random one. Unlike :mod:`onnxsim.quarot`,
this needs calibration data (the whole point is to target the *specific*
channels the real activation distribution concentrates outliers in,
rather than relying on a probabilistic argument that ignores that
structure) -- the same trade-off :mod:`onnxsim.spinquant` already makes
relative to :mod:`onnxsim.quip_sharp`'s own random rotation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from onnxsim import backend
from onnxsim.adaround import _pack_int4
from onnxsim.bias_correction import _add_probe_outputs, _all_names, _unique_name
from onnxsim.calibration import Tensors, generate_random_calibration_data
from onnxsim.omniquant import _quantize_blockwise_int4_with_clip
from onnxsim.quip_sharp import _match_matmul_like, _random_orthogonal_matrix


def _has_min_opset(model: onnx.ModelProto, min_version: int) -> bool:
    return any(
        o.domain in ("", "ai.onnx") and o.version >= min_version
        for o in model.opset_import
    )


def _build_duquant_rotation(
    act_absmax: np.ndarray,
    block_size: int,
    outlier_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Builds the ``[K, K]`` combined permutation + block-local-rotation
    matrix ``U`` described in this module's own docstring, from a
    calibration activation's own per-channel abs-max magnitude.
    """
    k = act_absmax.shape[0]
    num_blocks = k // block_size
    num_outliers = max(1, int(round(outlier_fraction * k)))
    num_outliers = min(num_outliers, k)

    order = np.argsort(-act_absmax)  # channel indices, largest magnitude first
    outlier_channels = order[:num_outliers]
    rest_channels = order[num_outliers:]

    # Greedily assign each outlier channel (largest first) to whichever
    # block currently holds the least total outlier magnitude, so the
    # worst channels end up spread as evenly as possible across blocks.
    block_slots: List[List[int]] = [[] for _ in range(num_blocks)]
    block_load = np.zeros(num_blocks, dtype=np.float64)
    for ch in outlier_channels:
        b = int(np.argmin(block_load))
        block_slots[b].append(int(ch))
        block_load[b] += float(act_absmax[ch])

    # Fill every block's remaining slots with the non-outlier channels, in
    # their original relative (magnitude) order.
    rest_iter = iter(int(c) for c in rest_channels)
    for b in range(num_blocks):
        while len(block_slots[b]) < block_size:
            block_slots[b].append(next(rest_iter))

    perm = np.array([ch for block in block_slots for ch in block], dtype=np.int64)
    assert perm.shape[0] == k and set(perm.tolist()) == set(range(k))

    # x @ perm_matrix reorders x into the new, block-redistributed
    # channel order: (x @ perm_matrix)[i] == x[perm[i]].
    perm_matrix = np.eye(k, dtype=np.float64)[:, perm]

    block_rotation = np.zeros((k, k), dtype=np.float64)
    for b in range(num_blocks):
        start = b * block_size
        end = start + block_size
        block_rotation[start:end, start:end] = _random_orthogonal_matrix(
            block_size, rng
        )

    return perm_matrix @ block_rotation


def apply_duquant(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    block_size: int = 32,
    outlier_fraction: float = 0.05,
    epsilon: float = 1e-12,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Applies DuQuant-style calibrated permutation + block-local random
    rotation (see this module's own docstring) plus INT4 round-to-nearest
    quantization of *both* the weight and the activation to every
    MatMul/vanilla-Gemm layer with a constant 2-D float32 weight whose
    reduction dimension ``K`` is divisible by ``block_size``. Unlike
    :func:`onnxsim.apply_quarot`, this needs calibration data: the
    permutation specifically targets the channels the real activation
    distribution concentrates outliers in.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches (each a
            ``{input_name: np.ndarray}`` dict matching ``model``'s graph
            inputs) used to rank each layer's own input channels by
            outlier magnitude -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted) and :func:`onnxsim.load_huggingface_calibration_data`
            (real data, a more representative ranking than random input)
    :param num_samples: random batches to generate when ``calibration_data``
            is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied) and for the block-local
            rotation matrices (a fresh ``numpy.random.Generator`` is
            derived per matched layer, in graph node order)
    :param block_size: elements per quantization block along ``K``,
            matching :func:`onnxsim.quantize_weight_only_int4`'s own
            default
    :param outlier_fraction: fraction of a layer's own input channels (by
            count) ranked as outliers and redistributed one-per-block
            across the permutation, rather than left in their original
            positions
    :param epsilon: floor applied to a token's own max-abs activation
            value before using it as a scale, avoiding a divide-by-zero
            on an all-zero token
    :param providers: onnxruntime execution providers to run calibration on
    :returns: ``model`` with every matched layer's weight and activation
            replaced by permuted-and-rotated, INT4-quantized versions
            (plus the original bias, if any); output tensor name
            unchanged. Layers with a non-constant, non-2-D weight, a
            reduction dimension not divisible by ``block_size``, or no
            calibration activation available, are left untouched; a
            model with no matching layer, or an opset older than 21
            (INT4's tensor type and ``DequantizeLinear``'s ``block_size``
            attribute both need opset 21), is returned unchanged
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if not _has_min_opset(model, 21):
        return model

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
        x_name, w_name, bias_name, weight_transposed = match
        w_init = initializer_map.get(w_name)
        if (
            w_init is None
            or w_init.data_type != onnx.TensorProto.FLOAT
            or len(w_init.dims) != 2
        ):
            continue
        candidates.append((node, x_name, w_name, bias_name, weight_transposed))

    if not candidates:
        return out

    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )

    probe_names = sorted({x_name for _, x_name, _, _, _ in candidates})
    probe_model = _add_probe_outputs(model, probe_names)
    act_absmax: Dict[str, np.ndarray] = {}
    for batch in calibration_data:
        result = backend.run_model(probe_model, batch, providers=providers)
        for name in probe_names:
            x = np.asarray(result[name], dtype=np.float64)
            if x.ndim != 2:
                continue
            m = np.abs(x).max(axis=0)
            act_absmax[name] = (
                m if name not in act_absmax else np.maximum(act_absmax[name], m)
            )

    rng = np.random.default_rng(seed)

    for node, x_name, w_name, bias_name, weight_transposed in candidates:
        absmax = act_absmax.get(x_name)
        if absmax is None:
            continue

        w_init = initializer_map[w_name]
        w = onnx.numpy_helper.to_array(w_init).astype(np.float64)
        w_nk = w if weight_transposed else w.T  # [N, K], output channel first
        n, k = w_nk.shape
        if k % block_size != 0 or absmax.shape[0] != k:
            continue

        u = _build_duquant_rotation(absmax, block_size, outlier_fraction, rng)
        w_tilde_nk = w_nk @ u  # [N, K] -- exact before quantization

        codes_nk, scale_blocks_nk = _quantize_blockwise_int4_with_clip(
            w_tilde_nk, block_size, 1.0
        )
        codes_kn = codes_nk.T.astype(np.int64)  # [K, N], ready for a plain MatMul
        scale_kn = scale_blocks_nk.T.astype(np.float32)  # [K/block_size, N]

        prefix = f"{w_name}_duquant"
        codes_name = _unique_name(f"{prefix}_codes", taken_names)
        codes_tensor = onnx.TensorProto()
        codes_tensor.name = codes_name
        codes_tensor.data_type = onnx.TensorProto.INT4
        codes_tensor.dims.extend([k, n])
        codes_tensor.raw_data = _pack_int4(codes_kn)
        graph.initializer.append(codes_tensor)

        scale_name = _unique_name(f"{prefix}_scale", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(scale_kn, name=scale_name)
        )
        u_name = _unique_name(f"{prefix}_u", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(u.astype(np.float32), name=u_name)
        )
        eps_name = _unique_name(f"{prefix}_eps", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(
                np.array(epsilon, dtype=np.float32), name=eps_name
            )
        )
        seven_name = _unique_name(f"{prefix}_seven", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(
                np.array(7.0, dtype=np.float32), name=seven_name
            )
        )
        clip_min_name = _unique_name(f"{prefix}_clip_min", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(
                np.array(-7.0, dtype=np.float32), name=clip_min_name
            )
        )
        clip_max_name = _unique_name(f"{prefix}_clip_max", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(
                np.array(7.0, dtype=np.float32), name=clip_max_name
            )
        )
        axes_name = _unique_name(f"{prefix}_reduce_axes", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(np.array([-1], dtype=np.int64), name=axes_name)
        )

        new_nodes: List[onnx.NodeProto] = []

        def _new(op_type, inputs, out_suffix, **attrs):
            out_name = _unique_name(f"{prefix}_{out_suffix}", taken_names)
            n_ = onnx.helper.make_node(
                op_type,
                inputs,
                [out_name],
                name=_unique_name(f"{prefix}_{out_suffix}_node", taken_names),
                **attrs,
            )
            new_nodes.append(n_)
            return out_name

        x_rotated = _new("MatMul", [x_name, u_name], "x_rotated")

        # Data-free, per-token round-to-nearest INT4 activation
        # quantization -- same pattern as onnxsim.quarot, simulated via
        # an immediate dequantize (kept in float32) since X isn't
        # constant: scale = max(reduce_max(abs(x_rotated), axis=-1), eps) / 7
        abs_name = _new("Abs", [x_rotated], "x_abs")
        max_name = _new("ReduceMax", [abs_name, axes_name], "x_max", keepdims=1)
        safe_max_name = _new("Clip", [max_name, eps_name], "x_safe_max")
        x_scale = _new("Div", [safe_max_name, seven_name], "x_scale")
        x_scaled = _new("Div", [x_rotated, x_scale], "x_scaled")
        x_rounded = _new("Round", [x_scaled], "x_rounded")
        x_clipped = _new("Clip", [x_rounded, clip_min_name, clip_max_name], "x_clipped")
        x_dequant = _new("Mul", [x_clipped, x_scale], "x_dequant")

        w_dequant = _new(
            "DequantizeLinear",
            [codes_name, scale_name],
            "w_dequant",
            axis=0,
            block_size=block_size,
        )
        core = _new("MatMul", [x_dequant, w_dequant], "core")

        old_output = node.output[0]
        if bias_name is not None:
            final = onnx.helper.make_node(
                "Add",
                [core, bias_name],
                [old_output],
                name=_unique_name(f"{prefix}_bias_add_node", taken_names),
            )
        else:
            final = onnx.helper.make_node(
                "Identity",
                [core],
                [old_output],
                name=_unique_name(f"{prefix}_identity_node", taken_names),
            )
        new_nodes.append(final)

        node_idx = next(i for i, n_ in enumerate(graph.node) if n_ is node)
        for offset, new_node in enumerate(new_nodes):
            graph.node.insert(node_idx + offset, new_node)
        del graph.node[node_idx + len(new_nodes)]

    return out
