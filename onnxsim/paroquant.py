"""ParoQuant (Liang et al., 2025, "ParoQuant: Pairwise Rotation Quantization
for Efficient Reasoning LLM Inference", https://arxiv.org/abs/2511.10645,
ICLR 2026). onnxsim ports the algorithm, not any framework's code, per the
same rationale as :mod:`onnxsim.awq`/:mod:`onnxsim.gptq`/:mod:`onnxsim.spinquant`
(ParoQuant's own reference implementation optimizes rotation angles against
live PyTorch weights with no ONNX export path).

**How this differs from** :mod:`onnxsim.spinquant`: both modules conjugate a
weight by an orthogonal rotation before block-wise INT4 quantization, to
make the rotated weight less outlier-concentrated and therefore cheaper to
quantize -- the same rotate-then-quantize contract :mod:`onnxsim.quip_sharp`
and :mod:`onnxsim.duquant` also share. :mod:`onnxsim.spinquant` fits one
**dense** ``[K, K]`` rotation matrix per layer (SpinQuant's own "R1-only"
substitute: the eigenvector basis of the calibration-activation covariance),
which at inference time costs a full ``K x K`` matmul against the
activation. ParoQuant's own distinguishing contribution is replacing that
dense rotation with a set of many independent, cheap **2x2 (pairwise,
Givens) rotations** instead: each one mixes exactly *one pair* of channels
and leaves every other channel untouched, so the "rotation" as a whole is
a ``[K, K]`` matrix that is block-diagonal (2x2 blocks on the chosen pairs,
identity everywhere else) rather than dense. The paper reports this keeps
the rotation's own compute overhead under 10% of the layer's matmul cost
(a block-diagonal matmul only ever touches 2 columns per output column,
versus every dense rotation column touching all ``K``), while still
meaningfully narrowing the per-group dynamic range a dense rotation
targets -- narrower than doing nothing, even if not quite as narrow as a
fully dense rotation's.

A Givens rotation is the 2x2 orthogonal matrix
``[[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]`` applied to
exactly two coordinates of a vector; applying one to a pair of channels
``(i, j)`` of an activation ``X`` and the *same* rotation to the *same*
pair of rows of the weight is an exact algebraic identity
(``(X @ R) @ (R.T @ W) == X @ W`` for any orthogonal ``R``, here block-
diagonal with 2x2 Givens blocks on the chosen pairs and identity
elsewhere -- a block-diagonal matrix of orthogonal blocks is itself
orthogonal), so applying a whole set of independent pairwise rotations is
still lossless before quantization, the same "provably exact migration,
then quantize" contract :mod:`onnxsim.spinquant`/:mod:`onnxsim.smoothquant`
already use.

This module's pairing is the cheapest, most hardware-friendly one the paper
describes: fixed adjacent channels within each quantization block --
``(0, 1), (2, 3), (4, 5), ...`` -- rather than a data-chosen pairing (no
extra bookkeeping to describe or transmit a permutation, unlike
:mod:`onnxsim.duquant`'s own calibration-driven channel reassignment). Each
pair's own rotation angle is then fit to that layer's own weight via a
small grid search over candidate angles in ``[-pi/4, pi/4]`` (covering
every distinct 2x2 rotation, since a Givens rotation is pi-periodic and
symmetric about pi/4), minimizing that pair's own contribution to its
quantization block's INT4 round-to-nearest reconstruction error -- the same
"grid-search a scalar against measured reconstruction error, not a
closed-form guarantee" style :mod:`onnxsim.awq` already uses for its own
per-channel scale, applied here per-pair instead of once per layer. Pairs
within the same block are optimized in a fixed left-to-right order, each
one seeing the already-rotated state of every earlier pair in its block
(their combined effect changes the block's own outlier structure, so
sequencing the search lets later pairs adapt to it), which the paper's own
gradient-free per-pair angle search also does.

Finally, this module combines the pairwise rotation with a SmoothQuant-style
(:mod:`onnxsim.smoothquant`) per-channel scale migration -- ParoQuant's own
second ingredient, evening out channel magnitudes before the rotation search
runs, using the exact same closed-form, alpha-parameterized formula
(``s_j = max(|X_j|) ** alpha / max(|W_j|) ** (1 - alpha)``) and the same
"one fixed global alpha, not searched" practice :mod:`onnxsim.smoothquant`
already documents. Combining a diagonal scale with an orthogonal rotation
stays exact for the same reason either piece alone is: ``X @ W ==
((X / s) @ R) @ (R.T @ (diag(s) @ W))`` for any invertible diagonal
``diag(s)`` and orthogonal ``R``, regardless of composition order.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from onnxsim import backend
from onnxsim.adaround import _pack_int4
from onnxsim.bias_correction import _add_probe_outputs, _all_names, _unique_name
from onnxsim.calibration import Tensors, generate_random_calibration_data
from onnxsim.omniquant import _quantize_blockwise_int4_with_clip
from onnxsim.quip_sharp import _match_matmul_like


def _has_min_opset(model: onnx.ModelProto, min_version: int) -> bool:
    return any(
        o.domain in ("", "ai.onnx") and o.version >= min_version
        for o in model.opset_import
    )


def _fit_paroquant_pairwise_rotation(
    w_nk: np.ndarray, block_size: int, num_angle_steps: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Fits ParoQuant's block-diagonal pairwise (Givens) rotation, ``[K, K]``
    -- one independent 2x2 rotation per adjacent channel pair
    ``(0, 1), (2, 3), ...`` within each ``block_size``-wide quantization
    block, identity everywhere else -- against ``w_nk`` ([N, K], output
    channel first; already SmoothQuant-scaled by the caller). See this
    module's own docstring. Returns ``(r, w_rotated_nk)``: the rotation
    matrix and ``w_nk @ r`` (equal by construction, returned together since
    the fit already computes the rotated weight incrementally).

    Pairs are processed in a fixed left-to-right order within each block;
    each pair's own angle is grid-searched over ``num_angle_steps`` points
    in ``[-pi/4, pi/4]`` (``0`` always included when ``num_angle_steps`` is
    odd, so a pair the search can't improve keeps its original, unrotated
    columns) to minimize the mean squared INT4 round-to-nearest
    reconstruction error of its own block, evaluated on the block's
    already-partially-rotated state so later pairs in the same block adapt
    to earlier ones.
    """
    n, k = w_nk.shape
    assert block_size % 2 == 0 and k % block_size == 0
    r = np.eye(k, dtype=np.float64)
    w_work = w_nk.astype(np.float64).copy()
    thetas = np.linspace(-np.pi / 4.0, np.pi / 4.0, num_angle_steps)

    for start in range(0, k, block_size):
        end = start + block_size
        for i in range(start, end, 2):
            j = i + 1
            col_i = w_work[:, i].copy()
            col_j = w_work[:, j].copy()

            best_err: Optional[float] = None
            best_theta = 0.0
            best_ci, best_cj = col_i, col_j
            for theta in thetas:
                c, s = np.cos(theta), np.sin(theta)
                new_i = c * col_i - s * col_j
                new_j = s * col_i + c * col_j
                block = w_work[:, start:end].copy()
                block[:, i - start] = new_i
                block[:, j - start] = new_j
                codes, scale_blocks = _quantize_blockwise_int4_with_clip(
                    block, block_size, 1.0
                )
                recon = codes * np.repeat(scale_blocks, block_size, axis=1)
                err = float(np.mean((block - recon) ** 2))
                if best_err is None or err < best_err:
                    best_err, best_theta = err, theta
                    best_ci, best_cj = new_i, new_j

            w_work[:, i], w_work[:, j] = best_ci, best_cj
            c, s = np.cos(best_theta), np.sin(best_theta)
            r[i, i], r[i, j] = c, s
            r[j, i], r[j, j] = -s, c

    return r, w_work


def apply_paroquant(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    block_size: int = 32,
    alpha: float = 0.5,
    num_angle_steps: int = 9,
    epsilon: float = 1e-5,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Applies ParoQuant-style channel-wise SmoothQuant scaling plus
    pairwise (Givens) rotation preprocessing (see this module's own
    docstring) followed by block-wise INT4 quantization to every
    MatMul/vanilla-Gemm layer with a constant 2-D float32 weight whose
    reduction dimension ``K`` is divisible by ``block_size``.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches (each a
            ``{input_name: np.ndarray}`` dict matching ``model``'s graph
            inputs) used to measure each layer's own input channel
            activation range for the SmoothQuant-style scale -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted) and :func:`onnxsim.load_huggingface_calibration_data`
            (real data, a more representative scale than random input)
    :param num_samples: random batches to generate when ``calibration_data``
            is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param block_size: elements per quantization block along ``K``, and per
            pairwise-rotation grouping (must be even), matching
            :func:`onnxsim.quantize_weight_only_int4`'s own default
    :param alpha: the channel-scale migration strength, same meaning and
            default as :func:`onnxsim.apply_smoothquant`'s own ``alpha``
    :param num_angle_steps: grid points per pair for the Givens angle
            search, evenly spaced over ``[-pi/4, pi/4]`` inclusive; higher
            values search more finely at proportionally more cost (one
            block re-quantization and reconstruction-error measurement per
            candidate, per pair)
    :param epsilon: floor applied to every per-channel activation/weight
            max-abs value before computing the SmoothQuant-style scale,
            avoiding a divide-by-zero on an all-zero channel
    :param providers: onnxruntime execution providers to run calibration on
    :returns: ``model`` with every matched layer's weight and activation
            replaced by scaled-and-pairwise-rotated, INT4-quantized
            versions (plus the original bias, if any); output tensor name
            unchanged. Layers with a non-constant, non-2-D weight, a
            reduction dimension not divisible by ``block_size``, or no
            calibration activation available, are left untouched; a model
            with no matching layer, an odd ``block_size``, or an opset
            older than 21 (INT4's tensor type and ``DequantizeLinear``'s
            ``block_size`` attribute both need opset 21), is returned
            unchanged
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if not _has_min_opset(model, 21) or block_size % 2 != 0:
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

        act_channel = np.maximum(absmax, epsilon)
        weight_channel = np.maximum(np.abs(w_nk).max(axis=0), epsilon)  # [K]
        s = (act_channel**alpha) / (weight_channel ** (1.0 - alpha))
        s = np.maximum(s, epsilon)

        w_smooth_nk = w_nk * s[np.newaxis, :]
        r, w_tilde_nk = _fit_paroquant_pairwise_rotation(
            w_smooth_nk, block_size, num_angle_steps
        )

        codes_nk, scale_blocks_nk = _quantize_blockwise_int4_with_clip(
            w_tilde_nk, block_size, 1.0
        )
        codes_kn = codes_nk.T.astype(np.int64)  # [K, N], ready for a plain MatMul
        scale_kn = scale_blocks_nk.T.astype(np.float32)  # [K/block_size, N]

        prefix = f"{w_name}_paroquant"
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
        r_name = _unique_name(f"{prefix}_r", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(r.astype(np.float32), name=r_name)
        )
        inv_s_name = _unique_name(f"{prefix}_inv_scale", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array((1.0 / s).astype(np.float32), name=inv_s_name)
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

        x_scaled = _new("Mul", [x_name, inv_s_name], "x_scaled")
        x_rotated = _new("MatMul", [x_scaled, r_name], "x_rotated")
        w_dequant = _new(
            "DequantizeLinear",
            [codes_name, scale_name],
            "w_dequant",
            axis=0,
            block_size=block_size,
        )
        core = _new("MatMul", [x_rotated, w_dequant], "core")

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
