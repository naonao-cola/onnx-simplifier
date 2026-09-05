"""AdpQ (Ghaffari et al., 2024, "AdpQ: A Zero-shot Calibration Free Adaptive
Post Training Quantization Method for LLMs", https://arxiv.org/abs/2405.13358).

This repo's other salient/non-salient weight splitters --
:mod:`onnxsim.owq`, :mod:`onnxsim.spqr`, :mod:`onnxsim.gptq`,
:mod:`onnxsim.billm` -- all need real calibration *activations* to decide
which weights matter: each builds (a version of) the layer's Hessian
``H = X^T X`` from a batch of representative inputs, then ranks
sensitivity by some function of ``H`` (or its inverse). AdpQ's own headline
contribution is doing the same kind of split -- separate a small "salient"
set of weights from the rest, quantize only the rest to a low-bit grid --
**with no calibration data whatsoever**. Its closest sibling in spirit is
instead :mod:`onnxsim.nf4`: both decide every quantization choice purely
from a weight tensor's own values, nothing else. Where they part ways is
what that means in practice -- NF4 quantizes every element onto the same
fixed, data-independent 16-point codebook; AdpQ still does a salient/
non-salient *split* (like OWQ/SpQR/GPTQ/BiLLM), just deciding it from the
weight's own magnitude distribution rather than a Hessian.

**Picking which elements are salient.** AdpQ borrows its threshold rule
from Adaptive LASSO regression (Zou, 2006): ordinary LASSO soft-thresholds
every coefficient by the same fixed ``lambda``; Adaptive LASSO instead
scales that threshold per-coefficient by (a power of) that coefficient's
own estimated scale, so naturally larger-magnitude groups keep
comparatively more of their mass. This module applies the same idea
per-group along a layer's reduction dimension (groups of ``group_size``
elements, matching :func:`onnxsim.quantize_weight_only_int4`'s own
block-wise convention): for each ``(output channel, group)`` slice, a
robust scale estimate

    sigma_hat = 1.4826 * MAD(w_group)

(the median absolute deviation, scaled by the usual constant that makes it
a consistent estimator of the standard deviation for normally-distributed
data -- robust because a handful of large weights in the group can't drag a
*median*-based estimate the way they would a max-abs or plain-std one), and
an adaptive soft-threshold derived from it,

    threshold = lambda_ * sigma_hat ** (1 - gamma)

directly mirroring Adaptive LASSO's own per-coefficient threshold
``lambda * scale^(1-gamma)``. ``gamma in [0, 1)`` is the adaptive weighting
exponent: at ``gamma = 0`` this is just an ordinary constant multiple of
each group's own robust sigma (a plain robust z-score threshold); as
``gamma`` grows, the exponent on ``sigma_hat`` shrinks towards 0 and the
threshold flattens out towards the constant ``lambda_`` regardless of the
group's own scale -- i.e. groups with a naturally wider spread get
comparatively *more* of their mass counted salient, the same
scale-adaptive asymmetry Adaptive LASSO's weighting scheme produces for
regression coefficients. Every element whose magnitude exceeds its own
group's threshold is salient; the rest are not. Unlike
:mod:`onnxsim.spqr`'s ``outlier_fraction`` (a fixed target count picked in
advance), the number of salient elements here falls out of the threshold
rule itself and can differ from group to group and layer to layer -- the
same "adaptive" character the paper's own name refers to.

**Minimizing the weight distribution's own KL divergence, not the layer's
output error.** Every non-salient group is quantized onto a uniform
symmetric INT4 grid, using a scale computed *excluding* that group's own
salient elements (the same "exclude the outliers from the scale, so
everything else quantizes tighter" trick :mod:`onnxsim.spqr` uses for its
own per-element outliers) -- this keeps the quantized sub-population's own
empirical distribution close to the original weights' distribution
restricted to that same sub-population, rather than optimizing (as
:mod:`onnxsim.gptq`/:mod:`onnxsim.owq` do) for the layer's *output*
reconstruction error against real activations, since no activations are
available here to optimize against in the first place. Salient elements are
restored to exact float32 precision via a sparse correction overlay --
``W_reconstructed = block_quantized(W) + sparse_correction`` -- the same
``ScatterND``-over-``ConstantOfShape`` mechanics :mod:`onnxsim.spqr` already
uses, chosen here (over e.g. a separate INT8 sub-tensor for salient
elements) because it needs no second quantization grid or extra dequant
branch: salient elements simply reconstruct exactly, non-salient elements
reconstruct through the INT4 grid, and both paths share one
``DequantizeLinear``.
"""

from __future__ import annotations

from typing import List, Union

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from onnxsim.adaround import _pack_int4
from onnxsim.bias_correction import _all_names, _unique_name
from onnxsim.quip_sharp import _match_matmul_like


def _has_min_opset(model: onnx.ModelProto, min_version: int) -> bool:
    return any(
        o.domain in ("", "ai.onnx") and o.version >= min_version
        for o in model.opset_import
    )


def _adaptive_thresholds(
    blocks: np.ndarray, lambda_: float, gamma: float
) -> np.ndarray:
    """Per-``(row, group)`` Adaptive-LASSO-style soft threshold -- see this
    module's own docstring. ``blocks`` is ``[N, num_groups, group_size]``;
    returns ``[N, num_groups]``.
    """
    median = np.median(blocks, axis=2, keepdims=True)
    mad = np.median(np.abs(blocks - median), axis=2)
    sigma_hat = np.maximum(1.4826 * mad, 1e-12)
    return lambda_ * sigma_hat ** (1.0 - gamma)


def quantize_weight_only_adpq(
    model: Union[str, onnx.ModelProto],
    group_size: int = 128,
    lambda_: float = 3.0,
    gamma: float = 0.3,
) -> onnx.ModelProto:
    """Applies AdpQ-style calibration-free adaptive INT4 quantization (see
    this module's own docstring) to every MatMul/vanilla-Gemm layer with a
    constant 2-D float32 weight whose reduction dimension ``K`` is divisible
    by ``group_size``. Needs no calibration data: every quantization
    decision -- which elements are salient, and the scale used for the
    rest -- comes from the weight tensor's own values.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param group_size: elements per quantization group along ``K``, matching
            :func:`onnxsim.quantize_weight_only_int4`'s own block-wise
            granularity
    :param lambda_: overall threshold scale -- at ``gamma=0`` this is the
            number of robust sigmas (see ``sigma_hat`` in this module's own
            docstring) above which a weight counts as salient
    :param gamma: Adaptive LASSO exponent in ``[0, 1)`` controlling how much
            a group's own scale flattens the threshold -- ``0`` is an
            ordinary constant multiple of that group's robust sigma; values
            closer to ``1`` let wider-spread groups keep comparatively more
            salient mass (see this module's own docstring)
    :returns: ``model`` with every matched layer's weight replaced by
            group-wise INT4 codes (scale excluding that group's own salient
            elements) plus a sparse exact correction restoring salient
            elements to their original float32 value -- see the module
            docstring's diagram in :mod:`onnxsim.spqr` for the shared
            ``DequantizeLinear``/``ScatterND`` graph shape. Layers with a
            non-constant, non-2-D weight, or a reduction dimension not
            divisible by ``group_size``, are left untouched; a model with no
            matching layer, or an opset older than 21 (INT4's tensor type
            and ``DequantizeLinear``'s ``block_size`` attribute both need
            opset 21), is returned unchanged.
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

    candidates = []
    for node in list(graph.node):
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

    for node, x_name, w_name, bias_name, weight_transposed in candidates:
        w_init = initializer_map[w_name]
        w = onnx.numpy_helper.to_array(w_init).astype(np.float64)
        w_nk = w if weight_transposed else w.T  # [N, K], output channel first
        n, k = w_nk.shape
        if k % group_size != 0:
            continue

        num_groups = k // group_size
        blocks = w_nk.reshape(n, num_groups, group_size)

        threshold = _adaptive_thresholds(blocks, lambda_, gamma)  # [N, num_groups]
        threshold_full = np.repeat(threshold, group_size, axis=1)  # [N, K]
        salient_mask = np.abs(w_nk) > threshold_full

        mask_blocks = (~salient_mask).reshape(n, num_groups, group_size)
        abs_masked = np.where(mask_blocks, np.abs(blocks), 0.0)
        scale_blocks = (
            np.maximum(abs_masked.max(axis=2), 1e-12) / 7.0
        )  # [N, num_groups]
        scale_full = np.repeat(scale_blocks, group_size, axis=1)  # [N, K]

        codes_nk = np.clip(np.round(w_nk / scale_full), -7.0, 7.0)
        dequant_nk = codes_nk * scale_full

        prefix = f"{w_name}_adpq"
        codes_kn = codes_nk.T.astype(np.int64)  # [K, N]
        scale_kn = scale_blocks.T.astype(np.float32)  # [K/group_size, N]

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

        w_dequant = _new(
            "DequantizeLinear",
            [codes_name, scale_name],
            "w_dequant",
            axis=0,
            block_size=group_size,
        )

        salient_rows, salient_cols = np.nonzero(salient_mask)
        num_salient = salient_rows.shape[0]
        if num_salient > 0:
            correction_values = (
                w_nk[salient_rows, salient_cols]
                - dequant_nk[salient_rows, salient_cols]
            )
            # [K, N]-layout indices, matching codes_kn/scale_kn's own
            # transposed storage: index[i] = [k_pos, n_pos].
            salient_indices_kn = np.stack([salient_cols, salient_rows], axis=1)

            indices_name = _unique_name(f"{prefix}_salient_indices", taken_names)
            graph.initializer.append(
                onnx.numpy_helper.from_array(
                    salient_indices_kn.astype(np.int64), name=indices_name
                )
            )
            values_name = _unique_name(f"{prefix}_salient_values", taken_names)
            graph.initializer.append(
                onnx.numpy_helper.from_array(
                    correction_values.astype(np.float32), name=values_name
                )
            )
            shape_name = _unique_name(f"{prefix}_shape", taken_names)
            graph.initializer.append(
                onnx.numpy_helper.from_array(
                    np.array([k, n], dtype=np.int64), name=shape_name
                )
            )

            zeros = _new(
                "ConstantOfShape",
                [shape_name],
                "zeros",
                value=onnx.numpy_helper.from_array(np.array([0.0], dtype=np.float32)),
            )
            correction = _new(
                "ScatterND", [zeros, indices_name, values_name], "correction"
            )
            w_reconstructed = _new("Add", [w_dequant, correction], "w_reconstructed")
        else:
            w_reconstructed = w_dequant

        core = _new("MatMul", [x_name, w_reconstructed], "core")

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
