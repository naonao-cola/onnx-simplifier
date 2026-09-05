"""SVDQuant (Li, Lin, Zhang, et al., 2024, "SVDQuant: Absorbing Outliers by
Low-Rank Component for 4-Bit Diffusion Models",
https://arxiv.org/abs/2411.05007 -- MIT Han Lab, also shipped as the
"Nunchaku" inference engine). onnxsim ports the algorithm's *weight-side
decomposition*, not any framework's code (the paper's own reference
implementation is a diffusers/PyTorch pipeline plus a hand-written CUDA
kernel library, with no ONNX export path -- the same rationale as
:mod:`onnxsim.awq`/:mod:`onnxsim.gptq`).

**What SVDQuant actually does, confirmed against the paper (not recalled
from memory -- see this module's own PR description for the verification).**
Ordinary block-wise round-to-nearest quantization (what
:func:`onnxsim.quantize_weight_only_int4` does) struggles once a weight has
been *smoothed* (:mod:`onnxsim.smoothquant`'s own per-channel migration,
which the SVDQuant paper explicitly builds on as its first step): smoothing
moves outlier difficulty from activations onto the weight, so the weight
itself becomes harder to quantize even as the activation gets easier. The
paper's fix: before quantizing the now-more-outlier-heavy smoothed weight
``W'``, take its SVD and peel off a small-rank ``r`` "low-rank branch"
(``L1 @ L2``, built from the ``r`` dominant singular values/vectors -- the
directions an outlier-heavy matrix concentrates its largest singular values
in) and keep that branch at full precision. What's left, the residual
``R = W' - L1 @ L2``, has had its dominant/outlier structure removed and is
now much more uniform, so quantizing *it* to INT4 (instead of ``W'``
directly) loses far less. At inference, the layer's original ``Y = X @ W``
becomes two branches summed: ``Y = X @ dequant(quantize(R)) + (X @ L1) @
L2`` -- the same "keep a small extra piece at full precision, add it back
via extra small MatMuls" shape as :mod:`onnxsim.low_rank_compensation`, but
computed *before* quantizing (from the weight's own dominant structure)
rather than *after* (from an already-fixed quantization's leftover error).
That distinction is the whole point: LoRC's low-rank term chases whatever
error round-to-nearest happened to leave behind; SVDQuant's low-rank term
prevents most of that error from being *created* in the first place, by
routing the hardest-to-quantize part of the weight around the quantizer
entirely.

This module wires the two existing onnxsim pieces together: it optionally
runs :func:`onnxsim.apply_smoothquant` first (the paper's own preprocessing
step -- pass ``smooth_alpha=None`` to skip it and decompose the raw weight
instead), then for every matched MatMul/vanilla-Gemm layer with a constant
2-D float32 weight computes the low-rank/residual split above and quantizes
the residual with the same block-wise scheme
:func:`onnxsim.quantize_weight_only_int4` uses (reusing
:mod:`onnxsim.omniquant`'s ``_quantize_blockwise_int4_with_clip`` with
``clip_ratio=1.0``, and :mod:`onnxsim.adaround`'s ``_pack_int4`` for the
INT4 byte packing).

**Deliberately not ported** (see this module's own docstring for the
paper's full scope): the paper's headline result is *activation* quantization
too (W4A4, needed for the compute-bound diffusion-model speedups it
measures) -- this module, like every other onnxsim weight-only quantizer,
only quantizes the weight; activations stay float32, so this is closer to
the paper's own W4A16 ablation than its full W4A4 pipeline. Also not
ported: Nunchaku, the paper's specialized CUDA inference engine that fuses
the low-rank and residual branches' kernels to make the extra branch nearly
free on real hardware (onnxsim emits the low-rank branch as ordinary
``MatMul`` nodes -- correct, but with none of that fusion); the paper's
optional iterative refinement of the low-rank branch (repeatedly
re-decomposing ``W' - quantize(R)`` for a few rounds to squeeze out more
accuracy); the paper's noted GPTQ-for-residual ablation (this module always
quantizes the residual via plain round-to-nearest, matching
:func:`onnxsim.quantize_weight_only_int4`'s own scheme -- run
:func:`onnxsim.apply_gptq`/:func:`onnxsim.apply_awq` afterwards against
this module's own residual for that refinement, the same composable way
every other onnxsim INT4 refinement pass is meant to be layered); and the
paper's diffusion-model-specific targeting -- this module, like onnxsim's
own :func:`quantize_weight_only_int4`, is architecture-agnostic and targets
any MatMul/Gemm the same way.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from onnxsim.adaround import _pack_int4
from onnxsim.bias_correction import _all_names, _unique_name
from onnxsim.calibration import Tensors
from onnxsim.omniquant import _quantize_blockwise_int4_with_clip
from onnxsim.quip_sharp import _match_matmul_like
from onnxsim.smoothquant import apply_smoothquant


def _has_min_opset(model: onnx.ModelProto, min_version: int) -> bool:
    return any(
        o.domain in ("", "ai.onnx") and o.version >= min_version
        for o in model.opset_import
    )


def apply_svdquant(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    rank: int = 32,
    block_size: int = 32,
    smooth_alpha: Optional[float] = 0.5,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Applies SVDQuant-style low-rank-branch-plus-residual-quantization to
    every MatMul/vanilla-Gemm layer with a constant 2-D float32 weight whose
    reduction dimension ``K`` is divisible by ``block_size``. See this
    module's own docstring for the technique and its scope.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches, forwarded to
            :func:`onnxsim.apply_smoothquant` for the outlier-migration
            preprocessing step (ignored entirely when ``smooth_alpha`` is
            ``None``, since the low-rank/residual split itself needs no
            calibration data -- it's a static decomposition of the weight).
            Each batch is a ``{input_name: np.ndarray}`` dict matching
            ``model``'s graph inputs -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted) and :func:`onnxsim.load_huggingface_calibration_data`
    :param num_samples: random batches to generate when ``calibration_data``
            is omitted and ``smooth_alpha`` is not ``None``
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied, or if ``smooth_alpha`` is
            ``None``)
    :param rank: the low-rank branch's rank ``r`` (clamped to
            ``min(r, N, K)`` per layer; the paper's own experiments use 16
            or 32); larger values move more of the weight's dominant
            structure into the full-precision branch, at the cost of two
            proportionally larger extra ``MatMul`` nodes
    :param block_size: elements per quantization block along the residual's
            reduction dimension, matching
            :func:`onnxsim.quantize_weight_only_int4`'s own granularity
    :param smooth_alpha: migration strength forwarded to
            :func:`onnxsim.apply_smoothquant` as its own ``alpha`` (see that
            module's docstring); pass ``None`` to skip smoothing entirely
            and decompose the raw weight instead
    :param providers: onnxruntime execution providers to run the smoothing
            step's calibration on (ignored if ``smooth_alpha`` is ``None``)
    :returns: a model with every matched layer's weight replaced by a
            block-wise INT4-quantized residual plus a full-precision
            low-rank correction (``L1``/``L2`` initializers and two extra
            ``MatMul`` nodes summed into the layer's output); layers with a
            non-constant, non-2-D weight, or a reduction dimension not
            divisible by ``block_size``, are left untouched. A model with no
            matching layer, or an opset older than 21 (INT4's tensor type
            and ``DequantizeLinear``'s ``block_size`` attribute both need
            opset 21), is returned unchanged.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if not _has_min_opset(model, 21):
        return model

    if smooth_alpha is not None:
        model = apply_smoothquant(
            model,
            calibration_data=calibration_data,
            num_samples=num_samples,
            seed=seed,
            alpha=smooth_alpha,
            providers=providers,
        )

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
        w_kn = w.T if weight_transposed else w  # [K, N], matching X @ W
        k, n = w_kn.shape
        if k % block_size != 0:
            continue

        r = min(rank, k, n)
        if r <= 0:
            continue

        u, s, vt = np.linalg.svd(w_kn, full_matrices=False)
        l1_kr = (u[:, :r] * s[np.newaxis, :r]).astype(np.float32)  # [K, r]
        l2_rn = vt[:r, :].astype(np.float32)  # [r, N]
        residual_kn = w_kn - (l1_kr.astype(np.float64) @ l2_rn.astype(np.float64))

        residual_nk = residual_kn.T  # [N, K], output channel first
        codes_nk, scale_blocks = _quantize_blockwise_int4_with_clip(
            residual_nk, block_size, 1.0
        )

        prefix = f"{w_name}_svdquant"
        codes_kn = codes_nk.T.astype(np.int64)  # [K, N]
        scale_kn = scale_blocks.T.astype(np.float32)  # [K / block_size, N]

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

        l1_name = _unique_name(f"{prefix}_l1", taken_names)
        graph.initializer.append(onnx.numpy_helper.from_array(l1_kr, name=l1_name))
        l2_name = _unique_name(f"{prefix}_l2", taken_names)
        graph.initializer.append(onnx.numpy_helper.from_array(l2_rn, name=l2_name))

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
            block_size=block_size,
        )
        base = _new("MatMul", [x_name, w_dequant], "base")
        lowrank_tmp = _new("MatMul", [x_name, l1_name], "lowrank_tmp")
        lowrank = _new("MatMul", [lowrank_tmp, l2_name], "lowrank")
        summed = _new("Add", [base, lowrank], "sum")

        old_output = node.output[0]
        if bias_name is not None:
            final = onnx.helper.make_node(
                "Add",
                [summed, bias_name],
                [old_output],
                name=_unique_name(f"{prefix}_bias_add_node", taken_names),
            )
        else:
            final = onnx.helper.make_node(
                "Identity",
                [summed],
                [old_output],
                name=_unique_name(f"{prefix}_identity_node", taken_names),
            )
        new_nodes.append(final)

        node_idx = next(i for i, n_ in enumerate(graph.node) if n_ is node)
        for offset, new_node in enumerate(new_nodes):
            graph.node.insert(node_idx + offset, new_node)
        del graph.node[node_idx + len(new_nodes)]

    return out
