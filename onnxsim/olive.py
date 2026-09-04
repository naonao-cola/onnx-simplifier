"""OliVe (Guo, Zhang, Yang, Liu, Wang, Chen, Wu, Wang, Liu, Guo, Zhu, ISCA
2023, "OliVe: Accelerating Large Language Models via Hardware-friendly
Outlier-Victim Pair Quantization", https://arxiv.org/abs/2304.07493).
onnxsim ports the algorithm, not any framework's or hardware's code, per the
same rationale as :mod:`onnxsim.awq`/:mod:`onnxsim.gptq` (OliVe's own
contribution is a memory *encoding* co-designed with a systolic-array
processing element -- this module reproduces its quantization decisions in
plain numpy/ONNX ops, not that bit layout or any tensor-core format).

(This is unrelated to Microsoft's "Olive" ONNX optimization/export
toolchain mentioned elsewhere in this repo's comments -- see e.g.
``onnxsim/pruning.py``'s references to it. That is a different project; this
module is the outlier-victim-pair quantization *paper* above.)

**The paper's own idea.** An ordinary group-wise quantizer (what
:func:`onnxsim.quantize_weight_only_int4` does) picks one scale per block of
the reduction dimension, sized to the block's largest-magnitude element. A
single outlier forces that scale wide, wasting resolution on every other,
ordinary-magnitude element sharing the block -- the same problem
:mod:`onnxsim.spqr`/:mod:`onnxsim.hqq` also address, each a different way.
OliVe's own insight (Section 3 of the paper): a normal-magnitude weight
sitting immediately *next to* an outlier in memory can usually afford to
lose almost all of its own precision without materially hurting the layer's
overall reconstruction error, because the outlier's own error dominates the
block's total error far more than any single ordinary element's rounding
does. So instead of giving every element a fixed, equal bit-width, OliVe
pairs each outlier with its immediate memory-adjacent neighbor (the
"victim") and *locally* re-negotiates that one pair's own bit budget: the
outlier gets extra bits (a wider code, at the ordinary elements' own
quantization step, so it covers a larger dynamic range without clipping),
paid for by re-quantizing the victim far more coarsely than the group's
ordinary elements. The paper's own name for this is "Outlier-Victim Pair"
(OVP) encoding.

**How this differs from onnxsim's other outlier-aware weight-only
quantizers.** :mod:`onnxsim.spqr` excludes outlier elements from a block's
scale computation and stores an *exact* float32 correction for them,
``W_reconstructed = block_quantized(W) + sparse_correction`` -- a
correction stored *alongside* (added on top of) the quantized value, with
unbounded precision and no relationship to any other specific element's own
bit budget. :mod:`onnxsim.owq` keeps whole salient *columns* -- not
individual elements -- at exact float32 precision via a similar additive
correction. OliVe does neither: nothing is ever restored to exact float
precision (every element, outlier included, stays quantized, just at a
locally negotiated bit-width), the unit of "outlier handling" is one
*element* rather than a whole column, and -- the real distinguishing
mechanism -- an outlier's extra resolution is funded specifically by its own
paired neighbor's bits, not by an unrelated global sparse/full-precision
allowance. Two adjacent elements are the whole unit of account.

**This module's OVP encoding.** For each ``(output channel, block)`` group
of ``block_size`` weight elements along the reduction axis (``block_size``
must be even):

1. **Outlier detection.** ``typical_scale = median(|w|)`` over the block (a
   robust central-tendency estimate, unaffected by the few outliers it is
   meant to characterize -- unlike an absmax-based scale, computing this
   doesn't need outliers excluded first). An element is an outlier if
   ``|w| > outlier_threshold * typical_scale``.

2. **Pairing.** Elements are grouped into non-overlapping adjacent pairs
   ``(2i, 2i+1)`` within the block (memory-adjacent, matching the paper's
   own PE-local pairing granularity). A pair with *exactly one* outlier
   member becomes an OVP pair: the outlier member gets the outlier
   treatment below, the other member becomes its victim. A pair with *zero*
   or *two* outlier members is declined -- there is no unpaired non-outlier
   neighbor to act as victim, or nothing to rescue -- and both members fall
   back to plain group-wide quantization (bullet 3, ``ordinary``), taking
   the same clipping/error a plain group-wide quantizer would.

3. **Bit accounting (``bits`` ordinary group elements get by default 4).**
   Three code widths coexist, all stored as ``INT8`` (ONNX has no native
   3-/5-bit packed tensor type, so codes are stored one signed byte each --
   the same convention :mod:`onnxsim.billm` uses for its own sub-4-bit
   codes; the *scale* arrays, not the code dtype, are what stay compact,
   see below):

   - ``ordinary`` (declined pairs, and any element not part of an OVP
     pair): ``bits``-bit signed code, ``qmax = 2**(bits-1) - 1`` (7 for
     ``bits=4``), against the block's own ``base_scale`` -- the ordinary
     group-wide scale, computed from the block's *non-outlier* elements
     only (so a lone outlier that did get paired, or one that was declined
     and is about to clip badly, never drags this scale wide for its
     neighbors -- the same exclude-outliers-from-the-scale idea
     :mod:`onnxsim.spqr` also uses).
   - ``outlier`` (the outlier member of an OVP pair): a ``bits + 1``-bit
     signed code, ``qmax = 2**bits - 1`` (15 for ``bits=4``), against
     ``outlier_scale`` -- a *second*, per-``(output channel, block)`` scale
     computed the same absmax/qmax way as ``base_scale`` but from that
     block's outlier elements only. Because ``outlier_scale`` is fit to the
     block's actual outlier magnitudes (rather than reusing ``base_scale``,
     which would clip), and the extra bit doubles the code count, an
     outlier reconstructs with much lower relative error than group-wide
     quantization at ``base_scale``/``qmax`` would give it -- without
     needing a per-element (rather than per-block) scale the way an exact
     correction term would.
   - ``victim`` (the non-outlier member of an OVP pair): a ``bits - 1``-bit
     signed code, ``qmax = 2**(bits-2) - 1`` (3 for ``bits=4``), against the
     *same* ``base_scale`` as ordinary elements -- deliberately coarse: the
     victim shares its neighbors' quantization step but is confined to far
     fewer of that grid's levels, discarding most of its own precision.

   By construction, ``ordinary_bits + ordinary_bits == (bits) + (bits) ==
   2*bits``, and ``outlier_bits + victim_bits == (bits+1) + (bits-1) ==
   2*bits`` as well -- an OVP pair costs exactly the same total bit budget
   as two ordinary group-wide-quantized elements would, satisfying this
   repo's own convention of describing a technique's real compression ratio
   honestly (see :mod:`onnxsim.spqr`/:mod:`onnxsim.billm`): no bits are
   conjured from nowhere, they are strictly reallocated within each pair.

**ONNX encoding.** Every block gets a compact ``base_scale`` (one float32
per ``(output channel, block)``, the same overhead as plain
:func:`onnxsim.quantize_weight_only_int4`) and, only for blocks containing
at least one OVP pair, a second compact ``outlier_scale`` of the same
shape. A dense ``INT8`` ``outlier_mask`` (1 at each OVP pair's outlier
position, 0 elsewhere -- one byte per element, the same order of overhead
as a second code tensor, not a second float32 weight's worth of storage)
selects, per element, which of two block-wise dequantizations to keep:

    Before:
      Y = MatMul(X, W) [+ bias]                  -- W constant, [K, N], float32

    After:
      Wq           = <int8 codes: ordinary/victim bits against BaseScale,
                      outlier bits against OutlierScale>
      BaseScale    = <float32, [K/block_size, N]>
      OutlierScale = <float32, [K/block_size, N]>          -- if any OVP pair
      OutlierMask  = <int8, [K, N], 1 at OVP outlier positions>
      BaseDequant    = DequantizeLinear(Wq, BaseScale, axis=0, block_size=block_size)
      OutlierDequant = DequantizeLinear(Wq, OutlierScale, axis=0, block_size=block_size)
      Wreconstructed = Where(Cast(OutlierMask, BOOL), OutlierDequant, BaseDequant)
      Y = MatMul(X, Wreconstructed) [+ bias]

A layer with no OVP pair anywhere skips ``OutlierScale``/``OutlierMask``/
``Where`` entirely and dequantizes directly from ``BaseScale`` (plain
group-wide quantization, the natural degenerate case). Ordinary ONNX ops
only (``DequantizeLinear`` with a ``block_size`` attribute needs opset 21,
matching :mod:`onnxsim.spqr`/:mod:`onnxsim.hqq`; ``Where``/``Cast`` need
nothing newer), no contrib op, and calibration-free -- like
:mod:`onnxsim.hqq`, every decision here comes from the weight tensor's own
values, no activation probing needed.
"""

from __future__ import annotations

from typing import List, Union

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from onnxsim.bias_correction import _all_names, _unique_name
from onnxsim.quip_sharp import _match_matmul_like

_INT8 = onnx.TensorProto.INT8


def _has_min_opset(model: onnx.ModelProto, min_version: int) -> bool:
    return any(
        o.domain in ("", "ai.onnx") and o.version >= min_version
        for o in model.opset_import
    )


def _qmax(num_bits: int) -> int:
    return 2 ** (num_bits - 1) - 1


def _olive_quantize_blockwise(
    w_nk: np.ndarray, block_size: int, bits: int, outlier_threshold: float
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """OVP-encodes ``w_nk`` ([N, K], output channel first; ``K`` must be
    divisible by ``block_size``, itself even) -- see this module's own
    docstring. Returns ``(codes_nk, base_scale, outlier_scale,
    outlier_mask_nk)``: ``codes_nk`` int64 ``[N, K]``; ``base_scale``/
    ``outlier_scale`` float64 ``[N, K // block_size]``; ``outlier_mask_nk``
    bool ``[N, K]`` (True at each OVP pair's outlier position).
    """
    n, k = w_nk.shape
    num_blocks = k // block_size
    eps = 1e-12

    blocks = w_nk.reshape(n, num_blocks, block_size)
    abs_blocks = np.abs(blocks)

    typical_scale = np.median(abs_blocks, axis=2, keepdims=True)  # [n, nb, 1]
    is_outlier = abs_blocks > (outlier_threshold * typical_scale)  # [n, nb, bs]

    ordinary_qmax = _qmax(bits)
    outlier_qmax = _qmax(bits + 1)
    victim_qmax = _qmax(bits - 1)

    # base_scale: block-wide scale from non-outlier elements only, so a
    # block's ordinary elements quantize tightly, unaffected by whichever
    # of their neighbors are outliers (mirrors onnxsim.spqr's own
    # exclude-outliers-from-the-scale idea).
    non_outlier_abs = np.where(is_outlier, 0.0, abs_blocks)
    base_max = non_outlier_abs.max(axis=2)  # [n, nb]
    block_all_outlier = ~np.any(~is_outlier, axis=2)
    base_max = np.where(block_all_outlier, abs_blocks.max(axis=2), base_max)
    base_scale = np.maximum(base_max, eps) / ordinary_qmax  # [n, nb]

    # outlier_scale: a second block-wide scale fit to this block's own
    # outlier magnitudes only, so the wider code actually extends dynamic
    # range instead of just reusing base_scale's (too-narrow) coverage.
    outlier_abs_only = np.where(is_outlier, abs_blocks, 0.0)
    outlier_max = outlier_abs_only.max(axis=2)  # [n, nb]
    block_has_outlier = np.any(is_outlier, axis=2)
    outlier_scale = np.where(
        block_has_outlier, np.maximum(outlier_max, eps) / outlier_qmax, base_scale
    )

    # Adjacent, non-overlapping pairing within each block.
    pair_outlier = is_outlier.reshape(n, num_blocks, block_size // 2, 2)
    ovp_pair = pair_outlier[..., 0] != pair_outlier[..., 1]  # exactly one outlier
    ovp_pair_full = np.repeat(ovp_pair, 2, axis=2)  # [n, nb, bs]

    is_final_outlier = is_outlier & ovp_pair_full
    is_victim = (~is_outlier) & ovp_pair_full
    # Declined: an outlier with no unpaired non-outlier neighbor (its pair
    # partner is also an outlier) -- falls back to ordinary quantization,
    # same as any plain non-outlier element.
    is_ordinary = ~ovp_pair_full

    base_scale3 = base_scale[:, :, np.newaxis]
    outlier_scale3 = outlier_scale[:, :, np.newaxis]

    code_ordinary = np.clip(
        np.round(blocks / base_scale3), -ordinary_qmax, ordinary_qmax
    )
    code_victim = np.clip(np.round(blocks / base_scale3), -victim_qmax, victim_qmax)
    code_outlier = np.clip(
        np.round(blocks / outlier_scale3), -outlier_qmax, outlier_qmax
    )

    codes = np.where(is_ordinary, code_ordinary, 0.0)
    codes = np.where(is_victim, code_victim, codes)
    codes = np.where(is_final_outlier, code_outlier, codes)

    return (
        codes.reshape(n, k).astype(np.int64),
        base_scale,
        outlier_scale,
        is_final_outlier.reshape(n, k),
    )


def quantize_weight_only_olive(
    model: Union[str, onnx.ModelProto],
    bits: int = 4,
    block_size: int = 32,
    outlier_threshold: float = 4.0,
) -> onnx.ModelProto:
    """Applies OliVe-style Outlier-Victim Pair (OVP) quantization (see this
    module's own docstring for the technique) to every MatMul/vanilla-Gemm
    layer with a constant 2-D float32 weight whose reduction dimension
    ``K`` is divisible by ``block_size``. Needs no calibration data: every
    quantization decision comes from the weight tensor's own values.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param bits: the ordinary (non-outlier, non-victim) group-wide code
            width; must be at least 3 so the victim code width (``bits -
            1``) has at least 1 representable level. The outlier code width
            is ``bits + 1``
    :param block_size: elements per ``(output channel, block)``
            quantization group along the reduction dimension; must be even
            (elements are paired up within each block)
    :param outlier_threshold: an element is an outlier if its magnitude
            exceeds ``outlier_threshold`` times its block's median absolute
            value (the paper's own "victim-outlier" split is threshold-based
            rather than a fixed top-k fraction, since which elements are
            outliers is expected to vary block to block)
    :returns: ``model`` with every matched layer's weight replaced by its
            OVP-encoded reconstruction (see the module docstring's
            diagram); output tensor name unchanged. Layers with a
            non-constant, non-2-D weight, a reduction dimension not
            divisible by ``block_size``, or an odd ``block_size``, are left
            untouched; a model with no matching layer, or an opset older
            than 21 (``DequantizeLinear``'s ``block_size`` attribute needs
            opset 21), is returned unchanged
    """
    if bits < 3:
        raise ValueError(
            f"bits must be >= 3 (victim code width bits-1 needs >=1 level), got {bits}"
        )
    if block_size % 2 != 0:
        raise ValueError(f"block_size must be even, got {block_size}")

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
    for node in graph.node:
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

    for node, x_name, w_name, weight_transposed in candidates:
        w_init = initializer_map[w_name]
        w = onnx.numpy_helper.to_array(w_init).astype(np.float64)
        dim0, dim1 = w.shape
        w_nk = w if weight_transposed else w.T  # [N, K], output channel first
        n, k = w_nk.shape
        if k % block_size != 0:
            continue

        codes_nk, base_scale, outlier_scale, outlier_mask_nk = (
            _olive_quantize_blockwise(w_nk, block_size, bits, outlier_threshold)
        )
        has_any_outlier = bool(outlier_mask_nk.any())

        codes_orig = codes_nk if weight_transposed else codes_nk.T
        mask_orig = outlier_mask_nk if weight_transposed else outlier_mask_nk.T
        # base_scale/outlier_scale are [N, K // block_size]; DequantizeLinear
        # (axis=0, block_size=block_size) wants the scale shaped like the
        # weight with its axis-0 (reduction) dimension divided by
        # block_size -- so transpose to [K // block_size, N] to match
        # w_nk's own [N, K] -> [K, N] transpose below (mirrors
        # onnxsim.spqr's own scale_kn convention).
        base_scale_kn = base_scale.T
        outlier_scale_kn = outlier_scale.T
        assert codes_orig.shape == (dim0, dim1)

        prefix = f"{w_name}_olive"
        codes_kn = codes_orig if weight_transposed else codes_orig.T  # [K, N]
        mask_kn = mask_orig if weight_transposed else mask_orig.T  # [K, N]

        codes_name = _unique_name(f"{prefix}_codes", taken_names)
        wq = onnx.TensorProto()
        wq.name = codes_name
        wq.data_type = _INT8
        wq.dims.extend([k, n])
        wq.raw_data = codes_kn.astype(np.int8).tobytes()
        graph.initializer.append(wq)

        base_scale_name = _unique_name(f"{prefix}_base_scale", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(
                base_scale_kn.astype(np.float32), name=base_scale_name
            )
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

        base_dequant = _new(
            "DequantizeLinear",
            [codes_name, base_scale_name],
            "base_dequant",
            axis=0,
            block_size=block_size,
        )

        if has_any_outlier:
            outlier_scale_name = _unique_name(f"{prefix}_outlier_scale", taken_names)
            graph.initializer.append(
                onnx.numpy_helper.from_array(
                    outlier_scale_kn.astype(np.float32), name=outlier_scale_name
                )
            )
            mask_name = _unique_name(f"{prefix}_outlier_mask", taken_names)
            graph.initializer.append(
                onnx.numpy_helper.from_array(mask_kn.astype(np.int8), name=mask_name)
            )

            outlier_dequant = _new(
                "DequantizeLinear",
                [codes_name, outlier_scale_name],
                "outlier_dequant",
                axis=0,
                block_size=block_size,
            )
            mask_bool = _new("Cast", [mask_name], "mask_bool", to=onnx.TensorProto.BOOL)
            w_reconstructed = _new(
                "Where", [mask_bool, outlier_dequant, base_dequant], "w_reconstructed"
            )
        else:
            w_reconstructed = base_dequant

        old_output = node.output[0]
        bias_name = node.input[2] if len(node.input) > 2 else None
        node_idx = next(i for i, n_ in enumerate(graph.node) if n_ is node)

        core = _new("MatMul", [x_name, w_reconstructed], "core")
        if bias_name:
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

        for offset, new_node in enumerate(new_nodes):
            graph.node.insert(node_idx + offset, new_node)
        del graph.node[node_idx + len(new_nodes)]

    return out
