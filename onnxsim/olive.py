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

from typing import Union

import onnx

from onnxsim.onnx_simplifier import apply_olive_cpp


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

    Delegates to the verified C++ port (:func:`onnxsim.apply_olive_cpp`),
    which hardcodes this function's own defaults (``bits=4``,
    ``block_size=32``, ``outlier_threshold=4.0``) and folds the round trip
    directly into a replacement float32 initializer rather than building a
    real ``DequantizeLinear`` x2 + ``Cast`` + ``Where`` + ``MatMul``[+
    ``Add``] graph rewrite -- a storage-format change from this function's
    own former behavior, not a numeric one.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param bits: the ordinary (non-outlier, non-victim) group-wide code
            width. **The C++ port's own only supported value is 4** -- a
            non-default value raises ``ValueError``.
    :param block_size: elements per ``(output channel, block)``
            quantization group along the reduction dimension. **The C++
            port's own only supported value is 32** -- a non-default
            value raises ``ValueError``.
    :param outlier_threshold: an element is an outlier if its magnitude
            exceeds ``outlier_threshold`` times its block's median absolute
            value (see this module's own docstring). **The C++ port's own
            only supported value is 4.0** -- a non-default value raises
            ``ValueError``.
    :returns: ``model`` with every matched layer's weight replaced by its
            OliVe-quantized float32 version, stored under a new
            initializer. Layers with a non-constant, non-2-D weight, or a
            reduction dimension not divisible by ``block_size``, are left
            untouched; a model with no matching layer, or an opset older
            than 21, is returned unchanged.
    """
    if bits != 4 or block_size != 32 or outlier_threshold != 4.0:
        raise ValueError(
            "quantize_weight_only_olive now delegates to apply_olive_cpp, "
            "which hardcodes bits=4, block_size=32, outlier_threshold=4.0 "
            "and cannot honor other values"
        )
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return apply_olive_cpp(model)
