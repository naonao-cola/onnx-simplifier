"""HQQ -- Half-Quadratic Quantization (Badri & Shaji, 2023, "Half-Quadratic
Quantization of Large Machine Learning Models",
https://mobiusml.github.io/hqq_blog/). One of the notable weight-only PTQ
techniques ``torchao`` implements (its ``Int4WeightOnlyConfig`` offers
``int4_choose_qparams_algorithm="hqq"`` as an alternative to plain min/max --
onnxsim ports the *algorithm*, not that code, per the same rationale as
:mod:`onnxsim.awq`/:mod:`onnxsim.gptq` -- torchao has no ONNX export path).

Unlike every other onnxsim-native PTQ technique so far
(:mod:`onnxsim.adaround`, :mod:`onnxsim.awq`, :mod:`onnxsim.gptq`,
:mod:`onnxsim.autoround`), HQQ needs **no calibration data at all** -- it
never runs the model, only looks at each weight tensor's own values. Its
angle: a block's naive min/max-derived quantization range is set by
whichever one or two elements happen to be the most extreme, so a couple of
outliers can force a scale wide enough to blur every other, "normal"
element in that block down toward a handful of coarse levels. HQQ instead
picks the affine quantization parameters (scale, zero-point) to minimize a
**robust** loss -- an ``Lp`` norm with ``p < 1`` on the reconstruction
residual, which (unlike the ordinary ``L2``/mean-squared-error a naive
min/max range implicitly optimizes) tolerates a few large residuals on
outlier elements in exchange for a tighter, more accurate fit on the
majority.

This module optimizes the affine zero-point via **Iteratively Reweighted
Least Squares (IRLS)** -- a standard, textbook algorithm for minimizing an
``Lp`` (``p < 2``) loss by alternating (a) reweighting each element
inversely by its own current residual magnitude raised to the ``p - 2``
power, so elements with large residuals count for less in the next step,
and (b) re-solving the now-ordinary weighted-least-squares problem for the
zero-point in closed form. This converges to the same *kind* of solution
HQQ's own paper describes (a robust fit that downweights outliers) and
optimizes the same objective (:math:`\\sum_k |w_k - \\hat w_k|^p`); it is
not a line-for-line reproduction of the paper's own half-quadratic-splitting
solver, which this module does not claim to replicate exactly, only to
solve the same problem via a different, independently-verifiable classical
method.

The scale itself is set once from each block's own min/max range (the
standard affine-quantization initialization) and held fixed -- only the
zero-point is refined by IRLS, matching typical HQQ implementations, which
report most of the benefit comes from the zero-point fit.

Since ``quantize_weight_only_int4``'s own scheme is symmetric (zero-point
always 0 -- see ``weight_only_quantize_int4_matmul.h``), and HQQ's whole
premise is an *asymmetric* (nonzero zero-point) affine fit, this module
produces its own standalone quantization -- unlike
:mod:`onnxsim.adaround`/:mod:`onnxsim.awq`/:mod:`onnxsim.gptq` (which refine
an already-``quantize_weight_only_int4``-quantized model in place), this
takes the plain float model and quantizes it directly into a
``DequantizeLinear(Wq, Ws, Wz, axis=..., block_size=...)`` structure of its
own -- unsigned 4-bit codes and zero-point (``[0, 15]``), the natural
representation for an affine (as opposed to symmetric) quantizer.
"""

from __future__ import annotations

from typing import Union

import onnx

from onnxsim.onnx_simplifier import apply_hqq_cpp


def quantize_weight_only_int4_hqq(
    model: Union[str, onnx.ModelProto],
    block_size: int = 32,
    num_iterations: int = 10,
    lp_norm: float = 0.7,
) -> onnx.ModelProto:
    """Quantizes every MatMul/vanilla-Gemm layer with a constant 2-D
    float32 weight (whose reduction dimension ``K`` is evenly divisible by
    ``block_size``) into HQQ-style asymmetric block-wise INT4 -- see this
    module's own docstring for the technique. Needs no calibration data:
    every quantization decision comes from the weight tensor's own values.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param block_size: elements per (output-channel, block) quantization
            group along the reduction dimension
    :param num_iterations: IRLS reweighting steps refining each block's
            zero-point
    :param lp_norm: the robust loss exponent IRLS targets (``p < 2``; the
            HQQ paper's own typical default, ``0.7``, favors tolerating a
            few large residuals over spreading error evenly -- lower values
            downweight outliers more aggressively, ``p = 2`` would recover
            an ordinary (non-robust) least-squares fit)
    :returns: ``model`` with every matched layer's weight replaced by its
            HQQ-quantized float32 reconstruction (see the "Delegates to"
            note below for a storage-format caveat); layers with a
            non-constant, non-2-D, or non-block-divisible weight are left
            untouched

    Delegates to :func:`onnxsim.apply_hqq_cpp`, which hardcodes
    ``block_size=32``, ``num_iterations=10`` and ``lp_norm=0.7`` (this
    function's own defaults) and cannot honor other values -- a
    non-default value raises ``ValueError`` rather than being silently
    ignored. Unlike this function's own former implementation (which
    built a real, packed-UINT4 ``DequantizeLinear(Wq, Ws, Wz, ...)``
    initializer triple), the C++ port folds the quantize/dequantize round
    trip directly into a *plain float32* replacement initializer -- a
    genuine storage-format difference, not merely a numerical one: the
    result no longer carries real INT4-packed storage, only the
    *simulated* precision loss. If your caller specifically needs the
    packed-UINT4 graph shape, this function can no longer provide it. This
    function also no longer needs opset 21 (the C++ fold uses no
    ``DequantizeLinear``/native UINT4 tensor type at all).
    """
    if block_size != 32 or num_iterations != 10 or lp_norm != 0.7:
        raise ValueError(
            "quantize_weight_only_int4_hqq now delegates to apply_hqq_cpp, "
            "which hardcodes block_size=32, num_iterations=10, lp_norm=0.7 "
            "and cannot honor other values"
        )
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return apply_hqq_cpp(model)
