"""SliM-LLM (Huang, Shao, Dong, Luo, Qiao et al., 2024, "SliM-LLM:
Salience-Driven Mixed-Precision Quantization for Large Language Models",
https://arxiv.org/abs/2405.14917).

:mod:`onnxsim.mixed_precision` already assigns different bit-widths within
one model, but only at **layer** granularity: it picks one bit-width
(INT4 or INT8) for an entire matched MatMul/Gemm weight, based on that
layer's own sensitivity score. SliM-LLM's distinguishing contribution is a
finer granularity -- it picks a bit-width per **group** *within* a single
weight matrix: the same reduction-axis blocks
:func:`onnxsim.quantize_weight_only_int4`/:mod:`onnxsim.mixed_precision`
already partition a layer's weight into (``group_size`` contiguous
elements along ``K``) each get their own, independently chosen bit-width,
so two groups in the *same layer* can end up quantized at different
precisions. The key phrase: :mod:`onnxsim.mixed_precision` picks one
bit-width per *layer*; this module picks one bit-width per *group within a
layer*.

**The salience score.** Mirrors :mod:`onnxsim.owq`'s own Optimal Brain
Surgeon-style column saliency (itself built on :mod:`onnxsim.gptq`'s
Hessian machinery) rather than reinventing one: for a layer's weight ``W``
([N, K], output channel first) and calibration activations ``X``
([samples, K]), ``H = X^T X`` is the same per-layer Hessian
:mod:`onnxsim.gptq`/:mod:`onnxsim.owq` already compute, and

    sensitivity_j = mean_n[(W[n, j] - RTN_low(W[n, j]))^2] / [H^-1]_jj

is :mod:`onnxsim.owq`'s own per-*column* saliency -- ``RTN_low`` here is
round-to-nearest at ``low_bits`` (this module's least-precise tier), so the
numerator captures how much reconstruction error each column would suffer
at the aggressive end of the bit budget, and ``[H^-1]_jj`` discounts
columns whose error other columns could compensate for. This module
aggregates that per-column score to a per-*group* salience (the mean of
``sensitivity_j`` over each group's ``group_size`` columns), then ranks
groups by it -- the same score OWQ uses to rank columns, applied one level
coarser to rank groups instead.

**Bit assignment.** Each layer gets a two-tier budget, ``low_bits`` (the
default candidate set is ``{2, 4}``, matching the paper's own ultra-low-bit
regime) and ``high_bits``, mixed so the layer's own average bits/weight
hits ``target_bits``: solving
``fraction_high * high_bits + (1 - fraction_high) * low_bits = target_bits``
for ``fraction_high`` (clipped to ``[0, 1]``) gives the count of groups
(rounded, most-salient first) that get ``high_bits``; every other group
gets ``low_bits``. This mirrors :mod:`onnxsim.mixed_precision`'s own
``high_bits_fraction`` parameter, except computed from a target *average*
bit budget instead of being supplied directly, and applied to groups
within one layer instead of to whole layers across the model.

**Storage.** Both tiers' codes are symmetric round-to-nearest integers in
``[-(2^(bits-1) - 1), 2^(bits-1) - 1]``, stored as a single ``INT8``
initializer per layer (wide enough for every bit-width this module
supports, so no bit-packing is needed -- the same simplification
:mod:`onnxsim.mixed_precision`'s own INT8 tier already makes) with a
per-(group, output-channel) float32 scale, reconstructed via one
``DequantizeLinear(..., axis=0, block_size=group_size)`` -- identical graph
shape to :func:`onnxsim.quantize_weight_only_int8_block`, since a group's
chosen bit-width only changes what range its own codes and scale were
computed over, not how they are dequantized. Each layer's per-group
bit-width is additionally recorded in a parallel ``INT64`` initializer
(mirroring how :mod:`onnxsim.owq`/:mod:`onnxsim.spqr` already attach
per-group/per-column metadata alongside their packed codes) purely for
inspection -- the dequantization subgraph does not read it.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import onnx

from onnxsim.calibration import Tensors


def apply_slim_llm(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    target_bits: float = 3.0,
    low_bits: int = 2,
    high_bits: int = 4,
    group_size: int = 32,
    percdamp: float = 0.01,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Quantizes every MatMul/vanilla-Gemm layer with a constant 2-D
    float32 weight whose reduction dimension ``K`` is divisible by
    ``group_size`` to a per-group mix of ``low_bits``/``high_bits``
    integer codes, chosen from a calibration-driven per-group salience
    score so each layer's own average bits/weight lands at ``target_bits``
    -- see this module's own docstring.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches (each a
            ``{input_name: np.ndarray}`` dict matching ``model``'s graph
            inputs) used to build each layer's Hessian and per-column
            error -- see :func:`onnxsim.generate_random_calibration_data`
            (the default when omitted) and
            :func:`onnxsim.load_huggingface_calibration_data` (real data,
            a more representative salience ranking than random input)
    :param num_samples: random batches to generate when ``calibration_data``
            is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param target_bits: target average bits/weight for each matched
            layer, met by mixing ``low_bits``- and ``high_bits``-coded
            groups (most salient groups get ``high_bits``); must lie in
            ``[low_bits, high_bits]`` -- values outside that range are
            clipped to it
    :param low_bits: bit-width for the least salient groups in each layer
            (the paper's own regime is ultra-low-bit, so the default is 2)
    :param high_bits: bit-width for the most salient groups in each layer
    :param group_size: elements per quantization group along ``K``,
            matching :func:`onnxsim.quantize_weight_only_int4`'s own
            default
    :param percdamp: Hessian damping factor (fraction of the mean diagonal
            added to every diagonal entry before inversion), matching
            :mod:`onnxsim.gptq`/:mod:`onnxsim.owq`'s own default
    :param providers: onnxruntime execution providers to run calibration on
    :returns: ``model`` with every matched layer's weight replaced by
            per-group mixed ``low_bits``/``high_bits`` INT8-stored codes
            plus a per-group float32 scale
            (``DequantizeLinear(..., axis=0, block_size=group_size)``) and
            a parallel per-group bit-width INT64 initializer; output
            tensor name unchanged. Layers with a non-constant, non-2-D
            weight, a reduction dimension not divisible by ``group_size``,
            or no calibration activation available, are left untouched; a
            model with no matching layer, or an opset older than 21
            (``DequantizeLinear``'s ``block_size`` attribute needs opset
            21), is returned unchanged

    Full parity, unconditional delegation: this is a thin alias for the
    verified C++ port :func:`onnxsim.apply_slim_llm_cpp`
    (``onnxsim/slim_llm_entry.cpp``'s own ``ApplySlimLlm``), forwarding
    every argument unchanged. Exact (bit-for-bit) agreement was verified
    against this function's own pre-alias implementation (candidate
    matching, salience ranking, bit assignment, and the resulting
    codes/scale/group-bits initializers) -- see
    tests/test_slim_llm_cpp.py -- before this alias was made. Imported
    lazily (inside the function body, not at module scope) to avoid a
    circular import: ``onnxsim.onnx_simplifier`` already imports from this
    module, so importing it back at module load time here would deadlock
    the import machinery.
    """
    if low_bits < 2 or high_bits <= low_bits:
        raise ValueError("require 2 <= low_bits < high_bits")

    from onnxsim.onnx_simplifier import apply_slim_llm_cpp

    return apply_slim_llm_cpp(
        model,
        calibration_data=calibration_data,
        num_samples=num_samples,
        seed=seed,
        target_bits=target_bits,
        low_bits=low_bits,
        high_bits=high_bits,
        group_size=group_size,
        percdamp=percdamp,
        providers=providers,
    )
