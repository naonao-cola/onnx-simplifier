"""AWQ -- Activation-aware Weight Quantization (Lin et al., 2023, "AWQ:
Activation-aware Weight Quantization for LLM Compression and Acceleration",
https://arxiv.org/abs/2306.00978). Originally implemented for PyTorch/CUDA
LLM deployment (most visibly as part of Meta's ``torchao`` -- see this
repository's own research into whether ``torchao``/``mslk`` have any real
ONNX interop point, which they do not: they quantize live PyTorch
``nn.Module``s with runtime-only tensor subclasses, with no ONNX export
path). AWQ's *algorithm*, unlike its PyTorch implementation, is entirely
portable: it needs nothing but a weight tensor, a round-to-nearest
block-wise integer quantizer, and real calibration activations -- exactly
:func:`onnxsim.quantize_weight_only_int4`'s own scheme and
:mod:`onnxsim.adaround`'s own "post-hoc adjustment from real activations"
style. This module ports the algorithm, not any framework's code.

AWQ's key empirical observation: not every weight element matters equally to
a layer's output -- a weight column feeding a large-magnitude *activation*
channel dominates that channel's contribution to the output, so quantization
error there costs more than an equal-sized error on a column whose
activation is small. Round-to-nearest (what :func:`quantize_weight_only_int4`
does) treats every element identically regardless of this, and per-element
optimization of *which way* to round (:func:`onnxsim.apply_adaround`) never
touches the weight's own magnitude, only which quantization bin it lands in.
AWQ instead rescales entire input-channel *columns* of the weight upward in
proportion to that channel's own average activation magnitude before
quantizing -- inflating a salient column's share of its block's dynamic
range so round-to-nearest's fixed per-block step size costs it
proportionally less -- and applies the exact inverse scale to the
activation, via a new ``Mul`` node inserted right before the layer, so the
transformation is a no-op on the *unquantized* function and only changes how
much quantization error each column ends up absorbing.

For every :func:`onnxsim.quantize_weight_only_int4`-quantized MatMul/Gemm
present (by node output name) in both ``float_model`` and ``quantized_model``
(the same matching :mod:`onnxsim.adaround` uses -- see
``weight_only_quantize_int4_matmul.h``'s scheme this targets): measures each
input channel's average activation magnitude from real calibration data,
then grid-searches a single scalar exponent ``alpha`` (the per-channel scale
is ``activation_magnitude ** alpha``, geometric-mean-normalized to keep the
compensating activation scale well-conditioned) to minimize the layer's own
reconstruction error, re-quantizing the rescaled weight from scratch at each
candidate ``alpha`` and measuring it against real activations -- the same
grid search the AWQ paper itself uses, not a closed-form guarantee. ``alpha
= 0`` (uniform scale 1, i.e. plain round-to-nearest) is always one of the
candidates, so a layer AWQ can't improve keeps its original quantization and
gets no inserted node at all.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import onnx

from onnxsim.calibration import Tensors


def _quantize_blockwise_int4(
    w_nk: np.ndarray, block_size: int
) -> "tuple[np.ndarray, np.ndarray]":
    """Fresh round-to-nearest block-wise INT4 quantization of ``w_nk``
    ([N, K], output channel first), matching
    ``TryQuantizeWeightBlockwiseInt4InPlace``'s own scheme (one scale per
    ``(output channel, block-of-K)`` group, ``scale = max(|w| in block) /
    7``, codes clamped to ``[-7, 7]``). Returns ``(codes_nk, scale_blocks)``
    with ``scale_blocks`` shape ``[N, K // block_size]``. Assumes ``K %
    block_size == 0`` (true of every candidate this module matches, since
    ``block_size`` is read from the existing DequantizeLinear node that
    already quantized this same weight).
    """
    n, k = w_nk.shape
    num_blocks = k // block_size
    blocks = w_nk.reshape(n, num_blocks, block_size)
    scale_blocks = np.maximum(np.abs(blocks).max(axis=2), 1e-12) / 7.0
    scale_full = np.repeat(scale_blocks, block_size, axis=1)
    codes_nk = np.clip(np.round(w_nk / scale_full), -7.0, 7.0)
    return codes_nk, scale_blocks


def apply_awq(
    float_model: Union[str, onnx.ModelProto],
    quantized_model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    num_alpha_steps: int = 20,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Applies AWQ-style activation-aware per-channel weight rescaling to
    every ``quantize_weight_only_int4``-quantized MatMul/Gemm layer present
    (by node output name) in both ``float_model`` and ``quantized_model``,
    using real activations captured from ``float_model``. See this module's
    own docstring for the technique.

    :param float_model: the original (unquantized) onnx ModelProto or file
            path
    :param quantized_model: a quantized version of ``float_model`` (onnx
            ModelProto or file path), produced by
            :func:`onnxsim.quantize_weight_only_int4`. Layers quantized by
            any other scheme (or left unquantized), or whose activation
            input has no feature axis at all (rank < 2) are left
            untouched; a higher-rank ``[batch, seq, K]`` activation is
            flattened to ``[batch * seq, K]``, which is exact. Assumes
            ``quantized_model`` was produced from ``float_model`` without
            renaming any MatMul/Gemm node's own output tensor -- true of
            every onnxsim ``quantize_*`` function.
    :param calibration_data: representative input batches to search and
            measure the rescaling on. Each batch is a
            ``{input_name: np.ndarray}`` dict matching ``float_model``'s
            graph inputs -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted) and
            :func:`onnxsim.load_huggingface_calibration_data` (real data, a
            much more representative search than random input).
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param num_alpha_steps: grid points for the per-channel scale exponent
            ``alpha``, evenly spaced over ``[0, 1]`` inclusive (matching the
            AWQ paper's own grid search); higher values search more finely
            at proportionally more cost (one full re-quantization and
            reconstruction-error measurement per candidate, per layer)
    :param providers: onnxruntime execution providers to run ``float_model``
            on when capturing calibration activations
    :returns: ``quantized_model`` with every layer AWQ measurably improved
            rewritten: its INT4 weight and scale initializers replaced with
            the rescaled-and-requantized versions, and a new ``Mul`` node
            inserted before it applying the compensating inverse channel
            scale to its activation input. A layer AWQ found no improvement
            for (``alpha = 0`` best) is left completely untouched.

    The pure-Python grid search below has been retired in favor of the
    verified-bit-exact C++ port -- this is now a thin alias for
    :func:`onnxsim.apply_awq_cpp` (``onnxsim/awq_entry.cpp``'s own
    ``ApplyAwq``), forwarding every argument unchanged. Exact
    (bit-for-bit) agreement was verified against this function's own
    pre-alias implementation across MatMul/Gemm/transB-Gemm, grid
    densities, multi-batch and rank-3 calibration, winning and losing
    alphas, and every skip shape -- see tests/test_awq_cpp.py -- before
    this alias was made. ``_quantize_blockwise_int4`` stays in this
    module (imported by :mod:`onnxsim.quantease`); only the entry point
    is aliased. Imported lazily (inside the function body, not at module
    scope) to avoid a circular import: ``onnxsim.onnx_simplifier``
    already imports from this module, so importing it back at module load
    time here would deadlock the import machinery.
    """
    from onnxsim.onnx_simplifier import apply_awq_cpp

    return apply_awq_cpp(
        float_model,
        quantized_model,
        calibration_data=calibration_data,
        num_samples=num_samples,
        seed=seed,
        num_alpha_steps=num_alpha_steps,
        providers=providers,
    )
