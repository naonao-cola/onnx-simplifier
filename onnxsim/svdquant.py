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

from typing import Optional, Sequence, Union

import onnx

from onnxsim.calibration import Tensors


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

    Thin wrapper delegating to the verified C++ port
    (:func:`onnxsim.apply_svdquant_cpp`) -- full parameter parity, no
    functionality gap (see that function's own docstring, and
    ``onnxsim/svdquant_entry.h``, for the one documented, immaterial-to-
    -correctness numerical divergence: this port's own SVD is a hand-rolled
    Jacobi solver rather than ``numpy.linalg.svd``'s own LAPACK routine, so
    individual singular vectors/values need not match sign-for-sign or
    bit-for-bit, only that the reconstructed rank-``r`` low-rank branch
    itself agrees closely).
    """
    from onnxsim.onnx_simplifier import apply_svdquant_cpp

    return apply_svdquant_cpp(
        model,
        calibration_data=calibration_data,
        num_samples=num_samples,
        seed=seed,
        rank=rank,
        block_size=block_size,
        smooth_alpha=smooth_alpha,
        providers=providers,
    )
