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

from typing import List, Optional, Sequence, Union

import numpy as np
import onnx

from onnxsim.calibration import Tensors
from onnxsim.quip_sharp import _random_orthogonal_matrix


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

    Thin wrapper delegating to the verified C++ port
    (:func:`onnxsim.apply_duquant_cpp`) -- full parameter parity, no
    functionality gap (see that function's own docstring, and
    ``onnxsim/duquant_entry.h``, for the one documented, immaterial-to-
    -correctness numerical divergence: this port seeds a fresh RNG per
    matched layer -- rather than reproducing this module's own single,
    sequentially-advancing ``numpy.random.Generator`` thread across every
    layer and block -- and its own block-local rotation is a Gram-Schmidt
    construction, not :func:`onnxsim.quip_sharp._random_orthogonal_matrix`'s
    sign-corrected QR, so individual rotation columns and near-tied
    outlier-channel assignments need not match exactly, only that the
    permute-rotate-then-quantize composition stays exact before
    quantization for any orthogonal rotation).
    """
    from onnxsim.onnx_simplifier import apply_duquant_cpp

    return apply_duquant_cpp(
        model,
        calibration_data=calibration_data,
        num_samples=num_samples,
        seed=seed,
        block_size=block_size,
        outlier_fraction=outlier_fraction,
        epsilon=epsilon,
        providers=providers,
    )
