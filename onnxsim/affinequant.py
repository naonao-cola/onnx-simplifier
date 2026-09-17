"""AffineQuant (Ma et al., 2024, ICLR, "AffineQuant: Affine Transformation
Quantization for Large Language Models", https://arxiv.org/abs/2403.12544).
onnxsim ports the algorithm, not any framework's code, per the same
rationale as :mod:`onnxsim.omniquant` (AffineQuant's own reference
implementation, like OmniQuant's, optimizes live PyTorch modules with
backpropagation, with no ONNX export path).

**Relationship to :mod:`onnxsim.omniquant`.** AffineQuant is explicitly
framed by its own paper as a generalization of OmniQuant's Learnable
Equivalent Transformation (LET): OmniQuant (and, before it,
:mod:`onnxsim.smoothquant`) migrates activation quantization difficulty
into the weight via a *diagonal* per-channel transform -- a single scale
(and, for OmniQuant, a shift) applied independently to each activation
channel, with no mixing across channels. AffineQuant instead optimizes a
full *invertible affine transformation matrix* jointly with the
quantization: because a general matrix can also rotate/mix channels
against each other (not just rescale each one on its own), it has
strictly more representational freedom to reduce quantization error than
any diagonal transform can reach, at higher compensation cost (a matrix
multiply on the activation side and a matrix-weight product on the weight
side, instead of an elementwise scale). A diagonal matrix is a special
case of an affine matrix, so this module's own search is set up to never
do worse than :mod:`onnxsim.omniquant`'s own diagonal LET on the same
layer -- see "Search strategy" below.

**Scope: block-diagonal, not dense.** The paper's own experiments use a
*fully dense* per-layer transformation matrix, optimized end-to-end by
gradient descent. A dense ``[K, K]`` matrix does not fit this repo's
closed-form/grid-search style (see :mod:`onnxsim.omniquant`'s own
docstring for why this series prefers that over a hand-rolled autodiff
loop): choosing a good dense ``K x K`` matrix without gradients means
solving a ``K``-dimensional optimization problem, and *compensating* it
means inverting a dense ``K x K`` matrix, which for a real transformer's
hidden size (thousands) is both numerically fragile (a poorly-conditioned
dense inverse can blow up the compensated weight's dynamic range, making
quantization *worse*, not better) and expensive to insert as a runtime
``MatMul``. This module instead restricts the transform to
**block-diagonal**: the ``K`` activation channels are partitioned into
fixed-size blocks (``affine_block_size``, default 8; configurable), and
each block gets its own small, independently-invertible square matrix,
with zero coupling across blocks. This keeps the search per-block (a
handful of channels at a time, tractable to search or solve in closed
form) and the compensation an efficient block-diagonal matrix multiply
(a ``[K, K]`` matrix that is mostly zero, structurally similar to grouped
convolution) rather than a dense ``K x K`` solve. It is a strictly less
expressive family than the paper's own dense matrix, but strictly more
expressive than OmniQuant's diagonal one -- a deliberate middle point
documented here as this module's own tractability tradeoff.

**Search strategy.** Per compensated layer, three transform families are
tried and the empirically-best one (lowest weight reconstruction error
against real calibration activations, exactly as :mod:`onnxsim.omniquant`
already measures) is kept:

1. **No transform** -- plain Learnable Weight Clipping (LWC) only, reusing
   :mod:`onnxsim.omniquant`'s own grid-searched per-block clip ratio
   unchanged.
2. **Diagonal LET** -- :mod:`onnxsim.omniquant`'s own closed-form shift
   plus alpha-grid-searched per-channel scale, unchanged.
3. **Block-affine LET** -- this module's own contribution: within each
   ``affine_block_size``-channel block, the block's own calibration
   activation covariance (after the same closed-form mean-shift as (2))
   is diagonalized via ``numpy.linalg.eigh``, giving an orthonormal basis
   (a rotation) for that block. Stacking these per-block rotations along
   the diagonal gives a block-diagonal orthogonal matrix ``R`` for the
   whole layer -- trivially and exactly invertible (``R^-1 = R^T``, no
   numerical solve needed, unlike an arbitrary invertible matrix) by
   construction, which is what keeps this restriction numerically stable.
   OmniQuant's own alpha-grid-searched per-channel scale is then
   recomputed *in the rotated basis* (on the rotated activation and
   rotated weight column statistics) and searched exactly as in (2).
   Because a rotation is orthogonal, family (2) is exactly family (3)
   restricted to ``R = I`` (or, when ``affine_block_size == 1``, every
   block's "rotation" is a trivial ``1x1`` matrix that cannot mix
   anything) -- so trying (3) can only match or beat (2), and (2) can
   only match or beat (1), on the same reconstruction-error objective
   this module measures. This mirrors :mod:`onnxsim.omniquant`'s own
   "each stage's first candidate reproduces the previous stage" guarantee
   that keeps it never worse than plain RTN.

Like OmniQuant, and unlike a pure migration module such as
:mod:`onnxsim.outlier_suppression`/:mod:`onnxsim.smoothquant`, this
module does not return a float model for a later, separate quantizer to
consume: the paper's own design (following OmniQuant's LET framework
directly) jointly optimizes the equivalent transformation *against* the
weight-only INT4 quantization error itself, so the transform choice and
the quantization it enables are inseparable here -- exactly the same
scope and non-float-migration contract :mod:`onnxsim.omniquant` already
documents and this module deliberately keeps, for the same reason.

This targets the same shape :mod:`onnxsim.omniquant` targets: every
``quantize_weight_only_int4``-quantized MatMul/Gemm layer (by node output
name) present in both a float model and its quantized counterpart, whose
activation input is a plain 2-D tensor.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import onnx

from onnxsim.calibration import Tensors

# _block_diagonal_rotation is kept as a plain, directly reusable function
# (not folded into apply_affinequant's own now-C++ implementation below)
# because tests/test_affinequant.py's own
# test_block_diagonal_rotation_is_orthogonal_and_block_diagonal unit-tests
# it directly, exercising the rotation-construction step of this
# technique's own "Search strategy" in isolation from the full grid search.


def _block_diagonal_rotation(
    x_centered: np.ndarray, affine_block_size: int
) -> np.ndarray:
    """Builds the ``[K, K]`` block-diagonal orthogonal rotation matrix used
    by this module's block-affine LET candidate: each ``affine_block_size``
    -channel block gets the eigenvectors of that block's own calibration
    covariance (over ``x_centered``, already mean-shifted) as its local
    orthonormal basis. Orthogonal by construction (``numpy.linalg.eigh``
    always returns orthonormal eigenvectors of a symmetric matrix), so the
    result is exactly invertible via its own transpose -- no matrix solve.
    """
    num_samples, k = x_centered.shape
    rotation = np.zeros((k, k), dtype=np.float64)
    for start in range(0, k, affine_block_size):
        stop = min(start + affine_block_size, k)
        block = x_centered[:, start:stop]
        cov = (block.T @ block) / max(num_samples, 1)
        # eigh: cov is symmetric PSD by construction (a Gram matrix), so
        # its eigenvectors are real and orthonormal.
        _, eigvecs = np.linalg.eigh(cov)
        rotation[start:stop, start:stop] = eigvecs
    return rotation


def apply_affinequant(
    float_model: Union[str, onnx.ModelProto],
    quantized_model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    num_clip_steps: int = 20,
    num_alpha_steps: int = 20,
    min_clip_ratio: float = 0.5,
    affine_block_size: int = 8,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Applies AffineQuant-style learnable weight clipping and block-affine
    learnable equivalent transformation to every
    ``quantize_weight_only_int4``-quantized MatMul/Gemm layer present (by
    node output name) in both ``float_model`` and ``quantized_model``,
    using real activations captured from ``float_model``. See this
    module's own docstring for the technique and how it generalizes
    :func:`onnxsim.apply_omniquant`'s diagonal transform.

    :param float_model: the original (unquantized) onnx ModelProto or file
            path
    :param quantized_model: a quantized version of ``float_model`` (onnx
            ModelProto or file path), produced by
            :func:`onnxsim.quantize_weight_only_int4`. Layers quantized by
            any other scheme (or left unquantized), or whose activation
            input has no feature axis at all (rank < 2) are left
            untouched; a higher-rank ``[batch, seq, K]`` activation is
            flattened to ``[batch * seq, K]``, which is exact.
    :param calibration_data: representative input batches to search and
            measure the transform on. Each batch is a
            ``{input_name: np.ndarray}`` dict matching ``float_model``'s
            graph inputs -- see :func:`onnxsim.generate_random_calibration_data`
            (the default when omitted) and
            :func:`onnxsim.load_huggingface_calibration_data` (real data).
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param num_clip_steps: grid points for the LWC clip ratio, evenly
            spaced over ``[min_clip_ratio, 1.0]`` inclusive (``1.0``,
            i.e. plain min/max scaling, is always the grid's last point,
            so a block LWC can't improve keeps its original scale)
    :param num_alpha_steps: grid points for the LET migration-strength
            exponent, evenly spaced over ``[0, 1]`` inclusive (``0``,
            i.e. no scale/shift transform at all, is always the grid's
            first point), used identically for both the diagonal and the
            block-affine candidate
    :param min_clip_ratio: the LWC grid's lower bound
    :param affine_block_size: the size of each independently-rotated block
            of activation channels in the block-affine candidate (see this
            module's own "Scope" docstring section). Larger blocks can
            capture more cross-channel structure but cost a bigger
            per-block eigendecomposition and a bigger inserted block of
            the compensating ``MatMul``; must be a positive divisor
            consideration only -- layers whose K isn't evenly divisible by
            it simply skip the block-affine candidate (falling back to (2)
            or (1)) rather than erroring.
    :param providers: onnxruntime execution providers to run ``float_model``
            on when capturing calibration activations
    :returns: ``quantized_model`` with every layer AffineQuant measurably
            improved rewritten: its INT4 weight/scale replaced by the
            LWC-reclipped, LET-transformed-and-requantized versions, and
            (only when a diagonal or block-affine transform was found to
            help) a new ``Sub``/(optionally ``MatMul``)/``Mul`` inserted
            before it transforming its activation input plus a new
            ``Add`` folding in the constant bias correction after it. A
            layer AffineQuant found no LET improvement for still gets its
            LWC-only reclipping, with no inserted activation-side nodes.

    Thin wrapper delegating to the verified C++ port
    (:func:`onnxsim.apply_affinequant_cpp`) -- full parameter parity, no
    functionality gap (see that function's own docstring, and
    ``onnxsim/affinequant_entry.h``, for the one documented, immaterial-
    -to-correctness numerical divergence: the block-affine candidate's own
    hand-rolled cyclic Jacobi eigendecomposition does not reproduce
    ``numpy.linalg.eigh``'s own eigenvector basis bit-for-bit on a block
    with repeated/near-degenerate eigenvalues -- both are equally valid
    orthonormal bases of the same eigenspace, verified to reach the same
    reconstruction-error quality, not merely "close").
    """
    from onnxsim.onnx_simplifier import apply_affinequant_cpp

    return apply_affinequant_cpp(
        float_model,
        quantized_model,
        calibration_data=calibration_data,
        num_samples=num_samples,
        seed=seed,
        num_clip_steps=num_clip_steps,
        num_alpha_steps=num_alpha_steps,
        min_clip_ratio=min_clip_ratio,
        affine_block_size=affine_block_size,
        providers=providers,
    )
