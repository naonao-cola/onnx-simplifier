"""SpQR (Dettmers et al., 2023, "SpQR: A Sparse-Quantized Representation
for Near-Lossless LLM Weight Compression", https://arxiv.org/abs/2306.03078).
onnxsim ports the algorithm, not any framework's code, per the same
rationale as :mod:`onnxsim.awq`/:mod:`onnxsim.gptq`/:mod:`onnxsim.spinquant`
(SpQR's own reference implementation quantizes live PyTorch weights via a
custom outlier-detection-and-packing pipeline, with no ONNX export path).

Ordinary block-wise RTN quantization (what
:func:`onnxsim.quantize_weight_only_int4` does) picks one scale per block
of the reduction dimension. A single unusually large weight *within* a
block forces that block's whole scale up, wasting resolution on every
other, ordinary-magnitude element sharing it -- the same "outlier" problem
:mod:`onnxsim.llm_int8`/:mod:`onnxsim.smoothquant` address for
*activations*, but here it is individual *weight* elements, scattered
throughout the matrix rather than confined to a few channels, so excluding
them channel-wise (like :mod:`onnxsim.llm_int8`) doesn't fit. SpQR's own
idea: identify the small fraction of weight elements whose quantization
error actually matters, exclude them from each block's own scale
computation (so the *rest* of the block quantizes tighter), and store an
exact correction for those specific elements as an explicit sparse
overlay -- ``W_reconstructed = block_quantized(W) + sparse_correction``,
with the correction defined as exactly ``W - block_quantized(W)`` at each
outlier position, so it cancels that position's quantization error
completely regardless of how the block-quantized value there was rounded.

**Picking which elements are outliers.** SpQR's own reference
implementation uses each weight's true contribution to the OBQ/GPTQ
objective, computed from the full inverse-Hessian of the layer's
calibration data -- expensive, and (like GPTQ's own column-by-column
update order) not independently verifiable without re-deriving the same
numerically delicate procedure. This module uses the classical
**diagonal-Hessian approximation** to that same objective instead: for a
squared-error objective with Hessian ``H = 2 X^T X``, OBQ's per-weight
error contribution ``w_k^2 / [H^{-1}]_{kk}`` reduces, when ``H`` is
approximated as diagonal, to ``w_k^2 * H_{kk} = w_k^2 * mean(X[:, k]^2)``
-- an ordinary, closed-form sensitivity score computed directly from the
weight and calibration activations, no matrix inversion involved. The
elements with the largest score (by default the top 1%, tunable via
``outlier_fraction``) are excluded from their block's scale computation
and become the sparse correction; every other element is quantized
normally.

**Storing the sparse correction efficiently.** Naively adding a dense
``[N, K]`` correction matrix back would cost as much storage as the
original float32 weight, defeating the point. Instead, only the
``num_outliers`` outlier ``(row, col)`` positions and their correction
values are stored (an initializer of shape ``[num_outliers, 2]`` plus one
of shape ``[num_outliers]`` -- at 1% density, a small fraction of the
dense weight's own footprint), and the graph reconstructs the full dense
correction at runtime via ``ScatterND`` into a ``ConstantOfShape``-produced
zero tensor -- both ordinary ONNX ops, no custom sparse tensor type or
contrib op needed:

    Before:
      Y = MatMul(X, W) [+ bias]                  -- W constant, [K, N], float32

    After:
      Wq  = <int4, per-(block, column) symmetric, outlier positions
             excluded from each block's own scale>
      Ws  = <float32, [K/block_size, N]>
      Wdq = DequantizeLinear(Wq, Ws, axis=0, block_size=block_size)  -- float32
      zeros = ConstantOfShape([K, N], value=0.0)
      correction = ScatterND(zeros, outlier_indices, outlier_values)  -- float32
      Wreconstructed = Wdq + correction
      Y = MatMul(X, Wreconstructed) [+ bias]

Every outlier position reconstructs *exactly* (the correction is defined
as the exact residual there, independent of how that position happened to
round), while the excluded-from-scale blocks quantize every ordinary
element more tightly than plain :func:`onnxsim.quantize_weight_only_int4`
would with the same outliers still dragging the scale up.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import onnx

from onnxsim.calibration import Tensors
from onnxsim.onnx_simplifier import apply_spqr_cpp


def quantize_weight_only_spqr(
    model: Union[str, onnx.ModelProto],
    block_size: int = 16,
    outlier_fraction: float = 0.01,
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Applies SpQR-style outlier-aware block-wise INT4 quantization (see
    this module's own docstring) to every MatMul/vanilla-Gemm layer with a
    constant 2-D float32 weight whose reduction dimension ``K`` is
    divisible by ``block_size``.

    Delegates to the verified C++ port (:func:`onnxsim.apply_spqr_cpp`,
    ``spqr_entry.h``), which builds the exact same outlier-selection and
    block-quantization pipeline (see that port's own "ACCEPTED, PERMANENT
    DIVERGENCE" note: outlier *selection order* can differ from numpy's
    own unspecified ``argpartition`` tie-handling, but the SET of outlier
    positions and the reconstructed model are the same).

    :param model: the original (unquantized) onnx ModelProto or file path
    :param block_size: elements per quantization block along ``K``,
            matching :func:`onnxsim.quantize_weight_only_int4`'s own
            default granularity (SpQR's own typical choice is finer,
            8-32)
    :param outlier_fraction: fraction of each layer's weight elements
            (by count) excluded from block-scale computation and stored
            as an exact sparse correction instead -- SpQR's own paper
            uses roughly 1%
    :param calibration_data: representative input batches (each a
            ``{input_name: np.ndarray}`` dict matching ``model``'s graph
            inputs) used to compute each weight element's sensitivity
            score -- see :func:`onnxsim.generate_random_calibration_data`
            (the default when omitted) and
            :func:`onnxsim.load_huggingface_calibration_data` (real data,
            a more representative sensitivity ranking than random input)
    :param num_samples: random batches to generate when ``calibration_data``
            is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param providers: onnxruntime execution providers to run calibration on
    :returns: ``model`` with every matched layer's weight replaced by
            block-wise INT4 codes plus a sparse outlier correction (see
            the module docstring's diagram); output tensor name unchanged.
            Layers with a non-constant, non-2-D weight, a reduction
            dimension not divisible by ``block_size``, or no calibration
            activation available, are left untouched; a model with no
            matching layer, or an opset older than 21 (INT4's tensor type
            and ``DequantizeLinear``'s ``block_size`` attribute both need
            opset 21), is returned unchanged
    """
    return apply_spqr_cpp(
        model,
        calibration_data=calibration_data,
        num_samples=num_samples,
        seed=seed,
        block_size=block_size,
        outlier_fraction=outlier_fraction,
        providers=providers,
    )
