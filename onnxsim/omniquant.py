"""OmniQuant (Shao et al., 2023, "OmniQuant: Omnidirectionally Calibrated
Quantization for Large Language Models", https://arxiv.org/abs/2308.13137).
onnxsim ports the algorithm, not any framework's code, per the same
rationale as :mod:`onnxsim.awq`/:mod:`onnxsim.gptq` (OmniQuant's own
reference implementation optimizes live PyTorch modules block by block
with backpropagation, with no ONNX export path).

OmniQuant combines two calibrated (data-driven, but *not* requiring a
full backprop training loop the way the paper's own block-wise gradient
descent does -- see below for what replaces it here) adjustments on top
of plain round-to-nearest block quantization:

- **Learnable Weight Clipping (LWC).** :func:`onnxsim.quantize_weight_only_int4`
  and every other onnxsim INT4 scheme derive each block's scale directly
  from that block's own min/max -- exactly the point a couple of outlier
  elements can distort (the same problem :mod:`onnxsim.hqq` addresses with
  a robust Lp loss instead). OmniQuant's LWC scales the block's min/max by
  a *learned* per-block clipping ratio (`` in (0, 1]``) instead, letting a
  block deliberately clip its most extreme elements when doing so reduces
  the block's own overall reconstruction error more than it costs. The
  paper learns this ratio by gradient descent through a straight-through
  rounding relaxation (the same relaxation :mod:`onnxsim.adaround` uses
  for a different parameter -- which bin each element rounds to, not the
  scale itself). This module instead **grid-searches** the ratio per
  block, the same search strategy :mod:`onnxsim.awq` already uses for its
  own per-channel scale: the objective is one-dimensional and well-behaved
  enough per block that a grid reliably finds as good an optimum as a few
  steps of noisy straight-through gradient descent would, without the risk
  of a hand-rolled autodiff-adjacent implementation silently getting a
  gradient wrong.
- **Learnable Equivalent Transformation (LET).** Like :mod:`onnxsim.smoothquant`,
  LET migrates activation quantization difficulty into the weight via a
  per-channel scale -- but it also *shifts* the activation by its own
  per-channel mean first, letting the transform also absorb a channel-wise
  DC offset (useful when the preceding op, e.g. a LayerNorm, leaves
  activations asymmetric around zero) before the scale is even applied.
  The paper learns both the scale and the shift jointly with LWC via the
  same block-wise gradient descent. This module instead sets the shift in
  **closed form** (each channel's own mean over the calibration set --
  the natural choice for "center the activation before scaling it") and
  reuses :mod:`onnxsim.smoothquant`'s own closed-form per-channel scale
  formula on the now-centered activation, then greedily re-searches LWC's
  clip ratio once more against the transformed weight. Because the shift
  is constant (fixed once calibration finishes), its effect on the
  layer's output is a constant too -- ``shift @ W`` -- so it costs nothing
  at inference beyond one extra additive bias, not a runtime shift
  operation.

Both simplifications trade the paper's own joint gradient-descent
optimization for cheaper, closed-form-or-grid-searched alternatives that
target the same two objectives (a learned clip ratio, a learned
scale-and-shift activation transform) -- consistent with how every other
refinement pass in this series (:mod:`onnxsim.hqq`'s IRLS in place of
half-quadratic splitting, :mod:`onnxsim.squeezellm`'s sensitivity-weighted
k-means in place of the paper's own solver) substitutes an independently
verifiable classical technique for a paper's own bespoke or
gradient-based one, rather than risk an unverifiable line-for-line
reproduction.

Like :mod:`onnxsim.apply_awq`/:mod:`onnxsim.apply_gptq`, this only
rewrites :func:`onnxsim.quantize_weight_only_int4`'s own INT4 codes/scale
in place -- it never touches that ``DequantizeLinear``'s ``axis``
attribute -- so a plain (non-``transB=1``) MatMul/Gemm layer this module
touches is affected by the same ONNX Runtime ``MatMulNBitsFusion``
optimizer bug :mod:`onnxsim.ort_matmul_nbits_workaround` works around
(verified transparently compatible with that workaround, the same as
every other technique built on ``quantize_weight_only_int4``'s output).
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import onnx

from onnxsim.calibration import Tensors

# _quantize_blockwise_int4_with_clip is kept as a plain, directly reusable
# function (not folded into apply_omniquant's own now-C++ implementation
# below) because it is reused by other modules' own pure-Python searches --
# onnxsim.quarot, onnxsim.imatrix_quant, and onnxsim.mixed_precision all
# import it directly, and it is not itself a hot loop worth porting on its
# own (apply_omniquant_cpp's own C++ port re-derives the identical formula
# internally, since it operates at the protobuf level, not via this
# function).


def _quantize_blockwise_int4_with_clip(
    w_nk: np.ndarray, block_size: int, clip_ratio: float
) -> "tuple[np.ndarray, np.ndarray]":
    """Round-to-nearest block-wise INT4 quantization of ``w_nk`` ([N, K],
    output channel first), with each block's scale computed from
    ``clip_ratio * max(|w| in block) / 7`` instead of the plain (
    ``clip_ratio = 1``) min/max scale -- OmniQuant's Learnable Weight
    Clipping, here with the ratio grid-searched rather than learned. See
    this module's own docstring.
    """
    n, k = w_nk.shape
    num_blocks = k // block_size
    blocks = w_nk.reshape(n, num_blocks, block_size)
    scale_blocks = np.maximum(np.abs(blocks).max(axis=2) * clip_ratio, 1e-12) / 7.0
    scale_full = np.repeat(scale_blocks, block_size, axis=1)
    codes_nk = np.clip(np.round(w_nk / scale_full), -7.0, 7.0)
    return codes_nk, scale_blocks


def apply_omniquant(
    float_model: Union[str, onnx.ModelProto],
    quantized_model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    num_clip_steps: int = 20,
    num_alpha_steps: int = 20,
    min_clip_ratio: float = 0.5,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Applies OmniQuant-style learnable weight clipping and learnable
    equivalent transformation to every ``quantize_weight_only_int4``-quantized
    MatMul/Gemm layer present (by node output name) in both
    ``float_model`` and ``quantized_model``, using real activations
    captured from ``float_model``. See this module's own docstring for
    the technique.

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
            i.e. no LET transform at all, is always the grid's first
            point, so a layer LET can't improve gets no inserted nodes)
    :param min_clip_ratio: the LWC grid's lower bound
    :param providers: onnxruntime execution providers to run ``float_model``
            on when capturing calibration activations
    :returns: ``quantized_model`` with every layer OmniQuant measurably
            improved rewritten: its INT4 weight/scale replaced by the
            LWC-reclipped, LET-transformed-and-requantized versions, and
            (only when LET was found to help) a new ``Sub``/``Mul``
            inserted before it transforming its activation input plus a
            new ``Add`` folding in the constant bias correction after it.
            A layer OmniQuant found no LET improvement for (``alpha = 0``
            best) still gets its LWC-only reclipping, with no inserted
            activation-side nodes.

    Thin wrapper delegating to the verified C++ port
    (:func:`onnxsim.apply_omniquant_cpp`) -- full parameter parity, no
    functionality gap: this is a bounded, deterministic grid search (no
    RNG, no hand-rolled dense linear algebra), and the C++ port has been
    verified to track this module's own numpy arithmetic bit-for-bit
    across every shape/GEMM-orientation/3-D-activation configuration this
    module's own test suite exercises (see
    ``tests/test_omniquant_cpp.py``).
    """
    from onnxsim.onnx_simplifier import apply_omniquant_cpp

    return apply_omniquant_cpp(
        float_model,
        quantized_model,
        calibration_data=calibration_data,
        num_samples=num_samples,
        seed=seed,
        num_clip_steps=num_clip_steps,
        num_alpha_steps=num_alpha_steps,
        min_clip_ratio=min_clip_ratio,
        providers=providers,
    )
