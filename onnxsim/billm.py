"""BiLLM (Huang, Liu, Qin, Li, Zhang, Liu, Magno, Qi, 2024, ICML 2024,
"BiLLM: Pushing the Limit of Post-Training Quantization for LLMs",
https://arxiv.org/abs/2402.04291) -- a genuine 1-bit-average **weight
binarizer**, not another rounding-refinement lever on top of an
already-fixed-scale quantizer the way :mod:`onnxsim.adaround`/
:mod:`onnxsim.awq`/:mod:`onnxsim.gptq`/:mod:`onnxsim.flexround` all are (each
of those takes an *already* ``quantize_weight_only_int4``-quantized model and
only changes which grid point each element rounds to). BiLLM instead takes
an ordinary dense float32 MatMul/Gemm layer straight from the float model and
produces a genuinely different (binary/near-binary) representation of it --
the same family as :func:`onnxsim.quantize_weight_only_int4`/
:func:`onnxsim.quantize_weight_only_nf4`/
:func:`onnxsim.quantize_weight_only_kmeans`, pushed to the most extreme end
of that family's bit-width range.

This is *not* the same problem :func:`onnxsim.quantize_ternary` solves (see
``docs/ternary-quantization.md``): that pass only **detects** a weight that
is *already*, structurally, exactly ``{-s, 0, +s}`` (a lossless rewrite of a
BitNet-family model someone else already trained ternary) and leaves
anything else -- including an ordinary dense float32 weight -- untouched.
BiLLM instead **quantizes** an ordinary dense float32 weight itself, lossily,
down to close to 1 bit/element on average. The two are complementary, not
overlapping: run :func:`onnxsim.quantize_ternary` on a BitNet export (nothing
to binarize, the weight already is ternary); run this module's
:func:`quantize_weight_only_billm` on a normal pretrained dense model's
Linear layers (genuinely new information loss, deliberately accepted for
the compression).

The technique, following the paper's own Algorithm 1/Algorithm 2 fairly
closely (this module ports the *algorithm*, in plain numpy, from the paper
itself -- there is no ONNX export path from the authors' own PyTorch
reference implementation, the same reason :mod:`onnxsim.gptq`/
:mod:`onnxsim.awq` port their own source papers rather than any upstream
code):

1. **Per-layer Hessian**, exactly as :mod:`onnxsim.gptq`: ``H = X^T X`` from
   real calibration activations (reusing
   :func:`onnxsim.gptq._inverse_hessian_cholesky` for the same damped-
   Cholesky-of-``H^-1`` reformulation GPTQ itself introduces, since BiLLM's
   own block-wise error compensation, Algorithm 1 lines 12-13, is the same
   OBC/GPTQ mechanism applied to whichever binarization a block picked
   rather than to a rounded integer).

2. **Block-wise processing** (``block_size`` columns of the reduction
   dimension at a time, paper default 128): within each block,

   a. **Structured salient-column selection** (paper Section 3.1,
      Algorithm 2's ``salient()``): per-column sensitivity
      ``s_i = w_i^2 / [H_c]_ii^2`` (``H_c`` the damped-Hessian-inverse
      Cholesky factor -- the same "how much does perturbing this weight
      hurt the layer's output" argument OBS/GPTQ use, evaluated per column
      via a column-sum rather than per individual element, since the paper
      finds salient Hessian mass concentrates in whole columns for
      attention-projection layers). Columns are ranked by total column
      sensitivity and a small bounded search (this module searches
      ``1..min(30, block_size - 1)`` candidate salient-column counts,
      matching the paper's own stated 3-30 search range) picks the count
      that minimizes plain-binary reconstruction error for the block --
      the *number* of salient columns is data-dependent per block/layer,
      not a fixed global fraction.

   b. **Binary residual approximation for salient columns** (paper
      Section 3.1 "Binary Residual Approximation", Algorithm 2's
      ``res_approximation()``, Equations 6-7): ``B1 = sign(W) * mean(|W|)``
      (one scalar scale for the whole salient sub-block -- the paper's own
      ``binary()`` primitive, Equation 4, whose L2-optimal closed-form
      scale is exactly ``mean(|W|)``), then a *second* binarization of the
      residual ``R = W - B1``: ``B2 = sign(R) * mean(|R|)``, giving
      ``W ~= B1 + B2`` -- effectively 2 bits for salient columns, versus a
      naive scheme that would need 8-16 bits to protect them, while the
      paper proves (Eq. 8) this residual reconstruction strictly dominates
      keeping only ``B1``.

   c. **Plain flat binary for non-salient columns**: ``sign(W) * mean(|W|)``
      (again the paper's own ``binary()`` primitive, one scalar scale for
      the whole non-salient sub-block) -- **this is a deliberate,
      documented simplification** of the paper's own Section 3.2 "Bell-
      shaped Distribution Splitting" (Algorithm 2's ``seg_search()``),
      which further splits non-salient weights *elementwise* by magnitude
      into a "concentrated" and a "sparse" region (each with its own
      scale, chosen by a 9-point percentile search minimizing Eq. 11) to
      account for their bell-shaped, non-uniform distribution. The paper
      itself reports this second-level split contributes far less to
      accuracy than the salient/residual mechanism (Table 1: ~0.02 extra
      average bits from the whole non-salient path, versus the salient
      path's own contribution) -- it is not this scheme's headline result.
      Skipping it also keeps this module's ONNX encoding uniform: the
      elementwise concentrated/sparse split has no clean per-column
      broadcast shape the way every other quantity here does (see below),
      and would force a per-element rather than a compact per-column
      scale. A faithful port of ``seg_search`` is a reasonable future
      addition, not attempted here.

   d. **Block-wise OBC-style error compensation** (paper Algorithm 1 lines
      12-13, the same mechanism :mod:`onnxsim.gptq` already implements):
      whatever this block's binarization couldn't represent is charged
      forward into every not-yet-processed column, in proportion to
      ``H_c``'s own off-diagonal structure -- so later blocks' own
      binarization partially compensates for earlier blocks' error, the
      same second-order argument GPTQ uses for its own rounding.

Because the paper's own scales (``mean(|W|)`` in Equation 4/12, and this
module's own non-salient scale) are single scalars *per block*, not per
individual weight element or even per output channel the way onnxsim's other
INT4/NF4/k-means schemes are, and because within a block every column is
either "salient" (gets both a level-1 and level-2 scale) or "non-salient"
(gets only a level-1 scale, with its level-2 contribution forced to zero),
this module's encoding is a **compact per-input-channel (per-column) pair of
scale vectors** (length ``K``, the reduction dimension) rather than the
per-(output-channel, block) 2-D scale grid :mod:`onnxsim.nf4`/
:func:`onnxsim.quantize_weight_only_int4` use:

    Before:
      Y = MatMul(X, W) [+ bias]        -- W constant, [K, N], float32

    After:
      Code1: initializer, int8, [K, N], values in {-1, +1}  -- sign(W), or
             sign(B1) for a salient column
      Code2: initializer, int8, [K, N], values in {-1, 0, +1}  -- sign of
             the salient residual for a salient column, exactly 0 (no
             correction) for a non-salient column
      Scale1: initializer, float32, [K, 1]  -- per-column level-1 scale
              (this block's alpha_salient for a salient column, this
              block's alpha_nonsalient for a non-salient one)
      Scale2: initializer, float32, [K, 1]  -- per-column level-2 scale,
              exactly 0 for every non-salient column
      What_hat = Cast(Code1, float) * Scale1 + Cast(Code2, float) * Scale2
      Y = MatMul(X, What_hat) [+ bias]

No ``Gather``/codebook lookup is needed the way :mod:`onnxsim.nf4`/
:mod:`onnxsim.kmeans_quantization` need one: since the "codebook" here is
just ``{-1, +1}`` (and ``{-1, 0, +1}`` for the residual level), the code
tensor's own values, cast to float, already *are* the sign -- multiplying by
the per-column scale directly reconstructs the weight, with no lookup table
in between. Ordinary ONNX ops only (``Cast``/``Mul``/``Add``), opset 11+
(this module needs nothing newer than that), no contrib op.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Union

import numpy as np
import onnx

from onnxsim.calibration import Tensors
from onnxsim.onnx_simplifier import apply_billm_cpp


def _sign(w: np.ndarray) -> np.ndarray:
    """``sign(x) = 1 if x >= 0 else -1`` (paper Equation 2) -- note this is
    *not* ``np.sign``, which maps exactly 0 to 0 rather than +1.
    """
    return np.where(w >= 0.0, 1.0, -1.0)


def quantize_weight_only_billm(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    block_size: int = 128,
    percdamp: float = 0.01,
    max_salient_search: int = 30,
    skip_names: Optional[Iterable[str]] = None,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Binarizes every MatMul/vanilla-Gemm layer with a constant 2-D
    float32 weight to close to 1 bit/element on average, using BiLLM's
    Hessian-guided salient-column residual approximation -- see this
    module's own docstring for the technique and its documented
    simplification relative to the source paper.

    Unlike :func:`onnxsim.quantize_weight_only_nf4`/
    :func:`onnxsim.quantize_weight_only_kmeans` (calibration-free: every
    decision comes from the weight tensor's own values), this needs real
    calibration activations to compute each layer's Hessian, the same as
    :mod:`onnxsim.gptq`/:mod:`onnxsim.awq` -- an inherent requirement of
    BiLLM's own salient-column selection, not a simplification.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches to compute each
            layer's Hessian from -- see :func:`onnxsim.gptq.apply_gptq`'s
            own parameter of the same name
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param block_size: columns of the reduction dimension processed
            together -- both BiLLM's own salient-column search granularity
            and the OBC-style error-compensation block, paper default 128
    :param percdamp: Hessian damping factor, matching
            :func:`onnxsim.gptq.apply_gptq`'s own parameter of the same
            name and default
    :param max_salient_search: upper bound on how many leading (most
            Hessian-salient) columns of a block the search considers as
            candidates for "the salient group" -- the paper's own stated
            search range is 3-30; this module searches ``1..min(
            max_salient_search, block_size - 1)`` and lets the search
            itself settle on however few (down to 0) or many turn out
            optimal
    :param skip_names: weight initializer names to leave unquantized even
            if otherwise eligible -- ``apply_billm_cpp`` (which this
            function now delegates to) does not support this, so a
            non-empty value raises ``ValueError`` rather than being
            silently ignored
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations
    :returns: ``model`` with every matched layer's weight replaced by
            ``Add(Mul(Cast(Code1), Scale1), Mul(Cast(Code2), Scale2))``
            feeding the original MatMul/Gemm node -- ordinary ONNX ops
            only, opset 11+. Layers with a non-constant, non-2-D, or
            non-float32 weight, or whose activation input has no feature
            axis at all (rank < 2), are left untouched; a higher-rank
            ``[batch, seq, K]`` activation is flattened to
            ``[batch * seq, K]``, which is exact.
    """
    if skip_names:
        raise ValueError(
            "quantize_weight_only_billm now delegates to apply_billm_cpp, "
            "which does not support skip_names"
        )
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return apply_billm_cpp(
        model,
        calibration_data=calibration_data,
        num_samples=num_samples,
        seed=seed,
        block_size=block_size,
        percdamp=percdamp,
        max_salient_search=max_salient_search,
        providers=providers,
    )
