"""TesseraQ (Dong, Yang, Wan, Wu, Cao, Cheng, Wang, Cheng, Yale University,
2024, "TesseraQ: Ultra Low-Bit LLM Post-Training Quantization with Block
Reconstruction", https://arxiv.org/abs/2410.19103).

:mod:`onnxsim.adaround` already ports this paper's closest relative -- Nagel
et al.'s AdaRound -- which optimizes a per-weight-element soft rounding
variable via the "rectified sigmoid" relaxation, pulled from soft (fully
continuous, any value in ``[0, 1]``) to hard (0 or 1) by a *single*,
smoothly-annealed regularization term applied uniformly to every element at
once: all elements are equally "soft" until late in the schedule, when the
regularizer's exponent sweep (``beta_range``) simultaneously sharpens every
element's own decision together. TesseraQ's own distinguishing
contribution, "Progressive Adaptive Rounding" (PAR), replaces that single
monolithic anneal with a **coarse-to-fine, element-by-element hardening
schedule**: reconstruction runs in a handful of rounds, and at the end of
each round the elements whose current soft value already sits closest to a
hard 0/1 decision (this module's confidence score: ``|h - 0.5]``, the
distance from the relaxation's own undecided midpoint) are permanently
frozen at that decision for the rest of optimization, while the *remaining*
still-soft elements keep being optimized against a reconstruction loss that
now sees the frozen elements as fixed constants rather than as more
optimization variables. Each round therefore reconstructs against a
smaller, better-conditioned free variable set than the last -- confident
elements stop absorbing gradient noise from indecisive neighbours, and by
construction 100% of the weight is hardened by the final round, unlike
AdaRound's `beta`-anneal which only pushes every element toward (never
strictly to) 0/1 without an explicit committal step. The paper reports this
stabilizes reconstruction specifically in the 2-3 bit regime, where
AdaRound's simultaneous-anneal tends to leave a residue of contested,
still-fractional elements that a single hard round-off at the end handles
poorly.

TesseraQ's second contribution ported here: unlike AdaRound (and
:mod:`onnxsim.adaquant`, which jointly optimizes weight rounding with the
*activation's* clip range but still leaves the *weight's* own dequantization
scale exactly as calibration left it), this module also treats each
weight block's own dequantization scale as a reconstruction-loss variable,
optimized in log-space by the same Adam loop that optimizes the rounding --
"jointly", in the sense that both live in the same computation graph and
the same per-iteration gradient step, not alternated. This matters
specifically because this module supports narrowing the *effective* bit
width below what :func:`onnxsim.quantize_weight_only_int4` originally
calibrated its scale for (INT4's own ``[-7, 7]`` symmetric range): asking
PAR to round every element into a narrower ``[-3, 3]`` (3-bit) or
``[-1, 1]`` (2-bit) range while leaving the *scale* fixed at whatever
calibration picked for the *wider* INT4 range would waste most of the
narrower range's resolution outside the weight's actual distribution --
jointly shrinking the scale during reconstruction (initialized as the
original scale, not re-derived from scratch) is what makes low bit widths
usable at all rather than merely "more clipping".

This targets exactly the same graph shape and the same
:func:`onnxsim.quantize_weight_only_int4`-produced starting point
:mod:`onnxsim.adaround` does (``DequantizeLinear(Wq, Ws, axis=<reduction
axis>, block_size=...)`` feeding a MatMul/Gemm, ``Wq`` INT4 storage, one
scale per ``(block, output channel)``) -- see
``weight_only_quantize_int4_matmul.h`` -- and reuses that module's own
candidate matcher, rectified-sigmoid relaxation helpers, and INT4 nibble
packing outright rather than re-deriving them. The *storage* container
stays the ONNX INT4 tensor type regardless of ``num_bits``: a narrower
``num_bits`` just constrains PAR's own optimization range (and the
hardened codes it ultimately writes) to a subset of the INT4 nibble's
``[-8, 7]`` range, the same way :mod:`onnxsim.spqr`/:mod:`onnxsim.billm`
reuse INT4/INT8 containers for effective bit widths below their nominal
size -- no new ONNX tensor type or contrib op is introduced.

Everything here is plain numpy with hand-derived gradients and a small
hand-rolled Adam loop, exactly :mod:`onnxsim.adaround`/
:mod:`onnxsim.adaquant`'s own style -- no autodiff framework, no
calibration data beyond what :mod:`onnxsim.calibration` already provides.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import onnx
import onnx.numpy_helper

from onnxsim.calibration import Tensors


def apply_tesseraq(
    float_model: Union[str, onnx.ModelProto],
    quantized_model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    num_bits: int = 4,
    num_iterations: int = 400,
    par_rounds: int = 4,
    learning_rate: float = 0.1,
    scale_learning_rate: float = 0.01,
    reg_param: float = 0.01,
    warm_start: float = 0.2,
    beta_range: Tuple[float, float] = (20.0, 2.0),
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Optimizes TesseraQ-style Progressive Adaptive Rounding, jointly with
    each weight block's own dequantization scale, for every
    ``quantize_weight_only_int4``-quantized MatMul/Gemm layer present (by
    node output name) in both ``float_model`` and ``quantized_model``, using
    real activations captured from ``float_model``. See this module's own
    docstring for the technique and its relationship to
    :func:`onnxsim.apply_adaround`.

    :param float_model: the original (unquantized) onnx ModelProto or file
            path
    :param quantized_model: a quantized version of ``float_model`` (onnx
            ModelProto or file path), produced by
            :func:`onnxsim.quantize_weight_only_int4`. Layers quantized by
            any other scheme (or left unquantized) are left untouched.
            Assumes ``quantized_model`` was produced from ``float_model``
            without renaming any MatMul/Gemm node's own output tensor --
            true of every onnxsim ``quantize_*`` function.
    :param calibration_data: representative input batches to optimize the
            reconstruction against. Each batch is a ``{input_name:
            np.ndarray}`` dict matching ``float_model``'s graph inputs --
            see :func:`onnxsim.generate_random_calibration_data` (the
            default when omitted) and
            :func:`onnxsim.load_huggingface_calibration_data` (real data, a
            much more representative optimization target than random
            input).
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param num_bits: effective signed bit width PAR rounds each element
            into, ``n_max = 2 ** (num_bits - 1) - 1`` (symmetric, same
            convention as :func:`onnxsim.quantize_weight_only_int4`'s own
            ``[-7, 7]``). Must be between 2 and 4 -- the codes are always
            packed into the underlying INT4 nibble storage
            :func:`onnxsim.quantize_weight_only_int4` already allocated,
            regardless of ``num_bits``; a value below 4 only narrows PAR's
            own optimization/rounding range (and relies on the jointly
            -optimized scale -- see this module's own docstring -- to make
            that narrower range usable)
    :param num_iterations: total Adam steps to run per layer, split evenly
            across ``par_rounds``
    :param par_rounds: number of Progressive Adaptive Rounding rounds. Each
            round (except the last, which hardens every remaining element)
            permanently hardens an additional ``1 / par_rounds`` of the
            *original* element count -- the most-confident still-soft
            elements first -- then continues optimizing only what remains
            soft. ``par_rounds=1`` degenerates to a single reconstruction
            round followed by one hardening step, with no progressive
            coarse-to-fine schedule
    :param learning_rate: Adam learning rate for the per-element rounding
            relaxation (same role as :func:`onnxsim.apply_adaround`'s
            ``learning_rate``)
    :param scale_learning_rate: Adam learning rate for each weight block's
            dequantization scale (optimized in log-space, as a
            multiplicative correction on top of ``quantized_model``'s own
            calibrated scale)
    :param reg_param: weight of the regularization term that pulls each
            still-soft element's relaxation toward a hard 0/1 (floor/ceil)
            decision ahead of its round's own hardening step
    :param warm_start: fraction of the total iteration budget (from the
            start, across every round) run with the regularization term
            disabled
    :param beta_range: ``(beta_start, beta_end)`` for the regularization
            term's exponent, linearly annealed across the iterations after
            ``warm_start``
    :param providers: onnxruntime execution providers to run ``float_model``
            on when capturing calibration activations
    :returns: ``quantized_model`` with every matched layer's INT4 weight
            codes and per-block scale initializers rewritten to their
            PAR-optimized values (same shape and dtype -- only the codes
            and the scale's own values change)

    The pure-Python PAR/Adam implementation above has been retired in
    favor of the verified C++ port -- this is now a thin alias for
    :func:`onnxsim.apply_tesseraq_cpp` (``onnxsim/tesseraq_entry.cpp``'s
    own ``ApplyTesseraq``), forwarding every argument unchanged.
    **Behavior change from earlier onnxsim versions:** this is an
    iterative Adam optimization, not a closed-form computation, so
    floating-point summation-order differences between the C++ port's
    own scalar dense-matmul kernels and this module's own numpy `@` can
    compound across iterations -- measured (tests/test_tesseraq_cpp.py)
    to agree with this function's own pre-alias implementation exactly
    in most configurations, but not always: a tiny, measured fraction of
    elements can land on the opposite side of the rectified sigmoid's
    0.5 soft-decision boundary (always its immediate grid neighbor) in
    the most demanding configurations tested. ``_optimize_tesseraq`` has
    been removed (nothing else in the codebase imported it); every other
    helper this module used stays available from its own home module
    (``onnxsim.adaround``'s ``_GAMMA``/``_ZETA``/``_Candidate``/
    ``_find_int4_matmul_candidates``/``_h_and_dhdv``/``_pack_int4``).
    Imported lazily (inside the function body, not at module scope) to
    avoid a circular import: ``onnxsim.onnx_simplifier`` already imports
    from this module, so importing it back at module load time here
    would deadlock the import machinery.
    """
    if not 2 <= num_bits <= 4:
        raise ValueError(f"num_bits must be between 2 and 4, got {num_bits}")

    from onnxsim.onnx_simplifier import apply_tesseraq_cpp

    return apply_tesseraq_cpp(
        float_model,
        quantized_model,
        calibration_data=calibration_data,
        num_samples=num_samples,
        seed=seed,
        num_bits=num_bits,
        num_iterations=num_iterations,
        par_rounds=par_rounds,
        learning_rate=learning_rate,
        scale_learning_rate=scale_learning_rate,
        reg_param=reg_param,
        warm_start=warm_start,
        beta_range=beta_range,
        providers=providers,
    )
