"""Sensitivity-based mixed-precision weight quantization.

Every quantizer already in onnxsim applies **one uniform scheme to the
whole model**: :func:`onnxsim.quantize_weight_only_int4` block-INT4s every
matched layer, :func:`onnxsim.accuracy.recommend_quantization` searches
across *global* schemes (try INT4-everywhere, then INT8-everywhere, ...)
and returns whichever single one meets the accuracy budget -- but nothing
in onnxsim assigns *different* bit-widths to *different layers* within one
model. That leaves real compression on the table: in any real network,
some layers are far more sensitive to quantization error than others (the
premise behind the mixed-precision/bit-width-search literature -- e.g.
HAQ, Dettmers & Zettlemoyer's LLM.int8() outlier analysis, GPTQ's own
per-layer error reporting), so spending the same number of bits on every
layer either wastes precision on layers that tolerate INT4 fine, or loses
too much on the few layers that don't.

This module is deliberately not a new *algorithm* the way
:mod:`onnxsim.spinquant`/:mod:`onnxsim.duquant` are -- it is a dispatcher
over two schemes onnxsim already has (block-wise INT4 and, for the most
sensitive layers, block-wise INT8), choosing which one each layer gets
from a data-driven **sensitivity score**, then reusing existing
graph-construction machinery for both.

**The sensitivity score.** For a layer with weight ``W`` ([N, K]) and
calibration activation ``X`` ([rows, K]), this asks "how much would this
layer's *output* change if ``W`` were quantized to INT4?" under the same
local quadratic reconstruction-error model :mod:`onnxsim.gptq` builds its
own per-weight correction from: to first order, an error ``E = W -
INT4_dequant(W)`` changes the layer's squared reconstruction error by
``mean_n(e_n @ H @ e_n^T)``, where ``H = X^T X`` (the very same per-layer
Hessian :mod:`onnxsim.gptq` computes, for a different purpose) and ``e_n``
is row ``n`` of ``E``. ``sensitivity_metric`` selects how that quantity is
turned into a per-layer score:

- ``"hessian_diag"`` (the default): drops ``H``'s off-diagonal
  (cross-input-channel) terms, i.e. treats each channel's contribution to
  the loss as independent of every other channel's:

      diag_h[k] = mean_rows(X[:, k] ** 2)          -- diag(H) / rows
      sensitivity = mean_n( sum_k diag_h[k] * e[n, k] ** 2 )

  an Optimal-Brain-Damage-style saliency, ``O(N*K)`` and needing only
  ``H``'s diagonal -- cheap enough to be the default. It improves on a
  plain ``mean((W - INT4_dequant(W))^2) * mean(X^2)`` scalar product (this
  module's very first scheme) exactly when a layer's quantization error is
  concentrated in the *same* channels its activation energy is
  concentrated in -- an outlier channel that is both large-valued and
  strongly activated, exactly the case mixed precision exists to catch --
  since that scalar product averages error and energy independently and so
  can't see that correlation.
- ``"full_hessian"``: the exact quadratic form, ``mean_n(e_n @ H @
  e_n^T)``, with no diagonal approximation -- and, since ``E`` here is
  already fully realized (every column of ``W`` was independently rounded
  to its nearest grid point, unlike a single weight's OBS-style *removal*
  in the GPTQ/OBQ sense), no matrix inversion needed either: just ``H``
  itself. ``O(N*K^2)`` time and ``O(K^2)`` memory per candidate layer (the
  full Gram matrix, not just its diagonal) to additionally capture
  *cross*-channel correlation that ``"hessian_diag"`` cannot see -- e.g.
  two input channels that tend to be large together, whose combined error
  contributes more (or less, if they tend to offset) to the loss than
  either channel's diagonal term alone would suggest. ``"hessian_diag"``
  is exactly this quantity's diagonal approximation, and matches it
  whenever ``H`` itself happens to be diagonal.

Whichever metric is used, layers are ranked by its score; the top
``high_bits_fraction`` (by count, most sensitive first) get block-wise INT8
(:func:`onnxsim.quantize_weight_only_int8_block`'s
own granularity, reimplemented locally here since that function quantizes
a whole model uniformly and can't be dispatched per-layer); every other
layer gets ordinary block-wise INT4
(:mod:`onnxsim.omniquant`'s own ``_quantize_blockwise_int4_with_clip``,
the same backend :mod:`onnxsim.spinquant`/:mod:`onnxsim.spqr` already use).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import numpy as np
import onnx
import onnx.numpy_helper

# onnxsim.accuracy does not import anything from this module (checked by
# grep), so this is not a cycle -- it is in fact the reverse of
# onnxsim/__init__.py's own import order (`accuracy` is imported well before
# `mixed_precision`), so `onnxsim.accuracy` is already fully initialized in
# `sys.modules` by the time this module is loaded as part of `import onnxsim`.
from onnxsim.accuracy import AccuracyDropReport, measure_accuracy_drop
from onnxsim.calibration import Tensors, generate_random_calibration_data


def _has_min_opset(model: onnx.ModelProto, min_version: int) -> bool:
    return any(
        o.domain in ("", "ai.onnx") and o.version >= min_version
        for o in model.opset_import
    )


SENSITIVITY_METRICS = ("hessian_diag", "full_hessian")


def _hessian_diag_sensitivity(err_nk: np.ndarray, diag_h: np.ndarray) -> float:
    """``sensitivity_metric="hessian_diag"``'s score: the diagonal
    (channel-independent) approximation of ``mean_n(e_n @ H @ e_n^T)`` --
    see module docstring. ``err_nk`` is ``E`` ([N, K]); ``diag_h`` is
    ``diag(H)`` ([K])."""
    return float(np.mean((err_nk**2) @ diag_h))


def _full_hessian_sensitivity(err_nk: np.ndarray, h: np.ndarray) -> float:
    """``sensitivity_metric="full_hessian"``'s score: the exact quadratic
    form ``mean_n(e_n @ H @ e_n^T)`` -- see module docstring. ``err_nk`` is
    ``E`` ([N, K]); ``h`` is ``H`` ([K, K]). Reduces to
    :func:`_hessian_diag_sensitivity` whenever ``h`` is itself diagonal."""
    return float(np.mean(np.sum((err_nk @ h) * err_nk, axis=1)))


def apply_mixed_precision_quantization(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    high_bits_fraction: float = 0.2,
    block_size: int = 32,
    sensitivity_metric: str = "hessian_diag",
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Quantizes every MatMul/vanilla-Gemm layer with a constant 2-D
    float32 weight whose reduction dimension ``K`` is divisible by
    ``block_size`` to either block-wise INT8 or block-wise INT4, chosen
    per layer from a calibration-driven sensitivity score -- see this
    module's own docstring.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches (each a
            ``{input_name: np.ndarray}`` dict matching ``model``'s graph
            inputs) used to measure each layer's own per-input-channel
            activation energy (and, for ``sensitivity_metric="full_hessian"``,
            cross-channel correlation) -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted) and :func:`onnxsim.load_huggingface_calibration_data`
            (real data, a more representative ranking than random input)
    :param num_samples: random batches to generate when ``calibration_data``
            is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param high_bits_fraction: fraction of matched layers (by count, most
            sensitive first) that get block-wise INT8 instead of INT4;
            ``0.0`` quantizes every layer to INT4 (matching
            :func:`onnxsim.quantize_weight_only_int4`'s own behavior),
            ``1.0`` quantizes every layer to INT8
    :param block_size: elements per quantization block along ``K``, for
            both the INT4 and INT8 tiers -- matching
            :func:`onnxsim.quantize_weight_only_int4`'s own default
    :param sensitivity_metric: ``"hessian_diag"`` (the default) or
            ``"full_hessian"`` -- which per-layer sensitivity score to rank
            candidate layers by; see this module's own docstring for the
            formulas and their cost/accuracy tradeoff. Raises
            :class:`ValueError` for any other value.
    :param providers: onnxruntime execution providers to run calibration on
    :returns: ``model`` with every matched layer's weight replaced by
            block-wise INT4 or INT8 codes plus a per-block float32 scale
            (``DequantizeLinear(..., axis=0, block_size=block_size)``);
            output tensor name unchanged. Layers with a non-constant,
            non-2-D weight, a reduction dimension not divisible by
            ``block_size``, or no calibration activation available, are
            left untouched; a model with no matching layer, or an opset
            older than 21 (INT4's tensor type and ``DequantizeLinear``'s
            ``block_size`` attribute both need opset 21), is returned
            unchanged

    Delegates to the verified C++ port
    (:func:`onnxsim.apply_mixed_precision_quantization_cpp`), which
    reimplements this function's own candidate matching, Hessian-diagonal/
    full-Hessian accumulation, and block-wise INT4/INT8 RTN quantization
    exactly (a closed-form computation with no RNG or gradient descent --
    see ``mixed_precision_entry.h`` for the full scope). This function's
    own former pure-Python implementation is preserved as-is in this
    module's own git history; the standalone ``_hessian_diag_sensitivity``/
    ``_full_hessian_sensitivity``/``_quantize_blockwise_int8`` helpers
    above remain (the first two are still directly unit-tested by
    ``tests/test_mixed_precision.py`` at the pure-math level, independent
    of this function).
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if sensitivity_metric not in SENSITIVITY_METRICS:
        raise ValueError(
            f"sensitivity_metric must be one of {SENSITIVITY_METRICS}, "
            f"got {sensitivity_metric!r}"
        )
    from onnxsim.onnx_simplifier import apply_mixed_precision_quantization_cpp

    return apply_mixed_precision_quantization_cpp(
        model,
        calibration_data=calibration_data,
        num_samples=num_samples,
        seed=seed,
        high_bits_fraction=high_bits_fraction,
        block_size=block_size,
        sensitivity_metric=sensitivity_metric,
        providers=providers,
    )


DEFAULT_SEARCH_FRACTIONS: Sequence[float] = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
"""Default ``fractions`` sweep for :func:`search_mixed_precision_for_budget`,
front-loaded towards small ``high_bits_fraction`` values (dense near 0,
sparse near 1). Most of the compression benefit of mixed precision comes
from keeping ``high_bits_fraction`` small -- only the few outlier-sensitive
layers actually need INT8 -- so most models that can meet a reasonable
accuracy budget at all will meet it at one of these early, small fractions.
Trying small fractions first lets the search stop (see
:func:`search_mixed_precision_for_budget`'s early-stopping) after as few
:func:`onnxsim.accuracy.measure_accuracy_drop` calls as possible, since each
one re-runs both the float and quantized model on every calibration sample.
"""


@dataclass
class MixedPrecisionSearchResult:
    """One :func:`search_mixed_precision_for_budget` result: the winning (or,
    if none met the budget, least-lossy -- i.e. highest ``high_bits_fraction``
    -- one tried) fraction, its measured accuracy drop, and the model already
    quantized with it. Mirrors :class:`onnxsim.accuracy.QuantizationRecommendation`'s
    shape, adapted to a search over ``high_bits_fraction`` within the mixed-
    precision scheme rather than over whole quantization schemes.
    """

    high_bits_fraction: float
    report: AccuracyDropReport
    quantized_model: onnx.ModelProto
    meets_budget: bool
    fractions_tried: List[float]


def search_mixed_precision_for_budget(
    model: Union[str, onnx.ModelProto],
    accuracy_budget: float = 0.02,
    fractions: Optional[Sequence[float]] = None,
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    block_size: int = 32,
    providers: Optional[Sequence[str]] = None,
) -> MixedPrecisionSearchResult:
    """Accuracy-aware search over :func:`apply_mixed_precision_quantization`'s
    own ``high_bits_fraction`` parameter: tries ``fractions`` in order
    (default :data:`DEFAULT_SEARCH_FRACTIONS`, smallest -- most compressed --
    first), stopping as soon as one fraction's measured accuracy drop
    (:func:`onnxsim.accuracy.measure_accuracy_drop`) meets ``accuracy_budget``.

    This is the "iterate until target met" control loop described as the
    missing piece over :func:`apply_mixed_precision_quantization`'s own
    single-shot per-layer sensitivity dispatcher -- promoting more of the
    most-sensitive layers to INT8 (by trying larger and larger
    ``high_bits_fraction`` values) until the model's *actual measured*
    accuracy drop, not just its estimated per-layer sensitivity score, is
    under budget. It is not a replacement for
    :func:`onnxsim.accuracy.recommend_quantization`, which searches across
    whole quantization *schemes* rather than per-layer bit-width assignment
    within this one scheme.

    ``calibration_data`` is generated once (if not supplied) and reused,
    unchanged, for every fraction tried -- both to quantize
    (:func:`apply_mixed_precision_quantization`'s own sensitivity ranking)
    and to measure (:func:`onnxsim.accuracy.measure_accuracy_drop`). Measuring
    every fraction against the same data is what makes their accuracy drops
    comparable at all; regenerating fresh random data per fraction would make
    each measurement a comparison against different noise.

    If ``model`` has no eligible mixed-precision candidate,
    :func:`apply_mixed_precision_quantization` returns it unchanged
    regardless of ``high_bits_fraction`` -- every fraction would measure the
    same (no-op) accuracy drop, so this doesn't special-case that: it simply
    tries ``fractions`` in order as usual and stops at ``fractions[0]``
    (typically meeting even a tight budget, since nothing was quantized) or
    proceeds through the full list if ``fractions[0]`` itself doesn't meet
    budget for some other reason (e.g. an already-embedded quantization
    error in ``model``, or an unreasonably tight ``accuracy_budget``).

    :param model: the original (unquantized) onnx ModelProto or file path
    :param accuracy_budget: maximum acceptable worst-case relative L2 error
            (see :attr:`onnxsim.accuracy.AccuracyDropReport.worst_relative_l2`)
            for a fraction to be accepted
    :param fractions: ``high_bits_fraction`` values to try, in the order
            given -- defaults to :data:`DEFAULT_SEARCH_FRACTIONS`. Must be
            non-empty. Not required to be sorted, but since the search stops
            at the first fraction that meets budget, an order other than
            increasing-fraction defeats the "stop as early as possible on the
            most-compressed option" intent
    :param calibration_data: representative input batches, used both to rank
            per-layer sensitivity (:func:`apply_mixed_precision_quantization`)
            and to measure accuracy drop
            (:func:`onnxsim.accuracy.measure_accuracy_drop`) at every
            fraction tried. See :func:`onnxsim.generate_random_calibration_data`
            (the default when omitted) and
            :func:`onnxsim.load_huggingface_calibration_data` (real data, a
            more representative search than random input)
    :param num_samples: random batches to generate when ``calibration_data``
            is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param block_size: elements per quantization block, forwarded to
            :func:`apply_mixed_precision_quantization` for every fraction
            tried
    :param providers: onnxruntime execution providers to calibrate/measure on
    :returns: the winning (or, if none met budget, last-tried) fraction's
            result. ``result.fractions_tried`` lists every fraction actually
            measured, in the order tried, so a caller can tell e.g. that only
            ``fractions[0]`` was needed
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if fractions is None:
        fractions = DEFAULT_SEARCH_FRACTIONS
    if not fractions:
        raise ValueError("`fractions` must be non-empty")
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )

    fractions_tried: List[float] = []
    result: Optional[MixedPrecisionSearchResult] = None
    for frac in fractions:
        fractions_tried.append(frac)
        quantized = apply_mixed_precision_quantization(
            model,
            calibration_data=calibration_data,
            num_samples=num_samples,
            seed=seed,
            high_bits_fraction=frac,
            block_size=block_size,
            providers=providers,
        )
        report = measure_accuracy_drop(
            model,
            quantized,
            calibration_data=calibration_data,
            num_samples=num_samples,
            seed=seed,
            providers=providers,
        )
        meets_budget = report.all_finite and report.worst_relative_l2 < accuracy_budget
        result = MixedPrecisionSearchResult(
            high_bits_fraction=frac,
            report=report,
            quantized_model=quantized,
            meets_budget=meets_budget,
            fractions_tried=list(fractions_tried),
        )
        if meets_budget:
            return result

    assert result is not None  # `fractions` was checked non-empty above
    return result
