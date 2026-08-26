"""AIMET's AutoQuant -- Qualcomm AIMET's automated, escalating pipeline that
strings its own PTQ techniques together and stops as soon as a model's
accuracy is good enough, instead of applying every technique regardless of
whether it's needed. The last of AIMET's four flagship PTQ techniques ported
to onnxsim, after Cross-Layer Equalization
(:func:`onnxsim.cross_layer_equalize`), empirical Bias Correction
(:func:`onnxsim.correct_bias`), and AdaRound (:func:`onnxsim.apply_adaround`).

AIMET's own AutoQuant targets a fixed W8A8 baseline and escalates through
CLE+BC, then AdaRound, when a user-supplied eval function reports the
baseline isn't accurate enough. onnxsim has no notion of a downstream eval
function (no labels, no task-specific metric) -- :func:`auto_quantize_int4`
instead escalates against :func:`onnxsim.measure_accuracy_drop`'s
model-agnostic relative-L2 comparison against the float model, matching
:func:`onnxsim.recommend_quantization`'s own accuracy-budget convention. It
targets :func:`onnxsim.quantize_weight_only_int4` specifically (rather than
searching bit widths/schemes the way :func:`onnxsim.recommend_quantization`
does) because AdaRound -- the strongest of the three refinements, and the
one this pipeline saves for last -- only optimizes that scheme's own output;
see :mod:`onnxsim.adaround`.

The escalation, cheapest and least invasive first, each stage strictly
cumulative on the last:

1. **Baseline**: :func:`onnxsim.quantize_weight_only_int4` alone.
2. **+ Cross-Layer Equalization**: :func:`onnxsim.cross_layer_equalize` the
   float model first (data-free, changes nothing but weight
   parameterization -- see its own docstring), then re-quantize the
   equalized model.
3. **+ AdaRound**: :func:`onnxsim.apply_adaround` on top of stage 2's
   quantized output, using real calibration data.
4. **+ Bias Correction**: :func:`onnxsim.correct_bias` on top of stage 3's
   quantized output, using the same calibration data.

AdaRound runs before Bias Correction, not after: :func:`onnxsim.correct_bias`
renames its corrected layer's own output node (inserting an ``Add`` right
after it to reclaim the original name -- see its own docstring) and
:func:`onnxsim.apply_adaround` matches a MatMul/Gemm between the float and
quantized graphs *by that same output name*, so a Bias-Correction-then-
AdaRound order would make every corrected layer invisible to AdaRound's own
candidate search. Running AdaRound first, while every layer still has its
original name, avoids the problem entirely -- and reads naturally besides:
optimize each layer's own rounding first, then correct whatever per-channel
bias is still left over once the rounding is already as good as it gets.

Stops and returns as soon as a stage's measured accuracy meets
``accuracy_budget``; if none do, returns the least-lossy stage reached
(``meets_budget=False``), same fallback convention as
:func:`onnxsim.recommend_quantization`. Every stage after the baseline is
strictly more expensive than the last (CLE is cheap and data-free; Bias
Correction and AdaRound both run real calibration data through the model,
AdaRound by far the most expensive of the three -- a per-layer gradient
optimization), so stopping early is not just an accuracy convenience but a
real cost saving, exactly AIMET AutoQuant's own point.

This module implements no actual *quantization-aware training* (QAT) --
onnxsim is a graph-transformation tool with no training loop, optimizer, or
labeled data of its own, and grafting one on would be a different, far
larger project than the rest of this PTQ-technique port. AIMET's own
AutoQuant is itself PTQ-only (it falls back to recommending the user try QAT
separately when even AdaRound doesn't meet the target) -- this module
matches that scope exactly, it does not fall short of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import onnx

from onnxsim.accuracy import AccuracyDropReport, measure_accuracy_drop
from onnxsim.adaround import apply_adaround
from onnxsim.bias_correction import correct_bias
from onnxsim.calibration import Tensors, generate_random_calibration_data
from onnxsim.onnx_simplifier import cross_layer_equalize, quantize_weight_only_int4


@dataclass
class AutoQuantResult:
    """One :func:`auto_quantize_int4` result: the quantized model reached,
    which AIMET refinement techniques were layered on top of the baseline
    INT4 weight-only quantization to reach it (in application order --
    ``[]`` means the unrefined baseline already met budget, a prefix of
    ``["cross_layer_equalization", "bias_correction", "adaround"]``
    otherwise), its measured accuracy drop, and whether it met
    ``accuracy_budget``.
    """

    quantized_model: onnx.ModelProto
    report: AccuracyDropReport
    techniques_applied: List[str]
    meets_budget: bool


def auto_quantize_int4(
    model: Union[str, onnx.ModelProto],
    accuracy_budget: float = 0.1,
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    providers: Optional[Sequence[str]] = None,
    num_adaround_iterations: int = 300,
) -> AutoQuantResult:
    """Quantizes ``model`` to INT4 weight-only (see
    :func:`onnxsim.quantize_weight_only_int4`), escalating through AIMET's
    Cross-Layer Equalization, Bias Correction, and AdaRound refinements one
    at a time until the measured accuracy (see
    :func:`onnxsim.measure_accuracy_drop`) meets ``accuracy_budget`` or every
    refinement has been tried. See this module's own docstring for the full
    escalation order and rationale.

    Every stage is measured against the *original* ``model``, not the
    previous stage, so accuracy numbers are always directly comparable to
    each other and to a plain :func:`onnxsim.quantize_weight_only_int4` call.
    ``accuracy_budget``'s default (``0.1``) is looser than
    :func:`onnxsim.recommend_quantization`'s (``0.02``): INT4 weight-only
    quantization is considerably lossier than the 8/16-bit schemes that
    function searches over (see ``test_weight_only_quantize_int4.py``'s own
    ~0.07-0.16 untuned baseline range), so a comparably tight budget would
    make even AdaRound-refined INT4 fail it on most models.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param accuracy_budget: maximum acceptable worst-case relative L2 error
            (see :attr:`onnxsim.AccuracyDropReport.worst_relative_l2`) for a
            stage to be accepted
    :param calibration_data: representative input batches, used by every
            data-driven stage (Bias Correction, AdaRound) and to measure
            each stage's accuracy drop. Each batch is a
            ``{input_name: np.ndarray}`` dict matching ``model``'s graph
            inputs -- see :func:`onnxsim.generate_random_calibration_data`
            (the default when omitted) and
            :func:`onnxsim.load_huggingface_calibration_data` (real data, a
            much more representative search than random input).
    :param num_samples: random batches to generate when ``calibration_data``
            is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param providers: onnxruntime execution providers to calibrate/run on
    :param num_adaround_iterations: ``num_iterations`` forwarded to
            :func:`onnxsim.apply_adaround`, only reached if every earlier
            stage misses ``accuracy_budget``
    :returns: the best stage reached -- ``meets_budget=True`` as soon as one
            does, otherwise the least-lossy stage after all four have been
            tried
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )

    def _measure(quantized: onnx.ModelProto) -> AccuracyDropReport:
        return measure_accuracy_drop(
            model,
            quantized,
            calibration_data=calibration_data,
            num_samples=num_samples,
            seed=seed,
            providers=providers,
        )

    def _meets(report: AccuracyDropReport) -> bool:
        return report.all_finite and report.worst_relative_l2 < accuracy_budget

    def _better(a: AccuracyDropReport, b: AccuracyDropReport) -> bool:
        # Whether report `a` should replace best-so-far report `b`.
        if a.all_finite and not b.all_finite:
            return True
        if not a.all_finite:
            return False
        return a.worst_relative_l2 < b.worst_relative_l2

    baseline = quantize_weight_only_int4(model)
    baseline_report = _measure(baseline)
    best = AutoQuantResult(baseline, baseline_report, [], _meets(baseline_report))
    if best.meets_budget:
        return best

    float_cle = cross_layer_equalize(model)

    cle_quantized = quantize_weight_only_int4(float_cle)
    cle_report = _measure(cle_quantized)
    cle_result = AutoQuantResult(
        cle_quantized, cle_report, ["cross_layer_equalization"], _meets(cle_report)
    )
    if _better(cle_report, best.report):
        best = cle_result
    if cle_result.meets_budget:
        return cle_result

    ada_quantized = apply_adaround(
        float_cle,
        cle_quantized,
        calibration_data=calibration_data,
        providers=providers,
        num_iterations=num_adaround_iterations,
    )
    ada_report = _measure(ada_quantized)
    ada_result = AutoQuantResult(
        ada_quantized,
        ada_report,
        ["cross_layer_equalization", "adaround"],
        _meets(ada_report),
    )
    if _better(ada_report, best.report):
        best = ada_result
    if ada_result.meets_budget:
        return ada_result

    bc_quantized = correct_bias(
        float_cle,
        ada_quantized,
        calibration_data=calibration_data,
        providers=providers,
    )
    bc_report = _measure(bc_quantized)
    bc_result = AutoQuantResult(
        bc_quantized,
        bc_report,
        ["cross_layer_equalization", "adaround", "bias_correction"],
        _meets(bc_report),
    )
    if _better(bc_report, best.report):
        best = bc_result
    return best
