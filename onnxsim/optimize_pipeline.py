"""An end-to-end optimization pipeline chaining onnxsim's own recently added
techniques together, in the same escalating-and-cumulative spirit as
:mod:`onnxsim.autoquant`'s ``auto_quantize_int4`` (Qualcomm AIMET's
AutoQuant), but spanning pruning, weight-only-vs-rotated quantization
schemes, and post-hoc scale compression rather than just AIMET's own four
techniques on top of a single fixed INT4 baseline.

:func:`apply_optimization_pipeline` runs, cheapest and least invasive first:

1. **Simplify** (:func:`onnxsim.simplify`) -- always, as a fusion/cleanup
   baseline every later stage builds on.
2. **Prune** -- shrinking the model *before* quantizing means later stages
   spend their whole error budget on weights that survive, rather than on
   ones that will end up zeroed/removed anyway. Two independent
   sub-stages, each optional and order-independent (they target disjoint
   node sets -- see this module's own "C++ vs. Python" note below):
   - (optional, ``attention_sparsity``) drops whole attention heads (or,
     for grouped-query attention, whole KV groups) from every matched
     fused self-attention block --
     :func:`onnxsim.apply_attention_head_pruning_cpp` (data-free,
     magnitude-ranked, the default) or, with ``pruning_method="wanda"``,
     :func:`onnxsim.apply_attention_head_wanda_pruning` (calibrated,
     ranked by weight norm times the real activation norm flowing through
     each unit).
   - (optional, ``sparsity``) channel/filter pruning of the remaining
     MatMul/Gemm/Conv layers -- :func:`onnxsim.apply_structured_pruning_cpp`
     (data-free, the default) or, with ``pruning_method="wanda"``,
     :func:`onnxsim.apply_structured_wanda_pruning` (calibrated).
3. **Cross-Layer Equalize** (:func:`onnxsim.cross_layer_equalize`) -- always;
   data-free, changes nothing but weight parameterization (see its own
   docstring), and reduces the error every later quantization stage
   introduces.
4. **Quantize** -- one of two mutually exclusive schemes:
   - ``bit_selection="uniform"`` (default): :func:`onnxsim.quantize_weight_only_int4`,
     or, if ``calibration_data`` is available, ``bit_selection="mixed_precision"``
     upgrades this to :func:`onnxsim.apply_mixed_precision_quantization`
     (per-layer INT4/INT8 chosen by measured sensitivity).
   - ``use_rotation=True``: rotates the residual stream so *both* the
     weight and the activation can drop to INT4, a strictly more
     aggressive alternative to the weight-only scheme above, at the cost
     of the refinement stage below (AdaRound/Bias Correction only
     optimize the weight-only scheme's own ``DequantizeLinear`` shape).
     ``rotation_method`` picks which rotation: ``"quarot"`` (the default)
     -- :func:`onnxsim.apply_quarot_cpp`, data-free, a random orthogonal
     rotation; ``"duquant"`` -- :func:`onnxsim.apply_duquant`, calibrated
     permutation plus block-local random rotation, targeting the channels
     the real activation distribution concentrates outliers in;
     ``"spinquant"`` -- :func:`onnxsim.apply_spinquant`, a calibrated
     learned rotation fit in closed form. The two calibrated options need
     ``calibration_data`` (or its random-data fallback) the same way
     ``bit_selection="mixed_precision"`` does.
5. **Refine** (only reached if step 4 didn't meet ``accuracy_budget``, and
   only when ``use_rotation`` is ``False``) -- :func:`onnxsim.apply_adaround`
   then :func:`onnxsim.correct_bias`, reusing :mod:`onnxsim.autoquant`'s own
   escalation order and its reasoning for that order (AdaRound before Bias
   Correction, not after -- see that module's own docstring).
6. **Compress** (:func:`onnxsim.apply_double_quantization_cpp`) -- always
   attempted last, on whatever stage 4/5 produced: a second-level
   quantization of the scale tensors themselves, essentially free in
   accuracy (QLoRA's own finding) but not free in *risk*, so this pipeline
   still measures it and only keeps it when it doesn't measurably worsen the
   chosen stage's own accuracy.

Every stage is measured with :func:`onnxsim.measure_accuracy_drop` against
the *original* (simplified, pre-quantization) float model -- never against
the previous stage -- matching :mod:`onnxsim.autoquant`'s own convention, so
every stage's reported accuracy is directly comparable. Escalation stops as
soon as a stage meets ``accuracy_budget``; if none do, the least-lossy stage
reached is returned (``meets_budget=False``), the same fallback
:func:`onnxsim.auto_quantize_int4`/:func:`onnxsim.recommend_quantization`
already use.

**Scope note.** This targets the pipeline's own six stages above, not every
onnxsim quantization/pruning algorithm -- :func:`onnxsim.apply_awq`/
:func:`onnxsim.apply_gptq`/:func:`onnxsim.apply_smoothquant`/etc., and
SparseGPT-calibrated pruning are all still directly callable on their own,
just not wired into this particular escalation. Likewise, ``sparsity``/
``attention_sparsity`` always target the *default* topology each pruning
function matches -- ``pruning.py``'s own newer options on top of that
(``global_sparsity``, ``importance_norm``, MoE expert pruning, ...) stay
directly callable on their own too, not exposed as pipeline parameters
here.

**C++ vs. Python.** The magnitude-based pruning sub-stages (stage 2's own
default), rotation (stage 4b's own ``rotation_method="quarot"`` default),
and compression (stage 6) use C++-backed ports
(:func:`onnxsim.apply_attention_head_pruning_cpp`,
:func:`onnxsim.apply_structured_pruning_cpp`, :func:`onnxsim.apply_quarot_cpp`,
:func:`onnxsim.apply_double_quantization_cpp`) -- all four are at full scope
parity with their pure-Python counterparts for the way this pipeline calls
them (default ``block_size``/``epsilon``/``min_elements``/``sparsity``), so
the native path is strictly faster with no behavior gap. This needs
revisiting if ``pruning.py``'s own data-free ``apply_structured_pruning``/
``apply_attention_head_pruning`` grows a new *default-topology* capability
the C++ ports (``structured_pruning_entry.cpp``) haven't caught up to yet
-- ``pruning.py`` has, in fact, grown several since this port last reached
parity (packed-QKV-then-Split GroupQueryAttention, attention-mask/bias
slicing during head pruning, block-aligned grouped-Conv Concat-chain
consumers, MoE pruning), so this default path may already be behind; the
pure-Python functions remain the always-current reference regardless.
``rotation_method="duquant"``/``"spinquant"`` and ``pruning_method="wanda"``
are pure-Python only -- no C++ port exists for either (calibrated
techniques are out of this codebase's established C++-port scope, data-free/
closed-form only).

The two pruning sub-stages target disjoint node sets by construction --
:func:`onnxsim.apply_structured_pruning_cpp`'s own chain finders require a
MatMul/Gemm/Conv *consumer*, which a fused self-attention op is never
matched as, so it never touches the Q/K/V/output-projection weights
:func:`onnxsim.apply_attention_head_pruning_cpp` prunes -- so running both,
in either order, is safe. This holds for the Wanda-calibrated pair too,
same topology matching, calibrated ranking only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

import onnx

from onnxsim.accuracy import AccuracyDropReport, measure_accuracy_drop
from onnxsim.adaround import apply_adaround
from onnxsim.bias_correction import correct_bias
from onnxsim.calibration import Tensors, generate_random_calibration_data
from onnxsim.duquant import apply_duquant
from onnxsim.mixed_precision import apply_mixed_precision_quantization
from onnxsim.onnx_simplifier import (
    apply_attention_head_pruning_cpp,
    apply_double_quantization_cpp,
    apply_quarot_cpp,
    apply_structured_pruning_cpp,
    cross_layer_equalize,
    quantize_weight_only_int4,
    simplify,
)
from onnxsim.pruning import (
    apply_attention_head_wanda_pruning,
    apply_structured_wanda_pruning,
)
from onnxsim.spinquant import apply_spinquant


@dataclass
class OptimizationPipelineResult:
    """One :func:`apply_optimization_pipeline` result: the optimized model
    reached, which stages were applied to reach it (in application order,
    a prefix of ``["attention_head_pruning" | "attention_head_wanda_pruning",
    "structured_pruning" | "structured_wanda_pruning",
    "cross_layer_equalization", "quantize_weight_only_int4" |
    "apply_mixed_precision_quantization" | "quarot" | "duquant" |
    "spinquant", "adaround", "bias_correction", "double_quantization"]``),
    its measured accuracy drop against the original float model, and
    whether it met ``accuracy_budget``.
    """

    optimized_model: onnx.ModelProto
    report: AccuracyDropReport
    stages_applied: List[str] = field(default_factory=list)
    meets_budget: bool = False


def apply_optimization_pipeline(
    model: Union[str, onnx.ModelProto],
    accuracy_budget: float = 0.1,
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    providers: Optional[Sequence[str]] = None,
    sparsity: Optional[float] = None,
    attention_sparsity: Optional[float] = None,
    pruning_method: str = "magnitude",
    bit_selection: str = "uniform",
    use_rotation: bool = False,
    rotation_method: str = "quarot",
    num_adaround_iterations: int = 300,
) -> OptimizationPipelineResult:
    """Runs ``model`` through the escalating pipeline described in this
    module's own docstring, stopping as soon as a stage's measured accuracy
    (:func:`onnxsim.measure_accuracy_drop`) meets ``accuracy_budget`` or
    every stage has been tried.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param accuracy_budget: maximum acceptable worst-case relative L2 error
            (see :attr:`onnxsim.AccuracyDropReport.worst_relative_l2`) for a
            stage to be accepted
    :param calibration_data: representative input batches, used by every
            data-driven stage (mixed-precision bit selection, AdaRound, Bias
            Correction) and to measure each stage's accuracy drop. Each
            batch is a ``{input_name: np.ndarray}`` dict matching ``model``'s
            graph inputs -- see :func:`onnxsim.generate_random_calibration_data`
            (the default when omitted)
    :param num_samples: random batches to generate when ``calibration_data``
            is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied) and for :func:`onnxsim.apply_quarot_cpp`'s
            own per-layer rotations
    :param providers: onnxruntime execution providers to calibrate/run on
    :param sparsity: if given, :func:`onnxsim.apply_structured_pruning_cpp`'s
            own target fraction of output channels to remove, applied to the
            float model before quantizing; ``None`` (the default) skips
            pruning entirely
    :param attention_sparsity: if given, the target fraction of attention
            heads (or, for grouped-query attention, whole KV groups) to
            remove, applied to the float model before quantizing (and
            before ``sparsity``'s own channel pruning); ``None`` (the
            default) skips attention-head pruning entirely. Independent of
            ``sparsity`` -- either, both, or neither may be given
    :param pruning_method: ``"magnitude"`` (the default) uses the data-free,
            C++-backed :func:`onnxsim.apply_structured_pruning_cpp`/
            :func:`onnxsim.apply_attention_head_pruning_cpp`; ``"wanda"``
            instead uses the calibrated, pure-Python
            :func:`onnxsim.apply_structured_wanda_pruning`/
            :func:`onnxsim.apply_attention_head_wanda_pruning` (ranked by
            weight norm times the real activation norm over
            ``calibration_data``, no C++ port -- see this module's own
            "C++ vs. Python" note). Applies to both pruning sub-stages
            alike; has no effect when neither ``sparsity`` nor
            ``attention_sparsity`` is given
    :param bit_selection: ``"uniform"`` (the default) quantizes every layer
            to :func:`onnxsim.quantize_weight_only_int4`; ``"mixed_precision"``
            instead uses :func:`onnxsim.apply_mixed_precision_quantization`
            to pick INT4 or INT8 per layer from calibration-driven
            sensitivity. Ignored when ``use_rotation`` is ``True``.
    :param use_rotation: if ``True``, replaces the weight-only quantization
            stage with ``rotation_method``'s own rotation preprocessing
            (both weight and activation quantized to INT4); skips the
            AdaRound/Bias Correction refinement stage, since those only
            optimize the weight-only scheme's own ``DequantizeLinear``
            shape
    :param rotation_method: which rotation ``use_rotation=True`` applies --
            ``"quarot"`` (the default): :func:`onnxsim.apply_quarot_cpp`,
            data-free, no ``calibration_data`` needed; ``"duquant"``:
            :func:`onnxsim.apply_duquant`, calibrated permutation plus
            block-local random rotation; ``"spinquant"``:
            :func:`onnxsim.apply_spinquant`, a calibrated learned rotation.
            Ignored when ``use_rotation`` is ``False``
    :param num_adaround_iterations: ``num_iterations`` forwarded to
            :func:`onnxsim.apply_adaround`, only reached if quantization
            alone misses ``accuracy_budget`` and ``use_rotation`` is
            ``False``
    :returns: the best stage reached -- ``meets_budget=True`` as soon as one
            does, otherwise the least-lossy stage after every applicable
            stage has been tried
    """
    if bit_selection not in ("uniform", "mixed_precision"):
        raise ValueError(
            f'bit_selection must be "uniform" or "mixed_precision", got {bit_selection!r}'
        )
    if pruning_method not in ("magnitude", "wanda"):
        raise ValueError(
            f'pruning_method must be "magnitude" or "wanda", got {pruning_method!r}'
        )
    if rotation_method not in ("quarot", "duquant", "spinquant"):
        raise ValueError(
            'rotation_method must be "quarot", "duquant", or "spinquant", '
            f"got {rotation_method!r}"
        )
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    model, _ = simplify(model)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )

    def _measure(candidate: onnx.ModelProto) -> AccuracyDropReport:
        return measure_accuracy_drop(
            model,
            candidate,
            calibration_data=calibration_data,
            num_samples=num_samples,
            seed=seed,
            providers=providers,
        )

    def _meets(report: AccuracyDropReport) -> bool:
        return report.all_finite and report.worst_relative_l2 < accuracy_budget

    def _better(a: AccuracyDropReport, b: AccuracyDropReport) -> bool:
        if a.all_finite and not b.all_finite:
            return True
        if not a.all_finite:
            return False
        return a.worst_relative_l2 < b.worst_relative_l2

    stages: List[str] = []
    float_model = model
    if attention_sparsity is not None:
        if pruning_method == "wanda":
            float_model = apply_attention_head_wanda_pruning(
                float_model,
                calibration_data=calibration_data,
                num_samples=num_samples,
                seed=seed,
                sparsity=attention_sparsity,
                providers=providers,
            )
            stages.append("attention_head_wanda_pruning")
        else:
            float_model = apply_attention_head_pruning_cpp(
                float_model, sparsity=attention_sparsity
            )
            stages.append("attention_head_pruning")
    if sparsity is not None:
        if pruning_method == "wanda":
            float_model = apply_structured_wanda_pruning(
                float_model,
                calibration_data=calibration_data,
                num_samples=num_samples,
                seed=seed,
                sparsity=sparsity,
                providers=providers,
            )
            stages.append("structured_wanda_pruning")
        else:
            float_model = apply_structured_pruning_cpp(float_model, sparsity=sparsity)
            stages.append("structured_pruning")

    float_model = cross_layer_equalize(float_model)
    stages.append("cross_layer_equalization")

    if use_rotation:
        if rotation_method == "duquant":
            quantized = apply_duquant(
                float_model,
                calibration_data=calibration_data,
                num_samples=num_samples,
                seed=seed,
                providers=providers,
            )
            stages.append("duquant")
        elif rotation_method == "spinquant":
            quantized = apply_spinquant(
                float_model,
                calibration_data=calibration_data,
                num_samples=num_samples,
                seed=seed,
                providers=providers,
            )
            stages.append("spinquant")
        else:
            quantized = apply_quarot_cpp(float_model, seed=seed)
            stages.append("quarot")
    elif bit_selection == "mixed_precision":
        quantized = apply_mixed_precision_quantization(
            float_model,
            calibration_data=calibration_data,
            num_samples=num_samples,
            seed=seed,
            providers=providers,
        )
        stages.append("apply_mixed_precision_quantization")
    else:
        quantized = quantize_weight_only_int4(float_model)
        stages.append("quantize_weight_only_int4")

    report = _measure(quantized)
    best = OptimizationPipelineResult(quantized, report, list(stages), _meets(report))

    if not best.meets_budget and not use_rotation:
        ada_quantized = apply_adaround(
            float_model,
            quantized,
            calibration_data=calibration_data,
            providers=providers,
            num_iterations=num_adaround_iterations,
        )
        ada_report = _measure(ada_quantized)
        stages.append("adaround")
        ada_result = OptimizationPipelineResult(
            ada_quantized, ada_report, list(stages), _meets(ada_report)
        )
        if _better(ada_report, best.report):
            best = ada_result

        if not ada_result.meets_budget:
            bc_quantized = correct_bias(
                float_model,
                ada_quantized,
                calibration_data=calibration_data,
                providers=providers,
            )
            bc_report = _measure(bc_quantized)
            stages.append("bias_correction")
            bc_result = OptimizationPipelineResult(
                bc_quantized, bc_report, list(stages), _meets(bc_report)
            )
            if _better(bc_report, best.report):
                best = bc_result

    # Compression is attempted last, on whatever stage produced `best`, and
    # kept only when it doesn't measurably worsen that stage's own accuracy
    # -- see this module's own docstring for why this isn't budget-gated
    # escalation like the stages above.
    compressed = apply_double_quantization_cpp(best.optimized_model)
    compressed_report = _measure(compressed)
    if compressed_report.all_finite and (
        not best.report.all_finite
        or compressed_report.worst_relative_l2 <= best.report.worst_relative_l2 * 1.05
    ):
        best = OptimizationPipelineResult(
            compressed,
            compressed_report,
            best.stages_applied + ["double_quantization"],
            _meets(compressed_report),
        )

    return best
