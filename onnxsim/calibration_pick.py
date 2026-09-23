"""Pick one calibration method for a whole model by what it does to the
model's *outputs*, not to each tensor on its own.

:func:`onnxsim.calibrate`'s ``method="auto"`` chooses a clip range per tensor
by that tensor's own expected quantization error. That is cheap and usually
right, but a tensor's error is not the task's error: YOLO11n's class logits
are almost all background, so every clipping method looks *better* on them
tensor-wise while it caps the rare confident scores that are the detections.
:func:`pick_calibration` instead quantizes the model once per candidate
method -- all from **one** calibration run (:func:`onnxsim.calibration.
collect_calibration_stats`: the histograms every method's ranges derive from)
-- runs each quantized model on held-out data, and keeps the candidate a
metric scores best: by default the worst output's SQNR, or any
``metric(float_outputs, quant_outputs) -> score`` the caller has (e.g. a
detector's matched-box count).
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import onnx

from onnxsim.calibration import (
    CalibrationStats,
    Tensors,
    _StaticPlan,
    collect_calibration_stats,
    generate_random_calibration_data,
)

Outputs = List[Dict[str, np.ndarray]]  # one {output_name: array} per batch

DEFAULT_PICK_CANDIDATES = (
    "minmax",
    "mse",
    "percentile:99.999",
    "percentile:99.99",
    "entropy",
    "auto",
)


@dataclass
class CalibrationPick:
    """:func:`pick_calibration`'s result."""

    method: str  # the winning candidate
    score: float  # its metric score (higher is better)
    scores: Dict[str, float]  # every candidate's score, in candidate order
    ranges: Dict[str, Tuple[float, float]]  # the winner's calibration ranges
    model: onnx.ModelProto  # the winner's quantized model
    # per-tensor methods "auto" chose (set when "auto" was a candidate)
    auto_choices: Dict[str, str] = field(default_factory=dict)


def worst_output_sqnr(float_outputs: Outputs, quant_outputs: Outputs) -> float:
    """The lowest per-output signal-to-quantization-noise ratio in dB over
    every batch (outputs compared one by one, so a small, crushed output --
    a score head next to a large box head -- is not averaged away). ``-inf``
    if any quantized output is not finite; capped at 200 dB (exact)."""
    worst = float("inf")
    for name in float_outputs[0]:
        f = np.concatenate(
            [np.ravel(b[name]).astype(np.float64) for b in float_outputs]
        )
        q = np.concatenate(
            [np.ravel(b[name]).astype(np.float64) for b in quant_outputs]
        )
        if f.shape != q.shape or not np.all(np.isfinite(q)):
            return float("-inf")
        noise = float(np.sum((f - q) ** 2))
        signal = float(np.sum(f**2))
        sqnr = 200.0 if noise == 0.0 else 10.0 * np.log10(max(signal, 1e-30) / noise)
        worst = min(worst, min(sqnr, 200.0))
    return worst


def run_outputs(
    model: onnx.ModelProto,
    data: Sequence[Tensors],
    providers: Optional[Sequence[str]] = None,
) -> Outputs:
    """Every output of ``model`` on every batch, through ONNX Runtime at its
    *basic* optimization level: the extended level fuses DQ -> Conv/Gemm/
    MatMul -> Q into u8s8 integer kernels that saturate on x86 CPUs without
    VNNI, so the ranking would depend on the host CPU instead of the
    calibration. Basic keeps every QDQ pair (no QLinear/integer op appears)
    and is ~4.7x faster than no optimization on YOLO11n."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess = ort.InferenceSession(
        model.SerializeToString(),
        so,
        providers=list(providers) if providers else ["CPUExecutionProvider"],
    )
    names = [o.name for o in sess.get_outputs()]
    return [dict(zip(names, sess.run(names, batch))) for batch in data]


def pick_calibration(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    eval_data: Optional[Sequence[Tensors]] = None,
    metric: Optional[Callable[[Outputs, Outputs], float]] = None,
    candidates: Sequence[str] = DEFAULT_PICK_CANDIDATES,
    providers: Optional[Sequence[str]] = None,
    full_graph: bool = False,
    per_channel: bool = True,
    nodes_to_exclude: Optional[Sequence[str]] = None,
    op_types_to_exclude: Optional[Sequence[str]] = None,
    activation_type: str = "uint8",
    minmax_tensor_names: Optional[Sequence[str]] = None,
    auto_options: Optional[Dict] = None,
    folds: int = 0,
    num_calibration_samples: int = 8,
    seed: int = 0,
    verbose: bool = False,
    **range_options,
) -> CalibrationPick:
    """Quantize ``model`` (as :func:`onnxsim.quantize_static` would, with the
    same ``full_graph``/exclusion/``activation_type`` options) once per
    ``candidates`` method and return the one ``metric`` scores highest on
    ``eval_data``.

    Without ``folds`` the model runs over ``calibration_data`` once
    (:func:`onnxsim.calibration.collect_calibration_stats`, two streaming
    passes); every candidate's ranges come from those statistics. Then each
    candidate's quantized model runs over ``eval_data``, as does the float
    model once, all through ORT without its QDQ -> integer-kernel fusion (see
    :func:`run_outputs`).

    :param calibration_data: batches to calibrate on (default: random, see
            :func:`onnxsim.generate_random_calibration_data` -- a poor proxy
            for a real distribution; pass real data)
    :param eval_data: held-out batches to score on. Defaults to
            ``calibration_data`` -- which favors whatever fits those exact
            batches; hold some out when the data allows, or use ``folds``.
    :param folds: ``k >= 2`` cross-fits instead of using ``eval_data``: the
            calibration batches are split into ``k`` folds (every ``k``-th
            batch), each fold is scored by candidates calibrated on the other
            ``k - 1``, and a candidate's score is the size-weighted mean of
            its fold scores; the winner is then calibrated on all the data.
            Every batch gets scored, none by a model calibrated on it, at the
            cost of ``k + 1`` calibration runs (each shared by all
            candidates). On YOLO11n a single 16-image holdout ranked
            percentile 99.99 first; 4 folds over the same 64 images ranked
            mse first, as 128 separate images do.
    :param metric: ``metric(float_outputs, quant_outputs) -> float``, higher
            is better; each argument is a list with one
            ``{output_name: array}`` per ``eval_data`` batch. Default:
            :func:`worst_output_sqnr`. A task metric (matched detections,
            top-1 agreement) is what catches a method that looks fine
            tensor-wise but clips the part of a tensor the task needs.
    :param candidates: methods to compare, in order (ties keep the earlier
            one): any :func:`onnxsim.calibrate` method,
            ``"percentile:<p>"``, or ``"auto"``
    :param auto_options: passed to
            :meth:`onnxsim.calibration.CalibrationStats.auto_ranges` for the
            ``"auto"`` candidate
    :param range_options: passed to
            :meth:`onnxsim.calibration.CalibrationStats.ranges` (e.g.
            ``num_mse_candidates``)
    :returns: a :class:`CalibrationPick`
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if not candidates:
        raise ValueError("no candidates")
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_calibration_samples, seed=seed
        )
    elif not isinstance(calibration_data, Sequence):
        calibration_data = list(calibration_data)
    if folds >= 2 and eval_data is not None:
        raise ValueError("folds cross-fits on calibration_data: pass no eval_data")
    if eval_data is None:
        eval_data = calibration_data
    metric = metric or worst_output_sqnr
    plan = _StaticPlan(
        model,
        full_graph=full_graph,
        per_channel=per_channel,
        nodes_to_exclude=nodes_to_exclude,
        op_types_to_exclude=op_types_to_exclude,
        activation_type=activation_type,
        minmax_tensor_names=minmax_tensor_names,
    )
    auto_kw = dict(dict(activation_type=activation_type), **(auto_options or {}))

    def candidate_ranges(stats: CalibrationStats, c: str):
        if c == "auto":
            return stats.auto_ranges(
                minmax_tensor_names=plan.minmax_tensor_names, **auto_kw
            )
        ranges = stats.ranges(
            c, minmax_tensor_names=plan.minmax_tensor_names, **range_options
        )
        return ranges, {}

    def calibrated(data: Sequence[Tensors]) -> CalibrationStats:
        return collect_calibration_stats(
            model, data, providers=providers, tensor_names=plan.tensor_names
        )

    scores: Dict[str, float] = {c: 0.0 for c in candidates}
    if folds >= 2:
        # Cross-fitting: fold f is scored by models calibrated on the others.
        n = len(calibration_data)
        if n < folds:
            raise ValueError(f"{n} calibration batches are too few for {folds} folds")
        for f in range(folds):
            ev = [b for i, b in enumerate(calibration_data) if i % folds == f]
            cal = [b for i, b in enumerate(calibration_data) if i % folds != f]
            stats = calibrated(cal)
            float_out = run_outputs(model, ev, providers)
            for c in candidates:
                q = plan.apply(candidate_ranges(stats, c)[0])
                s = float(metric(float_out, run_outputs(q, ev, providers)))
                scores[c] += s * len(ev) / n
                if verbose:
                    print(f"  fold {f} calibration {c}: {s:.6g}")
        winner = max(candidates, key=lambda c: (scores[c], -candidates.index(c)))
        final, final_choices = candidate_ranges(calibrated(calibration_data), winner)
        if verbose:
            for c in candidates:
                print(f"  calibration {c}: {scores[c]:.6g} (mean over {folds} folds)")
        return CalibrationPick(
            winner, scores[winner], scores, final, plan.apply(final), final_choices
        )

    stats = calibrated(calibration_data)
    float_out = run_outputs(model, eval_data, providers)
    best: Optional[CalibrationPick] = None
    auto_choices: Dict[str, str] = {}
    for c in candidates:
        ranges, choices = candidate_ranges(stats, c)
        auto_choices = choices or auto_choices
        q = plan.apply(ranges)
        score = float(metric(float_out, run_outputs(q, eval_data, providers)))
        scores[c] = score
        if verbose:
            print(f"  calibration {c}: {score:.6g}")
        if best is None or score > best.score:
            best = CalibrationPick(c, score, scores, ranges, q)
    assert best is not None
    best.auto_choices = auto_choices
    return best
