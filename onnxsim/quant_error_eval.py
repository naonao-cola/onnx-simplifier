"""Evaluates onnxsim's own existing LLM quantization passes -- specifically
their channel-scaling and rotation choices -- against the theoretical
framework of Narita & Sato, 2026 ("Understanding LLM Quantization through
Activation-Guided Compensation and Orthogonal Residuals",
https://arxiv.org/abs/2609.21450; see :mod:`onnxsim.quant_error_analysis`
for the paper's own math, ported separately).

This module does not reverse-engineer an already-*applied* graph
transformation (fragile: :mod:`onnxsim.quarot`/:mod:`onnxsim.spinquant`
fuse their rotation into a quantized weight, and recovering the exact
pre-quantization float rotation from that result is not always possible).
Instead, each existing method's own *documented, closed-form* channel
statistic is recomputed directly from captured calibration data --
:func:`smoothquant_scale` below is a one-line transcription of
:mod:`onnxsim.smoothquant`'s own formula, not a graph inspection -- and
scored against the paper's bounds. This mirrors how this repo already
treats "porting the algorithm, not the framework's code" elsewhere.

:func:`evaluate_model` is the top-level entry point: given a model and
calibration data, it captures each matched layer's activation/weight pair
and returns one :class:`LayerEvaluation` per layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import onnx
import onnx.numpy_helper

from onnxsim.calibration import Tensors, generate_random_calibration_data
from onnxsim.quant_error_analysis import (
    detect_persistent_outlier_channels,
    fixed_rotation_co_bound,
    frobenius_surrogate,
    l2_channel_scale,
    linf_channel_scale,
    persistent_co_split,
    randomized_rotation_co_bound,
    sampled_rotation_co_bound,
)
from onnxsim.smoothquant import _match_matmul_like


def smoothquant_scale(
    X: np.ndarray, W: np.ndarray, alpha: float = 0.5, epsilon: float = 1e-5
) -> np.ndarray:
    """Recomputes :func:`onnxsim.smoothquant.apply_smoothquant`'s (and
    :func:`onnxsim.outlier_suppression.apply_outlier_suppression`'s) own
    per-channel migration factor, ``s_j = max(|X_j|)**alpha / max(|W_j|)**(1-alpha)``
    (see that module's docstring), directly from calibration data. At the
    default ``alpha=0.5`` this is bit-for-bit
    :func:`onnxsim.quant_error_analysis.linf_channel_scale` up to the two
    functions' (both small, both epsilon-guarding-zero) floor conventions --
    the paper's own Section 4.5 identifies this as the L-infinity relaxation
    of its L2-optimal rule, so onnxsim's existing SmoothQuant/Outlier
    Suppression passes are already, by construction, an instance of that
    relaxation.

    :param X: activations at the layer's input, ``[T, K]``
    :param W: the layer's weight, ``[K, N]``
    :param alpha: migration strength, matching ``apply_smoothquant``'s own
            parameter and default
    :param epsilon: floor, matching ``apply_smoothquant``'s own default
    """
    x_inf = np.max(np.abs(X), axis=0)
    w_inf = np.max(np.abs(W), axis=1)
    return (np.maximum(x_inf, epsilon) ** alpha) / (
        np.maximum(w_inf, epsilon) ** (1 - alpha)
    )


@dataclass
class ChannelScalingEvaluation:
    """Scores a per-channel scaling vector against Proposition 1's L2-optimal
    rule and its L-infinity relaxation, via the Frobenius surrogate ``R^2``
    (Eq. 13-16) each is derived to minimize. A ``r2_ratio_to_l2_optimal`` of
    ``1.0`` means the scored method already achieves the paper's derived
    optimum on this layer's calibration statistics; values above ``1.0``
    quantify how much slack it leaves in the bound.
    """

    method: str
    r2: float
    r2_l2_optimal: float
    r2_linf_relaxation: float
    r2_ratio_to_l2_optimal: float
    r2_ratio_to_linf_relaxation: float


def evaluate_channel_scaling(
    method: str, lam: np.ndarray, X_reg: np.ndarray, W: np.ndarray
) -> ChannelScalingEvaluation:
    """Scores an arbitrary per-channel scaling vector ``lam`` -- typically
    from :func:`smoothquant_scale`, but any positive-diagonal scaling works
    -- against :func:`onnxsim.quant_error_analysis.l2_channel_scale` (the
    paper's derived optimum) and
    :func:`onnxsim.quant_error_analysis.linf_channel_scale` (its
    SmoothQuant-style relaxation), all evaluated on the same
    ``(X_reg, W)`` via :func:`onnxsim.quant_error_analysis.frobenius_surrogate`.
    """
    lam_l2 = l2_channel_scale(X_reg, W)
    lam_linf = linf_channel_scale(X_reg, W)
    r2 = frobenius_surrogate(lam, W, X_reg)
    r2_l2 = frobenius_surrogate(lam_l2, W, X_reg)
    r2_linf = frobenius_surrogate(lam_linf, W, X_reg)
    return ChannelScalingEvaluation(
        method=method,
        r2=r2,
        r2_l2_optimal=r2_l2,
        r2_linf_relaxation=r2_linf,
        r2_ratio_to_l2_optimal=r2 / r2_l2 if r2_l2 > 0 else float("nan"),
        r2_ratio_to_linf_relaxation=r2 / r2_linf if r2_linf > 0 else float("nan"),
    )


@dataclass
class RotationCoEvaluation:
    """Diagnoses how well a given (already-fit) input-side rotation controls
    the persistent-outlier quantity ``J_co`` (Eq. 7) that Theorem 1's
    residual bound depends on, alongside Theorem 2's three closed-form
    bounds for a *signed Hadamard* transform of the same dimension --
    the paper's own proposed alternative (Section 4.6's "Signed Online
    Rotation").

    **Caveat:** Theorem 2's bounds are proved specifically for a
    ``Lambda^-1 D H^T``-structured transform (a diagonal scaling composed
    with a signed Hadamard matrix); :mod:`onnxsim.quarot`'s and
    :mod:`onnxsim.spinquant`'s own rotations are instead a generic
    (Haar-random, or data-fit) orthogonal matrix with ``Lambda = I``. The
    ``theorem2_*_bound`` fields are therefore a *reference point* -- what
    the paper's own proposed construction would achieve on this same
    calibration data -- not a proven upper bound on ``j_co_actual``.
    """

    method: str
    n_co: int
    l_max: float
    j_co_actual: float
    theorem2_fixed_bound: float
    theorem2_randomized_bound: float
    theorem2_sampled_bound: float


def evaluate_rotation_co_control(
    method: str,
    rotation: np.ndarray,
    X: np.ndarray,
    co_ratio_threshold: float = 4.0,
    delta: float = 0.05,
    n_s: int = 10,
) -> RotationCoEvaluation:
    """Computes the actual ``J_co`` (Eq. 7) a given orthogonal ``rotation``
    ([K, K]) achieves on persistent-outlier channels detected in ``X``, and
    Theorem 2's three reference bounds (Eq. 10-12) for a signed Hadamard
    transform of the same dimension -- see :class:`RotationCoEvaluation`'s
    own caveat about what these bounds do and don't prove for a
    non-Hadamard rotation.

    :param method: a label for the rotation being evaluated (e.g.
            ``"quarot"``, ``"spinquant"``), carried through to the result
    :param rotation: the fitted/sampled rotation, ``[K, K]``, orthogonal
    :param X: activations at the layer's input, ``[T, K]``
    :param co_ratio_threshold: passed to
            :func:`onnxsim.quant_error_analysis.detect_persistent_outlier_channels`
    :param delta: failure probability for the randomized/sampled bounds
    :param n_s: number of sampled sign patterns for the sampled bound
    """
    T, K = X.shape
    co_mask = detect_persistent_outlier_channels(X, co_ratio_threshold)
    n_co = int(np.sum(co_mask))
    if n_co == 0:
        return RotationCoEvaluation(
            method=method,
            n_co=0,
            l_max=0.0,
            j_co_actual=0.0,
            theorem2_fixed_bound=0.0,
            theorem2_randomized_bound=0.0,
            theorem2_sampled_bound=0.0,
        )
    X_co, _X_reg, levels = persistent_co_split(X, co_mask)
    j_co_actual = float(np.sum(np.max(np.abs(X_co @ rotation), axis=1) ** 2))
    l_max = float(np.max(np.abs(levels[co_mask])))
    return RotationCoEvaluation(
        method=method,
        n_co=n_co,
        l_max=l_max,
        j_co_actual=j_co_actual,
        theorem2_fixed_bound=fixed_rotation_co_bound(T, n_co, K, l_max),
        theorem2_randomized_bound=randomized_rotation_co_bound(
            T, n_co, K, l_max, delta
        ),
        theorem2_sampled_bound=sampled_rotation_co_bound(T, n_co, K, l_max, delta, n_s),
    )


def quarot_style_rotation(K: int, seed: int = 0) -> np.ndarray:
    """Reproduces :mod:`onnxsim.quarot`'s own per-layer rotation -- a
    Haar-random orthogonal matrix via
    :func:`onnxsim.quip_sharp._random_orthogonal_matrix` -- for direct use
    with :func:`evaluate_rotation_co_control`.
    """
    from onnxsim.quip_sharp import _random_orthogonal_matrix

    return _random_orthogonal_matrix(K, np.random.default_rng(seed))


def spinquant_style_rotation(X: np.ndarray) -> np.ndarray:
    """Reproduces :mod:`onnxsim.spinquant`'s own per-layer rotation -- the
    eigenvector basis of the calibration-activation covariance matrix --
    for direct use with :func:`evaluate_rotation_co_control`.
    """
    cov = X.T @ X
    _eigvals, eigvecs = np.linalg.eigh(cov)
    return eigvecs


# ---------------------------------------------------------------------------
# Model-level harness
# ---------------------------------------------------------------------------


def _capture_matmul_like_io(
    model: onnx.ModelProto,
    calibration_data: Sequence[Tensors],
    providers: Optional[Sequence[str]] = None,
) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """Runs ``model`` over ``calibration_data``, capturing ``(weight_name,
    X, W)`` for every MatMul/vanilla-Gemm node with a constant 2-D float32
    weight, normalized to onnxsim's ``[K, N]`` convention. Reuses the same
    "append as an extra graph output, run once" technique
    :func:`onnxsim.calibration.calibrate` already uses, rather than
    depending on :mod:`onnxsim.bias_correction`'s probe machinery.
    """
    import onnxruntime as ort

    initializer_map = {t.name: t for t in model.graph.initializer}
    candidates = []
    for node in model.graph.node:
        match = _match_matmul_like(node)
        if match is None:
            continue
        x_name, w_name, weight_transposed = match
        w_init = initializer_map.get(w_name)
        if (
            w_init is None
            or w_init.data_type != onnx.TensorProto.FLOAT
            or len(w_init.dims) != 2
        ):
            continue
        candidates.append((x_name, w_name, weight_transposed))

    if not candidates:
        return []

    probe_model = onnx.ModelProto()
    probe_model.CopyFrom(model)
    existing_outputs = {o.name for o in probe_model.graph.output}
    probe_names = sorted({x_name for x_name, _w, _t in candidates})
    for x_name in probe_names:
        if x_name not in existing_outputs:
            probe_model.graph.output.append(onnx.ValueInfoProto(name=x_name))

    sess = ort.InferenceSession(
        probe_model.SerializeToString(),
        providers=list(providers) if providers else None,
    )
    output_names = [o.name for o in sess.get_outputs()]

    collected: Dict[str, List[np.ndarray]] = {name: [] for name in probe_names}
    for batch in calibration_data:
        outputs = sess.run(output_names, batch)
        for name, value in zip(output_names, outputs):
            if name not in collected:
                continue
            arr = np.asarray(value)
            if arr.ndim >= 2:
                collected[name].append(arr.reshape(-1, arr.shape[-1]))

    results: List[Tuple[str, np.ndarray, np.ndarray]] = []
    seen_weights = set()
    for x_name, w_name, weight_transposed in candidates:
        if w_name in seen_weights:
            continue
        chunks = collected.get(x_name, [])
        if not chunks:
            continue
        X = np.concatenate(chunks, axis=0).astype(np.float64)
        w = onnx.numpy_helper.to_array(initializer_map[w_name]).astype(np.float64)
        W = w.T if weight_transposed else w  # normalize to [K, N]
        if X.shape[1] != W.shape[0]:
            continue
        results.append((w_name, X, W))
        seen_weights.add(w_name)
    return results


@dataclass
class LayerEvaluation:
    """Every evaluation this module can produce for one matched layer."""

    weight_name: str
    num_tokens: int
    k: int
    n: int
    smoothquant: ChannelScalingEvaluation
    quarot: RotationCoEvaluation
    spinquant: RotationCoEvaluation


def evaluate_model(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    providers: Optional[Sequence[str]] = None,
    alpha: float = 0.5,
    co_ratio_threshold: float = 4.0,
    delta: float = 0.05,
    n_s: int = 10,
) -> List[LayerEvaluation]:
    """Evaluates every matched MatMul/vanilla-Gemm layer of ``model``
    against the paper's analysis: for each layer's captured (calibration
    activation, weight) pair, scores
    :func:`onnxsim.smoothquant.apply_smoothquant`'s own channel-scaling rule
    against Proposition 1's L2-optimal rule
    (:func:`evaluate_channel_scaling`), and
    :func:`onnxsim.quarot.apply_quarot`'s and
    :func:`onnxsim.spinquant.apply_spinquant`'s own rotations against
    Theorem 2's Hadamard-rotation reference bounds
    (:func:`evaluate_rotation_co_control`).

    The paper's own ``X_reg`` (Section 4.4) is approximated throughout by
    the full, unsplit captured activation, exactly as the paper's own
    Section 4.6 practical recipe does ("since ``X_reg`` is not explicitly
    identified... we approximate its channel-wise statistics using the
    empirical full activations").

    :param model: the onnx ModelProto or file path to evaluate
    :param calibration_data: representative input batches -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted) and :func:`onnxsim.load_huggingface_calibration_data`
    :param num_samples: random batches to generate when ``calibration_data``
            is omitted
    :param seed: seed for the random calibration data and for
            :func:`quarot_style_rotation`
    :param providers: onnxruntime execution providers to capture on
    :param alpha: passed to :func:`smoothquant_scale`
    :param co_ratio_threshold: passed to
            :func:`onnxsim.quant_error_analysis.detect_persistent_outlier_channels`
    :param delta: passed to :func:`evaluate_rotation_co_control`
    :param n_s: passed to :func:`evaluate_rotation_co_control`
    :returns: one :class:`LayerEvaluation` per matched layer with usable
            calibration data; a model with no such layer returns ``[]``
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )

    layers = []
    for w_name, X, W in _capture_matmul_like_io(model, calibration_data, providers):
        sq_lam = smoothquant_scale(X, W, alpha=alpha)
        sq_eval = evaluate_channel_scaling("smoothquant", sq_lam, X, W)

        quarot_rotation = quarot_style_rotation(W.shape[0], seed=seed)
        quarot_eval = evaluate_rotation_co_control(
            "quarot", quarot_rotation, X, co_ratio_threshold, delta, n_s
        )

        spinquant_rotation = spinquant_style_rotation(X)
        spinquant_eval = evaluate_rotation_co_control(
            "spinquant", spinquant_rotation, X, co_ratio_threshold, delta, n_s
        )

        layers.append(
            LayerEvaluation(
                weight_name=w_name,
                num_tokens=X.shape[0],
                k=W.shape[0],
                n=W.shape[1],
                smoothquant=sq_eval,
                quarot=quarot_eval,
                spinquant=spinquant_eval,
            )
        )
    return layers
