"""Local weight-activation quantization error analysis (Narita & Sato, 2026,
"Understanding LLM Quantization through Activation-Guided Compensation and
Orthogonal Residuals", https://arxiv.org/abs/2609.21450). onnxsim ports the
paper's analysis framework itself -- an exact error decomposition plus
theoretical bounds, not any framework's implementation of a quantizer -- so
its own existing quantization passes (:mod:`onnxsim.smoothquant`,
:mod:`onnxsim.outlier_suppression`, :mod:`onnxsim.quarot`,
:mod:`onnxsim.spinquant`, :mod:`onnxsim.gptaq`, ...) can be *evaluated*
against it. See :mod:`onnxsim.quant_error_eval` for that evaluation harness;
this module only implements the paper's own math.

**Setting (Section 3.1).** For a linear layer ``Y = X @ W`` with activation
``X in R^{T x K}`` (``T`` calibration tokens, ``K`` the reduction/input
dimension) and weight ``W in R^{K x N}`` (``N`` output channels) -- onnxsim's
own convention throughout this codebase (see e.g. ``onnxsim/spinquant.py``),
the transpose of the paper's own ``Y = X W^T`` with ``W in R^{N x K}``. An
invertible transformation ``P in R^{K x K}`` gives ``Z = X @ P``,
``V = P^{-1} @ W`` (this module's transpose of the paper's own
``V = W P^{-T}``), so that ``X @ W == Z @ V`` -- ``P`` only changes what the
quantizers see, not the float output. Every formula below is translated into
this row/column convention; each docstring cites the paper's own equation
number so it can be checked back against the source.

**Persistent channel-wise outlier (CO) model (Section 3.2, Eq. 3).** A
"persistent" outlier channel is one whose magnitude stays large across
essentially every token, as opposed to one that is merely large on a few
tokens; such channels are observed to dominate per-token quantization scales
(Xiao et al., 2023). The paper models this by writing
``X = X_co + X_reg``, where ``X_co`` holds a token-shared "level" ``L_k`` on
each detected CO channel ``k`` (and zero elsewhere) and ``X_reg`` is the
remainder. The paper itself does not give a closed-form CO *detector* (it
cites Raman et al., 2025 for one); :func:`detect_persistent_outlier_channels`
below is onnxsim's own simple proxy for that detector, not part of the
paper's own claims -- see that function's docstring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Section 3.2 -- persistent channel-wise outlier decomposition
# ---------------------------------------------------------------------------


def detect_persistent_outlier_channels(
    X: np.ndarray, ratio_threshold: float = 4.0
) -> np.ndarray:
    """Flags channels whose activation magnitude is *persistently* large
    across calibration tokens -- onnxsim's own heuristic proxy for the
    paper's ``C_co(X)`` (Section 3.2), since the paper's own text does not
    give a closed-form criterion and instead cites Raman et al. (2025) for
    one. This uses each channel's median absolute value (a level that stays
    large across most tokens raises the *median*, unlike a channel that is
    merely bursty on a few tokens) relative to the median such level across
    all channels; a channel is flagged when that ratio exceeds
    ``ratio_threshold``.

    :param X: activations, ``[T, K]``
    :param ratio_threshold: a channel's median-abs value must exceed this
            multiple of the cross-channel median to be flagged
    :returns: boolean mask, ``[K]``
    """
    med_abs = np.median(np.abs(X), axis=0)
    baseline = np.median(med_abs)
    if baseline <= 0.0:
        return np.zeros(X.shape[1], dtype=bool)
    return med_abs > ratio_threshold * baseline


def persistent_co_split(
    X: np.ndarray, co_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Eq. 3: decomposes ``X = X_co + X_reg`` under the persistent sparse CO
    model, given a channel mask (e.g. from
    :func:`detect_persistent_outlier_channels`). Each CO channel's level
    ``L_k`` is estimated as that channel's median value across tokens (a
    signed, robust estimate of the token-shared level the paper's ``L_k``
    denotes); non-CO channels get ``L_k = 0`` by definition.

    :param X: activations, ``[T, K]``
    :param co_mask: boolean mask, ``[K]``
    :returns: ``(X_co, X_reg, levels)`` -- ``X_co``/``X_reg`` are ``[T, K]``
            and sum to ``X``; ``levels`` is ``[K]``, zero outside ``co_mask``
    """
    levels = np.zeros(X.shape[1], dtype=X.dtype)
    if np.any(co_mask):
        levels[co_mask] = np.median(X[:, co_mask], axis=0)
    X_co = np.broadcast_to(levels, X.shape) * co_mask
    X_reg = X - X_co
    return X_co, X_reg.copy(), levels


# ---------------------------------------------------------------------------
# Section 4.1 -- Theorem 1: exact decomposition and residual upper bound
# ---------------------------------------------------------------------------


def local_reconstruction_error(
    Z: np.ndarray, Z_tilde: np.ndarray, V: np.ndarray, V_tilde: np.ndarray
) -> float:
    """Eq. 2: ``J(P, Vtilde) = (1/T) * ||Z_tilde @ Vtilde - Z @ V||_F^2``.

    :param Z: unquantized transformed activation, ``[T, K]``
    :param Z_tilde: quantized transformed activation, ``[T, K]``
    :param V: unquantized transformed weight, ``[K, N]``
    :param V_tilde: quantized transformed weight, ``[K, N]``
    """
    T = Z.shape[0]
    diff = Z_tilde @ V_tilde - Z @ V
    return float(np.sum(diff * diff) / T)


@dataclass
class LocalErrorDecomposition:
    """Theorem 1's exact decomposition of :func:`local_reconstruction_error`
    into a weight-compensable term and an orthogonal residual. ``j_total``
    equals ``j_agwc + j_residual`` to floating-point precision -- this is an
    algebraic identity (Eq. 4), not an approximation."""

    j_total: float
    j_agwc: float
    j_residual: float
    v_star: np.ndarray  # [K, N] -- the continuous target Vtilde is measured against


def decompose_local_error(
    Z: np.ndarray, Z_tilde: np.ndarray, V: np.ndarray, V_tilde: np.ndarray
) -> LocalErrorDecomposition:
    """Theorem 1 (Eq. 4, 6, proof Eq. 25-31): exactly decomposes
    :func:`local_reconstruction_error` into

    - ``j_agwc`` -- the activation-guided weight compensation term: it lies
      in the column space of ``Z_tilde`` and is therefore the part of the
      error a weight-only optimization (GPTQ/GPTAQ-style) can still reduce,
      by driving ``V_tilde`` towards the continuous target ``v_star``
      (returned alongside it) rather than the original ``V``;
    - ``j_residual`` -- the orthogonal complement, unreachable by *any*
      choice of ``V_tilde`` for this fixed transformation ``P``.

    Both a quantized activation (``Z_tilde != Z``) and an unquantized one
    (``Z_tilde == Z``, where ``j_residual`` is exactly zero and
    ``v_star == V``) are handled -- the latter is Section 4.2's "shared-input
    reconstruction" special case, where this reduces to plain GPTQ's own
    objective.

    :param Z: unquantized transformed activation, ``[T, K]``
    :param Z_tilde: quantized transformed activation, ``[T, K]``
    :param V: unquantized transformed weight, ``[K, N]``
    :param V_tilde: quantized transformed weight, ``[K, N]``
    """
    T = Z.shape[0]
    A = Z_tilde - Z  # [T, K]
    Z_tilde_pinv = np.linalg.pinv(Z_tilde)  # [K, T]
    v_star = V - Z_tilde_pinv @ (A @ V)  # [K, N], Z_tilde^+ A V

    diff_agwc = Z_tilde @ (V_tilde - v_star)  # [T, N]
    j_agwc = float(np.sum(diff_agwc * diff_agwc) / T)

    proj = Z_tilde @ Z_tilde_pinv  # [T, T], orthogonal projector onto col(Z_tilde)
    residual = (np.eye(T) - proj) @ (A @ V)  # [T, N]
    j_residual = float(np.sum(residual * residual) / T)

    j_total = local_reconstruction_error(Z, Z_tilde, V, V_tilde)
    return LocalErrorDecomposition(
        j_total=j_total, j_agwc=j_agwc, j_residual=j_residual, v_star=v_star
    )


def residual_upper_bound(
    W: np.ndarray,
    lam: np.ndarray,
    signs: np.ndarray,
    H: np.ndarray,
    X_co: np.ndarray,
    X_reg: np.ndarray,
    q_a: float,
) -> Tuple[float, float, float]:
    """Eq. 5, 7, 8: the transformation-dependent upper bound on the
    orthogonal residual, for ``P^-1 = Lambda^-1 @ D @ H^T`` (a diagonal
    scaling ``Lambda``, a diagonal sign matrix ``D``, and an orthogonal
    matrix ``H``, canonically the normalized Hadamard matrix -- see
    :func:`normalized_hadamard_matrix`).

    :param W: the *untransformed* weight, ``[K, N]`` (onnxsim's row/column
            transpose of the paper's own ``W in R^{N x K}``)
    :param lam: per-channel scale, ``Lambda``'s diagonal, ``[K]``
    :param signs: per-channel sign, ``D``'s diagonal, ``[K]`` of ``+-1``
    :param H: orthogonal matrix, ``[K, K]``
    :param X_co: persistent-outlier activation component, ``[T, K]``
    :param X_reg: regular activation component, ``[T, K]``
    :param q_a: ``2**(bit_width - 1) - 1``, the activation quantizer's max
            representable magnitude
    :returns: ``(j_co, j_reg, bound)`` -- Eq. 7's, Eq. 8's, and Eq. 5's
            second term (the bound on ``j_residual``), respectively.
            ``j_reg`` already includes the ``||W @ Lambda||_2^2`` factor.
    """
    T, K = X_co.shape
    # P^-1 = Lambda^-1 @ D @ H^T; since Lambda^-1 and D are diagonal, this is
    # H^T with row k scaled by signs[k] / lam[k].
    row_scale = signs / lam
    m = row_scale[:, None] * H.T  # [K, K]

    co_proj = X_co @ m  # [T, K]
    j_co = float(np.sum(np.max(np.abs(co_proj), axis=1) ** 2))

    reg_proj = X_reg @ m  # [T, K]
    reg_inf_sq_sum = float(np.sum(np.max(np.abs(reg_proj), axis=1) ** 2))
    # ||W @ Lambda||_2 (paper's ||V||_2, V = W @ Lambda in this convention)
    w_lambda_spec2 = float(np.linalg.norm(lam[:, None] * W, ord=2) ** 2)
    j_reg = w_lambda_spec2 * reg_inf_sq_sum

    bound = (K / (4 * q_a**2 * T)) * (
        np.sqrt(w_lambda_spec2 * j_co) + np.sqrt(j_reg)
    ) ** 2
    return j_co, j_reg, float(bound)


def normalized_hadamard_matrix(d: int) -> np.ndarray:
    """The normalized (orthogonal) Hadamard matrix used throughout Section 4
    as ``H``: entries ``+-1/sqrt(d)``, symmetric, ``H @ H == I``. Only
    defined for ``d`` a power of two (the standard Sylvester construction);
    raises :class:`ValueError` otherwise, matching onnxsim's own convention
    elsewhere of declining rather than silently padding/approximating (see
    e.g. :mod:`onnxsim.quip_sharp`'s own docstring on why it uses a generic
    random-orthogonal construction instead of a Hadamard one for exactly
    this reason).
    """
    if d <= 0 or (d & (d - 1)) != 0:
        raise ValueError(f"normalized_hadamard_matrix needs a power of two, got {d}")
    h = np.array([[1.0]])
    while h.shape[0] < d:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(d)


# ---------------------------------------------------------------------------
# Section 4.3 -- Theorem 2: Hadamard rotation bounds on J_co
# ---------------------------------------------------------------------------


def fixed_rotation_co_bound(T: int, n_co: int, d: int, l_lambda_max: float) -> float:
    """Eq. 10: ``J_co(Lambda, I) <= T * n_co^2 / d * l_lambda_max^2`` for a
    fixed (unsigned, ``D = I``) Hadamard rotation."""
    return T * n_co**2 / d * l_lambda_max**2


def randomized_rotation_co_bound(
    T: int, n_co: int, d: int, l_lambda_max: float, delta: float
) -> float:
    """Eq. 11: with probability >= ``1 - delta``,
    ``J_co(Lambda, D) <= 2*T*n_co/d * l_lambda_max^2 * (log(2d) + log(1/delta))``
    for a Hadamard rotation with independent Rademacher signs ``D``."""
    return 2 * T * n_co / d * l_lambda_max**2 * (np.log(2 * d) + np.log(1.0 / delta))


def sampled_rotation_co_bound(
    T: int, n_co: int, d: int, l_lambda_max: float, delta: float, n_s: int
) -> float:
    """Eq. 12: with probability >= ``1 - delta``, the *best* of ``n_s``
    independently sampled sign patterns achieves
    ``min_D J_co(Lambda, D) <= 2*T*n_co/d * l_lambda_max^2 * (log(2d) + log(1/delta)/n_s)``.
    Monotonically non-increasing in ``n_s`` (``n_s=1`` recovers
    :func:`randomized_rotation_co_bound`), matching the paper's own point
    that sampling more candidates can only help."""
    return (
        2 * T * n_co / d * l_lambda_max**2 * (np.log(2 * d) + np.log(1.0 / delta) / n_s)
    )


# ---------------------------------------------------------------------------
# Section 4.4-4.5 -- Proposition 1: L2 scaling and the L-infinity relaxation
# ---------------------------------------------------------------------------


def frobenius_surrogate(lam: np.ndarray, W: np.ndarray, X_reg: np.ndarray) -> float:
    """``R^2(Lambda) = ||W @ Lambda||_F^2 * ||X_reg @ Lambda^-1||_F^2``, the
    Frobenius surrogate Proposition 1 minimizes over positive diagonal
    ``Lambda`` (this module's row/column transpose of the paper's own
    ``||W Lambda||_F^2 ||X_reg Lambda^-1||_F^2``, which is transpose-
    invariant since the Frobenius norm is). Useful on its own as a scalar
    score: for any candidate scaling ``lam`` (e.g. one recovered from an
    existing method), ``frobenius_surrogate(lam, W, X_reg)`` divided by the
    same quantity at :func:`l2_channel_scale`'s optimum gives how much
    slack that candidate leaves in the bound, per Eq. 13.
    """
    w_term = float(np.sum((lam[:, None] * W) ** 2))
    x_term = float(np.sum((X_reg / lam[None, :]) ** 2))
    return w_term * x_term


def l2_channel_scale(
    X_reg: np.ndarray, W: np.ndarray, epsilon: float = 1e-12
) -> np.ndarray:
    """Eq. 14: the channel-wise scaling coefficient minimizing
    :func:`frobenius_surrogate` (a Cauchy-Schwarz equality condition):

        lambda_k = sqrt( ||(X_reg)_{:,k}||_2 / ||(W)_{k,:}||_2 )

    (onnxsim's row/column transpose of the paper's own
    ``sqrt(||(X_reg)_{:,k}||_2 / ||(W)_{:,k}||_2)`` -- channel ``k`` is a
    *column* of ``X_reg`` and a *row* of onnxsim's ``[K, N]``-shaped ``W``,
    matching :mod:`onnxsim.smoothquant`'s own per-channel convention).

    :param X_reg: the regular activation component, ``[T, K]`` (in
            practice, the paper's own Section 4.6 approximates this with
            the full, unsplit activation -- see
            :mod:`onnxsim.quant_error_eval`)
    :param W: the weight, ``[K, N]``
    :param epsilon: floor on the weight-side norm, avoiding a divide-by-zero
            on an all-zero channel
    """
    x_norm = np.linalg.norm(X_reg, axis=0)
    w_norm = np.linalg.norm(W, axis=1)
    return np.sqrt(x_norm / np.maximum(w_norm, epsilon))


def linf_channel_scale(
    X_reg: np.ndarray, W: np.ndarray, epsilon: float = 1e-12
) -> np.ndarray:
    """Eq. 16: the L-infinity relaxation of :func:`l2_channel_scale`,
    obtained by further bounding the Frobenius surrogate with channel-wise
    maxima instead of second moments (Eq. 15):

        lambda_k = sqrt( ||(X_reg)_{:,k}||_inf / ||(W)_{k,:}||_inf )

    This is, at ``alpha = 0.5``, *exactly* :mod:`onnxsim.smoothquant`'s own
    (and :mod:`onnxsim.outlier_suppression`'s own) migration factor
    ``s_j = max(|X_j|)**0.5 / max(|W_j|)**0.5`` -- the paper's Section 4.5
    connects L2 and SmoothQuant-style L-infinity scaling through exactly
    this relaxation chain. See :func:`onnxsim.quant_error_eval.smoothquant_scale`.
    """
    x_inf = np.max(np.abs(X_reg), axis=0)
    w_inf = np.max(np.abs(W), axis=1)
    return np.sqrt(x_inf / np.maximum(w_inf, epsilon))


def proposition1_jreg_bound(
    lam: np.ndarray, W: np.ndarray, X_reg: np.ndarray, d: int, T: int, delta: float
) -> float:
    """Eq. 13: with probability >= ``1 - delta`` over a random-sign ``D``,
    ``J_reg(Lambda, D) <= (2 * log(2*d*T/delta) / d) * ||W @ Lambda||_2^2 * ||X_reg @ Lambda^-1||_F^2``.
    """
    w_lambda_spec2 = float(np.linalg.norm(lam[:, None] * W, ord=2) ** 2)
    x_term = float(np.sum((X_reg / lam[None, :]) ** 2))
    return (2 * np.log(2 * d * T / delta) / d) * w_lambda_spec2 * x_term
