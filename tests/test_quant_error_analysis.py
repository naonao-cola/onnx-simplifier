"""Tests for :mod:`onnxsim.quant_error_analysis` -- the ported math from
Narita & Sato, 2026 (arXiv:2609.21450). These exercise the paper's own
claims directly on synthetic numpy arrays (Theorem 1's exact identity,
Theorem 2's bound orderings, Proposition 1's optimality), independent of
any ONNX graph -- see ``tests/test_quant_error_eval.py`` for the
model-level harness built on top of this module.
"""

import numpy as np
import pytest

from onnxsim.quant_error_analysis import (
    decompose_local_error,
    detect_persistent_outlier_channels,
    fixed_rotation_co_bound,
    frobenius_surrogate,
    l2_channel_scale,
    linf_channel_scale,
    local_reconstruction_error,
    normalized_hadamard_matrix,
    persistent_co_split,
    randomized_rotation_co_bound,
    residual_upper_bound,
    sampled_rotation_co_bound,
)


def _random_problem(T=64, K=16, N=8, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((T, K))
    A = rng.standard_normal((T, K)) * 0.1
    Z_tilde = Z + A
    V = rng.standard_normal((K, N))
    V_tilde = V + rng.standard_normal((K, N)) * 0.2
    return Z, Z_tilde, V, V_tilde


def test_decomposition_identity_exact():
    Z, Z_tilde, V, V_tilde = _random_problem()
    d = decompose_local_error(Z, Z_tilde, V, V_tilde)
    assert d.j_total == pytest.approx(d.j_agwc + d.j_residual, rel=1e-8, abs=1e-10)
    assert d.j_total == pytest.approx(
        local_reconstruction_error(Z, Z_tilde, V, V_tilde), rel=1e-10
    )


def test_no_activation_error_collapses_to_gptq_case():
    # A = 0: Section 4.2's "shared-input reconstruction" special case --
    # v_star == V and the residual is exactly zero.
    Z, _Z_tilde, V, V_tilde = _random_problem()
    d = decompose_local_error(Z, Z, V, V_tilde)
    assert d.j_residual == pytest.approx(0.0, abs=1e-8)
    assert np.allclose(d.v_star, V, atol=1e-8)
    assert d.j_total == pytest.approx(d.j_agwc, rel=1e-8)


def test_perfect_weight_reconstruction_leaves_only_residual():
    Z, Z_tilde, V, _V_tilde = _random_problem()
    d = decompose_local_error(Z, Z_tilde, V, V.copy())
    # Vtilde == V is not generally v_star when A != 0, so j_agwc need not be
    # zero here -- but the decomposition must still hold exactly.
    assert d.j_total == pytest.approx(d.j_agwc + d.j_residual, rel=1e-8, abs=1e-10)
    # Feeding v_star itself as Vtilde should drive j_agwc to (numerically) 0.
    d_star = decompose_local_error(Z, Z_tilde, V, d.v_star)
    assert d_star.j_agwc == pytest.approx(0.0, abs=1e-8)
    assert d_star.j_residual == pytest.approx(d.j_residual, rel=1e-6)


def test_residual_independent_of_weight_choice():
    # Theorem 1's central claim: for a fixed transformation, weight
    # optimization changes j_agwc but never j_residual.
    Z, Z_tilde, V, _ = _random_problem()
    rng = np.random.default_rng(1)
    V_tilde_a = V + rng.standard_normal(V.shape) * 0.3
    V_tilde_b = V + rng.standard_normal(V.shape) * 5.0
    d_a = decompose_local_error(Z, Z_tilde, V, V_tilde_a)
    d_b = decompose_local_error(Z, Z_tilde, V, V_tilde_b)
    assert d_a.j_residual == pytest.approx(d_b.j_residual, rel=1e-8, abs=1e-10)
    assert d_a.j_agwc != pytest.approx(d_b.j_agwc, rel=1e-3)


def test_persistent_outlier_detection():
    rng = np.random.default_rng(2)
    T, K = 200, 20
    X = rng.standard_normal((T, K)) * 0.1
    outlier_channels = [3, 11]
    for k in outlier_channels:
        X[:, k] = 10.0 + rng.standard_normal(T) * 0.1
    mask = detect_persistent_outlier_channels(X, ratio_threshold=4.0)
    assert np.where(mask)[0].tolist() == outlier_channels

    X_co, X_reg, levels = persistent_co_split(X, mask)
    assert np.allclose(X_co + X_reg, X)
    assert np.all(levels[~mask] == 0.0)
    for k in outlier_channels:
        assert levels[k] == pytest.approx(np.median(X[:, k]))


def test_detect_no_outliers_on_uniform_noise():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((100, 10))
    mask = detect_persistent_outlier_channels(X, ratio_threshold=4.0)
    assert not np.any(mask)


def test_l2_scaling_minimizes_frobenius_surrogate():
    rng = np.random.default_rng(4)
    T, K, N = 128, 12, 6
    X_reg = rng.standard_normal((T, K)) * rng.uniform(0.1, 10.0, size=K)
    W = rng.standard_normal((K, N)) * rng.uniform(0.1, 10.0, size=(K, 1))
    lam_opt = l2_channel_scale(X_reg, W)
    r2_opt = frobenius_surrogate(lam_opt, W, X_reg)
    for trial in range(20):
        perturb = rng.uniform(0.2, 5.0, size=K)
        r2_other = frobenius_surrogate(lam_opt * perturb, W, X_reg)
        assert r2_other >= r2_opt - 1e-9 * max(1.0, abs(r2_opt))


def test_l2_scaling_scale_invariant_optimum():
    # A global rescaling of lambda changes R^2 (it is not scale-invariant
    # itself -- lambda_k * c for all k scales the weight term by c^2 and the
    # activation term by 1/c^2, leaving R^2 unchanged); verify that specific
    # invariance directly, since it's what "the equality condition is unique
    # up to a global positive constant" (Eq. 14) means in practice.
    rng = np.random.default_rng(5)
    X_reg = rng.standard_normal((50, 8))
    W = rng.standard_normal((8, 4))
    lam = l2_channel_scale(X_reg, W)
    r2 = frobenius_surrogate(lam, W, X_reg)
    r2_scaled = frobenius_surrogate(lam * 3.7, W, X_reg)
    assert r2_scaled == pytest.approx(r2, rel=1e-8)


def test_linf_relaxation_matches_smoothquant_default_formula():
    rng = np.random.default_rng(6)
    X = rng.standard_normal((40, 10))
    W = rng.standard_normal((10, 5))
    lam = linf_channel_scale(X, W)
    x_inf = np.max(np.abs(X), axis=0)
    w_inf = np.max(np.abs(W), axis=1)
    expected = np.sqrt(x_inf / w_inf)
    assert np.allclose(lam, expected)


def test_theorem2_bounds_ordering():
    T, d, delta = 100, 64, 0.05
    n_co, l_max = 20, 3.0  # large enough n_co that interference dominates
    fixed = fixed_rotation_co_bound(T, n_co, d, l_max)
    randomized = randomized_rotation_co_bound(T, n_co, d, l_max, delta)
    sampled = sampled_rotation_co_bound(T, n_co, d, l_max, delta, n_s=10)
    # The paper's own point: random signs reduce N_co^2 interference to
    # N_co log(d) -- for enough persistent-outlier channels, fixed exceeds
    # randomized.
    assert fixed > randomized
    # Sampling more candidates can only tighten the bound further.
    assert sampled <= randomized


def test_sampled_bound_monotone_in_num_samples():
    T, d, delta, n_co, l_max = 50, 32, 0.05, 5, 1.0
    b1 = sampled_rotation_co_bound(T, n_co, d, l_max, delta, n_s=1)
    b4 = sampled_rotation_co_bound(T, n_co, d, l_max, delta, n_s=4)
    b16 = sampled_rotation_co_bound(T, n_co, d, l_max, delta, n_s=16)
    assert b1 >= b4 >= b16
    assert b1 == pytest.approx(randomized_rotation_co_bound(T, n_co, d, l_max, delta))


def test_normalized_hadamard_matrix_orthogonal():
    for d in (1, 2, 4, 16):
        H = normalized_hadamard_matrix(d)
        assert H.shape == (d, d)
        assert np.allclose(H @ H, np.eye(d), atol=1e-10)
        assert np.allclose(H, H.T)


def test_normalized_hadamard_matrix_rejects_non_power_of_two():
    with pytest.raises(ValueError):
        normalized_hadamard_matrix(6)


def test_residual_upper_bound_dominates_actual_residual():
    # Eq. 5: J_AGWC + the bound's second term upper-bounds J. Build a
    # concrete P = Lambda^-1 D H^T, derive Z = X @ P, and check the exact
    # residual (Theorem 1) never exceeds the bound built from the same
    # Lambda/D/H/W/X split.
    rng = np.random.default_rng(7)
    T, K, N = 300, 8, 4
    q_a = 7.0  # INT4 symmetric max code

    X = rng.standard_normal((T, K))
    outliers = [1, 5]
    for k in outliers:
        X[:, k] += 8.0
    W = rng.standard_normal((K, N))

    lam = np.ones(K)
    signs = rng.choice([-1.0, 1.0], size=K)
    H = normalized_hadamard_matrix(K)

    P = np.diag(1.0 / lam) @ np.diag(signs) @ H.T
    Z = X @ P
    V = np.linalg.inv(P) @ W  # V = P^-1 @ W so that X @ W == Z @ V

    # No-clipping dynamic per-token symmetric activation quantization
    # (Section 3.1, Eq. 1).
    row_max = np.max(np.abs(Z), axis=1, keepdims=True)
    delta_t = row_max / q_a
    Z_tilde = delta_t * np.round(Z / delta_t)

    V_tilde = V + rng.standard_normal(V.shape) * 0.05  # some weight quantizer

    d = decompose_local_error(Z, Z_tilde, V, V_tilde)

    mask = detect_persistent_outlier_channels(X, ratio_threshold=4.0)
    assert np.where(mask)[0].tolist() == outliers
    X_co, X_reg, _levels = persistent_co_split(X, mask)

    _j_co, _j_reg, bound = residual_upper_bound(W, lam, signs, H, X_co, X_reg, q_a)
    assert d.j_residual <= bound + 1e-9
