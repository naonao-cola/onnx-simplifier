"""Tests for :mod:`onnxsim.quant_error_eval` -- the harness that scores
onnxsim's own existing quantization passes (SmoothQuant-style scaling,
QuaRot/SpinQuant-style rotation) against arXiv:2609.21450's analysis. Model
construction follows this repo's own convention (see CLAUDE.md): built via
``onnx.parser`` rather than ``onnx.helper.make_node`` chains.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

from onnxsim.quant_error_eval import (
    evaluate_channel_scaling,
    evaluate_model,
    evaluate_rotation_co_control,
    quarot_style_rotation,
    smoothquant_scale,
    spinquant_style_rotation,
)

ort = pytest.importorskip("onnxruntime")


def _f32(array, name):
    return onnx.numpy_helper.from_array(np.asarray(array, dtype=np.float32), name)


def _model(body, initializer=(), opset=13, ir_version=10):
    model = parser.parse_model(
        f"""
        <
          ir_version: {ir_version},
          opset_import: ["": {opset}]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model


def _matmul_model(K=16, N=8, seed=0):
    rng = np.random.default_rng(seed)
    weight = (rng.standard_normal((K, N)) * 0.5).astype(np.float32)
    return _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f32(weight, "W")],
    )


def _calibration_with_outliers(K=16, num_samples=64, seed=1, outlier_channels=(2, 9)):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((num_samples, K)).astype(np.float32)
    for k in outlier_channels:
        x[:, k] += 12.0
    return [{"X": x}]


def test_evaluate_model_returns_one_report_per_layer():
    model = _matmul_model(K=16, N=8)
    calib = _calibration_with_outliers(K=16)
    reports = evaluate_model(model, calibration_data=calib, seed=0)
    assert len(reports) == 1
    layer = reports[0]
    assert layer.weight_name == "W"
    assert layer.k == 16 and layer.n == 8
    assert layer.num_tokens == 64

    for field in (
        layer.smoothquant.r2,
        layer.smoothquant.r2_l2_optimal,
        layer.smoothquant.r2_ratio_to_l2_optimal,
        layer.quarot.j_co_actual,
        layer.spinquant.j_co_actual,
    ):
        assert np.isfinite(field)


def test_smoothquant_ratio_to_l2_optimal_is_at_least_one():
    # The L2 rule is derived to *minimize* the Frobenius surrogate (Prop.
    # 1); SmoothQuant's own L-infinity relaxation of it can only match or
    # exceed that minimum on real (non-degenerate) calibration statistics.
    model = _matmul_model(K=16, N=8, seed=2)
    calib = _calibration_with_outliers(K=16, seed=3)
    reports = evaluate_model(model, calibration_data=calib, seed=0)
    assert reports[0].smoothquant.r2_ratio_to_l2_optimal >= 1.0 - 1e-6
    assert reports[0].smoothquant.r2_ratio_to_linf_relaxation == pytest.approx(
        1.0, rel=1e-6
    )


def test_smoothquant_scale_matches_apply_smoothquant_formula():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((32, 6)).astype(np.float64)
    W = rng.standard_normal((6, 3)).astype(np.float64)
    lam = smoothquant_scale(X, W, alpha=0.5, epsilon=1e-5)
    x_inf = np.maximum(np.max(np.abs(X), axis=0), 1e-5)
    w_inf = np.maximum(np.max(np.abs(W), axis=1), 1e-5)
    expected = np.sqrt(x_inf) / np.sqrt(w_inf)
    assert np.allclose(lam, expected)


def test_evaluate_channel_scaling_worse_candidate_has_larger_ratio():
    rng = np.random.default_rng(5)
    X = rng.standard_normal((50, 8))
    W = rng.standard_normal((8, 4))
    good = evaluate_channel_scaling("good", np.ones(8), X, W)
    bad_lam = np.ones(8)
    bad_lam[0] = 1e-3  # a badly chosen scale on one channel
    bad = evaluate_channel_scaling("bad", bad_lam, X, W)
    assert bad.r2_ratio_to_l2_optimal > good.r2_ratio_to_l2_optimal


def test_rotation_co_control_no_outliers_is_zero():
    rng = np.random.default_rng(6)
    X = rng.standard_normal((40, 8))
    rotation = np.eye(8)
    result = evaluate_rotation_co_control("identity", rotation, X)
    assert result.n_co == 0
    assert result.j_co_actual == 0.0


def test_rotation_co_control_detects_outliers_and_reports_bounds():
    rng = np.random.default_rng(7)
    K = 32
    X = rng.standard_normal((100, K)) * 0.1
    for k in (0, 5, 17):
        X[:, k] += 10.0
    rotation = quarot_style_rotation(K, seed=0)
    result = evaluate_rotation_co_control("quarot", rotation, X)
    assert result.n_co == 3
    assert result.j_co_actual > 0.0
    assert result.theorem2_randomized_bound > 0.0
    assert result.theorem2_sampled_bound <= result.theorem2_randomized_bound


def test_spinquant_style_rotation_is_orthogonal():
    rng = np.random.default_rng(8)
    X = rng.standard_normal((50, 10))
    U = spinquant_style_rotation(X)
    assert U.shape == (10, 10)
    assert np.allclose(U.T @ U, np.eye(10), atol=1e-8)


def test_evaluate_model_no_matching_layer_returns_empty():
    model = _model(
        """
        g (float[batch,4] X) => (float[batch,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    reports = evaluate_model(model, seed=0)
    assert reports == []
