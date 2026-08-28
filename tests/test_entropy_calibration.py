"""Tests for entropy (KL-divergence) calibration in ``onnxsim.calibration``.

``onnxsim.calibrate``'s default ``method="minmax"`` uses each tensor's raw
observed range; ``method="entropy"`` instead searches for a symmetric clip
threshold minimizing the KL divergence between the observed and simulated-
quantized distributions (TensorRT's entropy calibration). These tests check
the threshold search in isolation, then the full ``quantize_static`` pipeline
end-to-end through real ONNX Runtime inference, matching the pattern
``test_static_quantize_matmul.py`` uses for the default method.
"""

import collections

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.calibration import _entropy_threshold, _kl_divergence, _smooth_distribution

# A bare ``import onnxruntime`` would fail collection (not skip the test) on
# platforms onnxruntime doesn't ship wheels for (e.g. s390x).
ort = pytest.importorskip("onnxruntime")


def _model(body, initializer=(), opset=13, ir_version=10):
    # Pin a low IR version by default so the model loads under the older
    # onnxruntime bundled with some CI wheels (which cap at IR version 11).
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


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _op_counts(model):
    return collections.Counter(n.op_type for n in model.graph.node)


def _assert_close(float_outputs, quant_outputs, threshold=0.1):
    for f, q in zip(float_outputs, quant_outputs):
        f = np.asarray(f, dtype=np.float64).ravel()
        q = np.asarray(q, dtype=np.float64).ravel()
        assert np.all(np.isfinite(q))
        rel_l2 = np.linalg.norm(f - q) / max(np.linalg.norm(f), 1e-6)
        assert rel_l2 < threshold, f"relative L2 error too large: {rel_l2:.4f}"


def test_smooth_distribution_removes_zeros():
    p = np.array([10.0, 0.0, 5.0, 0.0, 5.0])
    smoothed = _smooth_distribution(p)
    assert smoothed is not None
    assert np.all(smoothed > 0)
    # Total mass is conserved: smoothing only redistributes it.
    assert np.isclose(smoothed.sum(), p.sum())


def test_smooth_distribution_returns_none_when_infeasible():
    # A single tiny non-zero bin can't give up enough mass to fill many zero
    # bins with eps and stay non-negative.
    p = np.array([1e-8] + [0.0] * 1000)
    assert _smooth_distribution(p, eps=1e-4) is None


def test_kl_divergence_zero_for_identical_distributions():
    p = np.array([0.2, 0.3, 0.5])
    assert _kl_divergence(p, p) == pytest.approx(0.0, abs=1e-12)


def test_kl_divergence_positive_for_different_distributions():
    p = np.array([0.9, 0.1])
    q = np.array([0.1, 0.9])
    assert _kl_divergence(p, q) > 0.0


def test_entropy_threshold_clips_heavy_tailed_outliers():
    # Mostly small values (the real signal) plus a handful of huge-magnitude
    # outliers -- entropy calibration should find a threshold that keeps the
    # bulk of the distribution intact rather than being stretched out by the
    # rare outliers, unlike a plain max.
    rng = np.random.default_rng(0)
    core = rng.normal(0, 1.0, size=20000)
    outliers = rng.normal(0, 1.0, size=20) * 50
    values = np.concatenate([core, outliers])

    threshold = _entropy_threshold(values, num_bins=2048, num_quantized_bins=128)
    raw_max = np.abs(values).max()
    assert threshold < raw_max
    # The core signal (>99.9% of the data) should survive essentially intact.
    assert threshold > np.percentile(np.abs(core), 99.0)


def test_entropy_threshold_handles_sparse_data():
    # Too little data to build a meaningful histogram: falls back to the
    # plain max rather than erroring or returning something degenerate.
    values = np.array([1.0, -2.0, 3.0])
    threshold = _entropy_threshold(values, num_bins=2048, num_quantized_bins=128)
    assert threshold == pytest.approx(3.0)


def test_entropy_threshold_zero_values():
    assert _entropy_threshold(np.zeros(10000)) == 0.0


def test_calibrate_entropy_tighter_than_minmax_with_outliers():
    # A calibration set where one batch has a rare, huge-magnitude spike:
    # minmax's range has to stretch to cover it; entropy calibration should
    # clip it instead, giving a visibly tighter (min, max).
    weight = _f32(np.random.default_rng(0).standard_normal((16, 8)), "W")
    model = _model(
        """
        g (float[4,16] X) => (float[4,8] Y)
        {
          Y = MatMul(X, W)
        }
        """,
        [weight],
    )

    rng = np.random.default_rng(1)
    calibration_data = [
        {"X": rng.standard_normal((4, 16)).astype(np.float32)} for _ in range(63)
    ]
    spike = rng.standard_normal((4, 16)).astype(np.float32)
    spike[0, 0] = 500.0
    calibration_data.append({"X": spike})

    minmax_ranges = onnxsim.calibrate(model, calibration_data, method="minmax")
    entropy_ranges = onnxsim.calibrate(model, calibration_data, method="entropy")

    minmax_lo, minmax_hi = minmax_ranges["X"]
    entropy_lo, entropy_hi = entropy_ranges["X"]
    assert minmax_hi == pytest.approx(500.0)
    assert entropy_hi < minmax_hi
    # Still wide enough to cover the actual (non-spike) signal.
    assert entropy_hi > 2.0


def test_quantize_static_entropy_method_end_to_end():
    rng = np.random.default_rng(2)
    K, N = 32, 16
    weight = _f32(rng.standard_normal((K, N)) * 0.5, "W")
    model = _model(
        f"""
        g (float[4,{K}] X) => (float[4,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [weight],
    )

    calibration_data = [
        {"X": rng.standard_normal((4, K)).astype(np.float32)} for _ in range(128)
    ]
    quant = onnxsim.quantize_static(
        model, calibration_data=calibration_data, method="entropy"
    )
    onnx.checker.check_model(quant)
    ops = _op_counts(quant)
    assert ops["MatMul"] == 1
    assert ops["QuantizeLinear"] == 1
    assert ops["DequantizeLinear"] == 2

    x = rng.standard_normal((4, K)).astype(np.float32)
    _assert_close(_run(model, {"X": x}), _run(quant, {"X": x}))


def test_calibrate_unknown_method_raises():
    weight = _f32(np.random.randn(8, 4).astype(np.float32), "W")
    model = _model(
        """
        g (float[2,8] X) => (float[2,4] Y)
        {
          Y = MatMul(X, W)
        }
        """,
        [weight],
    )
    data = [{"X": np.zeros((2, 8), dtype=np.float32)}]
    with pytest.raises(ValueError):
        onnxsim.calibrate(model, data, method="bogus")
