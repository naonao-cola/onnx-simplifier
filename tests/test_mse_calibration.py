"""Tests for MSE (direct reconstruction-error) calibration in
``onnxsim.calibration`` -- ``method="mse"``. Unlike ``"entropy"`` (which
minimizes KL divergence between the observed and simulated-quantized
distributions via a histogram proxy), ``"mse"`` searches for the symmetric
clip threshold minimizing quantized reconstruction error measured directly
against the observed values -- the technique Outlier Suppression's own
"Token-Wise Clipping" uses, and the gap ``onnxsim.outlier_suppression``'s
own docstring explicitly leaves for this module. These tests check the
threshold search in isolation, then the full ``quantize_static`` pipeline
end-to-end through real ONNX Runtime inference, matching the pattern
``test_entropy_calibration.py`` uses for its own method.
"""

import collections

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.calibration import _mse_threshold

# A bare ``import onnxruntime`` would fail collection (not skip the test) on
# platforms onnxruntime doesn't ship wheels for (e.g. s390x).
ort = pytest.importorskip("onnxruntime")


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


def _quantize_mse(v, threshold):
    scale = threshold / 127.0
    return np.clip(np.round(v / scale), -127, 127) * scale


def test_mse_threshold_clips_heavy_tailed_outliers():
    rng = np.random.default_rng(0)
    core = rng.normal(0, 1.0, size=20000)
    outliers = rng.normal(0, 1.0, size=20) * 50
    values = np.concatenate([core, outliers])

    threshold = _mse_threshold(values, num_candidates=100, min_coverage=0.5)
    raw_max = np.abs(values).max()
    assert threshold < raw_max
    # The core signal should survive essentially intact.
    assert threshold > np.percentile(np.abs(core), 99.0)


def test_mse_threshold_actually_minimizes_mse_among_candidates():
    # The threshold the search returns should have reconstruction error no
    # worse than every other candidate it considered, including the raw max
    # (which is always one of the endpoints of the search grid).
    rng = np.random.default_rng(1)
    core = rng.normal(0, 1.0, size=5000)
    outliers = rng.normal(0, 1.0, size=5) * 80
    values = np.concatenate([core, outliers])

    num_candidates = 50
    threshold = _mse_threshold(values, num_candidates=num_candidates, min_coverage=0.5)
    best_mse = np.mean((values - _quantize_mse(values, threshold)) ** 2)

    abs_max = np.abs(values).max()
    floor = max(np.percentile(np.abs(values), 50.0), abs_max * 1e-6)
    for t in np.linspace(floor, abs_max, num_candidates):
        mse = np.mean((values - _quantize_mse(values, t)) ** 2)
        assert best_mse <= mse + 1e-9


def test_mse_threshold_handles_sparse_data():
    # Too few points that the floor/max collapse: falls back to the plain
    # max rather than erroring or returning something degenerate.
    values = np.array([1.0, -2.0, 3.0])
    threshold = _mse_threshold(values, num_candidates=100, min_coverage=0.5)
    assert threshold == pytest.approx(3.0)


def test_mse_threshold_zero_values():
    assert _mse_threshold(np.zeros(10000)) == 0.0


def test_calibrate_mse_tighter_than_minmax_with_outliers():
    # Unlike entropy calibration (which treats outliers as cheap-to-sacrifice
    # low-probability-mass histogram bins and so clips even a single isolated
    # spike), MSE calibration weighs candidates by *squared* magnitude -- a
    # single extreme point contributes too much squared error to ever be
    # worth clipping. So, matching
    # ``test_mse_threshold_clips_heavy_tailed_outliers`` above, this uses a
    # heavier tail (a handful of outlier batches, not just one) at a more
    # moderate magnitude for MSE to show its own advantage on.
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
        {"X": rng.standard_normal((4, 16)).astype(np.float32)} for _ in range(60)
    ]
    for _ in range(4):
        spike = rng.standard_normal((4, 16)).astype(np.float32)
        spike[0, 0] = 50.0
        calibration_data.append({"X": spike})

    minmax_ranges = onnxsim.calibrate(model, calibration_data, method="minmax")
    mse_ranges = onnxsim.calibrate(model, calibration_data, method="mse")

    minmax_lo, minmax_hi = minmax_ranges["X"]
    mse_lo, mse_hi = mse_ranges["X"]
    assert minmax_hi == pytest.approx(50.0)
    assert mse_hi < minmax_hi
    # Still wide enough to cover the actual (non-spike) signal.
    assert mse_hi > 2.0


def test_quantize_static_mse_method_end_to_end():
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
        model, calibration_data=calibration_data, method="mse"
    )
    onnx.checker.check_model(quant)
    ops = _op_counts(quant)
    assert ops["MatMul"] == 1
    assert ops["QuantizeLinear"] == 1
    assert ops["DequantizeLinear"] == 2

    x = rng.standard_normal((4, K)).astype(np.float32)
    _assert_close(_run(model, {"X": x}), _run(quant, {"X": x}))
