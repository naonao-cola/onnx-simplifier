"""Tests for ``onnxsim.apply_easyquant`` (EasyQuant, see
``onnxsim/easyquant.py``) -- W8A8 quantization with per-output-channel
weight scale and per-tensor activation scale chosen by a coordinate-descent
search against real calibration activations, rather than a standalone
min/max threshold.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim

ort = pytest.importorskip("onnxruntime")


def _model(body, initializer=(), opset=13, ir_version=8):
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


def _matmul_model(w, K, N, batch="batch"):
    return _model(
        f"""
        g (float[{batch},{K}] X) => (float[{batch},{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(w, "W")],
    )


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-9)


def _outlier_calibration(K, num_samples, batch=32, seed=0):
    # Deliberately gives a handful of channels much larger magnitude than
    # the rest -- the scenario where a naive single-shot max-abs scale
    # (dominated by the outlier channels) wastes most of its resolution,
    # and a per-channel searched scale has real room to do better. Which
    # channels are outliers is a fixed property of "this layer's own
    # activation distribution" (derived from K alone, not the per-call
    # seed) -- exactly what makes calibration data representative of
    # held-out data in the first place; only the per-batch *values* are
    # randomized per seed.
    outlier_rng = np.random.default_rng(1234 + K)
    outlier_channels = outlier_rng.choice(K, size=max(1, K // 8), replace=False)
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((num_samples, batch, K)).astype(np.float32)
    x[:, :, outlier_channels] *= 15.0
    return [{"X": x[i]} for i in range(num_samples)]


def _naive_w8a8_round_trip(model, x):
    # A plain, unsearched W8A8 baseline: one max-abs scale per output
    # channel for the weight, one max-abs scale for the whole activation
    # tensor -- exactly EasyQuant's own starting point, before any search.
    w = onnx.numpy_helper.to_array(model.graph.initializer[0]).astype(np.float64)
    w_scale = np.maximum(np.max(np.abs(w), axis=0), 1e-12) / 127.0  # [N], W is [K,N]
    a_scale = max(float(np.max(np.abs(x))), 1e-12) / 127.0
    w_q = np.clip(np.round(w / w_scale), -127, 127) * w_scale
    x_q = np.clip(np.round(x / a_scale), -127, 127) * a_scale
    return x_q @ w_q


def test_easyquant_replaces_weight_and_inserts_activation_round_trip():
    rng = np.random.default_rng(0)
    K, N = 16, 8
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(w, K, N)
    calibration_data = _outlier_calibration(K, num_samples=8, seed=1)

    q = onnxsim.apply_easyquant(model, calibration_data=calibration_data)
    onnx.checker.check_model(q)
    assert any(n.op_type == "Round" for n in q.graph.node)
    assert any(n.op_type == "Clip" for n in q.graph.node)
    # The MatMul's own weight input was rewired to a new (quantize-
    # dequantize round-tripped) initializer, not the original "W".
    matmul_node = next(n for n in q.graph.node if n.op_type == "MatMul")
    assert matmul_node.input[1] != "W"
    new_w = onnx.numpy_helper.to_array(
        next(t for t in q.graph.initializer if t.name == matmul_node.input[1])
    )
    assert not np.array_equal(new_w, w)


def test_easyquant_search_beats_naive_max_abs_scale():
    # The core empirical claim: searching scales against the actual layer
    # output (rather than each tensor's own max-abs alone) should reduce
    # reconstruction error on a scenario deliberately shaped for it
    # (outlier channels distorting the naive scale).
    rng = np.random.default_rng(2)
    K, N = 32, 12
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(w, K, N)

    fit_data = _outlier_calibration(K, num_samples=16, batch=64, seed=3)
    q = onnxsim.apply_easyquant(
        model, calibration_data=fit_data, num_iterations=3, num_candidates=25
    )
    onnx.checker.check_model(q)

    held_out = _outlier_calibration(K, num_samples=1, batch=512, seed=4)[0]["X"]
    (float_y,) = _run(model, {"X": held_out})
    (easyquant_y,) = _run(q, {"X": held_out})
    naive_y = _naive_w8a8_round_trip(model, held_out.astype(np.float64))

    easyquant_err = _rel_l2(float_y, easyquant_y)
    naive_err = _rel_l2(float_y, naive_y)
    assert easyquant_err < naive_err


def test_easyquant_output_stays_finite_and_reasonably_close():
    rng = np.random.default_rng(5)
    K, N = 20, 10
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.3
    model = _matmul_model(w, K, N)
    calibration_data = _outlier_calibration(K, num_samples=8, batch=16, seed=6)

    q = onnxsim.apply_easyquant(model, calibration_data=calibration_data)
    onnx.checker.check_model(q)

    x = calibration_data[0]["X"]
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.5


def test_easyquant_noop_when_no_matmul_gemm_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_easyquant(
        model, calibration_data=[{"X": np.zeros((4, 4), dtype=np.float32)}]
    )
    assert result.SerializeToString() == model.SerializeToString()


def test_easyquant_skips_non_2d_weight():
    rng = np.random.default_rng(7)
    w = rng.standard_normal((2, 3, 3, 3)).astype(np.float32)
    model = _model(
        """
        g (float[1,2,8,8] X) => (float[1,3,6,6] Y)
        {
          Y = Conv(X, W)
        }
        """,
        [_f32(w, "W")],
    )
    result = onnxsim.apply_easyquant(
        model,
        calibration_data=[{"X": rng.standard_normal((1, 2, 8, 8)).astype(np.float32)}],
    )
    assert result.SerializeToString() == model.SerializeToString()
