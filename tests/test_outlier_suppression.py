"""Tests for ``onnxsim.apply_outlier_suppression`` (Outlier Suppression's
"Gamma Migration", see ``onnxsim/outlier_suppression.py``) -- folds a
per-channel migration scale directly into a ``LayerNormalization``'s own
``scale``/``bias`` parameters (dividing both), compensated by multiplying
every downstream MatMul/Gemm consumer's weight by the same scale, with zero
new nodes inserted -- only when every consumer of that LayerNormalization's
output is one of those compensated layers.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim

ort = pytest.importorskip("onnxruntime")


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _model(body, initializer=(), opset=17, ir_version=8):
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


def _run(model, feeds, output_names=None):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    names = output_names or [o.name for o in sess.get_outputs()]
    return dict(zip(names, sess.run(names, feeds)))


def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-6)


def _with_extra_output(model, name):
    out = onnx.ModelProto()
    out.CopyFrom(model)
    existing = {o.name for o in out.graph.output}
    if name not in existing:
        out.graph.output.append(onnx.ValueInfoProto(name=name))
    return out


def _ln_matmul_model(K=32, N=8, outlier_channels=(3,), gamma_outlier=20.0, seed=0):
    rng = np.random.default_rng(seed)
    gamma = np.ones(K, dtype=np.float32)
    for c in outlier_channels:
        gamma[c] = gamma_outlier  # LayerNorm's own gamma amplifies this channel
    beta = rng.standard_normal(K).astype(np.float32) * 0.1
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Ln = LayerNormalization<axis = -1>(X, Gamma, Beta)
          Y = MatMul(Ln, W)
        }}
        """,
        [_f32(gamma, "Gamma"), _f32(beta, "Beta"), _f32(weight, "W")],
    )
    return model


def _calibration(K=32, num_samples=64, seed=1):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((num_samples, K)).astype(np.float32)


def test_outlier_suppression_output_matches_float_almost_exactly():
    model = _ln_matmul_model(K=32, N=8, seed=0)
    x = _calibration(K=32, num_samples=64, seed=1)
    calibration_data = [{"X": x}]

    osup_model = onnxsim.apply_outlier_suppression(
        model, calibration_data=calibration_data
    )
    onnx.checker.check_model(osup_model)

    # No new nodes -- same node count/types, only initializer values change.
    assert [n.op_type for n in osup_model.graph.node] == [
        n.op_type for n in model.graph.node
    ]

    float_out = _run(model, {"X": x})
    osup_out = _run(osup_model, {"X": x}, output_names=["Y"])
    assert np.all(np.isfinite(osup_out["Y"]))
    assert _rel_l2(float_out["Y"], osup_out["Y"]) < 1e-4


def test_outlier_suppression_migrates_gamma_beta_and_reduces_channel_spread():
    K = 32
    model = _ln_matmul_model(K=K, N=8, outlier_channels=(3, 7), seed=2)
    x = _calibration(K=K, num_samples=64, seed=3)
    calibration_data = [{"X": x}]

    osup_model = onnxsim.apply_outlier_suppression(
        model, calibration_data=calibration_data
    )
    onnx.checker.check_model(osup_model)

    gamma_before = onnx.numpy_helper.to_array(model.graph.initializer[0])
    beta_before = onnx.numpy_helper.to_array(model.graph.initializer[1])
    gamma_after = onnx.numpy_helper.to_array(osup_model.graph.initializer[0])
    beta_after = onnx.numpy_helper.to_array(osup_model.graph.initializer[1])
    assert not np.allclose(gamma_before, gamma_after)
    assert not np.allclose(beta_before, beta_after)
    # gamma/beta were both divided by the exact same per-channel scale.
    ratio_gamma = gamma_before.astype(np.float64) / gamma_after.astype(np.float64)
    ratio_beta = beta_before.astype(np.float64) / beta_after.astype(np.float64)
    np.testing.assert_allclose(ratio_gamma, ratio_beta, rtol=1e-4)

    ln_node = next(
        n for n in osup_model.graph.node if n.op_type == "LayerNormalization"
    )
    probe_model = _with_extra_output(osup_model, ln_node.output[0])
    result = _run(probe_model, {"X": x}, output_names=[ln_node.output[0]])
    migrated = result[ln_node.output[0]]

    float_ln = next(n for n in model.graph.node if n.op_type == "LayerNormalization")
    float_probe = _with_extra_output(model, float_ln.output[0])
    float_result = _run(float_probe, {"X": x}, output_names=[float_ln.output[0]])
    original = float_result[float_ln.output[0]]

    original_spread = np.abs(original).max(axis=0)
    migrated_spread = np.abs(migrated).max(axis=0)
    assert (migrated_spread.max() / migrated_spread.min()) < (
        original_spread.max() / original_spread.min()
    )


def test_outlier_suppression_handles_multiple_consumers_exactly():
    K, N = 32, 8
    rng = np.random.default_rng(4)
    gamma = np.ones(K, dtype=np.float32)
    gamma[5] = 15.0
    beta = rng.standard_normal(K).astype(np.float32) * 0.1
    w_q = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    w_k = rng.standard_normal((K, N)).astype(np.float32) * 0.3
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Q, float[batch,{N}] Kk)
        {{
          Ln = LayerNormalization<axis = -1>(X, Gamma, Beta)
          Q = MatMul(Ln, Wq)
          Kk = MatMul(Ln, Wk)
        }}
        """,
        [_f32(gamma, "Gamma"), _f32(beta, "Beta"), _f32(w_q, "Wq"), _f32(w_k, "Wk")],
    )
    x = _calibration(K=K, num_samples=64, seed=5)
    calibration_data = [{"X": x}]

    osup_model = onnxsim.apply_outlier_suppression(
        model, calibration_data=calibration_data
    )
    onnx.checker.check_model(osup_model)

    float_out = _run(model, {"X": x})
    osup_out = _run(osup_model, {"X": x})
    assert _rel_l2(float_out["Q"], osup_out["Q"]) < 1e-4
    assert _rel_l2(float_out["Kk"], osup_out["Kk"]) < 1e-4


def test_outlier_suppression_declines_when_non_matmul_consumer_present():
    K, N = 16, 8
    rng = np.random.default_rng(6)
    gamma = np.ones(K, dtype=np.float32)
    beta = np.zeros(K, dtype=np.float32)
    weight = rng.standard_normal((K, N)).astype(np.float32)
    bias = rng.standard_normal(K).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y, float[batch,{K}] Z)
        {{
          Ln = LayerNormalization<axis = -1>(X, Gamma, Beta)
          Y = MatMul(Ln, W)
          Z = Add(Ln, Bias)
        }}
        """,
        [
            _f32(gamma, "Gamma"),
            _f32(beta, "Beta"),
            _f32(weight, "W"),
            _f32(bias, "Bias"),
        ],
    )
    x = _calibration(K=K, num_samples=16, seed=7)
    result = onnxsim.apply_outlier_suppression(model, calibration_data=[{"X": x}])
    assert result.SerializeToString() == model.SerializeToString()


def test_outlier_suppression_declines_when_layernorm_output_is_graph_output():
    K, N = 16, 8
    rng = np.random.default_rng(8)
    gamma = np.ones(K, dtype=np.float32)
    beta = np.zeros(K, dtype=np.float32)
    weight = rng.standard_normal((K, N)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y, float[batch,{K}] Ln)
        {{
          Ln = LayerNormalization<axis = -1>(X, Gamma, Beta)
          Y = MatMul(Ln, W)
        }}
        """,
        [_f32(gamma, "Gamma"), _f32(beta, "Beta"), _f32(weight, "W")],
    )
    x = _calibration(K=K, num_samples=16, seed=9)
    result = onnxsim.apply_outlier_suppression(model, calibration_data=[{"X": x}])
    assert result.SerializeToString() == model.SerializeToString()


def test_outlier_suppression_layernorm_without_bias_input():
    K, N = 16, 8
    rng = np.random.default_rng(10)
    gamma = np.ones(K, dtype=np.float32)
    gamma[2] = 12.0
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Ln = LayerNormalization<axis = -1>(X, Gamma)
          Y = MatMul(Ln, W)
        }}
        """,
        [_f32(gamma, "Gamma"), _f32(weight, "W")],
    )
    x = _calibration(K=K, num_samples=32, seed=11)
    calibration_data = [{"X": x}]

    osup_model = onnxsim.apply_outlier_suppression(
        model, calibration_data=calibration_data
    )
    onnx.checker.check_model(osup_model)

    float_out = _run(model, {"X": x})
    osup_out = _run(osup_model, {"X": x}, output_names=["Y"])
    assert _rel_l2(float_out["Y"], osup_out["Y"]) < 1e-4


def test_outlier_suppression_noop_when_no_layernorm_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_outlier_suppression(
        model, calibration_data=[{"X": np.zeros((4, 4), dtype=np.float32)}]
    )
    assert result.SerializeToString() == model.SerializeToString()
