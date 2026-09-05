"""Tests for ``onnxsim.apply_norm_tweaking`` (Norm Tweaking, see
``onnxsim/norm_tweaking.py``) -- recalibrates a LayerNormalization node's
own ``scale``/``bias`` parameters in place so its output distribution
matches the float model's, correcting for the distribution shift a
quantized upstream layer introduces.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim

ort = pytest.importorskip("onnxruntime")


def _model(body, initializer=(), opset=21, ir_version=10):
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
    return onnx.numpy_helper.from_array(np.asarray(array, dtype=np.float32), name)


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-9)


def _matmul_ln_model(w, scale, bias, K, N, batch=8):
    return _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          M = MatMul(X, W)
          Y = LayerNormalization<axis = -1>(M, scale, bias)
        }}
        """,
        initializer=[_f32(w, "W"), _f32(scale, "scale"), _f32(bias, "bias")],
    )


def _random_calibration(K, num_samples, batch=8, seed=0):
    rng = np.random.default_rng(seed)
    return [
        {"X": rng.standard_normal((batch, K)).astype(np.float32)}
        for _ in range(num_samples)
    ]


def test_norm_tweaking_recovers_known_scale_bias_corruption():
    # The pre-LayerNorm activation (M = MatMul(X, W)) is identical between
    # float_model and quantized_model here -- only the LayerNorm's own
    # scale/bias were deliberately corrupted, simulating "wrong" norm
    # parameters. Since normalize(x) is then exactly the same in both
    # models, the closed-form moment-matching correction should recover the
    # float model's output almost exactly.
    rng = np.random.default_rng(0)
    K, N = 16, 12
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    scale = rng.uniform(0.5, 1.5, N).astype(np.float32)
    bias = rng.standard_normal(N).astype(np.float32)
    float_model = _matmul_ln_model(w, scale, bias, K, N)

    corrupted_scale = scale * rng.uniform(0.3, 3.0, N).astype(np.float32)
    corrupted_bias = bias + rng.standard_normal(N).astype(np.float32) * 2.0
    quantized_model = _matmul_ln_model(w, corrupted_scale, corrupted_bias, K, N)

    calibration_data = _random_calibration(K, num_samples=32, seed=1)
    tweaked = onnxsim.apply_norm_tweaking(
        float_model, quantized_model, calibration_data=calibration_data
    )
    onnx.checker.check_model(tweaked)

    x = np.concatenate([b["X"] for b in calibration_data], axis=0)
    (float_y,) = _run(float_model, {"X": x})
    (corrupted_y,) = _run(quantized_model, {"X": x})
    (tweaked_y,) = _run(tweaked, {"X": x})

    corrupted_err = _rel_l2(float_y, corrupted_y)
    tweaked_err = _rel_l2(float_y, tweaked_y)
    assert tweaked_err < 1e-4
    assert tweaked_err < corrupted_err


def test_norm_tweaking_reduces_error_after_real_int4_quantization():
    # A realistic scenario: only the preceding MatMul's weight is actually
    # quantized (via quantize_weight_only_int4), which leaves the
    # LayerNorm's own scale/bias completely untouched even though the
    # distribution flowing into it has shifted. Fit the tweak on one
    # calibration batch and evaluate on a *held-out* batch, so this isn't
    # just checking that the closed form reproduces its own fitting data.
    rng = np.random.default_rng(2)
    K, N = 32, 16
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    scale = rng.uniform(0.5, 1.5, N).astype(np.float32)
    bias = rng.standard_normal(N).astype(np.float32)
    float_model = _matmul_ln_model(w, scale, bias, K, N)

    quant = onnxsim.quantize_weight_only_int4(float_model)
    # quantize_weight_only_int4 must not have touched the LayerNorm node.
    assert any(n.op_type == "LayerNormalization" for n in quant.graph.node)

    fit_data = _random_calibration(K, num_samples=64, seed=3)
    tweaked = onnxsim.apply_norm_tweaking(float_model, quant, calibration_data=fit_data)
    onnx.checker.check_model(tweaked)

    held_out = _random_calibration(K, num_samples=1, batch=256, seed=4)[0]["X"]
    (float_y,) = _run(float_model, {"X": held_out})
    (quant_y,) = _run(quant, {"X": held_out})
    (tweaked_y,) = _run(tweaked, {"X": held_out})

    quant_err = _rel_l2(float_y, quant_y)
    tweaked_err = _rel_l2(float_y, tweaked_y)
    assert tweaked_err < quant_err


def test_norm_tweaking_preserves_scale_and_bias_shape():
    rng = np.random.default_rng(5)
    K, N = 8, 6
    w = rng.standard_normal((K, N)).astype(np.float32)
    scale = rng.uniform(0.5, 1.5, N).astype(np.float32)
    bias = rng.standard_normal(N).astype(np.float32)
    float_model = _matmul_ln_model(w, scale, bias, K, N, batch=4)
    quant = onnxsim.quantize_weight_only_int4(float_model)

    tweaked = onnxsim.apply_norm_tweaking(
        float_model, quant, calibration_data=_random_calibration(K, 8, batch=4, seed=6)
    )
    ln_node = next(n for n in tweaked.graph.node if n.op_type == "LayerNormalization")
    scale_t = next(t for t in tweaked.graph.initializer if t.name == ln_node.input[1])
    bias_t = next(t for t in tweaked.graph.initializer if t.name == ln_node.input[2])
    assert list(scale_t.dims) == [N]
    assert list(bias_t.dims) == [N]


def test_norm_tweaking_noop_when_no_layernorm_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_norm_tweaking(
        model, model, calibration_data=[{"X": np.zeros((4, 4), dtype=np.float32)}]
    )
    assert result.SerializeToString() == model.SerializeToString()
