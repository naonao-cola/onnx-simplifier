"""Tests for ``onnxsim.apply_norm_tweaking_cpp`` -- the C++-backed port of
``onnxsim.apply_norm_tweaking`` (Norm Tweaking, see
``onnxsim/norm_tweaking_entry.h``). Unlike most other calibration-driven
``*_cpp`` ports, this one runs BOTH the float model and the quantized
model through a real ``onnxruntime``-backed executor for every calibration
batch (never a fake/mock executor).

``onnxsim.apply_norm_tweaking`` now delegates to this C++ port (see
``onnxsim/norm_tweaking.py``), so ``tests/test_norm_tweaking.py`` covers
delegation-level behavior/edge cases while this file covers the C++
port's own structural and numerical details directly, including
close-to-exact agreement with the pure-Python closed-form reference (a
deterministic moment-matching fit with no RNG or iterative solver, so
agreement is expected to be very close, modulo double-precision
summation-order differences -- not necessarily bit-exact).
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.onnx_simplifier import apply_norm_tweaking_cpp

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


def _scale_bias(model):
    ln_node = next(n for n in model.graph.node if n.op_type == "LayerNormalization")
    scale_t = next(t for t in model.graph.initializer if t.name == ln_node.input[1])
    bias_t = next(t for t in model.graph.initializer if t.name == ln_node.input[2])
    return onnx.numpy_helper.to_array(scale_t), onnx.numpy_helper.to_array(bias_t)


def test_norm_tweaking_cpp_recovers_known_scale_bias_corruption():
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
    tweaked = apply_norm_tweaking_cpp(
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


def test_norm_tweaking_cpp_matches_python_reference_closely():
    rng = np.random.default_rng(2)
    K, N = 32, 16
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    scale = rng.uniform(0.5, 1.5, N).astype(np.float32)
    bias = rng.standard_normal(N).astype(np.float32)
    float_model = _matmul_ln_model(w, scale, bias, K, N)

    quant = onnxsim.quantize_weight_only_int4(float_model)
    fit_data = _random_calibration(K, num_samples=16, seed=3)

    py = onnxsim.apply_norm_tweaking(float_model, quant, calibration_data=fit_data)
    cpp = apply_norm_tweaking_cpp(float_model, quant, calibration_data=fit_data)
    onnx.checker.check_model(cpp)

    py_scale, py_bias = _scale_bias(py)
    cpp_scale, cpp_bias = _scale_bias(cpp)
    np.testing.assert_allclose(cpp_scale, py_scale, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(cpp_bias, py_bias, rtol=1e-4, atol=1e-5)


def test_norm_tweaking_cpp_reduces_error_after_real_int4_quantization():
    rng = np.random.default_rng(2)
    K, N = 32, 16
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    scale = rng.uniform(0.5, 1.5, N).astype(np.float32)
    bias = rng.standard_normal(N).astype(np.float32)
    float_model = _matmul_ln_model(w, scale, bias, K, N)

    quant = onnxsim.quantize_weight_only_int4(float_model)
    assert any(n.op_type == "LayerNormalization" for n in quant.graph.node)

    fit_data = _random_calibration(K, num_samples=64, seed=3)
    tweaked = apply_norm_tweaking_cpp(float_model, quant, calibration_data=fit_data)
    onnx.checker.check_model(tweaked)

    held_out = _random_calibration(K, num_samples=1, batch=256, seed=4)[0]["X"]
    (float_y,) = _run(float_model, {"X": held_out})
    (quant_y,) = _run(quant, {"X": held_out})
    (tweaked_y,) = _run(tweaked, {"X": held_out})

    quant_err = _rel_l2(float_y, quant_y)
    tweaked_err = _rel_l2(float_y, tweaked_y)
    assert tweaked_err < quant_err


def test_norm_tweaking_cpp_preserves_scale_and_bias_shape():
    rng = np.random.default_rng(5)
    K, N = 8, 6
    w = rng.standard_normal((K, N)).astype(np.float32)
    scale = rng.uniform(0.5, 1.5, N).astype(np.float32)
    bias = rng.standard_normal(N).astype(np.float32)
    float_model = _matmul_ln_model(w, scale, bias, K, N, batch=4)
    quant = onnxsim.quantize_weight_only_int4(float_model)

    tweaked = apply_norm_tweaking_cpp(
        float_model, quant, calibration_data=_random_calibration(K, 8, batch=4, seed=6)
    )
    ln_node = next(n for n in tweaked.graph.node if n.op_type == "LayerNormalization")
    scale_t = next(t for t in tweaked.graph.initializer if t.name == ln_node.input[1])
    bias_t = next(t for t in tweaked.graph.initializer if t.name == ln_node.input[2])
    assert list(scale_t.dims) == [N]
    assert list(bias_t.dims) == [N]
    assert scale_t.name.endswith("_norm_tweak_scale")
    assert bias_t.name.endswith("_norm_tweak_bias")


def test_norm_tweaking_cpp_adds_bias_when_layernorm_has_none():
    rng = np.random.default_rng(7)
    K, N = 8, 6
    w = rng.standard_normal((K, N)).astype(np.float32)
    scale = rng.uniform(0.5, 1.5, N).astype(np.float32)
    float_model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          M = MatMul(X, W)
          Y = LayerNormalization<axis = -1>(M, scale)
        }}
        """,
        initializer=[_f32(w, "W"), _f32(scale, "scale")],
    )
    corrupted_scale = scale * 2.0
    quantized_model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          M = MatMul(X, W)
          Y = LayerNormalization<axis = -1>(M, scale)
        }}
        """,
        initializer=[_f32(w, "W"), _f32(corrupted_scale, "scale")],
    )
    ln_node = next(
        n for n in quantized_model.graph.node if n.op_type == "LayerNormalization"
    )
    assert len(ln_node.input) == 2

    calib = _random_calibration(K, num_samples=8, batch=4, seed=8)
    tweaked = apply_norm_tweaking_cpp(
        float_model, quantized_model, calibration_data=calib
    )
    onnx.checker.check_model(tweaked)
    tweaked_ln = next(
        n for n in tweaked.graph.node if n.op_type == "LayerNormalization"
    )
    assert len(tweaked_ln.input) == 3


def test_norm_tweaking_cpp_noop_when_no_layernorm_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = apply_norm_tweaking_cpp(
        model, model, calibration_data=[{"X": np.zeros((4, 4), dtype=np.float32)}]
    )
    assert result.SerializeToString() == model.SerializeToString()


def test_norm_tweaking_cpp_empty_calibration_data_is_noop():
    rng = np.random.default_rng(9)
    K, N = 8, 6
    w = rng.standard_normal((K, N)).astype(np.float32)
    scale = rng.uniform(0.5, 1.5, N).astype(np.float32)
    bias = rng.standard_normal(N).astype(np.float32)
    float_model = _matmul_ln_model(w, scale, bias, K, N, batch=4)
    quant = onnxsim.quantize_weight_only_int4(float_model)
    result = apply_norm_tweaking_cpp(float_model, quant, calibration_data=[])
    # No calibration batch was ever run, so no candidate's scale/bias could
    # have been measured -- the LayerNorm's own scale/bias stay exactly
    # the originals (still `scale`/`bias` by name, not a new
    # `_norm_tweak_scale`/`_norm_tweak_bias` initializer).
    ln_node = next(n for n in result.graph.node if n.op_type == "LayerNormalization")
    assert ln_node.input[1] == "scale"
    assert ln_node.input[2] == "bias"


def test_norm_tweaking_cpp_missing_calibration_input_raises():
    rng = np.random.default_rng(10)
    K, N = 8, 6
    w = rng.standard_normal((K, N)).astype(np.float32)
    scale = rng.uniform(0.5, 1.5, N).astype(np.float32)
    bias = rng.standard_normal(N).astype(np.float32)
    float_model = _matmul_ln_model(w, scale, bias, K, N, batch=4)
    quant = onnxsim.quantize_weight_only_int4(float_model)
    with pytest.raises((RuntimeError, ValueError)):
        apply_norm_tweaking_cpp(
            float_model,
            quant,
            calibration_data=[{"NotX": np.zeros((4, K), dtype=np.float32)}],
        )
