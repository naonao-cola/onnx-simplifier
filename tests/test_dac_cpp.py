"""Tests for ``onnxsim.apply_dac_cpp`` -- the C++-backed port of
``onnxsim.apply_dac`` (D2Quant's Deviation-Aware Correction, see
``onnxsim/dac_entry.h``). Like ``test_norm_tweaking_cpp.py``, this runs
BOTH the float model and the quantized model through a real
``onnxruntime``-backed executor for every calibration batch (never a
fake/mock executor).

``onnxsim.apply_dac`` now delegates to this C++ port (see
``onnxsim/d2quant.py``), so ``tests/test_d2quant.py``'s own DAC section
covers delegation-level behavior/edge cases while this file covers the
C++ port's own structural and numerical details directly, including
close-to-exact agreement with the pure-Python closed-form reference.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.onnx_simplifier import apply_dac_cpp

ort = pytest.importorskip("onnxruntime")


def _model(body, initializer=(), opset=17, ir_version=10):
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


def _ln_shift_models(K=16, shift=None, with_beta=True, seed=0):
    rng = np.random.default_rng(seed)
    gamma = (np.ones(K) + rng.standard_normal(K) * 0.05).astype(np.float32)
    beta = (rng.standard_normal(K) * 0.1).astype(np.float32)
    if shift is None:
        shift = np.zeros(K, dtype=np.float32)

    beta_arg = ", Beta" if with_beta else ""
    beta_init = [_f32(beta, "Beta")] if with_beta else []

    float_model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{K}] Y)
        {{
          Y = LayerNormalization<axis = -1>(X, Gamma{beta_arg})
        }}
        """,
        [_f32(gamma, "Gamma")] + beta_init,
    )
    quantized_model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{K}] Y)
        {{
          Xq = Add(X, Shift)
          Y = LayerNormalization<axis = -1>(Xq, Gamma{beta_arg})
        }}
        """,
        [_f32(gamma, "Gamma"), _f32(shift, "Shift")] + beta_init,
    )
    return float_model, quantized_model, gamma, beta if with_beta else None


def test_dac_cpp_reduces_deviation_from_shifted_layernorm_input():
    K = 16
    shift = np.zeros(K, dtype=np.float32)
    shift[3] = 2.0
    shift[9] = -1.5
    float_model, quantized_model, gamma, beta = _ln_shift_models(
        K=K, shift=shift, seed=0
    )

    rng = np.random.default_rng(1)
    calib = [{"X": rng.standard_normal((32, K)).astype(np.float32)} for _ in range(8)]

    corrected = apply_dac_cpp(float_model, quantized_model, calibration_data=calib)
    onnx.checker.check_model(corrected)

    beta_after = onnx.numpy_helper.to_array(
        next(t for t in corrected.graph.initializer if t.name == "Beta")
    )
    assert not np.allclose(beta_after, beta)
    gamma_after = onnx.numpy_helper.to_array(
        next(t for t in corrected.graph.initializer if t.name == "Gamma")
    )
    np.testing.assert_array_equal(gamma_after, gamma)  # DAC never touches gamma

    eval_x = rng.standard_normal((64, K)).astype(np.float32)
    (float_y,) = _run(float_model, {"X": eval_x}, output_names=["Y"]).values()
    (before_y,) = _run(quantized_model, {"X": eval_x}, output_names=["Y"]).values()
    (after_y,) = _run(corrected, {"X": eval_x}, output_names=["Y"]).values()
    assert _rel_l2(float_y, after_y) < _rel_l2(float_y, before_y)


def test_dac_cpp_matches_python_reference_closely():
    K = 16
    shift = np.zeros(K, dtype=np.float32)
    shift[2] = 1.5
    shift[7] = -2.0
    float_model, quantized_model, _gamma, _beta = _ln_shift_models(
        K=K, shift=shift, seed=5
    )
    rng = np.random.default_rng(6)
    calib = [{"X": rng.standard_normal((32, K)).astype(np.float32)} for _ in range(8)]

    py = onnxsim.apply_dac(float_model, quantized_model, calibration_data=calib)
    cpp = apply_dac_cpp(float_model, quantized_model, calibration_data=calib)
    onnx.checker.check_model(cpp)

    py_beta = onnx.numpy_helper.to_array(
        next(t for t in py.graph.initializer if t.name == "Beta")
    )
    cpp_beta = onnx.numpy_helper.to_array(
        next(t for t in cpp.graph.initializer if t.name == "Beta")
    )
    np.testing.assert_allclose(cpp_beta, py_beta, rtol=1e-4, atol=1e-5)


def test_dac_cpp_adds_bias_when_layernorm_has_none():
    K = 16
    shift = np.zeros(K, dtype=np.float32)
    shift[2] = 3.0
    float_model, quantized_model, _, _ = _ln_shift_models(
        K=K, shift=shift, with_beta=False, seed=2
    )
    ln_node = next(
        n for n in quantized_model.graph.node if n.op_type == "LayerNormalization"
    )
    assert len(ln_node.input) == 2

    rng = np.random.default_rng(3)
    calib = [{"X": rng.standard_normal((32, K)).astype(np.float32)} for _ in range(8)]
    corrected = apply_dac_cpp(float_model, quantized_model, calibration_data=calib)
    onnx.checker.check_model(corrected)

    corrected_ln = next(
        n for n in corrected.graph.node if n.op_type == "LayerNormalization"
    )
    assert len(corrected_ln.input) == 3
    bias_init = next(
        t for t in corrected.graph.initializer if t.name == corrected_ln.input[2]
    )
    assert bias_init.name.endswith("_dac_bias")
    assert onnx.numpy_helper.to_array(bias_init)[2] != 0.0


def test_dac_cpp_existing_bias_mutated_in_place_same_name():
    # An existing FLOAT bias of the right width is overwritten IN PLACE
    # (same initializer name) -- mirrors d2quant.py's own
    # _apply_ln_bias_correction exactly (no rewiring needed).
    K = 16
    shift = np.zeros(K, dtype=np.float32)
    shift[4] = 2.5
    float_model, quantized_model, _, beta = _ln_shift_models(K=K, shift=shift, seed=9)
    rng = np.random.default_rng(10)
    calib = [{"X": rng.standard_normal((32, K)).astype(np.float32)} for _ in range(8)]
    corrected = apply_dac_cpp(float_model, quantized_model, calibration_data=calib)
    ln_node = next(n for n in corrected.graph.node if n.op_type == "LayerNormalization")
    assert ln_node.input[2] == "Beta"
    beta_after = onnx.numpy_helper.to_array(
        next(t for t in corrected.graph.initializer if t.name == "Beta")
    )
    assert not np.allclose(beta_after, beta)


def test_dac_cpp_noop_when_no_deviation():
    K = 16
    float_model, quantized_model, _, beta = _ln_shift_models(K=K, shift=None, seed=4)

    rng = np.random.default_rng(5)
    calib = [{"X": rng.standard_normal((16, K)).astype(np.float32)} for _ in range(4)]
    corrected = apply_dac_cpp(float_model, quantized_model, calibration_data=calib)

    beta_after = onnx.numpy_helper.to_array(
        next(t for t in corrected.graph.initializer if t.name == "Beta")
    )
    np.testing.assert_allclose(beta_after, beta)


def test_dac_cpp_noop_when_no_layernorm_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    corrected = apply_dac_cpp(
        model, model, calibration_data=[{"X": np.zeros((4, 4), dtype=np.float32)}]
    )
    assert corrected.SerializeToString() == model.SerializeToString()


def test_dac_cpp_empty_calibration_data_is_noop():
    K = 16
    shift = np.zeros(K, dtype=np.float32)
    shift[1] = 4.0
    float_model, quantized_model, _, beta = _ln_shift_models(K=K, shift=shift, seed=11)
    corrected = apply_dac_cpp(float_model, quantized_model, calibration_data=[])
    beta_after = onnx.numpy_helper.to_array(
        next(t for t in corrected.graph.initializer if t.name == "Beta")
    )
    np.testing.assert_array_equal(beta_after, beta)


def test_dac_cpp_missing_calibration_input_raises():
    K = 16
    shift = np.zeros(K, dtype=np.float32)
    shift[0] = 1.0
    float_model, quantized_model, _, _ = _ln_shift_models(K=K, shift=shift, seed=12)
    with pytest.raises((RuntimeError, ValueError)):
        apply_dac_cpp(
            float_model,
            quantized_model,
            calibration_data=[{"NotX": np.zeros((4, K), dtype=np.float32)}],
        )


def test_dac_cpp_correction_threshold_gates_tiny_deviation():
    K = 16
    shift = np.zeros(K, dtype=np.float32)
    shift[5] = 1e-4  # Tiny, deliberately below a large correction_threshold.
    float_model, quantized_model, _, beta = _ln_shift_models(K=K, shift=shift, seed=13)
    rng = np.random.default_rng(14)
    calib = [{"X": rng.standard_normal((32, K)).astype(np.float32)} for _ in range(8)]
    corrected = apply_dac_cpp(
        float_model,
        quantized_model,
        calibration_data=calib,
        correction_threshold=1.0,
    )
    beta_after = onnx.numpy_helper.to_array(
        next(t for t in corrected.graph.initializer if t.name == "Beta")
    )
    np.testing.assert_array_equal(beta_after, beta)
