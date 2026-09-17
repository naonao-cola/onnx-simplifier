"""Tests for ``onnxsim.apply_slim_llm_cpp`` -- the C++-backed port of
``onnxsim.apply_slim_llm`` (SliM-LLM, see ``onnxsim/slim_llm.py``).
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

from onnxsim.onnx_simplifier import apply_slim_llm_cpp
from onnxsim.slim_llm import apply_slim_llm

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
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-6)


def _matmul_model(K, N, seed=0, opset=21):
    rng = np.random.default_rng(seed)
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.1
    return _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
        opset=opset,
    )


def _matmul_model_with_outlier_group(K, N, group_size, outlier_group, seed=0):
    rng = np.random.default_rng(seed)
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.05
    lo, hi = outlier_group * group_size, (outlier_group + 1) * group_size
    weight[lo:hi, :] = rng.standard_normal((group_size, N)).astype(np.float32) * 10.0
    return _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )


def _calibration(K, num_samples=32, seed=1):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((num_samples, K)).astype(np.float32)


def _group_bits(model, prefix="W"):
    tensor = next(
        t
        for t in model.graph.initializer
        if t.name.startswith(prefix) and t.name.endswith("_group_bits")
    )
    return np.frombuffer(tensor.raw_data, dtype=np.int64).copy()


def test_slim_llm_cpp_picks_the_more_salient_group_for_high_bits():
    K, N, group_size, outlier_group = 64, 8, 16, 2
    model = _matmul_model_with_outlier_group(
        K=K, N=N, group_size=group_size, outlier_group=outlier_group, seed=0
    )
    x = _calibration(K=K, num_samples=64, seed=1)
    q = apply_slim_llm_cpp(
        model,
        calibration_data=[{"X": x}],
        target_bits=3.0,
        low_bits=2,
        high_bits=4,
        group_size=group_size,
    )
    onnx.checker.check_model(q)

    bits = _group_bits(q)
    assert len(bits) == K // group_size
    assert bits[outlier_group] == 4
    assert np.any(bits == 2)


def test_slim_llm_cpp_matches_python_reference_group_bits():
    K, N, group_size = 128, 8, 16
    model = _matmul_model(K=K, N=N, seed=2)
    x = _calibration(K=K, num_samples=32, seed=3)
    kwargs = dict(
        calibration_data=[{"X": x}],
        target_bits=2.5,
        low_bits=2,
        high_bits=4,
        group_size=group_size,
    )
    py_q = apply_slim_llm(model, **kwargs)
    cpp_q = apply_slim_llm_cpp(model, **kwargs)

    py_bits = _group_bits(py_q)
    cpp_bits = _group_bits(cpp_q)
    np.testing.assert_array_equal(py_bits, cpp_bits)

    py_codes = next(
        t for t in py_q.graph.initializer if t.name.endswith("_slimllm_codes")
    )
    cpp_codes = next(
        t for t in cpp_q.graph.initializer if t.name.endswith("_slimllm_codes")
    )
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(py_codes), onnx.numpy_helper.to_array(cpp_codes)
    )


def test_slim_llm_cpp_reconstruction_matches_codes_times_scale():
    K, N, group_size = 64, 8, 16
    model = _matmul_model(K=K, N=N, seed=6)
    x = _calibration(K=K, num_samples=32, seed=7)
    q = apply_slim_llm_cpp(
        model,
        calibration_data=[{"X": x}],
        target_bits=3.0,
        group_size=group_size,
    )

    codes_init = next(
        t for t in q.graph.initializer if t.name.endswith("_slimllm_codes")
    )
    scale_init = next(
        t for t in q.graph.initializer if t.name.endswith("_slimllm_scale")
    )
    codes = onnx.numpy_helper.to_array(codes_init).astype(np.float64)  # [K, N]
    scale = onnx.numpy_helper.to_array(scale_init).astype(np.float64)  # [K/gs, N]
    scale_full = np.repeat(scale, group_size, axis=0)
    dequant_kn = codes * scale_full

    (dequant_y,) = _run(q, {"X": np.eye(K, dtype=np.float32)})
    np.testing.assert_allclose(dequant_y, dequant_kn, rtol=1e-2, atol=1e-2)


def test_slim_llm_cpp_output_stays_finite_via_onnxruntime():
    K, N, group_size = 64, 8, 16
    model = _matmul_model(K=K, N=N, seed=8)
    x = _calibration(K=K, num_samples=32, seed=9)
    q = apply_slim_llm_cpp(
        model,
        calibration_data=[{"X": x}],
        target_bits=3.0,
        group_size=group_size,
    )
    onnx.checker.check_model(q)

    rng = np.random.default_rng(10)
    x_eval = rng.standard_normal((4, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x_eval})
    (q_y,) = _run(q, {"X": x_eval})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 1.5


def test_slim_llm_cpp_declines_when_k_not_divisible_by_group_size():
    rng = np.random.default_rng(11)
    weight = rng.standard_normal((20, 4)).astype(np.float32)
    model = _model(
        """
        g (float[batch,20] X) => (float[batch,4] Y)
        {
          Y = MatMul(X, W)
        }
        """,
        [_f32(weight, "W")],
    )
    q = apply_slim_llm_cpp(
        model,
        calibration_data=[{"X": np.zeros((1, 20), dtype=np.float32)}],
        group_size=16,
    )
    assert q.SerializeToString() == model.SerializeToString()


def test_slim_llm_cpp_declines_non_constant_weight():
    model = _model(
        """
        g (float[4,32] X, float[32,4] W) => (float[4,4] Y)
        {
          Y = MatMul(X, W)
        }
        """
    )
    q = apply_slim_llm_cpp(
        model, calibration_data=[{"X": np.zeros((4, 32), dtype=np.float32)}]
    )
    assert q.SerializeToString() == model.SerializeToString()


def test_slim_llm_cpp_noop_when_no_matmul_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = apply_slim_llm_cpp(
        model, calibration_data=[{"X": np.zeros((4, 4), dtype=np.float32)}]
    )
    assert result.SerializeToString() == model.SerializeToString()


def test_slim_llm_cpp_declines_below_opset21():
    model = _matmul_model(K=32, N=8, opset=13)
    result = apply_slim_llm_cpp(
        model, calibration_data=[{"X": np.zeros((1, 32), dtype=np.float32)}]
    )
    assert result.SerializeToString() == model.SerializeToString()


def test_slim_llm_cpp_rejects_invalid_bit_range():
    model = _matmul_model(K=32, N=8)
    calib = [{"X": np.zeros((1, 32), dtype=np.float32)}]
    with pytest.raises((RuntimeError, ValueError)):
        apply_slim_llm_cpp(model, calibration_data=calib, low_bits=4, high_bits=4)
    with pytest.raises((RuntimeError, ValueError)):
        apply_slim_llm_cpp(model, calibration_data=calib, low_bits=1, high_bits=4)


def test_slim_llm_cpp_missing_calibration_input_raises():
    model = _matmul_model(K=32, N=8)
    with pytest.raises((RuntimeError, ValueError)):
        apply_slim_llm_cpp(model, calibration_data=[{"WrongName": np.zeros((1, 32))}])
