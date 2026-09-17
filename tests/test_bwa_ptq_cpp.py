"""Tests for ``onnxsim.apply_bwa_ptq_cpp`` -- the C++-backed port of
``onnxsim.apply_bwa_ptq`` (Binary Weight-Activation PTQ's Hessian-weighted
two-scale binary EM, see ``onnxsim/bwa_ptq_entry.h``). Like
``test_billm_cpp.py``, this runs the float model over real calibration
data through a real ``onnxruntime``-backed executor -- never a fake/mock
executor -- and checks agreement against the pure-Python reference: both
sides join the same candidates, accumulate the same Hessian diagonal, and
run the identical alternating EM update to the identical stopping
condition, so any divergence beyond ordinary floating-point rounding is a
bug, not an accepted tolerance.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.bwa_ptq import apply_bwa_ptq

ort = pytest.importorskip("onnxruntime")


def _model(body, initializer=(), opset=18, ir_version=9):
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


def _matmul_model(K=64, N=16, seed=0):
    rng = np.random.default_rng(seed)
    w = (rng.standard_normal((K, N)) * 0.5).astype(np.float32)
    return _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f32(w, "W")],
    )


def _correlated_calibration(K=64, num_samples=64, rank=6, seed=1):
    # Same low-rank-plus-noise shape test_billm_cpp.py/test_gptq_cpp.py's
    # own _correlated_calibration uses -- a well-conditioned but
    # non-trivial (non-identity) Hessian diagonal.
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((num_samples, rank)).astype(np.float32)
    projection = rng.standard_normal((rank, K)).astype(np.float32)
    x = latent @ projection
    x += rng.standard_normal((num_samples, K)).astype(np.float32) * 0.05
    return [{"X": x}]


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _assert_close_parity(model, calibration_data, **kwargs):
    py = apply_bwa_ptq(model, calibration_data, **kwargs)
    cpp = onnxsim.apply_bwa_ptq_cpp(model, calibration_data, **kwargs)
    onnx.checker.check_model(cpp)

    K = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value
    rng = np.random.default_rng(42)
    x = rng.standard_normal((4, K)).astype(np.float32)
    (y_py,) = _run(py, {"X": x})
    (y_cpp,) = _run(cpp, {"X": x})
    np.testing.assert_allclose(y_cpp, y_py, rtol=1e-5, atol=1e-6)
    return cpp


def test_bwa_ptq_cpp_matches_python_exactly():
    _assert_close_parity(_matmul_model(), _correlated_calibration())


def test_bwa_ptq_cpp_matches_python_across_shapes_and_groups():
    for K, N, seed, group_size in [
        (128, 32, 5, 64),
        (256, 64, 11, 128),
        (96, 24, 21, 48),
        (33, 8, 23, 16),  # K not a multiple of group_size
    ]:
        model = _matmul_model(K=K, N=N, seed=seed)
        cals = _correlated_calibration(K=K, seed=seed + 100)
        _assert_close_parity(model, cals, group_size=group_size)


def test_bwa_ptq_cpp_matches_python_with_bounded_em_iters():
    K, N = 64, 16
    model = _matmul_model(K=K, N=N, seed=30)
    cals = _correlated_calibration(K=K, seed=31)
    _assert_close_parity(model, cals, max_em_iters=1)
    _assert_close_parity(model, cals, max_em_iters=50)


def test_bwa_ptq_cpp_gemm_transb():
    rng = np.random.default_rng(8)
    K, N = 96, 12
    w = (rng.standard_normal((N, K)) * 0.5).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm<transB = 1>(X, W)
        }}
        """,
        initializer=[_f32(w, "W")],
    )
    cals = _correlated_calibration(K=K, seed=9)
    _assert_close_parity(model, cals)


def test_bwa_ptq_cpp_gemm_with_bias_untouched():
    rng = np.random.default_rng(15)
    K, N = 32, 8
    w = (rng.standard_normal((K, N)) * 0.5).astype(np.float32)
    b = (rng.standard_normal(N) * 0.1).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm(X, W, B)
        }}
        """,
        initializer=[_f32(w, "W"), _f32(b, "B")],
    )
    cals = _correlated_calibration(K=K, seed=16)
    cpp = _assert_close_parity(model, cals)
    # Bias is never touched by this pass.
    b_out = onnx.numpy_helper.to_array(
        next(t for t in cpp.graph.initializer if t.name == "B")
    )
    np.testing.assert_array_equal(b_out, b)


def test_bwa_ptq_cpp_produces_one_sign_bit_one_group_bit_per_element():
    K, N = 64, 16
    model = _matmul_model(K=K, N=N, seed=40)
    cals = _correlated_calibration(K=K, seed=41)
    cpp = onnxsim.apply_bwa_ptq_cpp(model, cals)
    onnx.checker.check_model(cpp)

    sign_t = next(t for t in cpp.graph.initializer if t.name.endswith("_bwa_sign"))
    group_t = next(
        t for t in cpp.graph.initializer if t.name.endswith("_bwa_group_select")
    )
    sign = onnx.numpy_helper.to_array(sign_t)
    group = onnx.numpy_helper.to_array(group_t)
    assert set(np.unique(sign).tolist()) <= {-1, 1}
    assert set(np.unique(group).tolist()) <= {0, 1}
    assert sign.shape == (K, N)


def test_bwa_ptq_cpp_noop_without_matmul_gemm():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    cal = [{"X": np.zeros((2, 4), dtype=np.float32)}]
    result = onnxsim.apply_bwa_ptq_cpp(model, cal)
    assert result.SerializeToString() == model.SerializeToString()


def test_bwa_ptq_cpp_missing_calibration_input_raises():
    model = _matmul_model(K=32, N=8, seed=50)
    bad_cal = [{"wrong_name": np.zeros((2, 32), dtype=np.float32)}]
    with pytest.raises(ValueError):
        onnxsim.apply_bwa_ptq_cpp(model, bad_cal)
