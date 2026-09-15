"""Tests for ``onnxsim.apply_qronos_cpp`` -- the C++-backed port of
``onnxsim.apply_qronos`` (a sequential, whole-model generalization of GPTQ,
see ``onnxsim/qronos.py``). Like ``test_gptq_cpp.py``, this runs the float
model over real calibration data through a real ``onnxruntime``-backed
executor -- never a fake/mock executor -- and checks exact (bit-for-bit)
agreement against the pure-Python reference: both sides join the same
candidates, process layers in the same forward-execution order, factor the
same Hessians, and pack the same codes, so any divergence is a bug, not an
accepted tolerance. Unlike GPTQ, this pass invokes the executor once per
matched layer (re-probing the progressively-corrected quantized model), so
these tests also exercise multi-layer models where a downstream layer's own
correction depends on an upstream layer's already-applied one.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.onnx_simplifier import apply_qronos_cpp
from onnxsim.qronos import apply_qronos

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


def _matmul_model(K=64, N=16, seed=0):
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


def _two_layer_model(K1=64, N1=32, N2=16, seed=0):
    rng = np.random.default_rng(seed)
    w1 = rng.standard_normal((K1, N1)).astype(np.float32) * 0.5
    w2 = rng.standard_normal((N1, N2)).astype(np.float32) * 0.5
    model = _model(
        f"""
        g (float[batch,{K1}] X) => (float[batch,{N2}] Y2)
        {{
          Y1 = MatMul(X, W1)
          Y2 = MatMul(Y1, W2)
        }}
        """,
        [_f32(w1, "W1"), _f32(w2, "W2")],
    )
    model.graph.value_info.append(
        onnx.helper.make_tensor_value_info("Y1", onnx.TensorProto.FLOAT, ["batch", N1])
    )
    return model


def _correlated_calibration(K=64, num_samples=64, rank=6, seed=1):
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((num_samples, rank)).astype(np.float32)
    projection = rng.standard_normal((rank, K)).astype(np.float32)
    x = latent @ projection
    x += rng.standard_normal((num_samples, K)).astype(np.float32) * 0.05
    return [{"X": x}]


def _assert_exact_parity(float_model, calibration_data, **kwargs):
    quant = onnxsim.quantize_weight_only_int4(float_model)
    py = apply_qronos(float_model, quant, calibration_data, **kwargs)
    cpp = apply_qronos_cpp(float_model, quant, calibration_data, **kwargs)
    onnx.checker.check_model(cpp)
    py_inits = sorted(py.graph.initializer, key=lambda t: t.name)
    cpp_inits = sorted(cpp.graph.initializer, key=lambda t: t.name)
    assert [t.name for t in py_inits] == [t.name for t in cpp_inits]
    for a, b in zip(py_inits, cpp_inits):
        assert a.data_type == b.data_type, a.name
        assert list(a.dims) == list(b.dims), a.name
        assert a.raw_data == b.raw_data, a.name
    return cpp


def test_qronos_cpp_matches_python_exactly_single_layer():
    # A single-layer model has no upstream-quantized predecessor, so this
    # also exercises Qronos's own exact GPTQ reduction (dx == 0).
    _assert_exact_parity(_matmul_model(), _correlated_calibration())


def test_qronos_cpp_matches_python_exactly_two_layer():
    # The real cross-layer scenario: Y2's own correction depends on Y1's
    # already-applied (Qronos-corrected) quantization error.
    model = _two_layer_model(K1=64, N1=32, N2=16, seed=2)
    calib = _correlated_calibration(K=64, seed=3)
    _assert_exact_parity(model, calib)


def test_qronos_cpp_matches_python_across_shapes_and_blocks():
    for K, N, seed, procb in [
        (128, 32, 5, 64),
        (256, 64, 11, 128),
        (96, 24, 21, 48),
        (32, 8, 23, 16),
    ]:
        model = _matmul_model(K=K, N=N, seed=seed)
        cals = _correlated_calibration(K=K, seed=seed + 100)
        _assert_exact_parity(model, cals, proc_block_size=procb)


def test_qronos_cpp_gemm_transb():
    rng = np.random.default_rng(8)
    K, N = 96, 12
    weight = (rng.standard_normal((N, K)) * 0.5).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm<transB = 1>(X, W)
        }}
        """,
        initializer=[_f32(weight, "W")],
    )
    _assert_exact_parity(model, _correlated_calibration(K=K, seed=9))


def test_qronos_cpp_ill_conditioned_calibration():
    # Dead channels plus near-duplicate channels: the Hessian is
    # singular without the dead-fix and damping, exercising exactly the
    # regularization paths most likely to diverge between LAPACK and
    # scalar kernels -- agreement must still be exact.
    K, N = 64, 16
    model = _matmul_model(K=K, N=N, seed=10)
    rng = np.random.default_rng(11)
    x = rng.standard_normal((64, K)).astype(np.float32)
    x[:, 4] = 0.0  # dead channel
    x[:, 8] = x[:, 0] * 1.0000001  # near-duplicate channel
    _assert_exact_parity(model, [{"X": x}])


def test_qronos_cpp_percdamp_matches():
    model = _matmul_model(K=64, N=16, seed=12)
    calib = _correlated_calibration(K=64, seed=13)
    _assert_exact_parity(model, calib, percdamp=0.05)


def test_qronos_cpp_noop_when_no_int4_matmul_present():
    model = _model(
        """
        g (float[batch,4] X) => (float[batch,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    quant = onnxsim.quantize_weight_only_int4(model)
    cpp = apply_qronos_cpp(
        model, quant, calibration_data=[{"X": np.zeros((1, 4), dtype=np.float32)}]
    )
    assert cpp.SerializeToString() == quant.SerializeToString()
