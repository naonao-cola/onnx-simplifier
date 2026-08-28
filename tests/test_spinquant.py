"""Tests for ``onnxsim.apply_spinquant`` -- see ``onnxsim/spinquant.py`` for
the technique (a calibrated, closed-form eigenvector-basis rotation --
SpinQuant's own "R1-only" idea, fit via PCA instead of a learned Cayley
optimizer -- followed by block-wise INT4 quantization).
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
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _matmul_model(K=32, N=8, seed=0, opset=21):
    rng = np.random.default_rng(seed)
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    return _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f32(weight, "W")],
        opset=opset,
    )


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-6)


def test_spinquant_output_stays_close_to_float_via_onnxruntime():
    model = _matmul_model(K=32, N=8, seed=0)
    q = onnxsim.apply_spinquant(model, block_size=8, num_samples=16, seed=1)
    onnx.checker.check_model(q)

    op_types = [n.op_type for n in q.graph.node]
    assert op_types.count("MatMul") == 2
    assert "DequantizeLinear" in op_types

    rng = np.random.default_rng(2)
    x = rng.standard_normal((8, 32)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.3


def test_spinquant_rotation_matrix_is_orthogonal():
    model = _matmul_model(K=16, N=4, seed=3)
    q = onnxsim.apply_spinquant(model, block_size=4, num_samples=32, seed=4)
    u_init = next(t for t in q.graph.initializer if t.name.endswith("_spinquant_u"))
    u = onnx.numpy_helper.to_array(u_init).astype(np.float64)
    assert np.allclose(u @ u.T, np.eye(u.shape[0]), atol=1e-4)


def test_spinquant_gemm_transb_with_bias():
    rng = np.random.default_rng(5)
    K, N = 32, 8
    weight = rng.standard_normal((N, K)).astype(np.float32) * 0.5
    bias = rng.standard_normal((N,)).astype(np.float32) * 0.1
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm<transB=1>(X, W, B)
        }}
        """,
        initializer=[_f32(weight, "W"), _f32(bias, "B")],
    )
    q = onnxsim.apply_spinquant(model, block_size=8, num_samples=16, seed=6)
    onnx.checker.check_model(q)
    assert "Add" in [n.op_type for n in q.graph.node]

    x = rng.standard_normal((4, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert _rel_l2(float_y, q_y) < 0.3


def test_spinquant_declines_when_k_not_divisible_by_block_size():
    model = _matmul_model(K=20, N=4, seed=7)  # 20 is not a multiple of 8
    q = onnxsim.apply_spinquant(model, block_size=8)
    assert q.SerializeToString() == model.SerializeToString()


def test_spinquant_declines_non_constant_weight():
    model = _model(
        """
        g (float[4,32] X, float[32,4] W) => (float[4,4] Y)
        {
          Y = MatMul(X, W)
        }
        """
    )
    q = onnxsim.apply_spinquant(model)
    assert q.SerializeToString() == model.SerializeToString()


def test_spinquant_noop_when_no_matmul_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_spinquant(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_spinquant_declines_below_opset21():
    model = _matmul_model(K=32, N=8, opset=13)
    result = onnxsim.apply_spinquant(model)
    assert result.SerializeToString() == model.SerializeToString()
