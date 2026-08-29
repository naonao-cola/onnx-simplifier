"""Tests for ``onnxsim.apply_quarot_cpp`` -- the C++-backed port of
``onnxsim.apply_quarot`` (see ``onnxsim/passes/quarot.h`` and
``onnxsim/passes/random_orthogonal.h``). Unlike the MXFP4/double-quantization
C++ ports, this pass draws a fresh random rotation per layer using its own
independent RNG derivation (not a numpy Generator sequenced across matches
in graph node order), so its output is expected to be *accurate*, not
bit-identical to the Python port -- these tests check structure and
numerical accuracy rather than exact equality.
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


def _matmul_model(K=32, N=8, weight=None, seed=0, opset=21):
    if weight is None:
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


def test_cpp_quarot_quantizes_matmul_with_standard_ops_only():
    model = _matmul_model(K=32, N=8, seed=0)
    q = onnxsim.apply_quarot_cpp(model, seed=0)
    onnx.checker.check_model(q)

    op_types = {n.op_type for n in q.graph.node}
    assert op_types <= {
        "MatMul",
        "Abs",
        "ReduceMax",
        "Clip",
        "Div",
        "Round",
        "Mul",
        "DequantizeLinear",
        "Add",
        "Identity",
    }
    assert all(n.domain in ("", "ai.onnx") for n in q.graph.node)


def test_cpp_quarot_rotation_is_orthogonal():
    model = _matmul_model(K=32, N=8, seed=1)
    q = onnxsim.apply_quarot_cpp(model, seed=2)
    u = next(
        onnx.numpy_helper.to_array(t)
        for t in q.graph.initializer
        if list(t.dims) == [32, 32]
    )
    identity = u.astype(np.float64) @ u.astype(np.float64).T
    assert np.allclose(identity, np.eye(32), atol=1e-4)


def test_cpp_quarot_output_stays_close_to_float_via_onnxruntime():
    model = _matmul_model(K=32, N=8, seed=3)
    q = onnxsim.apply_quarot_cpp(model, seed=3)
    onnx.checker.check_model(q)

    rng = np.random.default_rng(4)
    x = rng.standard_normal((8, 32)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.5


def test_cpp_quarot_gemm_with_bias():
    rng = np.random.default_rng(5)
    K, N = 64, 12
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    bias = rng.standard_normal((N,)).astype(np.float32) * 0.1
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm(X, W, B)
        }}
        """,
        initializer=[_f32(weight, "W"), _f32(bias, "B")],
    )
    q = onnxsim.apply_quarot_cpp(model, seed=6)
    onnx.checker.check_model(q)
    assert any(n.op_type == "Add" for n in q.graph.node)

    x = rng.standard_normal((4, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.5


def test_cpp_quarot_skips_non_block_divisible_k():
    model = _matmul_model(K=48, N=8, seed=7)  # 48 is not a multiple of 32
    q = onnxsim.apply_quarot_cpp(model, seed=0)
    assert q.SerializeToString() == model.SerializeToString()


def test_cpp_quarot_declines_pre_opset21():
    model = _matmul_model(K=32, N=8, seed=8, opset=13)
    q = onnxsim.apply_quarot_cpp(model, seed=0)
    assert q.SerializeToString() == model.SerializeToString()


def test_cpp_quarot_is_deterministic_for_a_given_seed():
    model = _matmul_model(K=32, N=8, seed=9)
    q1 = onnxsim.apply_quarot_cpp(model, seed=42)
    q2 = onnxsim.apply_quarot_cpp(model, seed=42)
    assert q1.SerializeToString() == q2.SerializeToString()


def test_cpp_quarot_different_seeds_give_different_rotations():
    model = _matmul_model(K=32, N=8, seed=10)
    q1 = onnxsim.apply_quarot_cpp(model, seed=1)
    q2 = onnxsim.apply_quarot_cpp(model, seed=2)
    assert q1.SerializeToString() != q2.SerializeToString()
