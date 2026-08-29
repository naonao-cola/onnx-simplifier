"""Tests for ``onnxsim.quantize_weight_only_mxfp4_cpp`` -- the C++-backed
port of ``onnxsim.quantize_weight_only_mxfp4`` (see
``onnxsim/passes/weight_only_quantize_mxfp4_matmul.h`` and
``onnxsim/passes/quantize_mxfp4_common.h``). Cross-checks the C++ port
against the pure-Python reference implementation on the same input, in
addition to the same standalone checks ``test_mx_quantization.py`` runs.
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


def _matmul_model(K=64, N=16, weight=None, seed=0):
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


def test_cpp_mxfp4_quantizes_matmul_with_standard_ops_only():
    model = _matmul_model(K=64, N=16, seed=0)
    q = onnxsim.quantize_weight_only_mxfp4_cpp(model)
    onnx.checker.check_model(q)

    op_types = {n.op_type for n in q.graph.node}
    assert op_types <= {"MatMul", "Cast", "Gather", "Reshape", "Mul"}
    assert all(n.domain in ("", "ai.onnx") for n in q.graph.node)


def test_cpp_mxfp4_block_scale_is_a_power_of_two():
    rng = np.random.default_rng(1)
    weight = rng.standard_normal((64, 16)).astype(np.float32) * 3.7
    model = _matmul_model(weight=weight)
    q = onnxsim.quantize_weight_only_mxfp4_cpp(model)

    # The scale initializer is whichever one is 2-D with the block-count
    # shape [K // 32, N] -- unlike the pure-Python port, the C++ pass mints
    # anonymous initializer names, so it's located by shape/dtype instead.
    scale_init = next(
        t
        for t in q.graph.initializer
        if t.data_type == onnx.TensorProto.FLOAT and list(t.dims) == [64 // 32, 16]
    )
    scale = onnx.numpy_helper.to_array(scale_init).astype(np.float64).ravel()
    log2_scale = np.log2(scale)
    assert np.all(np.abs(log2_scale - np.round(log2_scale)) < 1e-9)


def test_cpp_mxfp4_output_stays_close_to_float_via_onnxruntime():
    model = _matmul_model(K=64, N=16, seed=3)
    q = onnxsim.quantize_weight_only_mxfp4_cpp(model)
    onnx.checker.check_model(q)

    rng = np.random.default_rng(4)
    x = rng.standard_normal((8, 64)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.3


def test_cpp_mxfp4_gemm_transb():
    rng = np.random.default_rng(5)
    K, N = 128, 12
    weight = rng.standard_normal((N, K)).astype(np.float32) * 0.5
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm<transB = 1>(X, W)
        }}
        """,
        initializer=[_f32(weight, "W")],
    )
    q = onnxsim.quantize_weight_only_mxfp4_cpp(model)
    onnx.checker.check_model(q)

    x = rng.standard_normal((4, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert _rel_l2(float_y, q_y) < 0.3


def test_cpp_mxfp4_skips_non_block_divisible_k():
    model = _matmul_model(K=48, N=8, seed=7)  # 48 is not a multiple of 32
    q = onnxsim.quantize_weight_only_mxfp4_cpp(model)
    assert q.SerializeToString() == model.SerializeToString()


def test_cpp_mxfp4_skips_non_constant_weight():
    model = _model(
        """
        g (float[4,64] X, float[64,4] W) => (float[4,4] Y)
        {
          Y = MatMul(X, W)
        }
        """
    )
    q = onnxsim.quantize_weight_only_mxfp4_cpp(model)
    assert q.SerializeToString() == model.SerializeToString()


def test_cpp_mxfp4_matches_python_reference_output():
    # The C++ port and the pure-Python reference implement the same
    # block-wise MXFP4 algorithm (round-to-nearest onto the same fixed E2M1
    # codebook, the same power-of-two scale rule) -- on the same weight they
    # should produce numerically equivalent dequantized outputs, not just
    # separately "close to float32".
    rng = np.random.default_rng(9)
    weight = rng.standard_normal((64, 16)).astype(np.float32) * 0.7
    model = _matmul_model(weight=weight)

    q_py = onnxsim.quantize_weight_only_mxfp4(model, block_size=32)
    q_cpp = onnxsim.quantize_weight_only_mxfp4_cpp(model)
    onnx.checker.check_model(q_py)
    onnx.checker.check_model(q_cpp)

    x = rng.standard_normal((4, 64)).astype(np.float32)
    (y_py,) = _run(q_py, {"X": x})
    (y_cpp,) = _run(q_cpp, {"X": x})
    assert np.allclose(y_py, y_cpp, rtol=1e-4, atol=1e-4)
