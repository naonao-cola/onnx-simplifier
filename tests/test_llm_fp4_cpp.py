"""Tests for ``onnxsim.quantize_weight_only_llm_fp4_cpp`` -- the C++-backed
port of ``onnxsim.quantize_weight_only_llm_fp4`` (LLM-FP4's searched
sign/exponent/mantissa 4-bit float format, see
``onnxsim/passes/llm_fp4.h``). Unlike this repo's fold-to-a-single-
initializer weight-only ports (e.g. ``nf4.h``), this port keeps the real
Gather/Reshape/Mul dequantization graph visible, matching
``llm_fp4.py``'s own choice exactly (the same reasoning
``weight_only_quantize_mxfp4_matmul.h``'s own identical rewrite already
documents) -- so recovering the *effective* dequantized weight, on either
side, means running the model through onnxruntime with an identity-matrix
input rather than reading a single initializer.

This is a closed-form grid search with no RNG anywhere (see
``passes/llm_fp4.h``'s own "ACCEPTED NUMERICAL SCOPE" note), so these tests
expect a tight numeric match against the pure-Python reference, not just a
comparable-quality one.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.llm_fp4 import quantize_weight_only_llm_fp4

ort = pytest.importorskip("onnxruntime")

_BLOCK_SIZE = 32  # llm_fp4.py's own default block_size, matching passes/llm_fp4.h


def _f32(array, name):
    return onnx.numpy_helper.from_array(np.asarray(array, dtype=np.float32), name)


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


def _matmul_model(w, K, N, batch="batch"):
    return _model(
        f"""
        g (float[{batch},{K}] X) => (float[{batch},{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(w, "W")],
    )


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-9)


def _effective_weight(model, k):
    # Neither side stores the dequantized weight as a single flat
    # initializer -- both keep the real Gather/Reshape/Mul chain visible
    # (see this module's own docstring). Feed the K x K identity matrix as
    # X: row i of Y then equals row i of the effective (dequantized)
    # weight, since Y = I @ W' == W'.
    (y,) = _run(model, {"X": np.eye(k, dtype=np.float32)})
    return y.astype(np.float64)


def test_cpp_replaces_weight_and_keeps_gather_chain_visible():
    rng = np.random.default_rng(0)
    K, N = _BLOCK_SIZE * 2, 8
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(w, K, N)

    q = onnxsim.quantize_weight_only_llm_fp4_cpp(model)
    onnx.checker.check_model(q)
    op_types = [n.op_type for n in q.graph.node]
    assert "Gather" in op_types
    assert op_types.count("Reshape") == 3
    assert op_types.count("Mul") == 1
    assert op_types.count("Cast") == 1
    assert op_types.count("MatMul") == 1  # the original node, kept in place

    codebook_t = next(t for t in q.graph.initializer if t.dims == [16])
    assert len(np.unique(onnx.numpy_helper.to_array(codebook_t))) <= 16

    new_w = _effective_weight(q, K)
    w64 = w.astype(np.float64)
    assert not np.allclose(new_w, w64)


def test_cpp_at_most_16_distinct_levels_per_channel_block():
    rng = np.random.default_rng(1)
    K, N = _BLOCK_SIZE * 3, 4
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(w, K, N)

    q = onnxsim.quantize_weight_only_llm_fp4_cpp(model)
    new_w = _effective_weight(q, K)  # [K, N]: reduction axis is axis 0 here

    for n in range(N):
        for start in range(0, K, _BLOCK_SIZE):
            block = new_w[start : start + _BLOCK_SIZE, n]
            assert len(np.unique(np.round(block, 6))) <= 16


def test_cpp_matches_python_reference_tightly():
    # No RNG anywhere in this technique (a closed-form grid search over a
    # deterministic MSE objective on both sides), so a genuinely tight
    # numeric cross-check is expected here, unlike this repo's k-means/
    # rotation-family ports.
    rng = np.random.default_rng(2)
    K, N = _BLOCK_SIZE * 4, 6
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.4
    model = _matmul_model(w, K, N)

    py_q = quantize_weight_only_llm_fp4(model)
    cpp_q = onnxsim.quantize_weight_only_llm_fp4_cpp(model)
    onnx.checker.check_model(cpp_q)

    py_w = _effective_weight(py_q, K)
    cpp_w = _effective_weight(cpp_q, K)
    np.testing.assert_allclose(cpp_w, py_w, rtol=1e-4, atol=1e-5)


def test_cpp_reduces_reconstruction_error_versus_naive_uniform_int4():
    # LLM-FP4's core empirical claim, checked against the C++ port's own
    # actual output: a searched (format, per-block scale) floating-point
    # codebook should reconstruct roughly-Gaussian weights at least as well
    # as a naive uniform 4-bit grid at the same block size.
    rng = np.random.default_rng(3)
    K, N = _BLOCK_SIZE * 8, 2
    w = rng.standard_normal((K, N)).astype(np.float32)
    model = _matmul_model(w, K, N)

    q = onnxsim.quantize_weight_only_llm_fp4_cpp(model)
    new_w = _effective_weight(q, K)
    w64 = w.astype(np.float64)

    naive_out = np.empty_like(w64)
    for n in range(N):
        for start in range(0, K, _BLOCK_SIZE):
            block = w64[start : start + _BLOCK_SIZE, n]
            scale = max(np.abs(block).max(), 1e-12) / 7.0
            codes = np.clip(np.round(block / scale), -7, 7)
            naive_out[start : start + _BLOCK_SIZE, n] = codes * scale

    fp4_mse = float(np.mean((w64 - new_w) ** 2))
    naive_mse = float(np.mean((w64 - naive_out) ** 2))
    assert fp4_mse <= naive_mse


def test_cpp_gemm_with_bias():
    rng = np.random.default_rng(5)
    K, N = _BLOCK_SIZE * 2, 8
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
    q = onnxsim.quantize_weight_only_llm_fp4_cpp(model)
    onnx.checker.check_model(q)

    x = rng.standard_normal((4, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.2


def test_cpp_gemm_transb_uses_channel_first_layout():
    # transB=1 stores W as [N, K] -- the reduction axis moves to axis 1,
    # exercising passes/llm_fp4.h's own weight_transposed branch.
    rng = np.random.default_rng(8)
    K, N = _BLOCK_SIZE * 2, 6
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
    q = onnxsim.quantize_weight_only_llm_fp4_cpp(model)
    onnx.checker.check_model(q)

    py_q = quantize_weight_only_llm_fp4(model)
    py_w = _effective_weight(py_q, K)
    cpp_w = _effective_weight(q, K)
    np.testing.assert_allclose(cpp_w, py_w, rtol=1e-4, atol=1e-5)


def test_cpp_noop_when_no_matmul_gemm_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.quantize_weight_only_llm_fp4_cpp(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_cpp_skips_non_2d_weight():
    rng = np.random.default_rng(7)
    w = rng.standard_normal((2, 4, 4, 4)).astype(np.float32)
    model = _model(
        """
        g (float[1,2,8,8] X) => (float[1,2,5,5] Y)
        {
          Y = Conv(X, W)
        }
        """,
        [_f32(w, "W")],
    )
    result = onnxsim.quantize_weight_only_llm_fp4_cpp(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_cpp_skips_non_block_divisible_weight():
    # K not a multiple of block_size (32) -- llm_fp4.py's own encoder skips
    # this layer entirely, and this port matches that exactly.
    rng = np.random.default_rng(9)
    K, N = _BLOCK_SIZE + 5, 4
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.4
    model = _matmul_model(w, K, N)
    result = onnxsim.quantize_weight_only_llm_fp4_cpp(model)
    assert result.SerializeToString() == model.SerializeToString()
