"""Tests for ``onnxsim.apply_deepseek_fp8_cpp`` -- the C++-backed port of
just the *weight* half of ``onnxsim.apply_deepseek_fp8`` (see
``onnxsim/passes/deepseek_fp8.h``). ``apply_deepseek_fp8`` is a W8A8
scheme: it also inserts new graph nodes to block-quantize the activation
at run time. This port is explicitly out of scope for that half -- it
only reproduces the weight-side 128x128-block FP8 E4M3 round trip, folded
directly into a replaced initializer, matching every other weight-only
``*_cpp`` port in this repo. Unlike this repo's GGUF-family ports, both
directions here are a real, fully-specified FLOAT8E4M3FN cast (not an
honestly-scoped heuristic), so this port is expected to track
``quantize_dequantize_block_fp8`` unusually closely -- checked directly
here, not just via structural/algebraic properties.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.deepseek_fp8 import quantize_dequantize_block_fp8

ort = pytest.importorskip("onnxruntime")

_BLOCK_SIZE = 128


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


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


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


def _current_weight(model, weight_input_index=1):
    node = next(n for n in model.graph.node if n.op_type in ("MatMul", "Gemm"))
    w_name = node.input[weight_input_index]
    w_init = next(t for t in model.graph.initializer if t.name == w_name)
    return onnx.numpy_helper.to_array(w_init)


def _python_weight_side_reference(w, weight_transposed):
    # Mirrors apply_deepseek_fp8's own orientation handling: quantize the
    # weight's logical [N, K] (output-channel-first) view, then transpose
    # back if the on-disk weight isn't already stored that way.
    w64 = w.astype(np.float64)
    w_nk = w64 if weight_transposed else w64.T
    q_nk = quantize_dequantize_block_fp8(w_nk, _BLOCK_SIZE)
    return q_nk if weight_transposed else q_nk.T


def test_cpp_replaces_weight_with_same_shape_float():
    rng = np.random.default_rng(0)
    K, N = _BLOCK_SIZE * 2, 8
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(w, K, N)

    q = onnxsim.apply_deepseek_fp8_cpp(model)
    onnx.checker.check_model(q)
    new_w = _current_weight(q)
    assert new_w.shape == w.shape
    assert new_w.dtype == np.float32
    assert not np.array_equal(new_w, w)
    # The original initializer is left in the graph, unused -- matching
    # every other weight-only *_cpp port's established convention.
    assert any(t.name == "W" for t in q.graph.initializer)


def test_cpp_matches_python_weight_side_reference_exactly_matmul():
    # Both directions of this format are a real, fully-specified
    # FLOAT8E4M3FN cast (unlike the GGUF family's honestly-scoped
    # heuristics), so this is checked for close-to-exact agreement, not
    # just comparable reconstruction error.
    rng = np.random.default_rng(1)
    K, N = _BLOCK_SIZE * 2, 24
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.7
    model = _matmul_model(w, K, N)

    q = onnxsim.apply_deepseek_fp8_cpp(model)
    cpp_w = _current_weight(q).astype(np.float64)
    py_w = _python_weight_side_reference(w, weight_transposed=False)

    np.testing.assert_allclose(cpp_w, py_w, rtol=1e-5, atol=1e-6)


def test_cpp_matches_python_weight_side_reference_exactly_gemm_transb():
    rng = np.random.default_rng(2)
    K, N = _BLOCK_SIZE + 32, _BLOCK_SIZE * 2 + 16
    weight = rng.standard_normal((N, K)).astype(np.float32) * 0.3  # [N, K]
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm<transB = 1>(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )

    q = onnxsim.apply_deepseek_fp8_cpp(model)
    onnx.checker.check_model(q)
    cpp_w = _current_weight(q).astype(np.float64)
    py_w = _python_weight_side_reference(weight, weight_transposed=True)

    np.testing.assert_allclose(cpp_w, py_w, rtol=1e-5, atol=1e-6)


def test_cpp_per_block_max_abs_survives_within_one_fp8_step():
    # A block's own largest-magnitude element maps to code +-448 (the
    # scale is defined so the block's own max hits FLOAT8E4M3FN's largest
    # finite value exactly) -- so the reconstructed max-abs of every
    # block should closely track the original, not be crushed by a
    # neighboring block's much larger scale.
    rng = np.random.default_rng(3)
    K, N = _BLOCK_SIZE * 2, _BLOCK_SIZE
    w = rng.standard_normal((K, N)).astype(np.float32)
    w[: _BLOCK_SIZE // 2, : _BLOCK_SIZE // 2] *= 100.0  # one large-scale block
    model = _matmul_model(w, K, N)

    q = onnxsim.apply_deepseek_fp8_cpp(model)
    new_w = _current_weight(q).astype(np.float64)
    w64 = w.astype(np.float64)

    for k0 in range(0, K, _BLOCK_SIZE):
        for n0 in range(0, N, _BLOCK_SIZE):
            k1, n1 = min(k0 + _BLOCK_SIZE, K), min(n0 + _BLOCK_SIZE, N)
            orig_max = np.abs(w64[k0:k1, n0:n1]).max()
            recon_max = np.abs(new_w[k0:k1, n0:n1]).max()
            assert recon_max > orig_max * 0.9


def test_cpp_gemm_with_bias():
    rng = np.random.default_rng(5)
    K, N = _BLOCK_SIZE * 2, 16
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
    q = onnxsim.apply_deepseek_fp8_cpp(model)
    onnx.checker.check_model(q)

    x = rng.standard_normal((4, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.2


def test_cpp_ragged_last_block_matches_python_zero_padding():
    # Neither K nor N a multiple of 128 -- deepseek_fp8.h's own scope note
    # claims this ragged-last-block approach is mathematically identical
    # to deepseek_fp8.py's zero-pad-then-discard one, since padding zeros
    # can never change a block's own max(|.|). Checked directly here.
    rng = np.random.default_rng(6)
    K, N = _BLOCK_SIZE + 5, _BLOCK_SIZE + 40
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.4
    model = _matmul_model(w, K, N)

    q = onnxsim.apply_deepseek_fp8_cpp(model)
    cpp_w = _current_weight(q).astype(np.float64)
    py_w = _python_weight_side_reference(w, weight_transposed=False)

    np.testing.assert_allclose(cpp_w, py_w, rtol=1e-5, atol=1e-6)


def test_cpp_noop_when_no_matmul_gemm_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_deepseek_fp8_cpp(model)
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
    result = onnxsim.apply_deepseek_fp8_cpp(model)
    assert result.SerializeToString() == model.SerializeToString()
