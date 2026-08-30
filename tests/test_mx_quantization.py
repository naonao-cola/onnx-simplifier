"""Tests for ``onnxsim.quantize_weight_only_mxfp4`` (OCP Microscaling
MXFP4, see ``onnxsim/mx_quantization.py``) -- block-wise quantization onto
E2M1's own fixed 16-value codebook with a per-block power-of-two scale,
represented in the ONNX graph via ordinary Gather/Reshape/Mul (no contrib
op, no opset-21 features).
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.mx_quantization import MXFP4_CODEBOOK

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


def _dequantize_mxfp4_by_hand(model, w_name="W", block_size=32):
    wq = next(t for t in model.graph.initializer if t.name == f"{w_name}_mxfp4_q")
    ws = next(t for t in model.graph.initializer if t.name == f"{w_name}_mxfp4_scale")
    codes = onnx.numpy_helper.to_array(wq).astype(np.int64)
    scale = onnx.numpy_helper.to_array(ws).astype(np.float64)
    codebook = np.asarray(MXFP4_CODEBOOK, dtype=np.float64)

    dim0, dim1 = codes.shape
    num_blocks = scale.shape[0]
    block_size_actual = dim0 // num_blocks
    assert block_size_actual == block_size
    values = codebook[codes]  # [dim0, dim1]
    scale_full = np.repeat(scale, block_size, axis=0)  # [dim0, dim1]
    return values * scale_full


def test_mxfp4_quantizes_matmul_with_standard_ops_only():
    model = _matmul_model(K=64, N=16, seed=0)
    q = onnxsim.quantize_weight_only_mxfp4(model, block_size=32)
    onnx.checker.check_model(q)

    op_types = {n.op_type for n in q.graph.node}
    assert op_types <= {"MatMul", "Cast", "Gather", "Reshape", "Mul"}
    assert all(n.domain in ("", "ai.onnx") for n in q.graph.node)


def test_mxfp4_block_scale_is_a_power_of_two():
    rng = np.random.default_rng(1)
    weight = rng.standard_normal((64, 16)).astype(np.float32) * 3.7
    model = _matmul_model(weight=weight)
    q = onnxsim.quantize_weight_only_mxfp4(model, block_size=32)

    ws = next(t for t in q.graph.initializer if t.name == "W_mxfp4_scale")
    scale = onnx.numpy_helper.to_array(ws).astype(np.float64).ravel()
    log2_scale = np.log2(scale)
    # A pure power of two has an exactly-integer base-2 logarithm.
    assert np.all(np.abs(log2_scale - np.round(log2_scale)) < 1e-9)


def test_mxfp4_dequantized_values_match_hand_decoded_reference():
    rng = np.random.default_rng(2)
    weight = rng.standard_normal((64, 16)).astype(np.float32) * 0.3
    model = _matmul_model(weight=weight)
    q = onnxsim.quantize_weight_only_mxfp4(model, block_size=32)

    w_hand = _dequantize_mxfp4_by_hand(q, block_size=32)

    # Every element's dequantization error must be within half the largest
    # codebook gap (scaled by that element's own block scale) -- a real
    # per-element correctness check, not just an aggregate error bound.
    codebook = np.asarray(MXFP4_CODEBOOK)
    max_gap = np.max(np.diff(codebook))
    ws = next(t for t in q.graph.initializer if t.name == "W_mxfp4_scale")
    scale = onnx.numpy_helper.to_array(ws).astype(np.float64)
    scale_full = np.repeat(scale, 32, axis=0)
    assert np.all(
        np.abs(w_hand - weight.astype(np.float64)) <= max_gap * scale_full / 2 + 1e-6
    )


def test_mxfp4_output_stays_close_to_float_via_onnxruntime():
    model = _matmul_model(K=64, N=16, seed=3)
    q = onnxsim.quantize_weight_only_mxfp4(model, block_size=32)
    onnx.checker.check_model(q)

    rng = np.random.default_rng(4)
    x = rng.standard_normal((8, 64)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.3


def test_mxfp4_gemm_transb():
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
    q = onnxsim.quantize_weight_only_mxfp4(model, block_size=32)
    onnx.checker.check_model(q)

    x = rng.standard_normal((4, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert _rel_l2(float_y, q_y) < 0.3


def test_mxfp4_codes_stay_in_range():
    rng = np.random.default_rng(6)
    weight = rng.standard_normal((64, 8)).astype(np.float32) * 3
    model = _matmul_model(weight=weight)
    q = onnxsim.quantize_weight_only_mxfp4(model, block_size=32)
    wq = next(t for t in q.graph.initializer if t.name == "W_mxfp4_q")
    codes = onnx.numpy_helper.to_array(wq)
    assert np.all(codes >= 0) and np.all(codes <= 15)


def test_mxfp4_skips_non_block_divisible_k():
    model = _matmul_model(K=48, N=8, seed=7)  # 48 is not a multiple of 32
    q = onnxsim.quantize_weight_only_mxfp4(model, block_size=32)
    assert q.SerializeToString() == model.SerializeToString()


def test_mxfp4_skip_names_leaves_matched_weight_untouched():
    rng = np.random.default_rng(8)
    w_base = rng.standard_normal((64, 16)).astype(np.float32) * 0.5
    w_other = rng.standard_normal((64, 4)).astype(np.float32) * 0.1
    model = _model(
        """
        g (float[batch,64] X) => (float[batch,16] Y, float[batch,4] H)
        {
          Y = MatMul(X, W)
          H = MatMul(X, W_other)
        }
        """,
        initializer=[_f32(w_base, "W"), _f32(w_other, "W_other")],
    )
    q = onnxsim.quantize_weight_only_mxfp4(model, block_size=32, skip_names=["W_other"])
    onnx.checker.check_model(q)

    names = {t.name for t in q.graph.initializer}
    assert "W_mxfp4_q" in names
    assert "W_other_mxfp4_q" not in names
    other_out = next(
        onnx.numpy_helper.to_array(t)
        for t in q.graph.initializer
        if t.name == "W_other"
    )
    assert np.array_equal(other_out, w_other)


def test_mxfp4_skips_non_constant_weight():
    model = _model(
        """
        g (float[4,64] X, float[64,4] W) => (float[4,4] Y)
        {
          Y = MatMul(X, W)
        }
        """
    )
    q = onnxsim.quantize_weight_only_mxfp4(model)
    assert q.SerializeToString() == model.SerializeToString()


def test_mxfp4_codebook_is_well_formed():
    codebook = np.asarray(MXFP4_CODEBOOK)
    assert codebook.shape == (16,)
    assert np.all(np.diff(codebook) >= 0)  # non-decreasing (E2M1 has two zeros)
    assert codebook[0] == -6.0 and codebook[-1] == 6.0
    assert list(codebook).count(0.0) == 2  # +0.0 and -0.0, distinct bit patterns
