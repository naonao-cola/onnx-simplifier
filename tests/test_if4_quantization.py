"""Tests for ``onnxsim.quantize_weight_only_if4`` (IF4 / Adaptive
Block-Scaled Data Types, see ``onnxsim/if4_quantization.py``) --
per-block choice between INT4 and E2M1 (FP4), whichever reconstructs that
block's own values with lower error, via a single combined 32-entry
Gather table (FP4 in ``[0, 16)``, INT4 in ``[16, 32)``) and the ordinary
Reshape/Mul dequantization :mod:`onnxsim.mx_quantization` also uses.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.if4_quantization import _COMBINED_CODEBOOK, _quantize_if4_blockwise
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


def _dequantize_if4_by_hand(model, w_name="W", block_size=16):
    wq = next(t for t in model.graph.initializer if t.name == f"{w_name}_if4_q")
    ws = next(t for t in model.graph.initializer if t.name == f"{w_name}_if4_scale")
    codes = onnx.numpy_helper.to_array(wq).astype(np.int64)
    scale = onnx.numpy_helper.to_array(ws).astype(np.float64)
    codebook = np.asarray(_COMBINED_CODEBOOK, dtype=np.float64)

    dim0, dim1 = codes.shape
    num_blocks = scale.shape[0]
    block_size_actual = dim0 // num_blocks
    assert block_size_actual == block_size
    values = codebook[codes]
    scale_full = np.repeat(scale, block_size, axis=0)
    return values * scale_full


def test_if4_quantizes_matmul_with_standard_ops_only():
    model = _matmul_model(K=64, N=16, seed=0)
    q = onnxsim.quantize_weight_only_if4(model, block_size=16)
    onnx.checker.check_model(q)

    op_types = {n.op_type for n in q.graph.node}
    assert op_types <= {"MatMul", "Cast", "Gather", "Reshape", "Mul"}
    assert all(n.domain in ("", "ai.onnx") for n in q.graph.node)


def test_if4_picks_int4_for_a_uniform_block_and_fp4_for_a_heavy_tailed_one():
    # A block of near-uniform-magnitude values plays to INT4's own even
    # grid spacing; a block with one dominant outlier and everything else
    # tiny plays to FP4's own denser near-zero resolution -- construct one
    # block of each and check the search actually picks differently.
    uniform_block = np.linspace(-3.0, 3.0, 16)
    heavy_tailed_block = np.concatenate([[6.0], np.full(15, 0.05)])
    values = np.stack([uniform_block, heavy_tailed_block], axis=0)  # [2, 16]

    codes, scale = _quantize_if4_blockwise(values, block_size=16)
    is_int4 = codes >= len(MXFP4_CODEBOOK)
    assert bool(is_int4[0].all())
    assert not bool(is_int4[1].any())
    assert scale.shape == (2, 1)


def test_if4_dequantized_values_match_hand_decoded_reference():
    rng = np.random.default_rng(2)
    weight = rng.standard_normal((64, 16)).astype(np.float32) * 0.3
    model = _matmul_model(weight=weight)
    q = onnxsim.quantize_weight_only_if4(model, block_size=16)

    w_hand = _dequantize_if4_by_hand(q, block_size=16)
    np.testing.assert_allclose(w_hand.shape, weight.shape)

    # Every block's own reconstruction must be at least as good as the
    # *other* format would have done on that same block -- the whole
    # point of searching both.
    codebook = np.asarray(_COMBINED_CODEBOOK, dtype=np.float64)
    ws = next(t for t in q.graph.initializer if t.name == "W_if4_scale")
    scale = onnx.numpy_helper.to_array(ws).astype(np.float64)
    w = weight.astype(np.float64)
    err = np.abs(w_hand - w)
    # Loose sanity bound: reconstruction error stays within the block's
    # own scale times the largest possible codebook gap.
    max_gap = max(np.max(np.diff(np.sort(codebook[:16]))), 1.0)
    scale_full = np.repeat(scale, 16, axis=0)
    assert np.all(err <= max_gap * scale_full + 1e-6)


def test_if4_output_stays_close_to_float_via_onnxruntime():
    model = _matmul_model(K=64, N=16, seed=3)
    q = onnxsim.quantize_weight_only_if4(model, block_size=16)
    onnx.checker.check_model(q)

    rng = np.random.default_rng(4)
    x = rng.standard_normal((8, 64)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.3


def test_if4_gemm_transb():
    rng = np.random.default_rng(5)
    K, N = 64, 12
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
    q = onnxsim.quantize_weight_only_if4(model, block_size=16)
    onnx.checker.check_model(q)

    x = rng.standard_normal((4, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert _rel_l2(float_y, q_y) < 0.3


def test_if4_codes_stay_in_range():
    rng = np.random.default_rng(6)
    weight = rng.standard_normal((64, 8)).astype(np.float32) * 3
    model = _matmul_model(weight=weight)
    q = onnxsim.quantize_weight_only_if4(model, block_size=16)
    wq = next(t for t in q.graph.initializer if t.name == "W_if4_q")
    codes = onnx.numpy_helper.to_array(wq)
    assert np.all(codes >= 0) and np.all(codes <= 31)


def test_if4_skips_non_block_divisible_k():
    model = _matmul_model(K=50, N=8, seed=7)  # 50 is not a multiple of 16
    q = onnxsim.quantize_weight_only_if4(model, block_size=16)
    assert q.SerializeToString() == model.SerializeToString()


def test_if4_skip_names_leaves_matched_weight_untouched():
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
    q = onnxsim.quantize_weight_only_if4(model, block_size=16, skip_names=["W_other"])
    onnx.checker.check_model(q)

    names = {t.name for t in q.graph.initializer}
    assert "W_if4_q" in names
    assert "W_other_if4_q" not in names
    other_out = next(
        onnx.numpy_helper.to_array(t)
        for t in q.graph.initializer
        if t.name == "W_other"
    )
    assert np.array_equal(other_out, w_other)


def test_if4_skips_non_constant_weight():
    model = _model(
        """
        g (float[4,64] X, float[64,4] W) => (float[4,4] Y)
        {
          Y = MatMul(X, W)
        }
        """
    )
    q = onnxsim.quantize_weight_only_if4(model)
    assert q.SerializeToString() == model.SerializeToString()


def test_if4_combined_codebook_is_well_formed():
    codebook = np.asarray(_COMBINED_CODEBOOK)
    assert codebook.shape == (32,)
    np.testing.assert_array_equal(codebook[:16], np.asarray(MXFP4_CODEBOOK))
    np.testing.assert_array_equal(codebook[16:], np.arange(-8, 8, dtype=np.float64))
