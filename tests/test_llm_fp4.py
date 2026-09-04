"""Tests for ``onnxsim.quantize_weight_only_llm_fp4`` (LLM-FP4, see
``onnxsim/llm_fp4.py``) -- block-wise quantization onto a searched
sign/exponent/mantissa FP4 format (bit split searched per tensor) with a
per-block *real-valued* scale (searched jointly, standing in for the
paper's own "pre-shifted exponent bias"), represented in the ONNX graph via
ordinary Gather/Reshape/Mul (no contrib op, no opset-21 features).

Per this repo's own platform-numerics lesson (onnxruntime's MatMul kernel
reduction order is not bit-exact across CPU architectures), value
correctness is checked by dequantizing the written initializers directly in
numpy and comparing against the original float weight with a tight
*relative* tolerance -- never by comparing onnxruntime outputs with an
absolute tolerance. Any onnxruntime round-trip check below is a separate,
much looser sanity check.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.llm_fp4 import FP4_FORMATS, _fp4_codebook, _fp4_magnitudes
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


def _codebook_used_by(model, w_name="W"):
    gather = next(
        n
        for n in model.graph.node
        if n.op_type == "Gather" and n.input[1] == f"{w_name}_llmfp4_codes_i64"
    )
    codebook_init = next(
        t for t in model.graph.initializer if t.name == gather.input[0]
    )
    return onnx.numpy_helper.to_array(codebook_init).astype(np.float64)


def _dequantize_llm_fp4_by_hand(model, w_name="W", block_size=32):
    """Independent reference decode: reads Wq/Ws/the winning codebook
    straight from the initializers and dequantizes via numpy, without using
    any of the ops this module inserts into the graph.
    """
    wq = next(t for t in model.graph.initializer if t.name == f"{w_name}_llmfp4_q")
    ws = next(t for t in model.graph.initializer if t.name == f"{w_name}_llmfp4_scale")
    codes = onnx.numpy_helper.to_array(wq).astype(np.int64)
    scale = onnx.numpy_helper.to_array(ws).astype(np.float64)
    codebook = _codebook_used_by(model, w_name)

    dim0, dim1 = codes.shape
    num_blocks = scale.shape[0]
    block_size_actual = dim0 // num_blocks
    assert block_size_actual == block_size
    values = codebook[codes]  # [dim0, dim1]
    scale_full = np.repeat(scale, block_size, axis=0)  # [dim0, dim1]
    return values * scale_full


def test_llm_fp4_quantizes_matmul_with_standard_ops_only():
    model = _matmul_model(K=64, N=16, seed=0)
    q = onnxsim.quantize_weight_only_llm_fp4(model, block_size=32)
    onnx.checker.check_model(q)

    op_types = {n.op_type for n in q.graph.node}
    assert op_types <= {"MatMul", "Cast", "Gather", "Reshape", "Mul"}
    assert all(n.domain in ("", "ai.onnx") for n in q.graph.node)


def test_llm_fp4_dequantized_values_match_independently_recomputed_nearest_codebook():
    # Unlike onnxsim.mx_quantization/onnxsim.nf4 (whose block scale always
    # keeps the block's own max-abs element within the codebook's range),
    # this module's own scale search can deliberately choose a *tighter*
    # scale that clips outliers in exchange for lower total MSE -- so a
    # fixed "within half a codebook gap" per-element bound (those other
    # modules' own test pattern) does not hold here. Instead: trust only
    # the search's chosen per-block *scale* (Ws), and independently
    # recompute -- via nearest-codebook-index search, entirely in numpy,
    # not using any op this module inserts into the graph -- what the
    # *codes* (Wq) should be for that scale. This still catches any bug in
    # either the code assignment or the Gather/Reshape/Mul dequantization
    # subgraph, without assuming anything about how tightly the search
    # itself clips.
    rng = np.random.default_rng(1)
    weight = rng.standard_normal((64, 16)).astype(np.float32) * 0.3
    model = _matmul_model(weight=weight)
    q = onnxsim.quantize_weight_only_llm_fp4(model, block_size=32)

    w_hand = _dequantize_llm_fp4_by_hand(q, block_size=32)
    codebook = _codebook_used_by(q)

    ws = next(t for t in q.graph.initializer if t.name == "W_llmfp4_scale")
    scale = onnx.numpy_helper.to_array(ws).astype(np.float64)
    scale_full = np.repeat(scale, 32, axis=0)

    normalized = weight.astype(np.float64) / scale_full
    diffs = np.abs(normalized[..., np.newaxis] - codebook[np.newaxis, np.newaxis, :])
    nearest_idx = np.argmin(diffs, axis=-1)
    independent_dequant = codebook[nearest_idx] * scale_full

    assert np.allclose(w_hand, independent_dequant, atol=1e-4, rtol=1e-4)

    # A loose sanity bound on overall reconstruction quality (this weight's
    # own measured relative L2 error is ~0.09) -- catches a search that
    # regressed to picking wildly bad scales/formats, without pinning an
    # exact number.
    rel_l2 = np.linalg.norm(w_hand - weight.astype(np.float64)) / np.linalg.norm(
        weight.astype(np.float64)
    )
    assert rel_l2 < 0.2


def test_llm_fp4_output_stays_close_to_float_via_onnxruntime():
    model = _matmul_model(K=64, N=16, seed=2)
    q = onnxsim.quantize_weight_only_llm_fp4(model, block_size=32)
    onnx.checker.check_model(q)

    rng = np.random.default_rng(3)
    x = rng.standard_normal((8, 64)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    # Loose, absolute-precision-agnostic sanity check only -- the tight,
    # authoritative correctness check is the numpy hand-decode above.
    assert _rel_l2(float_y, q_y) < 0.3


def test_llm_fp4_gemm_transb():
    rng = np.random.default_rng(4)
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
    q = onnxsim.quantize_weight_only_llm_fp4(model, block_size=32)
    onnx.checker.check_model(q)

    x = rng.standard_normal((4, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert _rel_l2(float_y, q_y) < 0.3


def test_llm_fp4_codes_stay_in_range():
    rng = np.random.default_rng(5)
    weight = rng.standard_normal((64, 8)).astype(np.float32) * 3
    model = _matmul_model(weight=weight)
    q = onnxsim.quantize_weight_only_llm_fp4(model, block_size=32)
    wq = next(t for t in q.graph.initializer if t.name == "W_llmfp4_q")
    codes = onnx.numpy_helper.to_array(wq)
    assert np.all(codes >= 0) and np.all(codes <= 15)


def test_llm_fp4_skips_non_block_divisible_k():
    model = _matmul_model(K=48, N=8, seed=6)  # 48 is not a multiple of 32
    q = onnxsim.quantize_weight_only_llm_fp4(model, block_size=32)
    assert q.SerializeToString() == model.SerializeToString()


def test_llm_fp4_skip_names_leaves_matched_weight_untouched():
    rng = np.random.default_rng(7)
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
    q = onnxsim.quantize_weight_only_llm_fp4(
        model, block_size=32, skip_names=["W_other"]
    )
    onnx.checker.check_model(q)

    names = {t.name for t in q.graph.initializer}
    assert "W_llmfp4_q" in names
    assert "W_other_llmfp4_q" not in names
    other_out = next(
        onnx.numpy_helper.to_array(t)
        for t in q.graph.initializer
        if t.name == "W_other"
    )
    assert np.array_equal(other_out, w_other)


def test_llm_fp4_skips_non_constant_weight():
    model = _model(
        """
        g (float[4,64] X, float[64,4] W) => (float[4,4] Y)
        {
          Y = MatMul(X, W)
        }
        """
    )
    q = onnxsim.quantize_weight_only_llm_fp4(model)
    assert q.SerializeToString() == model.SerializeToString()


def test_llm_fp4_rejects_unknown_format():
    model = _matmul_model()
    with pytest.raises(ValueError):
        onnxsim.quantize_weight_only_llm_fp4(model, formats=["not_a_format"])


def test_llm_fp4_e2m1_codebook_matches_mxfp4():
    # E2M1 is exactly MXFP4's own element format -- the two codebooks must
    # be identical sets of magnitudes (this module's own layout convention
    # matches onnxsim.mx_quantization's, so they should be byte-identical).
    e2m1 = np.asarray(_fp4_codebook(*FP4_FORMATS["e2m1"]))
    mxfp4 = np.asarray(MXFP4_CODEBOOK)
    assert np.array_equal(e2m1, mxfp4)


def test_llm_fp4_formats_have_eight_magnitudes_each():
    for e_bits, m_bits in FP4_FORMATS.values():
        magnitudes = _fp4_magnitudes(e_bits, m_bits)
        assert len(magnitudes) == 8
        assert magnitudes[0] == 0.0
        assert np.all(np.diff(magnitudes) > 0)  # strictly increasing


def test_llm_fp4_codebooks_are_well_formed():
    for fmt in FP4_FORMATS:
        e_bits, m_bits = FP4_FORMATS[fmt]
        codebook = np.asarray(_fp4_codebook(e_bits, m_bits))
        assert codebook.shape == (16,)
        assert np.all(np.diff(codebook) >= 0)  # non-decreasing (duplicate zero)
        assert list(codebook).count(0.0) == 2  # +0.0 and -0.0
        assert codebook[0] == -codebook[-1]  # symmetric


def test_llm_fp4_scale_is_not_restricted_to_power_of_two():
    # Unlike onnxsim.mx_quantization's MXFP4 (E8M0: power-of-two only), this
    # module's own per-block scale is real-valued -- the whole point of
    # realizing the paper's "pre-shifted exponent bias" as a per-block
    # scale search rather than reusing MX's narrower E8M0 restriction.
    rng = np.random.default_rng(9)
    weight = rng.standard_normal((64, 16)).astype(np.float32) * 3.7
    model = _matmul_model(weight=weight)
    q = onnxsim.quantize_weight_only_llm_fp4(model, block_size=32)

    ws = next(t for t in q.graph.initializer if t.name == "W_llmfp4_scale")
    scale = onnx.numpy_helper.to_array(ws).astype(np.float64).ravel()
    log2_scale = np.log2(scale)
    assert not np.all(np.abs(log2_scale - np.round(log2_scale)) < 1e-9)


def test_llm_fp4_format_search_beats_a_forced_single_format():
    # A weight tailor-made to reconstruct much better under E3M0 (wide
    # dynamic range, coarse mantissa -- octave-spaced magnitudes) than
    # E1M2 (narrow range, fine mantissa): the full search (default
    # `formats`) must find a reconstruction at least as good as forcing
    # E1M2 alone, and strictly better on this adversarial tensor.
    rng = np.random.default_rng(10)
    exponents = rng.integers(-6, 7, size=(64, 16))
    weight = (2.0**exponents).astype(np.float32) * rng.choice(
        [-1.0, 1.0], size=(64, 16)
    ).astype(np.float32)
    model = _matmul_model(weight=weight)

    q_search = onnxsim.quantize_weight_only_llm_fp4(model, block_size=32)
    q_e1m2 = onnxsim.quantize_weight_only_llm_fp4(
        model, block_size=32, formats=["e1m2"]
    )

    w_search = _dequantize_llm_fp4_by_hand(q_search, block_size=32)
    w_e1m2 = _dequantize_llm_fp4_by_hand(q_e1m2, block_size=32)
    w64 = weight.astype(np.float64)

    err_search = np.sum((w_search - w64) ** 2)
    err_e1m2 = np.sum((w_e1m2 - w64) ** 2)
    assert err_search <= err_e1m2
    assert err_search < err_e1m2 * 0.9  # strictly, meaningfully better


def test_llm_fp4_restricting_formats_only_uses_requested_ones():
    rng = np.random.default_rng(11)
    weight = rng.standard_normal((64, 16)).astype(np.float32) * 0.5
    model = _matmul_model(weight=weight)
    q = onnxsim.quantize_weight_only_llm_fp4(model, block_size=32, formats=["e3m0"])

    names = {t.name for t in q.graph.initializer}
    assert "llm_fp4_codebook_e3m0" in names
    assert "llm_fp4_codebook_e1m2" not in names
    assert "llm_fp4_codebook_e2m1" not in names
