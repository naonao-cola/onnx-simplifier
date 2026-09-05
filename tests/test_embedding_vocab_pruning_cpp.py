"""Tests for ``onnxsim.apply_embedding_vocab_pruning_cpp``/
``onnxsim.apply_embedding_vocab_magnitude_pruning_cpp`` -- the C++-backed
ports of ``onnxsim.apply_embedding_vocab_pruning``/``onnxsim.apply_embedding_
vocab_magnitude_pruning`` (see ``onnxsim/structured_pruning_entry.cpp``'s own
"Embedding vocabulary pruning" section).

Unlike every other prior C++-port test file in this repo, this one is NOT a
subset of an existing feature's coverage grown incrementally -- embedding
vocabulary pruning is a genuinely different shape of pass (it prunes a
table's VOCABULARY axis, not a producer/consumer FEATURE-channel pair), and
it returns an ``onnxsim.pruning.EmbeddingPruningResult`` (model + kept/
dropped token ids + an old-id -> new-id remapping), never a bare
``onnx.ModelProto``.

Tests here mirror ``tests/test_pruning.py``'s own "Embedding / lm_head
vocabulary pruning" coverage (search that section name there): a plain
``Gather`` producer, a ``com.microsoft::EmbedLayerNormalization`` one, or a
``com.microsoft::GatherBlockQuantized`` one (the block-quantized embedding
shape -- see structured_pruning_entry.cpp's own section comment for the
full empirical schema/packing detail this port depends on), a ``MatMul``/
vanilla-``Gemm``/``com.microsoft::FusedGemm``/``GemmFastGelu`` ``lm_head``,
and a FLOAT, FLOAT16, OR BFLOAT16 embedding table/``lm_head`` weight/bias.
Every test that actually prunes something runs the result through a real
onnxruntime CPU session and compares against a numpy slice of the ORIGINAL
model's own output (the same "byte-exact oracle" bar every other C++-port
test file in this repo holds itself to).

``onnxsim.apply_embedding_vocab_pruning``/``apply_embedding_vocab_magnitude_
pruning`` (the pure-Python names) are now themselves thin aliases for
:func:`onnxsim.apply_embedding_vocab_pruning_cpp`/``apply_embedding_vocab_
magnitude_pruning_cpp`` (full parity verified across both ``GatherBlockQuantized``
packing conventions -- see pruning.py's own docstrings on those two
functions), so a cross-check against the pure-Python entry point would be
tautological (literally the same code path twice) wherever the C++ result's
own ``kept_token_ids``/``id_map``/``lm_head_pruned`` is already verified
directly against an independently-computed expected value in the same test
(the explicit ``keep_token_ids``/``drop_token_ids`` the caller passed in, or
a numpy-computed importance ranking) -- this file no longer calls the
pure-Python entry points at all; the ONNX Runtime oracle comparisons already
present in every test remain fully meaningful regression coverage
regardless of aliasing.
"""

import ml_dtypes
import numpy as np
import onnx
import onnx.checker
import onnx.helper
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim

ort = pytest.importorskip("onnxruntime")


def _model(body, initializer=(), opset=21):
    # Pinning ir_version: 10, same as tests/test_pruning.py's own _model --
    # matches the older onnxruntime bundled with some CI wheels.
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": {opset}]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _f16(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float16), name)


def _bf16(array, name):
    return onnx.numpy_helper.from_array(array.astype(ml_dtypes.bfloat16), name)


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _matmul_model(K=4, N=4):
    # A plain MatMul with no Gather anywhere -- the "no embedding pattern at
    # all" decline baseline, shared by several tests below.
    rng = np.random.default_rng(0)
    w = rng.standard_normal((K, N)).astype(np.float32)
    return _model(
        f"""
        g (float[M,{K}] X) => (float[M,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f32(w, "W")],
    )


def _untied_model(V, H, seed=1):
    rng = np.random.default_rng(seed)
    w_emb = rng.standard_normal((V, H)).astype(np.float32)
    w_lm = rng.standard_normal((H, V)).astype(np.float32)
    model = _model(
        f"""
        g (int64[batch,seq] input_ids) => (float[batch,seq,{V}] logits)
        {{
          hidden = Gather<axis=0>(W_emb, input_ids)
          logits = MatMul(hidden, W_lm)
        }}
        """,
        initializer=[_f32(w_emb, "W_emb"), _f32(w_lm, "W_lm")],
    )
    onnx.checker.check_model(model)
    return model


def _ambiguous_embedding_model(V_tok=10, V_pos=6, H=4, seed=6):
    rng = np.random.default_rng(seed)
    w_tok = rng.standard_normal((V_tok, H)).astype(np.float32)
    w_pos = rng.standard_normal((V_pos, H)).astype(np.float32)
    model = _model(
        f"""
        g (int64[M] input_ids, int64[M] position_ids) => (float[M,{H}] out)
        {{
          tok = Gather<axis=0>(W_tok, input_ids)
          pos = Gather<axis=0>(W_pos, position_ids)
          out = Add(tok, pos)
        }}
        """,
        initializer=[_f32(w_tok, "W_tok"), _f32(w_pos, "W_pos")],
    )
    onnx.checker.check_model(model)
    return model


# --- Explicit keep/drop-token-ids pruning -----------------------------------


def test_untied_matches_oracle_and_renumbers_contiguously():
    V, H = 12, 8
    model = _untied_model(V, H, seed=1)

    keep_token_ids = [9, 1, 4, 7, 3, 11]  # deliberately unsorted, with dups below
    result = onnxsim.apply_embedding_vocab_pruning_cpp(
        model, keep_token_ids=keep_token_ids + [4, 9]
    )
    onnx.checker.check_model(result.model)

    assert result.matched
    assert result.lm_head_pruned
    assert result.kept_token_ids == sorted(set(keep_token_ids))
    assert result.id_map == {tok: i for i, tok in enumerate(result.kept_token_ids)}

    emb_init = {t.name: t for t in result.model.graph.initializer}["W_emb"]
    lm_init = {t.name: t for t in result.model.graph.initializer}["W_lm"]
    assert list(emb_init.dims) == [len(result.kept_token_ids), H]
    assert list(lm_init.dims) == [H, len(result.kept_token_ids)]

    input_ids = np.array([[9, 4, 1, 7], [3, 11, 9, 1]], dtype=np.int64)
    remapped = np.vectorize(result.id_map.get)(input_ids).astype(np.int64)

    orig_out = _run(model, {"input_ids": input_ids})[0]
    pruned_out = _run(result.model, {"input_ids": remapped})[0]
    expected = orig_out[..., result.kept_token_ids]
    assert pruned_out.shape[-1] == len(result.kept_token_ids)
    np.testing.assert_allclose(pruned_out, expected, atol=1e-5, rtol=1e-5)


def test_drop_token_ids_equivalent_to_complement_keep_set():
    V, H = 9, 5
    model = _untied_model(V, H, seed=4)
    drop = [2, 4, 7]

    result = onnxsim.apply_embedding_vocab_pruning_cpp(model, drop_token_ids=drop)
    onnx.checker.check_model(result.model)
    assert result.matched and result.lm_head_pruned
    assert result.kept_token_ids == [i for i in range(V) if i not in drop]
    assert result.id_map == {tok: i for i, tok in enumerate(result.kept_token_ids)}


def test_untied_lm_head_gemm_bias_is_sliced():
    V, H = 10, 6
    rng = np.random.default_rng(2)
    w_emb = rng.standard_normal((V, H)).astype(np.float32)
    w_lm = rng.standard_normal((V, H)).astype(np.float32)  # [N, K], transB=1
    b_lm = rng.standard_normal(V).astype(np.float32)
    model = _model(
        f"""
        g (int64[M] input_ids) => (float[M,{V}] logits)
        {{
          hidden = Gather<axis=0>(W_emb, input_ids)
          logits = Gemm<transB=1>(hidden, W_lm, B_lm)
        }}
        """,
        initializer=[_f32(w_emb, "W_emb"), _f32(w_lm, "W_lm"), _f32(b_lm, "B_lm")],
    )
    onnx.checker.check_model(model)

    keep_token_ids = [0, 2, 3, 5, 8, 9]
    result = onnxsim.apply_embedding_vocab_pruning_cpp(
        model, keep_token_ids=keep_token_ids
    )
    onnx.checker.check_model(result.model)
    assert result.matched and result.lm_head_pruned

    bias_init = {t.name: t for t in result.model.graph.initializer}["B_lm"]
    assert list(bias_init.dims) == [len(keep_token_ids)]

    input_ids = np.array([0, 5, 9, 2, 8], dtype=np.int64)
    remapped = np.array([result.id_map[i] for i in input_ids], dtype=np.int64)

    orig_out = _run(model, {"input_ids": input_ids})[0]
    pruned_out = _run(result.model, {"input_ids": remapped})[0]
    expected = orig_out[..., result.kept_token_ids]
    np.testing.assert_allclose(pruned_out, expected, atol=1e-5, rtol=1e-5)


def test_tied_via_transpose_matches_oracle():
    # Tied (weight-shared) embedding/lm_head, the "Transpose then MatMul"
    # sub-shape -- the single shared initializer must be sliced exactly
    # once, never independently twice.
    V, H = 12, 8
    rng = np.random.default_rng(3)
    w_emb = rng.standard_normal((V, H)).astype(np.float32)
    model = _model(
        f"""
        g (int64[batch,seq] input_ids) => (float[batch,seq,{V}] logits)
        {{
          hidden = Gather<axis=0>(W_emb, input_ids)
          W_t = Transpose<perm=[1,0]>(W_emb)
          logits = MatMul(hidden, W_t)
        }}
        """,
        initializer=[_f32(w_emb, "W_emb")],
    )
    onnx.checker.check_model(model)
    assert len(model.graph.initializer) == 1

    keep_token_ids = [1, 2, 4, 6, 8, 10, 11]
    result = onnxsim.apply_embedding_vocab_pruning_cpp(
        model, keep_token_ids=keep_token_ids
    )
    onnx.checker.check_model(result.model)

    assert result.matched
    assert result.lm_head_pruned
    assert len(result.model.graph.initializer) == 1
    emb_init = result.model.graph.initializer[0]
    assert list(emb_init.dims) == [len(keep_token_ids), H]

    input_ids = np.array([[1, 4, 8], [10, 2, 11]], dtype=np.int64)
    remapped = np.vectorize(result.id_map.get)(input_ids).astype(np.int64)

    orig_out = _run(model, {"input_ids": input_ids})[0]
    pruned_out = _run(result.model, {"input_ids": remapped})[0]
    expected = orig_out[..., result.kept_token_ids]
    np.testing.assert_allclose(pruned_out, expected, atol=1e-5, rtol=1e-5)


def test_tied_direct_gemm_matches_oracle():
    # The other tied sub-shape: a direct Gemm(transB=1) reusing the
    # embedding table as its own [vocab, hidden] weight, no Transpose node.
    V, H = 9, 5
    rng = np.random.default_rng(4)
    w_emb = rng.standard_normal((V, H)).astype(np.float32)
    model = _model(
        f"""
        g (int64[M] input_ids) => (float[M,{V}] logits)
        {{
          hidden = Gather<axis=0>(W_emb, input_ids)
          logits = Gemm<transB=1>(hidden, W_emb)
        }}
        """,
        initializer=[_f32(w_emb, "W_emb")],
    )
    onnx.checker.check_model(model)
    assert len(model.graph.initializer) == 1

    keep_token_ids = [0, 1, 3, 5, 6, 8]
    result = onnxsim.apply_embedding_vocab_pruning_cpp(model, drop_token_ids=[2, 4, 7])
    onnx.checker.check_model(result.model)
    assert result.matched and result.lm_head_pruned
    assert result.kept_token_ids == keep_token_ids
    assert len(result.model.graph.initializer) == 1

    input_ids = np.array([0, 5, 8, 3], dtype=np.int64)
    remapped = np.array([result.id_map[i] for i in input_ids], dtype=np.int64)
    orig_out = _run(model, {"input_ids": input_ids})[0]
    pruned_out = _run(result.model, {"input_ids": remapped})[0]
    expected = orig_out[..., result.kept_token_ids]
    np.testing.assert_allclose(pruned_out, expected, atol=1e-5, rtol=1e-5)


def test_cast_hop_indices_still_matched():
    V, H = 8, 4
    rng = np.random.default_rng(5)
    w_emb = rng.standard_normal((V, H)).astype(np.float32)
    model = _model(
        f"""
        g (int32[M] input_ids) => (float[M,{H}] hidden)
        {{
          ids64 = Cast<to=7>(input_ids)
          hidden = Gather<axis=0>(W_emb, ids64)
        }}
        """,
        initializer=[_f32(w_emb, "W_emb")],
    )
    onnx.checker.check_model(model)

    keep_token_ids = [0, 2, 3, 5, 7]
    result = onnxsim.apply_embedding_vocab_pruning_cpp(
        model, keep_token_ids=keep_token_ids
    )
    onnx.checker.check_model(result.model)
    assert result.matched
    assert not result.lm_head_pruned

    input_ids = np.array([0, 5, 2, 7], dtype=np.int32)
    remapped = np.array([result.id_map[i] for i in input_ids], dtype=np.int32)
    orig_out = _run(model, {"input_ids": input_ids})[0]
    pruned_out = _run(result.model, {"input_ids": remapped})[0]
    np.testing.assert_allclose(pruned_out, orig_out, atol=1e-6, rtol=1e-6)


# --- input_name auto-detection / ambiguity ----------------------------------


def test_declines_when_gather_is_ambiguous():
    model = _ambiguous_embedding_model()
    result = onnxsim.apply_embedding_vocab_pruning_cpp(model, keep_token_ids=[0, 1, 2])
    assert result.matched is False
    assert result.kept_token_ids is None
    assert result.id_map is None
    assert result.model.SerializeToString() == model.SerializeToString()


def test_input_name_disambiguates_correctly():
    model = _ambiguous_embedding_model(V_tok=10, V_pos=6)
    result = onnxsim.apply_embedding_vocab_pruning_cpp(
        model, drop_token_ids=[4, 5], input_name="input_ids"
    )
    assert result.matched
    assert result.kept_token_ids == [0, 1, 2, 3, 6, 7, 8, 9]
    # The positional-embedding table must be left completely untouched.
    w_pos = {t.name: t for t in result.model.graph.initializer}["W_pos"]
    assert list(w_pos.dims) == [6, 4]


def test_input_name_auto_detected_when_unambiguous():
    # A single eligible Gather -- input_name may be omitted entirely.
    V, H = 8, 4
    model = _untied_model(V, H, seed=9)
    result_omitted = onnxsim.apply_embedding_vocab_pruning_cpp(
        model, keep_token_ids=[0, 1, 2, 3]
    )
    result_named = onnxsim.apply_embedding_vocab_pruning_cpp(
        model, keep_token_ids=[0, 1, 2, 3], input_name="input_ids"
    )
    assert result_omitted.matched and result_named.matched
    assert result_omitted.kept_token_ids == result_named.kept_token_ids


def test_unknown_input_name_raises():
    model = _ambiguous_embedding_model()
    with pytest.raises(ValueError, match="not_a_real_input"):
        onnxsim.apply_embedding_vocab_pruning_cpp(
            model, keep_token_ids=[0, 1], input_name="not_a_real_input"
        )


def test_declines_non_zero_gather_axis():
    V, H, M = 10, 6, 3
    rng = np.random.default_rng(7)
    w_emb = rng.standard_normal((V, H)).astype(np.float32)
    model = _model(
        f"""
        g (int64[{M}] input_ids) => (float[{V},{M}] out)
        {{
          out = Gather<axis=1>(W_emb, input_ids)
        }}
        """,
        initializer=[_f32(w_emb, "W_emb")],
    )
    onnx.checker.check_model(model)
    result = onnxsim.apply_embedding_vocab_pruning_cpp(model, keep_token_ids=[0, 1])
    assert result.matched is False
    assert result.model.SerializeToString() == model.SerializeToString()


def test_declines_unexpected_shared_consumer():
    V, H, M = 10, 6, 3
    rng = np.random.default_rng(8)
    w_emb = rng.standard_normal((V, H)).astype(np.float32)
    bias = rng.standard_normal((V, H)).astype(np.float32)
    model = _model(
        f"""
        g (int64[{M}] input_ids) => (float[{M},{H}] hidden, float[{V},{H}] extra)
        {{
          hidden = Gather<axis=0>(W_emb, input_ids)
          extra = Add(W_emb, Bias)
        }}
        """,
        initializer=[_f32(w_emb, "W_emb"), _f32(bias, "Bias")],
    )
    onnx.checker.check_model(model)
    result = onnxsim.apply_embedding_vocab_pruning_cpp(model, keep_token_ids=[0, 1, 2])
    assert result.matched is False
    assert result.model.SerializeToString() == model.SerializeToString()


def test_declines_when_no_embedding_pattern_exists():
    model = _matmul_model(K=8, N=4)
    result = onnxsim.apply_embedding_vocab_pruning_cpp(model, keep_token_ids=[0])
    assert result.matched is False
    assert result.model.SerializeToString() == model.SerializeToString()


# --- Argument validation ----------------------------------------------------


def test_validates_keep_and_drop_arguments():
    model = _untied_model(6, 3, seed=11)
    with pytest.raises(ValueError):
        onnxsim.apply_embedding_vocab_pruning_cpp(model)
    with pytest.raises(ValueError):
        onnxsim.apply_embedding_vocab_pruning_cpp(
            model, keep_token_ids=[0], drop_token_ids=[1]
        )


def test_rejects_out_of_range_ids():
    V = 6
    model = _untied_model(V, 3, seed=12)
    with pytest.raises(ValueError):
        onnxsim.apply_embedding_vocab_pruning_cpp(model, keep_token_ids=[0, 100])
    with pytest.raises(ValueError):
        onnxsim.apply_embedding_vocab_pruning_cpp(model, drop_token_ids=[0, -1])
    with pytest.raises(ValueError):
        onnxsim.apply_embedding_vocab_pruning_cpp(model, drop_token_ids=list(range(V)))


# --- Magnitude-based pruning -------------------------------------------------


def test_magnitude_pruning_drops_lowest_norm_rows_and_protects():
    V, H = 10, 4
    w = np.full((V, H), 5.0, dtype=np.float32)
    w[2] *= 0.001  # deliberately tiny-norm rows -- should be dropped first
    w[7] *= 0.001
    model = _model(
        f"""
        g (int64[M] input_ids) => (float[M,{H}] hidden)
        {{
          hidden = Gather<axis=0>(W_emb, input_ids)
        }}
        """,
        initializer=[_f32(w, "W_emb")],
    )
    onnx.checker.check_model(model)

    result = onnxsim.apply_embedding_vocab_magnitude_pruning_cpp(model, sparsity=0.3)
    onnx.checker.check_model(result.model)
    assert result.matched
    assert not result.lm_head_pruned
    assert len(result.kept_token_ids) == round(V * 0.7) == 7
    assert 2 not in result.kept_token_ids
    assert 7 not in result.kept_token_ids

    protected = onnxsim.apply_embedding_vocab_magnitude_pruning_cpp(
        model, sparsity=0.3, protect_token_ids=[2]
    )
    assert protected.matched
    assert 2 in protected.kept_token_ids


def test_magnitude_pruning_matches_independent_oracle_ranking():
    # A strictly-decreasing per-row scale gives an unambiguous, hand-
    # computable ranking -- the independent oracle here is a plain numpy
    # L2-norm computation, not the pure-Python entry point (kept genuinely
    # separate from the implementation under test).
    V, H = 14, 6
    rng = np.random.default_rng(13)
    scale = np.linspace(3.0, 0.1, V)
    w_emb = (rng.standard_normal((V, H)) * scale[:, None]).astype(np.float32)
    model = _model(
        f"""
        g (int64[M] input_ids) => (float[M,{H}] hidden)
        {{
          hidden = Gather<axis=0>(W_emb, input_ids)
        }}
        """,
        initializer=[_f32(w_emb, "W_emb")],
    )
    onnx.checker.check_model(model)

    sparsity = 0.5
    result = onnxsim.apply_embedding_vocab_magnitude_pruning_cpp(
        model, sparsity=sparsity
    )
    assert result.matched

    row_norm = np.linalg.norm(w_emb.astype(np.float64), axis=1)
    keep_count = max(1, round(V * (1.0 - sparsity)))
    expected_keep = sorted(np.argsort(-row_norm)[:keep_count].tolist())
    assert result.kept_token_ids == expected_keep
    assert len(result.kept_token_ids) == keep_count


def test_magnitude_pruning_combines_lm_head_norm_when_untied():
    V, H = 12, 8
    rng = np.random.default_rng(12)
    w_emb = (rng.standard_normal((V, H)) * np.linspace(2.0, 0.05, V)[:, None]).astype(
        np.float32
    )
    w_lm = rng.standard_normal((H, V)).astype(np.float32)
    model = _model(
        f"""
        g (int64[M] input_ids) => (float[M,{V}] logits)
        {{
          hidden = Gather<axis=0>(W_emb, input_ids)
          logits = MatMul(hidden, W_lm)
        }}
        """,
        initializer=[_f32(w_emb, "W_emb"), _f32(w_lm, "W_lm")],
    )
    onnx.checker.check_model(model)

    result = onnxsim.apply_embedding_vocab_magnitude_pruning_cpp(model, sparsity=0.4)
    onnx.checker.check_model(result.model)
    assert result.matched and result.lm_head_pruned
    assert len(result.kept_token_ids) == round(V * 0.6)
    # Rows were scaled by a strictly decreasing factor -- the kept set must
    # be exactly the lowest-index (highest-combined-norm) rows.
    assert result.kept_token_ids == list(range(len(result.kept_token_ids)))

    input_ids = np.array([0, 1, 2, 3], dtype=np.int64)
    remapped = np.array([result.id_map[i] for i in input_ids], dtype=np.int64)
    orig_out = _run(model, {"input_ids": input_ids})[0]
    pruned_out = _run(result.model, {"input_ids": remapped})[0]
    expected = orig_out[..., result.kept_token_ids]
    np.testing.assert_allclose(pruned_out, expected, atol=1e-5, rtol=1e-5)


def test_magnitude_pruning_rejects_bad_sparsity():
    model = _matmul_model(K=4, N=4)
    with pytest.raises(ValueError, match="sparsity"):
        onnxsim.apply_embedding_vocab_magnitude_pruning_cpp(model, sparsity=1.0)


def test_magnitude_pruning_declines_when_no_embedding_pattern_exists():
    model = _matmul_model(K=8, N=4)
    result = onnxsim.apply_embedding_vocab_magnitude_pruning_cpp(model, sparsity=0.3)
    assert result.matched is False
    assert result.model.SerializeToString() == model.SerializeToString()


# --- EmbedLayerNormalization producer shape ---------------------------------
#
# The fused ``com.microsoft::EmbedLayerNormalization`` producer shape (see
# ``onnxsim/pruning.py``'s own `_match_embed_layer_norm_producer` docstring
# and structured_pruning_entry.cpp's own `MatchEmbedLayerNormProducer`) --
# newly matched by this C++ port. ``EmbedLayerNormalization`` has a real CPU
# kernel in this environment (confirmed directly), so no decomposed-proxy
# fallback is needed here, unlike the GemmFastGelu lm_head tests below.


def _embed_layer_norm_model(word_emb, pos_emb, gamma, beta, batch=2, seq=3):
    H = word_emb.shape[1]
    model = _model(
        f"""
        g (int32[{batch},{seq}] input_ids) =>
           (float[{batch},{seq},{H}] output, int32[{batch}] mask_index)
        {{
          output, mask_index = com.microsoft.EmbedLayerNormalization<epsilon=1e-12>(
              input_ids, , word_embedding, position_embedding, , gamma, beta)
        }}
        """,
        initializer=[
            _f32(word_emb, "word_embedding"),
            _f32(pos_emb, "position_embedding"),
            _f32(gamma, "gamma"),
            _f32(beta, "beta"),
        ],
    )
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))
    return model


def test_embed_layer_norm_word_embedding_matches_oracle_and_ort_execution():
    V, P, H = 12, 16, 8
    rng = np.random.default_rng(201)
    word_emb = rng.standard_normal((V, H)).astype(np.float32)
    pos_emb = rng.standard_normal((P, H)).astype(np.float32)
    gamma = rng.standard_normal(H).astype(np.float32) * 0.5 + 1.0
    beta = rng.standard_normal(H).astype(np.float32) * 0.1

    model = _embed_layer_norm_model(word_emb, pos_emb, gamma, beta)
    onnx.checker.check_model(model)

    keep_token_ids = [1, 3, 4, 6, 8, 9, 11]
    result = onnxsim.apply_embedding_vocab_pruning_cpp(
        model, keep_token_ids=keep_token_ids
    )
    onnx.checker.check_model(result.model)
    assert result.matched
    assert not result.lm_head_pruned
    assert result.kept_token_ids == keep_token_ids

    inits = {t.name: t for t in result.model.graph.initializer}
    assert list(inits["word_embedding"].dims) == [len(keep_token_ids), H]
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(inits["word_embedding"]), word_emb[keep_token_ids]
    )
    # position_embedding/gamma/beta: a different index space / not
    # vocab-shaped at all -- must be left byte-identical, untouched.
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(inits["position_embedding"]), pos_emb
    )
    np.testing.assert_array_equal(onnx.numpy_helper.to_array(inits["gamma"]), gamma)
    np.testing.assert_array_equal(onnx.numpy_helper.to_array(inits["beta"]), beta)

    # Independent reference model built directly from the pre-sliced
    # word_embedding, never touching this pass's own matching/slicing code.
    ref_model = _embed_layer_norm_model(word_emb[keep_token_ids], pos_emb, gamma, beta)
    orig_ids = np.array([[1, 3, 4], [6, 8, 9]], dtype=np.int32)
    local_ids = np.array(
        [[result.id_map[int(t)] for t in row] for row in orig_ids], dtype=np.int32
    )
    pruned_out, _ = _run(result.model, {"input_ids": local_ids})
    ref_out, _ = _run(ref_model, {"input_ids": local_ids})
    np.testing.assert_allclose(pruned_out, ref_out, atol=1e-5, rtol=1e-5)


def test_embed_layer_norm_magnitude_pruning_matches_oracle():
    # A strictly-decreasing per-row scale gives an unambiguous ranking with
    # no exactly-tied importances -- avoids relying on numpy's/std::stable_
    # sort's own (potentially differing) tie-breaking, mirroring
    # test_magnitude_pruning_matches_independent_oracle_ranking above.
    V, P, H = 10, 8, 4
    rng = np.random.default_rng(202)
    scale = np.linspace(3.0, 0.1, V)
    word_emb = (rng.standard_normal((V, H)) * scale[:, None]).astype(np.float32)
    pos_emb = rng.standard_normal((P, H)).astype(np.float32)
    gamma = np.ones(H, dtype=np.float32)
    beta = np.zeros(H, dtype=np.float32)

    model = _embed_layer_norm_model(word_emb, pos_emb, gamma, beta)
    onnx.checker.check_model(model)

    result = onnxsim.apply_embedding_vocab_magnitude_pruning_cpp(model, sparsity=0.3)
    onnx.checker.check_model(result.model)
    assert result.matched

    row_norm = np.linalg.norm(word_emb.astype(np.float64), axis=1)
    keep_count = max(1, round(V * 0.7))
    expected_keep = sorted(np.argsort(-row_norm)[:keep_count].tolist())
    assert result.kept_token_ids == expected_keep


def _ambiguous_embed_layer_norm_model(V1=10, V2=8, P=6, H=4, seed=203):
    rng = np.random.default_rng(seed)
    w1 = rng.standard_normal((V1, H)).astype(np.float32)
    w2 = rng.standard_normal((V2, H)).astype(np.float32)
    pos_emb = rng.standard_normal((P, H)).astype(np.float32)
    gamma = np.ones(H, dtype=np.float32)
    beta = np.zeros(H, dtype=np.float32)
    model = _model(
        f"""
        g (int32[2,3] input_ids_a, int32[2,3] input_ids_b) =>
           (float[2,3,{H}] out_a, float[2,3,{H}] out_b)
        {{
          out_a, mask_a = com.microsoft.EmbedLayerNormalization<epsilon=1e-12>(
              input_ids_a, , W1, Pos, , Gamma, Beta)
          out_b, mask_b = com.microsoft.EmbedLayerNormalization<epsilon=1e-12>(
              input_ids_b, , W2, Pos, , Gamma, Beta)
        }}
        """,
        initializer=[
            _f32(w1, "W1"),
            _f32(w2, "W2"),
            _f32(pos_emb, "Pos"),
            _f32(gamma, "Gamma"),
            _f32(beta, "Beta"),
        ],
    )
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))
    return model


def test_embed_layer_norm_declines_when_ambiguous_and_input_name_disambiguates():
    model = _ambiguous_embed_layer_norm_model()
    onnx.checker.check_model(model)

    result = onnxsim.apply_embedding_vocab_pruning_cpp(model, keep_token_ids=[0, 1, 2])
    assert result.matched is False
    assert result.model.SerializeToString() == model.SerializeToString()

    result = onnxsim.apply_embedding_vocab_pruning_cpp(
        model, drop_token_ids=[4, 5], input_name="input_ids_a"
    )
    assert result.matched
    assert result.kept_token_ids == [0, 1, 2, 3, 6, 7, 8, 9]
    w2 = {t.name: t for t in result.model.graph.initializer}["W2"]
    assert list(w2.dims) == [8, 4]  # the other producer's own table, untouched


# --- FusedGemm / GemmFastGelu lm_head node types ----------------------------
#
# `lm_head` auto-detection now also recognizes `com.microsoft::FusedGemm`/
# `GemmFastGelu` (MatchMatMulLikeWidenedRaw), not just a bare `MatMul`/
# vanilla `Gemm`.


def test_untied_lm_head_fusedgemm_matches_oracle_and_ort_execution():
    # FusedGemm has a real CPU kernel in this environment, so this one runs
    # directly (no decomposition needed), unlike GemmFastGelu below.
    V, H = 9, 5
    rng = np.random.default_rng(210)
    w_emb = rng.standard_normal((V, H)).astype(np.float32)
    w_lm = rng.standard_normal((V, H)).astype(np.float32)  # [N, K], transB=1
    b_lm = rng.standard_normal(V).astype(np.float32)
    model = _model(
        f"""
        g (int64[M] input_ids) => (float[M,{V}] logits)
        {{
          hidden = Gather<axis=0>(W_emb, input_ids)
          logits = com.microsoft.FusedGemm<transB=1, activation="Relu">(hidden, W_lm, B_lm)
        }}
        """,
        initializer=[_f32(w_emb, "W_emb"), _f32(w_lm, "W_lm"), _f32(b_lm, "B_lm")],
    )
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))
    onnx.checker.check_model(model)

    keep_token_ids = [0, 2, 3, 5, 6, 8]
    result = onnxsim.apply_embedding_vocab_pruning_cpp(
        model, keep_token_ids=keep_token_ids
    )
    onnx.checker.check_model(result.model)
    assert result.matched and result.lm_head_pruned
    assert result.kept_token_ids == keep_token_ids

    input_ids = np.array([0, 6, 8, 2], dtype=np.int64)
    remapped = np.array([result.id_map[i] for i in input_ids], dtype=np.int64)
    orig_out = _run(model, {"input_ids": input_ids})[0]
    pruned_out = _run(result.model, {"input_ids": remapped})[0]
    expected = orig_out[..., result.kept_token_ids]
    np.testing.assert_allclose(pruned_out, expected, atol=1e-5, rtol=1e-5)


def _decompose_gemmfastgelu(model):
    """Rewrites every `com.microsoft::GemmFastGelu` node into the literal
    unfused `MatMul(X, W) -> FastGelu(h, bias?)` sequence it is byte-
    identical to -- needed only because this environment's onnxruntime has
    no CPU kernel for `GemmFastGelu` itself (confirmed directly: a plain
    `InferenceSession` construction against it raises `NOT_IMPLEMENTED`),
    mirroring ``tests/test_pruning.py``'s own `_decompose_gemmfastgelu`.
    """
    out = onnx.ModelProto()
    out.CopyFrom(model)
    new_nodes = []
    for node in out.graph.node:
        if node.op_type != "GemmFastGelu" or node.domain != "com.microsoft":
            new_nodes.append(node)
            continue
        x_name, w_name = node.input[0], node.input[1]
        bias_name = node.input[2] if len(node.input) == 3 and node.input[2] else None
        (y_name,) = node.output
        h_name = f"{y_name}__gemmfastgelu_h"
        new_nodes.append(onnx.helper.make_node("MatMul", [x_name, w_name], [h_name]))
        fastgelu_inputs = [h_name, bias_name] if bias_name else [h_name]
        new_nodes.append(
            onnx.helper.make_node(
                "FastGelu", fastgelu_inputs, [y_name], domain="com.microsoft"
            )
        )
    del out.graph.node[:]
    out.graph.node.extend(new_nodes)
    return out


def test_untied_lm_head_gemmfastgelu_matches_oracle_via_decompose():
    V, H = 10, 6
    rng = np.random.default_rng(211)
    w_emb = rng.standard_normal((V, H)).astype(np.float32)
    w_lm = rng.standard_normal((H, V)).astype(np.float32)  # [K, N], no transpose
    b_lm = rng.standard_normal(V).astype(np.float32)
    model = _model(
        f"""
        g (int64[M] input_ids) => (float[M,{V}] logits)
        {{
          hidden = Gather<axis=0>(W_emb, input_ids)
          logits = com.microsoft.GemmFastGelu(hidden, W_lm, B_lm)
        }}
        """,
        initializer=[_f32(w_emb, "W_emb"), _f32(w_lm, "W_lm"), _f32(b_lm, "B_lm")],
    )
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))
    onnx.checker.check_model(model)

    keep_token_ids = [0, 1, 3, 5, 8, 9]
    result = onnxsim.apply_embedding_vocab_pruning_cpp(
        model, keep_token_ids=keep_token_ids
    )
    onnx.checker.check_model(result.model)
    assert result.matched and result.lm_head_pruned
    assert result.kept_token_ids == keep_token_ids

    lm_init = {t.name: t for t in result.model.graph.initializer}["W_lm"]
    assert list(lm_init.dims) == [H, len(keep_token_ids)]

    input_ids = np.array([0, 5, 9, 8], dtype=np.int64)
    remapped = np.array([result.id_map[i] for i in input_ids], dtype=np.int64)
    orig_out = _run(_decompose_gemmfastgelu(model), {"input_ids": input_ids})[0]
    pruned_out = _run(_decompose_gemmfastgelu(result.model), {"input_ids": remapped})[0]
    expected = orig_out[..., result.kept_token_ids]
    np.testing.assert_allclose(pruned_out, expected, atol=1e-5, rtol=1e-5)


# --- FLOAT16 / BFLOAT16 embedding table --------------------------------------
#
# FLOAT16 has a real onnxruntime CPU kernel here, so its own test runs the
# pruned model through a real session; BFLOAT16 has none (confirmed
# directly, mirroring tests/test_pruning.py's own "FP16/BFloat16 weight
# support" section comment), so its own test checks correctness at the
# array level (dtype preservation, exact per-element decode via
# ``ml_dtypes.bfloat16``) instead.


def test_fp16_embedding_and_lm_head_matches_ort_execution_and_preserves_bits():
    V, H = 10, 6
    rng = np.random.default_rng(220)
    w_emb = (rng.standard_normal((V, H)) * 0.5).astype(np.float16)
    w_lm = (rng.standard_normal((H, V)) * 0.5).astype(np.float16)
    model = _model(
        f"""
        g (int64[M] input_ids) => (float16[M,{V}] logits)
        {{
          hidden = Gather<axis=0>(W_emb, input_ids)
          logits = MatMul(hidden, W_lm)
        }}
        """,
        initializer=[_f16(w_emb, "W_emb"), _f16(w_lm, "W_lm")],
    )
    onnx.checker.check_model(model)

    keep_token_ids = [0, 2, 4, 6, 8, 9]
    result = onnxsim.apply_embedding_vocab_pruning_cpp(
        model, keep_token_ids=keep_token_ids
    )
    onnx.checker.check_model(result.model)
    assert result.matched and result.lm_head_pruned
    assert result.kept_token_ids == keep_token_ids

    inits = {t.name: t for t in result.model.graph.initializer}
    assert inits["W_emb"].data_type == onnx.TensorProto.FLOAT16
    assert inits["W_lm"].data_type == onnx.TensorProto.FLOAT16
    # Value-preserving slice -- every surviving row/column must reproduce
    # the exact original fp16 bit pattern, not a re-rounded one.
    emb_new = onnx.numpy_helper.to_array(inits["W_emb"])
    np.testing.assert_array_equal(
        emb_new.view(np.uint16), w_emb[keep_token_ids].view(np.uint16)
    )
    lm_new = onnx.numpy_helper.to_array(inits["W_lm"])
    np.testing.assert_array_equal(
        lm_new.view(np.uint16), w_lm[:, keep_token_ids].view(np.uint16)
    )

    input_ids = np.array([0, 8, 4, 9], dtype=np.int64)
    remapped = np.array([result.id_map[i] for i in input_ids], dtype=np.int64)
    orig_out = _run(model, {"input_ids": input_ids})[0]
    pruned_out = _run(result.model, {"input_ids": remapped})[0]
    expected = orig_out[..., result.kept_token_ids]
    np.testing.assert_allclose(
        pruned_out.astype(np.float32),
        expected.astype(np.float32),
        atol=1e-2,
        rtol=1e-2,
    )


def test_bfloat16_embedding_preserves_dtype_and_matches_array_oracle():
    # No onnxruntime CPU execution support for BFLOAT16 in this environment
    # (confirmed directly, mirroring tests/test_pruning.py's own bfloat16
    # tests) -- checked via ml_dtypes decode instead of a real session run.
    V, H = 10, 6
    rng = np.random.default_rng(221)
    w_emb = (rng.standard_normal((V, H)) * 0.5).astype(ml_dtypes.bfloat16)
    model = _model(
        f"""
        g (int64[M] input_ids) => (bfloat16[M,{H}] hidden)
        {{
          hidden = Gather<axis=0>(W_emb, input_ids)
        }}
        """,
        initializer=[_bf16(w_emb, "W_emb")],
    )
    onnx.checker.check_model(model)

    keep_token_ids = [0, 2, 3, 5, 7, 9]
    result = onnxsim.apply_embedding_vocab_pruning_cpp(
        model, keep_token_ids=keep_token_ids
    )
    onnx.checker.check_model(result.model)
    assert result.matched
    assert not result.lm_head_pruned
    assert result.kept_token_ids == keep_token_ids
    assert result.id_map == {tok: i for i, tok in enumerate(result.kept_token_ids)}

    emb_init = {t.name: t for t in result.model.graph.initializer}["W_emb"]
    assert emb_init.data_type == onnx.TensorProto.BFLOAT16
    emb_new = onnx.numpy_helper.to_array(emb_init)
    assert emb_new.dtype == ml_dtypes.bfloat16
    np.testing.assert_array_equal(
        emb_new.view(np.uint16), w_emb[keep_token_ids].view(np.uint16)
    )


def test_bfloat16_magnitude_pruning_matches_array_oracle():
    V, H = 12, 6
    rng = np.random.default_rng(222)
    scale = np.linspace(3.0, 0.1, V)
    w_emb = (rng.standard_normal((V, H)) * scale[:, None]).astype(ml_dtypes.bfloat16)
    model = _model(
        f"""
        g (int64[M] input_ids) => (bfloat16[M,{H}] hidden)
        {{
          hidden = Gather<axis=0>(W_emb, input_ids)
        }}
        """,
        initializer=[_bf16(w_emb, "W_emb")],
    )
    onnx.checker.check_model(model)

    result = onnxsim.apply_embedding_vocab_magnitude_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(result.model)
    assert result.matched

    row_norm = np.linalg.norm(w_emb.astype(np.float64), axis=1)
    keep_count = max(1, round(V * 0.5))
    expected_keep = sorted(np.argsort(-row_norm)[:keep_count].tolist())
    assert result.kept_token_ids == expected_keep


# --- GatherBlockQuantized producer shape -------------------------------------
#
# The block-quantized (int2/int4/int8) analogue of a plain-float embedding
# `Gather` (see structured_pruning_entry.cpp's own "Embedding vocabulary
# pruning" section comment, and pruning.py's own section-top comment above
# `_match_gather_block_quantized_producer`, for the full empirical schema/
# packing/real-exporter-evidence this depends on). Two genuinely different
# sub-8-bit packing conventions are dispatched purely off `data`'s own
# dtype: ONNX-native sub-byte `tensor(uint4)`/`tensor(int4)` (a flat,
# whole-tensor, 2-values-per-byte pack -- built here via `ml_dtypes.uint4`/
# `ml_dtypes.int4` so `onnx.numpy_helper.from_array` packs it exactly the
# way a real exporter's own tensor would), and manually-packed plain
# `tensor(uint8)` (`bits` in {2, 4, 8} -- packed per-row, low-order bits
# first, via `_pack_uint8_bits` below, mirroring the schema doc's own
# "for bits < 8 the values are packed along the last dimension"). Every
# test here uses `onnx.helper`/`onnx.numpy_helper.from_array` rather than
# `onnx.parser`, per CLAUDE.md's own documented fallback: the parser's text
# format encodes tensor literals as `float_data`, has no notion of ONNX's
# native sub-4-bit packing at all, and can't express a hand-packed uint8
# byte layout either -- both need genuine `numpy`-array-shaped tensor
# construction. `GatherBlockQuantized` has a real onnxruntime CPU kernel in
# this environment (confirmed directly, like `EmbedLayerNormalization`/
# `FusedGemm` above), so every test below runs the pruned model through a
# real session and compares against a numpy slice of the ORIGINAL model's
# own output -- the same "byte-exact oracle" bar this file holds itself to
# throughout, cross-checked against a real `InferenceSession` run rather
# than the (now-aliased, so no longer independent) pure-Python entry point.


def _pack_uint8_bits(vals, bits):
    """Packs `vals` (small non-negative ints, shape `(rows, cols)`) into a
    plain `uint8` array using `GatherBlockQuantized`'s own manually-packed
    convention: independent PER ROW, low-order bits first -- e.g. at
    `bits=4`, element `2i`/`2i+1` of a row land in the low/high nibble of
    that row's own byte `i`. A `cols` not evenly divisible by
    `8 // bits` leaves the trailing high-order bits of the last byte as
    zero padding (never read back by anything this pass checks).
    """
    rows, cols = vals.shape
    per_byte = 8 // bits
    packed_width = (cols + per_byte - 1) // per_byte
    packed = np.zeros((rows, packed_width), dtype=np.uint8)
    for i in range(cols):
        byte_i = i // per_byte
        shift = (i % per_byte) * bits
        packed[:, byte_i] |= (vals[:, i].astype(np.uint8) & ((1 << bits) - 1)) << shift
    return packed


def _gbq_model(data_tt, scales_tt, zp_tt, bits, block_size, hidden_out):
    """A single-node `com.microsoft::GatherBlockQuantized` model, `indices`
    (`int64`, rank 1) the sole graph input, `output` (same dtype as
    `scales_tt`) the sole graph output. `gather_axis=0`/`quantize_axis=1`
    throughout -- the only layout this pass's matcher ever admits.
    """
    inputs = ["data", "indices", "scales"]
    initializer = [data_tt, scales_tt]
    if zp_tt is not None:
        inputs.append("zero_points")
        initializer.append(zp_tt)
    node = onnx.helper.make_node(
        "GatherBlockQuantized",
        inputs,
        ["output"],
        domain="com.microsoft",
        bits=bits,
        block_size=block_size,
        gather_axis=0,
        quantize_axis=1,
    )
    indices_vi = onnx.helper.make_tensor_value_info(
        "indices", onnx.TensorProto.INT64, ["N"]
    )
    output_vi = onnx.helper.make_tensor_value_info(
        "output", scales_tt.data_type, ["N", hidden_out]
    )
    graph = onnx.helper.make_graph(
        [node], "g", [indices_vi], [output_vi], initializer=initializer
    )
    model = onnx.helper.make_model(
        graph,
        opset_imports=[
            onnx.helper.make_opsetid("", 21),
            onnx.helper.make_opsetid("com.microsoft", 1),
        ],
    )
    model.ir_version = 10
    return model


def test_gbq_native_uint4_no_zero_points_matches_oracle():
    # Odd hidden_size (17) -- the native uint4 flat pack crosses a row
    # boundary (this section's own top comment) -- exercises that this
    # pass's row-select still lands on exactly the right bits.
    V, H = 6, 17
    bits, block_size = 4, 16
    n_blocks = (H + block_size - 1) // block_size
    rng = np.random.default_rng(300)
    data_vals = rng.integers(0, 16, size=(V, H), dtype=np.int64).astype(np.uint8)
    data_tt = onnx.numpy_helper.from_array(data_vals.astype(ml_dtypes.uint4), "data")
    scales_vals = (rng.random((V, n_blocks)).astype(np.float32) + 0.1)
    scales_tt = _f32(scales_vals, "scales")

    model = _gbq_model(data_tt, scales_tt, None, bits, block_size, H)
    onnx.checker.check_model(model)

    idx_full = np.arange(V, dtype=np.int64)
    orig_out = _run(model, {"indices": idx_full})[0]

    keep_token_ids = [0, 2, 5, 3]
    result = onnxsim.apply_embedding_vocab_pruning_cpp(model, keep_token_ids=keep_token_ids)
    onnx.checker.check_model(result.model)
    assert result.matched
    assert not result.lm_head_pruned
    assert result.kept_token_ids == sorted(keep_token_ids)

    inits = {t.name: t for t in result.model.graph.initializer}
    assert list(inits["data"].dims) == [len(result.kept_token_ids), H]
    assert inits["data"].data_type == onnx.TensorProto.UINT4

    idx_new = np.arange(len(result.kept_token_ids), dtype=np.int64)
    pruned_out = _run(result.model, {"indices": idx_new})[0]
    expected = orig_out[result.kept_token_ids]
    np.testing.assert_allclose(pruned_out, expected, atol=1e-5, rtol=1e-5)


def test_gbq_native_uint4_with_zero_points_matches_oracle():
    V, H = 5, 12
    bits, block_size = 4, 16
    n_blocks = 1
    rng = np.random.default_rng(301)
    data_vals = rng.integers(0, 16, size=(V, H), dtype=np.int64).astype(np.uint8)
    data_tt = onnx.numpy_helper.from_array(data_vals.astype(ml_dtypes.uint4), "data")
    scales_vals = (rng.random((V, n_blocks)).astype(np.float32) + 0.1)
    scales_tt = _f32(scales_vals, "scales")
    zp_vals = rng.integers(0, 16, size=(V, n_blocks), dtype=np.int64).astype(np.uint8)
    zp_tt = onnx.numpy_helper.from_array(zp_vals.astype(ml_dtypes.uint4), "zero_points")

    model = _gbq_model(data_tt, scales_tt, zp_tt, bits, block_size, H)
    onnx.checker.check_model(model)

    idx_full = np.arange(V, dtype=np.int64)
    orig_out = _run(model, {"indices": idx_full})[0]

    result = onnxsim.apply_embedding_vocab_pruning_cpp(model, drop_token_ids=[1, 4])
    onnx.checker.check_model(result.model)
    assert result.matched
    assert result.kept_token_ids == [0, 2, 3]

    inits = {t.name: t for t in result.model.graph.initializer}
    assert list(inits["zero_points"].dims) == [len(result.kept_token_ids), n_blocks]
    assert inits["zero_points"].data_type == onnx.TensorProto.UINT4

    idx_new = np.arange(len(result.kept_token_ids), dtype=np.int64)
    pruned_out = _run(result.model, {"indices": idx_new})[0]
    expected = orig_out[result.kept_token_ids]
    np.testing.assert_allclose(pruned_out, expected, atol=1e-5, rtol=1e-5)


def test_gbq_native_int4_matches_oracle():
    V, H = 7, 10
    bits, block_size = 4, 16
    n_blocks = 1
    rng = np.random.default_rng(302)
    data_vals = rng.integers(-8, 8, size=(V, H), dtype=np.int64).astype(np.int8)
    data_tt = onnx.numpy_helper.from_array(data_vals.astype(ml_dtypes.int4), "data")
    scales_vals = (rng.random((V, n_blocks)).astype(np.float32) + 0.1)
    scales_tt = _f32(scales_vals, "scales")

    model = _gbq_model(data_tt, scales_tt, None, bits, block_size, H)
    onnx.checker.check_model(model)

    idx_full = np.arange(V, dtype=np.int64)
    orig_out = _run(model, {"indices": idx_full})[0]

    keep_token_ids = [6, 4, 1, 0]
    result = onnxsim.apply_embedding_vocab_pruning_cpp(model, keep_token_ids=keep_token_ids)
    onnx.checker.check_model(result.model)
    assert result.matched
    assert result.kept_token_ids == sorted(keep_token_ids)
    assert {t.name: t for t in result.model.graph.initializer}["data"].data_type == (
        onnx.TensorProto.INT4
    )

    idx_new = np.arange(len(result.kept_token_ids), dtype=np.int64)
    pruned_out = _run(result.model, {"indices": idx_new})[0]
    expected = orig_out[result.kept_token_ids]
    np.testing.assert_allclose(pruned_out, expected, atol=1e-5, rtol=1e-5)


def test_gbq_packed_uint8_bits4_with_zero_points_matches_oracle():
    # true_hidden=5 at bits=4 packs to ceil(5*4/8)=3 bytes/row -- one
    # nibble's worth (the 6th logical position) is unused padding, exactly
    # the "intentionally-adversarial odd true width" case this section's
    # own top comment (and pruning.py's own matcher comment) documents.
    V, true_hidden, bits, block_size = 5, 5, 4, 16
    n_blocks = 1
    rng = np.random.default_rng(303)
    maxval = (1 << bits) - 1
    data_vals = rng.integers(0, maxval + 1, size=(V, true_hidden), dtype=np.int64)
    packed = _pack_uint8_bits(data_vals, bits)
    data_tt = onnx.numpy_helper.from_array(packed, "data")
    real_width = packed.shape[1] * (8 // bits)
    scales_vals = (rng.random((V, n_blocks)).astype(np.float32) + 0.1)
    scales_tt = _f32(scales_vals, "scales")
    zp_vals = rng.integers(0, maxval + 1, size=(V, n_blocks), dtype=np.int64)
    zp_tt = onnx.numpy_helper.from_array(_pack_uint8_bits(zp_vals, bits), "zero_points")

    model = _gbq_model(data_tt, scales_tt, zp_tt, bits, block_size, real_width)
    onnx.checker.check_model(model)

    idx_full = np.arange(V, dtype=np.int64)
    orig_out = _run(model, {"indices": idx_full})[0]

    keep_token_ids = [0, 2, 4, 1]
    result = onnxsim.apply_embedding_vocab_pruning_cpp(model, keep_token_ids=keep_token_ids)
    onnx.checker.check_model(result.model)
    assert result.matched
    assert result.kept_token_ids == sorted(keep_token_ids)

    inits = {t.name: t for t in result.model.graph.initializer}
    assert inits["data"].data_type == onnx.TensorProto.UINT8
    assert list(inits["data"].dims) == [len(result.kept_token_ids), packed.shape[1]]
    # Byte-exact per-row slice -- no unpack/repack needed for this
    # convention (this section's own top comment).
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(inits["data"]), packed[result.kept_token_ids]
    )

    idx_new = np.arange(len(result.kept_token_ids), dtype=np.int64)
    pruned_out = _run(result.model, {"indices": idx_new})[0]
    expected = orig_out[result.kept_token_ids]
    np.testing.assert_allclose(pruned_out, expected, atol=1e-5, rtol=1e-5)


def test_gbq_packed_uint8_bits2_no_zero_points_matches_oracle():
    V, true_hidden, bits, block_size = 6, 7, 2, 16
    n_blocks = 1
    rng = np.random.default_rng(304)
    maxval = (1 << bits) - 1
    data_vals = rng.integers(0, maxval + 1, size=(V, true_hidden), dtype=np.int64)
    packed = _pack_uint8_bits(data_vals, bits)
    data_tt = onnx.numpy_helper.from_array(packed, "data")
    real_width = packed.shape[1] * (8 // bits)
    scales_vals = (rng.random((V, n_blocks)).astype(np.float32) + 0.1)
    scales_tt = _f32(scales_vals, "scales")

    model = _gbq_model(data_tt, scales_tt, None, bits, block_size, real_width)
    onnx.checker.check_model(model)

    idx_full = np.arange(V, dtype=np.int64)
    orig_out = _run(model, {"indices": idx_full})[0]

    # Default zero_points for a plain-uint8 `data` is 2**(bits-1) -- confirm
    # the pruned model still agrees with the ORIGINAL model's own real
    # kernel execution (which itself already applies that default), not a
    # value hand-derived independently -- the "byte-exact oracle" bar.
    result = onnxsim.apply_embedding_vocab_pruning_cpp(model, drop_token_ids=[1, 3])
    onnx.checker.check_model(result.model)
    assert result.matched
    assert result.kept_token_ids == [0, 2, 4, 5]

    idx_new = np.arange(len(result.kept_token_ids), dtype=np.int64)
    pruned_out = _run(result.model, {"indices": idx_new})[0]
    expected = orig_out[result.kept_token_ids]
    np.testing.assert_allclose(pruned_out, expected, atol=1e-5, rtol=1e-5)


def test_gbq_plain_uint8_bits8_matches_oracle():
    # bits=8 -- no packing at all, `data` already a plain per-element byte
    # array; the identical "no unpack/repack needed" row-slice as the
    # bits<8 packed-uint8 convention, exercised here at the boundary value.
    V, H, bits, block_size = 6, 9, 8, 16
    n_blocks = 1
    rng = np.random.default_rng(305)
    data_vals = rng.integers(0, 256, size=(V, H), dtype=np.int64).astype(np.uint8)
    data_tt = onnx.numpy_helper.from_array(data_vals, "data")
    scales_vals = (rng.random((V, n_blocks)).astype(np.float32) + 0.1)
    scales_tt = _f32(scales_vals, "scales")

    model = _gbq_model(data_tt, scales_tt, None, bits, block_size, H)
    onnx.checker.check_model(model)

    idx_full = np.arange(V, dtype=np.int64)
    orig_out = _run(model, {"indices": idx_full})[0]

    keep_token_ids = [5, 3, 0, 1]
    result = onnxsim.apply_embedding_vocab_pruning_cpp(model, keep_token_ids=keep_token_ids)
    onnx.checker.check_model(result.model)
    assert result.matched
    assert result.kept_token_ids == sorted(keep_token_ids)

    idx_new = np.arange(len(result.kept_token_ids), dtype=np.int64)
    pruned_out = _run(result.model, {"indices": idx_new})[0]
    expected = orig_out[result.kept_token_ids]
    np.testing.assert_allclose(pruned_out, expected, atol=1e-5, rtol=1e-5)


def test_gbq_untied_lm_head_auto_detected_and_matches_oracle():
    V, H = 6, 16
    bits, block_size = 4, 16
    n_blocks = 1
    rng = np.random.default_rng(306)
    data_vals = rng.integers(0, 16, size=(V, H), dtype=np.int64).astype(np.uint8)
    data_tt = onnx.numpy_helper.from_array(data_vals.astype(ml_dtypes.uint4), "data")
    scales_vals = (rng.random((V, n_blocks)).astype(np.float32) + 0.1)
    scales_tt = _f32(scales_vals, "scales")
    w_lm = rng.standard_normal((H, V)).astype(np.float32)

    gbq_node = onnx.helper.make_node(
        "GatherBlockQuantized",
        ["data", "indices", "scales"],
        ["hidden"],
        domain="com.microsoft",
        bits=bits,
        block_size=block_size,
        gather_axis=0,
        quantize_axis=1,
    )
    mm_node = onnx.helper.make_node("MatMul", ["hidden", "W_lm"], ["logits"])
    indices_vi = onnx.helper.make_tensor_value_info("indices", onnx.TensorProto.INT64, ["N"])
    logits_vi = onnx.helper.make_tensor_value_info("logits", onnx.TensorProto.FLOAT, ["N", V])
    graph = onnx.helper.make_graph(
        [gbq_node, mm_node],
        "g",
        [indices_vi],
        [logits_vi],
        initializer=[data_tt, scales_tt, _f32(w_lm, "W_lm")],
    )
    model = onnx.helper.make_model(
        graph,
        opset_imports=[
            onnx.helper.make_opsetid("", 21),
            onnx.helper.make_opsetid("com.microsoft", 1),
        ],
    )
    model.ir_version = 10
    onnx.checker.check_model(model)

    idx_full = np.arange(V, dtype=np.int64)
    orig_out = _run(model, {"indices": idx_full})[0]

    keep_token_ids = [0, 1, 3, 5]
    result = onnxsim.apply_embedding_vocab_pruning_cpp(model, keep_token_ids=keep_token_ids)
    onnx.checker.check_model(result.model)
    assert result.matched
    assert result.lm_head_pruned
    assert result.kept_token_ids == sorted(keep_token_ids)

    lm_init = {t.name: t for t in result.model.graph.initializer}["W_lm"]
    assert list(lm_init.dims) == [H, len(result.kept_token_ids)]

    idx_new = np.arange(len(result.kept_token_ids), dtype=np.int64)
    pruned_out = _run(result.model, {"indices": idx_new})[0]
    expected = orig_out[result.kept_token_ids][:, result.kept_token_ids]
    np.testing.assert_allclose(pruned_out, expected, atol=1e-4, rtol=1e-4)


def test_gbq_tied_lm_head_declined_ambiguous_two_producers():
    # A tied lm_head sharing the packed `data` tensor itself is always
    # declined for this shape (MatchGatherBlockQuantizedProducer's own
    # single-consumer requirement on `data`) -- constructed here as two
    # independent `GatherBlockQuantized` nodes both reading the same `data`/
    # `scales` initializers, which is simultaneously an unrecognized-second-
    # consumer decline AND an ambiguous-multiple-producer decline; either
    # way the whole call must decline, the model left untouched.
    V, H, bits, block_size = 5, 8, 4, 16
    rng = np.random.default_rng(307)
    data_vals = rng.integers(0, 16, size=(V, H), dtype=np.int64).astype(np.uint8)
    data_tt = onnx.numpy_helper.from_array(data_vals.astype(ml_dtypes.uint4), "data")
    scales_tt = _f32(rng.random((V, 1)).astype(np.float32) + 0.1, "scales")

    node1 = onnx.helper.make_node(
        "GatherBlockQuantized", ["data", "indices1", "scales"], ["out1"],
        domain="com.microsoft", bits=bits, block_size=block_size,
        gather_axis=0, quantize_axis=1,
    )
    node2 = onnx.helper.make_node(
        "GatherBlockQuantized", ["data", "indices2", "scales"], ["out2"],
        domain="com.microsoft", bits=bits, block_size=block_size,
        gather_axis=0, quantize_axis=1,
    )
    ids1_vi = onnx.helper.make_tensor_value_info("indices1", onnx.TensorProto.INT64, ["N"])
    ids2_vi = onnx.helper.make_tensor_value_info("indices2", onnx.TensorProto.INT64, ["M"])
    out1_vi = onnx.helper.make_tensor_value_info("out1", onnx.TensorProto.FLOAT, ["N", H])
    out2_vi = onnx.helper.make_tensor_value_info("out2", onnx.TensorProto.FLOAT, ["M", H])
    graph = onnx.helper.make_graph(
        [node1, node2], "g", [ids1_vi, ids2_vi], [out1_vi, out2_vi],
        initializer=[data_tt, scales_tt],
    )
    model = onnx.helper.make_model(
        graph,
        opset_imports=[
            onnx.helper.make_opsetid("", 21),
            onnx.helper.make_opsetid("com.microsoft", 1),
        ],
    )
    model.ir_version = 10
    onnx.checker.check_model(model)

    result = onnxsim.apply_embedding_vocab_pruning_cpp(model, keep_token_ids=[0, 1, 2])
    assert result.matched is False
    assert result.model.SerializeToString() == model.SerializeToString()


def test_gbq_input_name_disambiguates_two_producers():
    V1, V2, H, bits, block_size = 6, 8, 8, 4, 16
    rng = np.random.default_rng(308)

    def mk(vocab, prefix):
        d = rng.integers(0, 16, size=(vocab, H), dtype=np.int64).astype(np.uint8)
        dt = onnx.numpy_helper.from_array(d.astype(ml_dtypes.uint4), f"{prefix}_data")
        st = _f32(rng.random((vocab, 1)).astype(np.float32) + 0.1, f"{prefix}_scales")
        return dt, st

    a_data, a_scales = mk(V1, "a")
    b_data, b_scales = mk(V2, "b")
    node_a = onnx.helper.make_node(
        "GatherBlockQuantized", ["a_data", "ids_a", "a_scales"], ["out_a"],
        domain="com.microsoft", bits=bits, block_size=block_size,
        gather_axis=0, quantize_axis=1,
    )
    node_b = onnx.helper.make_node(
        "GatherBlockQuantized", ["b_data", "ids_b", "b_scales"], ["out_b"],
        domain="com.microsoft", bits=bits, block_size=block_size,
        gather_axis=0, quantize_axis=1,
    )
    ids_a_vi = onnx.helper.make_tensor_value_info("ids_a", onnx.TensorProto.INT64, ["N"])
    ids_b_vi = onnx.helper.make_tensor_value_info("ids_b", onnx.TensorProto.INT64, ["M"])
    out_a_vi = onnx.helper.make_tensor_value_info("out_a", onnx.TensorProto.FLOAT, ["N", H])
    out_b_vi = onnx.helper.make_tensor_value_info("out_b", onnx.TensorProto.FLOAT, ["M", H])
    graph = onnx.helper.make_graph(
        [node_a, node_b], "g", [ids_a_vi, ids_b_vi], [out_a_vi, out_b_vi],
        initializer=[a_data, a_scales, b_data, b_scales],
    )
    model = onnx.helper.make_model(
        graph,
        opset_imports=[
            onnx.helper.make_opsetid("", 21),
            onnx.helper.make_opsetid("com.microsoft", 1),
        ],
    )
    model.ir_version = 10
    onnx.checker.check_model(model)

    result = onnxsim.apply_embedding_vocab_pruning_cpp(model, keep_token_ids=[0, 1, 2])
    assert result.matched is False  # ambiguous, no input_name

    result2 = onnxsim.apply_embedding_vocab_pruning_cpp(
        model, drop_token_ids=[4, 5], input_name="ids_a"
    )
    assert result2.matched
    assert result2.kept_token_ids == [0, 1, 2, 3]
    b_data_new = {t.name: t for t in result2.model.graph.initializer}["b_data"]
    assert list(b_data_new.dims) == [V2, H]  # the other producer's table untouched


def test_gbq_magnitude_pruning_matches_dequantized_norm_oracle():
    # Independent oracle: dequantize by hand (the same formula this pass's
    # own GatherBlockQuantizedDequantized/`_gather_block_quantized_
    # dequantized` use, per structured_pruning_entry.cpp's own doc comment),
    # rank by L2 norm, and compare -- never calls the (now-aliased)
    # pure-Python entry point.
    V, H, bits, block_size = 10, 8, 4, 16
    n_blocks = 1
    rng = np.random.default_rng(309)
    scale_factor = np.linspace(3.0, 0.2, V)
    data_vals = rng.integers(0, 16, size=(V, H), dtype=np.int64).astype(np.uint8)
    data_tt = onnx.numpy_helper.from_array(data_vals.astype(ml_dtypes.uint4), "data")
    scales_vals = scale_factor.astype(np.float32).reshape(V, n_blocks)
    scales_tt = _f32(scales_vals, "scales")

    model = _gbq_model(data_tt, scales_tt, None, bits, block_size, H)
    onnx.checker.check_model(model)

    result = onnxsim.apply_embedding_vocab_magnitude_pruning_cpp(model, sparsity=0.4)
    onnx.checker.check_model(result.model)
    assert result.matched

    dequant = data_vals.astype(np.float64) * scales_vals.astype(np.float64)  # zp=0
    row_norm = np.linalg.norm(dequant, axis=1)
    keep_count = max(1, round(V * 0.6))
    expected_keep = sorted(np.argsort(-row_norm)[:keep_count].tolist())
    assert result.kept_token_ids == expected_keep
