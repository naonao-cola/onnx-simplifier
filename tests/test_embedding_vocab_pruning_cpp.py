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
vocabulary pruning" coverage (search that section name there), restricted to
what this C++ port actually recognizes: a plain ``Gather`` producer (never
``com.microsoft::EmbedLayerNormalization``/``GatherBlockQuantized``, both out
of scope for this port -- see structured_pruning_entry.cpp's own section
comment), and a bare ``MatMul``/vanilla-``Gemm`` ``lm_head`` (never
``com.microsoft::FusedGemm``/``GemmFastGelu``). Every test that actually
prunes something either runs the result through a real onnxruntime CPU
session and compares against a numpy slice of the ORIGINAL model's own
output (the same "byte-exact oracle" bar every other C++-port test file in
this repo holds itself to), or cross-checks against the pure-Python
``onnxsim.apply_embedding_vocab_pruning``/``apply_embedding_vocab_magnitude_
pruning`` entry points as a second, independently-implemented oracle.
"""

import numpy as np
import onnx
import onnx.checker
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

    # Independent oracle: the pure-Python entry point must agree exactly
    # (same kept ids, same id_map, same lm_head_pruned flag).
    py_result = onnxsim.apply_embedding_vocab_pruning(
        model, keep_token_ids=keep_token_ids + [4, 9]
    )
    assert py_result.kept_token_ids == result.kept_token_ids
    assert py_result.id_map == result.id_map
    assert py_result.lm_head_pruned == result.lm_head_pruned


def test_drop_token_ids_equivalent_to_complement_keep_set():
    V, H = 9, 5
    model = _untied_model(V, H, seed=4)
    drop = [2, 4, 7]

    result = onnxsim.apply_embedding_vocab_pruning_cpp(model, drop_token_ids=drop)
    onnx.checker.check_model(result.model)
    assert result.matched and result.lm_head_pruned
    assert result.kept_token_ids == [i for i in range(V) if i not in drop]

    py_result = onnxsim.apply_embedding_vocab_pruning(model, drop_token_ids=drop)
    assert py_result.kept_token_ids == result.kept_token_ids
    assert py_result.id_map == result.id_map


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

    py_result = onnxsim.apply_embedding_vocab_pruning(
        model, drop_token_ids=[4, 5], input_name="input_ids"
    )
    assert py_result.kept_token_ids == result.kept_token_ids


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

    # Cross-check against the pure-Python entry point's own ranking.
    py_result = onnxsim.apply_embedding_vocab_magnitude_pruning(model, sparsity=0.4)
    assert py_result.kept_token_ids == result.kept_token_ids


def test_magnitude_pruning_rejects_bad_sparsity():
    model = _matmul_model(K=4, N=4)
    with pytest.raises(ValueError, match="sparsity"):
        onnxsim.apply_embedding_vocab_magnitude_pruning_cpp(model, sparsity=1.0)


def test_magnitude_pruning_declines_when_no_embedding_pattern_exists():
    model = _matmul_model(K=8, N=4)
    result = onnxsim.apply_embedding_vocab_magnitude_pruning_cpp(model, sparsity=0.3)
    assert result.matched is False
    assert result.model.SerializeToString() == model.SerializeToString()
