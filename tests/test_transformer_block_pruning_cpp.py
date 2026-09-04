"""Tests for ``onnxsim.apply_transformer_block_pruning_cpp`` -- the C++-backed
port of ``onnxsim.apply_transformer_block_pruning`` (see
``onnxsim/structured_pruning_entry.cpp``'s "Transformer block (depth) pruning"
section and ``ApplyTransformerBlockPruning``). Like
``onnxsim.apply_structured_wanda_pruning_cpp``, this runs the model over real
calibration data through a real ``onnxruntime``-backed
:class:`onnxsim.onnx_simplifier.PyModelExecutor` (via
``onnxsim.onnx_simplifier._get_model_executor``) -- never a fake/mock
executor. Unlike every other C++-backed pruning entry point, this one
performs real graph surgery (node deletion + consumer rewiring), not tensor
slicing, so several of these tests confirm the rewiring itself is correct
via an independently hand-built "already pruned" reference model, not merely
that *some* pruning happened.

See ``tests/test_pruning.py``'s own "apply_transformer_block_pruning" section
for the exact matched pattern (a pre-norm residual sub-block whose merge is a
bare ``Add`` and whose entry norm is a plain LayerNormalization/
RMSNormalization/SimplifiedLayerNormalization node or a fused
SkipLayerNormalization-family node's own optional fourth output) and the full
set of decline conditions this file's tests cross-check against.
"""

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


# --- Basic drop: matches a hand-built "already pruned" oracle ---------------


def test_transformer_block_pruning_cpp_drops_near_identity_mlp_block_and_matches_manual_removal_oracle():
    # Two stacked pre-norm MLP residual blocks, block 0 engineered
    # near-identity (tiny down-projection weight) -- the canonical
    # "redundant block" case the calibrated mean-cosine-similarity ranking
    # should flag as the one to drop. Mirrors
    # test_pruning.py's own
    # test_transformer_block_pruning_drops_near_identity_mlp_block_and_matches_manual_removal_oracle
    # exactly, just through the C++-backed entry point.
    H = 8
    rng = np.random.default_rng(0)
    ln0_scale = np.ones(H, dtype=np.float32)
    ln0_bias = np.zeros(H, dtype=np.float32)
    w0 = (rng.standard_normal((H, H)) * 1e-6).astype(np.float32)
    ln1_scale = np.ones(H, dtype=np.float32)
    ln1_bias = np.zeros(H, dtype=np.float32)
    w1 = rng.standard_normal((H, H)).astype(np.float32)

    model = _model(
        f"""
        g (float[1,4,{H}] x0) => (float[1,4,{H}] y)
        {{
          ln0 = LayerNormalization<axis=-1>(x0, Ln0Scale, Ln0Bias)
          h0 = MatMul(ln0, W0)
          x1 = Add(x0, h0)

          ln1 = LayerNormalization<axis=-1>(x1, Ln1Scale, Ln1Bias)
          h1 = MatMul(ln1, W1)
          x2 = Add(x1, h1)

          y = Identity(x2)
        }}
        """,
        initializer=[
            _f32(ln0_scale, "Ln0Scale"),
            _f32(ln0_bias, "Ln0Bias"),
            _f32(w0, "W0"),
            _f32(ln1_scale, "Ln1Scale"),
            _f32(ln1_bias, "Ln1Bias"),
            _f32(w1, "W1"),
        ],
    )
    onnx.checker.check_model(model)

    ref = _model(
        f"""
        g (float[1,4,{H}] x0) => (float[1,4,{H}] y)
        {{
          ln1 = LayerNormalization<axis=-1>(x0, Ln1Scale, Ln1Bias)
          h1 = MatMul(ln1, W1)
          x2 = Add(x0, h1)
          y = Identity(x2)
        }}
        """,
        initializer=[
            _f32(ln1_scale, "Ln1Scale"),
            _f32(ln1_bias, "Ln1Bias"),
            _f32(w1, "W1"),
        ],
    )
    onnx.checker.check_model(ref)

    pruned = onnxsim.apply_transformer_block_pruning_cpp(
        model, num_blocks_to_drop=1, seed=0, num_samples=4
    )
    onnx.checker.check_model(pruned)

    assert [n.op_type for n in pruned.graph.node] == [
        "LayerNormalization",
        "MatMul",
        "Add",
        "Identity",
    ]

    x = np.random.default_rng(42).standard_normal((1, 4, H)).astype(np.float32)
    (pruned_y,) = _run(pruned, {"x0": x})
    (ref_y,) = _run(ref, {"x0": x})
    np.testing.assert_array_equal(pruned_y, ref_y)

    # Cross-check against the pure-Python reference on the identical model +
    # calibration seed -- the primary port-correctness signal.
    pruned_py = onnxsim.apply_transformer_block_pruning(
        model, num_blocks_to_drop=1, seed=0, num_samples=4
    )
    assert pruned.SerializeToString() == pruned_py.SerializeToString()


def test_transformer_block_pruning_cpp_adversarial_ranking_prefers_more_redundant_block_regardless_of_position():
    # The redundant block is placed *second*, proving selection follows the
    # calibrated cosine-similarity ranking, not simply "whichever candidate
    # is found first".
    H = 8
    rng = np.random.default_rng(7)
    ln0_scale = np.ones(H, dtype=np.float32)
    ln0_bias = np.zeros(H, dtype=np.float32)
    w0 = rng.standard_normal((H, H)).astype(np.float32)
    ln1_scale = np.ones(H, dtype=np.float32)
    ln1_bias = np.zeros(H, dtype=np.float32)
    w1 = (rng.standard_normal((H, H)) * 1e-6).astype(np.float32)

    model = _model(
        f"""
        g (float[1,4,{H}] x0) => (float[1,4,{H}] y)
        {{
          ln0 = LayerNormalization<axis=-1>(x0, Ln0Scale, Ln0Bias)
          h0 = MatMul(ln0, W0)
          x1 = Add(x0, h0)

          ln1 = LayerNormalization<axis=-1>(x1, Ln1Scale, Ln1Bias)
          h1 = MatMul(ln1, W1)
          x2 = Add(x1, h1)

          y = Identity(x2)
        }}
        """,
        initializer=[
            _f32(ln0_scale, "Ln0Scale"),
            _f32(ln0_bias, "Ln0Bias"),
            _f32(w0, "W0"),
            _f32(ln1_scale, "Ln1Scale"),
            _f32(ln1_bias, "Ln1Bias"),
            _f32(w1, "W1"),
        ],
    )

    ref = _model(
        f"""
        g (float[1,4,{H}] x0) => (float[1,4,{H}] y)
        {{
          ln0 = LayerNormalization<axis=-1>(x0, Ln0Scale, Ln0Bias)
          h0 = MatMul(ln0, W0)
          y = Add(x0, h0)
        }}
        """,
        initializer=[
            _f32(ln0_scale, "Ln0Scale"),
            _f32(ln0_bias, "Ln0Bias"),
            _f32(w0, "W0"),
        ],
    )

    pruned = onnxsim.apply_transformer_block_pruning_cpp(
        model, num_blocks_to_drop=1, seed=1, num_samples=4
    )
    onnx.checker.check_model(pruned)
    assert [n.op_type for n in pruned.graph.node] == [
        "LayerNormalization",
        "MatMul",
        "Add",
        "Identity",
    ]
    survivor = next(n for n in pruned.graph.node if n.op_type == "MatMul")
    w_name = survivor.input[1]
    inits = {t.name: t for t in pruned.graph.initializer}
    np.testing.assert_array_equal(onnx.numpy_helper.to_array(inits[w_name]), w0)

    x = np.random.default_rng(99).standard_normal((1, 4, H)).astype(np.float32)
    (pruned_y,) = _run(pruned, {"x0": x})
    (ref_y,) = _run(ref, {"x0": x})
    np.testing.assert_array_equal(pruned_y, ref_y)


def test_transformer_block_pruning_cpp_drops_attention_block_and_matches_manual_removal_oracle():
    # A single self-attention residual block (Q/K/V feeding a real
    # com.microsoft::GroupQueryAttention node, not a plain MLP) -- confirms
    # F need not be an MLP at all.
    K, NH, D = 16, 2, 8
    Nqkv = NH * D
    rng = np.random.default_rng(5)
    scale = np.ones(K, dtype=np.float32)
    bias = np.zeros(K, dtype=np.float32)
    wq = (rng.standard_normal((K, Nqkv)) * 1e-6).astype(np.float32)
    wk = (rng.standard_normal((K, Nqkv)) * 1e-6).astype(np.float32)
    wv = (rng.standard_normal((K, Nqkv)) * 1e-6).astype(np.float32)
    wout = (rng.standard_normal((Nqkv, K)) * 1e-6).astype(np.float32)
    seq = 5
    seqlens_k = np.full((2,), seq - 1, dtype=np.int32)
    total_seq = np.array(seq, dtype=np.int32)

    model = _model(
        f"""
        g (float[2,{seq},{K}] x0) => (float[2,{seq},{K}] y)
        {{
          ln = LayerNormalization<axis=-1>(x0, Scale, Bias)
          q = MatMul(ln, Wq)
          k = MatMul(ln, Wk)
          v = MatMul(ln, Wv)
          ctx, present_k, present_v = com.microsoft.GroupQueryAttention <num_heads={NH}, kv_num_heads={NH}> (q, k, v, , , SeqLensK, TotalSeq)
          fout = MatMul(ctx, Wout)
          y = Add(x0, fout)
        }}
        """,
        initializer=[
            _f32(scale, "Scale"),
            _f32(bias, "Bias"),
            _f32(wq, "Wq"),
            _f32(wk, "Wk"),
            _f32(wv, "Wv"),
            _f32(wout, "Wout"),
            onnx.numpy_helper.from_array(seqlens_k, "SeqLensK"),
            onnx.numpy_helper.from_array(total_seq, "TotalSeq"),
        ],
    )
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_transformer_block_pruning_cpp(
        model, num_blocks_to_drop=1, seed=0, num_samples=4
    )
    onnx.checker.check_model(pruned)

    assert [n.op_type for n in pruned.graph.node] == ["Identity"]
    assert list(pruned.graph.node[0].input) == ["x0"]
    assert list(pruned.graph.node[0].output) == ["y"]

    x = np.random.default_rng(17).standard_normal((2, 5, K)).astype(np.float32)
    (pruned_y,) = _run(pruned, {"x0": x})
    np.testing.assert_array_equal(pruned_y, x)


# --- Fused SkipLayerNormalization entry norm ---------------------------------


def test_transformer_block_pruning_cpp_matches_fused_skip_layer_normalization_entry():
    # A fused com.microsoft::SkipLayerNormalization node as the block's own
    # *entry* norm -- exactly what onnxruntime's transformer optimizer
    # produces by fusing the *previous* residual Add with the following
    # LayerNormalization. This block's own merge itself stays an ordinary,
    # unfused Add.
    H = 8
    rng = np.random.default_rng(41)
    gamma = rng.standard_normal(H).astype(np.float32)
    beta = rng.standard_normal(H).astype(np.float32)
    w = (rng.standard_normal((H, H)) * 1e-6).astype(np.float32)

    def _skip_ln_node(sum_output_name):
        return onnx.helper.make_node(
            "SkipLayerNormalization",
            ["input", "skip", "Gamma", "Beta"],
            ["ln_out", "", "", sum_output_name],
            domain="com.microsoft",
            epsilon=1e-5,
        )

    initializer = [_f32(gamma, "Gamma"), _f32(beta, "Beta")]
    inputs = [
        onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 4, H]),
        onnx.helper.make_tensor_value_info("skip", onnx.TensorProto.FLOAT, [1, 4, H]),
    ]

    skip_ln = _skip_ln_node("sum_out")
    h = onnx.helper.make_node("MatMul", ["ln_out", "W"], ["h"])
    y = onnx.helper.make_node("Add", ["sum_out", "h"], ["y"])
    graph = onnx.helper.make_graph(
        [skip_ln, h, y],
        "g",
        inputs,
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 4, H])],
        initializer=initializer + [_f32(w, "W")],
        # Plain onnx.shape_inference has no inference function for a
        # com.microsoft op, so this declared value_info is what makes
        # ApplyTransformerBlockPruning's own TransformerBlockShapesMatch
        # check actually confirm the match (see test_pruning.py's own
        # identical test for the full reasoning).
        value_info=[
            onnx.helper.make_tensor_value_info(
                "sum_out", onnx.TensorProto.FLOAT, [1, 4, H]
            )
        ],
    )
    model = onnx.helper.make_model(
        graph,
        opset_imports=[
            onnx.helper.make_opsetid("", 17),
            onnx.helper.make_opsetid("com.microsoft", 1),
        ],
        ir_version=10,
    )
    onnx.checker.check_model(model)

    ref_skip_ln = _skip_ln_node("y")
    ref_graph = onnx.helper.make_graph(
        [ref_skip_ln],
        "g",
        inputs,
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 4, H])],
        initializer=initializer,
    )
    ref = onnx.helper.make_model(
        ref_graph,
        opset_imports=[
            onnx.helper.make_opsetid("", 17),
            onnx.helper.make_opsetid("com.microsoft", 1),
        ],
        ir_version=10,
    )
    onnx.checker.check_model(ref)

    pruned = onnxsim.apply_transformer_block_pruning_cpp(
        model, num_blocks_to_drop=1, seed=0, num_samples=4
    )
    onnx.checker.check_model(pruned)

    # The fused SkipLayerNormalization node itself survives, unchanged --
    # deleting it would delete the very tensor (sum_out) every rewired
    # x_out consumer now reads.
    assert [n.op_type for n in pruned.graph.node] == [
        "SkipLayerNormalization",
        "Identity",
    ]

    x_in = np.random.default_rng(42).standard_normal((1, 4, H)).astype(np.float32)
    x_skip = np.random.default_rng(43).standard_normal((1, 4, H)).astype(np.float32)
    (pruned_y,) = _run(pruned, {"input": x_in, "skip": x_skip})
    (ref_y,) = _run(ref, {"input": x_in, "skip": x_skip})
    np.testing.assert_array_equal(pruned_y, ref_y)


# --- sparsity vs num_blocks_to_drop sizing -----------------------------------


def test_transformer_block_pruning_cpp_sparsity_selects_fraction_of_matched_candidates():
    # Three stacked blocks, 0 and 2 engineered near-identity, block 1 a real
    # transform. sparsity=2/3 of 3 matched candidates rounds to 2 -- both
    # near-identity blocks should be dropped, leaving only block 1, rewired
    # straight from x0. Also exercises two NON-adjacent committed drops (0
    # and 2, sandwiching a kept block): block 2's own x_in is itself
    # downstream of block 0's own now-deleted merge, not a graph input --
    # CommitTransformerBlockDrops' own `resolve` chaining must handle this.
    H = 8
    rng = np.random.default_rng(3)
    scale = np.ones(H, dtype=np.float32)
    bias = np.zeros(H, dtype=np.float32)
    w0 = (rng.standard_normal((H, H)) * 1e-6).astype(np.float32)
    w1 = rng.standard_normal((H, H)).astype(np.float32)
    w2 = (rng.standard_normal((H, H)) * 1e-6).astype(np.float32)

    model = _model(
        f"""
        g (float[1,4,{H}] x0) => (float[1,4,{H}] y)
        {{
          ln0 = LayerNormalization<axis=-1>(x0, Scale, Bias)
          h0 = MatMul(ln0, W0)
          x1 = Add(x0, h0)

          ln1 = LayerNormalization<axis=-1>(x1, Scale, Bias)
          h1 = MatMul(ln1, W1)
          x2 = Add(x1, h1)

          ln2 = LayerNormalization<axis=-1>(x2, Scale, Bias)
          h2 = MatMul(ln2, W2)
          x3 = Add(x2, h2)

          y = Identity(x3)
        }}
        """,
        initializer=[
            _f32(scale, "Scale"),
            _f32(bias, "Bias"),
            _f32(w0, "W0"),
            _f32(w1, "W1"),
            _f32(w2, "W2"),
        ],
    )

    ref = _model(
        f"""
        g (float[1,4,{H}] x0) => (float[1,4,{H}] y)
        {{
          ln1 = LayerNormalization<axis=-1>(x0, Scale, Bias)
          h1 = MatMul(ln1, W1)
          y = Add(x0, h1)
        }}
        """,
        initializer=[_f32(scale, "Scale"), _f32(bias, "Bias"), _f32(w1, "W1")],
    )

    pruned = onnxsim.apply_transformer_block_pruning_cpp(
        model, sparsity=2 / 3, seed=0, num_samples=4
    )
    onnx.checker.check_model(pruned)
    assert [n.op_type for n in pruned.graph.node].count("LayerNormalization") == 1

    x = np.random.default_rng(11).standard_normal((1, 4, H)).astype(np.float32)
    (pruned_y,) = _run(pruned, {"x0": x})
    (ref_y,) = _run(ref, {"x0": x})
    np.testing.assert_array_equal(pruned_y, ref_y)

    pruned_py = onnxsim.apply_transformer_block_pruning(
        model, sparsity=2 / 3, seed=0, num_samples=4
    )
    assert pruned.SerializeToString() == pruned_py.SerializeToString()


def test_transformer_block_pruning_cpp_num_blocks_to_drop_caps_at_matched_candidate_count():
    # Only 2 candidates exist (both engineered near-identity);
    # num_blocks_to_drop=10 is silently capped rather than erroring -- both
    # get dropped, and the model collapses to a straight identity.
    H = 8
    rng = np.random.default_rng(31)
    scale = np.ones(H, dtype=np.float32)
    bias = np.zeros(H, dtype=np.float32)
    w0 = (rng.standard_normal((H, H)) * 1e-6).astype(np.float32)
    w1 = (rng.standard_normal((H, H)) * 1e-6).astype(np.float32)

    model = _model(
        f"""
        g (float[1,4,{H}] x0) => (float[1,4,{H}] y)
        {{
          ln0 = LayerNormalization<axis=-1>(x0, Scale, Bias)
          h0 = MatMul(ln0, W0)
          x1 = Add(x0, h0)

          ln1 = LayerNormalization<axis=-1>(x1, Scale, Bias)
          h1 = MatMul(ln1, W1)
          y = Add(x1, h1)
        }}
        """,
        initializer=[
            _f32(scale, "Scale"),
            _f32(bias, "Bias"),
            _f32(w0, "W0"),
            _f32(w1, "W1"),
        ],
    )

    pruned = onnxsim.apply_transformer_block_pruning_cpp(
        model, num_blocks_to_drop=10, seed=0, num_samples=4
    )
    onnx.checker.check_model(pruned)
    assert [n.op_type for n in pruned.graph.node] == ["Identity"]

    x = np.random.default_rng(33).standard_normal((1, 4, H)).astype(np.float32)
    (pruned_y,) = _run(pruned, {"x0": x})
    np.testing.assert_array_equal(pruned_y, x)

    pruned_py = onnxsim.apply_transformer_block_pruning(
        model, num_blocks_to_drop=10, seed=0, num_samples=4
    )
    assert pruned.SerializeToString() == pruned_py.SerializeToString()


def test_transformer_block_pruning_cpp_num_blocks_to_drop_takes_priority_over_sparsity():
    # Both given: num_blocks_to_drop wins (mirrors pruning.py's own keyword
    # precedence -- see apply_transformer_block_pruning's own signature).
    H = 8
    rng = np.random.default_rng(200)
    scale = np.ones(H, dtype=np.float32)
    bias = np.zeros(H, dtype=np.float32)
    w0 = (rng.standard_normal((H, H)) * 1e-6).astype(np.float32)
    w1 = (rng.standard_normal((H, H)) * 1e-6).astype(np.float32)

    model = _model(
        f"""
        g (float[1,4,{H}] x0) => (float[1,4,{H}] y)
        {{
          ln0 = LayerNormalization<axis=-1>(x0, Scale, Bias)
          h0 = MatMul(ln0, W0)
          x1 = Add(x0, h0)

          ln1 = LayerNormalization<axis=-1>(x1, Scale, Bias)
          h1 = MatMul(ln1, W1)
          y = Add(x1, h1)
        }}
        """,
        initializer=[
            _f32(scale, "Scale"),
            _f32(bias, "Bias"),
            _f32(w0, "W0"),
            _f32(w1, "W1"),
        ],
    )

    # sparsity=0.0 alone would drop nothing; num_blocks_to_drop=1 overrides.
    pruned = onnxsim.apply_transformer_block_pruning_cpp(
        model, sparsity=0.0, num_blocks_to_drop=1, seed=0, num_samples=4
    )
    assert [n.op_type for n in pruned.graph.node].count("LayerNormalization") == 1
    onnx.checker.check_model(pruned)


# --- Decline cases: left completely untouched --------------------------------


def test_transformer_block_pruning_cpp_declines_fused_entry_when_sum_output_absent():
    H = 8
    rng = np.random.default_rng(49)
    gamma = rng.standard_normal(H).astype(np.float32)
    beta = rng.standard_normal(H).astype(np.float32)
    w = rng.standard_normal((H, H)).astype(np.float32)

    skip_ln = onnx.helper.make_node(
        "SkipLayerNormalization",
        ["input", "skip", "Gamma", "Beta"],
        ["ln_out"],  # no fourth output declared at all
        domain="com.microsoft",
        epsilon=1e-5,
    )
    h = onnx.helper.make_node("MatMul", ["ln_out", "W"], ["h"])
    y = onnx.helper.make_node("Add", ["input", "h"], ["y"])
    graph = onnx.helper.make_graph(
        [skip_ln, h, y],
        "g",
        [
            onnx.helper.make_tensor_value_info(
                "input", onnx.TensorProto.FLOAT, [1, 4, H]
            ),
            onnx.helper.make_tensor_value_info(
                "skip", onnx.TensorProto.FLOAT, [1, 4, H]
            ),
        ],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 4, H])],
        initializer=[_f32(gamma, "Gamma"), _f32(beta, "Beta"), _f32(w, "W")],
    )
    model = onnx.helper.make_model(
        graph,
        opset_imports=[
            onnx.helper.make_opsetid("", 17),
            onnx.helper.make_opsetid("com.microsoft", 1),
        ],
        ir_version=10,
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_transformer_block_pruning_cpp(
        model, num_blocks_to_drop=1, seed=0, num_samples=4
    )
    assert pruned.SerializeToString() == model.SerializeToString()


def test_transformer_block_pruning_cpp_declines_kv_cache_bearing_attention_block():
    # present_k/present_v are declared as graph outputs -- the generic "no
    # block-internal node's own output may leak outside the block" check
    # catches this, with no KV-cache-specific detection at all.
    K, NH, D = 8, 2, 4
    Nqkv = NH * D
    rng = np.random.default_rng(19)
    scale = np.ones(K, dtype=np.float32)
    bias = np.zeros(K, dtype=np.float32)
    wq = rng.standard_normal((K, Nqkv)).astype(np.float32)
    wk = rng.standard_normal((K, Nqkv)).astype(np.float32)
    wv = rng.standard_normal((K, Nqkv)).astype(np.float32)
    wout = rng.standard_normal((Nqkv, K)).astype(np.float32)

    model = _model(
        f"""
        g (float[2,5,{K}] x0) => (float[2,5,{K}] y, float[2,5,{Nqkv}] present_k, float[2,5,{Nqkv}] present_v)
        {{
          ln = LayerNormalization<axis=-1>(x0, Scale, Bias)
          q = MatMul(ln, Wq)
          k = MatMul(ln, Wk)
          v = MatMul(ln, Wv)
          ctx, present_k, present_v = com.microsoft.GroupQueryAttention <num_heads={NH}, kv_num_heads={NH}> (q, k, v)
          fout = MatMul(ctx, Wout)
          y = Add(x0, fout)
        }}
        """,
        initializer=[
            _f32(scale, "Scale"),
            _f32(bias, "Bias"),
            _f32(wq, "Wq"),
            _f32(wk, "Wk"),
            _f32(wv, "Wv"),
            _f32(wout, "Wout"),
        ],
    )
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_transformer_block_pruning_cpp(
        model, num_blocks_to_drop=1, seed=0, num_samples=4
    )
    assert pruned.SerializeToString() == model.SerializeToString()


def test_transformer_block_pruning_cpp_declines_when_intermediate_tensor_has_external_consumer():
    H = 8
    rng = np.random.default_rng(15)
    scale = np.ones(H, dtype=np.float32)
    bias = np.zeros(H, dtype=np.float32)
    w = rng.standard_normal((H, H)).astype(np.float32)

    model = _model(
        f"""
        g (float[1,4,{H}] x0) => (float[1,4,{H}] y, float[1,4,{H}] ln_out)
        {{
          ln = LayerNormalization<axis=-1>(x0, Scale, Bias)
          h = MatMul(ln, W)
          y = Add(x0, h)
          ln_out = Identity(ln)
        }}
        """,
        initializer=[_f32(scale, "Scale"), _f32(bias, "Bias"), _f32(w, "W")],
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_transformer_block_pruning_cpp(
        model, num_blocks_to_drop=1, seed=0, num_samples=4
    )
    assert pruned.SerializeToString() == model.SerializeToString()


def test_transformer_block_pruning_cpp_declines_shape_broadcasting_merge():
    # F's own final output is tiled to a *wider* batch dimension than
    # x_in's own -- the residual Add would silently broadcast x_in up to
    # match, so replacing every x_out consumer with (narrower) x_in
    # directly would be shape-unsafe. Declined via TransformerBlockShapesMatch,
    # not guessed at -- this exercises the real onnx::shape_inference path
    # (not just whatever value_info the input model happens to carry).
    H = 8
    rng = np.random.default_rng(21)
    scale = np.ones(H, dtype=np.float32)
    bias = np.zeros(H, dtype=np.float32)
    w = rng.standard_normal((H, H)).astype(np.float32)
    repeats = np.array([3, 1, 1], dtype=np.int64)

    model = _model(
        f"""
        g (float[1,4,{H}] x0) => (float[3,4,{H}] y)
        {{
          ln = LayerNormalization<axis=-1>(x0, Scale, Bias)
          h = MatMul(ln, W)
          h_tiled = Tile(h, Repeats)
          y = Add(x0, h_tiled)
        }}
        """,
        initializer=[
            _f32(scale, "Scale"),
            _f32(bias, "Bias"),
            _f32(w, "W"),
            onnx.numpy_helper.from_array(repeats, "Repeats"),
        ],
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_transformer_block_pruning_cpp(
        model, num_blocks_to_drop=1, seed=0, num_samples=4
    )
    assert pruned.SerializeToString() == model.SerializeToString()


def test_transformer_block_pruning_cpp_declines_when_f_reads_x_in_directly():
    # No LayerNorm/RMSNorm at all -- F reads x0 raw. No candidate is even
    # found.
    H = 8
    rng = np.random.default_rng(25)
    w = rng.standard_normal((H, H)).astype(np.float32)

    model = _model(
        f"""
        g (float[1,4,{H}] x0) => (float[1,4,{H}] y)
        {{
          h = MatMul(x0, W)
          y = Add(x0, h)
        }}
        """,
        initializer=[_f32(w, "W")],
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_transformer_block_pruning_cpp(
        model, num_blocks_to_drop=1, seed=0, num_samples=4
    )
    assert pruned.SerializeToString() == model.SerializeToString()


def test_transformer_block_pruning_cpp_no_candidates_returns_unchanged_copy():
    H = 8
    rng = np.random.default_rng(35)
    w = rng.standard_normal((H, H)).astype(np.float32)
    model = _model(
        f"""
        g (float[1,4,{H}] x0) => (float[1,4,{H}] y)
        {{
          y = MatMul(x0, W)
        }}
        """,
        initializer=[_f32(w, "W")],
    )

    pruned = onnxsim.apply_transformer_block_pruning_cpp(model, num_blocks_to_drop=1)
    assert pruned.SerializeToString() == model.SerializeToString()


def test_transformer_block_pruning_cpp_zero_sparsity_leaves_model_untouched():
    H = 8
    rng = np.random.default_rng(37)
    scale = np.ones(H, dtype=np.float32)
    bias = np.zeros(H, dtype=np.float32)
    w = (rng.standard_normal((H, H)) * 1e-6).astype(np.float32)
    model = _model(
        f"""
        g (float[1,4,{H}] x0) => (float[1,4,{H}] y)
        {{
          ln = LayerNormalization<axis=-1>(x0, Scale, Bias)
          h = MatMul(ln, W)
          y = Add(x0, h)
        }}
        """,
        initializer=[_f32(scale, "Scale"), _f32(bias, "Bias"), _f32(w, "W")],
    )

    pruned = onnxsim.apply_transformer_block_pruning_cpp(model, sparsity=0.0)
    assert pruned.SerializeToString() == model.SerializeToString()


# --- Interior-overlap-skip: two independently-matched candidates whose own
# block_nodes overlap -----------------------------------------------------


def test_transformer_block_pruning_cpp_interior_overlap_skip_matches_python_reference():
    # Constructs a genuine "candidate B's own interior fully contains
    # candidate A's own interior" topology -- the unusual-but-possible case
    # SelectDroppableBlocks/_select_droppable_blocks' own docstring
    # describes, where only ONE of two independently-VALID candidates can
    # ever safely be committed:
    #
    #   Block A: x1 = x0 + MatMul(LN0(x0), WA)               (candidate A)
    #   Block B: y  = x2 + (MatMul(LN_B(x2), WB2) + x1)      (candidate B)
    #
    # Block B's own F (its "other" merge operand) is built by summing a
    # normal LN_B(x2)-derived term with x1 -- Block A's own raw output --
    # used as a plain additive term, not through any norm. Walking B's own
    # backward search from its own merge therefore passes straight through
    # Add_A (the node that produces x1) and everything upstream of it
    # (MatMul_A, LN0), collecting them as ordinary interior nodes of B, in
    # ADDITION to B's own genuine LN_B(x2) boundary elsewhere in the sum.
    # Since x1's own only real consumer is that one sum node inside F_B (no
    # other reader, no graph-output exposure), and x_out (a merge node's own
    # primary output) is exempt from the "no external consumer" check for
    # ITS OWN candidate, BOTH candidate A (interior {Add_A, MatMul_A, LN0})
    # and candidate B (interior a strict superset: {Add_B, the sum node,
    # MatMul_B2, LN_B, Add_A, MatMul_A, LN0}) independently pass every
    # safety check and are both matched -- with block_nodes(A) subset-of
    # block_nodes(B).
    #
    # num_blocks_to_drop=2 (both "found" candidates) can therefore never
    # actually commit two independent drops: whichever candidate the
    # ranking tries first gets committed, and the second is SKIPPED outright
    # (its own block_nodes overlaps the first commit's), never causing the
    # whole call to decline. The primary assertion here is exact parity
    # with the pure-Python reference on the identical model + calibration
    # data -- proving the C++ port makes the identical skip decision the
    # Python original does, whichever way the (calibration-dependent)
    # ranking happens to resolve.
    H = 8
    rng = np.random.default_rng(500)
    scale0 = np.ones(H, dtype=np.float32)
    bias0 = np.zeros(H, dtype=np.float32)
    wa = rng.standard_normal((H, H)).astype(np.float32) * 0.1
    scale_b = np.ones(H, dtype=np.float32)
    bias_b = np.zeros(H, dtype=np.float32)
    wb2 = rng.standard_normal((H, H)).astype(np.float32) * 0.1

    model = _model(
        f"""
        g (float[1,4,{H}] x0, float[1,4,{H}] x2) => (float[1,4,{H}] y)
        {{
          ln0 = LayerNormalization<axis=-1>(x0, Scale0, Bias0)
          hA = MatMul(ln0, WA)
          x1 = Add(x0, hA)

          lnB = LayerNormalization<axis=-1>(x2, ScaleB, BiasB)
          hB2 = MatMul(lnB, WB2)
          hBFinal = Add(hB2, x1)
          y = Add(x2, hBFinal)
        }}
        """,
        initializer=[
            _f32(scale0, "Scale0"),
            _f32(bias0, "Bias0"),
            _f32(wa, "WA"),
            _f32(scale_b, "ScaleB"),
            _f32(bias_b, "BiasB"),
            _f32(wb2, "WB2"),
        ],
    )
    onnx.checker.check_model(model)

    rng_cal = np.random.default_rng(501)
    calibration_data = [
        {
            "x0": rng_cal.standard_normal((1, 4, H)).astype(np.float32),
            "x2": rng_cal.standard_normal((1, 4, H)).astype(np.float32),
        }
        for _ in range(3)
    ]

    pruned_cpp = onnxsim.apply_transformer_block_pruning_cpp(
        model, calibration_data=calibration_data, num_blocks_to_drop=2
    )
    onnx.checker.check_model(pruned_cpp)
    pruned_py = onnxsim.apply_transformer_block_pruning(
        model, calibration_data=calibration_data, num_blocks_to_drop=2
    )
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    # Confirm the graph really did change (at least one candidate was
    # committed) -- this isn't a "nothing matched" no-op.
    assert pruned_cpp.SerializeToString() != model.SerializeToString()

    # And confirm the result still executes correctly (the rewiring/
    # deletion mechanics produced a valid, runnable graph even in this
    # overlap-heavy topology).
    x0 = np.random.default_rng(502).standard_normal((1, 4, H)).astype(np.float32)
    x2 = np.random.default_rng(503).standard_normal((1, 4, H)).astype(np.float32)
    _run(pruned_cpp, {"x0": x0, "x2": x2})  # must not raise


# --- Cross-check against the pure-Python reference on random multi-block
# models -----------------------------------------------------------------


def test_transformer_block_pruning_cpp_matches_python_reference_random_model():
    H = 12
    rng = np.random.default_rng(600)
    scales = [np.ones(H, dtype=np.float32) for _ in range(4)]
    biases = [np.zeros(H, dtype=np.float32) for _ in range(4)]
    # A mix of near-identity and real-transform blocks so the ranking is
    # meaningfully exercised.
    weights = [
        (rng.standard_normal((H, H)) * (1e-6 if i % 2 == 0 else 1.0)).astype(np.float32)
        for i in range(4)
    ]

    body_lines = []
    prev = "x0"
    for i in range(4):
        body_lines.append(
            f"ln{i} = LayerNormalization<axis=-1>({prev}, Scale{i}, Bias{i})"
        )
        body_lines.append(f"h{i} = MatMul(ln{i}, W{i})")
        nxt = "y" if i == 3 else f"x{i + 1}"
        body_lines.append(f"{nxt} = Add({prev}, h{i})")
        prev = nxt
    body = "\n".join(body_lines)

    initializer = []
    for i in range(4):
        initializer.append(_f32(scales[i], f"Scale{i}"))
        initializer.append(_f32(biases[i], f"Bias{i}"))
        initializer.append(_f32(weights[i], f"W{i}"))

    model = _model(
        f"""
        g (float[1,4,{H}] x0) => (float[1,4,{H}] y)
        {{
        {body}
        }}
        """,
        initializer=initializer,
    )
    onnx.checker.check_model(model)

    rng_cal = np.random.default_rng(601)
    calibration_data = [
        {"x0": rng_cal.standard_normal((1, 4, H)).astype(np.float32)} for _ in range(3)
    ]

    pruned_cpp = onnxsim.apply_transformer_block_pruning_cpp(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    pruned_py = onnxsim.apply_transformer_block_pruning(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned_cpp)
    assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()
    # Sanity: some real reduction happened.
    assert len(pruned_cpp.graph.node) < len(model.graph.node)


# --- Error handling ------------------------------------------------------


def test_transformer_block_pruning_cpp_missing_calibration_input_raises():
    H = 8
    rng = np.random.default_rng(700)
    scale = np.ones(H, dtype=np.float32)
    bias = np.zeros(H, dtype=np.float32)
    w = rng.standard_normal((H, H)).astype(np.float32)
    model = _model(
        f"""
        g (float[1,4,{H}] x0) => (float[1,4,{H}] y)
        {{
          ln = LayerNormalization<axis=-1>(x0, Scale, Bias)
          h = MatMul(ln, W)
          y = Add(x0, h)
        }}
        """,
        initializer=[_f32(scale, "Scale"), _f32(bias, "Bias"), _f32(w, "W")],
    )
    bad_batch = {"NotX0": np.zeros((1, 4, H), dtype=np.float32)}
    with pytest.raises(Exception):
        onnxsim.apply_transformer_block_pruning_cpp(
            model, calibration_data=[bad_batch], num_blocks_to_drop=1
        )


def test_transformer_block_pruning_cpp_negative_num_blocks_to_drop_raises():
    H = 8
    rng = np.random.default_rng(701)
    scale = np.ones(H, dtype=np.float32)
    bias = np.zeros(H, dtype=np.float32)
    w = rng.standard_normal((H, H)).astype(np.float32)
    model = _model(
        f"""
        g (float[1,4,{H}] x0) => (float[1,4,{H}] y)
        {{
          ln = LayerNormalization<axis=-1>(x0, Scale, Bias)
          h = MatMul(ln, W)
          y = Add(x0, h)
        }}
        """,
        initializer=[_f32(scale, "Scale"), _f32(bias, "Bias"), _f32(w, "W")],
    )
    with pytest.raises(Exception):
        onnxsim.apply_transformer_block_pruning_cpp(model, num_blocks_to_drop=-1)


def test_transformer_block_pruning_cpp_sparsity_out_of_range_raises():
    H = 8
    rng = np.random.default_rng(702)
    scale = np.ones(H, dtype=np.float32)
    bias = np.zeros(H, dtype=np.float32)
    w = rng.standard_normal((H, H)).astype(np.float32)
    model = _model(
        f"""
        g (float[1,4,{H}] x0) => (float[1,4,{H}] y)
        {{
          ln = LayerNormalization<axis=-1>(x0, Scale, Bias)
          h = MatMul(ln, W)
          y = Add(x0, h)
        }}
        """,
        initializer=[_f32(scale, "Scale"), _f32(bias, "Bias"), _f32(w, "W")],
    )
    with pytest.raises(Exception):
        onnxsim.apply_transformer_block_pruning_cpp(model, sparsity=1.5)


# --- Default (auto-generated) calibration data -------------------------------


def test_transformer_block_pruning_cpp_default_calibration_data_runs():
    H = 8
    rng = np.random.default_rng(800)
    scale = np.ones(H, dtype=np.float32)
    bias = np.zeros(H, dtype=np.float32)
    w = (rng.standard_normal((H, H)) * 1e-6).astype(np.float32)
    model = _model(
        f"""
        g (float[1,4,{H}] x0) => (float[1,4,{H}] y)
        {{
          ln = LayerNormalization<axis=-1>(x0, Scale, Bias)
          h = MatMul(ln, W)
          y = Add(x0, h)
        }}
        """,
        initializer=[_f32(scale, "Scale"), _f32(bias, "Bias"), _f32(w, "W")],
    )
    pruned = onnxsim.apply_transformer_block_pruning_cpp(
        model, num_samples=4, seed=5, num_blocks_to_drop=1
    )
    onnx.checker.check_model(pruned)
    assert [n.op_type for n in pruned.graph.node] == ["Identity"]
