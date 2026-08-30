"""Tests for ``onnxsim.apply_attention_head_pruning_cpp`` -- the C++-backed
port of ``onnxsim.apply_attention_head_pruning`` (see
``onnxsim/structured_pruning_entry.cpp``'s "Attention-head pruning" section).
Data-free (magnitude/Frobenius-norm) only -- the calibration-driven Wanda
variant (``onnxsim.apply_attention_head_wanda_pruning``) is not ported,
matching this codebase's established C++-port scope decision. Tests here are
adapted from ``test_pruning.py``'s own ``apply_attention_head_pruning``
coverage for all three matched op families: plain ``com.microsoft::Attention``
(merged QKV weight), ``com.microsoft::GroupQueryAttention``, and the plain
``ai.onnx::Attention`` op (opset 24+).
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


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _head_idx(keep_heads, d):
    return np.concatenate([np.arange(h * d, (h + 1) * d) for h in keep_heads])


def _oracle_keep_heads(wqkv, nq, nk, nv, num_heads, keep_count):
    dq, dk, dv = nq // num_heads, nk // num_heads, nv // num_heads
    wq, wk, wv = wqkv[:, :nq], wqkv[:, nq : nq + nk], wqkv[:, nq + nk :]
    importance = np.zeros(num_heads)
    for h in range(num_heads):
        block = np.concatenate(
            [
                wq[:, h * dq : (h + 1) * dq],
                wk[:, h * dk : (h + 1) * dk],
                wv[:, h * dv : (h + 1) * dv],
            ],
            axis=1,
        )
        importance[h] = np.linalg.norm(block)
    return np.sort(np.argsort(-importance)[:keep_count])


def _oracle_keep_groups(wq, wk, wv, num_heads, kv_num_heads, head_size, keep_count):
    group_size = num_heads // kv_num_heads
    importance = np.zeros(kv_num_heads)
    for kv in range(kv_num_heads):
        q_block = np.concatenate(
            [
                wq[:, h * head_size : (h + 1) * head_size]
                for h in range(kv * group_size, (kv + 1) * group_size)
            ],
            axis=1,
        )
        k_block = wk[:, kv * head_size : (kv + 1) * head_size]
        v_block = wv[:, kv * head_size : (kv + 1) * head_size]
        importance[kv] = np.linalg.norm(
            np.concatenate([q_block, k_block, v_block], axis=1)
        )
    return np.sort(np.argsort(-importance)[:keep_count])


def _group_q_heads(keep_groups, group_size):
    return np.concatenate(
        [np.arange(g * group_size, (g + 1) * group_size) for g in keep_groups]
    )


# --- Plain com.microsoft::Attention (merged QKV weight) ---------------------


def _attention_model(
    K=8,
    H=4,
    D=4,
    Out=6,
    seed=0,
    batch=2,
    seq=5,
    bias=True,
    with_reshape=False,
    wqkv=None,
    bqkv=None,
    wout=None,
    num_heads=None,
):
    rng = np.random.default_rng(seed)
    Nq = Nk = Nv = H * D
    if wqkv is None:
        wqkv = rng.standard_normal((K, Nq + Nk + Nv)).astype(np.float32)
    if wout is None:
        wout = rng.standard_normal((Nv, Out)).astype(np.float32)
    if bias and bqkv is None:
        bqkv = rng.standard_normal((Nq + Nk + Nv,)).astype(np.float32)
    heads = H if num_heads is None else num_heads

    initializer = [_f32(wqkv, "Wqkv"), _f32(wout, "Wout")]
    qkv_inputs = "X, Wqkv"
    if bias:
        initializer.append(_f32(bqkv, "Bqkv"))
        qkv_inputs = "X, Wqkv, Bqkv"

    if with_reshape:
        shape = np.array([batch, seq, Nv], dtype=np.int64)
        initializer.append(onnx.numpy_helper.from_array(shape, "Shape"))
        tail = "ctx2 = Reshape(ctx, Shape)\n          Y = MatMul(ctx2, Wout)"
    else:
        tail = "Y = MatMul(ctx, Wout)"

    body = f"""
        g (float[batch,seq,{K}] X) => (float[batch,seq,{Out}] Y)
        {{
          ctx = com.microsoft.Attention <num_heads={heads}, qkv_hidden_sizes=[{Nq},{Nk},{Nv}]> ({qkv_inputs})
          {tail}
        }}
        """
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 17, "com.microsoft": 1]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model, dict(
        K=K, H=H, D=D, Out=Out, Nq=Nq, Nk=Nk, Nv=Nv, wqkv=wqkv, bqkv=bqkv, wout=wout
    )


def _attention_node(model):
    return next(n for n in model.graph.node if n.op_type == "Attention")


def _attention_attrs(node):
    num_heads = next(a.i for a in node.attribute if a.name == "num_heads")
    qkv = next(list(a.ints) for a in node.attribute if a.name == "qkv_hidden_sizes")
    return num_heads, qkv


def test_cpp_attention_head_pruning_shrinks_matched_block():
    model, cfg = _attention_model(K=8, H=4, D=4, Out=6)
    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    node = _attention_node(pruned)
    num_heads, qkv = _attention_attrs(node)
    assert num_heads == 2
    assert qkv == [8, 8, 8]

    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["Wqkv"].dims) == [8, 24]
    assert list(inits["Bqkv"].dims) == [24]
    assert list(inits["Wout"].dims) == [8, 6]


def test_cpp_attention_head_pruning_matches_manual_head_deletion_exactly():
    model, cfg = _attention_model(K=8, H=4, D=4, Out=6, seed=1)
    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    keep = _oracle_keep_heads(cfg["wqkv"], cfg["Nq"], cfg["Nk"], cfg["Nv"], cfg["H"], 2)
    d = cfg["D"]
    qi, ki, vi = (
        _head_idx(keep, d),
        _head_idx(keep, d) + cfg["Nq"],
        _head_idx(keep, d) + cfg["Nq"] + cfg["Nk"],
    )
    all_idx = np.concatenate([qi, ki, vi])
    oracle_wqkv = cfg["wqkv"][:, all_idx]
    oracle_bqkv = cfg["bqkv"][all_idx]
    oracle_wout = cfg["wout"][_head_idx(keep, d), :]
    oracle, _ = _attention_model(
        K=cfg["K"],
        H=2,
        D=d,
        Out=cfg["Out"],
        seed=1,
        wqkv=oracle_wqkv,
        bqkv=oracle_bqkv,
        wout=oracle_wout,
        num_heads=2,
    )

    rng = np.random.default_rng(2)
    x = rng.standard_normal((2, 5, cfg["K"])).astype(np.float32)
    (y_pruned,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y_pruned, y_oracle, rtol=1e-4, atol=1e-4)


def test_cpp_attention_head_pruning_reshape_hop_is_recognized_and_shape_updated():
    model, cfg = _attention_model(K=8, H=4, D=4, Out=6, seed=3, with_reshape=True)
    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.25)
    onnx.checker.check_model(pruned)
    assert [n.op_type for n in pruned.graph.node] == ["Attention", "Reshape", "MatMul"]

    node = _attention_node(pruned)
    num_heads, _ = _attention_attrs(node)
    assert num_heads == 3  # round(4 - 4*0.25) == 3

    shape = onnx.numpy_helper.to_array(
        next(t for t in pruned.graph.initializer if t.name == "Shape")
    )
    assert shape[-1] == 3 * cfg["D"]  # updated to the new (post-prune) Nv

    rng = np.random.default_rng(4)
    x = rng.standard_normal((2, 5, cfg["K"])).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    assert y.shape == (2, 5, cfg["Out"])


def test_cpp_attention_head_pruning_mismatched_consumer_reduction_dim_is_left_untouched():
    K, H, D, Out = 8, 4, 4, 6
    Nqkv = H * D
    rng = np.random.default_rng(6)
    wqkv = rng.standard_normal((K, 3 * Nqkv)).astype(np.float32)
    wout_wrong = rng.standard_normal((Nqkv + 1, Out)).astype(np.float32)  # off-by-one
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 17, "com.microsoft": 1]
        >
        g (float[2,5,{K}] X) => (float[2,5,{Out}] Y)
        {{
          ctx = com.microsoft.Attention <num_heads={H}, qkv_hidden_sizes=[{Nqkv},{Nqkv},{Nqkv}]> (X, Wqkv)
          padded = Pad <pads = [0,0,0,0,0,1]> (ctx)
          Y = MatMul(padded, Wout)
        }}
        """
    )
    model.graph.initializer.extend([_f32(wqkv, "Wqkv"), _f32(wout_wrong, "Wout")])

    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    inits_before = {
        t.name: onnx.numpy_helper.to_array(t) for t in model.graph.initializer
    }
    inits_after = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer
    }
    for name in inits_before:
        np.testing.assert_array_equal(inits_before[name], inits_after[name])


def test_cpp_attention_head_pruning_zero_sparsity_is_a_no_op():
    model, cfg = _attention_model(K=8, H=4, D=4, Out=6, seed=7)
    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.0)
    node = _attention_node(pruned)
    num_heads, qkv = _attention_attrs(node)
    assert num_heads == cfg["H"]
    assert qkv == [cfg["Nq"], cfg["Nk"], cfg["Nv"]]
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["Wqkv"], cfg["wqkv"])


def test_cpp_attention_head_pruning_invalid_sparsity_raises():
    model, _ = _attention_model(K=8, H=4, D=4, Out=6)
    with pytest.raises(Exception):
        onnxsim.apply_attention_head_pruning_cpp(model, sparsity=1.0)
    with pytest.raises(Exception):
        onnxsim.apply_attention_head_pruning_cpp(model, sparsity=-0.1)


# --- com.microsoft::GroupQueryAttention (separate Q/K/V producers) ---------


def _gqa_model(
    K=8,
    H=4,
    KVH=2,
    D=8,
    Out=6,
    seed=0,
    batch=2,
    seq=5,
    bias=False,
    with_reshape=False,
    wq=None,
    wk=None,
    wv=None,
    bq=None,
    bk=None,
    bv=None,
    wout=None,
    past_kv=None,  # None (empty) | "nonempty" (constant) | "dynamic" (graph input)
):
    rng = np.random.default_rng(seed)
    Nq, Nkv = H * D, KVH * D
    if wq is None:
        wq = rng.standard_normal((K, Nq)).astype(np.float32)
    if wk is None:
        wk = rng.standard_normal((K, Nkv)).astype(np.float32)
    if wv is None:
        wv = rng.standard_normal((K, Nkv)).astype(np.float32)
    if wout is None:
        wout = rng.standard_normal((Nq, Out)).astype(np.float32)

    initializer = [_f32(wq, "Wq"), _f32(wk, "Wk"), _f32(wv, "Wv"), _f32(wout, "Wout")]
    q_op, k_op, v_op = "MatMul(X, Wq)", "MatMul(X, Wk)", "MatMul(X, Wv)"
    if bias:
        if bq is None:
            bq = rng.standard_normal((Nq,)).astype(np.float32)
        if bk is None:
            bk = rng.standard_normal((Nkv,)).astype(np.float32)
        if bv is None:
            bv = rng.standard_normal((Nkv,)).astype(np.float32)
        initializer += [_f32(bq, "Bq"), _f32(bk, "Bk"), _f32(bv, "Bv")]
        q_op, k_op, v_op = "Gemm(X, Wq, Bq)", "Gemm(X, Wk, Bk)", "Gemm(X, Wv, Bv)"

    initializer.append(
        onnx.numpy_helper.from_array(
            np.full((batch,), seq - 1, dtype=np.int32), "SeqLensK"
        )
    )
    initializer.append(
        onnx.numpy_helper.from_array(np.array(seq, dtype=np.int32), "TotalSeq")
    )

    operands = ["q", "k", "v"]
    extra_graph_inputs = ""
    if past_kv == "nonempty":
        past_key = rng.standard_normal((batch, KVH, 1, D)).astype(np.float32)
        past_value = rng.standard_normal((batch, KVH, 1, D)).astype(np.float32)
        initializer += [_f32(past_key, "PastKey"), _f32(past_value, "PastValue")]
        operands += ["PastKey", "PastValue"]
    elif past_kv == "dynamic":
        operands += ["PastKeyIn", "PastValueIn"]
        extra_graph_inputs = (
            f", float[{batch},{KVH},1,{D}] PastKeyIn"
            f", float[{batch},{KVH},1,{D}] PastValueIn"
        )
    else:
        operands += ["", ""]
    operands += ["SeqLensK", "TotalSeq"]

    if with_reshape:
        shape = np.array([batch, seq, Nq], dtype=np.int64)
        initializer.append(onnx.numpy_helper.from_array(shape, "Shape"))
        tail = "ctx2 = Reshape(ctx, Shape)\n          Y = MatMul(ctx2, Wout)"
    else:
        tail = "Y = MatMul(ctx, Wout)"

    body = f"""
        g (float[{batch},{seq},{K}] X{extra_graph_inputs}) => (float[{batch},{seq},{Out}] Y)
        {{
          q = {q_op}
          k = {k_op}
          v = {v_op}
          ctx, pk, pv = com.microsoft.GroupQueryAttention <num_heads={H}, kv_num_heads={KVH}> ({", ".join(operands)})
          {tail}
        }}
        """

    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 17, "com.microsoft": 1]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model, dict(
        K=K,
        H=H,
        KVH=KVH,
        D=D,
        Out=Out,
        Nq=Nq,
        Nkv=Nkv,
        wq=wq,
        wk=wk,
        wv=wv,
        bq=bq,
        bk=bk,
        bv=bv,
        wout=wout,
        batch=batch,
        seq=seq,
    )


def _gqa_node(model):
    return next(n for n in model.graph.node if n.op_type == "GroupQueryAttention")


def _gqa_attrs(node):
    num_heads = next(a.i for a in node.attribute if a.name == "num_heads")
    kv_num_heads = next(a.i for a in node.attribute if a.name == "kv_num_heads")
    return num_heads, kv_num_heads


def test_cpp_gqa_pruning_shrinks_matched_block():
    model, cfg = _gqa_model(K=8, H=4, KVH=4, D=8, Out=6)
    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    node = _gqa_node(pruned)
    num_heads, kv_num_heads = _gqa_attrs(node)
    assert num_heads == 2
    assert kv_num_heads == 2

    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["Wq"].dims) == [8, 16]
    assert list(inits["Wk"].dims) == [8, 16]
    assert list(inits["Wv"].dims) == [8, 16]
    assert list(inits["Wout"].dims) == [16, 6]


def test_cpp_gqa_pruning_zero_sparsity_is_a_no_op():
    model, cfg = _gqa_model(K=8, H=8, KVH=2, D=8, Out=6, seed=7)
    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.0)
    node = _gqa_node(pruned)
    num_heads, kv_num_heads = _gqa_attrs(node)
    assert num_heads == cfg["H"]
    assert kv_num_heads == cfg["KVH"]
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["Wq"], cfg["wq"])
    np.testing.assert_array_equal(inits["Wk"], cfg["wk"])
    np.testing.assert_array_equal(inits["Wv"], cfg["wv"])


def test_cpp_gqa_pruning_unequal_heads_drops_whole_groups_and_preserves_ratio():
    model, cfg = _gqa_model(K=8, H=8, KVH=4, D=8, Out=6, seed=11)
    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    node = _gqa_node(pruned)
    num_heads, kv_num_heads = _gqa_attrs(node)
    assert kv_num_heads == 2
    assert num_heads == 4
    assert num_heads // kv_num_heads == cfg["H"] // cfg["KVH"]

    group_size = cfg["H"] // cfg["KVH"]
    keep_groups = _oracle_keep_groups(
        cfg["wq"], cfg["wk"], cfg["wv"], cfg["H"], cfg["KVH"], cfg["D"], 2
    )
    keep_q_heads = _group_q_heads(keep_groups, group_size)
    d = cfg["D"]

    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["Wq"], cfg["wq"][:, _head_idx(keep_q_heads, d)])
    np.testing.assert_array_equal(inits["Wk"], cfg["wk"][:, _head_idx(keep_groups, d)])
    np.testing.assert_array_equal(inits["Wv"], cfg["wv"][:, _head_idx(keep_groups, d)])


def test_cpp_gqa_pruning_matches_oracle_exactly():
    model, cfg = _gqa_model(K=8, H=8, KVH=2, D=8, Out=6, seed=1)
    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    node = _gqa_node(pruned)
    num_heads, kv_num_heads = _gqa_attrs(node)
    group_size = cfg["H"] // cfg["KVH"]
    assert kv_num_heads == 1
    assert num_heads == kv_num_heads * group_size

    keep_groups = _oracle_keep_groups(
        cfg["wq"], cfg["wk"], cfg["wv"], cfg["H"], cfg["KVH"], cfg["D"], kv_num_heads
    )
    keep_q_heads = _group_q_heads(keep_groups, group_size)
    d = cfg["D"]
    q_idx, kv_idx = _head_idx(keep_q_heads, d), _head_idx(keep_groups, d)

    oracle, _ = _gqa_model(
        K=cfg["K"],
        H=num_heads,
        KVH=kv_num_heads,
        D=d,
        Out=cfg["Out"],
        seed=1,
        wq=cfg["wq"][:, q_idx],
        wk=cfg["wk"][:, kv_idx],
        wv=cfg["wv"][:, kv_idx],
        wout=cfg["wout"][q_idx, :],
        batch=cfg["batch"],
        seq=cfg["seq"],
    )

    rng = np.random.default_rng(2)
    x = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(np.float32)
    (y_pruned,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y_pruned, y_oracle, rtol=1e-4, atol=1e-4)


def test_cpp_gqa_pruning_slices_bias_when_producer_has_one():
    model, cfg = _gqa_model(K=8, H=4, KVH=2, D=8, Out=6, seed=14, bias=True)
    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)

    node = _gqa_node(pruned)
    num_heads, kv_num_heads = _gqa_attrs(node)
    group_size = cfg["H"] // cfg["KVH"]
    assert kv_num_heads == 1
    assert num_heads == group_size

    keep_groups = _oracle_keep_groups(
        cfg["wq"], cfg["wk"], cfg["wv"], cfg["H"], cfg["KVH"], cfg["D"], kv_num_heads
    )
    keep_q_heads = _group_q_heads(keep_groups, group_size)
    d = cfg["D"]
    q_idx, kv_idx = _head_idx(keep_q_heads, d), _head_idx(keep_groups, d)

    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["Wq"], cfg["wq"][:, q_idx])
    np.testing.assert_array_equal(inits["Wk"], cfg["wk"][:, kv_idx])
    np.testing.assert_array_equal(inits["Wv"], cfg["wv"][:, kv_idx])
    np.testing.assert_array_equal(inits["Bq"], cfg["bq"][q_idx])
    np.testing.assert_array_equal(inits["Bk"], cfg["bk"][kv_idx])
    np.testing.assert_array_equal(inits["Bv"], cfg["bv"][kv_idx])


def test_cpp_gqa_pruning_reshape_hop_is_recognized_and_shape_updated():
    model, cfg = _gqa_model(K=8, H=4, KVH=2, D=8, Out=6, seed=3, with_reshape=True)
    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    assert [n.op_type for n in pruned.graph.node] == [
        "MatMul",
        "MatMul",
        "MatMul",
        "GroupQueryAttention",
        "Reshape",
        "MatMul",
    ]

    node = _gqa_node(pruned)
    num_heads, kv_num_heads = _gqa_attrs(node)
    assert kv_num_heads == 1
    assert num_heads == 2

    shape = onnx.numpy_helper.to_array(
        next(t for t in pruned.graph.initializer if t.name == "Shape")
    )
    assert shape[-1] == num_heads * cfg["D"]

    rng = np.random.default_rng(4)
    x = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    assert y.shape == (cfg["batch"], cfg["seq"], cfg["Out"])


def test_cpp_gqa_pruning_nonempty_past_kv_constant_is_left_untouched():
    model, cfg = _gqa_model(K=8, H=8, KVH=2, D=8, Out=6, seed=12, past_kv="nonempty")
    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    inits_before = {
        t.name: onnx.numpy_helper.to_array(t) for t in model.graph.initializer
    }
    inits_after = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer
    }
    for name in inits_before:
        np.testing.assert_array_equal(inits_before[name], inits_after[name])


def test_cpp_gqa_pruning_dynamic_past_kv_input_is_still_pruned():
    model, cfg = _gqa_model(K=8, H=8, KVH=2, D=8, Out=6, seed=13, past_kv="dynamic")
    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    node = _gqa_node(pruned)
    num_heads, kv_num_heads = _gqa_attrs(node)
    group_size = cfg["H"] // cfg["KVH"]
    assert kv_num_heads == 1
    assert num_heads == group_size


def test_cpp_attention_head_pruning_group_query_attention_missing_required_inputs_is_left_untouched():
    K, H, D, Out = 8, 4, 4, 6
    Nqkv = H * D
    rng = np.random.default_rng(5)
    wq = rng.standard_normal((K, Nqkv)).astype(np.float32)
    wk = rng.standard_normal((K, Nqkv)).astype(np.float32)
    wv = rng.standard_normal((K, Nqkv)).astype(np.float32)
    wout = rng.standard_normal((Nqkv, Out)).astype(np.float32)
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 17, "com.microsoft": 1]
        >
        g (float[2,5,{K}] X) => (float[2,5,{Out}] Y)
        {{
          q = MatMul(X, Wq)
          k = MatMul(X, Wk)
          v = MatMul(X, Wv)
          ctx, present_k, present_v = com.microsoft.GroupQueryAttention <num_heads={H}, kv_num_heads={H}> (q, k, v)
          Y = MatMul(ctx, Wout)
        }}
        """
    )
    model.graph.initializer.extend(
        [_f32(wq, "Wq"), _f32(wk, "Wk"), _f32(wv, "Wv"), _f32(wout, "Wout")]
    )

    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    inits_before = {
        t.name: onnx.numpy_helper.to_array(t) for t in model.graph.initializer
    }
    inits_after = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer
    }
    for name in inits_before:
        np.testing.assert_array_equal(inits_before[name], inits_after[name])


# --- Plain ai.onnx::Attention (opset 24+) -----------------------------------


def _onnx_attention_model(
    K=8,
    H=4,
    KVH=2,
    D=4,
    Out=6,
    seed=0,
    batch=2,
    seq=5,
    bias=False,
    with_reshape=False,
    wq=None,
    wk=None,
    wv=None,
    bq=None,
    bk=None,
    bv=None,
    wout=None,
    attn_mask=None,  # None (omitted) | "nonempty" (constant) | "dynamic" (graph input)
    past_kv=None,  # None (omitted) | "nonempty" (constant) | "dynamic" (graph input)
):
    rng = np.random.default_rng(seed)
    Nq, Nkv = H * D, KVH * D
    if wq is None:
        wq = rng.standard_normal((K, Nq)).astype(np.float32)
    if wk is None:
        wk = rng.standard_normal((K, Nkv)).astype(np.float32)
    if wv is None:
        wv = rng.standard_normal((K, Nkv)).astype(np.float32)
    if wout is None:
        wout = rng.standard_normal((Nq, Out)).astype(np.float32)

    initializer = [_f32(wq, "Wq"), _f32(wk, "Wk"), _f32(wv, "Wv"), _f32(wout, "Wout")]
    q_op, k_op, v_op = "MatMul(X, Wq)", "MatMul(X, Wk)", "MatMul(X, Wv)"
    if bias:
        if bq is None:
            bq = rng.standard_normal((Nq,)).astype(np.float32)
        if bk is None:
            bk = rng.standard_normal((Nkv,)).astype(np.float32)
        if bv is None:
            bv = rng.standard_normal((Nkv,)).astype(np.float32)
        initializer += [_f32(bq, "Bq"), _f32(bk, "Bk"), _f32(bv, "Bv")]
        q_op, k_op, v_op = "Gemm(X, Wq, Bq)", "Gemm(X, Wk, Bk)", "Gemm(X, Wv, Bv)"

    operands = ["q", "k", "v"]
    extra_graph_inputs = ""

    if attn_mask == "nonempty":
        mask = np.zeros((seq, seq), dtype=np.float32)
        initializer.append(_f32(mask, "AttnMask"))
        operands.append("AttnMask")
    elif attn_mask == "dynamic":
        operands.append("AttnMaskIn")
        extra_graph_inputs += f", float[{seq},{seq}] AttnMaskIn"
    else:
        operands.append("")

    if past_kv == "nonempty":
        past_key = rng.standard_normal((batch, KVH, 1, D)).astype(np.float32)
        past_value = rng.standard_normal((batch, KVH, 1, D)).astype(np.float32)
        initializer += [_f32(past_key, "PastKey"), _f32(past_value, "PastValue")]
        operands += ["PastKey", "PastValue"]
    elif past_kv == "dynamic":
        operands += ["PastKeyIn", "PastValueIn"]
        extra_graph_inputs += (
            f", float[{batch},{KVH},1,{D}] PastKeyIn"
            f", float[{batch},{KVH},1,{D}] PastValueIn"
        )
    else:
        operands += ["", ""]

    while operands and operands[-1] == "":
        operands.pop()

    if with_reshape:
        shape = np.array([batch, seq, Nq], dtype=np.int64)
        initializer.append(onnx.numpy_helper.from_array(shape, "Shape"))
        tail = "ctx2 = Reshape(ctx, Shape)\n          Y = MatMul(ctx2, Wout)"
    else:
        tail = "Y = MatMul(ctx, Wout)"

    body = f"""
        g (float[{batch},{seq},{K}] X{extra_graph_inputs}) => (float[{batch},{seq},{Out}] Y)
        {{
          q = {q_op}
          k = {k_op}
          v = {v_op}
          ctx = Attention <q_num_heads={H}, kv_num_heads={KVH}> ({", ".join(operands)})
          {tail}
        }}
        """

    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 24]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model, dict(
        K=K,
        H=H,
        KVH=KVH,
        D=D,
        Out=Out,
        Nq=Nq,
        Nkv=Nkv,
        wq=wq,
        wk=wk,
        wv=wv,
        bq=bq,
        bk=bk,
        bv=bv,
        wout=wout,
        batch=batch,
        seq=seq,
    )


def _onnx_attention_node(model):
    return next(
        n for n in model.graph.node if n.op_type == "Attention" and n.domain == ""
    )


def _onnx_attention_attrs(node):
    q_num_heads = next(a.i for a in node.attribute if a.name == "q_num_heads")
    kv_num_heads = next(a.i for a in node.attribute if a.name == "kv_num_heads")
    return q_num_heads, kv_num_heads


def test_cpp_onnx_attention_pruning_shrinks_matched_block():
    model, cfg = _onnx_attention_model(K=8, H=4, KVH=2, D=4, Out=6)
    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    node = _onnx_attention_node(pruned)
    q_num_heads, kv_num_heads = _onnx_attention_attrs(node)
    assert kv_num_heads == 1
    assert q_num_heads == kv_num_heads * (cfg["H"] // cfg["KVH"])


def test_cpp_onnx_attention_pruning_matches_oracle_exactly():
    model, cfg = _onnx_attention_model(K=8, H=8, KVH=2, D=4, Out=6, seed=1)
    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    node = _onnx_attention_node(pruned)
    q_num_heads, kv_num_heads = _onnx_attention_attrs(node)
    group_size = cfg["H"] // cfg["KVH"]
    assert kv_num_heads == 1
    assert q_num_heads == kv_num_heads * group_size

    keep_groups = _oracle_keep_groups(
        cfg["wq"], cfg["wk"], cfg["wv"], cfg["H"], cfg["KVH"], cfg["D"], kv_num_heads
    )
    keep_q_heads = _group_q_heads(keep_groups, group_size)
    d = cfg["D"]
    q_idx, kv_idx = _head_idx(keep_q_heads, d), _head_idx(keep_groups, d)

    oracle, _ = _onnx_attention_model(
        K=cfg["K"],
        H=len(keep_q_heads),
        KVH=len(keep_groups),
        D=d,
        Out=cfg["Out"],
        seed=1,
        wq=cfg["wq"][:, q_idx],
        wk=cfg["wk"][:, kv_idx],
        wv=cfg["wv"][:, kv_idx],
        wout=cfg["wout"][q_idx, :],
        batch=cfg["batch"],
        seq=cfg["seq"],
    )

    rng = np.random.default_rng(2)
    x = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(np.float32)
    (y_pruned,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y_pruned, y_oracle, rtol=1e-4, atol=1e-4)


def test_cpp_onnx_attention_pruning_slices_bias_when_producer_has_one():
    model, cfg = _onnx_attention_model(K=8, H=4, KVH=2, D=4, Out=6, seed=14, bias=True)
    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)

    node = _onnx_attention_node(pruned)
    q_num_heads, kv_num_heads = _onnx_attention_attrs(node)
    group_size = cfg["H"] // cfg["KVH"]
    assert kv_num_heads == 1
    assert q_num_heads == group_size

    keep_groups = _oracle_keep_groups(
        cfg["wq"], cfg["wk"], cfg["wv"], cfg["H"], cfg["KVH"], cfg["D"], kv_num_heads
    )
    keep_q_heads = _group_q_heads(keep_groups, group_size)
    d = cfg["D"]
    q_idx, kv_idx = _head_idx(keep_q_heads, d), _head_idx(keep_groups, d)

    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["Wq"], cfg["wq"][:, q_idx])
    np.testing.assert_array_equal(inits["Wk"], cfg["wk"][:, kv_idx])
    np.testing.assert_array_equal(inits["Wv"], cfg["wv"][:, kv_idx])
    np.testing.assert_array_equal(inits["Bq"], cfg["bq"][q_idx])
    np.testing.assert_array_equal(inits["Bk"], cfg["bk"][kv_idx])
    np.testing.assert_array_equal(inits["Bv"], cfg["bv"][kv_idx])


def test_cpp_onnx_attention_pruning_nonempty_attn_mask_constant_is_left_untouched():
    model, cfg = _onnx_attention_model(
        K=8, H=4, KVH=2, D=4, Out=6, seed=17, attn_mask="nonempty"
    )
    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    inits_before = {
        t.name: onnx.numpy_helper.to_array(t) for t in model.graph.initializer
    }
    inits_after = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer
    }
    for name in inits_before:
        np.testing.assert_array_equal(inits_before[name], inits_after[name])


def test_cpp_onnx_attention_pruning_diff_v_head_size_is_left_untouched():
    # This op's real schema (unlike GroupQueryAttention, which fuse_gqa.h
    # always emits with equal Q/K/V head_size) genuinely allows V its own,
    # independent head_size. This pass reuses the shared, uniform-head_size
    # slicing body unmodified rather than a parallel implementation, so a
    # node whose V head size actually differs from Q/K's is declined here
    # rather than mis-sliced.
    K, H, KVH, D, Dv, Out = 8, 4, 4, 4, 6, 5
    Nq, Nk, Nv = H * D, KVH * D, KVH * Dv
    rng = np.random.default_rng(19)
    wq = rng.standard_normal((K, Nq)).astype(np.float32)
    wk = rng.standard_normal((K, Nk)).astype(np.float32)
    wv = rng.standard_normal((K, Nv)).astype(np.float32)
    wout = rng.standard_normal((Nq, Out)).astype(np.float32)
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 24]
        >
        g (float[2,5,{K}] X) => (float[2,5,{Out}] Y)
        {{
          q = MatMul(X, Wq)
          k = MatMul(X, Wk)
          v = MatMul(X, Wv)
          ctx = Attention <q_num_heads={H}, kv_num_heads={KVH}> (q, k, v)
          Y = MatMul(ctx, Wout)
        }}
        """
    )
    model.graph.initializer.extend(
        [_f32(wq, "Wq"), _f32(wk, "Wk"), _f32(wv, "Wv"), _f32(wout, "Wout")]
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    inits_before = {
        t.name: onnx.numpy_helper.to_array(t) for t in model.graph.initializer
    }
    inits_after = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer
    }
    for name in inits_before:
        np.testing.assert_array_equal(inits_before[name], inits_after[name])


# --- Cross-check against the pure-Python reference --------------------------


def test_cpp_attention_head_pruning_matches_python_reference_output():
    model, _ = _attention_model(K=8, H=4, D=4, Out=6, seed=21)
    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned_py)
    onnx.checker.check_model(pruned_cpp)

    rng = np.random.default_rng(22)
    x = rng.standard_normal((2, 5, 8)).astype(np.float32)
    (y_py,) = _run(pruned_py, {"X": x})
    (y_cpp,) = _run(pruned_cpp, {"X": x})
    np.testing.assert_allclose(y_py, y_cpp, rtol=1e-4, atol=1e-4)
