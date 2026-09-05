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
import onnx.helper
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


# --- Subgraph recursion (If) -------------------------------------------------
#
# Covers `structured_pruning_entry.cpp`'s own `IterSubgraphs` and the
# `ApplyAttentionHeadPruning` loop built on it -- a straight C++ port of
# `onnxsim/pruning.py`'s own `_iter_subgraphs`/`apply_attention_head_pruning`
# subgraph-recursion round (see that module's "Subgraph recursion" section
# comment, and `structured_pruning_entry.cpp`'s own copy of it above
# `IterSubgraphs`'s definition, for the full design rationale). Model shape
# mirrors `tests/test_pruning.py`'s own `_if_attention_model`/
# `test_attention_head_pruning_prunes_blocks_inside_if_branches` fixture,
# just driven through `apply_attention_head_pruning_cpp` instead of the
# pure-Python reference.
#
# `onnx.parser.parse_model`'s text format has no way to spell a graph-typed
# node attribute (an `If`'s `then_branch`/`else_branch`), so the model below
# uses `onnx.helper.make_node`/`make_graph` directly, per this repo's own
# CLAUDE.md guidance for exactly this case -- see `test_structured_pruning_
# cpp.py`'s own matching "Subgraph recursion" section for the `Loop`-body
# half of this coverage (the two C++ port test files split the `If`/`Loop`
# cases between them rather than duplicating both in each).


def _attention_branch_nodes(K, H, D, Out, prefix, seed):
    rng = np.random.default_rng(seed)
    Nq = Nk = Nv = H * D
    wqkv = rng.standard_normal((K, Nq + Nk + Nv)).astype(np.float32)
    bqkv = rng.standard_normal((Nq + Nk + Nv,)).astype(np.float32)
    wout = rng.standard_normal((Nv, Out)).astype(np.float32)
    nodes = [
        onnx.helper.make_node(
            "Attention",
            ["Xb", f"{prefix}Wqkv", f"{prefix}Bqkv"],
            [f"{prefix}ctx"],
            domain="com.microsoft",
            num_heads=H,
            qkv_hidden_sizes=[Nq, Nk, Nv],
        ),
        onnx.helper.make_node("MatMul", [f"{prefix}ctx", f"{prefix}Wout"], ["Yb"]),
    ]
    inits = [
        _f32(wqkv, f"{prefix}Wqkv"),
        _f32(bqkv, f"{prefix}Bqkv"),
        _f32(wout, f"{prefix}Wout"),
    ]
    return nodes, inits, dict(wqkv=wqkv, bqkv=bqkv, wout=wout)


def _if_attention_model(K=8, H=4, D=4, Out=6):
    then_nodes, then_inits, then_cfg = _attention_branch_nodes(
        K, H, D, Out, "then_", seed=1
    )
    else_nodes, else_inits, else_cfg = _attention_branch_nodes(
        K, H, D, Out, "else_", seed=2
    )
    out_vi = onnx.helper.make_tensor_value_info(
        "Yb", onnx.TensorProto.FLOAT, ["batch", "seq", Out]
    )
    then_graph = onnx.helper.make_graph(
        then_nodes, "then_graph", [], [out_vi], initializer=then_inits
    )
    else_graph = onnx.helper.make_graph(
        else_nodes, "else_graph", [], [out_vi], initializer=else_inits
    )
    if_node = onnx.helper.make_node(
        "If", ["cond"], ["Y1"], then_branch=then_graph, else_branch=else_graph
    )
    xb = onnx.helper.make_tensor_value_info(
        "Xb", onnx.TensorProto.FLOAT, ["batch", "seq", K]
    )
    cond = onnx.helper.make_tensor_value_info("cond", onnx.TensorProto.BOOL, [])
    y1 = onnx.helper.make_tensor_value_info(
        "Y1", onnx.TensorProto.FLOAT, ["batch", "seq", Out]
    )
    graph = onnx.helper.make_graph([if_node], "g", [xb, cond], [y1])
    model = onnx.helper.make_model(
        graph,
        opset_imports=[
            onnx.helper.make_opsetid("", 17),
            onnx.helper.make_opsetid("com.microsoft", 1),
        ],
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model, dict(then=then_cfg, else_=else_cfg)


def _then_else_graphs(pruned_model):
    if_node = next(n for n in pruned_model.graph.node if n.op_type == "If")
    then_g = else_g = None
    for attr in if_node.attribute:
        if attr.name == "then_branch":
            then_g = attr.g
        elif attr.name == "else_branch":
            else_g = attr.g
    return then_g, else_g


def test_cpp_attention_head_pruning_prunes_blocks_inside_if_branches():
    # The core repro: apply_attention_head_pruning_cpp must match and prune
    # the merged-QKV Attention block inside BOTH `then_branch` and
    # `else_branch` -- each with its own independent weights and its own
    # independently-computed importance ranking/kept-head set -- not just a
    # top-level block. Verified by initializer shape, by the node's own
    # updated `num_heads` attribute, and by an exact oracle cross-check per
    # branch (mirroring `test_cpp_attention_head_pruning_matches_manual_
    # head_deletion_exactly`'s own oracle, just built independently once per
    # branch to prove neither branch's ranking leaked into the other's).
    K, H, D, Out = 8, 4, 4, 6
    model, cfg = _if_attention_model(K=K, H=H, D=D, Out=Out)
    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    then_g, else_g = _then_else_graphs(pruned)
    rng = np.random.default_rng(4)
    xb = rng.standard_normal((2, 3, K)).astype(np.float32)

    for g, prefix, branch_cfg, cond in [
        (then_g, "then_", cfg["then"], True),
        (else_g, "else_", cfg["else_"], False),
    ]:
        inits = {t.name: t for t in g.initializer}
        assert list(inits[f"{prefix}Wqkv"].dims) == [K, 3 * (H // 2) * D]
        assert list(inits[f"{prefix}Wout"].dims) == [(H // 2) * D, Out]
        node = next(n for n in g.node if n.op_type == "Attention")
        num_heads = next(a.i for a in node.attribute if a.name == "num_heads")
        assert num_heads == H // 2

        wqkv, bqkv, wout = branch_cfg["wqkv"], branch_cfg["bqkv"], branch_cfg["wout"]
        Nq = Nk = Nv = H * D
        keep = _oracle_keep_heads(wqkv, Nq, Nk, Nv, H, H // 2)
        qi, ki, vi = (
            _head_idx(keep, D),
            _head_idx(keep, D) + Nq,
            _head_idx(keep, D) + Nq + Nk,
        )
        all_idx = np.concatenate([qi, ki, vi])
        np.testing.assert_allclose(
            onnx.numpy_helper.to_array(inits[f"{prefix}Wqkv"]),
            wqkv[:, all_idx],
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            onnx.numpy_helper.to_array(inits[f"{prefix}Wout"]),
            wout[_head_idx(keep, D), :],
            rtol=1e-5,
            atol=1e-5,
        )

        oracle_bqkv = bqkv[all_idx]
        np.testing.assert_allclose(
            onnx.numpy_helper.to_array(inits[f"{prefix}Bqkv"]),
            oracle_bqkv,
            rtol=1e-5,
            atol=1e-5,
        )

        (yb,) = _run(pruned, {"Xb": xb, "cond": np.array(cond)})
        assert yb.shape == (2, 3, Out)
        assert np.all(np.isfinite(yb))


def test_cpp_attention_head_pruning_matches_python_reference_output_with_if_subgraph():
    # Cross-check against onnxsim.apply_attention_head_pruning (the
    # pure-Python reference this C++ port mirrors) on a model where the
    # only prunable attention block lives inside the `If`'s own branches --
    # both `cond` values are driven through InferenceSession so both
    # branches' own subgraph-recursion behavior is exercised.
    K, H, D, Out = 8, 4, 4, 6
    model, _cfg = _if_attention_model(K=K, H=H, D=D, Out=Out)

    pruned_py = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned_py)
    onnx.checker.check_model(pruned_cpp)

    rng = np.random.default_rng(23)
    xb = rng.standard_normal((2, 3, K)).astype(np.float32)
    for cond in (True, False):
        feeds = {"Xb": xb, "cond": np.array(cond)}
        (y_py,) = _run(pruned_py, feeds)
        (y_cpp,) = _run(pruned_cpp, feeds)
        np.testing.assert_allclose(y_py, y_cpp, rtol=1e-4, atol=1e-4)


# --- com.microsoft::MatMulNBitsQkv (fused, block-quantized Q/K/V projection
# --- feeding GroupQueryAttention) -------------------------------------------
#
# Tests for the fused ``MatMulNBitsQkv`` chain family
# (``onnxsim/structured_pruning_entry.cpp``'s own "MatMulNBitsMlp/
# MatMulNBitsQkv" subsection -- despite living in that file's own
# "MatMulNBits" section, this chain kind is wired into
# `ApplyAttentionHeadPruning`/``apply_attention_head_pruning_cpp`` rather
# than `ApplyStructuredPruning`/``apply_structured_pruning_cpp``, since
# pruning a whole KV group needs THIS function's own GQA head-count
# matching machinery -- see that subsection's own top comment for the full
# reasoning), mirroring ``tests/test_pruning.py``'s own
# ``test_matmul_nbits_qkv_pruning_matches_decomposed_oracle``/
# ``test_matmul_nbits_qkv_pruning_declines_non_block_aligned_consumer``.
# Like ``MatMulNBitsMlp``, neither this op nor its Q/K/V branches has a
# ``zero_points`` input at all -- every weight slot uses the schema's own
# DEFAULT zero point (``2 ** (bits - 1)``, i.e. 8 for ``bits=4``). Unlike
# ``MatMulNBitsMlp``, ``MatMulNBitsQkv`` itself ALSO cannot be executed via a
# real CPU-EP ``InferenceSession`` here (confirmed the same empirical way) --
# so its own oracle test below decomposes the PRUNED fused node's own
# tensors into a real ``SimplifiedLayerNormalization`` + 3x real
# ``MatMulNBits`` (both genuine CPU kernels) and checks their own outputs
# against an independent RMSNorm + dequantize-then-matmul numpy oracle,
# mirroring ``test_pruning.py``'s own identical proxy-topology technique.
# The downstream ``GroupQueryAttention``/consumer half of this pass's own
# slicing -- num_heads/kv_num_heads attribute rewrite, consumer weight
# slicing -- is exactly the same code path every plain-GQA test above
# already runs end to end through a real ``GroupQueryAttention`` CPU kernel
# (`ApplyOneGqaChain`'s own head-count/attribute handling, reused verbatim
# by `ApplyMatMulNBitsQkvChains` -- see that function's own comment), so
# this section's own tests below check it via direct assertion instead
# (attribute values, byte-exact consumer-weight slices) rather than
# re-proving already-oracle-tested machinery a second time.


def _nbits_pack_nibbles(vals):
    """Independent reference nibble packer: last axis (uint8 in [0, 15]),
    2-per-byte, LOW nibble first -- the schema's own documented layout.
    """
    count = vals.shape[-1]
    nbytes = (count + 1) // 2
    out = np.zeros(vals.shape[:-1] + (nbytes,), dtype=np.uint8)
    for j in range(nbytes):
        lo = vals[..., 2 * j]
        hi = vals[..., 2 * j + 1] if 2 * j + 1 < count else np.zeros_like(lo)
        out[..., j] = (lo & 0xF) | ((hi & 0xF) << 4)
    return out


def _nbits_pack_b(qcodes, n, k_blocks, block_size):
    blob_size = block_size * 4 // 8
    b = np.zeros((n, k_blocks, blob_size), dtype=np.uint8)
    for kb in range(k_blocks):
        k0 = kb * block_size
        b[:, kb, :] = _nbits_pack_nibbles(qcodes[:, k0 : k0 + block_size])
    return b


def _nbits_quantize_default_zp(w, block_size, bits=4):
    """Independent reference block quantizer using the schema's own DEFAULT
    zero point (``2 ** (bits - 1)``) -- the only encoding ``MatMulNBitsQkv``
    (and ``MatMulNBitsMlp``) supports, since neither has a ``zero_points``
    input at all. Returns ``(qcodes uint8 [N, K], scales float32 [N,
    k_blocks], k_blocks)``.
    """
    n, k = w.shape
    assert k % block_size == 0
    k_blocks = k // block_size
    qmax = (1 << bits) - 1
    zp = float(1 << (bits - 1))
    scales = np.zeros((n, k_blocks), dtype=np.float32)
    qcodes = np.zeros((n, k), dtype=np.uint8)
    for row in range(n):
        for kb in range(k_blocks):
            k0, k1 = kb * block_size, (kb + 1) * block_size
            block = w[row, k0:k1]
            maxabs = max(float(np.max(np.abs(block))), 1e-8)
            scale = maxabs / max(zp, qmax - zp)
            scales[row, kb] = scale
            codes = np.round(block / scale + zp).clip(0, qmax)
            qcodes[row, k0:k1] = codes.astype(np.uint8)
    return qcodes, scales, k_blocks


def _nbits_dequant(qcodes, scales, block_size, bits=4):
    """Independent reference dequantizer, schema DEFAULT zero point only."""
    n, k = qcodes.shape
    k_blocks = k // block_size
    out = np.zeros((n, k), dtype=np.float64)
    zp = float(1 << (bits - 1))
    for row in range(n):
        for kb in range(k_blocks):
            k0, k1 = kb * block_size, (kb + 1) * block_size
            out[row, k0:k1] = (qcodes[row, k0:k1].astype(np.float64) - zp) * scales[
                row, kb
            ]
    return out


def _nbits_no_zp_initializers(w, block_size, prefix, bits=4):
    """Quantizes ``w`` (``[N, K]``) with the schema's own default zero point
    and returns ``(initializer_list, info_dict)`` -- the zero_points-free
    analogue of a ``_nbits_weight_initializers`` helper, since neither
    ``MatMulNBitsQkv`` nor ``MatMulNBitsMlp`` ever carries one.
    """
    qcodes, scales, k_blocks = _nbits_quantize_default_zp(w, block_size, bits)
    b = _nbits_pack_b(qcodes, w.shape[0], k_blocks, block_size)
    inits = [
        onnx.numpy_helper.from_array(b, name=f"{prefix}_B"),
        onnx.numpy_helper.from_array(scales, name=f"{prefix}_scales"),
    ]
    return inits, dict(
        qcodes=qcodes,
        scales=scales,
        k_blocks=k_blocks,
        b_name=f"{prefix}_B",
        scales_name=f"{prefix}_scales",
    )


def _matmul_nbits_qkv_model(
    num_heads,
    kv_num_heads,
    d,
    K,
    block_size,
    N2,
    w_q,
    w_k,
    w_v,
    bias_q,
    bias_k,
    bias_v,
    norm_scale,
    batch=2,
    seq=5,
    consumer="plain",
    consumer_block_size=None,
    attention_bias=None,
):
    """Builds ``A -> MatMulNBitsQkv(qkv) -> (Q, K, V) ->
    GroupQueryAttention(attn) -> MatMul/MatMulNBits(down) -> Z``. ``Q``/``K``/
    ``V`` feed the attention node's own query/key/value inputs DIRECTLY (no
    per-head norm/RoPE hop -- a deliberate, documented scope boundary, see
    ``structured_pruning_entry.cpp``'s own section comment).
    ``consumer="plain"`` builds a plain-float output projection;
    ``consumer="nbits"`` a real ``MatMulNBits`` one (block size
    `consumer_block_size`), to exercise the block-alignment decline path.
    `attention_bias`, if given, is wired as a constant GQA `attention_bias`
    input (index 10) -- to exercise `MatMulNBitsQkvAttentionExtrasSafe`'s own
    decline path.
    """
    Nq = num_heads * d
    Nkv = kv_num_heads * d
    inits_q, info_q = _nbits_no_zp_initializers(w_q, block_size, "qkvq")
    inits_k, info_k = _nbits_no_zp_initializers(w_k, block_size, "qkvk")
    inits_v, info_v = _nbits_no_zp_initializers(w_v, block_size, "qkvv")

    seqlens_k = np.full((batch,), seq - 1, dtype=np.int32)
    total_seq = np.array(seq, dtype=np.int32)
    initializer = [
        *inits_q,
        *inits_k,
        *inits_v,
        _f32(bias_q, "qkv_bias_q"),
        _f32(bias_k, "qkv_bias_k"),
        _f32(bias_v, "qkv_bias_v"),
        _f32(norm_scale, "qkv_norm_scale"),
        onnx.numpy_helper.from_array(seqlens_k, "qkv_seqlens_k"),
        onnx.numpy_helper.from_array(total_seq, "qkv_total_seq"),
    ]

    if consumer == "plain":
        rng = np.random.default_rng(9202)
        down_w = (rng.standard_normal((Nq, N2)) * 0.3).astype(np.float32)
        initializer.append(_f32(down_w, "qkv_down_w"))
        consumer_body = "Z = MatMul(ctx, qkv_down_w)"
        consumer_info = dict(down_w=down_w)
    else:
        assert consumer_block_size is not None
        rng = np.random.default_rng(9203)
        w_c = (rng.standard_normal((N2, Nq)) * 0.3).astype(np.float32)
        inits_c, info_c = _nbits_no_zp_initializers(w_c, consumer_block_size, "qkvdown")
        initializer += inits_c
        consumer_body = (
            f"Z = com.microsoft.MatMulNBits<K={Nq},N={N2},bits=4,"
            f"block_size={consumer_block_size}>"
            f"(ctx, {info_c['b_name']}, {info_c['scales_name']})"
        )
        consumer_info = dict(
            qcodes_c=info_c["qcodes"], scales_c=info_c["scales"], kbc=info_c["k_blocks"]
        )

    gqa_extra_inputs = ""
    if attention_bias is not None:
        initializer.append(_f32(attention_bias, "qkv_attention_bias"))
        # attention_bias is input 10 -- pad cos_cache/sin_cache/position_ids
        # (7/8/9) with blanks.
        gqa_extra_inputs = ", , , , qkv_attention_bias"

    body = f"""
        g (float[{batch},{seq},{K}] A) => (float[{batch},{seq},{N2}] Z)
        {{
          Q, Kt, V = com.microsoft.MatMulNBitsQkv<block_size={block_size},bits=4,Nq={Nq},Nkv={Nkv},K={K}>(A, , qkv_norm_scale, {info_q["b_name"]}, {info_q["scales_name"]}, qkv_bias_q, {info_k["b_name"]}, {info_k["scales_name"]}, qkv_bias_k, {info_v["b_name"]}, {info_v["scales_name"]}, qkv_bias_v)
          ctx, pk, pv = com.microsoft.GroupQueryAttention<num_heads={num_heads}, kv_num_heads={kv_num_heads}>(Q, Kt, V, , , qkv_seqlens_k, qkv_total_seq{gqa_extra_inputs})
          {consumer_body}
        }}
        """

    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 21, "com.microsoft": 1]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    info = dict(
        qcodes_q=info_q["qcodes"],
        scales_q=info_q["scales"],
        qcodes_k=info_k["qcodes"],
        scales_k=info_k["scales"],
        qcodes_v=info_v["qcodes"],
        scales_v=info_v["scales"],
        kb=info_q["k_blocks"],
        Nq=Nq,
        Nkv=Nkv,
    )
    info.update(consumer_info)
    return model, info


def test_cpp_matmul_nbits_qkv_pruning_matches_decomposed_oracle():
    # 4 query heads, 2 KV heads (group_size=2), head_size=2. KV group 0
    # (query heads 0,1 + kv head 0) engineered LARGE (kept); KV group 1
    # (query heads 2,3 + kv head 1) engineered SMALL (dropped). sparsity=0.5
    # -> keep exactly 1 of 2 groups: group 0 -- num_heads 4->2, kv_num_heads
    # 2->1, preserving the 2:1 group ratio exactly.
    num_heads, kv_num_heads, d, K, block_size, N2 = 4, 2, 2, 32, 32, 5
    rng = np.random.default_rng(9300)
    w_q = (rng.standard_normal((num_heads * d, K)) * 0.3).astype(np.float32)
    w_k = (rng.standard_normal((kv_num_heads * d, K)) * 0.3).astype(np.float32)
    w_v = (rng.standard_normal((kv_num_heads * d, K)) * 0.3).astype(np.float32)
    w_q[:4] *= 8.0
    w_q[4:] *= 0.05
    w_k[:2] *= 8.0
    w_k[2:] *= 0.05
    w_v[:2] *= 8.0
    w_v[2:] *= 0.05
    bias_q = (rng.standard_normal(num_heads * d) * 0.05).astype(np.float32)
    bias_k = (rng.standard_normal(kv_num_heads * d) * 0.05).astype(np.float32)
    bias_v = (rng.standard_normal(kv_num_heads * d) * 0.05).astype(np.float32)
    norm_scale = (1.0 + rng.standard_normal(K) * 0.1).astype(np.float32)

    model, info = _matmul_nbits_qkv_model(
        num_heads,
        kv_num_heads,
        d,
        K,
        block_size,
        N2,
        w_q,
        w_k,
        w_v,
        bias_q,
        bias_k,
        bias_v,
        norm_scale,
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    q_keep = np.array([0, 1, 2, 3])  # heads 0,1 (group 0), d=2 -> rows 0-3
    kv_keep = np.array([0, 1])  # kv head 0, d=2 -> rows 0-1

    qkv_node = next(n for n in pruned.graph.node if n.op_type == "MatMulNBitsQkv")
    assert next(a.i for a in qkv_node.attribute if a.name == "Nq") == 4
    assert next(a.i for a in qkv_node.attribute if a.name == "Nkv") == 2
    attn_node = next(n for n in pruned.graph.node if n.op_type == "GroupQueryAttention")
    assert next(a.i for a in attn_node.attribute if a.name == "num_heads") == 2
    assert next(a.i for a in attn_node.attribute if a.name == "kv_num_heads") == 1

    inits = {t.name: t for t in pruned.graph.initializer}
    q_B_expected = _nbits_pack_b(info["qcodes_q"][q_keep], 4, info["kb"], block_size)
    k_B_expected = _nbits_pack_b(info["qcodes_k"][kv_keep], 2, info["kb"], block_size)
    v_B_expected = _nbits_pack_b(info["qcodes_v"][kv_keep], 2, info["kb"], block_size)
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(inits["qkvq_B"]), q_B_expected
    )
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(inits["qkvk_B"]), k_B_expected
    )
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(inits["qkvv_B"]), v_B_expected
    )
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(inits["qkv_down_w"]), info["down_w"][q_keep]
    )
    np.testing.assert_allclose(
        onnx.numpy_helper.to_array(inits["qkv_bias_q"]), bias_q[q_keep]
    )
    np.testing.assert_allclose(
        onnx.numpy_helper.to_array(inits["qkv_bias_k"]), bias_k[kv_keep]
    )
    np.testing.assert_allclose(
        onnx.numpy_helper.to_array(inits["qkv_bias_v"]), bias_v[kv_keep]
    )

    # Decompose the fused MatMulNBitsQkv node's own (PRUNED) tensors into a
    # real SimplifiedLayerNormalization + 3x real MatMulNBits and run THOSE
    # through a CPU-kernel InferenceSession against an independent RMSNorm +
    # dequantize-then-matmul numpy oracle -- see this section's own top
    # comment for why the fused node itself cannot be executed here.
    decomposed = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 21, "com.microsoft": 1]
        >
        g (float[batch,{K}] A) => (float[batch,4] q_out, float[batch,2] k_out, float[batch,2] v_out)
        {{
          A_norm = SimplifiedLayerNormalization<axis=-1,epsilon=1e-5>(A, qkv_norm_scale)
          q_out = com.microsoft.MatMulNBits<K={K},N=4,bits=4,block_size={block_size}>(A_norm, qkvq_B, qkvq_scales, , , qkv_bias_q)
          k_out = com.microsoft.MatMulNBits<K={K},N=2,bits=4,block_size={block_size}>(A_norm, qkvk_B, qkvk_scales, , , qkv_bias_k)
          v_out = com.microsoft.MatMulNBits<K={K},N=2,bits=4,block_size={block_size}>(A_norm, qkvv_B, qkvv_scales, , , qkv_bias_v)
        }}
        """
    )
    decomposed.graph.initializer.extend(
        [
            inits["qkv_norm_scale"],
            inits["qkvq_B"],
            inits["qkvq_scales"],
            inits["qkv_bias_q"],
            inits["qkvk_B"],
            inits["qkvk_scales"],
            inits["qkv_bias_k"],
            inits["qkvv_B"],
            inits["qkvv_scales"],
            inits["qkv_bias_v"],
        ]
    )
    # No onnx.checker.check_model here -- the plain ONNX checker doesn't
    # recognize `SimplifiedLayerNormalization` (an onnxruntime-only op
    # registered under the "" domain, confirmed via live schema
    # introspection) even though onnxruntime itself executes it fine;
    # mirrors test_pruning.py's own identical omission for this same
    # decomposed proxy graph.

    x = np.random.default_rng(9301).standard_normal((3, K)).astype(np.float32)
    q_actual, k_actual, v_actual = _run(decomposed, {"A": x})

    def _rmsnorm(a, scale, eps):
        rms = np.sqrt(np.mean(a.astype(np.float64) ** 2, axis=-1, keepdims=True) + eps)
        return (a.astype(np.float64) / rms) * scale.astype(np.float64)

    a_norm_ref = _rmsnorm(x, norm_scale, 1e-5)
    w_q_dequant = _nbits_dequant(info["qcodes_q"], info["scales_q"], block_size)
    w_k_dequant = _nbits_dequant(info["qcodes_k"], info["scales_k"], block_size)
    w_v_dequant = _nbits_dequant(info["qcodes_v"], info["scales_v"], block_size)
    q_ref = a_norm_ref @ w_q_dequant[q_keep].T + bias_q[q_keep]
    k_ref = a_norm_ref @ w_k_dequant[kv_keep].T + bias_k[kv_keep]
    v_ref = a_norm_ref @ w_v_dequant[kv_keep].T + bias_v[kv_keep]

    np.testing.assert_allclose(q_actual, q_ref, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(k_actual, k_ref, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(v_actual, v_ref, rtol=1e-3, atol=1e-3)


def test_cpp_matmul_nbits_qkv_pruning_declines_non_block_aligned_consumer():
    # head_size=3 (unlike the block-aligned test above's head_size=2):
    # keep_q_heads=[0, 1] (group 0) -> q_idx = rows [0..5] (6 elements),
    # which straddles the MatMulNBits consumer's own block boundary at row 4
    # (block_size=4: blocks [0,4), [4,8), [8,12)) -- rows 4, 5 are only PART
    # of block 1, so this keep-set is NOT block-aligned. The whole chain
    # (qkv node, attention node, AND consumer) must be left completely
    # untouched, mirroring the plain-MatMulNBits consumer's own identical
    # decline precedent.
    num_heads, kv_num_heads, d, K, block_size, N2 = 4, 2, 3, 32, 32, 4
    consumer_block_size = 4
    rng = np.random.default_rng(9310)
    w_q = (rng.standard_normal((num_heads * d, K)) * 0.3).astype(np.float32)
    w_k = (rng.standard_normal((kv_num_heads * d, K)) * 0.3).astype(np.float32)
    w_v = (rng.standard_normal((kv_num_heads * d, K)) * 0.3).astype(np.float32)
    w_q[:6] *= 8.0
    w_q[6:] *= 0.05
    w_k[:3] *= 8.0
    w_k[3:] *= 0.05
    w_v[:3] *= 8.0
    w_v[3:] *= 0.05
    bias_q = (rng.standard_normal(num_heads * d) * 0.05).astype(np.float32)
    bias_k = (rng.standard_normal(kv_num_heads * d) * 0.05).astype(np.float32)
    bias_v = (rng.standard_normal(kv_num_heads * d) * 0.05).astype(np.float32)
    norm_scale = (1.0 + rng.standard_normal(K) * 0.1).astype(np.float32)

    model, _info = _matmul_nbits_qkv_model(
        num_heads,
        kv_num_heads,
        d,
        K,
        block_size,
        N2,
        w_q,
        w_k,
        w_v,
        bias_q,
        bias_k,
        bias_v,
        norm_scale,
        consumer="nbits",
        consumer_block_size=consumer_block_size,
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)

    inits_before = {t.name: t.SerializeToString() for t in model.graph.initializer}
    inits_after = {t.name: t.SerializeToString() for t in pruned.graph.initializer}
    assert inits_before == inits_after
    attrs_before = [
        (n.name, [(a.name, a.i) for a in n.attribute]) for n in model.graph.node
    ]
    attrs_after = [
        (n.name, [(a.name, a.i) for a in n.attribute]) for n in pruned.graph.node
    ]
    assert attrs_before == attrs_after


def test_cpp_matmul_nbits_qkv_pruning_declines_when_attention_bias_present():
    # A genuinely per-head-shaped `attention_bias` (dims[1] == num_heads,
    # not 1) on the downstream GroupQueryAttention node -- this port has no
    # dynamic-Gather-insertion machinery to correctly re-slice it (see
    # `MatMulNBitsQkvAttentionExtrasSafe`'s own comment), so the whole chain
    # must be declined outright rather than silently leave it stale.
    num_heads, kv_num_heads, d, K, block_size, N2 = 4, 2, 2, 32, 32, 5
    batch, seq = 2, 5
    rng = np.random.default_rng(9320)
    w_q = (rng.standard_normal((num_heads * d, K)) * 0.3).astype(np.float32)
    w_k = (rng.standard_normal((kv_num_heads * d, K)) * 0.3).astype(np.float32)
    w_v = (rng.standard_normal((kv_num_heads * d, K)) * 0.3).astype(np.float32)
    bias_q = (rng.standard_normal(num_heads * d) * 0.05).astype(np.float32)
    bias_k = (rng.standard_normal(kv_num_heads * d) * 0.05).astype(np.float32)
    bias_v = (rng.standard_normal(kv_num_heads * d) * 0.05).astype(np.float32)
    norm_scale = (1.0 + rng.standard_normal(K) * 0.1).astype(np.float32)
    attention_bias = rng.standard_normal((1, num_heads, seq, seq)).astype(np.float32)

    model, _info = _matmul_nbits_qkv_model(
        num_heads,
        kv_num_heads,
        d,
        K,
        block_size,
        N2,
        w_q,
        w_k,
        w_v,
        bias_q,
        bias_k,
        bias_v,
        norm_scale,
        batch=batch,
        seq=seq,
        attention_bias=attention_bias,
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    inits_before = {t.name: t.SerializeToString() for t in model.graph.initializer}
    inits_after = {t.name: t.SerializeToString() for t in pruned.graph.initializer}
    assert inits_before == inits_after
    attrs_before = [
        (n.name, [(a.name, a.i) for a in n.attribute]) for n in model.graph.node
    ]
    attrs_after = [
        (n.name, [(a.name, a.i) for a in n.attribute]) for n in pruned.graph.node
    ]
    assert attrs_before == attrs_after


def test_cpp_matmul_nbits_qkv_pruning_zero_sparsity_is_a_no_op():
    num_heads, kv_num_heads, d, K, block_size, N2 = 4, 2, 2, 32, 32, 5
    rng = np.random.default_rng(9330)
    w_q = (rng.standard_normal((num_heads * d, K)) * 0.3).astype(np.float32)
    w_k = (rng.standard_normal((kv_num_heads * d, K)) * 0.3).astype(np.float32)
    w_v = (rng.standard_normal((kv_num_heads * d, K)) * 0.3).astype(np.float32)
    bias_q = (rng.standard_normal(num_heads * d) * 0.05).astype(np.float32)
    bias_k = (rng.standard_normal(kv_num_heads * d) * 0.05).astype(np.float32)
    bias_v = (rng.standard_normal(kv_num_heads * d) * 0.05).astype(np.float32)
    norm_scale = (1.0 + rng.standard_normal(K) * 0.1).astype(np.float32)

    model, _info = _matmul_nbits_qkv_model(
        num_heads,
        kv_num_heads,
        d,
        K,
        block_size,
        N2,
        w_q,
        w_k,
        w_v,
        bias_q,
        bias_k,
        bias_v,
        norm_scale,
    )
    pruned = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.0)
    qkv_node = next(n for n in pruned.graph.node if n.op_type == "MatMulNBitsQkv")
    assert next(a.i for a in qkv_node.attribute if a.name == "Nq") == num_heads * d


# --- importance_norm ("l1" vs "l2") ------------------------------------------
#
# Adapted from test_pruning.py's own `test_attention_head_pruning_l1_norm_
# favors_total_magnitude`/`test_gqa_pruning_l1_norm_favors_total_magnitude`:
# adversarial per-head/per-group weight blocks engineered so L2 (Frobenius)
# and L1 (entrywise abs-sum) importance disagree on which unit survives --
# a bug that silently keeps ranking by L2 under the hood even when "l1" is
# requested would keep the WRONG head/group, not merely score it slightly
# differently.


def test_cpp_attention_head_pruning_importance_norm_l1_matches_python_reference_and_differs_from_l2():
    K, H, D, Out = 16, 4, 4, 3
    Nq = Nk = Nv = H * D
    rng_qk = np.random.default_rng(52)
    wqkv = np.zeros((K, Nq + Nk + Nv), dtype=np.float32)
    wqkv[:, :Nq] = rng_qk.standard_normal((K, Nq)).astype(np.float32) * 0.01
    wqkv[:, Nq : Nq + Nk] = rng_qk.standard_normal((K, Nk)).astype(np.float32) * 0.01
    v_offset = Nq + Nk
    wqkv[0, v_offset + 0] = 16.0  # head 0 ("concentrated")
    wqkv[:, v_offset + D : v_offset + 2 * D] = 1.0  # head 1 ("spread")
    wqkv[2, v_offset + 2 * D] = 1000.0  # head 2 ("filler_high")
    wqkv[3, v_offset + 3 * D] = 0.001  # head 3 ("filler_low")
    bqkv = np.zeros((Nq + Nk + Nv,), dtype=np.float32)

    model, _cfg = _attention_model(
        K=K, H=H, D=D, Out=Out, seed=50, bias=True, wqkv=wqkv, bqkv=bqkv
    )

    for norm in ("l2", "l1"):
        pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(
            model, sparsity=0.5, importance_norm=norm
        )
        pruned_py = onnxsim.apply_attention_head_pruning(
            model, sparsity=0.5, importance_norm=norm
        )
        onnx.checker.check_model(pruned_cpp)
        assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    kept_l2 = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    kept_l1 = onnxsim.apply_attention_head_pruning_cpp(
        model, sparsity=0.5, importance_norm="l1"
    )
    # "l2" keeps {concentrated, filler_high} (16 & 1000 dominate Frobenius),
    # "l1" keeps {spread, filler_high} (64 total magnitude beats 16) --
    # provably different surviving Wqkv shapes/values, not just a different
    # score.
    assert kept_l2.SerializeToString() != kept_l1.SerializeToString()


def test_cpp_gqa_pruning_importance_norm_l1_matches_python_reference_and_differs_from_l2():
    K, H, KVH, D, Out = 8, 4, 2, 8, 3
    Nq, Nkv = H * D, KVH * D
    wq = np.zeros((K, Nq), dtype=np.float32)
    wk = np.zeros((K, Nkv), dtype=np.float32)
    wv = np.zeros((K, Nkv), dtype=np.float32)
    wv[0, 0] = 16.0  # KV group 0's own V slice -- concentrated
    wv[:, D : 2 * D] = 1.0  # KV group 1's own V slice -- spread

    model, _cfg = _gqa_model(
        K=K, H=H, KVH=KVH, D=D, Out=Out, seed=60, wq=wq, wk=wk, wv=wv
    )

    for norm in ("l2", "l1"):
        pruned_cpp = onnxsim.apply_attention_head_pruning_cpp(
            model, sparsity=0.5, importance_norm=norm
        )
        pruned_py = onnxsim.apply_attention_head_pruning(
            model, sparsity=0.5, importance_norm=norm
        )
        onnx.checker.check_model(pruned_cpp)
        assert pruned_cpp.SerializeToString() == pruned_py.SerializeToString()

    kept_l2 = onnxsim.apply_attention_head_pruning_cpp(model, sparsity=0.5)
    kept_l1 = onnxsim.apply_attention_head_pruning_cpp(
        model, sparsity=0.5, importance_norm="l1"
    )
    # "l2" keeps KV group 0 (Frobenius 16 > 8), "l1" keeps KV group 1
    # (total magnitude 64 > 16) -- a real flip in which group survives.
    assert kept_l2.SerializeToString() != kept_l1.SerializeToString()
