"""Tests for ``onnxsim.apply_structured_pruning_cpp`` -- the C++-backed port
of ``onnxsim.apply_structured_pruning`` (see
``onnxsim/structured_pruning_entry.cpp``). Scope note: this port covers only
the two "plain chain" topologies (a MatMul/vanilla-Gemm or Conv producer
feeding, through shape-preserving elementwise ops, exactly one consumer of
the same family) -- not the pure-Python implementation's gated-FFN or
residual/skip-connection chain support. Tests here are adapted from
``test_pruning.py``'s own ``apply_structured_pruning`` coverage, scoped to
the topologies this port actually implements, plus a couple of tests
confirming the unported topologies are left untouched (never guessed at).
"""

import numpy as np
import onnx
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


def _oracle_keep_indices(w1, keep_count):
    importance = np.linalg.norm(w1.T, axis=1)
    return np.sort(np.argsort(-importance)[:keep_count])


def _oracle_keep_indices_conv(w, keep_count):
    importance = np.linalg.norm(w.reshape(w.shape[0], -1).astype(np.float64), axis=1)
    return np.sort(np.argsort(-importance)[:keep_count])


def _oracle_keep_indices_conv_grouped(w, group, sparsity):
    out_channels = w.shape[0]
    block = out_channels // group
    per_group_keep = max(1, round(block * (1.0 - sparsity)))
    parts = []
    for gi in range(group):
        lo, hi = gi * block, (gi + 1) * block
        parts.append(_oracle_keep_indices_conv(w[lo:hi], per_group_keep) + lo)
    return np.concatenate(parts)


def _oracle_slice_grouped_consumer_conv(w2, keep, group, n_channels):
    out_channels = w2.shape[0]
    out_per_group = out_channels // group
    block = n_channels // group
    parts = []
    for gi in range(group):
        lo, hi = gi * block, (gi + 1) * block
        local_keep = keep[(keep >= lo) & (keep < hi)] - lo
        parts.append(w2[gi * out_per_group : (gi + 1) * out_per_group][:, local_keep])
    return np.concatenate(parts, axis=0)


def _mlp_model(K=8, H=32, Out=4, bias=True, activation="Relu", seed=0):
    rng = np.random.default_rng(seed)
    w1 = rng.standard_normal((K, H)).astype(np.float32)
    w2 = rng.standard_normal((H, Out)).astype(np.float32)
    initializer = [_f32(w1, "W1"), _f32(w2, "W2")]
    if bias:
        b1 = rng.standard_normal((H,)).astype(np.float32)
        gemm1 = "h = Gemm(X, W1, B1)"
        initializer.append(_f32(b1, "B1"))
    else:
        gemm1 = "h = MatMul(X, W1)"
    return _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          {gemm1}
          a = {activation}(h)
          Y = MatMul(a, W2)
        }}
        """,
        initializer=initializer,
    )


def _conv_pair_model(w1, w2, b1=None, spatial=10, activation="Relu"):
    Cin, C2 = w1.shape[1], w2.shape[0]
    initializer = [_f32(w1, "W1"), _f32(w2, "W2")]
    if b1 is not None:
        conv1 = "h = Conv<kernel_shape=[3,3]>(X, W1, B1)"
        initializer.append(_f32(b1, "B1"))
    else:
        conv1 = "h = Conv<kernel_shape=[3,3]>(X, W1)"
    out_spatial = spatial - 4
    return _model(
        f"""
        g (float[N,{Cin},{spatial},{spatial}] X) => (float[N,{C2},{out_spatial},{out_spatial}] Y)
        {{
          {conv1}
          a = {activation}(h)
          Y = Conv<kernel_shape=[3,3]>(a, W2)
        }}
        """,
        initializer=initializer,
    )


def _grouped_conv_pair_model(
    w1, w2, group1=1, group2=1, b1=None, spatial=10, activation="Relu"
):
    Cin, C2 = w1.shape[1] * group1, w2.shape[0]
    initializer = [_f32(w1, "W1"), _f32(w2, "W2")]
    g1 = f", group={group1}" if group1 != 1 else ""
    g2 = f", group={group2}" if group2 != 1 else ""
    if b1 is not None:
        conv1 = f"h = Conv<kernel_shape=[3,3]{g1}>(X, W1, B1)"
        initializer.append(_f32(b1, "B1"))
    else:
        conv1 = f"h = Conv<kernel_shape=[3,3]{g1}>(X, W1)"
    out_spatial = spatial - 4
    return _model(
        f"""
        g (float[N,{Cin},{spatial},{spatial}] X) => (float[N,{C2},{out_spatial},{out_spatial}] Y)
        {{
          {conv1}
          a = {activation}(h)
          Y = Conv<kernel_shape=[3,3]{g2}>(a, W2)
        }}
        """,
        initializer=initializer,
    )


def _depthwise_pair_model(w1, dw_hops, w2, b1=None, spatial=10, activation="Relu"):
    Cin, C2 = w1.shape[1], w2.shape[0]
    initializer = [_f32(w1, "W1"), _f32(w2, "W2")]
    if b1 is not None:
        lines = ["h0 = Conv<kernel_shape=[3,3]>(X, W1, B1)"]
        initializer.append(_f32(b1, "B1"))
    else:
        lines = ["h0 = Conv<kernel_shape=[3,3]>(X, W1)"]
    lines.append(f"a0 = {activation}(h0)")
    cur = "a0"
    n_convs = 1
    for i, (wd, bd) in enumerate(dw_hops):
        group = wd.shape[0]
        w_name, b_name = f"WD{i}", f"BD{i}"
        initializer.append(_f32(wd, w_name))
        if bd is not None:
            initializer.append(_f32(bd, b_name))
            lines.append(
                f"hd{i} = Conv<kernel_shape=[3,3], group={group}>"
                f"({cur}, {w_name}, {b_name})"
            )
        else:
            lines.append(
                f"hd{i} = Conv<kernel_shape=[3,3], group={group}>({cur}, {w_name})"
            )
        lines.append(f"ad{i} = {activation}(hd{i})")
        cur = f"ad{i}"
        n_convs += 1
    lines.append(f"Y = Conv<kernel_shape=[3,3]>({cur}, W2)")
    n_convs += 1
    out_spatial = spatial - 2 * n_convs
    body = "\n          ".join(lines)
    return _model(
        f"""
        g (float[N,{Cin},{spatial},{spatial}] X) => (float[N,{C2},{out_spatial},{out_spatial}] Y)
        {{
          {body}
        }}
        """,
        initializer=initializer,
    )


# --- MatMul/Gemm plain chains -----------------------------------------------


def test_cpp_structured_pruning_shrinks_matched_layers():
    model = _mlp_model(K=8, H=32, Out=4)
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1"].dims) == [8, 16]
    assert list(inits["B1"].dims) == [16]
    assert list(inits["W2"].dims) == [16, 4]


def test_cpp_structured_pruning_matches_manual_channel_deletion_exactly():
    K, H, Out = 8, 32, 4
    model = _mlp_model(K=K, H=H, Out=Out, bias=True)
    orig = {t.name: onnx.numpy_helper.to_array(t) for t in model.graph.initializer}
    w1, b1, w2 = orig["W1"], orig["B1"], orig["W2"]

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    keep = _oracle_keep_indices(w1, H // 2)

    rng = np.random.default_rng(1)
    x = rng.standard_normal((6, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})

    h = x @ w1[:, keep] + b1[keep]
    a = np.maximum(h, 0)
    y_oracle = a @ w2[keep, :]
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_matmul_only_chain_matches_oracle():
    K, H, Out = 8, 24, 4
    model = _mlp_model(K=K, H=H, Out=Out, bias=False, activation="Sigmoid")
    w1 = onnx.numpy_helper.to_array(model.graph.initializer[0])
    w2 = onnx.numpy_helper.to_array(model.graph.initializer[1])

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.25)
    keep = _oracle_keep_indices(w1, H - round(H * 0.25))

    rng = np.random.default_rng(2)
    x = rng.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})

    h = x @ w1[:, keep]
    a = 1.0 / (1.0 + np.exp(-h))
    y_oracle = a @ w2[keep, :]
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_bias_add_between_matmuls_matches_oracle():
    K, H, Out = 8, 16, 4
    rng = np.random.default_rng(3)
    w1 = rng.standard_normal((K, H)).astype(np.float32)
    bias = rng.standard_normal((H,)).astype(np.float32)
    w2 = rng.standard_normal((H, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          h = MatMul(X, W1)
          hb = Add(h, Bias)
          a = Relu(hb)
          Y = MatMul(a, W2)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(bias, "Bias"), _f32(w2, "W2")],
    )

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["Bias"].dims) == [H // 2]

    keep = _oracle_keep_indices(w1, H // 2)
    x = rng.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    h = x @ w1[:, keep] + bias[keep]
    a = np.maximum(h, 0)
    y_oracle = a @ w2[keep, :]
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_skips_branching_output():
    K, H, Out = 8, 16, 4
    model = _mlp_model(K=K, H=H, Out=Out, bias=False)
    model.graph.output.append(
        onnx.helper.make_tensor_value_info("h", onnx.TensorProto.FLOAT, ["batch", H])
    )

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1"].dims) == [K, H]
    assert list(inits["W2"].dims) == [H, Out]


def test_cpp_structured_pruning_skips_multi_consumer_branch():
    K, H, Out = 8, 16, 4
    rng = np.random.default_rng(4)
    w1 = rng.standard_normal((K, H)).astype(np.float32)
    w2 = rng.standard_normal((H, Out)).astype(np.float32)
    w3 = rng.standard_normal((H, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y1, float[batch,{Out}] Y2)
        {{
          h = MatMul(X, W1)
          Y1 = MatMul(h, W2)
          Y2 = MatMul(h, W3)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2"), _f32(w3, "W3")],
    )

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1"].dims) == [K, H]


def test_cpp_structured_pruning_zero_sparsity_is_a_no_op():
    model = _mlp_model(K=8, H=16, Out=4)
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.0)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1"].dims) == [8, 16]


def test_cpp_structured_pruning_invalid_sparsity_raises():
    model = _mlp_model(K=8, H=16, Out=4)
    with pytest.raises(Exception):
        onnxsim.apply_structured_pruning_cpp(model, sparsity=1.0)
    with pytest.raises(Exception):
        onnxsim.apply_structured_pruning_cpp(model, sparsity=-0.1)


def test_cpp_structured_pruning_chains_through_a_third_layer():
    K, H1, H2, Out = 8, 16, 20, 4
    rng = np.random.default_rng(5)
    w1 = rng.standard_normal((K, H1)).astype(np.float32)
    w2 = rng.standard_normal((H1, H2)).astype(np.float32)
    w3 = rng.standard_normal((H2, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          h1 = MatMul(X, W1)
          a1 = Relu(h1)
          h2 = MatMul(a1, W2)
          a2 = Relu(h2)
          Y = MatMul(a2, W3)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2"), _f32(w3, "W3")],
    )

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1"].dims) == [K, H1 // 2]
    assert list(inits["W2"].dims) == [H1 // 2, H2 // 2]
    assert list(inits["W3"].dims) == [H2 // 2, Out]

    keep1 = _oracle_keep_indices(w1, H1 // 2)
    keep2 = _oracle_keep_indices(w2[keep1, :], H2 // 2)

    rng2 = np.random.default_rng(6)
    x = rng2.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    a1 = np.maximum(x @ w1[:, keep1], 0)
    a2 = np.maximum(a1 @ w2[np.ix_(keep1, keep2)], 0)
    y_oracle = a2 @ w3[keep2, :]
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


# --- Gated FFN: unsupported by this port, must stay untouched --------------


def test_cpp_structured_pruning_gated_ffn_is_left_untouched():
    K, H, Out = 8, 16, 4
    rng = np.random.default_rng(7)
    wg = rng.standard_normal((K, H)).astype(np.float32)
    wu = rng.standard_normal((K, H)).astype(np.float32)
    wd = rng.standard_normal((H, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          gate = MatMul(X, Wg)
          gate_act = Sigmoid(gate)
          up = MatMul(X, Wu)
          h = Mul(gate_act, up)
          Y = MatMul(h, Wd)
        }}
        """,
        initializer=[_f32(wg, "Wg"), _f32(wu, "Wu"), _f32(wd, "Wd")],
    )
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["Wg"], wg)
    np.testing.assert_array_equal(inits["Wu"], wu)
    np.testing.assert_array_equal(inits["Wd"], wd)


# --- Conv plain chains -------------------------------------------------------


def test_cpp_structured_pruning_conv_chain_shrinks_matched_layers():
    Cin, C1, C2 = 3, 16, 8
    rng = np.random.default_rng(30)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    b1 = rng.standard_normal((C1,)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    model = _conv_pair_model(w1, w2, b1=b1)
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1"].dims) == [C1 // 2, Cin, 3, 3]
    assert list(inits["B1"].dims) == [C1 // 2]
    assert list(inits["W2"].dims) == [C2, C1 // 2, 3, 3]


def test_cpp_structured_pruning_conv_chain_matches_manual_channel_deletion_exactly():
    Cin, C1, C2 = 3, 16, 8
    rng = np.random.default_rng(30)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    b1 = rng.standard_normal((C1,)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    model = _conv_pair_model(w1, w2, b1=b1)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    keep = _oracle_keep_indices_conv(w1, C1 // 2)
    oracle = _conv_pair_model(w1[keep], w2[:, keep], b1=b1[keep])

    rng_x = np.random.default_rng(31)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_conv_only_chain_matches_oracle_no_bias():
    Cin, C1, C2 = 4, 12, 6
    rng = np.random.default_rng(32)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    model = _conv_pair_model(w1, w2, activation="Sigmoid")

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.25)
    onnx.checker.check_model(pruned)

    keep = _oracle_keep_indices_conv(w1, C1 - round(C1 * 0.25))
    oracle = _conv_pair_model(w1[keep], w2[:, keep], activation="Sigmoid")

    rng_x = np.random.default_rng(33)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_skips_grouped_producer_conv():
    C = 8
    rng = np.random.default_rng(34)
    w1 = rng.standard_normal((C, 1, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C, C, 3, 3)).astype(np.float32)
    model = _model(
        f"""
        g (float[N,{C},10,10] X) => (float[N,{C},6,6] Y)
        {{
          h = Conv<kernel_shape=[3,3], group={C}>(X, W1)
          a = Relu(h)
          Y = Conv<kernel_shape=[3,3]>(a, W2)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2")],
    )
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["W2"], w2)


def test_cpp_structured_pruning_skips_grouped_consumer_conv():
    Cin, C1 = 3, 8
    rng = np.random.default_rng(35)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C1, 1, 3, 3)).astype(np.float32)
    model = _model(
        f"""
        g (float[N,{Cin},10,10] X) => (float[N,{C1},6,6] Y)
        {{
          h = Conv<kernel_shape=[3,3]>(X, W1)
          a = Relu(h)
          Y = Conv<kernel_shape=[3,3], group={C1}>(a, W2)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2")],
    )
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["W2"], w2)


def test_cpp_structured_pruning_conv_into_non_pass_through_op_is_left_untouched():
    Cin, C1, Out = 3, 8, 4
    rng = np.random.default_rng(36)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C1, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[N,{Cin},10,10] X) => (float[N,{Out}] Y)
        {{
          h = Conv<kernel_shape=[3,3]>(X, W1)
          p = GlobalAveragePool(h)
          f = Flatten<axis=1>(p)
          Y = MatMul(f, W2)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2")],
    )
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["W2"], w2)


def test_cpp_structured_pruning_conv_chain_scale_between_convs_is_left_untouched():
    Cin, C1, C2 = 3, 8, 4
    rng = np.random.default_rng(37)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    scale = rng.standard_normal((1, C1, 1, 1)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    model = _model(
        f"""
        g (float[N,{Cin},10,10] X) => (float[N,{C2},6,6] Y)
        {{
          h = Conv<kernel_shape=[3,3]>(X, W1)
          s = Mul(h, Scale)
          a = Relu(s)
          Y = Conv<kernel_shape=[3,3]>(a, W2)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(scale, "Scale"), _f32(w2, "W2")],
    )
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["W2"], w2)


# --- Depthwise Conv pass-through hops ----------------------------------------


def test_cpp_structured_pruning_depthwise_pass_through_matches_manual_channel_deletion_exactly():
    Cin, C1, C2 = 3, 16, 8
    rng = np.random.default_rng(50)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    b1 = rng.standard_normal((C1,)).astype(np.float32)
    wd = rng.standard_normal((C1, 1, 3, 3)).astype(np.float32)
    bd = rng.standard_normal((C1,)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    model = _depthwise_pair_model(w1, [(wd, bd)], w2, b1=b1)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    assert inits["WD0"].shape == (C1 // 2, 1, 3, 3)
    assert inits["BD0"].shape == (C1 // 2,)
    dw_node = next(n for n in pruned.graph.node if "WD0" in n.input)
    group_attr = next(a for a in dw_node.attribute if a.name == "group")
    assert group_attr.i == C1 // 2

    keep = _oracle_keep_indices_conv(w1, C1 // 2)
    oracle = _depthwise_pair_model(
        w1[keep], [(wd[keep], bd[keep])], w2[:, keep], b1=b1[keep]
    )

    rng_x = np.random.default_rng(51)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_multiple_consecutive_depthwise_pass_through_hops_matches_oracle():
    Cin, C1, C2 = 3, 12, 6
    rng = np.random.default_rng(52)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    wd1 = rng.standard_normal((C1, 1, 3, 3)).astype(np.float32)
    bd1 = rng.standard_normal((C1,)).astype(np.float32)
    wd2 = rng.standard_normal((C1, 1, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    model = _depthwise_pair_model(w1, [(wd1, bd1), (wd2, None)], w2, spatial=14)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.25)
    onnx.checker.check_model(pruned)

    keep = _oracle_keep_indices_conv(w1, C1 - round(C1 * 0.25))
    oracle = _depthwise_pair_model(
        w1[keep], [(wd1[keep], bd1[keep]), (wd2[keep], None)], w2[:, keep], spatial=14
    )

    rng_x = np.random.default_rng(53)
    x = rng_x.standard_normal((2, Cin, 14, 14)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_depthwise_pass_through_no_bias_matches_oracle():
    Cin, C1, C2 = 4, 10, 5
    rng = np.random.default_rng(54)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    wd = rng.standard_normal((C1, 1, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    model = _depthwise_pair_model(w1, [(wd, None)], w2, activation="Sigmoid")

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.3)
    onnx.checker.check_model(pruned)

    keep = _oracle_keep_indices_conv(w1, C1 - round(C1 * 0.3))
    oracle = _depthwise_pair_model(
        w1[keep], [(wd[keep], None)], w2[:, keep], activation="Sigmoid"
    )

    rng_x = np.random.default_rng(55)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_depthwise_pass_through_branch_is_left_untouched():
    Cin, C1 = 3, 8
    rng = np.random.default_rng(56)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    wd = rng.standard_normal((C1, 1, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C1, C1, 3, 3)).astype(np.float32)
    model = _model(
        f"""
        g (float[N,{Cin},10,10] X) => (float[N,{C1},4,4] Y1, float[N,{C1},6,6] Y2)
        {{
          h = Conv<kernel_shape=[3,3]>(X, W1)
          a = Relu(h)
          d = Conv<kernel_shape=[3,3], group={C1}>(a, WD)
          Y1 = Conv<kernel_shape=[3,3]>(d, W2)
          Y2 = Relu(d)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(wd, "WD"), _f32(w2, "W2")],
    )
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["WD"], wd)
    np.testing.assert_array_equal(inits["W2"], w2)


# --- General grouped Conv -----------------------------------------------------


def test_cpp_structured_pruning_general_grouped_producer_conv_prunes_per_group_independently():
    Cin, C1, C2, group = 4, 8, 4, 2
    rng = np.random.default_rng(80)
    w1 = rng.standard_normal((C1, Cin // group, 3, 3)).astype(np.float32)
    w1[:4] *= 10.0
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    model = _grouped_conv_pair_model(w1, w2, group1=group)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    keep_grouped = _oracle_keep_indices_conv_grouped(w1, group, 0.5)
    assert sum(i < 4 for i in keep_grouped) == 2
    assert sum(i >= 4 for i in keep_grouped) == 2

    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1[keep_grouped])
    dw_node = next(n for n in pruned.graph.node if "W1" in n.input)
    group_attr = next(a.i for a in dw_node.attribute if a.name == "group")
    assert group_attr == group

    oracle = _grouped_conv_pair_model(
        w1[keep_grouped], w2[:, keep_grouped], group1=group
    )
    rng_x = np.random.default_rng(81)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_general_grouped_consumer_conv_matches_manual_channel_deletion_exactly():
    Cin, C1, C2, group = 3, 8, 6, 2
    rng = np.random.default_rng(82)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    w1[:4] *= 8.0
    w2 = rng.standard_normal((C2, C1 // group, 3, 3)).astype(np.float32)
    model = _grouped_conv_pair_model(w1, w2, group2=group)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    keep = _oracle_keep_indices_conv_grouped(w1, group, 0.5)
    w2_sliced = _oracle_slice_grouped_consumer_conv(w2, keep, group, C1)

    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1[keep])
    np.testing.assert_array_equal(inits["W2"], w2_sliced)

    oracle = _grouped_conv_pair_model(w1[keep], w2_sliced, group2=group)
    rng_x = np.random.default_rng(83)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_both_sides_grouped_matching_group_count_matches_oracle():
    Cin, C1, C2, group = 4, 8, 6, 2
    rng = np.random.default_rng(84)
    w1 = rng.standard_normal((C1, Cin // group, 3, 3)).astype(np.float32)
    w1[:4] *= 6.0
    w2 = rng.standard_normal((C2, C1 // group, 3, 3)).astype(np.float32)
    model = _grouped_conv_pair_model(w1, w2, group1=group, group2=group)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    keep = _oracle_keep_indices_conv_grouped(w1, group, 0.5)
    w2_sliced = _oracle_slice_grouped_consumer_conv(w2, keep, group, C1)
    oracle = _grouped_conv_pair_model(w1[keep], w2_sliced, group1=group, group2=group)

    rng_x = np.random.default_rng(85)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_skips_mismatched_grouped_producer_and_consumer():
    Cin, C1, C2, gp, gc = 4, 8, 8, 2, 4
    rng = np.random.default_rng(86)
    w1 = rng.standard_normal((C1, Cin // gp, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1 // gc, 3, 3)).astype(np.float32)
    model = _grouped_conv_pair_model(w1, w2, group1=gp, group2=gc)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["W2"], w2)


# --- Cross-check against the pure-Python reference --------------------------


def test_cpp_structured_pruning_matches_python_reference_output():
    K, H, Out = 8, 32, 4
    model = _mlp_model(K=K, H=H, Out=Out, bias=True, seed=9)
    pruned_py = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    pruned_cpp = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned_py)
    onnx.checker.check_model(pruned_cpp)

    rng = np.random.default_rng(10)
    x = rng.standard_normal((6, K)).astype(np.float32)
    (y_py,) = _run(pruned_py, {"X": x})
    (y_cpp,) = _run(pruned_cpp, {"X": x})
    np.testing.assert_allclose(y_py, y_cpp, rtol=1e-5, atol=1e-5)
