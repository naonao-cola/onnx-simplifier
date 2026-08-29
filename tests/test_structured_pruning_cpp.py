"""Tests for ``onnxsim.apply_structured_pruning_cpp`` -- the C++-backed port
of ``onnxsim.apply_structured_pruning`` (see
``onnxsim/structured_pruning_entry.cpp``). Scope note: this port covers the
"plain chain" topologies (a MatMul/vanilla-Gemm or Conv producer feeding,
through shape-preserving elementwise ops, exactly one consumer of the same
family), the gated-FFN (SwiGLU/GeGLU) topology (two producers combined by
``Mul`` or the native ``SwiGLU`` op, pruned to a shared combined-importance
channel set), and Conv/MatMul residual (skip-connection) chains (a
channel-preserving merge point -- a bare ``Add(a, b)`` for either family, or,
MatMul/Gemm only, a fused
``com.microsoft::SkipLayerNormalization``/``SkipSimplifiedLayerNormalization``
node -- resolved via backward walk plus union-find grouping). Tests here are
adapted from ``test_pruning.py``'s own ``apply_structured_pruning`` coverage,
plus a couple of tests confirming unmatched topologies are left untouched
(never guessed at).
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


# --- Gated FFN (SwiGLU/GeGLU) -----------------------------------------------


def _combined_keep_indices(w_gate, w_up, keep_count):
    importance = np.sqrt(
        np.square(np.linalg.norm(w_gate.T, axis=1))
        + np.square(np.linalg.norm(w_up.T, axis=1))
    )
    return np.sort(np.argsort(-importance)[:keep_count])


def _swiglu_mlp_model(K=8, H=16, Out=4, gate_activation="Sigmoid", seed=0):
    rng = np.random.default_rng(seed)
    wg = rng.standard_normal((K, H)).astype(np.float32)
    wu = rng.standard_normal((K, H)).astype(np.float32)
    wd = rng.standard_normal((H, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          gate = MatMul(X, Wg)
          gate_act = {gate_activation}(gate)
          up = MatMul(X, Wu)
          h = Mul(gate_act, up)
          Y = MatMul(h, Wd)
        }}
        """,
        initializer=[_f32(wg, "Wg"), _f32(wu, "Wu"), _f32(wd, "Wd")],
    )
    return model, wg, wu, wd


def test_cpp_structured_pruning_gated_ffn_matches_oracle():
    K, H, Out = 8, 16, 4
    model, wg, wu, wd = _swiglu_mlp_model(K=K, H=H, Out=Out)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["Wg"].dims) == [K, H // 2]
    assert list(inits["Wu"].dims) == [K, H // 2]
    assert list(inits["Wd"].dims) == [H // 2, Out]

    keep = _combined_keep_indices(wg, wu, H // 2)
    rng = np.random.default_rng(10)
    x = rng.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})

    gate = 1.0 / (1.0 + np.exp(-(x @ wg[:, keep])))
    up = x @ wu[:, keep]
    y_oracle = (gate * up) @ wd[keep, :]
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_gated_ffn_prunes_both_branches_to_same_channels():
    # The real bug this pattern risks: gate and up disagreeing on which
    # channels survive, which would silently break the elementwise
    # product's alignment. Assert they select the identical index set,
    # not just that both shrank to the same *count*.
    K, H, Out = 8, 20, 4
    model, wg, wu, _ = _swiglu_mlp_model(K=K, H=H, Out=Out, seed=1)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.3)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    keep = _combined_keep_indices(wg, wu, H - round(H * 0.3))

    np.testing.assert_array_equal(inits["Wg"], wg[:, keep])
    np.testing.assert_array_equal(inits["Wu"], wu[:, keep])


def test_cpp_structured_pruning_gelu_gated_ffn_matches_oracle():
    # GeGLU: same gated topology, a different (still-unary) gate activation.
    # Uses Gelu's tanh approximation so the oracle needs no scipy/erf.
    K, H, Out = 8, 16, 4
    rng = np.random.default_rng(11)
    wg = rng.standard_normal((K, H)).astype(np.float32)
    wu = rng.standard_normal((K, H)).astype(np.float32)
    wd = rng.standard_normal((H, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          gate = MatMul(X, Wg)
          gate_act = Gelu<approximate = "tanh">(gate)
          up = MatMul(X, Wu)
          h = Mul(gate_act, up)
          Y = MatMul(h, Wd)
        }}
        """,
        initializer=[_f32(wg, "Wg"), _f32(wu, "Wu"), _f32(wd, "Wd")],
    )

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    keep = _combined_keep_indices(wg, wu, H // 2)

    x = rng.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})

    g = x @ wg[:, keep]
    gate = 0.5 * g * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (g + 0.044715 * g**3)))
    up = x @ wu[:, keep]
    y_oracle = (gate * up) @ wd[keep, :]
    np.testing.assert_allclose(y, y_oracle, rtol=1e-4, atol=1e-4)


def test_cpp_structured_pruning_ungated_mul_of_two_producers_still_matches_oracle():
    # No activation at all on either branch -- a plain (unactivated) GLU,
    # both Mul operands are raw producer outputs directly.
    K, H, Out = 8, 12, 4
    rng = np.random.default_rng(2)
    w1 = rng.standard_normal((K, H)).astype(np.float32)
    w2 = rng.standard_normal((K, H)).astype(np.float32)
    w3 = rng.standard_normal((H, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          a = MatMul(X, W1)
          b = MatMul(X, W2)
          h = Mul(a, b)
          Y = MatMul(h, W3)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2"), _f32(w3, "W3")],
    )

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    keep = _combined_keep_indices(w1, w2, H // 2)

    x = rng.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    y_oracle = ((x @ w1[:, keep]) * (x @ w2[:, keep])) @ w3[keep, :]
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_gated_mul_against_constant_scale_is_not_a_gate():
    # Mul(a, constant) is the existing per-channel-scale chain continuation
    # (already covered elsewhere), not a two-producer gated pair -- the
    # constant operand must never be mistaken for a second producer.
    K, H, Out = 8, 12, 4
    rng = np.random.default_rng(3)
    w1 = rng.standard_normal((K, H)).astype(np.float32)
    scale = rng.standard_normal((H,)).astype(np.float32)
    w2 = rng.standard_normal((H, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          a = MatMul(X, W1)
          h = Mul(a, Scale)
          Y = MatMul(h, W2)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(scale, "Scale"), _f32(w2, "W2")],
    )

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1"].dims) == [K, H // 2]
    assert list(inits["Scale"].dims) == [H // 2]
    assert list(inits["W2"].dims) == [H // 2, Out]


def test_cpp_structured_pruning_gated_ffn_skips_when_a_branch_also_feeds_elsewhere():
    # "up" also feeding a second consumer directly means pruning its
    # channels would silently change what that other consumer sees --
    # must be left completely untouched, same bar as the plain-chain case.
    K, H, Out = 8, 12, 4
    rng = np.random.default_rng(4)
    wg = rng.standard_normal((K, H)).astype(np.float32)
    wu = rng.standard_normal((K, H)).astype(np.float32)
    wd = rng.standard_normal((H, Out)).astype(np.float32)
    wother = rng.standard_normal((H, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y1, float[batch,{Out}] Y2)
        {{
          gate = MatMul(X, Wg)
          gate_act = Sigmoid(gate)
          up = MatMul(X, Wu)
          h = Mul(gate_act, up)
          Y1 = MatMul(h, Wd)
          Y2 = MatMul(up, Wother)
        }}
        """,
        initializer=[
            _f32(wg, "Wg"),
            _f32(wu, "Wu"),
            _f32(wd, "Wd"),
            _f32(wother, "Wother"),
        ],
    )

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["Wg"].dims) == [K, H]
    assert list(inits["Wu"].dims) == [K, H]
    assert list(inits["Wd"].dims) == [H, Out]


def test_cpp_structured_pruning_native_swiglu_node_prunes_both_producers_together():
    # ONNX's native fused SwiGLU(a, b) = swish(a) * b (opset 28+): the
    # activation lives entirely inside the op, so a/b must be raw producer
    # outputs with no separate activation node in between. Not yet
    # supported by the installed onnx checker/onnxruntime in this
    # environment (opset 28 is still under development upstream), so this
    # verifies the graph surgery directly via tensor values rather than
    # onnx.checker/onnxruntime execution.
    K, H, Out = 8, 16, 4
    rng = np.random.default_rng(5)
    wg = rng.standard_normal((K, H)).astype(np.float32)
    wu = rng.standard_normal((K, H)).astype(np.float32)
    wd = rng.standard_normal((H, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          gate = MatMul(X, Wg)
          up = MatMul(X, Wu)
          h = SwiGLU(gate, up)
          Y = MatMul(h, Wd)
        }}
        """,
        initializer=[_f32(wg, "Wg"), _f32(wu, "Wu"), _f32(wd, "Wd")],
        opset=28,
    )

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    keep = _combined_keep_indices(wg, wu, H // 2)

    np.testing.assert_array_equal(inits["Wg"], wg[:, keep])
    np.testing.assert_array_equal(inits["Wu"], wu[:, keep])
    np.testing.assert_array_equal(inits["Wd"], wd[keep, :])


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


# --- Conv residual (Add-merged) chains --------------------------------------


def _residual_diamond_model(w_f, w_s, w_out, spatial=10):
    # y = Conv_out(Relu(Add(Conv_f(X), Conv_s(X)))) -- a "projection
    # shortcut" residual block: two entirely independent Conv producers
    # merge via Add and must therefore share one surviving channel-index
    # set, feeding one real consumer.
    Cin = w_f.shape[1]
    Cout = w_out.shape[0]
    out_spatial = spatial - 4  # two chained 3x3 valid convs
    return _model(
        f"""
        g (float[N,{Cin},{spatial},{spatial}] X) => (float[N,{Cout},{out_spatial},{out_spatial}] Y)
        {{
          f = Conv<kernel_shape=[3,3]>(X, WF)
          s = Conv<kernel_shape=[3,3]>(X, WS)
          addr = Add(f, s)
          r = Relu(addr)
          Y = Conv<kernel_shape=[3,3]>(r, WOUT)
        }}
        """,
        initializer=[_f32(w_f, "WF"), _f32(w_s, "WS"), _f32(w_out, "WOUT")],
    )


def _residual_transitive_model(w_f1, w_s1, w_f2, w_out, spatial=10):
    # Two Add merges chained transitively, sharing one spine channel count,
    # with no branch anywhere along the chain: add1's own output feeds only
    # into add2, never reused elsewhere -- the union-find grouping extends
    # across both Adds into one group of three producers.
    Cin = w_f1.shape[1]
    Cz = w_f2.shape[1]
    Cout = w_out.shape[0]
    add1_spatial = spatial - 2  # one 3x3 valid conv each, from X
    out_spatial = add1_spatial - 2  # WOUT's own 3x3 valid conv
    return _model(
        f"""
        g (float[N,{Cin},{spatial},{spatial}] X, float[N,{Cz},{add1_spatial},{add1_spatial}] Z)
            => (float[N,{Cout},{out_spatial},{out_spatial}] Y)
        {{
          f1 = Conv<kernel_shape=[3,3]>(X, WF1)
          s1 = Conv<kernel_shape=[3,3]>(X, WS1)
          add1 = Add(f1, s1)
          f2 = Conv<kernel_shape=[1,1]>(Z, WF2)
          add2 = Add(f2, add1)
          r = Relu(add2)
          Y = Conv<kernel_shape=[3,3]>(r, WOUT)
        }}
        """,
        initializer=[
            _f32(w_f1, "WF1"),
            _f32(w_s1, "WS1"),
            _f32(w_f2, "WF2"),
            _f32(w_out, "WOUT"),
        ],
    )


def test_cpp_structured_pruning_conv_residual_add_matches_oracle():
    Cin, C, Cout = 3, 16, 8
    rng = np.random.default_rng(80)
    w_f = rng.standard_normal((C, Cin, 3, 3)).astype(np.float32)
    w_s = rng.standard_normal((C, Cin, 3, 3)).astype(np.float32)
    w_out = rng.standard_normal((Cout, C, 3, 3)).astype(np.float32)
    model = _residual_diamond_model(w_f, w_s, w_out)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["WF"].dims) == [C // 2, Cin, 3, 3]
    assert list(inits["WS"].dims) == [C // 2, Cin, 3, 3]
    assert list(inits["WOUT"].dims) == [Cout, C // 2, 3, 3]

    importance = np.sqrt(
        np.square(np.linalg.norm(w_f.reshape(C, -1).astype(np.float64), axis=1))
        + np.square(np.linalg.norm(w_s.reshape(C, -1).astype(np.float64), axis=1))
    )
    keep = np.sort(np.argsort(-importance)[: C // 2])
    oracle = _residual_diamond_model(w_f[keep], w_s[keep], w_out[:, keep])

    rng_x = np.random.default_rng(81)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_conv_residual_add_transitive_chain_matches_oracle():
    Cin, C, Cz, Cout = 3, 16, 5, 8
    rng = np.random.default_rng(82)
    w_f1 = rng.standard_normal((C, Cin, 3, 3)).astype(np.float32)
    w_s1 = rng.standard_normal((C, Cin, 3, 3)).astype(np.float32)
    w_f2 = rng.standard_normal((C, Cz, 1, 1)).astype(np.float32)
    w_out = rng.standard_normal((Cout, C, 3, 3)).astype(np.float32)
    model = _residual_transitive_model(w_f1, w_s1, w_f2, w_out)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    importance = np.sqrt(
        np.square(np.linalg.norm(w_f1.reshape(C, -1).astype(np.float64), axis=1))
        + np.square(np.linalg.norm(w_s1.reshape(C, -1).astype(np.float64), axis=1))
        + np.square(np.linalg.norm(w_f2.reshape(C, -1).astype(np.float64), axis=1))
    )
    keep = np.sort(np.argsort(-importance)[: C // 2])
    oracle = _residual_transitive_model(
        w_f1[keep], w_s1[keep], w_f2[keep], w_out[:, keep]
    )

    rng_x = np.random.default_rng(83)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    z = rng_x.standard_normal((2, Cz, 8, 8)).astype(np.float32)
    (y,) = _run(pruned, {"X": x, "Z": z})
    (y_oracle,) = _run(oracle, {"X": x, "Z": z})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_conv_residual_add_declines_on_fan_out_branch():
    # A realistic multi-block residual stage's interior boundary: `r`
    # (add1's own post-block tensor) is read twice -- once by the next
    # block's own first Conv, once unchanged as that next block's own Add
    # shortcut operand -- exactly the fan-out backward walk's
    # single-consumer bar declines. Left completely untouched.
    Cin, C, Cout = 3, 16, 8
    rng = np.random.default_rng(84)
    w_f = rng.standard_normal((C, Cin, 3, 3)).astype(np.float32)
    w_s = rng.standard_normal((C, Cin, 3, 3)).astype(np.float32)
    w_next = rng.standard_normal((C, C, 1, 1)).astype(np.float32)
    w_out = rng.standard_normal((Cout, C, 1, 1)).astype(np.float32)
    model = _model(
        f"""
        g (float[N,{Cin},10,10] X) => (float[N,{Cout},8,8] Y)
        {{
          f = Conv<kernel_shape=[3,3]>(X, WF)
          s = Conv<kernel_shape=[3,3]>(X, WS)
          add1 = Add(f, s)
          r = Relu(add1)
          nxt = Conv<kernel_shape=[1,1]>(r, WNEXT)
          add2 = Add(nxt, r)
          Y = Conv<kernel_shape=[1,1]>(add2, WOUT)
        }}
        """,
        initializer=[
            _f32(w_f, "WF"),
            _f32(w_s, "WS"),
            _f32(w_next, "WNEXT"),
            _f32(w_out, "WOUT"),
        ],
    )
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["WF"], w_f)
    np.testing.assert_array_equal(inits["WS"], w_s)
    np.testing.assert_array_equal(inits["WNEXT"], w_next)
    np.testing.assert_array_equal(inits["WOUT"], w_out)


def test_cpp_structured_pruning_conv_residual_add_declines_on_identity_shortcut():
    # y = Conv2(Relu(Add(Conv1(X), X))): a classic identity-shortcut
    # residual block with no Conv on the shortcut path at all. X has no
    # producer this pass owns (it's a graph input) and is itself read
    # twice (by Conv1 and directly by Add) -- either alone is enough to
    # decline.
    C, Cout = 8, 4
    rng = np.random.default_rng(85)
    w1 = rng.standard_normal((C, C, 1, 1)).astype(np.float32)
    w2 = rng.standard_normal((Cout, C, 1, 1)).astype(np.float32)
    model = _model(
        f"""
        g (float[N,{C},10,10] X) => (float[N,{Cout},10,10] Y)
        {{
          f = Conv<kernel_shape=[1,1]>(X, W1)
          add1 = Add(f, X)
          r = Relu(add1)
          Y = Conv<kernel_shape=[1,1]>(r, W2)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2")],
    )
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["W2"], w2)


def test_cpp_structured_pruning_conv_residual_add_declines_on_grouped_conv_consumer():
    # Two independent Conv branches merge via Add, but the downstream
    # consumer is a general grouped Conv (group=2) -- a composition the
    # residual finder explicitly declines (_chain_group's per-group top-k
    # assumes each producer feeds the consumer's full channel range, which a
    # residual group's combined-importance ranking doesn't establish).
    Cin, C, Cout, group = 3, 16, 8, 2
    rng = np.random.default_rng(89)
    w_f = rng.standard_normal((C, Cin, 3, 3)).astype(np.float32)
    w_s = rng.standard_normal((C, Cin, 3, 3)).astype(np.float32)
    w_out = rng.standard_normal((Cout, C // group, 1, 1)).astype(np.float32)
    model = _model(
        f"""
        g (float[N,{Cin},10,10] X) => (float[N,{Cout},8,8] Y)
        {{
          f = Conv<kernel_shape=[3,3]>(X, WF)
          s = Conv<kernel_shape=[3,3]>(X, WS)
          addr = Add(f, s)
          r = Relu(addr)
          Y = Conv<kernel_shape=[1,1],group={group}>(r, WOUT)
        }}
        """,
        initializer=[_f32(w_f, "WF"), _f32(w_s, "WS"), _f32(w_out, "WOUT")],
    )
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["WF"], w_f)
    np.testing.assert_array_equal(inits["WS"], w_s)
    np.testing.assert_array_equal(inits["WOUT"], w_out)


# --- MatMul/Gemm residual (Add-merged) chains -------------------------------


def _matmul_residual_diamond_model(wf, ws, wout):
    # y = MatMul_out(Relu(Add(MatMul_f(X), MatMul_s(X)))) -- the MatMul/Gemm
    # analogue of _residual_diamond_model.
    K, C = wf.shape
    Out = wout.shape[1]
    return _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          f = MatMul(X, WF)
          s = MatMul(X, WS)
          addr = Add(f, s)
          r = Relu(addr)
          Y = MatMul(r, WOUT)
        }}
        """,
        initializer=[_f32(wf, "WF"), _f32(ws, "WS"), _f32(wout, "WOUT")],
    )


def _matmul_residual_transitive_model(wf1, ws1, wf2, wout):
    # Two Add merges chained transitively, sharing one spine channel count
    # -- the MatMul/Gemm analogue of _residual_transitive_model.
    K, C = wf1.shape
    Kz = wf2.shape[0]
    Out = wout.shape[1]
    return _model(
        f"""
        g (float[batch,{K}] X, float[batch,{Kz}] Z) => (float[batch,{Out}] Y)
        {{
          f1 = MatMul(X, WF1)
          s1 = MatMul(X, WS1)
          add1 = Add(f1, s1)
          f2 = MatMul(Z, WF2)
          add2 = Add(f2, add1)
          r = Relu(add2)
          Y = MatMul(r, WOUT)
        }}
        """,
        initializer=[
            _f32(wf1, "WF1"),
            _f32(ws1, "WS1"),
            _f32(wf2, "WF2"),
            _f32(wout, "WOUT"),
        ],
    )


def test_cpp_structured_pruning_matmul_residual_add_matches_oracle():
    # Weights deliberately built so the two branches disagree about which
    # channels matter most, so the correct combined-importance keep set is
    # neither branch's own individual top-k.
    K, C, Out = 8, 16, 4
    rng = np.random.default_rng(90)
    scale_f = np.where(np.arange(C) < C // 2, 3.0, 0.3).astype(np.float32)
    scale_s = np.where(np.arange(C) < C // 2, 0.3, 3.0).astype(np.float32)
    wf = rng.standard_normal((K, C)).astype(np.float32) * scale_f
    ws = rng.standard_normal((K, C)).astype(np.float32) * scale_s
    wout = rng.standard_normal((C, Out)).astype(np.float32)
    model = _matmul_residual_diamond_model(wf, ws, wout)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    importance = np.sqrt(
        np.square(np.linalg.norm(wf.astype(np.float64), axis=0))
        + np.square(np.linalg.norm(ws.astype(np.float64), axis=0))
    )
    keep = np.sort(np.argsort(-importance)[: C // 2])
    assert np.any(keep < C // 2) and np.any(keep >= C // 2)
    oracle = _matmul_residual_diamond_model(wf[:, keep], ws[:, keep], wout[keep, :])

    rng_x = np.random.default_rng(91)
    x = rng_x.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_matmul_residual_add_transitive_chain_matches_oracle():
    K, C, Kz, Out = 8, 16, 5, 4
    rng = np.random.default_rng(92)
    wf1 = rng.standard_normal((K, C)).astype(np.float32)
    ws1 = rng.standard_normal((K, C)).astype(np.float32)
    wf2 = rng.standard_normal((Kz, C)).astype(np.float32)
    wout = rng.standard_normal((C, Out)).astype(np.float32)
    model = _matmul_residual_transitive_model(wf1, ws1, wf2, wout)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    importance = np.sqrt(
        np.square(np.linalg.norm(wf1.astype(np.float64), axis=0))
        + np.square(np.linalg.norm(ws1.astype(np.float64), axis=0))
        + np.square(np.linalg.norm(wf2.astype(np.float64), axis=0))
    )
    keep = np.sort(np.argsort(-importance)[: C // 2])
    oracle = _matmul_residual_transitive_model(
        wf1[:, keep], ws1[:, keep], wf2[:, keep], wout[keep, :]
    )

    rng_x = np.random.default_rng(93)
    x = rng_x.standard_normal((5, K)).astype(np.float32)
    z = rng_x.standard_normal((5, Kz)).astype(np.float32)
    (y,) = _run(pruned, {"X": x, "Z": z})
    (y_oracle,) = _run(oracle, {"X": x, "Z": z})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_matmul_residual_add_declines_on_fan_out_branch():
    K, C, Out = 8, 16, 4
    rng = np.random.default_rng(94)
    wf = rng.standard_normal((K, C)).astype(np.float32)
    ws = rng.standard_normal((K, C)).astype(np.float32)
    wnext = rng.standard_normal((C, C)).astype(np.float32)
    wout = rng.standard_normal((C, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          f = MatMul(X, WF)
          s = MatMul(X, WS)
          add1 = Add(f, s)
          r = Relu(add1)
          nxt = MatMul(r, WNEXT)
          add2 = Add(nxt, r)
          Y = MatMul(add2, WOUT)
        }}
        """,
        initializer=[
            _f32(wf, "WF"),
            _f32(ws, "WS"),
            _f32(wnext, "WNEXT"),
            _f32(wout, "WOUT"),
        ],
    )
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["WF"], wf)
    np.testing.assert_array_equal(inits["WS"], ws)
    np.testing.assert_array_equal(inits["WNEXT"], wnext)
    np.testing.assert_array_equal(inits["WOUT"], wout)


def test_cpp_structured_pruning_matmul_residual_add_declines_on_identity_shortcut():
    # y = MatMul2(Relu(Add(MatMul1(X), X))): the exact x = x + f(x)
    # transformer-residual identity-shortcut shape, no MatMul on the
    # shortcut path at all.
    C, Out = 8, 4
    rng = np.random.default_rng(95)
    w1 = rng.standard_normal((C, C)).astype(np.float32)
    w2 = rng.standard_normal((C, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{C}] X) => (float[batch,{Out}] Y)
        {{
          f = MatMul(X, W1)
          add1 = Add(f, X)
          r = Relu(add1)
          Y = MatMul(r, W2)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2")],
    )
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["W2"], w2)


def test_cpp_structured_pruning_matmul_residual_add_with_bias_hop_matches_oracle():
    # One branch has a per-channel bias Add (a separate node, not Gemm's own
    # bias input) between its producer and the residual merge -- exercises
    # the wider MatMul/Gemm-only hop set and the self-consistent-then-
    # revalidate check that tells this bias Add apart from an eligible
    # residual-merge Add.
    K, C, Out = 8, 16, 4
    rng = np.random.default_rng(96)
    w1 = rng.standard_normal((K, C)).astype(np.float32)
    bias = rng.standard_normal((C,)).astype(np.float32)
    ws = rng.standard_normal((K, C)).astype(np.float32)
    wout = rng.standard_normal((C, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          h = MatMul(X, W1)
          hb = Add(h, Bias)
          f = Relu(hb)
          s = MatMul(X, WS)
          addr = Add(f, s)
          r = Relu(addr)
          Y = MatMul(r, WOUT)
        }}
        """,
        initializer=[
            _f32(w1, "W1"),
            _f32(bias, "Bias"),
            _f32(ws, "WS"),
            _f32(wout, "WOUT"),
        ],
    )

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["Bias"].dims) == [C // 2]

    importance = np.sqrt(
        np.square(np.linalg.norm(w1.astype(np.float64), axis=0))
        + np.square(np.linalg.norm(ws.astype(np.float64), axis=0))
    )
    keep = np.sort(np.argsort(-importance)[: C // 2])

    rng_x = np.random.default_rng(97)
    x = rng_x.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    f = np.maximum(x @ w1[:, keep] + bias[keep], 0)
    s = x @ ws[:, keep]
    y_oracle = np.maximum(f + s, 0) @ wout[keep, :]
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_matmul_residual_add_transposed_gemm_producer_matches_oracle():
    # One branch is a Gemm with transB=1 (weight stored [N, K]) rather than
    # a plain MatMul's [K, N] -- a regression test for weight_transposed
    # being carried correctly through the backward walk.
    K, C, Out = 8, 16, 4
    rng = np.random.default_rng(98)
    w1t = rng.standard_normal((C, K)).astype(np.float32)  # [N, K] -- transB=1 layout
    ws = rng.standard_normal((K, C)).astype(np.float32)
    wout = rng.standard_normal((C, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          f = Gemm<transB = 1>(X, W1T)
          s = MatMul(X, WS)
          addr = Add(f, s)
          r = Relu(addr)
          Y = MatMul(r, WOUT)
        }}
        """,
        initializer=[_f32(w1t, "W1T"), _f32(ws, "WS"), _f32(wout, "WOUT")],
    )

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1T"].dims) == [C // 2, K]

    importance = np.sqrt(
        np.square(np.linalg.norm(w1t.astype(np.float64), axis=1))
        + np.square(np.linalg.norm(ws.astype(np.float64), axis=0))
    )
    keep = np.sort(np.argsort(-importance)[: C // 2])

    rng_x = np.random.default_rng(99)
    x = rng_x.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    f = x @ w1t[keep, :].T
    s = x @ ws[:, keep]
    y_oracle = np.maximum(f + s, 0) @ wout[keep, :]
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_matmul_residual_add_declines_on_gated_branch_with_no_projection():
    # A gated (SwiGLU-style) combine feeding directly into a residual Add,
    # with no output-projection MatMul between the Mul and the Add -- the
    # backward walk has no principled way to pick "the" one branch through a
    # Mul of two non-constant operands, so this falls straight through to
    # "fail" and the whole group is declined.
    K, C, Out = 8, 16, 4
    rng = np.random.default_rng(100)
    wg = rng.standard_normal((K, C)).astype(np.float32)
    wu = rng.standard_normal((K, C)).astype(np.float32)
    wp = rng.standard_normal((K, C)).astype(np.float32)
    wout = rng.standard_normal((C, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          gate = MatMul(X, WG)
          gate_act = Sigmoid(gate)
          up = MatMul(X, WU)
          h = Mul(gate_act, up)
          p = MatMul(X, WP)
          addr = Add(p, h)
          r = Relu(addr)
          Y = MatMul(r, WOUT)
        }}
        """,
        initializer=[
            _f32(wg, "WG"),
            _f32(wu, "WU"),
            _f32(wp, "WP"),
            _f32(wout, "WOUT"),
        ],
    )
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["WG"], wg)
    np.testing.assert_array_equal(inits["WU"], wu)
    np.testing.assert_array_equal(inits["WP"], wp)
    np.testing.assert_array_equal(inits["WOUT"], wout)


def test_cpp_structured_pruning_matmul_residual_add_declines_on_bare_gqa_shortcut():
    # A residual branch whose backward walk would need to cross a fused
    # self-attention op boundary to reach a real producer -- ctx (a
    # GroupQueryAttention node's own raw output) feeds directly into the
    # residual Add, with no output-projection MatMul in between. Neither
    # GroupQueryAttention nor its Q/K/V MatMul producers can be reached
    # through it.
    K, H, D, Out = 8, 4, 4, 6
    Nqkv = H * D
    rng = np.random.default_rng(101)
    wq = rng.standard_normal((K, Nqkv)).astype(np.float32)
    wk = rng.standard_normal((K, Nqkv)).astype(np.float32)
    wv = rng.standard_normal((K, Nqkv)).astype(np.float32)
    wp = rng.standard_normal((K, Nqkv)).astype(np.float32)
    wout = rng.standard_normal((Nqkv, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[2,5,{K}] X) => (float[2,5,{Out}] Y)
        {{
          q = MatMul(X, Wq)
          k = MatMul(X, Wk)
          v = MatMul(X, Wv)
          ctx, present_k, present_v = com.microsoft.GroupQueryAttention <num_heads={H}, kv_num_heads={H}> (q, k, v)
          p = MatMul(X, Wp)
          addr = Add(p, ctx)
          r = Relu(addr)
          Y = MatMul(r, Wout)
        }}
        """,
        initializer=[
            _f32(wq, "Wq"),
            _f32(wk, "Wk"),
            _f32(wv, "Wv"),
            _f32(wp, "Wp"),
            _f32(wout, "Wout"),
        ],
        opset=17,
    )
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["Wq"], wq)
    np.testing.assert_array_equal(inits["Wk"], wk)
    np.testing.assert_array_equal(inits["Wv"], wv)
    np.testing.assert_array_equal(inits["Wp"], wp)
    np.testing.assert_array_equal(inits["Wout"], wout)


# --- Fused SkipLayerNormalization residual merge ----------------------------


def _skip_layer_norm_residual_diamond_model(
    wf, ws, wout, gamma, beta=None, bias=None, simplified=False, epsilon=1e-5
):
    # y = SkipLayerNormalization(MatMul_f(X), MatMul_s(X), gamma, beta?,
    # bias?) -- the SkipLayerNormalization/SkipSimplifiedLayerNormalization
    # analogue of _matmul_residual_diamond_model: two entirely independent
    # MatMul producers merge via the fused node instead of a bare Add, and
    # must therefore still share one surviving channel-index set, feeding
    # one real consumer.
    K, C = wf.shape
    Out = wout.shape[1]
    op = "SkipSimplifiedLayerNormalization" if simplified else "SkipLayerNormalization"
    initializer = [
        _f32(wf, "WF"),
        _f32(ws, "WS"),
        _f32(wout, "WOUT"),
        _f32(gamma, "Gamma"),
    ]
    inputs = ["f", "s", "Gamma"]
    if not simplified:
        inputs.append("Beta" if beta is not None else "")
        if beta is not None:
            initializer.append(_f32(beta, "Beta"))
    if bias is not None:
        inputs.append("Bias")
        initializer.append(_f32(bias, "Bias"))
    while inputs and inputs[-1] == "":
        inputs.pop()
    ins = ", ".join(inputs)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          f = MatMul(X, WF)
          s = MatMul(X, WS)
          y = com.microsoft.{op} <epsilon={epsilon}> ({ins})
          Y = MatMul(y, WOUT)
        }}
        """,
        initializer=initializer,
        opset=17,
    )
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))
    return model


def _skip_layer_norm_keep(wf, ws, C):
    importance = np.sqrt(
        np.square(np.linalg.norm(wf.astype(np.float64), axis=0))
        + np.square(np.linalg.norm(ws.astype(np.float64), axis=0))
    )
    keep = np.sort(np.argsort(-importance)[: C // 2])
    assert np.any(keep < C // 2) and np.any(keep >= C // 2)
    return keep


def _conflicting_wf_ws(seed, K, C):
    rng = np.random.default_rng(seed)
    scale_f = np.where(np.arange(C) < C // 2, 3.0, 0.3).astype(np.float32)
    scale_s = np.where(np.arange(C) < C // 2, 0.3, 3.0).astype(np.float32)
    wf = rng.standard_normal((K, C)).astype(np.float32) * scale_f
    ws = rng.standard_normal((K, C)).astype(np.float32) * scale_s
    return rng, wf, ws


def test_cpp_structured_pruning_skip_layer_norm_residual_matches_oracle():
    K, C, Out = 8, 16, 4
    rng, wf, ws = _conflicting_wf_ws(110, K, C)
    wout = rng.standard_normal((C, Out)).astype(np.float32)
    gamma = rng.standard_normal((C,)).astype(np.float32)
    beta = rng.standard_normal((C,)).astype(np.float32)
    model = _skip_layer_norm_residual_diamond_model(wf, ws, wout, gamma, beta=beta)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["WF"].dims) == [K, C // 2]
    assert list(inits["WS"].dims) == [K, C // 2]
    assert list(inits["WOUT"].dims) == [C // 2, Out]
    assert list(inits["Gamma"].dims) == [C // 2]
    assert list(inits["Beta"].dims) == [C // 2]

    keep = _skip_layer_norm_keep(wf, ws, C)
    oracle = _skip_layer_norm_residual_diamond_model(
        wf[:, keep], ws[:, keep], wout[keep, :], gamma[keep], beta=beta[keep]
    )

    rng_x = np.random.default_rng(111)
    x = rng_x.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_skip_simplified_layer_norm_residual_matches_oracle():
    # SkipSimplifiedLayerNormalization -- the RMSNorm variant LLaMA-style
    # models use -- drops beta/mean-centering entirely.
    K, C, Out = 8, 16, 4
    rng, wf, ws = _conflicting_wf_ws(112, K, C)
    wout = rng.standard_normal((C, Out)).astype(np.float32)
    gamma = rng.standard_normal((C,)).astype(np.float32)
    model = _skip_layer_norm_residual_diamond_model(
        wf, ws, wout, gamma, simplified=True
    )

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert "Beta" not in inits

    keep = _skip_layer_norm_keep(wf, ws, C)
    oracle = _skip_layer_norm_residual_diamond_model(
        wf[:, keep], ws[:, keep], wout[keep, :], gamma[keep], simplified=True
    )

    rng_x = np.random.default_rng(113)
    x = rng_x.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_skip_layer_norm_residual_with_bias_matches_oracle():
    # bias present (and, deliberately, beta absent -- SkipLayerNorm's own
    # optional inputs are independent of each other): exercises the
    # bias-idx-shift in SkipLayerNormConstNames (bias lives at input index 4
    # when beta is declared) and confirms Bias is sliced alongside Gamma.
    K, C, Out = 8, 16, 4
    rng, wf, ws = _conflicting_wf_ws(114, K, C)
    wout = rng.standard_normal((C, Out)).astype(np.float32)
    gamma = rng.standard_normal((C,)).astype(np.float32)
    bias = rng.standard_normal((C,)).astype(np.float32)
    model = _skip_layer_norm_residual_diamond_model(wf, ws, wout, gamma, bias=bias)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["Bias"].dims) == [C // 2]

    keep = _skip_layer_norm_keep(wf, ws, C)
    oracle = _skip_layer_norm_residual_diamond_model(
        wf[:, keep], ws[:, keep], wout[keep, :], gamma[keep], bias=bias[keep]
    )

    rng_x = np.random.default_rng(115)
    x = rng_x.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_skip_layer_norm_residual_declines_on_nonconstant_beta():
    # Beta is a graph input, not a constant initializer -- gamma (also
    # required) is fine, but a present non-constant beta still means this
    # pass can't slice it, so the whole chain is declined and the model is
    # left byte-identical.
    K, C, Out = 8, 16, 4
    rng = np.random.default_rng(116)
    wf = rng.standard_normal((K, C)).astype(np.float32)
    ws = rng.standard_normal((K, C)).astype(np.float32)
    wout = rng.standard_normal((C, Out)).astype(np.float32)
    gamma = rng.standard_normal((C,)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X, float[{C}] Beta) => (float[batch,{Out}] Y)
        {{
          f = MatMul(X, WF)
          s = MatMul(X, WS)
          y = com.microsoft.SkipLayerNormalization <epsilon=1e-5> (f, s, Gamma, Beta)
          Y = MatMul(y, WOUT)
        }}
        """,
        initializer=[
            _f32(wf, "WF"),
            _f32(ws, "WS"),
            _f32(wout, "WOUT"),
            _f32(gamma, "Gamma"),
        ],
        opset=17,
    )
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    assert pruned.SerializeToString() == model.SerializeToString()


def test_cpp_structured_pruning_skip_layer_norm_residual_declines_on_consumed_mean_output():
    # The training-only mean output (index 1) is actually consumed here
    # (wired straight to a second graph output) -- onnxruntime's own CPU
    # kernel never actually populates it, and this pass has no basis for
    # whether pruning keeps it meaningful for whatever reads it, so the
    # whole chain is declined outright, leaving the model byte-identical.
    K, C, Out = 8, 16, 4
    rng = np.random.default_rng(117)
    wf = rng.standard_normal((K, C)).astype(np.float32)
    ws = rng.standard_normal((K, C)).astype(np.float32)
    wout = rng.standard_normal((C, Out)).astype(np.float32)
    gamma = rng.standard_normal((C,)).astype(np.float32)
    beta = rng.standard_normal((C,)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y, float[batch] MeanOut)
        {{
          f = MatMul(X, WF)
          s = MatMul(X, WS)
          y, MeanOut = com.microsoft.SkipLayerNormalization <epsilon=1e-5> (f, s, Gamma, Beta)
          Y = MatMul(y, WOUT)
        }}
        """,
        initializer=[
            _f32(wf, "WF"),
            _f32(ws, "WS"),
            _f32(wout, "WOUT"),
            _f32(gamma, "Gamma"),
            _f32(beta, "Beta"),
        ],
        opset=17,
    )
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    assert pruned.SerializeToString() == model.SerializeToString()


def test_cpp_structured_pruning_skip_layer_norm_residual_declines_on_consumed_sum_output():
    # The fourth output, input_skip_bias_sum (the raw, pre-normalization
    # f + s), is consumed directly here by a second graph output. Its shape
    # shrinks along with f/s, and this pass has no way to confirm the
    # outside consumer still expects the new, narrower width. Declined
    # outright, model left byte-identical.
    K, C, Out = 8, 16, 4
    rng = np.random.default_rng(119)
    wf = rng.standard_normal((K, C)).astype(np.float32)
    ws = rng.standard_normal((K, C)).astype(np.float32)
    wout = rng.standard_normal((C, Out)).astype(np.float32)
    gamma = rng.standard_normal((C,)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y, float[batch,{C}] SumOut)
        {{
          f = MatMul(X, WF)
          s = MatMul(X, WS)
          y, mean, inv_std, SumOut = com.microsoft.SkipSimplifiedLayerNormalization <epsilon=1e-5> (f, s, Gamma)
          Y = MatMul(y, WOUT)
        }}
        """,
        initializer=[
            _f32(wf, "WF"),
            _f32(ws, "WS"),
            _f32(wout, "WOUT"),
            _f32(gamma, "Gamma"),
        ],
        opset=17,
    )
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    assert pruned.SerializeToString() == model.SerializeToString()


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
