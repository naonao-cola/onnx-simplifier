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


# --- Split-merged (fused gate_up_proj) gated FFN chains ----------------------
#
# Real Phi-3/Phi-3.5 (onnxruntime-genai) exports use ONE gate_up_proj MatMul/
# Gemm whose 2*H-wide output is halved by a Split into a gate half and an up
# half, rather than two separate gate_proj/up_proj producers -- see
# onnxsim/structured_pruning_entry.cpp's own "Split-merged (fused
# gate_up_proj) gated FFN chains" section comment (and onnxsim/pruning.py's
# identically-named section, which this is ported from) for the full shape
# and co-selection semantics these tests exercise: "neuron" i of the
# intermediate dimension is represented by BOTH column i (gate) and column
# H + i (up) of the ONE combined weight tensor, and must always be kept or
# dropped together.


def _split_gate_up_keep_indices(w, H, keep_count):
    # The correct paired-importance ranking: combined (root-sum-square) norm
    # of the gate half (columns [0, H)) and the up half (columns [H, 2H)) of
    # the ONE combined weight `w` -- mirrors _combined_keep_indices's own
    # formula for the two-separate-producer case above.
    gate_half, up_half = w[:, :H], w[:, H:]
    importance = np.sqrt(
        np.square(np.linalg.norm(gate_half.T, axis=1))
        + np.square(np.linalg.norm(up_half.T, axis=1))
    )
    return np.sort(np.argsort(-importance)[:keep_count])


def _split_gate_up_mlp_model(
    K=8,
    H=16,
    Out=4,
    gate_activation="Sigmoid",
    seed=0,
    opset=21,
    split_attrs="axis=-1, num_outputs=2",
):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((K, 2 * H)).astype(np.float32)
    wd = rng.standard_normal((H, Out)).astype(np.float32)
    split_attr_text = f"<{split_attrs}>" if split_attrs else ""
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          combined = MatMul(X, W)
          gate, up = Split {split_attr_text} (combined)
          gate_act = {gate_activation}(gate)
          h = Mul(gate_act, up)
          Y = MatMul(h, Wd)
        }}
        """,
        initializer=[_f32(w, "W"), _f32(wd, "Wd")],
        opset=opset,
    )
    return model, w, wd


def test_cpp_structured_pruning_split_gated_ffn_matches_oracle():
    K, H, Out = 8, 16, 4
    model, w, wd = _split_gate_up_mlp_model(K=K, H=H, Out=Out)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W"].dims) == [K, H]  # 2 * (H // 2)
    assert list(inits["Wd"].dims) == [H // 2, Out]

    keep = _split_gate_up_keep_indices(w, H, H // 2)
    rng = np.random.default_rng(20)
    x = rng.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})

    gate = 1.0 / (1.0 + np.exp(-(x @ w[:, keep])))
    up = x @ w[:, H + keep]
    y_oracle = (gate * up) @ wd[keep, :]
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_split_gated_ffn_prunes_both_halves_of_one_tensor():
    # The real bug this pattern risks: only one of the two halves getting
    # sliced (or the two halves disagreeing on which columns survive) --
    # assert the SAME index set is dropped from both, out of the single
    # physical weight tensor.
    K, H, Out = 8, 20, 4
    model, w, wd = _split_gate_up_mlp_model(K=K, H=H, Out=Out, seed=1)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.3)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    keep = _split_gate_up_keep_indices(w, H, H - round(H * 0.3))

    np.testing.assert_array_equal(inits["W"][:, : len(keep)], w[:, keep])
    np.testing.assert_array_equal(inits["W"][:, len(keep) :], w[:, H + keep])
    np.testing.assert_array_equal(inits["Wd"], wd[keep, :])


def test_cpp_structured_pruning_split_gated_ffn_uses_combined_paired_importance():
    # Adversarial case: gate-half and up-half columns for the SAME neuron
    # index deliberately have very different magnitudes, constructed so that
    # ranking by EITHER half alone (a "sliced/ranked only one half" bug)
    # picks a DIFFERENT keep-set than the correct combined (root-sum-square)
    # ranking of the pair. K=2 with only row 0 non-zero makes each column's
    # own L2 norm exactly its row-0 value, so the desired per-half
    # magnitudes can be set directly and exactly.
    K, H, Out = 2, 5, 3
    gate_vals = np.array([9.0, 1.0, 6.0, 7.0, 0.5], dtype=np.float32)
    up_vals = np.array([1.0, 8.9, 6.0, 0.5, 6.9], dtype=np.float32)
    # combined (root-sum-square) importance per column:
    #   col0: sqrt(9.0^2+1.0^2) = 9.0554  (rank 1)
    #   col1: sqrt(1.0^2+8.9^2) = 8.9556  (rank 2)
    #   col2: sqrt(6.0^2+6.0^2) = 8.4853  (rank 3)
    #   col3: sqrt(7.0^2+0.5^2) = 7.0178  (rank 4)
    #   col4: sqrt(0.5^2+6.9^2) = 6.9181  (rank 5)
    # correct keep (top 3, combined) = [0, 1, 2]; gate-only top 3 would be
    # [0, 2, 3] and up-only top 3 would be [1, 2, 4] -- both wrong.
    w = np.zeros((K, 2 * H), dtype=np.float32)
    w[0, :H] = gate_vals
    w[0, H:] = up_vals
    rng = np.random.default_rng(2)
    wd = rng.standard_normal((H, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          combined = MatMul(X, W)
          gate, up = Split <axis=-1, num_outputs=2> (combined)
          gate_act = Sigmoid(gate)
          h = Mul(gate_act, up)
          Y = MatMul(h, Wd)
        }}
        """,
        initializer=[_f32(w, "W"), _f32(wd, "Wd")],
    )

    # H=5, sparsity=0.4 -> keep_count = 5 - round(5*0.4) = 3.
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.4)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    correct_keep = np.array([0, 1, 2])
    gate_only_keep = np.array([0, 2, 3])  # what a gate-only-ranked bug would pick
    up_only_keep = np.array([1, 2, 4])  # what an up-only-ranked bug would pick

    np.testing.assert_array_equal(inits["W"][:, :3], w[:, correct_keep])
    np.testing.assert_array_equal(inits["W"][:, 3:], w[:, H + correct_keep])
    assert not np.array_equal(inits["W"][0, :3], w[0, gate_only_keep])
    assert not np.array_equal(inits["W"][0, :3], w[0, up_only_keep])


def test_cpp_structured_pruning_split_gated_ffn_gelu_activation_matches_oracle():
    # GeGLU: same fused-gate_up_proj topology, a different (still-unary)
    # gate activation -- mirrors this file's own
    # test_cpp_structured_pruning_gelu_gated_ffn_matches_oracle above, but
    # for the single fused-producer shape. Uses Gelu's tanh approximation so
    # the oracle needs no scipy/erf. (Native ai.onnx Swish/HardSwish are not
    # exercised here: UnaryPassThroughOps() -- shared by every gated-chain
    # family in this port, not just this one -- does not yet recognize them,
    # a pre-existing gap outside this feature's own scope.)
    K, H, Out = 8, 16, 4
    model, w, wd = _split_gate_up_mlp_model(
        K=K, H=H, Out=Out, gate_activation='Gelu<approximate = "tanh">', seed=3
    )
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    keep = _split_gate_up_keep_indices(w, H, H // 2)
    rng = np.random.default_rng(30)
    x = rng.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})

    g = x @ w[:, keep]
    gate = 0.5 * g * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (g + 0.044715 * g**3)))
    up = x @ w[:, H + keep]
    y_oracle = (gate * up) @ wd[keep, :]
    np.testing.assert_allclose(y, y_oracle, rtol=1e-4, atol=1e-4)


def test_cpp_structured_pruning_split_gated_ffn_explicit_equal_split_input_matches_oracle():
    # opset 13+'s explicit `split` *input* (rather than the fully-automatic
    # even split) spelled out as literally [H, H] -- the same semantic
    # split, a different spelling; the pruned model's own Split input must
    # be rewritten to the new, still-even [h', h'].
    K, H, Out = 8, 16, 4
    rng = np.random.default_rng(4)
    w = rng.standard_normal((K, 2 * H)).astype(np.float32)
    wd = rng.standard_normal((H, Out)).astype(np.float32)
    sizes = onnx.numpy_helper.from_array(np.array([H, H], dtype=np.int64), name="Sizes")
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          combined = MatMul(X, W)
          gate, up = Split <axis=-1> (combined, Sizes)
          gate_act = Sigmoid(gate)
          h = Mul(gate_act, up)
          Y = MatMul(h, Wd)
        }}
        """,
        initializer=[_f32(w, "W"), _f32(wd, "Wd"), sizes],
    )

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    split_node = next(n for n in pruned.graph.node if n.op_type == "Split")
    sizes_init = next(
        t for t in pruned.graph.initializer if t.name == split_node.input[1]
    )
    assert list(onnx.numpy_helper.to_array(sizes_init)) == [H // 2, H // 2]

    keep = _split_gate_up_keep_indices(w, H, H // 2)
    x = rng.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    gate = 1.0 / (1.0 + np.exp(-(x @ w[:, keep])))
    up = x @ w[:, H + keep]
    y_oracle = (gate * up) @ wd[keep, :]
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_split_gated_ffn_native_swiglu_matches_oracle():
    # ONNX's native fused SwiGLU(a, b) = swish(a) * b (opset 28+), fed
    # directly by the Split's own two raw outputs -- mirrors
    # test_cpp_structured_pruning_native_swiglu_node_prunes_both_producers_together
    # above, but for the single fused-producer gate_up_proj shape. Not yet
    # supported by the installed onnx checker/onnxruntime in this
    # environment (opset 28 is still under development upstream), so this
    # verifies the graph surgery directly via tensor values rather than
    # onnx.checker/onnxruntime execution.
    K, H, Out = 8, 16, 4
    rng = np.random.default_rng(12)
    w = rng.standard_normal((K, 2 * H)).astype(np.float32)
    wd = rng.standard_normal((H, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          combined = MatMul(X, W)
          gate, up = Split <axis=-1, num_outputs=2> (combined)
          h = SwiGLU(gate, up)
          Y = MatMul(h, Wd)
        }}
        """,
        initializer=[_f32(w, "W"), _f32(wd, "Wd")],
        opset=28,
    )

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    keep = _split_gate_up_keep_indices(w, H, H // 2)

    np.testing.assert_array_equal(inits["W"][:, : H // 2], w[:, keep])
    np.testing.assert_array_equal(inits["W"][:, H // 2 :], w[:, H + keep])
    np.testing.assert_array_equal(inits["Wd"], wd[keep, :])


def test_cpp_structured_pruning_split_gated_ffn_gemm_producer_with_bias_matches_oracle():
    # The producer may also be a vanilla Gemm with a fused constant bias
    # (_match_producer/MatchProducer's own bias support) -- unlike a
    # *separate* MatMul -> Add(bias) hop before Split (declined, see
    # test_cpp_structured_pruning_split_gated_ffn_declines_bias_add_before_split
    # below), Gemm's own bias operand is a per-channel constant riding along
    # with the rest of the combined [K, 2H] weight, so it must be sliced at
    # the same two fixed offsets.
    K, H, Out = 8, 16, 4
    rng = np.random.default_rng(13)
    w = rng.standard_normal((K, 2 * H)).astype(np.float32)
    b = rng.standard_normal((2 * H,)).astype(np.float32)
    wd = rng.standard_normal((H, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          combined = Gemm(X, W, B)
          gate, up = Split <axis=-1, num_outputs=2> (combined)
          gate_act = Sigmoid(gate)
          h = Mul(gate_act, up)
          Y = MatMul(h, Wd)
        }}
        """,
        initializer=[_f32(w, "W"), _f32(b, "B"), _f32(wd, "Wd")],
    )

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W"].dims) == [K, H]
    assert list(inits["B"].dims) == [H]

    keep = _split_gate_up_keep_indices(w, H, H // 2)
    x = rng.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    combined = (
        x @ w[:, np.concatenate([keep, H + keep])] + b[np.concatenate([keep, H + keep])]
    )
    gate = 1.0 / (1.0 + np.exp(-combined[:, : H // 2]))
    up = combined[:, H // 2 :]
    y_oracle = (gate * up) @ wd[keep, :]
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_split_gated_ffn_declines_unequal_explicit_split():
    K, H = 8, 16
    rng = np.random.default_rng(5)
    w = rng.standard_normal((K, 2 * H)).astype(np.float32)
    sizes = onnx.numpy_helper.from_array(
        np.array([H + 2, H - 2], dtype=np.int64), name="Sizes"
    )
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{H + 2}] Gate, float[batch,{H - 2}] Up)
        {{
          combined = MatMul(X, W)
          Gate, Up = Split <axis=-1> (combined, Sizes)
        }}
        """,
        initializer=[_f32(w, "W"), sizes],
    )
    before = model.SerializeToString()
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    assert pruned.SerializeToString() == before


def test_cpp_structured_pruning_split_gated_ffn_declines_when_axis_defaults_to_zero():
    # Split's own schema default axis is 0, unlike Concat's *required*
    # attribute -- an un-annotated Split here would target the batch axis,
    # not the channel axis, and must be declined, not assumed.
    K, H, Out = 8, 16, 4
    model, w, wd = _split_gate_up_mlp_model(
        K=K, H=H, Out=Out, seed=10, split_attrs="num_outputs=2"
    )
    before = model.SerializeToString()
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    assert pruned.SerializeToString() == before


def test_cpp_structured_pruning_split_gated_ffn_declines_bias_add_before_split():
    # A separate MatMul -> Add(bias) -> Split, rather than the producer's
    # raw output feeding Split directly -- out of scope for this first pass
    # (see this section's own comment in structured_pruning_entry.cpp).
    K, H, Out = 8, 16, 4
    rng = np.random.default_rng(8)
    w = rng.standard_normal((K, 2 * H)).astype(np.float32)
    bias = rng.standard_normal((2 * H,)).astype(np.float32)
    wd = rng.standard_normal((H, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          combined = MatMul(X, W)
          combined_b = Add(combined, Bias)
          gate, up = Split <axis=-1, num_outputs=2> (combined_b)
          gate_act = Sigmoid(gate)
          h = Mul(gate_act, up)
          Y = MatMul(h, Wd)
        }}
        """,
        initializer=[_f32(w, "W"), _f32(bias, "Bias"), _f32(wd, "Wd")],
    )
    before = model.SerializeToString()
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    assert pruned.SerializeToString() == before


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


def test_cpp_structured_pruning_conv_residual_add_matches_oracle_on_fan_out_branch():
    # A realistic multi-block residual stage's interior boundary: `r`
    # (add1's own post-block tensor) is read twice -- once by the next
    # block's own first Conv, once unchanged as that next block's own Add
    # shortcut operand. The backward walkers no longer reject this
    # mid-walk; instead the "extra" reader (`nxt`) is resolved as its own
    # independent forward branch once the group's shared keep set is
    # established (see ResolveConvFanoutBranches), so `nxt` ends up
    # pruned on *both* axes of WNEXT: its own output channels (it's also a
    # leaf producer of add2's own merge) and, via this fan-out branch, its
    # input channels (it independently reads the group's own shared spine).
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
    onnx.checker.check_model(pruned)

    importance = np.sqrt(
        np.square(np.linalg.norm(w_f.reshape(C, -1).astype(np.float64), axis=1))
        + np.square(np.linalg.norm(w_s.reshape(C, -1).astype(np.float64), axis=1))
        + np.square(np.linalg.norm(w_next.reshape(C, -1).astype(np.float64), axis=1))
    )
    keep = np.sort(np.argsort(-importance)[: C // 2])
    oracle = _model(
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
            _f32(w_f[keep], "WF"),
            _f32(w_s[keep], "WS"),
            _f32(w_next[keep][:, keep], "WNEXT"),
            _f32(w_out[:, keep], "WOUT"),
        ],
    )

    rng_x = np.random.default_rng(85)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


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


def test_cpp_structured_pruning_conv_residual_add_matches_oracle_with_grouped_conv_consumer():
    # Two independent Conv branches merge via Add, and the downstream
    # consumer is a general grouped Conv (group=2) -- now matched (see
    # this module's own docstring for why per-`group`-block top-k is a
    # provably-safe generalization once every producer/branch agrees on
    # the same `group` count), one independent top-k per `group`-sized
    # block of the combined-importance vector.
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
    onnx.checker.check_model(pruned)

    importance = np.sqrt(
        np.square(np.linalg.norm(w_f.reshape(C, -1).astype(np.float64), axis=1))
        + np.square(np.linalg.norm(w_s.reshape(C, -1).astype(np.float64), axis=1))
    )
    block = C // group
    per_group_keep = block // 2
    keep = np.concatenate(
        [
            np.sort(
                np.argsort(-importance[gi * block : (gi + 1) * block])[:per_group_keep]
            )
            + gi * block
            for gi in range(group)
        ]
    )

    out_per_group = Cout // group
    out_parts = []
    for gi in range(group):
        local_keep = keep[(keep >= gi * block) & (keep < (gi + 1) * block)] - gi * block
        out_parts.append(
            w_out[gi * out_per_group : (gi + 1) * out_per_group, local_keep]
        )
    w_out_oracle = np.concatenate(out_parts, axis=0)

    oracle = _model(
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
        initializer=[
            _f32(w_f[keep], "WF"),
            _f32(w_s[keep], "WS"),
            _f32(w_out_oracle, "WOUT"),
        ],
    )

    rng_x = np.random.default_rng(90)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


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


def test_cpp_structured_pruning_matmul_residual_add_matches_oracle_on_fan_out_branch():
    # The MatMul/Gemm analogue of the Conv fan-out test above: `r` is read
    # both by `nxt` and, unchanged, by `add2` -- `nxt` ends up pruned on
    # both axes of WNEXT (its own output columns, as a leaf producer of
    # add2's own merge, and its own reduction rows, as an independent
    # fan-out branch reading the group's shared spine).
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
    onnx.checker.check_model(pruned)

    importance = np.sqrt(
        np.square(np.linalg.norm(wf.astype(np.float64), axis=0))
        + np.square(np.linalg.norm(ws.astype(np.float64), axis=0))
        + np.square(np.linalg.norm(wnext.astype(np.float64), axis=0))
    )
    keep = np.sort(np.argsort(-importance)[: C // 2])
    oracle = _model(
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
            _f32(wf[:, keep], "WF"),
            _f32(ws[:, keep], "WS"),
            _f32(wnext[keep][:, keep], "WNEXT"),
            _f32(wout[keep, :], "WOUT"),
        ],
    )

    rng_x = np.random.default_rng(95)
    x = rng_x.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


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


def test_cpp_structured_pruning_matmul_residual_add_matches_oracle_on_gated_branch_with_no_projection():
    # A gated (SwiGLU-style) combine feeding directly into a residual Add,
    # with no output-projection MatMul between the Mul and the Add -- now
    # resolved the same way a gated pair outside a residual chain already
    # is: both `gate`'s and `up`'s own producers join the group's shared
    # leaf-producer set (see WalkMatmulProducerBackward's own "gated"
    # outcome), ranked and pruned together with `p`.
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
    onnx.checker.check_model(pruned)

    importance = np.sqrt(
        np.square(np.linalg.norm(wg.astype(np.float64), axis=0))
        + np.square(np.linalg.norm(wu.astype(np.float64), axis=0))
        + np.square(np.linalg.norm(wp.astype(np.float64), axis=0))
    )
    keep = np.sort(np.argsort(-importance)[: C // 2])
    oracle = _model(
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
            _f32(wg[:, keep], "WG"),
            _f32(wu[:, keep], "WU"),
            _f32(wp[:, keep], "WP"),
            _f32(wout[keep, :], "WOUT"),
        ],
    )

    rng_x = np.random.default_rng(101)
    x = rng_x.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


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


# --- PRelu/Clip channel pass-through hops ------------------------------------
#
# Two "channel pass-through hop" features ported from pruning.py's own
# reference: a PRelu whose `slope` is either a scalar/single shared
# parameter (left untouched) or a genuine per-channel constant (sliced by
# the chain's own `keep` set, like a depthwise Conv hop's own weight), and a
# Clip (the `torch.nn.ReLU6` shape MobileNet/EfficientNet-Lite exports)
# crossed transparently whenever its `min`/`max` are each either omitted or
# a constant scalar. See _match_prelu_pass_through(_self,_matmul,
# _matmul_self) and _match_clip_channel_pass_through in pruning.py.


def test_cpp_structured_pruning_prelu_per_channel_pass_through_conv_matches_oracle():
    Cin, C1, C2 = 3, 16, 8
    rng = np.random.default_rng(200)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    b1 = rng.standard_normal((C1,)).astype(np.float32)
    slope = rng.uniform(0.05, 0.3, size=(C1, 1, 1)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)

    def _mk(w1, b1, slope, w2):
        return _model(
            f"""
            g (float[N,{Cin},10,10] X) => (float[N,{C2},6,6] Y)
            {{
              h = Conv<kernel_shape=[3,3]>(X, W1, B1)
              a = PRelu(h, Slope)
              Y = Conv<kernel_shape=[3,3]>(a, W2)
            }}
            """,
            initializer=[
                _f32(w1, "W1"),
                _f32(b1, "B1"),
                _f32(slope, "Slope"),
                _f32(w2, "W2"),
            ],
        )

    model = _mk(w1, b1, slope, w2)
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    assert inits["Slope"].shape == (C1 // 2, 1, 1)
    # The per-channel-slope hop reuses ConvPassThrough (same as a depthwise
    # Conv hop), but PRelu has no `group` attribute of its own -- confirm the
    # port doesn't erroneously bolt one on.
    prelu_node = next(n for n in pruned.graph.node if n.op_type == "PRelu")
    assert len(prelu_node.attribute) == 0

    keep = _oracle_keep_indices_conv(w1, C1 // 2)
    oracle = _mk(w1[keep], b1[keep], slope[keep], w2[:, keep])

    rng_x = np.random.default_rng(201)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_prelu_scalar_slope_left_untouched_on_conv_chain():
    Cin, C1, C2 = 3, 16, 8
    rng = np.random.default_rng(202)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    slope = np.array([0.2], dtype=np.float32)  # single shared parameter.
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)

    def _mk(w1, slope, w2):
        return _model(
            f"""
            g (float[N,{Cin},10,10] X) => (float[N,{C2},6,6] Y)
            {{
              h = Conv<kernel_shape=[3,3]>(X, W1)
              a = PRelu(h, Slope)
              Y = Conv<kernel_shape=[3,3]>(a, W2)
            }}
            """,
            initializer=[_f32(w1, "W1"), _f32(slope, "Slope"), _f32(w2, "W2")],
        )

    model = _mk(w1, slope, w2)
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    # Scalar slope: same value multiplies every channel, so it's left
    # completely untouched -- no "nothing of its own to slice" hop needed.
    np.testing.assert_array_equal(inits["Slope"], slope)

    keep = _oracle_keep_indices_conv(w1, C1 // 2)
    oracle = _mk(w1[keep], slope, w2[:, keep])

    rng_x = np.random.default_rng(203)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_clip_relu6_pass_through_conv_matches_oracle():
    Cin, C1, C2 = 3, 16, 8
    rng = np.random.default_rng(204)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    min_c = np.array(0.0, dtype=np.float32)
    max_c = np.array(6.0, dtype=np.float32)

    def _mk(w1, w2):
        return _model(
            f"""
            g (float[N,{Cin},10,10] X) => (float[N,{C2},6,6] Y)
            {{
              h = Conv<kernel_shape=[3,3]>(X, W1)
              a = Clip(h, Min, Max)
              Y = Conv<kernel_shape=[3,3]>(a, W2)
            }}
            """,
            initializer=[
                _f32(w1, "W1"),
                _f32(min_c, "Min"),
                _f32(max_c, "Max"),
                _f32(w2, "W2"),
            ],
        )

    model = _mk(w1, w2)
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1"].dims) == [C1 // 2, Cin, 3, 3]
    assert list(inits["W2"].dims) == [C2, C1 // 2, 3, 3]

    keep = _oracle_keep_indices_conv(w1, C1 // 2)
    oracle = _mk(w1[keep], w2[:, keep])

    rng_x = np.random.default_rng(205)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_prelu_bare_rank1_slope_declines_on_conv_chain():
    # A bare [C] slope is deliberately *not* treated as per-channel on a Conv
    # chain (unlike a MatMul/Gemm chain's own last-axis convention): ONNX's
    # unidirectional broadcasting would align it against the *trailing* (W)
    # axis, not axis 1 -- declined, never guessed at.
    Cin, C1, C2 = 3, 8, 4
    rng = np.random.default_rng(206)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    slope = rng.uniform(0.05, 0.3, size=(C1,)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    model = _model(
        f"""
        g (float[N,{Cin},10,10] X) => (float[N,{C2},6,6] Y)
        {{
          h = Conv<kernel_shape=[3,3]>(X, W1)
          a = PRelu(h, Slope)
          Y = Conv<kernel_shape=[3,3]>(a, W2)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(slope, "Slope"), _f32(w2, "W2")],
    )

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["Slope"], slope)
    np.testing.assert_array_equal(inits["W2"], w2)


def test_cpp_structured_pruning_prelu_nonconstant_slope_declines():
    Cin, C1, C2 = 3, 8, 4
    rng = np.random.default_rng(207)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    model = _model(
        f"""
        g (float[N,{Cin},10,10] X, float[{C1},1,1] Slope) => (float[N,{C2},6,6] Y)
        {{
          h = Conv<kernel_shape=[3,3]>(X, W1)
          a = PRelu(h, Slope)
          Y = Conv<kernel_shape=[3,3]>(a, W2)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2")],
    )

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["W2"], w2)


def test_cpp_structured_pruning_clip_nonconstant_bound_declines():
    Cin, C1, C2 = 3, 8, 4
    rng = np.random.default_rng(208)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    min_c = np.array(0.0, dtype=np.float32)
    model = _model(
        f"""
        g (float[N,{Cin},10,10] X, float Max) => (float[N,{C2},6,6] Y)
        {{
          h = Conv<kernel_shape=[3,3]>(X, W1)
          a = Clip(h, Min, Max)
          Y = Conv<kernel_shape=[3,3]>(a, W2)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(min_c, "Min"), _f32(w2, "W2")],
    )

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["W2"], w2)


def test_cpp_structured_pruning_prelu_per_channel_pass_through_matmul_matches_oracle():
    K, H, Out = 8, 32, 4
    rng = np.random.default_rng(209)
    w1 = rng.standard_normal((K, H)).astype(np.float32)
    b1 = rng.standard_normal((H,)).astype(np.float32)
    slope = rng.uniform(0.05, 0.3, size=(H,)).astype(np.float32)
    w2 = rng.standard_normal((H, Out)).astype(np.float32)

    def _mk(w1, b1, slope, w2):
        return _model(
            f"""
            g (float[batch,{K}] X) => (float[batch,{Out}] Y)
            {{
              h = Gemm(X, W1, B1)
              a = PRelu(h, Slope)
              Y = MatMul(a, W2)
            }}
            """,
            initializer=[
                _f32(w1, "W1"),
                _f32(b1, "B1"),
                _f32(slope, "Slope"),
                _f32(w2, "W2"),
            ],
        )

    model = _mk(w1, b1, slope, w2)
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    assert inits["Slope"].shape == (H // 2,)

    keep = _oracle_keep_indices(w1, H // 2)
    oracle = _mk(w1[:, keep], b1[keep], slope[keep], w2[keep, :])

    rng_x = np.random.default_rng(210)
    x = rng_x.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_prelu_scalar_slope_left_untouched_on_matmul_chain():
    K, H, Out = 8, 32, 4
    rng = np.random.default_rng(211)
    w1 = rng.standard_normal((K, H)).astype(np.float32)
    # Single shared parameter, shape [1] -- mirrors _match_prelu_pass_through*'s
    # own `if not dims: return None` bar, which (like the Conv-chain matcher)
    # declines a true rank-0 slope; [1]/[1,1,1] is the shape real exporters
    # (and this matcher) actually treat as "scalar".
    slope = np.array([0.25], dtype=np.float32)
    w2 = rng.standard_normal((H, Out)).astype(np.float32)

    def _mk(w1, slope, w2):
        return _model(
            f"""
            g (float[batch,{K}] X) => (float[batch,{Out}] Y)
            {{
              h = MatMul(X, W1)
              a = PRelu(h, Slope)
              Y = MatMul(a, W2)
            }}
            """,
            initializer=[_f32(w1, "W1"), _f32(slope, "Slope"), _f32(w2, "W2")],
        )

    model = _mk(w1, slope, w2)
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["Slope"], slope)

    keep = _oracle_keep_indices(w1, H // 2)
    oracle = _mk(w1[:, keep], slope, w2[keep, :])

    rng_x = np.random.default_rng(212)
    x = rng_x.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_clip_relu6_pass_through_matmul_matches_oracle():
    K, H, Out = 8, 32, 4
    rng = np.random.default_rng(213)
    w1 = rng.standard_normal((K, H)).astype(np.float32)
    w2 = rng.standard_normal((H, Out)).astype(np.float32)
    min_c = np.array(0.0, dtype=np.float32)
    max_c = np.array([6.0], dtype=np.float32)  # single-element shape [1].

    def _mk(w1, w2):
        return _model(
            f"""
            g (float[batch,{K}] X) => (float[batch,{Out}] Y)
            {{
              h = MatMul(X, W1)
              a = Clip(h, Min, Max)
              Y = MatMul(a, W2)
            }}
            """,
            initializer=[
                _f32(w1, "W1"),
                _f32(min_c, "Min"),
                _f32(max_c, "Max"),
                _f32(w2, "W2"),
            ],
        )

    model = _mk(w1, w2)
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1"].dims) == [K, H // 2]
    assert list(inits["W2"].dims) == [H // 2, Out]

    keep = _oracle_keep_indices(w1, H // 2)
    oracle = _mk(w1[:, keep], w2[keep, :])

    rng_x = np.random.default_rng(214)
    x = rng_x.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_conv_residual_prelu_pass_through_hop_matches_oracle():
    # A PRelu per-channel hop crossed by the *backward* walk
    # (WalkConvProducerBackward/MatchPreluPassThroughSelf), not just the
    # forward one -- exercises the residual-chain insertion point and the
    # ApplyChains "group" attribute guard (PRelu must not get one).
    Cin, C, Cout = 3, 16, 8
    rng = np.random.default_rng(215)
    w_f = rng.standard_normal((C, Cin, 3, 3)).astype(np.float32)
    slope = rng.uniform(0.05, 0.3, size=(C, 1, 1)).astype(np.float32)
    w_s = rng.standard_normal((C, Cin, 3, 3)).astype(np.float32)
    w_out = rng.standard_normal((Cout, C, 3, 3)).astype(np.float32)

    def _mk(w_f, slope, w_s, w_out):
        return _model(
            f"""
            g (float[N,{Cin},10,10] X) => (float[N,{Cout},6,6] Y)
            {{
              f0 = Conv<kernel_shape=[3,3]>(X, WF)
              f = PRelu(f0, Slope)
              s = Conv<kernel_shape=[3,3]>(X, WS)
              addr = Add(f, s)
              r = Relu(addr)
              Y = Conv<kernel_shape=[3,3]>(r, WOUT)
            }}
            """,
            initializer=[
                _f32(w_f, "WF"),
                _f32(slope, "Slope"),
                _f32(w_s, "WS"),
                _f32(w_out, "WOUT"),
            ],
        )

    model = _mk(w_f, slope, w_s, w_out)
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    assert inits["Slope"].shape == (C // 2, 1, 1)
    prelu_node = next(n for n in pruned.graph.node if n.op_type == "PRelu")
    assert len(prelu_node.attribute) == 0

    importance = np.sqrt(
        np.square(np.linalg.norm(w_f.reshape(C, -1).astype(np.float64), axis=1))
        + np.square(np.linalg.norm(w_s.reshape(C, -1).astype(np.float64), axis=1))
    )
    keep = np.sort(np.argsort(-importance)[: C // 2])
    oracle = _mk(w_f[keep], slope[keep], w_s[keep], w_out[:, keep])

    rng_x = np.random.default_rng(216)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_matmul_residual_clip_pass_through_hop_matches_oracle():
    # A Clip crossed by the *backward* MatMul/Gemm walk
    # (WalkMatmulProducerBackward) -- exercises that insertion point too.
    K, C, Out = 8, 16, 4
    rng = np.random.default_rng(217)
    w_f = rng.standard_normal((K, C)).astype(np.float32)
    w_s = rng.standard_normal((K, C)).astype(np.float32)
    w_out = rng.standard_normal((C, Out)).astype(np.float32)
    min_c = np.array(0.0, dtype=np.float32)
    max_c = np.array(6.0, dtype=np.float32)

    def _mk(w_f, w_s, w_out):
        return _model(
            f"""
            g (float[batch,{K}] X) => (float[batch,{Out}] Y)
            {{
              f0 = MatMul(X, WF)
              f = Clip(f0, Min, Max)
              s = MatMul(X, WS)
              addr = Add(f, s)
              r = Relu(addr)
              Y = MatMul(r, WOUT)
            }}
            """,
            initializer=[
                _f32(w_f, "WF"),
                _f32(min_c, "Min"),
                _f32(max_c, "Max"),
                _f32(w_s, "WS"),
                _f32(w_out, "WOUT"),
            ],
        )

    model = _mk(w_f, w_s, w_out)
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    importance = np.sqrt(
        np.square(np.linalg.norm(w_f.T.astype(np.float64), axis=1))
        + np.square(np.linalg.norm(w_s.T.astype(np.float64), axis=1))
    )
    keep = np.sort(np.argsort(-importance)[: C // 2])
    oracle = _mk(w_f[:, keep], w_s[:, keep], w_out[keep, :])

    rng_x = np.random.default_rng(218)
    x = rng_x.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


# --- Conv chain: GroupNormalization pass-through hop -------------------------
#
# `Conv -> GroupNormalization -> Conv`: mirrors test_pruning.py's own
# `_group_norm_conv_pair_model` and its group-norm-pass-through test coverage.
# Unlike PRelu/Clip, a mid-chain GroupNorm hop constrains `ChainGroup()`'s own
# per-block `keep` selection to its own `num_groups` (see GroupNormPassThrough
# and ChainGroup in structured_pruning_entry.cpp), so the oracle here uses
# `_oracle_keep_indices_conv_grouped`, not the plain `_oracle_keep_indices_conv`
# every other Conv-chain hop test uses.


def _group_norm_conv_pair_model(
    w1, w2, gn_scale, gn_bias, num_groups, group1=1, b1=None
):
    Cin, C2 = w1.shape[1] * group1, w2.shape[0]
    initializer = [
        _f32(w1, "W1"),
        _f32(w2, "W2"),
        _f32(gn_scale, "GNScale"),
        _f32(gn_bias, "GNBias"),
    ]
    g1 = f", group={group1}" if group1 != 1 else ""
    if b1 is not None:
        conv1 = f"h = Conv<kernel_shape=[3,3]{g1}>(X, W1, B1)"
        initializer.append(_f32(b1, "B1"))
    else:
        conv1 = f"h = Conv<kernel_shape=[3,3]{g1}>(X, W1)"
    return _model(
        f"""
        g (float[N,{Cin},10,10] X) => (float[N,{C2},6,6] Y)
        {{
          {conv1}
          gn = GroupNormalization<num_groups={num_groups}, epsilon=1e-05>(h, GNScale, GNBias)
          Y = Conv<kernel_shape=[3,3]>(gn, W2)
        }}
        """,
        initializer=initializer,
    )


def test_cpp_structured_pruning_group_norm_pass_through_matches_oracle():
    Cin, C1, C2, num_groups = 3, 16, 8, 4
    rng = np.random.default_rng(220)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    b1 = rng.standard_normal((C1,)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    gn_scale = rng.standard_normal((C1,)).astype(np.float32)
    gn_bias = rng.standard_normal((C1,)).astype(np.float32)
    model = _group_norm_conv_pair_model(w1, w2, gn_scale, gn_bias, num_groups, b1=b1)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    # `num_groups` (a node attribute, not a tensor) is unchanged; the
    # surviving channel count must still divide it evenly.
    assert inits["W1"].shape[0] % num_groups == 0
    assert inits["W1"].shape[0] < C1  # actually pruned, not a no-op
    assert inits["GNScale"].shape == inits["GNBias"].shape == (C1 // 2,)
    gn_node = next(n for n in pruned.graph.node if n.op_type == "GroupNormalization")
    assert next(a.i for a in gn_node.attribute if a.name == "num_groups") == num_groups

    keep = _oracle_keep_indices_conv_grouped(w1, num_groups, 0.5)
    oracle = _group_norm_conv_pair_model(
        w1[keep], w2[:, keep], gn_scale[keep], gn_bias[keep], num_groups, b1=b1[keep]
    )

    rng_x = np.random.default_rng(221)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_group_norm_num_groups_mismatch_with_grouped_conv_declines():
    # A mid-chain GroupNorm hop's own `num_groups` disagreeing with a
    # same-chain grouped Conv producer's own `group` -- the two partitions'
    # block boundaries wouldn't generally align, so the whole chain is
    # declined outright, never guessed at, the same bar a plain
    # producer_group != consumer_group mismatch already gets.
    Cin, C1, C2, group1, num_groups = 4, 8, 8, 2, 4
    rng = np.random.default_rng(222)
    w1 = rng.standard_normal((C1, Cin // group1, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    gn_scale = rng.standard_normal((C1,)).astype(np.float32)
    gn_bias = rng.standard_normal((C1,)).astype(np.float32)
    model = _group_norm_conv_pair_model(
        w1, w2, gn_scale, gn_bias, num_groups, group1=group1
    )

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["W2"], w2)
    np.testing.assert_array_equal(inits["GNScale"], gn_scale)
    np.testing.assert_array_equal(inits["GNBias"], gn_bias)


def test_cpp_structured_pruning_group_norm_tied_scale_bias_declines():
    # `scale`/`bias` naming the *same* tensor -- double-slicing it in
    # ApplyChains's own per-hop loop would corrupt it, so this is declined
    # outright, mirroring pruning.py's own tied-name bar
    # (_match_group_norm_pass_through).
    Cin, C1, C2, num_groups = 3, 8, 4, 2
    rng = np.random.default_rng(223)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    tied = rng.standard_normal((C1,)).astype(np.float32)
    model = _model(
        f"""
        g (float[N,{Cin},10,10] X) => (float[N,{C2},6,6] Y)
        {{
          h = Conv<kernel_shape=[3,3]>(X, W1)
          gn = GroupNormalization<num_groups={num_groups}, epsilon=1e-05>(h, Tied, Tied)
          Y = Conv<kernel_shape=[3,3]>(gn, W2)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2"), _f32(tied, "Tied")],
    )
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["W2"], w2)
    np.testing.assert_array_equal(inits["Tied"], tied)


# --- Conv chain: Resize channel-safe pass-through hop -------------------------
#
# `Conv -> Resize(scales, spatial-only) -> Conv`: the U-Net/diffusion-model-
# decoder-style upsampling shape. Mirrors test_pruning.py's own
# `_resize_conv_pair_model` and its Resize-pass-through test coverage.


def _resize_conv_pair_model(w1, w2, scales, b1=None, spatial=8, out_spatial_hw=None):
    Cin, C2 = w1.shape[1], w2.shape[0]
    initializer = [
        _f32(w1, "W1"),
        _f32(w2, "W2"),
        onnx.numpy_helper.from_array(np.asarray(scales, dtype=np.float32), "Scales"),
    ]
    if b1 is not None:
        conv1 = "h = Conv<kernel_shape=[3,3]>(X, W1, B1)"
        initializer.append(_f32(b1, "B1"))
    else:
        conv1 = "h = Conv<kernel_shape=[3,3]>(X, W1)"
    after_conv1 = spatial - 2
    if out_spatial_hw is None:
        mid_h = round(after_conv1 * scales[2])
        out_spatial_hw = mid_h - 2
    return _model(
        f"""
        g (float[N,{Cin},{spatial},{spatial}] X) => (float[N,{C2},{out_spatial_hw},{out_spatial_hw}] Y)
        {{
          {conv1}
          p = Resize<mode="nearest">(h, , Scales)
          Y = Conv<kernel_shape=[3,3]>(p, W2)
        }}
        """,
        initializer=initializer,
    )


def test_cpp_structured_pruning_resize_channel_safe_pass_through_matches_oracle():
    Cin, C1, C2 = 3, 16, 8
    rng = np.random.default_rng(224)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    b1 = rng.standard_normal((C1,)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    scales = [1.0, 1.0, 2.0, 2.0]
    model = _resize_conv_pair_model(w1, w2, scales, b1=b1)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1"].dims) == [C1 // 2, Cin, 3, 3]
    assert list(inits["W2"].dims) == [C2, C1 // 2, 3, 3]

    keep = _oracle_keep_indices_conv(w1, C1 // 2)
    oracle = _resize_conv_pair_model(w1[keep], w2[:, keep], scales, b1=b1[keep])

    rng_x = np.random.default_rng(225)
    x = rng_x.standard_normal((2, Cin, 8, 8)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_resize_channel_affecting_declines():
    # scales[1] (the channel axis) == 2.0 -- genuinely resizes the channel
    # axis itself, so it must be declined outright, never guessed at.
    Cin, C1, C2 = 3, 8, 4
    rng = np.random.default_rng(226)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    scales = [1.0, 2.0, 1.0, 1.0]
    w2 = rng.standard_normal((C2, C1 * 2, 3, 3)).astype(np.float32)
    model = _resize_conv_pair_model(w1, w2, scales, out_spatial_hw=4)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["W2"], w2)


def test_cpp_structured_pruning_resize_dynamic_scales_declines():
    # `scales` computed at runtime (Shape -> Cast) rather than a constant
    # initializer -- this pass cannot know which axis is affected without
    # evaluating the graph, so it must decline outright, never guessed at.
    Cin, C1, C2 = 3, 8, 4
    rng = np.random.default_rng(227)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    model = _model(
        f"""
        g (float[N,{Cin},8,8] X) => (float[N,{C2},4,4] Y)
        {{
          h = Conv<kernel_shape=[3,3]>(X, W1)
          shp = Shape(h)
          scales_dyn = Cast<to=1>(shp)
          p = Resize<mode="nearest">(h, , scales_dyn)
          Y = Conv<kernel_shape=[3,3]>(p, W2)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2")],
    )
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["W2"], w2)


def test_cpp_structured_pruning_conv_residual_resize_pass_through_hop_matches_oracle():
    # A channel-safe Resize crossed by the *backward* walk
    # (WalkConvProducerBackward) -- exercises the residual-chain insertion
    # point, not just the forward one.
    Cin, C, Cout = 3, 16, 8
    rng = np.random.default_rng(228)
    w_f = rng.standard_normal((C, Cin, 3, 3)).astype(np.float32)
    scales = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    w_s = rng.standard_normal((C, Cin, 3, 3)).astype(np.float32)
    w_out = rng.standard_normal((Cout, C, 3, 3)).astype(np.float32)

    def _mk(w_f, w_s, w_out):
        return _model(
            f"""
            g (float[N,{Cin},10,10] X) => (float[N,{Cout},6,6] Y)
            {{
              f0 = Conv<kernel_shape=[3,3]>(X, WF)
              f = Resize<mode="nearest">(f0, , Scales)
              s = Conv<kernel_shape=[3,3]>(X, WS)
              addr = Add(f, s)
              r = Relu(addr)
              Y = Conv<kernel_shape=[3,3]>(r, WOUT)
            }}
            """,
            initializer=[
                _f32(w_f, "WF"),
                onnx.numpy_helper.from_array(scales, "Scales"),
                _f32(w_s, "WS"),
                _f32(w_out, "WOUT"),
            ],
        )

    model = _mk(w_f, w_s, w_out)
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    importance = np.sqrt(
        np.square(np.linalg.norm(w_f.reshape(C, -1).astype(np.float64), axis=1))
        + np.square(np.linalg.norm(w_s.reshape(C, -1).astype(np.float64), axis=1))
    )
    keep = np.sort(np.argsort(-importance)[: C // 2])
    oracle = _mk(w_f[keep], w_s[keep], w_out[:, keep])

    rng_x = np.random.default_rng(229)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


# --- Conv chain: Pad channel-safe pass-through hop ----------------------------
#
# `Conv -> Pad -> Conv`: mirrors test_pruning.py's own `_pad_conv_pair_model`
# and its Pad-pass-through test coverage.


def _pad_conv_pair_model(w1, w2, pads, b1=None, spatial=8):
    """`Conv -> Pad -> Conv`, `pads` the raw 8-element (`2 * rank` for a
    rank-4 NCHW tensor) ONNX `pads` layout: `[x1_begin, ..., xk_begin,
    x1_end, ..., xk_end]`."""
    Cin, C2 = w1.shape[1], w2.shape[0]
    pads = np.asarray(pads, dtype=np.int64)
    rank = len(pads) // 2
    initializer = [
        _f32(w1, "W1"),
        _f32(w2, "W2"),
        onnx.numpy_helper.from_array(pads, "Pads"),
    ]
    if b1 is not None:
        conv1 = "h = Conv<kernel_shape=[3,3]>(X, W1, B1)"
        initializer.append(_f32(b1, "B1"))
    else:
        conv1 = "h = Conv<kernel_shape=[3,3]>(X, W1)"
    after_conv1 = spatial - 2
    mid_h = after_conv1 + pads[2] + pads[rank + 2]  # axis 2 (H) begin+end pad
    out_spatial = mid_h - 2
    return _model(
        f"""
        g (float[N,{Cin},{spatial},{spatial}] X) => (float[N,{C2},{out_spatial},{out_spatial}] Y)
        {{
          {conv1}
          p = Pad<mode="constant">(h, Pads)
          Y = Conv<kernel_shape=[3,3]>(p, W2)
        }}
        """,
        initializer=initializer,
    )


def test_cpp_structured_pruning_pad_channel_safe_pass_through_matches_oracle():
    Cin, C1, C2 = 3, 16, 8
    rng = np.random.default_rng(230)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    b1 = rng.standard_normal((C1,)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    pads = [0, 0, 1, 1, 0, 0, 1, 1]
    model = _pad_conv_pair_model(w1, w2, pads, b1=b1)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1"].dims) == [C1 // 2, Cin, 3, 3]
    assert list(inits["W2"].dims) == [C2, C1 // 2, 3, 3]

    keep = _oracle_keep_indices_conv(w1, C1 // 2)
    oracle = _pad_conv_pair_model(w1[keep], w2[:, keep], pads, b1=b1[keep])

    rng_x = np.random.default_rng(231)
    x = rng_x.standard_normal((2, Cin, 8, 8)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_pad_channel_affecting_declines():
    # Nonzero padding on axis 1 (channel) -- changes the output channel
    # count outright, so this must be declined, never guessed at.
    Cin, C1, C2 = 3, 8, 4
    rng = np.random.default_rng(232)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    pads = [0, 1, 0, 0, 0, 1, 0, 0]
    w2 = rng.standard_normal((C2, C1 + 2, 3, 3)).astype(np.float32)
    model = _pad_conv_pair_model(w1, w2, pads)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["W2"], w2)


def test_cpp_structured_pruning_pad_dynamic_pads_declines():
    # `pads` computed at runtime (a non-constant node output) rather than a
    # constant initializer -- declined outright for the same reason as a
    # dynamic Resize `scales` above.
    Cin, C1, C2 = 3, 8, 4
    rng = np.random.default_rng(233)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    model = _model(
        f"""
        g (float[N,{Cin},8,8] X, int64[8] PadsIn) => (float[N,{C2},4,4] Y)
        {{
          h = Conv<kernel_shape=[3,3]>(X, W1)
          pads_dyn = Identity(PadsIn)
          p = Pad<mode="constant">(h, pads_dyn)
          Y = Conv<kernel_shape=[3,3]>(p, W2)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2")],
    )
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["W2"], w2)


def test_cpp_structured_pruning_conv_residual_pad_pass_through_hop_matches_oracle():
    # A channel-safe Pad crossed by the *backward* walk
    # (WalkConvProducerBackward) -- exercises the residual-chain insertion
    # point, not just the forward one.
    Cin, C, Cout = 3, 16, 8
    rng = np.random.default_rng(234)
    w_f = rng.standard_normal((C, Cin, 3, 3)).astype(np.float32)
    # All-zero pads (a no-op Pad node) -- keeps both Add operands' shapes
    # equal (the merge point's own requirement) while still exercising the
    # channel-safety matcher (pads[1] == pads[rank+1] == 0 is trivially true).
    pads = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64)
    w_s = rng.standard_normal((C, Cin, 3, 3)).astype(np.float32)
    w_out = rng.standard_normal((Cout, C, 3, 3)).astype(np.float32)

    def _mk(w_f, w_s, w_out):
        return _model(
            f"""
            g (float[N,{Cin},10,10] X) => (float[N,{Cout},6,6] Y)
            {{
              f0 = Conv<kernel_shape=[3,3]>(X, WF)
              f = Pad<mode="constant">(f0, Pads)
              s = Conv<kernel_shape=[3,3]>(X, WS)
              addr = Add(f, s)
              r = Relu(addr)
              Y = Conv<kernel_shape=[3,3]>(r, WOUT)
            }}
            """,
            initializer=[
                _f32(w_f, "WF"),
                onnx.numpy_helper.from_array(pads, "Pads"),
                _f32(w_s, "WS"),
                _f32(w_out, "WOUT"),
            ],
        )

    model = _mk(w_f, w_s, w_out)
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    importance = np.sqrt(
        np.square(np.linalg.norm(w_f.reshape(C, -1).astype(np.float64), axis=1))
        + np.square(np.linalg.norm(w_s.reshape(C, -1).astype(np.float64), axis=1))
    )
    keep = np.sort(np.argsort(-importance)[: C // 2])
    oracle = _mk(w_f[keep], w_s[keep], w_out[:, keep])

    rng_x = np.random.default_rng(235)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


# --- Concat-merged (skip-connection) chains ----------------------------------


def test_cpp_structured_pruning_matmul_concat_matches_oracle():
    K, C1, C2, Out = 8, 6, 10, 4
    rng = np.random.default_rng(110)
    w1 = rng.standard_normal((K, C1)).astype(np.float32)
    w2 = rng.standard_normal((K, C2)).astype(np.float32)
    wout = rng.standard_normal((C1 + C2, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          a = MatMul(X, W1)
          b = MatMul(X, W2)
          m = Concat<axis = -1>(a, b)
          Y = MatMul(m, WOUT)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2"), _f32(wout, "WOUT")],
    )
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    keep1 = np.sort(
        np.argsort(-np.linalg.norm(w1.astype(np.float64), axis=0))[: C1 // 2]
    )
    keep2 = np.sort(
        np.argsort(-np.linalg.norm(w2.astype(np.float64), axis=0))[: C2 // 2]
    )
    global_keep = np.concatenate([keep1, keep2 + C1])
    oracle = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          a = MatMul(X, W1)
          b = MatMul(X, W2)
          m = Concat<axis = -1>(a, b)
          Y = MatMul(m, WOUT)
        }}
        """,
        initializer=[
            _f32(w1[:, keep1], "W1"),
            _f32(w2[:, keep2], "W2"),
            _f32(wout[global_keep, :], "WOUT"),
        ],
    )
    rng_x = np.random.default_rng(111)
    x = rng_x.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_conv_concat_matches_oracle():
    Cin, C1, C2, Cout = 3, 8, 12, 6
    rng = np.random.default_rng(112)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C2, Cin, 3, 3)).astype(np.float32)
    wout = rng.standard_normal((Cout, C1 + C2, 1, 1)).astype(np.float32)
    model = _model(
        f"""
        g (float[N,{Cin},10,10] X) => (float[N,{Cout},8,8] Y)
        {{
          a = Conv<kernel_shape=[3,3]>(X, W1)
          b = Conv<kernel_shape=[3,3]>(X, W2)
          m = Concat<axis = 1>(a, b)
          Y = Conv<kernel_shape=[1,1]>(m, WOUT)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2"), _f32(wout, "WOUT")],
    )
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    keep1 = np.sort(
        np.argsort(-np.linalg.norm(w1.reshape(C1, -1).astype(np.float64), axis=1))[
            : C1 // 2
        ]
    )
    keep2 = np.sort(
        np.argsort(-np.linalg.norm(w2.reshape(C2, -1).astype(np.float64), axis=1))[
            : C2 // 2
        ]
    )
    global_keep = np.concatenate([keep1, keep2 + C1])
    oracle = _model(
        f"""
        g (float[N,{Cin},10,10] X) => (float[N,{Cout},8,8] Y)
        {{
          a = Conv<kernel_shape=[3,3]>(X, W1)
          b = Conv<kernel_shape=[3,3]>(X, W2)
          m = Concat<axis = 1>(a, b)
          Y = Conv<kernel_shape=[1,1]>(m, WOUT)
        }}
        """,
        initializer=[
            _f32(w1[keep1], "W1"),
            _f32(w2[keep2], "W2"),
            _f32(wout[:, global_keep], "WOUT"),
        ],
    )
    rng_x = np.random.default_rng(113)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_matmul_concat_composed_residual_branch_matches_oracle():
    # One Concat operand ("r") resolves through a whole eligible-Add
    # residual group instead of a bare producer -- both WF/WS join that
    # branch's own combined-importance leaf-producer set, sharing one keep
    # index set, entirely independent of the other ("b") branch's own.
    K, C, C2, Out = 8, 16, 6, 4
    rng = np.random.default_rng(114)
    wf = rng.standard_normal((K, C)).astype(np.float32)
    ws = rng.standard_normal((K, C)).astype(np.float32)
    w2 = rng.standard_normal((K, C2)).astype(np.float32)
    wout = rng.standard_normal((C + C2, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          f = MatMul(X, WF)
          s = MatMul(X, WS)
          addr = Add(f, s)
          r = Relu(addr)
          b = MatMul(X, W2)
          m = Concat<axis = -1>(r, b)
          Y = MatMul(m, WOUT)
        }}
        """,
        initializer=[
            _f32(wf, "WF"),
            _f32(ws, "WS"),
            _f32(w2, "W2"),
            _f32(wout, "WOUT"),
        ],
    )
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    importance1 = np.sqrt(
        np.square(np.linalg.norm(wf.astype(np.float64), axis=0))
        + np.square(np.linalg.norm(ws.astype(np.float64), axis=0))
    )
    keep1 = np.sort(np.argsort(-importance1)[: C // 2])
    keep2 = np.sort(
        np.argsort(-np.linalg.norm(w2.astype(np.float64), axis=0))[: C2 // 2]
    )
    global_keep = np.concatenate([keep1, keep2 + C])
    oracle = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          f = MatMul(X, WF)
          s = MatMul(X, WS)
          addr = Add(f, s)
          r = Relu(addr)
          b = MatMul(X, W2)
          m = Concat<axis = -1>(r, b)
          Y = MatMul(m, WOUT)
        }}
        """,
        initializer=[
            _f32(wf[:, keep1], "WF"),
            _f32(ws[:, keep1], "WS"),
            _f32(w2[:, keep2], "W2"),
            _f32(wout[global_keep, :], "WOUT"),
        ],
    )
    rng_x = np.random.default_rng(115)
    x = rng_x.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_matmul_concat_gated_branch_matches_oracle():
    # A gated (SwiGLU-style) combine feeds a Concat operand directly, with
    # no real producer's raw output in between -- both `gate`'s and `up`'s
    # own producers become this one branch's own `producers` tuple, ranked
    # together by combined importance, entirely independent of `b`'s own.
    K, C, C2, Out = 8, 16, 6, 4
    rng = np.random.default_rng(121)
    wg = rng.standard_normal((K, C)).astype(np.float32)
    wu = rng.standard_normal((K, C)).astype(np.float32)
    w2 = rng.standard_normal((K, C2)).astype(np.float32)
    wout = rng.standard_normal((C + C2, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          gate = MatMul(X, WG)
          gate_act = Sigmoid(gate)
          up = MatMul(X, WU)
          h = Mul(gate_act, up)
          b = MatMul(X, W2)
          m = Concat<axis = -1>(h, b)
          Y = MatMul(m, WOUT)
        }}
        """,
        initializer=[
            _f32(wg, "WG"),
            _f32(wu, "WU"),
            _f32(w2, "W2"),
            _f32(wout, "WOUT"),
        ],
    )
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    importance1 = np.sqrt(
        np.square(np.linalg.norm(wg.astype(np.float64), axis=0))
        + np.square(np.linalg.norm(wu.astype(np.float64), axis=0))
    )
    keep1 = np.sort(np.argsort(-importance1)[: C // 2])
    keep2 = np.sort(
        np.argsort(-np.linalg.norm(w2.astype(np.float64), axis=0))[: C2 // 2]
    )
    global_keep = np.concatenate([keep1, keep2 + C])
    oracle = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          gate = MatMul(X, WG)
          gate_act = Sigmoid(gate)
          up = MatMul(X, WU)
          h = Mul(gate_act, up)
          b = MatMul(X, W2)
          m = Concat<axis = -1>(h, b)
          Y = MatMul(m, WOUT)
        }}
        """,
        initializer=[
            _f32(wg[:, keep1], "WG"),
            _f32(wu[:, keep1], "WU"),
            _f32(w2[:, keep2], "W2"),
            _f32(wout[global_keep, :], "WOUT"),
        ],
    )
    rng_x = np.random.default_rng(122)
    x = rng_x.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_matmul_concat_declines_on_fan_out_branch():
    # Branch `a` also feeds `Z` directly -- a real extra consumer a Concat
    # branch has no fan-out resolution for (unlike a residual/merge group)
    # -- the whole Concat node is declined, left completely untouched.
    K, C1, C2, Out = 8, 6, 10, 4
    rng = np.random.default_rng(116)
    w1 = rng.standard_normal((K, C1)).astype(np.float32)
    w2 = rng.standard_normal((K, C2)).astype(np.float32)
    wout = rng.standard_normal((C1 + C2, Out)).astype(np.float32)
    wextra = rng.standard_normal((C1, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y, float[batch,{Out}] Z)
        {{
          a = MatMul(X, W1)
          b = MatMul(X, W2)
          m = Concat<axis = -1>(a, b)
          Y = MatMul(m, WOUT)
          Z = MatMul(a, WEXTRA)
        }}
        """,
        initializer=[
            _f32(w1, "W1"),
            _f32(w2, "W2"),
            _f32(wout, "WOUT"),
            _f32(wextra, "WEXTRA"),
        ],
    )
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["W2"], w2)
    np.testing.assert_array_equal(inits["WOUT"], wout)


def test_cpp_structured_pruning_conv_concat_declines_on_grouped_consumer():
    Cin, C1, C2, Cout, group = 3, 8, 8, 8, 2
    rng = np.random.default_rng(120)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C2, Cin, 3, 3)).astype(np.float32)
    wout = rng.standard_normal((Cout, (C1 + C2) // group, 1, 1)).astype(np.float32)
    model = _model(
        f"""
        g (float[N,{Cin},10,10] X) => (float[N,{Cout},8,8] Y)
        {{
          a = Conv<kernel_shape=[3,3]>(X, W1)
          b = Conv<kernel_shape=[3,3]>(X, W2)
          m = Concat<axis = 1>(a, b)
          Y = Conv<kernel_shape=[1,1],group={group}>(m, WOUT)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2"), _f32(wout, "WOUT")],
    )
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["W2"], w2)
    np.testing.assert_array_equal(inits["WOUT"], wout)


def test_cpp_structured_pruning_matmul_concat_accepts_positive_last_axis_when_rank_known():
    K, C1, C2, Out = 8, 6, 10, 4
    rng = np.random.default_rng(117)
    w1 = rng.standard_normal((K, C1)).astype(np.float32)
    w2 = rng.standard_normal((K, C2)).astype(np.float32)
    wout = rng.standard_normal((C1 + C2, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          a = MatMul(X, W1)
          b = MatMul(X, W2)
          m = Concat<axis = 1>(a, b)
          Y = MatMul(m, WOUT)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2"), _f32(wout, "WOUT")],
    )
    # A positive axis is only recognized as "last" once at least one
    # operand's rank is confirmed via value_info -- add it directly rather
    # than running full-graph shape inference (whose validity elsewhere in
    # the graph this test doesn't care about).
    model.graph.value_info.append(
        onnx.helper.make_tensor_value_info("a", onnx.TensorProto.FLOAT, [None, C1])
    )
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    assert inits["W1"].shape == (K, C1 // 2)
    assert inits["W2"].shape == (K, C2 // 2)


def test_cpp_structured_pruning_matmul_concat_declines_on_positive_non_last_axis():
    K, C1, C2, Out = 8, 6, 10, 4
    rng = np.random.default_rng(118)
    w1 = rng.standard_normal((K, C1)).astype(np.float32)
    w2 = rng.standard_normal((K, C2)).astype(np.float32)
    wout = rng.standard_normal((C1 + C2, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          a = MatMul(X, W1)
          b = MatMul(X, W2)
          m = Concat<axis = 0>(a, b)
          Y = MatMul(m, WOUT)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2"), _f32(wout, "WOUT")],
    )
    model.graph.value_info.append(
        onnx.helper.make_tensor_value_info("a", onnx.TensorProto.FLOAT, [None, C1])
    )
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["W2"], w2)
    np.testing.assert_array_equal(inits["WOUT"], wout)


def test_cpp_structured_pruning_matmul_concat_declines_on_positive_axis_unknown_rank():
    K, C1, C2, Out = 8, 6, 10, 4
    rng = np.random.default_rng(119)
    w1 = rng.standard_normal((K, C1)).astype(np.float32)
    w2 = rng.standard_normal((K, C2)).astype(np.float32)
    wout = rng.standard_normal((C1 + C2, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          a = MatMul(X, W1)
          b = MatMul(X, W2)
          m = Concat<axis = 1>(a, b)
          Y = MatMul(m, WOUT)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2"), _f32(wout, "WOUT")],
    )
    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["W2"], w2)
    np.testing.assert_array_equal(inits["WOUT"], wout)


def test_cpp_structured_pruning_matches_python_reference_output_with_concat_chain():
    K, C1, C2, Out = 8, 6, 10, 4
    rng = np.random.default_rng(123)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          a = MatMul(X, W1)
          b = MatMul(X, W2)
          m = Concat<axis = -1>(a, b)
          Y = MatMul(m, WOUT)
        }}
        """,
        initializer=[
            _f32(rng.standard_normal((K, C1)), "W1"),
            _f32(rng.standard_normal((K, C2)), "W2"),
            _f32(rng.standard_normal((C1 + C2, Out)), "WOUT"),
        ],
    )
    pruned_py = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    pruned_cpp = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned_py)
    onnx.checker.check_model(pruned_cpp)

    rng_x = np.random.default_rng(124)
    x = rng_x.standard_normal((6, K)).astype(np.float32)
    (y_py,) = _run(pruned_py, {"X": x})
    (y_cpp,) = _run(pruned_cpp, {"X": x})
    np.testing.assert_allclose(y_py, y_cpp, rtol=1e-5, atol=1e-5)


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


# --- Subgraph recursion (If/Loop) --------------------------------------------
#
# Covers `structured_pruning_entry.cpp`'s own `IterSubgraphs` and the
# `ApplyStructuredPruning` loop built on it -- a straight C++ port of
# `onnxsim/pruning.py`'s own `_iter_subgraphs`/`apply_structured_pruning`
# subgraph-recursion round (see that module's "Subgraph recursion" section
# comment, and `structured_pruning_entry.cpp`'s own copy of it directly
# above `IterSubgraphs`'s definition, for the full design rationale). Model
# shapes below mirror `tests/test_pruning.py`'s own
# `_if_wrapped_mlp_model`/`test_structured_pruning_prunes_top_level_and_
# both_if_branches` fixture exactly, so a diff against those tests is the
# fastest way to see this is deliberately the same scenario, just driven
# through `apply_structured_pruning_cpp` instead of the pure-Python
# reference.
#
# `onnx.parser.parse_model`'s text format has no way to spell a graph-typed
# node attribute (an `If`'s `then_branch`/`else_branch`, a `Loop`'s
# `body`), so every model below uses `onnx.helper.make_node`/`make_graph`
# directly instead, per this repo's own CLAUDE.md guidance for exactly this
# case.


def _mlp_branch_nodes(K, H, Out, prefix, seed):
    # A minimal MatMul(Gemm)->Relu->MatMul chain, exactly `_mlp_model`'s own
    # shape, but returning bare nodes/initializers (not a whole model) so it
    # can be dropped into a subgraph's own `node`/`initializer` lists.
    rng = np.random.default_rng(seed)
    w1 = rng.standard_normal((K, H)).astype(np.float32)
    b1 = rng.standard_normal((H,)).astype(np.float32)
    w2 = rng.standard_normal((H, Out)).astype(np.float32)
    nodes = [
        onnx.helper.make_node(
            "Gemm", ["Xb", f"{prefix}W1", f"{prefix}B1"], [f"{prefix}h"]
        ),
        onnx.helper.make_node("Relu", [f"{prefix}h"], [f"{prefix}a"]),
        onnx.helper.make_node("MatMul", [f"{prefix}a", f"{prefix}W2"], ["Yb"]),
    ]
    inits = [
        _f32(w1, f"{prefix}W1"),
        _f32(b1, f"{prefix}B1"),
        _f32(w2, f"{prefix}W2"),
    ]
    return nodes, inits, dict(w1=w1, b1=b1, w2=w2)


def _if_wrapped_mlp_model(K0=8, H0=16, Out0=4, K1=6, H1=12, OutB=3):
    """A top-level MatMul(Gemm)->Relu->MatMul chain (`W1t`/`W2t`) PLUS an
    `If` node whose `then_branch`/`else_branch` each carry their OWN
    independent, identically-shaped MLP chain (`then_*`/`else_*`), with
    their own weights living only in that branch's own `initializer` list.
    `Xb` (the branch chains' shared activation input) is an ordinary
    top-level graph input, read by both branches purely via implicit
    capture (an `If` branch subgraph takes no formal inputs of its own).
    `cond` selects which branch actually executes at run time.
    """
    rng = np.random.default_rng(0)
    w1t = rng.standard_normal((K0, H0)).astype(np.float32)
    b1t = rng.standard_normal((H0,)).astype(np.float32)
    w2t = rng.standard_normal((H0, Out0)).astype(np.float32)
    top_nodes = [
        onnx.helper.make_node("Gemm", ["X0", "W1t", "B1t"], ["ht"]),
        onnx.helper.make_node("Relu", ["ht"], ["at"]),
        onnx.helper.make_node("MatMul", ["at", "W2t"], ["Y0"]),
    ]
    top_inits = [_f32(w1t, "W1t"), _f32(b1t, "B1t"), _f32(w2t, "W2t")]

    then_nodes, then_inits, then_cfg = _mlp_branch_nodes(K1, H1, OutB, "then_", seed=1)
    else_nodes, else_inits, else_cfg = _mlp_branch_nodes(K1, H1, OutB, "else_", seed=2)

    out_vi = onnx.helper.make_tensor_value_info(
        "Yb", onnx.TensorProto.FLOAT, ["batch", OutB]
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

    x0 = onnx.helper.make_tensor_value_info("X0", onnx.TensorProto.FLOAT, ["batch", K0])
    xb = onnx.helper.make_tensor_value_info("Xb", onnx.TensorProto.FLOAT, ["batch", K1])
    cond = onnx.helper.make_tensor_value_info("cond", onnx.TensorProto.BOOL, [])
    y0 = onnx.helper.make_tensor_value_info(
        "Y0", onnx.TensorProto.FLOAT, ["batch", Out0]
    )
    y1 = onnx.helper.make_tensor_value_info(
        "Y1", onnx.TensorProto.FLOAT, ["batch", OutB]
    )

    graph = onnx.helper.make_graph(
        [*top_nodes, if_node],
        "g",
        [x0, xb, cond],
        [y0, y1],
        initializer=top_inits,
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 17)]
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model, dict(w1t=w1t, b1t=b1t, w2t=w2t, then=then_cfg, else_=else_cfg)


def _then_else_graphs(pruned_model):
    if_node = next(n for n in pruned_model.graph.node if n.op_type == "If")
    then_g = else_g = None
    for attr in if_node.attribute:
        if attr.name == "then_branch":
            then_g = attr.g
        elif attr.name == "else_branch":
            else_g = attr.g
    return then_g, else_g


def _loop_wrapped_mlp_model(K0=8, H0=16, Out0=4, K1=6, H1=12, OutB=3, M=3):
    """The `Loop`-body counterpart of `_if_wrapped_mlp_model` above --
    covers the other half of "If/Loop/Scan" this file's own "Subgraph
    recursion" comment (and `structured_pruning_entry.cpp`'s own copy of
    it) names. A top-level MatMul(Gemm)->Relu->MatMul chain (`W1t`/`W2t`)
    PLUS a `Loop` node whose `body` carries its OWN independent MLP chain
    (`loop_*`), with its own weights living only in the body's own
    `initializer` list. `Xb` is read every iteration purely via implicit
    capture from the top-level graph's own input (`Loop`'s body takes only
    `iter_num`/`cond_in` as formal inputs -- no loop-carried dependency);
    `Yb` is emitted as a `scan_output`, stacked across all `M` iterations
    into `Ys`.
    """
    rng = np.random.default_rng(0)
    w1t = rng.standard_normal((K0, H0)).astype(np.float32)
    b1t = rng.standard_normal((H0,)).astype(np.float32)
    w2t = rng.standard_normal((H0, Out0)).astype(np.float32)
    top_nodes = [
        onnx.helper.make_node("Gemm", ["X0", "W1t", "B1t"], ["ht"]),
        onnx.helper.make_node("Relu", ["ht"], ["at"]),
        onnx.helper.make_node("MatMul", ["at", "W2t"], ["Y0"]),
    ]
    top_inits = [_f32(w1t, "W1t"), _f32(b1t, "B1t"), _f32(w2t, "W2t")]

    body_nodes, body_inits, body_cfg = _mlp_branch_nodes(K1, H1, OutB, "loop_", seed=1)
    cond_pass_through = onnx.helper.make_node("Identity", ["cond_in"], ["cond_out"])
    iter_num_vi = onnx.helper.make_tensor_value_info(
        "iter_num", onnx.TensorProto.INT64, []
    )
    cond_in_vi = onnx.helper.make_tensor_value_info(
        "cond_in", onnx.TensorProto.BOOL, []
    )
    cond_out_vi = onnx.helper.make_tensor_value_info(
        "cond_out", onnx.TensorProto.BOOL, []
    )
    yb_vi = onnx.helper.make_tensor_value_info(
        "Yb", onnx.TensorProto.FLOAT, ["batch", OutB]
    )
    body_graph = onnx.helper.make_graph(
        [*body_nodes, cond_pass_through],
        "loop_body",
        [iter_num_vi, cond_in_vi],
        [cond_out_vi, yb_vi],
        initializer=body_inits,
    )
    loop_node = onnx.helper.make_node("Loop", ["M", "cond"], ["Ys"], body=body_graph)

    x0 = onnx.helper.make_tensor_value_info("X0", onnx.TensorProto.FLOAT, ["batch", K0])
    xb = onnx.helper.make_tensor_value_info("Xb", onnx.TensorProto.FLOAT, ["batch", K1])
    m = onnx.helper.make_tensor_value_info("M", onnx.TensorProto.INT64, [])
    cond = onnx.helper.make_tensor_value_info("cond", onnx.TensorProto.BOOL, [])
    y0 = onnx.helper.make_tensor_value_info(
        "Y0", onnx.TensorProto.FLOAT, ["batch", Out0]
    )
    ys = onnx.helper.make_tensor_value_info(
        "Ys", onnx.TensorProto.FLOAT, [M, "batch", OutB]
    )

    graph = onnx.helper.make_graph(
        [*top_nodes, loop_node],
        "g",
        [x0, xb, m, cond],
        [y0, ys],
        initializer=top_inits,
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 17)]
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model, dict(w1t=w1t, b1t=b1t, w2t=w2t, loop=body_cfg)


def test_cpp_structured_pruning_prunes_top_level_and_both_if_branches():
    # The core repro: apply_structured_pruning_cpp must match and prune the
    # chain inside BOTH `then_branch` and `else_branch` (each with its own
    # independent weights) -- not just the top-level chain -- verified both
    # by initializer shape and by driving real execution (through
    # InferenceSession) into EACH branch via `cond`, comparing against an
    # independently reconstructed "already pruned" numpy oracle for that
    # branch's own weights. Also proves independence both ways: the
    # top-level chain's own pruning is unaffected by what's inside the `If`
    # (it lands on exactly the same oracle
    # `test_cpp_structured_pruning_matches_python_reference_output` already
    # checks for a subgraph-free model), and each branch's own pruning is
    # unaffected by the top-level chain or its sibling branch.
    K0, H0, Out0 = 8, 16, 4
    K1, H1, OutB = 6, 12, 3
    model, cfg = _if_wrapped_mlp_model(K0=K0, H0=H0, Out0=Out0, K1=K1, H1=H1, OutB=OutB)

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    top_inits = {t.name: t for t in pruned.graph.initializer}
    assert list(top_inits["W1t"].dims) == [K0, H0 // 2]
    assert list(top_inits["W2t"].dims) == [H0 // 2, Out0]

    then_g, else_g = _then_else_graphs(pruned)
    then_inits = {t.name: t for t in then_g.initializer}
    else_inits = {t.name: t for t in else_g.initializer}
    assert list(then_inits["then_W1"].dims) == [K1, H1 // 2]
    assert list(then_inits["then_W2"].dims) == [H1 // 2, OutB]
    assert list(else_inits["else_W1"].dims) == [K1, H1 // 2]
    assert list(else_inits["else_W2"].dims) == [H1 // 2, OutB]

    rng = np.random.default_rng(5)
    x0 = rng.standard_normal((3, K0)).astype(np.float32)
    xb = rng.standard_normal((3, K1)).astype(np.float32)

    y0_true, y1_then = _run(pruned, {"X0": x0, "Xb": xb, "cond": np.array(True)})
    y0_false, y1_else = _run(pruned, {"X0": x0, "Xb": xb, "cond": np.array(False)})

    def _oracle(branch_cfg, keep_count):
        w1, b1, w2 = branch_cfg["w1"], branch_cfg["b1"], branch_cfg["w2"]
        keep = _oracle_keep_indices(w1, keep_count)
        h = xb @ w1[:, keep] + b1[keep]
        a = np.maximum(h, 0)
        return a @ w2[keep, :]

    np.testing.assert_allclose(
        y1_then, _oracle(cfg["then"], H1 // 2), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        y1_else, _oracle(cfg["else_"], H1 // 2), rtol=1e-5, atol=1e-5
    )

    keep_top = _oracle_keep_indices(cfg["w1t"], H0 // 2)
    h0 = x0 @ cfg["w1t"][:, keep_top] + cfg["b1t"][keep_top]
    a0 = np.maximum(h0, 0)
    y0_oracle = a0 @ cfg["w2t"][keep_top, :]
    np.testing.assert_allclose(y0_true, y0_oracle, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(y0_false, y0_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_prunes_top_level_and_loop_body():
    # Same repro as the `If`-branch test above, but for a `Loop` body --
    # both the top-level chain and the chain living entirely inside the
    # `body` subgraph must be pruned to the same H1 // 2 width, each using
    # its own independent (in this case, identical-modulo-independent-
    # oracle) importance ranking, with neither affecting the other.
    K0, H0, Out0 = 8, 16, 4
    K1, H1, OutB = 6, 12, 3
    M = 3
    model, cfg = _loop_wrapped_mlp_model(
        K0=K0, H0=H0, Out0=Out0, K1=K1, H1=H1, OutB=OutB, M=M
    )

    pruned = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    top_inits = {t.name: t for t in pruned.graph.initializer}
    assert list(top_inits["W1t"].dims) == [K0, H0 // 2]
    assert list(top_inits["W2t"].dims) == [H0 // 2, Out0]

    loop_node = next(n for n in pruned.graph.node if n.op_type == "Loop")
    body = next(a.g for a in loop_node.attribute if a.name == "body")
    body_inits = {t.name: t for t in body.initializer}
    assert list(body_inits["loop_W1"].dims) == [K1, H1 // 2]
    assert list(body_inits["loop_W2"].dims) == [H1 // 2, OutB]

    rng = np.random.default_rng(6)
    x0 = rng.standard_normal((3, K0)).astype(np.float32)
    xb = rng.standard_normal((3, K1)).astype(np.float32)
    y0, ys = _run(
        pruned,
        {
            "X0": x0,
            "Xb": xb,
            "M": np.array(M, dtype=np.int64),
            "cond": np.array(True),
        },
    )

    def _oracle(branch_cfg, keep_count):
        w1, b1, w2 = branch_cfg["w1"], branch_cfg["b1"], branch_cfg["w2"]
        keep = _oracle_keep_indices(w1, keep_count)
        h = xb @ w1[:, keep] + b1[keep]
        a = np.maximum(h, 0)
        return a @ w2[keep, :]

    yb_oracle = _oracle(cfg["loop"], H1 // 2)
    # Every iteration recomputes the exact same thing (no loop-carried
    # state, `Xb` fixed across iterations), so every one of the M stacked
    # scan-output slices must equal the same oracle.
    np.testing.assert_allclose(
        ys, np.broadcast_to(yb_oracle, ys.shape), rtol=1e-5, atol=1e-5
    )

    keep_top = _oracle_keep_indices(cfg["w1t"], H0 // 2)
    h0 = x0 @ cfg["w1t"][:, keep_top] + cfg["b1t"][keep_top]
    a0 = np.maximum(h0, 0)
    y0_oracle = a0 @ cfg["w2t"][keep_top, :]
    np.testing.assert_allclose(y0, y0_oracle, rtol=1e-5, atol=1e-5)


def test_cpp_structured_pruning_matches_python_reference_output_with_if_subgraph():
    # Cross-check against onnxsim.apply_structured_pruning (the pure-Python
    # reference this C++ port mirrors) on a model where the only prunable
    # weight worth talking about lives inside the `If`'s own branches --
    # both `cond` values are driven through InferenceSession so both
    # branches' own subgraph-recursion behavior is exercised, not just
    # whichever one a single run happens to select.
    K0, H0, Out0 = 8, 16, 4
    K1, H1, OutB = 6, 12, 3
    model, _cfg = _if_wrapped_mlp_model(
        K0=K0, H0=H0, Out0=Out0, K1=K1, H1=H1, OutB=OutB
    )

    pruned_py = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    pruned_cpp = onnxsim.apply_structured_pruning_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned_py)
    onnx.checker.check_model(pruned_cpp)

    rng = np.random.default_rng(7)
    x0 = rng.standard_normal((3, K0)).astype(np.float32)
    xb = rng.standard_normal((3, K1)).astype(np.float32)
    for cond in (True, False):
        feeds = {"X0": x0, "Xb": xb, "cond": np.array(cond)}
        y0_py, y1_py = _run(pruned_py, feeds)
        y0_cpp, y1_cpp = _run(pruned_cpp, feeds)
        np.testing.assert_allclose(y0_py, y0_cpp, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(y1_py, y1_cpp, rtol=1e-5, atol=1e-5)
