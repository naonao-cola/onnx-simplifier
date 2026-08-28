"""Tests for ``onnxsim.pruning`` -- magnitude pruning (data-free baseline),
Wanda pruning (calibrated on activation norms), and structured (channel)
pruning, see ``onnxsim/pruning.py``.
"""

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper
import pytest

import onnxsim

ort = pytest.importorskip("onnxruntime")


def _vi(name, shape):
    return onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _model(nodes, inputs, outputs, initializer, opset=21):
    graph = onnx.helper.make_graph(nodes, "g", inputs, outputs, initializer)
    return onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", opset)], ir_version=10
    )


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _matmul_model(K=64, N=16, seed=0):
    rng = np.random.default_rng(seed)
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    return _model(
        nodes, [_vi("X", ["batch", K])], [_vi("Y", ["batch", N])], [_f32(weight, "W")]
    )


def _weight(model):
    return onnx.numpy_helper.to_array(model.graph.initializer[0])


def test_magnitude_pruning_reaches_target_sparsity():
    model = _matmul_model(K=64, N=16)
    pruned = onnxsim.apply_magnitude_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    assert onnxsim.weight_sparsity(pruned) == pytest.approx(0.5, abs=1e-9)
    # Shape is untouched -- this is a value-only rewrite.
    assert _weight(pruned).shape == _weight(model).shape


def test_magnitude_pruning_keeps_the_largest_entries_per_row():
    model = _matmul_model(K=64, N=16)
    pruned = onnxsim.apply_magnitude_pruning(model, sparsity=0.75)
    w = _weight(model).astype(np.float64)  # [K, N]
    w_pruned = _weight(pruned).astype(np.float64)
    # Per output column (row of W^T), the surviving entries must be exactly
    # the top-(1 - sparsity) fraction by magnitude.
    for col in range(w.shape[1]):
        kept = np.flatnonzero(w_pruned[:, col] != 0)
        assert len(kept) == 16  # round(64 * 0.25)
        threshold = np.abs(w[:, col])[kept].min()
        dropped_max = np.abs(w[:, col])[np.flatnonzero(w_pruned[:, col] == 0)].max()
        assert dropped_max <= threshold


def test_magnitude_pruning_zero_sparsity_is_a_no_op():
    model = _matmul_model(K=32, N=8)
    pruned = onnxsim.apply_magnitude_pruning(model, sparsity=0.0)
    np.testing.assert_array_equal(_weight(pruned), _weight(model))


def test_magnitude_pruning_nm_pattern():
    model = _matmul_model(K=64, N=16)
    pruned = onnxsim.apply_magnitude_pruning(model, n=2, m=4)
    w_pruned = _weight(pruned).T  # [N, K], row-major per output channel
    assert onnxsim.weight_sparsity(pruned) == pytest.approx(0.5, abs=1e-9)
    for row in w_pruned:
        for start in range(0, len(row), 4):
            group = row[start : start + 4]
            assert np.count_nonzero(group) <= 2


def test_magnitude_pruning_requires_n_and_m_together():
    model = _matmul_model(K=16, N=4)
    with pytest.raises(ValueError):
        onnxsim.apply_magnitude_pruning(model, n=2)
    with pytest.raises(ValueError):
        onnxsim.apply_magnitude_pruning(model, m=4)


def test_wanda_pruning_protects_high_activation_channels():
    # A handful of input channels carry much larger activation magnitude
    # than the rest but a merely-average weight magnitude -- Wanda's own
    # motivating scenario: plain |W| magnitude pruning is blind to this and
    # may prune those channels' weights anyway, while Wanda's
    # |W| * ||X||_2 metric should protect them.
    K, N = 64, 16
    salient = (3, 7, 40)
    rng = np.random.default_rng(0)
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    model = _model(
        nodes, [_vi("X", ["batch", K])], [_vi("Y", ["batch", N])], [_f32(weight, "W")]
    )

    x = rng.standard_normal((32, K)).astype(np.float32)
    for c in salient:
        x[:, c] *= 20.0
    calibration_data = [{"X": x}]

    magnitude_pruned = onnxsim.apply_magnitude_pruning(model, sparsity=0.5)
    wanda_pruned = onnxsim.apply_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(wanda_pruned)
    assert onnxsim.weight_sparsity(wanda_pruned) == pytest.approx(0.5, abs=1e-9)

    w_magnitude = _weight(magnitude_pruned)
    w_wanda = _weight(wanda_pruned)
    # Wanda must keep strictly more of the salient rows' entries than plain
    # magnitude pruning -- otherwise this is just re-testing magnitude
    # pruning under a different name.
    salient_kept_magnitude = np.count_nonzero(w_magnitude[list(salient), :])
    salient_kept_wanda = np.count_nonzero(w_wanda[list(salient), :])
    assert salient_kept_wanda > salient_kept_magnitude

    (float_y,) = _run(model, {"X": x})
    (magnitude_y,) = _run(magnitude_pruned, {"X": x})
    (wanda_y,) = _run(wanda_pruned, {"X": x})
    magnitude_err = np.linalg.norm(float_y.astype(np.float64) - magnitude_y)
    wanda_err = np.linalg.norm(float_y.astype(np.float64) - wanda_y)
    assert wanda_err < magnitude_err


def test_wanda_pruning_falls_back_to_magnitude_without_matching_activation():
    # X isn't 2-D at the probe point (it's 3-D), so Wanda never observes a
    # usable activation norm and must fall back to plain |W| pruning rather
    # than leaving the layer untouched or crashing.
    K, N = 32, 8
    rng = np.random.default_rng(0)
    weight = rng.standard_normal((K, N)).astype(np.float32)
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    model = _model(
        nodes,
        [_vi("X", ["batch", "seq", K])],
        [_vi("Y", ["batch", "seq", N])],
        [_f32(weight, "W")],
    )
    x = rng.standard_normal((2, 4, K)).astype(np.float32)

    magnitude_pruned = onnxsim.apply_magnitude_pruning(model, sparsity=0.5)
    wanda_pruned = onnxsim.apply_wanda_pruning(
        model, calibration_data=[{"X": x}], sparsity=0.5
    )
    onnx.checker.check_model(wanda_pruned)
    np.testing.assert_array_equal(_weight(wanda_pruned), _weight(magnitude_pruned))


def test_weight_sparsity_of_unpruned_model_is_zero():
    model = _matmul_model(K=16, N=4)
    assert onnxsim.weight_sparsity(model) == 0.0


def test_weight_sparsity_ignores_non_matching_layers():
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Relu", ["X"], ["Y"])],
        "g",
        [_vi("X", [4])],
        [_vi("Y", [4])],
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 21)], ir_version=10
    )
    assert onnxsim.weight_sparsity(model) == 0.0


# --- apply_structured_pruning ------------------------------------------------


def _mlp_model(K=8, H=32, Out=4, bias=True, activation="Relu", seed=0):
    rng = np.random.default_rng(seed)
    w1 = rng.standard_normal((K, H)).astype(np.float32)
    w2 = rng.standard_normal((H, Out)).astype(np.float32)
    initializer = [_f32(w1, "W1"), _f32(w2, "W2")]
    if bias:
        b1 = rng.standard_normal((H,)).astype(np.float32)
        gemm1 = onnx.helper.make_node("Gemm", ["X", "W1", "B1"], ["h"])
        initializer.append(_f32(b1, "B1"))
    else:
        gemm1 = onnx.helper.make_node("MatMul", ["X", "W1"], ["h"])
    nodes = [
        gemm1,
        onnx.helper.make_node(activation, ["h"], ["a"]),
        onnx.helper.make_node("MatMul", ["a", "W2"], ["Y"]),
    ]
    return _model(
        nodes, [_vi("X", ["batch", K])], [_vi("Y", ["batch", Out])], initializer
    )


def _oracle_keep_indices(w1, keep_count):
    importance = np.linalg.norm(w1.T, axis=1)
    return np.sort(np.argsort(-importance)[:keep_count])


def test_structured_pruning_shrinks_matched_layers():
    model = _mlp_model(K=8, H=32, Out=4)
    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1"].dims) == [8, 16]
    assert list(inits["B1"].dims) == [16]
    assert list(inits["W2"].dims) == [16, 4]


def test_structured_pruning_matches_manual_channel_deletion_exactly():
    # The real correctness bar isn't "close to the float model" (removing
    # half the hidden units on random weights changes the output a lot,
    # by design) -- it's exact equivalence to deleting the same channels
    # by hand in numpy.
    K, H, Out = 8, 32, 4
    model = _mlp_model(K=K, H=H, Out=Out, bias=True)
    orig = {t.name: onnx.numpy_helper.to_array(t) for t in model.graph.initializer}
    w1, b1, w2 = orig["W1"], orig["B1"], orig["W2"]

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    keep = _oracle_keep_indices(w1, H // 2)

    rng = np.random.default_rng(1)
    x = rng.standard_normal((6, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})

    h = x @ w1[:, keep] + b1[keep]
    a = np.maximum(h, 0)
    y_oracle = a @ w2[keep, :]

    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_structured_pruning_matmul_only_chain_matches_oracle():
    # No Gemm bias at all -- a plain MatMul -> activation -> MatMul chain.
    K, H, Out = 8, 24, 4
    model = _mlp_model(K=K, H=H, Out=Out, bias=False, activation="Sigmoid")
    w1 = onnx.numpy_helper.to_array(model.graph.initializer[0])
    w2 = onnx.numpy_helper.to_array(model.graph.initializer[1])

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.25)
    keep = _oracle_keep_indices(w1, H - round(H * 0.25))

    rng = np.random.default_rng(2)
    x = rng.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})

    h = x @ w1[:, keep]
    a = 1.0 / (1.0 + np.exp(-h))
    y_oracle = a @ w2[keep, :]

    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_structured_pruning_bias_add_between_matmuls_matches_oracle():
    # Bias as a separate Add node (not Gemm's own 3rd input) must be caught
    # by the elementwise chain-walk, not just Gemm's native bias slot.
    K, H, Out = 8, 16, 4
    rng = np.random.default_rng(3)
    w1 = rng.standard_normal((K, H)).astype(np.float32)
    bias = rng.standard_normal((H,)).astype(np.float32)
    w2 = rng.standard_normal((H, Out)).astype(np.float32)
    nodes = [
        onnx.helper.make_node("MatMul", ["X", "W1"], ["h"]),
        onnx.helper.make_node("Add", ["h", "Bias"], ["hb"]),
        onnx.helper.make_node("Relu", ["hb"], ["a"]),
        onnx.helper.make_node("MatMul", ["a", "W2"], ["Y"]),
    ]
    model = _model(
        nodes,
        [_vi("X", ["batch", K])],
        [_vi("Y", ["batch", Out])],
        [_f32(w1, "W1"), _f32(bias, "Bias"), _f32(w2, "W2")],
    )

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
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


def test_structured_pruning_skips_branching_output():
    # h feeds both the Relu->MatMul chain *and* is itself a graph output --
    # pruning it would silently change what the caller observes, so this
    # must be left completely untouched.
    K, H, Out = 8, 16, 4
    model = _mlp_model(K=K, H=H, Out=Out, bias=False)
    graph = model.graph
    graph.output.append(_vi("h", ["batch", H]))

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1"].dims) == [K, H]
    assert list(inits["W2"].dims) == [H, Out]


def test_structured_pruning_skips_multi_consumer_branch():
    # h feeds two separate downstream MatMuls -- not the single-consumer
    # chain this pass proves safe to cut, so it must be left untouched.
    K, H, Out = 8, 16, 4
    rng = np.random.default_rng(4)
    w1 = rng.standard_normal((K, H)).astype(np.float32)
    w2 = rng.standard_normal((H, Out)).astype(np.float32)
    w3 = rng.standard_normal((H, Out)).astype(np.float32)
    nodes = [
        onnx.helper.make_node("MatMul", ["X", "W1"], ["h"]),
        onnx.helper.make_node("MatMul", ["h", "W2"], ["Y1"]),
        onnx.helper.make_node("MatMul", ["h", "W3"], ["Y2"]),
    ]
    model = _model(
        nodes,
        [_vi("X", ["batch", K])],
        [_vi("Y1", ["batch", Out]), _vi("Y2", ["batch", Out])],
        [_f32(w1, "W1"), _f32(w2, "W2"), _f32(w3, "W3")],
    )

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1"].dims) == [K, H]


def test_structured_pruning_zero_sparsity_is_a_no_op():
    model = _mlp_model(K=8, H=16, Out=4)
    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.0)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1"].dims) == [8, 16]


def test_structured_pruning_invalid_sparsity_raises():
    model = _mlp_model(K=8, H=16, Out=4)
    with pytest.raises(ValueError):
        onnxsim.apply_structured_pruning(model, sparsity=1.0)
    with pytest.raises(ValueError):
        onnxsim.apply_structured_pruning(model, sparsity=-0.1)


def test_structured_pruning_chains_through_a_third_layer():
    # W2 is a producer for one chain (its own output channels feeding W3)
    # and a consumer for another (W1's output channels feeding into it) --
    # independent axes of the same tensor, both must be pruned correctly.
    K, H1, H2, Out = 8, 16, 20, 4
    rng = np.random.default_rng(5)
    w1 = rng.standard_normal((K, H1)).astype(np.float32)
    w2 = rng.standard_normal((H1, H2)).astype(np.float32)
    w3 = rng.standard_normal((H2, Out)).astype(np.float32)
    nodes = [
        onnx.helper.make_node("MatMul", ["X", "W1"], ["h1"]),
        onnx.helper.make_node("Relu", ["h1"], ["a1"]),
        onnx.helper.make_node("MatMul", ["a1", "W2"], ["h2"]),
        onnx.helper.make_node("Relu", ["h2"], ["a2"]),
        onnx.helper.make_node("MatMul", ["a2", "W3"], ["Y"]),
    ]
    model = _model(
        nodes,
        [_vi("X", ["batch", K])],
        [_vi("Y", ["batch", Out])],
        [_f32(w1, "W1"), _f32(w2, "W2"), _f32(w3, "W3")],
    )

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
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


# --- apply_structured_pruning: gated FFN (SwiGLU/GeGLU) ----------------------


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
    nodes = [
        onnx.helper.make_node("MatMul", ["X", "Wg"], ["gate"]),
        onnx.helper.make_node(gate_activation, ["gate"], ["gate_act"]),
        onnx.helper.make_node("MatMul", ["X", "Wu"], ["up"]),
        onnx.helper.make_node("Mul", ["gate_act", "up"], ["h"]),
        onnx.helper.make_node("MatMul", ["h", "Wd"], ["Y"]),
    ]
    model = _model(
        nodes,
        [_vi("X", ["batch", K])],
        [_vi("Y", ["batch", Out])],
        [_f32(wg, "Wg"), _f32(wu, "Wu"), _f32(wd, "Wd")],
    )
    return model, wg, wu, wd


def test_structured_pruning_gated_ffn_matches_oracle():
    K, H, Out = 8, 16, 4
    model, wg, wu, wd = _swiglu_mlp_model(K=K, H=H, Out=Out)

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
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


def test_structured_pruning_gated_ffn_prunes_both_branches_to_same_channels():
    # The real bug this pattern risks: gate and up disagreeing on which
    # channels survive, which would silently break the elementwise
    # product's alignment. Assert they select the identical index set,
    # not just that both shrank to the same *count*.
    K, H, Out = 8, 20, 4
    model, wg, wu, _ = _swiglu_mlp_model(K=K, H=H, Out=Out, seed=1)

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.3)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    keep = _combined_keep_indices(wg, wu, H - round(H * 0.3))

    np.testing.assert_array_equal(inits["Wg"], wg[:, keep])
    np.testing.assert_array_equal(inits["Wu"], wu[:, keep])


def test_structured_pruning_gelu_gated_ffn_matches_oracle():
    # GeGLU: same gated topology, a different (still-unary) gate activation.
    # Uses Gelu's tanh approximation so the oracle needs no scipy/erf.
    K, H, Out = 8, 16, 4
    rng = np.random.default_rng(11)
    wg = rng.standard_normal((K, H)).astype(np.float32)
    wu = rng.standard_normal((K, H)).astype(np.float32)
    wd = rng.standard_normal((H, Out)).astype(np.float32)
    nodes = [
        onnx.helper.make_node("MatMul", ["X", "Wg"], ["gate"]),
        onnx.helper.make_node("Gelu", ["gate"], ["gate_act"], approximate="tanh"),
        onnx.helper.make_node("MatMul", ["X", "Wu"], ["up"]),
        onnx.helper.make_node("Mul", ["gate_act", "up"], ["h"]),
        onnx.helper.make_node("MatMul", ["h", "Wd"], ["Y"]),
    ]
    model = _model(
        nodes,
        [_vi("X", ["batch", K])],
        [_vi("Y", ["batch", Out])],
        [_f32(wg, "Wg"), _f32(wu, "Wu"), _f32(wd, "Wd")],
    )

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    keep = _combined_keep_indices(wg, wu, H // 2)

    x = rng.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})

    g = x @ wg[:, keep]
    gate = 0.5 * g * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (g + 0.044715 * g**3)))
    up = x @ wu[:, keep]
    y_oracle = (gate * up) @ wd[keep, :]
    np.testing.assert_allclose(y, y_oracle, rtol=1e-4, atol=1e-4)


def test_structured_pruning_ungated_mul_of_two_producers_still_matches_oracle():
    # No activation at all on either branch -- a plain (unactivated) GLU,
    # both Mul operands are raw producer outputs directly.
    K, H, Out = 8, 12, 4
    rng = np.random.default_rng(2)
    w1 = rng.standard_normal((K, H)).astype(np.float32)
    w2 = rng.standard_normal((K, H)).astype(np.float32)
    w3 = rng.standard_normal((H, Out)).astype(np.float32)
    nodes = [
        onnx.helper.make_node("MatMul", ["X", "W1"], ["a"]),
        onnx.helper.make_node("MatMul", ["X", "W2"], ["b"]),
        onnx.helper.make_node("Mul", ["a", "b"], ["h"]),
        onnx.helper.make_node("MatMul", ["h", "W3"], ["Y"]),
    ]
    model = _model(
        nodes,
        [_vi("X", ["batch", K])],
        [_vi("Y", ["batch", Out])],
        [_f32(w1, "W1"), _f32(w2, "W2"), _f32(w3, "W3")],
    )

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    keep = _combined_keep_indices(w1, w2, H // 2)

    x = rng.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    y_oracle = ((x @ w1[:, keep]) * (x @ w2[:, keep])) @ w3[keep, :]
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_structured_pruning_gated_mul_against_constant_scale_is_not_a_gate():
    # Mul(a, constant) is the existing per-channel-scale chain continuation
    # (already covered elsewhere), not a two-producer gated pair -- the
    # constant operand must never be mistaken for a second producer.
    K, H, Out = 8, 12, 4
    rng = np.random.default_rng(3)
    w1 = rng.standard_normal((K, H)).astype(np.float32)
    scale = rng.standard_normal((H,)).astype(np.float32)
    w2 = rng.standard_normal((H, Out)).astype(np.float32)
    nodes = [
        onnx.helper.make_node("MatMul", ["X", "W1"], ["a"]),
        onnx.helper.make_node("Mul", ["a", "Scale"], ["h"]),
        onnx.helper.make_node("MatMul", ["h", "W2"], ["Y"]),
    ]
    model = _model(
        nodes,
        [_vi("X", ["batch", K])],
        [_vi("Y", ["batch", Out])],
        [_f32(w1, "W1"), _f32(scale, "Scale"), _f32(w2, "W2")],
    )

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1"].dims) == [K, H // 2]
    assert list(inits["Scale"].dims) == [H // 2]
    assert list(inits["W2"].dims) == [H // 2, Out]


def test_structured_pruning_gated_ffn_skips_when_a_branch_also_feeds_elsewhere():
    # "up" also feeding a second consumer directly means pruning its
    # channels would silently change what that other consumer sees --
    # must be left completely untouched, same bar as the plain-chain case.
    K, H, Out = 8, 12, 4
    rng = np.random.default_rng(4)
    wg = rng.standard_normal((K, H)).astype(np.float32)
    wu = rng.standard_normal((K, H)).astype(np.float32)
    wd = rng.standard_normal((H, Out)).astype(np.float32)
    wother = rng.standard_normal((H, Out)).astype(np.float32)
    nodes = [
        onnx.helper.make_node("MatMul", ["X", "Wg"], ["gate"]),
        onnx.helper.make_node("Sigmoid", ["gate"], ["gate_act"]),
        onnx.helper.make_node("MatMul", ["X", "Wu"], ["up"]),
        onnx.helper.make_node("Mul", ["gate_act", "up"], ["h"]),
        onnx.helper.make_node("MatMul", ["h", "Wd"], ["Y1"]),
        onnx.helper.make_node("MatMul", ["up", "Wother"], ["Y2"]),
    ]
    model = _model(
        nodes,
        [_vi("X", ["batch", K])],
        [_vi("Y1", ["batch", Out]), _vi("Y2", ["batch", Out])],
        [_f32(wg, "Wg"), _f32(wu, "Wu"), _f32(wd, "Wd"), _f32(wother, "Wother")],
    )

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["Wg"].dims) == [K, H]
    assert list(inits["Wu"].dims) == [K, H]
    assert list(inits["Wd"].dims) == [H, Out]


def test_structured_pruning_native_swiglu_node_prunes_both_producers_together():
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
    nodes = [
        onnx.helper.make_node("MatMul", ["X", "Wg"], ["gate"]),
        onnx.helper.make_node("MatMul", ["X", "Wu"], ["up"]),
        onnx.helper.make_node("SwiGLU", ["gate", "up"], ["h"]),
        onnx.helper.make_node("MatMul", ["h", "Wd"], ["Y"]),
    ]
    model = _model(
        nodes,
        [_vi("X", ["batch", K])],
        [_vi("Y", ["batch", Out])],
        [_f32(wg, "Wg"), _f32(wu, "Wu"), _f32(wd, "Wd")],
        opset=28,
    )

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    keep = _combined_keep_indices(wg, wu, H // 2)

    np.testing.assert_array_equal(inits["Wg"], wg[:, keep])
    np.testing.assert_array_equal(inits["Wu"], wu[:, keep])
    np.testing.assert_array_equal(inits["Wd"], wd[keep, :])


# --- apply_structured_wanda_pruning ------------------------------------------


def _kept_columns(pruned_model, weight_name, original_w):
    w_pruned = onnx.numpy_helper.to_array(
        next(t for t in pruned_model.graph.initializer if t.name == weight_name)
    )
    kept = []
    for j in range(original_w.shape[1]):
        col = original_w[:, j]
        if any(np.array_equal(col, w_pruned[:, jj]) for jj in range(w_pruned.shape[1])):
            kept.append(j)
    return kept


def test_structured_wanda_pruning_protects_channels_with_small_weight_but_large_activation():
    # A structured analogue of Wanda's own motivating scenario: a hidden
    # unit whose own weight column is deliberately *smaller* than typical
    # (so plain L2-norm structured pruning ranks it lowest and cuts it),
    # but which is wired to an input feature that calibration data makes
    # consistently huge -- its actual contribution to the network is large
    # even though its weight norm alone doesn't show it.
    K, H, Out = 8, 32, 4
    salient = (3, 7, 20)
    k0 = 0
    rng = np.random.default_rng(20)
    w1 = rng.standard_normal((K, H)).astype(np.float32) * 0.5
    w2 = rng.standard_normal((H, Out)).astype(np.float32)
    non_salient = [j for j in range(H) if j not in salient]
    w1[k0, non_salient] = 0.0  # only salient channels respond to k0 at all
    small_scale = 0.4
    for j in salient:
        w1[:, j] = 0.0
        w1[k0, j] = small_scale  # weight norm well below the ~1.4 typical column

    nodes = [
        onnx.helper.make_node("MatMul", ["X", "W1"], ["h"]),
        onnx.helper.make_node("Relu", ["h"], ["a"]),
        onnx.helper.make_node("MatMul", ["a", "W2"], ["Y"]),
    ]
    model = _model(
        nodes,
        [_vi("X", ["batch", K])],
        [_vi("Y", ["batch", Out])],
        [_f32(w1, "W1"), _f32(w2, "W2")],
    )

    x = rng.standard_normal((64, K)).astype(np.float32)
    x[:, k0] *= 40.0
    calibration_data = [{"X": x}]

    plain = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    wanda = onnxsim.apply_structured_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(plain)
    onnx.checker.check_model(wanda)

    plain_kept = _kept_columns(plain, "W1", w1)
    wanda_kept = _kept_columns(wanda, "W1", w1)
    assert all(j not in plain_kept for j in salient)
    assert all(j in wanda_kept for j in salient)


def test_structured_wanda_pruning_matches_oracle_exactly():
    K, H, Out = 8, 24, 4
    rng = np.random.default_rng(21)
    w1 = rng.standard_normal((K, H)).astype(np.float32)
    w2 = rng.standard_normal((H, Out)).astype(np.float32)
    nodes = [
        onnx.helper.make_node("MatMul", ["X", "W1"], ["h"]),
        onnx.helper.make_node("Relu", ["h"], ["a"]),
        onnx.helper.make_node("MatMul", ["a", "W2"], ["Y"]),
    ]
    model = _model(
        nodes,
        [_vi("X", ["batch", "seq", K])],
        [_vi("Y", ["batch", "seq", Out])],
        [_f32(w1, "W1"), _f32(w2, "W2")],
    )

    rng_cal = np.random.default_rng(22)
    x_cal = rng_cal.standard_normal((2, 16, K)).astype(np.float32)
    calibration_data = [{"X": x_cal}]

    pruned = onnxsim.apply_structured_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned)

    a_cal = np.maximum(x_cal.reshape(-1, K) @ w1, 0)
    act_norm = np.sqrt(np.mean(np.square(a_cal), axis=0))
    importance = np.linalg.norm(w1.T, axis=1) * np.maximum(act_norm, 1e-8)
    keep = np.sort(np.argsort(-importance)[: H // 2])

    x = rng_cal.standard_normal((3, 5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    h = np.maximum(x @ w1[:, keep], 0)
    y_oracle = h @ w2[keep, :]
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_structured_wanda_pruning_falls_back_to_plain_with_no_calibration_batches():
    K, H, Out = 8, 16, 4
    model = _mlp_model(K=K, H=H, Out=Out, bias=False)

    plain = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    wanda = onnxsim.apply_structured_wanda_pruning(
        model, calibration_data=[], sparsity=0.5
    )
    inits_plain = {
        t.name: onnx.numpy_helper.to_array(t) for t in plain.graph.initializer
    }
    inits_wanda = {
        t.name: onnx.numpy_helper.to_array(t) for t in wanda.graph.initializer
    }
    np.testing.assert_array_equal(inits_plain["W1"], inits_wanda["W1"])
    np.testing.assert_array_equal(inits_plain["W2"], inits_wanda["W2"])


def test_structured_wanda_pruning_gated_ffn_matches_oracle():
    K, H, Out = 8, 16, 4
    model, wg, wu, wd = _swiglu_mlp_model(K=K, H=H, Out=Out, seed=23)

    rng = np.random.default_rng(24)
    x_cal = rng.standard_normal((32, K)).astype(np.float32)
    calibration_data = [{"X": x_cal}]

    pruned = onnxsim.apply_structured_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    assert inits["Wg"].shape == (K, H // 2)
    assert inits["Wu"].shape == (K, H // 2)
    assert inits["Wd"].shape == (H // 2, Out)

    # Both branches must still select the identical channel-index set.
    kept_g = _kept_columns(pruned, "Wg", wg)
    kept_u = _kept_columns(pruned, "Wu", wu)
    assert kept_g == kept_u

    x = rng.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    keep = kept_g
    gate = 1.0 / (1.0 + np.exp(-(x @ wg[:, keep])))
    up = x @ wu[:, keep]
    y_oracle = (gate * up) @ wd[keep, :]
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_structured_wanda_pruning_zero_sparsity_is_a_no_op():
    model = _mlp_model(K=8, H=16, Out=4)
    pruned = onnxsim.apply_structured_wanda_pruning(model, sparsity=0.0)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1"].dims) == [8, 16]


def test_structured_wanda_pruning_invalid_sparsity_raises():
    model = _mlp_model(K=8, H=16, Out=4)
    with pytest.raises(ValueError):
        onnxsim.apply_structured_wanda_pruning(model, sparsity=1.0)
    with pytest.raises(ValueError):
        onnxsim.apply_structured_wanda_pruning(model, sparsity=-0.1)
