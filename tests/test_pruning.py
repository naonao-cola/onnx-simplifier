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


def _mlp_model(K=8, H=32, O=4, bias=True, activation="Relu", seed=0):
    rng = np.random.default_rng(seed)
    w1 = rng.standard_normal((K, H)).astype(np.float32)
    w2 = rng.standard_normal((H, O)).astype(np.float32)
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
        nodes, [_vi("X", ["batch", K])], [_vi("Y", ["batch", O])], initializer
    )


def _oracle_keep_indices(w1, keep_count):
    importance = np.linalg.norm(w1.T, axis=1)
    return np.sort(np.argsort(-importance)[:keep_count])


def test_structured_pruning_shrinks_matched_layers():
    model = _mlp_model(K=8, H=32, O=4)
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
    K, H, O = 8, 32, 4
    model = _mlp_model(K=K, H=H, O=O, bias=True)
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
    K, H, O = 8, 24, 4
    model = _mlp_model(K=K, H=H, O=O, bias=False, activation="Sigmoid")
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
    K, H, O = 8, 16, 4
    rng = np.random.default_rng(3)
    w1 = rng.standard_normal((K, H)).astype(np.float32)
    bias = rng.standard_normal((H,)).astype(np.float32)
    w2 = rng.standard_normal((H, O)).astype(np.float32)
    nodes = [
        onnx.helper.make_node("MatMul", ["X", "W1"], ["h"]),
        onnx.helper.make_node("Add", ["h", "Bias"], ["hb"]),
        onnx.helper.make_node("Relu", ["hb"], ["a"]),
        onnx.helper.make_node("MatMul", ["a", "W2"], ["Y"]),
    ]
    model = _model(
        nodes,
        [_vi("X", ["batch", K])],
        [_vi("Y", ["batch", O])],
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
    K, H, O = 8, 16, 4
    model = _mlp_model(K=K, H=H, O=O, bias=False)
    graph = model.graph
    graph.output.append(_vi("h", ["batch", H]))

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1"].dims) == [K, H]
    assert list(inits["W2"].dims) == [H, O]


def test_structured_pruning_skips_multi_consumer_branch():
    # h feeds two separate downstream MatMuls -- not the single-consumer
    # chain this pass proves safe to cut, so it must be left untouched.
    K, H, O = 8, 16, 4
    rng = np.random.default_rng(4)
    w1 = rng.standard_normal((K, H)).astype(np.float32)
    w2 = rng.standard_normal((H, O)).astype(np.float32)
    w3 = rng.standard_normal((H, O)).astype(np.float32)
    nodes = [
        onnx.helper.make_node("MatMul", ["X", "W1"], ["h"]),
        onnx.helper.make_node("MatMul", ["h", "W2"], ["Y1"]),
        onnx.helper.make_node("MatMul", ["h", "W3"], ["Y2"]),
    ]
    model = _model(
        nodes,
        [_vi("X", ["batch", K])],
        [_vi("Y1", ["batch", O]), _vi("Y2", ["batch", O])],
        [_f32(w1, "W1"), _f32(w2, "W2"), _f32(w3, "W3")],
    )

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1"].dims) == [K, H]


def test_structured_pruning_zero_sparsity_is_a_no_op():
    model = _mlp_model(K=8, H=16, O=4)
    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.0)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1"].dims) == [8, 16]


def test_structured_pruning_invalid_sparsity_raises():
    model = _mlp_model(K=8, H=16, O=4)
    with pytest.raises(ValueError):
        onnxsim.apply_structured_pruning(model, sparsity=1.0)
    with pytest.raises(ValueError):
        onnxsim.apply_structured_pruning(model, sparsity=-0.1)


def test_structured_pruning_chains_through_a_third_layer():
    # W2 is a producer for one chain (its own output channels feeding W3)
    # and a consumer for another (W1's output channels feeding into it) --
    # independent axes of the same tensor, both must be pruned correctly.
    K, H1, H2, O = 8, 16, 20, 4
    rng = np.random.default_rng(5)
    w1 = rng.standard_normal((K, H1)).astype(np.float32)
    w2 = rng.standard_normal((H1, H2)).astype(np.float32)
    w3 = rng.standard_normal((H2, O)).astype(np.float32)
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
        [_vi("Y", ["batch", O])],
        [_f32(w1, "W1"), _f32(w2, "W2"), _f32(w3, "W3")],
    )

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1"].dims) == [K, H1 // 2]
    assert list(inits["W2"].dims) == [H1 // 2, H2 // 2]
    assert list(inits["W3"].dims) == [H2 // 2, O]

    keep1 = _oracle_keep_indices(w1, H1 // 2)
    keep2 = _oracle_keep_indices(w2[keep1, :], H2 // 2)

    rng2 = np.random.default_rng(6)
    x = rng2.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    a1 = np.maximum(x @ w1[:, keep1], 0)
    a2 = np.maximum(a1 @ w2[np.ix_(keep1, keep2)], 0)
    y_oracle = a2 @ w3[keep2, :]
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)
