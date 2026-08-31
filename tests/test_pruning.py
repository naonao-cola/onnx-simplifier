"""Tests for ``onnxsim.pruning`` -- magnitude pruning (data-free baseline),
Wanda pruning (calibrated on activation norms), and structured (channel)
pruning, see ``onnxsim/pruning.py``.
"""

import ml_dtypes
import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper
import onnx.shape_inference
import pytest
from onnx import parser

import onnxsim

ort = pytest.importorskip("onnxruntime")


def _model(body, initializer=(), opset=21):
    # Pinning ir_version: 10 matches the older onnxruntime bundled with some
    # CI wheels (which cap at IR version 11); `_run` and onnxsim's own
    # checks below run these models through onnxruntime.
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


def _matmul_model(K=64, N=16, seed=0):
    rng = np.random.default_rng(seed)
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    return _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f32(weight, "W")],
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
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f32(weight, "W")],
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
    model = _model(
        f"""
        g (float[batch,seq,{K}] X) => (float[batch,seq,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f32(weight, "W")],
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
    model = _model(
        """
        g (float[4] X) => (float[4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    assert onnxsim.weight_sparsity(model) == 0.0


# --- magnitude/Wanda pruning: global_sparsity ----------------------------


def _two_scale_matmul_model(K=16, N=8, big_scale=100.0, small_scale=1.0, seed=0):
    # Two independent MatMul layers sharing one input, deliberately built at
    # very different weight-magnitude scales -- the adversarial case
    # `global_sparsity` exists for: `apply_magnitude_pruning`'s own
    # per-layer-uniform mode cuts both to the same *fraction* regardless of
    # scale, while `global_sparsity` should redistribute toward the
    # uniformly-smaller layer.
    rng = np.random.default_rng(seed)
    w_big = (rng.standard_normal((K, N)) * big_scale).astype(np.float32)
    w_small = (rng.standard_normal((K, N)) * small_scale).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y1, float[batch,{N}] Y2)
        {{
          Y1 = MatMul(X, Wbig)
          Y2 = MatMul(X, Wsmall)
        }}
        """,
        initializer=[_f32(w_big, "Wbig"), _f32(w_small, "Wsmall")],
    )
    return model, w_big, w_small


def test_magnitude_pruning_global_sparsity_redistributes_toward_small_magnitude_layer():
    K, N = 16, 8
    model, _, _ = _two_scale_matmul_model(K=K, N=N, big_scale=100.0, small_scale=1.0)

    local_pruned = onnxsim.apply_magnitude_pruning(model, sparsity=0.5)
    global_pruned = onnxsim.apply_magnitude_pruning(
        model, sparsity=0.5, global_sparsity=True
    )
    onnx.checker.check_model(global_pruned)

    inits_local = {t.name: t for t in local_pruned.graph.initializer}
    inits_global = {t.name: t for t in global_pruned.graph.initializer}

    def sparsity_of(inits, name):
        return float(np.mean(onnx.numpy_helper.to_array(inits[name]) == 0))

    # Per-layer-uniform (default) mode: both layers cut to exactly the same
    # fraction, regardless of scale.
    assert sparsity_of(inits_local, "Wbig") == pytest.approx(0.5, abs=1e-9)
    assert sparsity_of(inits_local, "Wsmall") == pytest.approx(0.5, abs=1e-9)

    # global_sparsity mode: the uniformly-100x-larger layer must be pruned
    # markedly less than the uniformly-small one -- the whole point of
    # pooling importance across layers instead of treating each layer's own
    # distribution in isolation.
    big_sparsity = sparsity_of(inits_global, "Wbig")
    small_sparsity = sparsity_of(inits_global, "Wsmall")
    assert big_sparsity < 0.5 < small_sparsity

    # Aggregate sparsity across both matched layers still hits the
    # requested global target exactly -- no per-row/per-layer floor to
    # introduce slack in this unstructured mode (see
    # apply_magnitude_pruning's own `global_sparsity` docstring).
    assert onnxsim.weight_sparsity(global_pruned) == pytest.approx(0.5, abs=1e-9)


def test_magnitude_pruning_global_sparsity_matches_pooled_threshold_oracle():
    K, N = 12, 6
    sparsity = 0.6
    model, w_big, w_small = _two_scale_matmul_model(
        K=K, N=N, big_scale=50.0, small_scale=0.3, seed=3
    )

    pruned = onnxsim.apply_magnitude_pruning(
        model, sparsity=sparsity, global_sparsity=True
    )
    inits = {t.name: t for t in pruned.graph.initializer}

    # Hand-built oracle: pool |W| across both layers' own [N, K]
    # (output-channel-first) entries, in the same node order `_candidates`
    # matches them in (program order here), rank globally, and zero the
    # lowest-scoring round(total * sparsity) entries -- a from-scratch
    # reimplementation of `_apply_global_unstructured_pruning`, not a call
    # into it.
    w_big_nk = w_big.T.astype(np.float64)  # [N, K]
    w_small_nk = w_small.T.astype(np.float64)
    pooled = np.concatenate(
        [np.abs(w_big_nk).reshape(-1), np.abs(w_small_nk).reshape(-1)]
    )
    total = pooled.size
    keep_count = round(total * (1.0 - sparsity))
    drop_count = total - keep_count
    order = np.argsort(pooled, kind="stable")
    drop_flat = np.zeros(total, dtype=bool)
    drop_flat[order[:drop_count]] = True

    big_drop = drop_flat[: w_big_nk.size].reshape(w_big_nk.shape)
    small_drop = drop_flat[w_big_nk.size :].reshape(w_small_nk.shape)
    expected_big = np.where(big_drop, 0.0, w_big_nk).T.astype(np.float32)
    expected_small = np.where(small_drop, 0.0, w_small_nk).T.astype(np.float32)

    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(inits["Wbig"]), expected_big
    )
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(inits["Wsmall"]), expected_small
    )


def test_magnitude_pruning_global_sparsity_rejects_nm():
    model = _matmul_model(K=16, N=4)
    with pytest.raises(ValueError):
        onnxsim.apply_magnitude_pruning(model, n=2, m=4, global_sparsity=True)


def _two_input_matmul_model(K=16, N=8, seed=0):
    # Two independent MatMul layers fed by *separate* graph inputs (rather
    # than sharing one, as `_two_scale_matmul_model` does) so calibration
    # data can give each layer's own activation a different magnitude while
    # both layers' weights stay the same scale -- isolates Wanda's own
    # ``||X_j||_2`` half of its importance metric from the plain
    # weight-magnitude effect `_two_scale_matmul_model` exercises.
    rng = np.random.default_rng(seed)
    w1 = rng.standard_normal((K, N)).astype(np.float32)
    w2 = rng.standard_normal((K, N)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X1, float[batch,{K}] X2) => (float[batch,{N}] Y1, float[batch,{N}] Y2)
        {{
          Y1 = MatMul(X1, W1)
          Y2 = MatMul(X2, W2)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2")],
    )
    return model, w1, w2


def test_wanda_pruning_global_sparsity_redistributes_by_combined_metric():
    K, N = 16, 8
    model, _, _ = _two_input_matmul_model(K=K, N=N, seed=5)
    rng = np.random.default_rng(9)
    x1 = (rng.standard_normal((32, K)) * 50.0).astype(np.float32)
    x2 = (rng.standard_normal((32, K)) * 1.0).astype(np.float32)
    calibration_data = [{"X1": x1, "X2": x2}]

    pruned = onnxsim.apply_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.5, global_sparsity=True
    )
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    sparsity1 = float(np.mean(onnx.numpy_helper.to_array(inits["W1"]) == 0))
    sparsity2 = float(np.mean(onnx.numpy_helper.to_array(inits["W2"]) == 0))

    # W1 sees a 50x larger activation than W2 despite the same weight
    # scale -- global Wanda importance must protect it accordingly.
    assert sparsity1 < 0.5 < sparsity2
    assert onnxsim.weight_sparsity(pruned) == pytest.approx(0.5, abs=1e-9)


def test_wanda_pruning_global_sparsity_matches_pooled_threshold_oracle():
    K, N = 10, 5
    sparsity = 0.55
    model, w1, w2 = _two_input_matmul_model(K=K, N=N, seed=11)
    rng = np.random.default_rng(13)
    x1 = (rng.standard_normal((16, K)) * 20.0).astype(np.float32)
    x2 = (rng.standard_normal((16, K)) * 3.0).astype(np.float32)
    calibration_data = [{"X1": x1, "X2": x2}]

    pruned = onnxsim.apply_wanda_pruning(
        model,
        calibration_data=calibration_data,
        sparsity=sparsity,
        global_sparsity=True,
    )
    inits = {t.name: t for t in pruned.graph.initializer}

    # Hand-built oracle: the same |W_ij| * ||X_j||_2 formula
    # apply_wanda_pruning itself uses, computed directly from the fixed
    # calibration input (MatMul(X, W)'s own probed activation *is* X,
    # unchanged by any op upstream of it -- no need to re-run onnxruntime),
    # pooled the same way _apply_global_unstructured_pruning pools any
    # per-entry importance metric.
    norm1 = np.sqrt(np.mean(np.square(x1.astype(np.float64)), axis=0))  # [K]
    norm2 = np.sqrt(np.mean(np.square(x2.astype(np.float64)), axis=0))
    w1_nk = w1.T.astype(np.float64)  # [N, K]
    w2_nk = w2.T.astype(np.float64)
    importance1 = np.abs(w1_nk) * np.maximum(norm1[np.newaxis, :], 1e-8)
    importance2 = np.abs(w2_nk) * np.maximum(norm2[np.newaxis, :], 1e-8)
    pooled = np.concatenate([importance1.reshape(-1), importance2.reshape(-1)])
    total = pooled.size
    keep_count = round(total * (1.0 - sparsity))
    drop_count = total - keep_count
    order = np.argsort(pooled, kind="stable")
    drop_flat = np.zeros(total, dtype=bool)
    drop_flat[order[:drop_count]] = True
    drop1 = drop_flat[: importance1.size].reshape(importance1.shape)
    drop2 = drop_flat[importance1.size :].reshape(importance2.shape)
    expected1 = np.where(drop1, 0.0, w1_nk).T.astype(np.float32)
    expected2 = np.where(drop2, 0.0, w2_nk).T.astype(np.float32)

    np.testing.assert_array_equal(onnx.numpy_helper.to_array(inits["W1"]), expected1)
    np.testing.assert_array_equal(onnx.numpy_helper.to_array(inits["W2"]), expected2)


def test_wanda_pruning_global_sparsity_rejects_nm():
    model = _matmul_model(K=16, N=4)
    with pytest.raises(ValueError):
        onnxsim.apply_wanda_pruning(model, n=2, m=4, global_sparsity=True)


# --- magnitude/Wanda pruning: Conv2D -------------------------------------


def _single_conv_model(w, spatial=10, extra_attrs="", out_spatial=None, group=1):
    # `w`'s shape must already be [Cout, Cin/group, kH, kW] when `group` > 1
    # -- same caller-responsibility convention `_grouped_conv_pair_model`
    # below has.
    Cout, Cin_per_group, kh, kw = w.shape
    Cin = Cin_per_group * group
    if out_spatial is None:
        out_spatial = spatial - kh + 1  # no padding, unit stride
    attrs = f"kernel_shape=[{kh},{kw}]"
    if group != 1:
        attrs += f", group={group}"
    if extra_attrs:
        attrs += ", " + extra_attrs
    return _model(
        f"""
        g (float[N,{Cin},{spatial},{spatial}] X) => (float[N,{Cout},{out_spatial},{out_spatial}] Y)
        {{
          Y = Conv<{attrs}>(X, W1)
        }}
        """,
        initializer=[_f32(w, "W1")],
    )


def _conv_weight(model):
    return onnx.numpy_helper.to_array(model.graph.initializer[0])


def test_magnitude_pruning_conv_reaches_target_sparsity():
    Cin, Cout = 4, 8  # K = Cin*3*3 = 36, evenly halved by sparsity=0.5
    rng = np.random.default_rng(60)
    w = rng.standard_normal((Cout, Cin, 3, 3)).astype(np.float32)
    model = _single_conv_model(w)

    pruned = onnxsim.apply_magnitude_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    assert onnxsim.weight_sparsity(pruned) == pytest.approx(0.5, abs=1e-9)
    assert _conv_weight(pruned).shape == w.shape


def test_magnitude_pruning_conv_keeps_the_largest_entries_per_filter():
    Cin, Cout = 4, 8
    rng = np.random.default_rng(61)
    w = rng.standard_normal((Cout, Cin, 3, 3)).astype(np.float32) * 0.5
    model = _single_conv_model(w)

    pruned = onnxsim.apply_magnitude_pruning(model, sparsity=0.75)
    K = Cin * 3 * 3
    w_flat = w.astype(np.float64).reshape(Cout, K)
    w_pruned_flat = _conv_weight(pruned).astype(np.float64).reshape(Cout, K)
    keep_count = round(K * 0.25)
    for row in range(Cout):
        kept = np.flatnonzero(w_pruned_flat[row] != 0)
        assert len(kept) == keep_count
        threshold = np.abs(w_flat[row])[kept].min()
        dropped_max = np.abs(w_flat[row])[np.flatnonzero(w_pruned_flat[row] == 0)].max()
        assert dropped_max <= threshold


def test_magnitude_pruning_conv_nm_pattern():
    Cin, Cout = 4, 8
    rng = np.random.default_rng(62)
    w = rng.standard_normal((Cout, Cin, 3, 3)).astype(np.float32)
    model = _single_conv_model(w)

    pruned = onnxsim.apply_magnitude_pruning(model, n=2, m=4)
    K = Cin * 3 * 3
    w_flat = _conv_weight(pruned).reshape(Cout, K)
    assert onnxsim.weight_sparsity(pruned) == pytest.approx(0.5, abs=1e-9)
    for row in w_flat:
        for start in range(0, len(row), 4):
            group = row[start : start + 4]
            assert np.count_nonzero(group) <= 2


def test_magnitude_pruning_conv_depthwise_reaches_target_sparsity_and_matches_oracle():
    # Unlike the earlier restriction this module used to draw (mirroring
    # structured pruning's producer/consumer channel-index coupling
    # problem, which does not actually apply to unstructured pruning at
    # all -- see this module's own docstring and
    # :func:`_match_conv_weight_only`'s), a depthwise Conv (group ==
    # in_channels == out_channels) is now pruned per-filter exactly like an
    # ordinary Conv's own filters: each of its `C` single-input-channel
    # filters is its own independent comparison group.
    C = 8
    rng = np.random.default_rng(63)
    w = rng.standard_normal((C, 1, 4, 4)).astype(np.float32)  # K=16, halved exactly
    model = _single_conv_model(w, spatial=10, group=C)

    pruned = onnxsim.apply_magnitude_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    assert onnxsim.weight_sparsity(pruned) == pytest.approx(0.5, abs=1e-9)
    w_pruned = _conv_weight(pruned)
    assert w_pruned.shape == w.shape
    # group/kernel_shape attributes (and hence output shape) are untouched
    # -- this is a value-only rewrite.
    conv_node = pruned.graph.node[0]
    assert next(a.i for a in conv_node.attribute if a.name == "group") == C

    K = 1 * 4 * 4  # each filter's own single-channel kernel
    w_flat = w.astype(np.float64).reshape(C, K)
    keep = round(K * 0.5)
    order = np.argsort(np.abs(w_flat), axis=1)
    mask = np.ones((C, K), dtype=bool)
    np.put_along_axis(mask, order[:, : K - keep], False, axis=1)
    oracle_w = np.where(mask, w_flat, 0.0).reshape(w.shape).astype(np.float32)
    np.testing.assert_array_equal(w_pruned, oracle_w)

    oracle = _single_conv_model(oracle_w, spatial=10, group=C)
    rng_x = np.random.default_rng(64)
    x = rng_x.standard_normal((2, C, 10, 10)).astype(np.float32)
    (y_pruned,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y_pruned, y_oracle, rtol=1e-5, atol=1e-5)


def test_magnitude_pruning_conv_general_grouped_reaches_target_sparsity_and_matches_oracle():
    # A general grouped Conv (1 < group < in_channels): each of the 8
    # output filters -- 4 per group -- is still its own independent
    # per-filter comparison group, exactly like the group=1 case; `group`
    # only changes what each filter's own [Cin/group, kH, kW] kernel
    # covers, never how filters are ranked against each other.
    Cin, Cout, group = 8, 8, 2  # K = (Cin/group)*3*3 = 36
    rng = np.random.default_rng(65)
    w = rng.standard_normal((Cout, Cin // group, 3, 3)).astype(np.float32)
    model = _single_conv_model(w, spatial=10, group=group)

    pruned = onnxsim.apply_magnitude_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    assert onnxsim.weight_sparsity(pruned) == pytest.approx(0.5, abs=1e-9)
    w_pruned = _conv_weight(pruned)
    assert w_pruned.shape == w.shape
    conv_node = pruned.graph.node[0]
    assert next(a.i for a in conv_node.attribute if a.name == "group") == group

    K = (Cin // group) * 3 * 3
    w_flat = w.astype(np.float64).reshape(Cout, K)
    keep = round(K * 0.5)
    order = np.argsort(np.abs(w_flat), axis=1)
    mask = np.ones((Cout, K), dtype=bool)
    np.put_along_axis(mask, order[:, : K - keep], False, axis=1)
    oracle_w = np.where(mask, w_flat, 0.0).reshape(w.shape).astype(np.float32)
    np.testing.assert_array_equal(w_pruned, oracle_w)

    oracle = _single_conv_model(oracle_w, spatial=10, group=group)
    rng_x = np.random.default_rng(66)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y_pruned,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y_pruned, y_oracle, rtol=1e-5, atol=1e-5)


def test_magnitude_pruning_skips_malformed_grouped_conv():
    # out_channels % group != 0 -- not a valid grouped-Conv shape (`group`
    # equal-sized output blocks is required, the same well-formedness
    # :func:`_match_conv_producer` already requires for structured pruning)
    # -- :func:`_match_conv_weight_only` declines it rather than guessing.
    Cout, group = 6, 4  # 6 % 4 != 0; Cin/group = 1 (Cin=4) is otherwise valid
    rng = np.random.default_rng(67)
    w = rng.standard_normal((Cout, 1, 3, 3)).astype(np.float32)
    model = _model(
        f"""
        g (float[N,4,10,10] X) => (float[N,{Cout},8,8] Y)
        {{
          Y = Conv<kernel_shape=[3,3], group={group}>(X, W1)
        }}
        """,
        initializer=[_f32(w, "W1")],
    )
    pruned = onnxsim.apply_magnitude_pruning(model, sparsity=0.5)
    np.testing.assert_array_equal(_conv_weight(pruned), w)


def _naive_conv_patch_sq_sum(x, attrs):
    # Slow, obviously-correct nested-loop reference for the per-``(in_
    # channel, kh, kw)`` offset activation statistic Wanda needs for Conv
    # -- the oracle :func:`onnxsim.pruning._conv_patch_sq_sum`'s own
    # ``sliding_window_view``-based implementation is checked against
    # below, before it's ever trusted inside the actual pruning path.
    n, cin, h, w = x.shape
    xp = np.pad(
        x,
        (
            (0, 0),
            (0, 0),
            (attrs.pad_top, attrs.pad_bottom),
            (attrs.pad_left, attrs.pad_right),
        ),
    )
    hp, wp = xp.shape[2], xp.shape[3]
    h_out = (hp - attrs.kh) // attrs.stride_h + 1
    w_out = (wp - attrs.kw) // attrs.stride_w + 1
    sq = np.zeros((cin, attrs.kh, attrs.kw), dtype=np.float64)
    count = 0
    for ni in range(n):
        for oh in range(h_out):
            for ow in range(w_out):
                count += 1
                for c in range(cin):
                    for i in range(attrs.kh):
                        for j in range(attrs.kw):
                            val = xp[
                                ni, c, oh * attrs.stride_h + i, ow * attrs.stride_w + j
                            ]
                            sq[c, i, j] += val * val
    return sq, count


def test_conv_patch_sq_sum_matches_naive_nested_loop_oracle():
    from onnxsim.pruning import _conv_patch_sq_sum, _ConvSpatialAttrs

    rng = np.random.default_rng(70)
    x = rng.standard_normal((2, 3, 4, 4))
    cases = [
        _ConvSpatialAttrs(
            kh=3,
            kw=3,
            pad_top=1,
            pad_left=1,
            pad_bottom=1,
            pad_right=1,
            stride_h=2,
            stride_w=2,
        ),
        _ConvSpatialAttrs(
            kh=2,
            kw=2,
            pad_top=0,
            pad_left=0,
            pad_bottom=0,
            pad_right=0,
            stride_h=1,
            stride_w=1,
        ),
        _ConvSpatialAttrs(
            kh=3,
            kw=3,
            pad_top=0,
            pad_left=2,
            pad_bottom=1,
            pad_right=0,
            stride_h=1,
            stride_w=2,
        ),
    ]
    for attrs in cases:
        sq_vec, count_vec = _conv_patch_sq_sum(x, attrs)
        sq_naive, count_naive = _naive_conv_patch_sq_sum(x, attrs)
        assert count_vec == count_naive
        np.testing.assert_allclose(sq_vec, sq_naive)


def test_wanda_pruning_conv_matches_manual_im2col_importance_oracle_exactly():
    # Same correctness bar as the MatMul/Gemm Wanda tests, but the oracle's
    # activation norm is computed by manually unfolding X into overlapping
    # kh*kw patches (a second, independent im2col implementation from the
    # one under test, see :func:`onnxsim.pruning._conv_patch_sq_sum`) and
    # reducing over batch and every output position.
    Cin, Cout, kh, kw, spatial = 3, 6, 3, 3, 8
    rng = np.random.default_rng(72)
    w = rng.standard_normal((Cout, Cin, kh, kw)).astype(np.float32)
    model = _single_conv_model(w, spatial=spatial)

    rng_x = np.random.default_rng(73)
    x = rng_x.standard_normal((3, Cin, spatial, spatial)).astype(np.float32)

    pruned = onnxsim.apply_wanda_pruning(
        model, calibration_data=[{"X": x}], sparsity=0.5
    )
    onnx.checker.check_model(pruned)

    out_spatial = spatial - kh + 1
    K = Cin * kh * kw
    patches = np.zeros((x.shape[0] * out_spatial * out_spatial, K), dtype=np.float64)
    idx = 0
    for ni in range(x.shape[0]):
        for oh in range(out_spatial):
            for ow in range(out_spatial):
                patch = x[ni, :, oh : oh + kh, ow : ow + kw].astype(np.float64)
                patches[idx] = patch.reshape(-1)
                idx += 1
    act_norm = np.sqrt(np.mean(np.square(patches), axis=0))

    w_flat = w.astype(np.float64).reshape(Cout, K)
    importance = np.abs(w_flat) * act_norm[np.newaxis, :]
    keep = round(K * 0.5)
    order = np.argsort(importance, axis=1)
    drop = order[:, : K - keep]
    mask = np.ones((Cout, K), dtype=bool)
    np.put_along_axis(mask, drop, False, axis=1)
    expected = np.where(mask, w_flat, 0.0).reshape(Cout, Cin, kh, kw)

    np.testing.assert_allclose(_conv_weight(pruned).astype(np.float64), expected)


def test_wanda_pruning_conv_protects_high_activation_channel():
    # Same motivating scenario as the MatMul Wanda test: one input channel
    # carries much larger activation magnitude than the rest, but a
    # merely-average weight magnitude -- Wanda's per-offset activation
    # norm (rolled up here to a whole input channel, since every (kh, kw)
    # offset of that channel gets boosted identically) should protect it
    # more than plain |W| magnitude pruning does.
    Cin, Cout, spatial = 3, 8, 10
    salient_channel = 1
    rng = np.random.default_rng(71)
    w = rng.standard_normal((Cout, Cin, 3, 3)).astype(np.float32) * 0.5
    model = _single_conv_model(w, spatial=spatial)

    x = rng.standard_normal((4, Cin, spatial, spatial)).astype(np.float32)
    x[:, salient_channel, :, :] *= 20.0
    calibration_data = [{"X": x}]

    magnitude_pruned = onnxsim.apply_magnitude_pruning(model, sparsity=0.5)
    wanda_pruned = onnxsim.apply_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(wanda_pruned)

    w_magnitude = _conv_weight(magnitude_pruned)
    w_wanda = _conv_weight(wanda_pruned)
    salient_kept_magnitude = np.count_nonzero(w_magnitude[:, salient_channel, :, :])
    salient_kept_wanda = np.count_nonzero(w_wanda[:, salient_channel, :, :])
    assert salient_kept_wanda > salient_kept_magnitude

    (float_y,) = _run(model, {"X": x})
    (magnitude_y,) = _run(magnitude_pruned, {"X": x})
    (wanda_y,) = _run(wanda_pruned, {"X": x})
    magnitude_err = np.linalg.norm(float_y.astype(np.float64) - magnitude_y)
    wanda_err = np.linalg.norm(float_y.astype(np.float64) - wanda_y)
    assert wanda_err < magnitude_err


def test_wanda_pruning_conv_auto_pad_matches_manual_im2col_importance_oracle_exactly():
    # An earlier version of this module declined auto_pad entirely
    # (_conv_spatial_attrs used to return None for any non-"NOTSET" value)
    # and fell back to plain magnitude. auto_pad's own padding is now
    # resolved per calibration batch from the input's own spatial size
    # (_resolve_conv_pads, per the ONNX Conv operator's own auto_pad
    # formula) instead. kh=3, stride=2, spatial=8 is deliberately chosen so
    # SAME_UPPER's own pad_total = 1 is *odd*, making the resolved padding
    # asymmetric (pad_top=0, pad_bottom=1): a bug putting the extra pixel
    # on the wrong edge, or splitting it evenly, would silently shift every
    # captured patch and diverge from this independently-padded oracle.
    Cin, Cout, kh, kw, spatial, stride = 3, 6, 3, 3, 8, 2
    rng = np.random.default_rng(101)
    w = rng.standard_normal((Cout, Cin, kh, kw)).astype(np.float32)
    out_spatial = 4  # ceil(8 / 2)
    model = _single_conv_model(
        w,
        spatial=spatial,
        extra_attrs=f'auto_pad="SAME_UPPER", strides=[{stride},{stride}]',
        out_spatial=out_spatial,
    )

    rng_x = np.random.default_rng(102)
    x = rng_x.standard_normal((3, Cin, spatial, spatial)).astype(np.float32)

    pruned = onnxsim.apply_wanda_pruning(
        model, calibration_data=[{"X": x}], sparsity=0.5
    )
    onnx.checker.check_model(pruned)
    # End-to-end sanity: the padded/strided pruned model still actually
    # runs through onnxruntime and produces the shape its own graph
    # declares.
    (float_y,) = _run(model, {"X": x})
    (pruned_y,) = _run(pruned, {"X": x})
    assert pruned_y.shape == float_y.shape == (3, Cout, out_spatial, out_spatial)
    assert np.all(np.isfinite(pruned_y))

    # Independent oracle: ONNX's own auto_pad formula
    # (https://onnx.ai/onnx/operators/onnx__Conv.html), computed fresh here
    # rather than calling onnxsim.pruning._resolve_conv_pads.
    pad_total = max(0, (out_spatial - 1) * stride + kh - spatial)
    pad_lo, pad_hi = pad_total // 2, pad_total - pad_total // 2
    assert (pad_lo, pad_hi) == (0, 1)  # confirms the deliberately odd split
    xp = np.pad(x, ((0, 0), (0, 0), (pad_lo, pad_hi), (pad_lo, pad_hi)))
    K = Cin * kh * kw
    patches = np.zeros((x.shape[0] * out_spatial * out_spatial, K), dtype=np.float64)
    idx = 0
    for ni in range(x.shape[0]):
        for oh in range(out_spatial):
            for ow in range(out_spatial):
                patch = xp[
                    ni,
                    :,
                    oh * stride : oh * stride + kh,
                    ow * stride : ow * stride + kw,
                ].astype(np.float64)
                patches[idx] = patch.reshape(-1)
                idx += 1
    act_norm = np.sqrt(np.mean(np.square(patches), axis=0))

    w_flat = w.astype(np.float64).reshape(Cout, K)
    importance = np.abs(w_flat) * act_norm[np.newaxis, :]
    keep = round(K * 0.5)
    order = np.argsort(importance, axis=1)
    drop = order[:, : K - keep]
    mask = np.ones((Cout, K), dtype=bool)
    np.put_along_axis(mask, drop, False, axis=1)
    expected = np.where(mask, w_flat, 0.0).reshape(Cout, Cin, kh, kw)

    np.testing.assert_allclose(_conv_weight(pruned).astype(np.float64), expected)


def test_wanda_pruning_conv_dilated_matches_manual_im2col_importance_oracle_exactly():
    # An earlier version of this module declined every non-unit dilation
    # entirely (sliding_window_view's own unit-offset window assumed
    # unit-spaced taps) and fell back to plain magnitude. Each of the
    # kh*kw taps is now extracted from its own dilation-offset strided
    # slice instead. Adversarial by construction: kernel_shape=[3,3],
    # dilations=[3,3] spaces the 9 taps 3 pixels apart within a 7x7
    # effective receptive field (spatial=13) -- if onnxsim's own tap
    # offsetting were off by even one pixel (e.g. silently treating
    # dilation as if it were 1, the bug the earlier decline specifically
    # guarded against), every captured patch would read the wrong pixels
    # and the exact-equality oracle comparison below would fail.
    Cin, Cout, kh, kw, spatial, dilation = 3, 6, 3, 3, 13, 3
    rng = np.random.default_rng(103)
    w = rng.standard_normal((Cout, Cin, kh, kw)).astype(np.float32)
    eff_k = (kh - 1) * dilation + 1  # 7
    out_spatial = spatial - eff_k + 1  # 7
    model = _single_conv_model(
        w,
        spatial=spatial,
        extra_attrs=f"dilations=[{dilation},{dilation}]",
        out_spatial=out_spatial,
    )

    rng_x = np.random.default_rng(104)
    x = rng_x.standard_normal((3, Cin, spatial, spatial)).astype(np.float32)

    pruned = onnxsim.apply_wanda_pruning(
        model, calibration_data=[{"X": x}], sparsity=0.5
    )
    onnx.checker.check_model(pruned)
    (float_y,) = _run(model, {"X": x})
    (pruned_y,) = _run(pruned, {"X": x})
    assert pruned_y.shape == float_y.shape == (3, Cout, out_spatial, out_spatial)
    assert np.all(np.isfinite(pruned_y))

    K = Cin * kh * kw
    patches = np.zeros((x.shape[0] * out_spatial * out_spatial, K), dtype=np.float64)
    idx = 0
    for ni in range(x.shape[0]):
        for oh in range(out_spatial):
            for ow in range(out_spatial):
                patch = x[
                    ni,
                    :,
                    oh : oh + eff_k : dilation,
                    ow : ow + eff_k : dilation,
                ].astype(np.float64)
                patches[idx] = patch.reshape(-1)
                idx += 1
    act_norm = np.sqrt(np.mean(np.square(patches), axis=0))

    w_flat = w.astype(np.float64).reshape(Cout, K)
    importance = np.abs(w_flat) * act_norm[np.newaxis, :]
    keep = round(K * 0.5)
    order = np.argsort(importance, axis=1)
    drop = order[:, : K - keep]
    mask = np.ones((Cout, K), dtype=bool)
    np.put_along_axis(mask, drop, False, axis=1)
    expected = np.where(mask, w_flat, 0.0).reshape(Cout, Cin, kh, kw)

    np.testing.assert_allclose(_conv_weight(pruned).astype(np.float64), expected)


def test_wanda_pruning_conv_grouped_uses_own_groups_activation_norm():
    # The test that actually proves _conv_group_relative_norm's correctness
    # (not just "it runs"): a general grouped Conv (group=2, 4 filters per
    # group) whose two groups' calibration input channels are engineered to
    # have wildly different activation magnitude -- group 0's input channel
    # carries a much larger norm than group 1's. Every filter's own weight
    # magnitude is identical across both groups (same `w` block, just
    # tiled), so plain magnitude pruning treats every filter identically,
    # but Wanda must not: group 0's filters should end up importance-
    # weighted (and therefore masked) according to group 0's own inflated
    # activation norm, and group 1's filters according to group 1's own
    # (much smaller, unmodified) norm -- not a global average and not one
    # group's statistic bleeding into the other's filters, which is exactly
    # the bug a single shared per-offset norm (as if every filter read the
    # full input) would silently produce for every group but the first.
    Cin_per_group, Cout, group, spatial = 2, 8, 2, 8
    filters_per_group = Cout // group
    rng = np.random.default_rng(90)
    # Same weight block reused for both groups so the two groups' own
    # weight magnitude carries no information -- any masking difference
    # between the two groups can only come from the activation norm.
    w_block = rng.standard_normal((filters_per_group, Cin_per_group, 3, 3)).astype(
        np.float32
    )
    w = np.concatenate([w_block, w_block], axis=0)
    model = _single_conv_model(w, spatial=spatial, group=group)

    Cin = Cin_per_group * group
    x = rng.standard_normal((4, Cin, spatial, spatial)).astype(np.float32)
    x[:, :Cin_per_group, :, :] *= 50.0  # group 0's input channels only

    pruned = onnxsim.apply_wanda_pruning(
        model, calibration_data=[{"X": x}], sparsity=0.5
    )
    onnx.checker.check_model(pruned)
    w_pruned = _conv_weight(pruned)

    # Group 0's filters (indices 0:4) see their own group's inflated norm,
    # group 1's (4:8) their own group's ordinary norm -- despite starting
    # from bit-identical weight blocks, the two groups' resulting masks
    # must therefore differ (a global-average or group-0-shared norm would
    # instead make every filter's mask identical, both groups included).
    group0_mask = w_pruned[:filters_per_group] != 0
    group1_mask = w_pruned[filters_per_group:] != 0
    assert not np.array_equal(group0_mask, group1_mask)

    # And each group's own mask must independently match a manually
    # unfolded, per-group im2col oracle -- not merely "differ from the
    # other group", but the *correct* per-group statistic.
    out_spatial = spatial - 3 + 1
    K = Cin_per_group * 3 * 3
    for g in range(group):
        x_group = x[:, g * Cin_per_group : (g + 1) * Cin_per_group, :, :]
        patches = np.zeros(
            (x.shape[0] * out_spatial * out_spatial, K), dtype=np.float64
        )
        idx = 0
        for ni in range(x.shape[0]):
            for oh in range(out_spatial):
                for ow in range(out_spatial):
                    patch = x_group[ni, :, oh : oh + 3, ow : ow + 3].astype(np.float64)
                    patches[idx] = patch.reshape(-1)
                    idx += 1
        act_norm = np.sqrt(np.mean(np.square(patches), axis=0))

        w_flat = w_block.astype(np.float64).reshape(filters_per_group, K)
        importance = np.abs(w_flat) * act_norm[np.newaxis, :]
        keep = round(K * 0.5)
        order = np.argsort(importance, axis=1)
        mask = np.ones((filters_per_group, K), dtype=bool)
        np.put_along_axis(mask, order[:, : K - keep], False, axis=1)
        expected = np.where(mask, w_flat, 0.0).reshape(w_block.shape)

        got = w_pruned[g * filters_per_group : (g + 1) * filters_per_group]
        np.testing.assert_allclose(got.astype(np.float64), expected)


def test_wanda_pruning_conv_depthwise_matches_manual_group_relative_oracle_exactly():
    # Depthwise Conv (group == in_channels == out_channels): each output
    # filter reads exactly one input channel, so _conv_group_relative_norm
    # degenerates to "each filter's own channel's own norm" -- checked here
    # against a manual per-channel im2col oracle, the same bar
    # test_wanda_pruning_conv_matches_manual_im2col_importance_oracle_exactly
    # already sets for the group=1 case.
    C, kh, kw, spatial = 6, 3, 3, 8
    rng = np.random.default_rng(91)
    w = rng.standard_normal((C, 1, kh, kw)).astype(np.float32)
    model = _single_conv_model(w, spatial=spatial, group=C)

    rng_x = np.random.default_rng(92)
    x = rng_x.standard_normal((3, C, spatial, spatial)).astype(np.float32)
    x[:, 2, :, :] *= 15.0  # one channel salient -- only its own filter cares

    pruned = onnxsim.apply_wanda_pruning(
        model, calibration_data=[{"X": x}], sparsity=0.5
    )
    onnx.checker.check_model(pruned)
    w_pruned = _conv_weight(pruned)

    out_spatial = spatial - kh + 1
    K = kh * kw
    expected = np.zeros_like(w)
    for c in range(C):
        patches = np.zeros(
            (x.shape[0] * out_spatial * out_spatial, K), dtype=np.float64
        )
        idx = 0
        for ni in range(x.shape[0]):
            for oh in range(out_spatial):
                for ow in range(out_spatial):
                    patch = x[ni, c, oh : oh + kh, ow : ow + kw].astype(np.float64)
                    patches[idx] = patch.reshape(-1)
                    idx += 1
        act_norm = np.sqrt(np.mean(np.square(patches), axis=0))
        w_flat = w[c].astype(np.float64).reshape(1, K)
        importance = np.abs(w_flat) * act_norm[np.newaxis, :]
        keep = round(K * 0.5)
        order = np.argsort(importance, axis=1)
        mask = np.ones((1, K), dtype=bool)
        np.put_along_axis(mask, order[:, : K - keep], False, axis=1)
        expected[c] = np.where(mask, w_flat, 0.0).reshape(1, kh, kw)

    np.testing.assert_allclose(w_pruned.astype(np.float64), expected)


def test_sparsegpt_pruning_conv_now_matches_depthwise_conv():
    # Unlike an earlier version of this module (which declined every
    # group != 1 Conv outright, the same restriction
    # test_sparsegpt_pruning_conv_skips_general_grouped_conv below used to
    # check), apply_sparsegpt_pruning now matches depthwise Conv
    # (group == in_channels == out_channels) too, via a genuinely per-group
    # Hessian and column-processing loop -- see the "apply_sparsegpt_
    # pruning: Conv2D, grouped/depthwise" section below for the full
    # oracle/reference-transliteration/reconstruction-error verification.
    # This test only confirms the layer is no longer left untouched.
    C = 8
    rng = np.random.default_rng(76)
    w = rng.standard_normal((C, 1, 3, 3)).astype(np.float32)
    model = _single_conv_model(w, spatial=10, group=C)
    x_cal = rng.standard_normal((4, C, 10, 10)).astype(np.float32)

    pruned = onnxsim.apply_sparsegpt_pruning(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5
    )
    onnx.checker.check_model(pruned)
    assert not np.array_equal(_conv_weight(pruned), w)
    # Each depthwise channel's own K = kh*kw = 9 is small enough that the
    # block-shared quantile's achievable fractions (k/9) land no closer to
    # 0.5 than 4/9 or 5/9 -- a coarser rounding granularity than the
    # larger-K cases elsewhere in this file, hence the wider tolerance.
    assert onnxsim.weight_sparsity(pruned) == pytest.approx(0.5, abs=0.1)


def test_sparsegpt_pruning_conv_now_matches_general_grouped_conv():
    # Same shape as the depthwise case above, for a general grouped Conv
    # (1 < group < in_channels): also now matched and actually pruned,
    # rather than left untouched.
    Cin, Cout, group = 8, 8, 2
    rng = np.random.default_rng(77)
    w = rng.standard_normal((Cout, Cin // group, 3, 3)).astype(np.float32)
    model = _single_conv_model(w, spatial=10, group=group)
    x_cal = rng.standard_normal((4, Cin, 10, 10)).astype(np.float32)

    pruned = onnxsim.apply_sparsegpt_pruning(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5
    )
    onnx.checker.check_model(pruned)
    assert not np.array_equal(_conv_weight(pruned), w)
    assert onnxsim.weight_sparsity(pruned) == pytest.approx(0.5, abs=0.05)


# --- apply_sparsegpt_pruning --------------------------------------------


def _reference_sparsegpt(w_nk, h, sparsity, n, m, percdamp, blocksize):
    # An independent transliteration of the reference implementation's own
    # ``SparseGPT.fasterprune`` (https://github.com/IST-DASLab/sparsegpt/
    # blob/master/sparsegpt.py), written fresh from that source rather than
    # copied from onnxsim/pruning.py, to give this an oracle that isn't
    # just "the same code twice". Uses the reference's own prunen/prunem
    # naming (prunen = number pruned per group of prunem), the mirror image
    # of onnxsim's own n/m ("n kept per group of m") convention.
    w = w_nk.copy().astype(np.float64)
    rows, cols = w.shape
    h = h.copy().astype(np.float64)
    dead = np.diag(h) == 0
    h[dead, dead] = 1.0
    w[:, dead] = 0.0

    damp = percdamp * np.mean(np.diag(h))
    diag = np.arange(cols)
    h[diag, diag] += damp
    hinv = np.linalg.cholesky(np.linalg.inv(h)).T

    prunen = 0 if n is None else m - n
    prunem = 0 if m is None else m

    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        count = i2 - i1
        w1 = w[:, i1:i2].copy()
        q1 = np.zeros_like(w1)
        err1 = np.zeros_like(w1)
        hinv1 = hinv[i1:i2, i1:i2]

        if prunen == 0:
            tmp = w1**2 / (np.diag(hinv1).reshape(1, -1)) ** 2
            thresh = np.sort(tmp.flatten())[int(tmp.size * sparsity)]
            mask1 = tmp <= thresh
        else:
            mask1 = np.zeros_like(w1, dtype=bool)

        for i in range(count):
            w_col = w1[:, i]
            d = hinv1[i, i]
            if prunen != 0 and i % prunem == 0:
                tmp = (
                    w1[:, i : i + prunem] ** 2
                    / (np.diag(hinv1)[i : i + prunem].reshape(1, -1)) ** 2
                )
                idx = np.argsort(tmp, axis=1)[:, :prunen]
                mask1[:, i : i + prunem] = False
                np.put_along_axis(mask1[:, i : i + prunem], idx, True, axis=1)
            q_col = w_col.copy()
            q_col[mask1[:, i]] = 0.0
            q1[:, i] = q_col
            err_col = (w_col - q_col) / d
            w1[:, i + 1 :] -= np.outer(err_col, hinv1[i, i + 1 :])
            err1[:, i] = err_col

        w[:, i1:i2] = q1
        w[:, i2:] -= err1 @ hinv[i1:i2, i2:]

    return w


def test_sparsegpt_pruning_matches_reference_transliteration_exactly():
    K, N = 32, 8
    rng = np.random.default_rng(50)
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(K=K, N=N, seed=50)
    x_cal = rng.standard_normal((48, K)).astype(np.float32)

    pruned = onnxsim.apply_sparsegpt_pruning(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5, proc_block_size=12
    )
    onnx.checker.check_model(pruned)

    w_nk = w.T.astype(np.float64)
    h = x_cal.astype(np.float64).T @ x_cal.astype(np.float64)
    expected_nk = _reference_sparsegpt(
        w_nk, h, sparsity=0.5, n=None, m=None, percdamp=0.01, blocksize=12
    )
    np.testing.assert_allclose(_weight(pruned).T, expected_nk, rtol=1e-6, atol=1e-6)


def test_sparsegpt_pruning_nm_pattern_matches_reference_transliteration():
    K, N = 32, 8
    rng = np.random.default_rng(51)
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(K=K, N=N, seed=51)
    x_cal = rng.standard_normal((48, K)).astype(np.float32)

    pruned = onnxsim.apply_sparsegpt_pruning(
        model, calibration_data=[{"X": x_cal}], n=2, m=4, proc_block_size=12
    )
    onnx.checker.check_model(pruned)

    w_nk = w.T.astype(np.float64)
    h = x_cal.astype(np.float64).T @ x_cal.astype(np.float64)
    expected_nk = _reference_sparsegpt(
        w_nk, h, sparsity=0.0, n=2, m=4, percdamp=0.01, blocksize=12
    )
    np.testing.assert_allclose(_weight(pruned).T, expected_nk, rtol=1e-6, atol=1e-6)

    w_pruned = _weight(pruned).T  # [N, K]
    for row in w_pruned:
        for start in range(0, len(row), 4):
            group = row[start : start + 4]
            if len(group) == 4:
                assert np.count_nonzero(group) == 2


def test_sparsegpt_pruning_zero_sparsity_is_a_no_op():
    K, N = 16, 4
    model = _matmul_model(K=K, N=N, seed=52)
    rng = np.random.default_rng(53)
    x_cal = rng.standard_normal((32, K)).astype(np.float32)
    pruned = onnxsim.apply_sparsegpt_pruning(
        model, calibration_data=[{"X": x_cal}], sparsity=0.0
    )
    np.testing.assert_array_equal(_weight(pruned), _weight(model))


def test_sparsegpt_pruning_no_calibration_batches_leaves_layer_untouched():
    model = _matmul_model(K=16, N=4, seed=54)
    pruned = onnxsim.apply_sparsegpt_pruning(model, calibration_data=[], sparsity=0.5)
    np.testing.assert_array_equal(_weight(pruned), _weight(model))


def test_sparsegpt_pruning_reconstructs_better_than_a_same_mask_style_baseline():
    # The actual point of the technique: given comparable calibration
    # signal, SparseGPT's Hessian-compensated result should reconstruct
    # the layer's output at least as well, on that same calibration data,
    # as simply zeroing the same-shaped lowest-magnitude entries with no
    # compensation at all -- isolating what the error-propagation step
    # buys over naive masking.
    K, N = 48, 12
    rng = np.random.default_rng(55)
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(K=K, N=N, seed=55)
    x_cal = rng.standard_normal((512, K)).astype(np.float32)  # well-conditioned H

    pruned = onnxsim.apply_sparsegpt_pruning(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5
    )
    w_sparsegpt = _weight(pruned).astype(np.float64)

    w64 = w.astype(np.float64)
    score = np.abs(w64)
    thresh = np.sort(score.flatten())[int(score.size * 0.5)]
    w_naive = np.where(score <= thresh, 0.0, w64)

    x64 = x_cal.astype(np.float64)
    y_orig = x64 @ w64
    err_sparsegpt = np.sum((y_orig - x64 @ w_sparsegpt) ** 2)
    err_naive = np.sum((y_orig - x64 @ w_naive) ** 2)
    assert err_sparsegpt <= err_naive


def test_sparsegpt_pruning_requires_n_and_m_together():
    model = _matmul_model(K=16, N=4)
    with pytest.raises(ValueError):
        onnxsim.apply_sparsegpt_pruning(model, n=2)
    with pytest.raises(ValueError):
        onnxsim.apply_sparsegpt_pruning(model, m=4)


# --- apply_sparsegpt_pruning: Conv2D ------------------------------------


def _naive_conv_patch_hessian(x, attrs):
    # Slow, obviously-correct nested-loop reference for the full [K, K]
    # im2col cross-covariance Hessian SparseGPT's Conv support needs --
    # built a completely different way from
    # onnxsim.pruning._conv_im2col_patches's own sliding_window_view-based
    # implementation (an explicit outer-product accumulation per output
    # position instead of any vectorized unfolding), the same bar
    # _naive_conv_patch_sq_sum already set for Wanda's diagonal-only
    # per-offset norm above.
    n, cin, h, w = x.shape
    xp = np.pad(
        x,
        (
            (0, 0),
            (0, 0),
            (attrs.pad_top, attrs.pad_bottom),
            (attrs.pad_left, attrs.pad_right),
        ),
    )
    hp, wp = xp.shape[2], xp.shape[3]
    h_out = (hp - attrs.kh) // attrs.stride_h + 1
    w_out = (wp - attrs.kw) // attrs.stride_w + 1
    k = cin * attrs.kh * attrs.kw
    hessian = np.zeros((k, k), dtype=np.float64)
    count = 0
    for ni in range(n):
        for oh in range(h_out):
            for ow in range(w_out):
                patch = np.zeros(k, dtype=np.float64)
                idx = 0
                for c in range(cin):
                    for i in range(attrs.kh):
                        for j in range(attrs.kw):
                            patch[idx] = xp[
                                ni, c, oh * attrs.stride_h + i, ow * attrs.stride_w + j
                            ]
                            idx += 1
                hessian += np.outer(patch, patch)
                count += 1
    return hessian, count


def test_sparsegpt_conv_hessian_matches_naive_nested_loop_oracle():
    from onnxsim.pruning import _conv_im2col_patches, _ConvSpatialAttrs

    rng = np.random.default_rng(90)
    x = rng.standard_normal((2, 3, 4, 4))
    cases = [
        _ConvSpatialAttrs(
            kh=3,
            kw=3,
            pad_top=1,
            pad_left=1,
            pad_bottom=1,
            pad_right=1,
            stride_h=2,
            stride_w=2,
        ),
        _ConvSpatialAttrs(
            kh=2,
            kw=2,
            pad_top=0,
            pad_left=0,
            pad_bottom=0,
            pad_right=0,
            stride_h=1,
            stride_w=1,
        ),
        _ConvSpatialAttrs(
            kh=3,
            kw=3,
            pad_top=0,
            pad_left=2,
            pad_bottom=1,
            pad_right=0,
            stride_h=1,
            stride_w=2,
        ),
    ]
    for attrs in cases:
        patches = _conv_im2col_patches(x, attrs)
        h_vec = patches.T @ patches
        h_naive, count_naive = _naive_conv_patch_hessian(x, attrs)
        assert patches.shape[0] == count_naive
        np.testing.assert_allclose(h_vec, h_naive)


def test_sparsegpt_pruning_conv_matches_reference_transliteration_exactly():
    # End-to-end correctness: an independent nested-loop patch unfold (not
    # onnxsim.pruning._conv_im2col_patches) feeds the *reference*
    # SparseGPT transliteration (_reference_sparsegpt, already validated
    # against the MatMul/Gemm path above) to build an expected Conv
    # weight, entirely independent of onnxsim's own Conv Hessian/pruning
    # code -- two independently-built pieces must agree with onnxsim's
    # actual output before this is trusted.
    Cin, Cout, kh, kw, spatial = 3, 6, 3, 3, 8
    rng = np.random.default_rng(91)
    w = rng.standard_normal((Cout, Cin, kh, kw)).astype(np.float32) * 0.5
    model = _single_conv_model(w, spatial=spatial)
    rng_x = np.random.default_rng(92)
    x = rng_x.standard_normal((3, Cin, spatial, spatial)).astype(np.float32)

    pruned = onnxsim.apply_sparsegpt_pruning(
        model, calibration_data=[{"X": x}], sparsity=0.5, proc_block_size=12
    )
    onnx.checker.check_model(pruned)

    out_spatial = spatial - kh + 1
    K = Cin * kh * kw
    patches = np.zeros((x.shape[0] * out_spatial * out_spatial, K), dtype=np.float64)
    idx = 0
    for ni in range(x.shape[0]):
        for oh in range(out_spatial):
            for ow in range(out_spatial):
                patch = x[ni, :, oh : oh + kh, ow : ow + kw].astype(np.float64)
                patches[idx] = patch.reshape(-1)
                idx += 1
    h = patches.T @ patches

    w_nk = w.astype(np.float64).reshape(Cout, K)
    expected_nk = _reference_sparsegpt(
        w_nk, h, sparsity=0.5, n=None, m=None, percdamp=0.01, blocksize=12
    )
    expected = expected_nk.reshape(Cout, Cin, kh, kw)
    np.testing.assert_allclose(
        _conv_weight(pruned).astype(np.float64), expected, rtol=1e-6, atol=1e-6
    )


def test_sparsegpt_pruning_conv_nm_pattern_matches_reference_transliteration():
    Cin, Cout, kh, kw, spatial = 4, 8, 3, 3, 8
    rng = np.random.default_rng(93)
    w = rng.standard_normal((Cout, Cin, kh, kw)).astype(np.float32) * 0.5
    model = _single_conv_model(w, spatial=spatial)
    rng_x = np.random.default_rng(94)
    x = rng_x.standard_normal((3, Cin, spatial, spatial)).astype(np.float32)

    pruned = onnxsim.apply_sparsegpt_pruning(
        model, calibration_data=[{"X": x}], n=2, m=4, proc_block_size=12
    )
    onnx.checker.check_model(pruned)

    out_spatial = spatial - kh + 1
    K = Cin * kh * kw
    patches = np.zeros((x.shape[0] * out_spatial * out_spatial, K), dtype=np.float64)
    idx = 0
    for ni in range(x.shape[0]):
        for oh in range(out_spatial):
            for ow in range(out_spatial):
                patch = x[ni, :, oh : oh + kh, ow : ow + kw].astype(np.float64)
                patches[idx] = patch.reshape(-1)
                idx += 1
    h = patches.T @ patches

    w_nk = w.astype(np.float64).reshape(Cout, K)
    expected_nk = _reference_sparsegpt(
        w_nk, h, sparsity=0.0, n=2, m=4, percdamp=0.01, blocksize=12
    )
    expected = expected_nk.reshape(Cout, Cin, kh, kw)
    np.testing.assert_allclose(
        _conv_weight(pruned).astype(np.float64), expected, rtol=1e-6, atol=1e-6
    )

    w_flat = _conv_weight(pruned).reshape(Cout, K)
    for row in w_flat:
        for start in range(0, len(row), 4):
            group = row[start : start + 4]
            if len(group) == 4:
                assert np.count_nonzero(group) == 2


def test_sparsegpt_pruning_conv_reaches_target_sparsity():
    # Unlike magnitude/Wanda's exact per-row quantile, SparseGPT's
    # block-shared threshold (this module's own docstring) picks the
    # target-index order statistic and keeps every entry <= it, so a tied
    # threshold value can round the actual count up slightly -- checked
    # with a small tolerance instead of exact equality, the same
    # "shared per-block, not per-row" caveat the MatMul/Gemm path already
    # documents.
    Cin, Cout, spatial = 4, 8, 10
    rng = np.random.default_rng(95)
    w = rng.standard_normal((Cout, Cin, 3, 3)).astype(np.float32) * 0.5
    model = _single_conv_model(w, spatial=spatial)
    x_cal = rng.standard_normal((16, Cin, spatial, spatial)).astype(np.float32)

    pruned = onnxsim.apply_sparsegpt_pruning(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5
    )
    onnx.checker.check_model(pruned)
    assert onnxsim.weight_sparsity(pruned) == pytest.approx(0.5, abs=0.02)


def test_sparsegpt_pruning_conv_zero_sparsity_is_a_no_op():
    Cin, Cout, spatial = 4, 8, 10
    rng = np.random.default_rng(96)
    w = rng.standard_normal((Cout, Cin, 3, 3)).astype(np.float32)
    model = _single_conv_model(w, spatial=spatial)
    x_cal = rng.standard_normal((8, Cin, spatial, spatial)).astype(np.float32)

    pruned = onnxsim.apply_sparsegpt_pruning(
        model, calibration_data=[{"X": x_cal}], sparsity=0.0
    )
    np.testing.assert_array_equal(_conv_weight(pruned), w)


def test_sparsegpt_pruning_conv_no_calibration_batches_leaves_layer_untouched():
    Cin, Cout, spatial = 4, 8, 10
    rng = np.random.default_rng(97)
    w = rng.standard_normal((Cout, Cin, 3, 3)).astype(np.float32)
    model = _single_conv_model(w, spatial=spatial)

    pruned = onnxsim.apply_sparsegpt_pruning(model, calibration_data=[], sparsity=0.5)
    np.testing.assert_array_equal(_conv_weight(pruned), w)


def test_sparsegpt_pruning_conv_auto_pad_matches_reference_transliteration():
    # An earlier version of this module declined auto_pad entirely and left
    # the layer completely untouched (no data-free fallback for
    # SparseGPT). auto_pad's own padding is now resolved per calibration
    # batch from the input's own spatial size (_resolve_conv_pads). Same
    # deliberately-odd-pad_total setup as the Wanda auto_pad oracle test
    # above (kh=3, stride=2, spatial=8 -> SAME_UPPER pad_total=1, split
    # pad_top=0/pad_bottom=1) so a wrong-edge or symmetric-split bug would
    # be caught, this time via the full im2col Hessian
    # (_reference_sparsegpt), not just a per-offset norm.
    Cin, Cout, kh, kw, spatial, stride = 3, 6, 3, 3, 8, 2
    rng = np.random.default_rng(98)
    w = rng.standard_normal((Cout, Cin, kh, kw)).astype(np.float32) * 0.5
    out_spatial = 4  # ceil(8 / 2)
    model = _single_conv_model(
        w,
        spatial=spatial,
        extra_attrs=f'auto_pad="SAME_UPPER", strides=[{stride},{stride}]',
        out_spatial=out_spatial,
    )
    rng_x = np.random.default_rng(198)
    x = rng_x.standard_normal((3, Cin, spatial, spatial)).astype(np.float32)

    pruned = onnxsim.apply_sparsegpt_pruning(
        model, calibration_data=[{"X": x}], sparsity=0.5, proc_block_size=12
    )
    onnx.checker.check_model(pruned)
    (float_y,) = _run(model, {"X": x})
    (pruned_y,) = _run(pruned, {"X": x})
    assert pruned_y.shape == float_y.shape == (3, Cout, out_spatial, out_spatial)
    assert np.all(np.isfinite(pruned_y))

    pad_total = max(0, (out_spatial - 1) * stride + kh - spatial)
    pad_lo, pad_hi = pad_total // 2, pad_total - pad_total // 2
    assert (pad_lo, pad_hi) == (0, 1)  # confirms the deliberately odd split
    xp = np.pad(x, ((0, 0), (0, 0), (pad_lo, pad_hi), (pad_lo, pad_hi)))
    K = Cin * kh * kw
    patches = np.zeros((x.shape[0] * out_spatial * out_spatial, K), dtype=np.float64)
    idx = 0
    for ni in range(x.shape[0]):
        for oh in range(out_spatial):
            for ow in range(out_spatial):
                patch = xp[
                    ni,
                    :,
                    oh * stride : oh * stride + kh,
                    ow * stride : ow * stride + kw,
                ].astype(np.float64)
                patches[idx] = patch.reshape(-1)
                idx += 1
    h = patches.T @ patches

    w_nk = w.astype(np.float64).reshape(Cout, K)
    expected_nk = _reference_sparsegpt(
        w_nk, h, sparsity=0.5, n=None, m=None, percdamp=0.01, blocksize=12
    )
    expected = expected_nk.reshape(Cout, Cin, kh, kw)
    np.testing.assert_allclose(
        _conv_weight(pruned).astype(np.float64), expected, rtol=1e-6, atol=1e-6
    )


def test_sparsegpt_pruning_conv_dilated_matches_reference_transliteration():
    # An earlier version of this module declined every non-unit dilation
    # entirely and left the layer completely untouched. Same
    # deliberately-spaced-out adversarial dilation setup as the Wanda
    # dilation oracle test above (kernel_shape=[3,3], dilations=[3,3],
    # spatial=13 -> taps 3 pixels apart within a 7x7 effective receptive
    # field): an off-by-one in tap offsetting would misalign every column
    # of the Hessian this test's own independent im2col unfold builds.
    Cin, Cout, kh, kw, spatial, dilation = 3, 6, 3, 3, 13, 3
    rng = np.random.default_rng(99)
    w = rng.standard_normal((Cout, Cin, kh, kw)).astype(np.float32) * 0.5
    eff_k = (kh - 1) * dilation + 1  # 7
    out_spatial = spatial - eff_k + 1  # 7
    model = _single_conv_model(
        w,
        spatial=spatial,
        extra_attrs=f"dilations=[{dilation},{dilation}]",
        out_spatial=out_spatial,
    )
    rng_x = np.random.default_rng(199)
    x = rng_x.standard_normal((3, Cin, spatial, spatial)).astype(np.float32)

    pruned = onnxsim.apply_sparsegpt_pruning(
        model, calibration_data=[{"X": x}], sparsity=0.5, proc_block_size=12
    )
    onnx.checker.check_model(pruned)
    (float_y,) = _run(model, {"X": x})
    (pruned_y,) = _run(pruned, {"X": x})
    assert pruned_y.shape == float_y.shape == (3, Cout, out_spatial, out_spatial)
    assert np.all(np.isfinite(pruned_y))

    K = Cin * kh * kw
    patches = np.zeros((x.shape[0] * out_spatial * out_spatial, K), dtype=np.float64)
    idx = 0
    for ni in range(x.shape[0]):
        for oh in range(out_spatial):
            for ow in range(out_spatial):
                patch = x[
                    ni,
                    :,
                    oh : oh + eff_k : dilation,
                    ow : ow + eff_k : dilation,
                ].astype(np.float64)
                patches[idx] = patch.reshape(-1)
                idx += 1
    h = patches.T @ patches

    w_nk = w.astype(np.float64).reshape(Cout, K)
    expected_nk = _reference_sparsegpt(
        w_nk, h, sparsity=0.5, n=None, m=None, percdamp=0.01, blocksize=12
    )
    expected = expected_nk.reshape(Cout, Cin, kh, kw)
    np.testing.assert_allclose(
        _conv_weight(pruned).astype(np.float64), expected, rtol=1e-6, atol=1e-6
    )


def test_sparsegpt_pruning_conv_reconstructs_better_than_a_same_mask_style_baseline():
    # The Conv analogue of
    # test_sparsegpt_pruning_reconstructs_better_than_a_same_mask_style_baseline:
    # given comparable calibration signal, SparseGPT's Hessian-compensated
    # Conv weight should reconstruct the layer's real (onnxruntime) output
    # at least as well as naively zeroing the same-shaped lowest-magnitude
    # entries with no compensation at all.
    Cin, Cout, spatial = 4, 8, 10
    rng = np.random.default_rng(100)
    w = rng.standard_normal((Cout, Cin, 3, 3)).astype(np.float32) * 0.5
    model = _single_conv_model(w, spatial=spatial)
    # Well-conditioned H: n_positions = 16*8*8 = 1024 >> K = 4*3*3 = 36.
    x_cal = rng.standard_normal((16, Cin, spatial, spatial)).astype(np.float32)

    pruned = onnxsim.apply_sparsegpt_pruning(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5
    )
    onnx.checker.check_model(pruned)

    K = Cin * 3 * 3
    w64 = w.astype(np.float64).reshape(Cout, K)
    score = np.abs(w64)
    thresh = np.sort(score.flatten())[int(score.size * 0.5)]
    w_naive = np.where(score <= thresh, 0.0, w64).reshape(Cout, Cin, 3, 3)
    naive_model = _single_conv_model(w_naive.astype(np.float32), spatial=spatial)

    (float_y,) = _run(model, {"X": x_cal})
    (sparsegpt_y,) = _run(pruned, {"X": x_cal})
    (naive_y,) = _run(naive_model, {"X": x_cal})
    err_sparsegpt = np.sum(
        (float_y.astype(np.float64) - sparsegpt_y.astype(np.float64)) ** 2
    )
    err_naive = np.sum((float_y.astype(np.float64) - naive_y.astype(np.float64)) ** 2)
    assert err_sparsegpt <= err_naive


# --- apply_sparsegpt_pruning: Conv2D, grouped/depthwise ------------------


def _naive_conv_group_hessian(x, attrs, group, cin_per_group):
    # Brute-force, per-group Hessian oracle: for each group g, slices x's
    # own global input-channel range [g*cin_per_group, (g+1)*cin_per_group)
    # and accumulates that group's own [K, K] Hessian via an explicit
    # outer-product-per-output-position nested Python loop
    # (_naive_conv_patch_hessian, already a completely independent
    # construction from onnxsim.pruning._conv_im2col_patches's vectorized
    # sliding_window_view unfolding) -- never touching any other group's
    # channels. Returns a list of `group` [K, K] arrays.
    return [
        _naive_conv_patch_hessian(
            x[:, g * cin_per_group : (g + 1) * cin_per_group, :, :], attrs
        )[0]
        for g in range(group)
    ]


def test_sparsegpt_conv_grouped_hessian_matches_naive_nested_loop_oracle():
    # Verification bar item 1: an independent brute-force nested-loop
    # oracle for each group's own Hessian, on calibration data engineered
    # so the two groups' own statistics are genuinely different -- a bug
    # that shares one Hessian across groups, or mixes up which group's
    # slice feeds which filters, would silently pass on symmetric data but
    # must fail here.
    from onnxsim.pruning import _conv_im2col_patches, _ConvSpatialAttrs

    group, cin_per_group = 2, 2
    rng = np.random.default_rng(140)
    x = rng.standard_normal((2, cin_per_group * group, 5, 5))
    x[:, :cin_per_group, :, :] *= 25.0  # group 0 only: wildly different scale

    attrs = _ConvSpatialAttrs(
        kh=3,
        kw=3,
        pad_top=1,
        pad_left=1,
        pad_bottom=1,
        pad_right=1,
        stride_h=1,
        stride_w=2,
    )

    expected = _naive_conv_group_hessian(x, attrs, group, cin_per_group)
    for g in range(group):
        x_g = x[:, g * cin_per_group : (g + 1) * cin_per_group, :, :]
        patches = _conv_im2col_patches(x_g, attrs)
        h_vec = patches.T @ patches
        np.testing.assert_allclose(h_vec, expected[g])
    # The two groups' own oracle Hessians must actually differ -- proof
    # this is exercising genuinely different per-group statistics, not two
    # identical matrices that would both trivially pass elementwise.
    assert not np.allclose(expected[0], expected[1])


def _grouped_sparsegpt_conv_setup(
    seed_w, seed_x, Cin_per_group, Cout, group, kh=3, kw=3, spatial=8
):
    filters_per_group = Cout // group
    rng_w = np.random.default_rng(seed_w)
    w = rng_w.standard_normal((Cout, Cin_per_group, kh, kw)).astype(np.float32) * 0.5
    model = _single_conv_model(w, spatial=spatial, group=group)

    Cin = Cin_per_group * group
    rng_x = np.random.default_rng(seed_x)
    x = rng_x.standard_normal((3, Cin, spatial, spatial)).astype(np.float32)
    # Different per-group activation scale -- a bug mixing up which group's
    # H feeds which filter rows produces a mismatch here rather than
    # accidentally agreeing on symmetric calibration data.
    x[:, :Cin_per_group, :, :] *= 8.0
    return w, model, x, filters_per_group


def _reference_grouped_conv_sparsegpt(
    w, x, group, cin_per_group, kh, kw, sparsity, n, m, percdamp, blocksize
):
    # Second, independent reference oracle: for each group, an independent
    # nested-loop-style im2col unfold (not onnxsim.pruning's own
    # _conv_im2col_patches) builds that group's own H, which then feeds
    # _reference_sparsegpt (the direct fasterprune transliteration, already
    # validated against the group=1 case above) with that group's own
    # correctly-sliced weight sub-block -- entirely independent of
    # onnxsim's own grouped-Conv Hessian/pruning code.
    cout = w.shape[0]
    filters_per_group = cout // group
    spatial = x.shape[2]
    out_spatial = spatial - kh + 1
    K = cin_per_group * kh * kw
    blocks = []
    for g in range(group):
        x_g = x[:, g * cin_per_group : (g + 1) * cin_per_group, :, :]
        patches = np.zeros(
            (x.shape[0] * out_spatial * out_spatial, K), dtype=np.float64
        )
        idx = 0
        for ni in range(x.shape[0]):
            for oh in range(out_spatial):
                for ow in range(out_spatial):
                    patch = x_g[ni, :, oh : oh + kh, ow : ow + kw].astype(np.float64)
                    patches[idx] = patch.reshape(-1)
                    idx += 1
        h_g = patches.T @ patches
        w_nk_g = (
            w[g * filters_per_group : (g + 1) * filters_per_group]
            .astype(np.float64)
            .reshape(filters_per_group, K)
        )
        expected_nk_g = _reference_sparsegpt(
            w_nk_g,
            h_g,
            sparsity=sparsity,
            n=n,
            m=m,
            percdamp=percdamp,
            blocksize=blocksize,
        )
        blocks.append(expected_nk_g.reshape(filters_per_group, cin_per_group, kh, kw))
    return np.concatenate(blocks, axis=0)


def test_sparsegpt_pruning_conv_grouped_matches_reference_transliteration_exactly():
    # Verification bar item 2 (unstructured sparsity): a general grouped
    # Conv (group=2), each group fed through the independent reference
    # oracle above, must match apply_sparsegpt_pruning's actual output
    # exactly.
    Cin_per_group, Cout, group, kh, kw, spatial = 2, 8, 2, 3, 3, 8
    w, model, x, _ = _grouped_sparsegpt_conv_setup(
        130, 131, Cin_per_group, Cout, group, kh, kw, spatial
    )

    pruned = onnxsim.apply_sparsegpt_pruning(
        model, calibration_data=[{"X": x}], sparsity=0.5, proc_block_size=6
    )
    onnx.checker.check_model(pruned)

    expected = _reference_grouped_conv_sparsegpt(
        w,
        x,
        group,
        Cin_per_group,
        kh,
        kw,
        sparsity=0.5,
        n=None,
        m=None,
        percdamp=0.01,
        blocksize=6,
    )
    np.testing.assert_allclose(
        _conv_weight(pruned).astype(np.float64), expected, rtol=1e-6, atol=1e-6
    )

    # And the two groups must have actually ended up differently pruned --
    # not merely "both correct", but proof neither group silently reused
    # the other's mask/compensation.
    filters_per_group = Cout // group
    w_pruned = _conv_weight(pruned)
    mask0 = w_pruned[:filters_per_group] != 0
    mask1 = w_pruned[filters_per_group:] != 0
    assert not np.array_equal(mask0, mask1)


def test_sparsegpt_pruning_conv_grouped_nm_pattern_matches_reference_transliteration():
    # Verification bar item 2 (N:M sparsity). proc_block_size=12 (a multiple
    # of m=4, matching test_sparsegpt_pruning_nm_pattern_matches_reference_
    # transliteration's own choice) keeps every N:M group boundary aligned
    # to an absolute multiple of 4 within each group's own K=18 columns --
    # _sparsegpt_prune_columns' own N:M grouping restarts at column 0 of
    # every proc_block_size-wide block (block-relative, not global), so a
    # block width that isn't itself a multiple of m would still prune a
    # mathematically correct (and reference-matching) N:M-per-block
    # pattern, just not one whose absolute-column groups of 4 each keep
    # exactly n=2 -- irrelevant to correctness, but this test's own
    # per-group manual check below assumes clean absolute alignment.
    Cin_per_group, Cout, group, kh, kw, spatial = 2, 8, 2, 3, 3, 8
    w, model, x, filters_per_group = _grouped_sparsegpt_conv_setup(
        132, 133, Cin_per_group, Cout, group, kh, kw, spatial
    )

    pruned = onnxsim.apply_sparsegpt_pruning(
        model, calibration_data=[{"X": x}], n=2, m=4, proc_block_size=12
    )
    onnx.checker.check_model(pruned)

    expected = _reference_grouped_conv_sparsegpt(
        w,
        x,
        group,
        Cin_per_group,
        kh,
        kw,
        sparsity=0.0,
        n=2,
        m=4,
        percdamp=0.01,
        blocksize=12,
    )
    np.testing.assert_allclose(
        _conv_weight(pruned).astype(np.float64), expected, rtol=1e-6, atol=1e-6
    )

    K = Cin_per_group * kh * kw
    w_flat = _conv_weight(pruned).reshape(Cout, K)
    for row in w_flat:
        for start in range(0, len(row), 4):
            group_vals = row[start : start + 4]
            if len(group_vals) == 4:
                assert np.count_nonzero(group_vals) == 2


def test_sparsegpt_pruning_conv_depthwise_matches_reference_transliteration_exactly():
    # Verification bar item 4: the group == Cin == Cout extreme
    # (Cin/group == 1) -- confirms the per-group Hessian degenerates
    # correctly to a [kh*kw, kh*kw] per-channel Hessian, verified against
    # the same independent reference oracle as the general-grouped case.
    C, kh, kw, spatial = 6, 3, 3, 8
    w, model, x, _ = _grouped_sparsegpt_conv_setup(134, 135, 1, C, C, kh, kw, spatial)

    pruned = onnxsim.apply_sparsegpt_pruning(
        model, calibration_data=[{"X": x}], sparsity=0.5, proc_block_size=4
    )
    onnx.checker.check_model(pruned)

    expected = _reference_grouped_conv_sparsegpt(
        w, x, C, 1, kh, kw, sparsity=0.5, n=None, m=None, percdamp=0.01, blocksize=4
    )
    np.testing.assert_allclose(
        _conv_weight(pruned).astype(np.float64), expected, rtol=1e-6, atol=1e-6
    )


def test_sparsegpt_pruning_conv_grouped_reaches_target_sparsity():
    Cin_per_group, Cout, group, spatial = 2, 8, 2, 10
    rng = np.random.default_rng(136)
    w = rng.standard_normal((Cout, Cin_per_group, 3, 3)).astype(np.float32) * 0.5
    model = _single_conv_model(w, spatial=spatial, group=group)
    x_cal = rng.standard_normal((16, Cin_per_group * group, spatial, spatial)).astype(
        np.float32
    )

    pruned = onnxsim.apply_sparsegpt_pruning(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5
    )
    onnx.checker.check_model(pruned)
    assert onnxsim.weight_sparsity(pruned) == pytest.approx(0.5, abs=0.02)


def test_sparsegpt_pruning_conv_grouped_auto_pad_reaches_target_sparsity():
    # An earlier version of this module declined auto_pad entirely, so a
    # grouped Conv with auto_pad got no data-free fallback (every group
    # left completely untouched). auto_pad is now resolved per calibration
    # batch (_resolve_conv_pads) before each group's own im2col Hessian is
    # built, so a grouped auto_pad Conv is pruned like any other -- the
    # padding/dilation resolution happens once per node, upstream of the
    # per-group channel-slicing this test's grouped siblings elsewhere in
    # this file already hold to a stricter (exact-oracle) bar.
    Cin_per_group, Cout, group, spatial = 2, 8, 2, 10
    rng = np.random.default_rng(137)
    w = rng.standard_normal((Cout, Cin_per_group, 3, 3)).astype(np.float32)
    model = _single_conv_model(
        w,
        spatial=spatial,
        group=group,
        extra_attrs='auto_pad="SAME_UPPER"',
        out_spatial=spatial,
    )
    x_cal = rng.standard_normal((16, Cin_per_group * group, spatial, spatial)).astype(
        np.float32
    )

    pruned = onnxsim.apply_sparsegpt_pruning(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5
    )
    onnx.checker.check_model(pruned)
    assert not np.array_equal(_conv_weight(pruned), w)  # actually pruned, not skipped
    assert onnxsim.weight_sparsity(pruned) == pytest.approx(0.5, abs=0.02)
    x = rng.standard_normal((2, Cin_per_group * group, spatial, spatial)).astype(
        np.float32
    )
    (float_y,) = _run(model, {"X": x})
    (pruned_y,) = _run(pruned, {"X": x})
    assert pruned_y.shape == float_y.shape
    assert np.all(np.isfinite(pruned_y))


def test_sparsegpt_pruning_conv_grouped_reconstructs_better_than_a_same_mask_style_baseline():
    # Verification bar item 3: end-to-end reconstruction-error property via
    # onnxruntime, for a grouped Conv -- SparseGPT's Hessian-compensated
    # result should reconstruct the real (onnxruntime) output at least as
    # well as naive same-mask zeroing with no compensation, on
    # well-conditioned calibration data.
    #
    # The naive baseline's threshold is computed independently *per group*
    # here, not globally across both groups' weights combined: since
    # apply_sparsegpt_pruning itself enforces its target sparsity within
    # each group's own [filters_per_group, K] sub-block (its own
    # block-shared threshold, scoped per group -- see this module's own
    # docstring), a fair same-sparsity-pattern-granularity comparison must
    # match that scoping. A single global-threshold naive baseline instead
    # gives naive an extra degree of freedom SparseGPT's own per-group
    # algorithm doesn't have -- uneven sparsity allocation between the two
    # groups' weight sub-blocks, favoring whichever group's magnitudes
    # happen to be more compressible -- which can (and, empirically,
    # reliably does for this weight distribution) make even a "naive"
    # baseline outperform SparseGPT's own necessarily-50%-per-group result,
    # despite SparseGPT correctly beating a same-mask-granularity
    # (per-group) naive baseline every time; not a difference in
    # correctness, just an unfair comparison.
    Cin_per_group, Cout, group, spatial = 2, 8, 2, 10
    rng = np.random.default_rng(138)
    w = rng.standard_normal((Cout, Cin_per_group, 3, 3)).astype(np.float32) * 0.5
    model = _single_conv_model(w, spatial=spatial, group=group)
    Cin = Cin_per_group * group
    # Well-conditioned H per group: n_positions = 16*8*8 = 1024 >> K = 18.
    x_cal = rng.standard_normal((16, Cin, spatial, spatial)).astype(np.float32)

    pruned = onnxsim.apply_sparsegpt_pruning(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5
    )
    onnx.checker.check_model(pruned)

    K = Cin_per_group * 3 * 3
    w64 = w.astype(np.float64).reshape(Cout, K)
    filters_per_group = Cout // group
    w_naive = np.zeros_like(w64)
    for g in range(group):
        rows = slice(g * filters_per_group, (g + 1) * filters_per_group)
        score = np.abs(w64[rows])
        thresh = np.sort(score.flatten())[int(score.size * 0.5)]
        w_naive[rows] = np.where(score <= thresh, 0.0, w64[rows])
    w_naive = w_naive.reshape(Cout, Cin_per_group, 3, 3)
    naive_model = _single_conv_model(
        w_naive.astype(np.float32), spatial=spatial, group=group
    )

    (float_y,) = _run(model, {"X": x_cal})
    (sparsegpt_y,) = _run(pruned, {"X": x_cal})
    (naive_y,) = _run(naive_model, {"X": x_cal})
    err_sparsegpt = np.sum(
        (float_y.astype(np.float64) - sparsegpt_y.astype(np.float64)) ** 2
    )
    err_naive = np.sum((float_y.astype(np.float64) - naive_y.astype(np.float64)) ** 2)
    assert err_sparsegpt <= err_naive


# --- apply_structured_pruning ------------------------------------------------


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
    model.graph.output.append(
        onnx.helper.make_tensor_value_info("h", onnx.TensorProto.FLOAT, ["batch", H])
    )

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


# --- apply_structured_pruning: global_sparsity ---------------------------


def _two_scale_mlp_model(K=8, H=16, Out=4, big_scale=50.0, small_scale=1.0, seed=0):
    # Two independent, ordinary (single-producer, group=1) MLP chains
    # sharing one input, deliberately built at very different weight-
    # magnitude scales -- the adversarial case `global_sparsity` exists
    # for: the default per-chain-uniform mode cuts both to the same output-
    # channel *count* regardless of scale, while `global_sparsity` should
    # redistribute toward the uniformly-smaller chain.
    rng = np.random.default_rng(seed)
    w1_big = (rng.standard_normal((K, H)) * big_scale).astype(np.float32)
    w2_big = rng.standard_normal((H, Out)).astype(np.float32)
    w1_small = (rng.standard_normal((K, H)) * small_scale).astype(np.float32)
    w2_small = rng.standard_normal((H, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Ybig, float[batch,{Out}] Ysmall)
        {{
          hbig = MatMul(X, W1big)
          abig = Relu(hbig)
          Ybig = MatMul(abig, W2big)
          hsmall = MatMul(X, W1small)
          asmall = Relu(hsmall)
          Ysmall = MatMul(asmall, W2small)
        }}
        """,
        initializer=[
            _f32(w1_big, "W1big"),
            _f32(w2_big, "W2big"),
            _f32(w1_small, "W1small"),
            _f32(w2_small, "W2small"),
        ],
    )
    return model, w1_big, w2_big, w1_small, w2_small


def _oracle_global_structured_keep(importances, sparsity):
    """From-scratch reimplementation of `_apply_chains_global`'s own
    selection algorithm (ascending pooled sort, drop the lowest-scoring
    round(total * sparsity) entries, then a per-chain floor of at least
    one kept channel), for exact-match testing -- not a call into
    production code. `importances` is a list of 1-D per-chain importance
    arrays (pooled in that order, matching `_candidates`'/`_find_chains`'
    own node-encounter order); returns a same-length list of 1-D boolean
    keep masks.
    """
    sizes = [imp.size for imp in importances]
    total = sum(sizes)
    pooled = np.concatenate(importances)
    keep_count_total = min(max(round(total * (1.0 - sparsity)), 0), total)
    drop_count_total = total - keep_count_total
    drop_flat = np.zeros(total, dtype=bool)
    if drop_count_total > 0:
        order = np.argsort(pooled, kind="stable")
        drop_flat[order[:drop_count_total]] = True
    masks = []
    offset = 0
    for imp, size in zip(importances, sizes):
        drop_here = drop_flat[offset : offset + size]
        offset += size
        if drop_here.all():
            drop_here = drop_here.copy()
            drop_here[np.argmax(imp)] = False
        masks.append(~drop_here)
    return masks


def test_structured_pruning_global_sparsity_redistributes_across_chains_and_matches_oracle():
    K, H, Out = 8, 16, 4
    sparsity = 0.5
    model, w1_big, w2_big, w1_small, w2_small = _two_scale_mlp_model(
        K=K, H=H, Out=Out, big_scale=50.0, small_scale=0.5, seed=7
    )

    local_pruned = onnxsim.apply_structured_pruning(model, sparsity=sparsity)
    global_pruned = onnxsim.apply_structured_pruning(
        model, sparsity=sparsity, global_sparsity=True
    )
    onnx.checker.check_model(global_pruned)

    inits_local = {t.name: t for t in local_pruned.graph.initializer}
    inits_global = {t.name: t for t in global_pruned.graph.initializer}

    # Per-chain-uniform (default) mode: both chains cut to exactly the same
    # channel count, regardless of scale.
    assert inits_local["W1big"].dims[1] == H // 2
    assert inits_local["W1small"].dims[1] == H // 2

    # global_sparsity mode: the uniformly-100x-larger chain must keep
    # strictly more channels than the uniformly-small one.
    big_kept = inits_global["W1big"].dims[1]
    small_kept = inits_global["W1small"].dims[1]
    assert big_kept > H // 2 > small_kept

    # Hand-built oracle: pool both chains' own per-channel L2-norm
    # importance and select via the same algorithm _apply_chains_global
    # itself implements, reimplemented from scratch.
    importance_big = np.linalg.norm(w1_big.T, axis=1)
    importance_small = np.linalg.norm(w1_small.T, axis=1)
    keep_big_mask, keep_small_mask = _oracle_global_structured_keep(
        [importance_big, importance_small], sparsity
    )
    keep_big = np.flatnonzero(keep_big_mask)
    keep_small = np.flatnonzero(keep_small_mask)
    assert big_kept == keep_big.size
    assert small_kept == keep_small.size

    # And the actual pruned model reproduces exactly what deleting those
    # same channels by hand in numpy would -- real equivalence, not just a
    # matching channel count.
    rng = np.random.default_rng(21)
    x = rng.standard_normal((6, K)).astype(np.float32)
    y_big, y_small = _run(global_pruned, {"X": x})

    ab = np.maximum(x @ w1_big[:, keep_big], 0)
    yb_oracle = ab @ w2_big[keep_big, :]
    as_ = np.maximum(x @ w1_small[:, keep_small], 0)
    ys_oracle = as_ @ w2_small[keep_small, :]

    np.testing.assert_allclose(y_big, yb_oracle, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(y_small, ys_oracle, rtol=1e-4, atol=1e-4)


def test_structured_pruning_global_sparsity_enforces_per_chain_floor():
    # A 2-channel chain with tiny weights alongside a 32-channel chain with
    # huge weights, at a high global sparsity that would naively want to
    # drop every one of the tiny chain's channels -- the floor must keep
    # exactly one instead of collapsing it to a zero-sized axis.
    K, Out = 8, 4
    H_tiny, H_big = 2, 32
    sparsity = 0.9
    rng = np.random.default_rng(41)
    w1_tiny = (rng.standard_normal((K, H_tiny)) * 0.01).astype(np.float32)
    w2_tiny = rng.standard_normal((H_tiny, Out)).astype(np.float32)
    w1_big = (rng.standard_normal((K, H_big)) * 100.0).astype(np.float32)
    w2_big = rng.standard_normal((H_big, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Ytiny, float[batch,{Out}] Ybig)
        {{
          ht = MatMul(X, W1tiny)
          at = Relu(ht)
          Ytiny = MatMul(at, W2tiny)
          hb = MatMul(X, W1big)
          ab = Relu(hb)
          Ybig = MatMul(ab, W2big)
        }}
        """,
        initializer=[
            _f32(w1_tiny, "W1tiny"),
            _f32(w2_tiny, "W2tiny"),
            _f32(w1_big, "W1big"),
            _f32(w2_big, "W2big"),
        ],
    )

    pruned = onnxsim.apply_structured_pruning(
        model, sparsity=sparsity, global_sparsity=True
    )
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    tiny_kept = inits["W1tiny"].dims[1]
    big_kept = inits["W1big"].dims[1]

    importance_tiny = np.linalg.norm(w1_tiny.T, axis=1)
    importance_big = np.linalg.norm(w1_big.T, axis=1)
    keep_tiny_mask, keep_big_mask = _oracle_global_structured_keep(
        [importance_tiny, importance_big], sparsity
    )
    assert tiny_kept == int(keep_tiny_mask.sum()) == 1
    assert big_kept == int(keep_big_mask.sum())


def test_structured_pruning_global_sparsity_leaves_gated_ffn_chain_untouched():
    # A gated (SwiGLU-style) pair must agree on one *shared* keep set
    # already (both branches ranked together, pruned to the same surviving
    # indices) -- global_sparsity declines to layer a second, global
    # agreement on top of that rather than guess at one (see
    # apply_structured_pruning's own `global_sparsity` docstring). Its own
    # weights are made uniformly tiny here, so a naive pooled ranking would
    # otherwise be very eager to prune it hard.
    K, H, Out = 8, 16, 4
    rng = np.random.default_rng(31)
    w1 = (rng.standard_normal((K, H)) * 50.0).astype(np.float32)
    w2 = rng.standard_normal((H, Out)).astype(np.float32)
    wg = (rng.standard_normal((K, H)) * 0.01).astype(np.float32)
    wu = (rng.standard_normal((K, H)) * 0.01).astype(np.float32)
    wd = rng.standard_normal((H, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Yplain, float[batch,{Out}] Ygated)
        {{
          h = MatMul(X, W1)
          a = Relu(h)
          Yplain = MatMul(a, W2)
          gate = MatMul(X, Wg)
          gate_act = Sigmoid(gate)
          up = MatMul(X, Wu)
          hg = Mul(gate_act, up)
          Ygated = MatMul(hg, Wd)
        }}
        """,
        initializer=[
            _f32(w1, "W1"),
            _f32(w2, "W2"),
            _f32(wg, "Wg"),
            _f32(wu, "Wu"),
            _f32(wd, "Wd"),
        ],
    )

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5, global_sparsity=True)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}

    # The gated pair is left completely untouched by global_sparsity mode.
    assert list(inits["Wg"].dims) == [K, H]
    assert list(inits["Wu"].dims) == [K, H]
    assert list(inits["Wd"].dims) == [H, Out]
    # The eligible plain chain is still pruned.
    assert inits["W1"].dims[1] < H


def test_structured_pruning_global_sparsity_leaves_grouped_conv_chain_untouched():
    # A general grouped Conv's own `keep` selection is already constrained
    # to a uniform count *per group block* -- global_sparsity declines to
    # fold it into a single pooled ranking that has no general way to land
    # on a block-uniform count for it (see apply_structured_pruning's own
    # `global_sparsity` docstring). Its own weights are made uniformly tiny
    # here, so a naive pooled ranking would otherwise be very eager to
    # prune it hard.
    K, H, Out = 8, 16, 4
    C, spatial, group = 8, 10, 2
    rng = np.random.default_rng(37)
    w1 = (rng.standard_normal((K, H)) * 50.0).astype(np.float32)
    w2 = rng.standard_normal((H, Out)).astype(np.float32)
    wc1 = (rng.standard_normal((C, C // group, 3, 3)) * 0.01).astype(np.float32)
    wc2 = rng.standard_normal((C, C, 3, 3)).astype(np.float32)
    out_spatial = spatial - 4  # two valid (no-pad) 3x3 convs
    model = _model(
        f"""
        g (float[batch,{K}] X, float[N,{C},{spatial},{spatial}] Xc)
            => (float[batch,{Out}] Yplain, float[N,{C},{out_spatial},{out_spatial}] Yconv)
        {{
          h = MatMul(X, W1)
          a = Relu(h)
          Yplain = MatMul(a, W2)
          hc = Conv<kernel_shape=[3,3], group={group}>(Xc, Wc1)
          ac = Relu(hc)
          Yconv = Conv<kernel_shape=[3,3]>(ac, Wc2)
        }}
        """,
        initializer=[
            _f32(w1, "W1"),
            _f32(w2, "W2"),
            _f32(wc1, "Wc1"),
            _f32(wc2, "Wc2"),
        ],
    )

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5, global_sparsity=True)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}

    # The grouped-Conv chain is left completely untouched by
    # global_sparsity mode.
    assert list(inits["Wc1"].dims) == list(wc1.shape)
    assert list(inits["Wc2"].dims) == list(wc2.shape)
    # The eligible plain chain is still pruned.
    assert inits["W1"].dims[1] < H


def test_structured_pruning_global_sparsity_invalid_sparsity_raises():
    model = _mlp_model(K=8, H=16, Out=4)
    with pytest.raises(ValueError):
        onnxsim.apply_structured_pruning(model, sparsity=1.0, global_sparsity=True)
    with pytest.raises(ValueError):
        onnxsim.apply_structured_pruning(model, sparsity=-0.1, global_sparsity=True)


# --- apply_structured_pruning: fused BiasGelu/FastGelu/QuickGelu hop --------
#
# onnxruntime's own transformer-optimizer tool typically fuses an FFN
# block's bias-add and Gelu-family activation into one node --
# `com.microsoft::BiasGelu(A, B) = Gelu(A + B)` (erf-based) and
# `com.microsoft::FastGelu(X[, bias])` (the tanh approximation, bias
# optional) both collapse `MatMul -> Add(bias) -> Gelu` into a single hop;
# `com.microsoft::QuickGelu(X) = X * Sigmoid(alpha * X)` is a third,
# bias-free fusion some model families use instead. Semantics confirmed
# against onnxruntime's own schema (`contrib_defs.cc`) and CPU kernel
# (`bias_gelu.cc`'s shared `AddBiasGelu`, `quick_gelu.cc`) and by direct
# execution before any of this was written -- see this module's own
# docstring for the exact arithmetic and matching functions.


def _erf_gelu(x):
    # math.erf rather than scipy.special.erf, matching this file's own
    # "needs no scipy/erf" convention for the tanh-approximated Gelu oracle
    # above -- math.erf is stdlib, no extra dependency either.
    import math

    return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / np.sqrt(2.0)))


def _tanh_gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def _quick_gelu(x, alpha=1.702):
    return x * (1.0 / (1.0 + np.exp(-alpha * x)))


def test_structured_pruning_bias_gelu_chain_matches_oracle():
    # up = MatMul(x, W1); h = BiasGelu(up, Bias1); down = MatMul(h, W2) --
    # the realistic fused-FFN shape this feature exists for. W1, Bias1, and
    # W2 must all be sliced to the same combined-importance keep set.
    K, H, Out = 8, 16, 4
    rng = np.random.default_rng(120)
    w1 = rng.standard_normal((K, H)).astype(np.float32)
    bias1 = rng.standard_normal((H,)).astype(np.float32)
    w2 = rng.standard_normal((H, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          up = MatMul(X, W1)
          h = com.microsoft.BiasGelu(up, Bias1)
          Y = MatMul(h, W2)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(bias1, "Bias1"), _f32(w2, "W2")],
        opset=17,
    )
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1"].dims) == [K, H // 2]
    assert list(inits["Bias1"].dims) == [H // 2]
    assert list(inits["W2"].dims) == [H // 2, Out]

    keep = _oracle_keep_indices(w1, H // 2)
    x = rng.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    h = _erf_gelu(x @ w1[:, keep] + bias1[keep])
    y_oracle = h @ w2[keep, :]
    np.testing.assert_allclose(y, y_oracle, rtol=1e-4, atol=1e-4)


def test_structured_pruning_bias_gelu_declines_on_nonconstant_bias():
    # BiasGelu's own schema makes its bias operand required, but here it's a
    # graph input rather than a constant initializer -- can't safely slice
    # it, so the whole chain is declined and the model is left
    # byte-identical, the same conservative bar a non-constant Gemm bias
    # already gets elsewhere in this module.
    K, H, Out = 8, 16, 4
    rng = np.random.default_rng(121)
    w1 = rng.standard_normal((K, H)).astype(np.float32)
    w2 = rng.standard_normal((H, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X, float[{H}] Bias1) => (float[batch,{Out}] Y)
        {{
          up = MatMul(X, W1)
          h = com.microsoft.BiasGelu(up, Bias1)
          Y = MatMul(h, W2)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2")],
        opset=17,
    )
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    assert pruned.SerializeToString() == model.SerializeToString()


def test_structured_pruning_fast_gelu_with_bias_chain_matches_oracle():
    # FastGelu's own bias is optional but present here -- same per-channel
    # slicing bar as BiasGelu's required one, just via a different schema
    # path (_match_fused_bias_gelu's own bias_required=False branch),
    # and the tanh-approximated Gelu formula rather than the erf-based one.
    K, H, Out = 8, 16, 4
    rng = np.random.default_rng(122)
    w1 = rng.standard_normal((K, H)).astype(np.float32)
    bias1 = rng.standard_normal((H,)).astype(np.float32)
    w2 = rng.standard_normal((H, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          up = MatMul(X, W1)
          h = com.microsoft.FastGelu(up, Bias1)
          Y = MatMul(h, W2)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(bias1, "Bias1"), _f32(w2, "W2")],
        opset=17,
    )
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["Bias1"].dims) == [H // 2]

    keep = _oracle_keep_indices(w1, H // 2)
    x = rng.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    h = _tanh_gelu(x @ w1[:, keep] + bias1[keep])
    y_oracle = h @ w2[keep, :]
    np.testing.assert_allclose(y, y_oracle, rtol=1e-4, atol=1e-4)


def test_structured_pruning_fast_gelu_no_bias_chain_matches_oracle():
    # FastGelu with its own optional bias genuinely absent -- exercises
    # _match_fused_bias_gelu's own "bias omitted entirely" branch (a plain
    # tanh-Gelu(x), no per-channel constant to slice at all).
    K, H, Out = 8, 16, 4
    rng = np.random.default_rng(123)
    w1 = rng.standard_normal((K, H)).astype(np.float32)
    w2 = rng.standard_normal((H, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          up = MatMul(X, W1)
          h = com.microsoft.FastGelu(up)
          Y = MatMul(h, W2)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2")],
        opset=17,
    )
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1"].dims) == [K, H // 2]
    assert list(inits["W2"].dims) == [H // 2, Out]

    keep = _oracle_keep_indices(w1, H // 2)
    x = rng.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    h = _tanh_gelu(x @ w1[:, keep])
    y_oracle = h @ w2[keep, :]
    np.testing.assert_allclose(y, y_oracle, rtol=1e-4, atol=1e-4)


def test_structured_pruning_quick_gelu_chain_matches_oracle():
    # QuickGelu(X) = X * Sigmoid(alpha * X) takes no bias operand at all
    # (alpha is a node attribute, not a second input), so -- unlike
    # BiasGelu/FastGelu -- it needed no dedicated hop machinery, only
    # joining `_UNARY_PASS_THROUGH`. Still gets its own oracle-verified
    # test rather than assuming that "just works" from set membership alone.
    K, H, Out = 8, 16, 4
    rng = np.random.default_rng(124)
    w1 = rng.standard_normal((K, H)).astype(np.float32)
    w2 = rng.standard_normal((H, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          up = MatMul(X, W1)
          h = com.microsoft.QuickGelu(up)
          Y = MatMul(h, W2)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2")],
        opset=17,
    )
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1"].dims) == [K, H // 2]
    assert list(inits["W2"].dims) == [H // 2, Out]

    keep = _oracle_keep_indices(w1, H // 2)
    x = rng.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    h = _quick_gelu(x @ w1[:, keep])
    y_oracle = h @ w2[keep, :]
    np.testing.assert_allclose(y, y_oracle, rtol=1e-4, atol=1e-4)


def test_structured_pruning_conv_bias_gelu_hop_is_not_recognized():
    # This fused-bias-activation hop is deliberately MatMul/Gemm-only (see
    # this module's own docstring): a real Conv already carries any bias in
    # its own third input, and neither optimizer fusion targets Conv graphs
    # in practice. A Conv -> BiasGelu -> Conv chain must therefore be left
    # completely untouched, the same as any other unrecognized hop.
    Cin, Cmid, Cout, spatial = 4, 8, 6, 10
    rng = np.random.default_rng(125)
    w1 = rng.standard_normal((Cmid, Cin, 3, 3)).astype(np.float32)
    bias1 = rng.standard_normal((Cmid,)).astype(np.float32)
    w2 = rng.standard_normal((Cout, Cmid, 3, 3)).astype(np.float32)
    mid_spatial = spatial - 2
    out_spatial = mid_spatial - 2
    model = _model(
        f"""
        g (float[N,{Cin},{spatial},{spatial}] X) => (float[N,{Cout},{out_spatial},{out_spatial}] Y)
        {{
          up = Conv<kernel_shape=[3,3]>(X, W1)
          h = com.microsoft.BiasGelu(up, Bias1)
          Y = Conv<kernel_shape=[3,3]>(h, W2)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(bias1, "Bias1"), _f32(w2, "W2")],
        opset=17,
    )
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    assert pruned.SerializeToString() == model.SerializeToString()


def test_structured_wanda_pruning_bias_gelu_chain_matches_oracle():
    # apply_structured_wanda_pruning picks up the fused BiasGelu hop for
    # free (the same _find_chains/_walk_to_consumer chain-finder as plain
    # apply_structured_pruning) -- oracle-verified with real calibration
    # data driving the activation-norm-weighted importance ranking.
    K, H, Out = 8, 16, 4
    rng = np.random.default_rng(126)
    w1 = rng.standard_normal((K, H)).astype(np.float32)
    bias1 = rng.standard_normal((H,)).astype(np.float32)
    w2 = rng.standard_normal((H, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          up = MatMul(X, W1)
          h = com.microsoft.BiasGelu(up, Bias1)
          Y = MatMul(h, W2)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(bias1, "Bias1"), _f32(w2, "W2")],
        opset=17,
    )
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))

    rng_cal = np.random.default_rng(127)
    x_cal = rng_cal.standard_normal((32, K)).astype(np.float32)
    calibration_data = [{"X": x_cal}]

    pruned = onnxsim.apply_structured_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned)

    # The activation norm is captured where the chain feeds its consumer --
    # i.e. after BiasGelu, not the producer's raw pre-activation output.
    h_cal = _erf_gelu(x_cal @ w1 + bias1)
    act_norm = np.sqrt(np.mean(np.square(h_cal), axis=0))
    importance = np.linalg.norm(w1.T, axis=1) * np.maximum(act_norm, 1e-8)
    keep = np.sort(np.argsort(-importance)[: H // 2])

    x = rng_cal.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    h = _erf_gelu(x @ w1[:, keep] + bias1[keep])
    y_oracle = h @ w2[keep, :]
    np.testing.assert_allclose(y, y_oracle, rtol=1e-4, atol=1e-4)


def test_structured_pruning_matmul_residual_add_bias_gelu_hop_matches_oracle():
    # One residual branch has a fused BiasGelu between its producer and the
    # merge point -- exercises _walk_matmul_producer_backward's own new
    # BiasGelu/FastGelu hop (the backward mirror of the forward walk's own
    # hop tested above), telling it apart from the bare Add residual merge
    # it sits next to.
    K, C, Out = 8, 16, 4
    rng = np.random.default_rng(128)
    w1 = rng.standard_normal((K, C)).astype(np.float32)
    bias1 = rng.standard_normal((C,)).astype(np.float32)
    ws = rng.standard_normal((K, C)).astype(np.float32)
    wout = rng.standard_normal((C, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          h = MatMul(X, W1)
          f = com.microsoft.BiasGelu(h, Bias1)
          s = MatMul(X, WS)
          addr = Add(f, s)
          r = Relu(addr)
          Y = MatMul(r, WOUT)
        }}
        """,
        initializer=[
            _f32(w1, "W1"),
            _f32(bias1, "Bias1"),
            _f32(ws, "WS"),
            _f32(wout, "WOUT"),
        ],
        opset=17,
    )
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["Bias1"].dims) == [C // 2]

    importance = np.sqrt(
        np.square(np.linalg.norm(w1.astype(np.float64), axis=0))
        + np.square(np.linalg.norm(ws.astype(np.float64), axis=0))
    )
    keep = np.sort(np.argsort(-importance)[: C // 2])

    x = rng.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    f = _erf_gelu(x @ w1[:, keep] + bias1[keep])
    s = x @ ws[:, keep]
    y_oracle = np.maximum(f + s, 0) @ wout[keep, :]
    np.testing.assert_allclose(y, y_oracle, rtol=1e-4, atol=1e-4)


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


def test_structured_pruning_quick_gelu_gated_ffn_matches_oracle():
    # A gate branch fused into com.microsoft::QuickGelu -- some real
    # gated-FFN model families use this in place of a plain Sigmoid/Gelu
    # gate. QuickGelu is unary (no bias operand at all), so it needed no
    # dedicated gated-chain machinery -- adding it to _UNARY_PASS_THROUGH
    # alone already lets _trace_gate_producer_backward's own unary-only
    # backward trace recognize it, the same way it already recognizes a
    # plain Sigmoid/Gelu gate above. Confirmed here rather than assumed:
    # this module's own docstring explicitly declines the *bias-carrying*
    # BiasGelu/FastGelu fusion on a gate branch (out of scope, see there),
    # but QuickGelu -- bias-free -- is not that case.
    K, H, Out = 8, 16, 4
    rng = np.random.default_rng(129)
    wg = rng.standard_normal((K, H)).astype(np.float32)
    wu = rng.standard_normal((K, H)).astype(np.float32)
    wd = rng.standard_normal((H, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          gate = MatMul(X, Wg)
          gate_act = com.microsoft.QuickGelu(gate)
          up = MatMul(X, Wu)
          h = Mul(gate_act, up)
          Y = MatMul(h, Wd)
        }}
        """,
        initializer=[_f32(wg, "Wg"), _f32(wu, "Wu"), _f32(wd, "Wd")],
        opset=17,
    )
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    keep = _combined_keep_indices(wg, wu, H // 2)

    x = rng.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})

    g = x @ wg[:, keep]
    gate = _quick_gelu(g)
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

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    keep = _combined_keep_indices(wg, wu, H // 2)

    np.testing.assert_array_equal(inits["Wg"], wg[:, keep])
    np.testing.assert_array_equal(inits["Wu"], wu[:, keep])
    np.testing.assert_array_equal(inits["Wd"], wd[keep, :])


# --- Conv2D structured pruning ------------------------------------------


def _conv_pair_model(w1, w2, b1=None, spatial=10, activation="Relu"):
    Cin, C2 = w1.shape[1], w2.shape[0]
    initializer = [_f32(w1, "W1"), _f32(w2, "W2")]
    if b1 is not None:
        conv1 = "h = Conv<kernel_shape=[3,3]>(X, W1, B1)"
        initializer.append(_f32(b1, "B1"))
    else:
        conv1 = "h = Conv<kernel_shape=[3,3]>(X, W1)"
    out_spatial = spatial - 4  # two valid (no-pad) 3x3 convs
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


def _conv_model(Cin=3, C1=16, C2=8, bias=True, activation="Relu", seed=0, spatial=10):
    rng = np.random.default_rng(seed)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    b1 = rng.standard_normal((C1,)).astype(np.float32) if bias else None
    return _conv_pair_model(w1, w2, b1=b1, spatial=spatial, activation=activation)


def _oracle_keep_indices_conv(w, keep_count):
    importance = np.linalg.norm(w.reshape(w.shape[0], -1).astype(np.float64), axis=1)
    return np.sort(np.argsort(-importance)[:keep_count])


def _depthwise_pair_model(w1, dw_hops, w2, b1=None, spatial=10, activation="Relu"):
    """The Conv-chain oracle builder, extended with zero or more depthwise
    pass-through hops between producer and consumer: `dw_hops` is a list of
    ``(weight[C1, 1, kH, kW], bias_or_None)`` depthwise Convs (``group`` is
    always `weight.shape[0]`, so slicing `w1`/`dw_hops`/`w2` down together
    -- as every test below does for its own "oracle" call -- keeps every
    depthwise hop's `group` attribute consistent with its sliced weight for
    free). Each hop, like the producer, is followed by `activation`.
    """
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
    out_spatial = spatial - 2 * n_convs  # each 3x3 valid conv shrinks by 2
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


def test_structured_pruning_depthwise_pass_through_matches_manual_channel_deletion_exactly():
    # A MobileNet/EfficientNet-style inverted-residual block:
    # Conv(group=1) -> Relu -> DepthwiseConv(group=C1) -> Relu ->
    # Conv(group=1). The depthwise layer mixes no channels at all -- output
    # channel i depends only on input channel i -- so the chain walk must
    # cross it transparently: the same channel-index set the real
    # producer/consumer pair is pruned to also slices the depthwise layer's
    # own weight and bias, and shrinks its `group` attribute to match.
    Cin, C1, C2 = 3, 16, 8
    rng = np.random.default_rng(50)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    b1 = rng.standard_normal((C1,)).astype(np.float32)
    wd = rng.standard_normal((C1, 1, 3, 3)).astype(np.float32)
    bd = rng.standard_normal((C1,)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    model = _depthwise_pair_model(w1, [(wd, bd)], w2, b1=b1)

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
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


def test_structured_pruning_multiple_consecutive_depthwise_pass_through_hops_matches_oracle():
    # Two depthwise Convs back to back (e.g. a wider spatial receptive
    # field built from stacked depthwise layers) -- both must be crossed
    # transparently by the same channel-index set, each sliced and
    # re-grouped independently. The second hop also has no bias, folding in
    # that case too.
    Cin, C1, C2 = 3, 12, 6
    rng = np.random.default_rng(52)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    wd1 = rng.standard_normal((C1, 1, 3, 3)).astype(np.float32)
    bd1 = rng.standard_normal((C1,)).astype(np.float32)
    wd2 = rng.standard_normal((C1, 1, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    model = _depthwise_pair_model(w1, [(wd1, bd1), (wd2, None)], w2, spatial=14)

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.25)
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


def test_structured_pruning_depthwise_pass_through_no_bias_matches_oracle():
    # A depthwise hop with no bias at all -- its own [C1, 1, kH, kW] weight
    # is the only thing that needs slicing for it.
    Cin, C1, C2 = 4, 10, 5
    rng = np.random.default_rng(54)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    wd = rng.standard_normal((C1, 1, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    model = _depthwise_pair_model(w1, [(wd, None)], w2, activation="Sigmoid")

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.3)
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


def test_structured_pruning_depthwise_pass_through_branch_is_left_untouched():
    # A depthwise Conv whose output feeds more than one consumer (a
    # branch) can't be crossed transparently either -- doing so would mean
    # picking one branch to carry the chain forward while silently leaving
    # the other reading a now-stale channel count. Left untouched, same as
    # any other branching point this pass declines to guess at (the same
    # single-consumer requirement every other hop in this pass already
    # holds every intermediate tensor to).
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
    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["WD"], wd)
    np.testing.assert_array_equal(inits["W2"], w2)


def test_structured_pruning_conv_chain_shrinks_matched_layers():
    Cin, C1, C2 = 3, 16, 8
    model = _conv_model(Cin=Cin, C1=C1, C2=C2, bias=True)
    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W1"].dims) == [C1 // 2, Cin, 3, 3]
    assert list(inits["B1"].dims) == [C1 // 2]
    assert list(inits["W2"].dims) == [C2, C1 // 2, 3, 3]


def test_structured_pruning_conv_chain_matches_manual_channel_deletion_exactly():
    # Same correctness bar as the MatMul/Gemm chain tests: exact
    # equivalence to deleting the same output filters by hand, not just
    # "close to the float model". Conv has no simple numpy one-liner
    # standing in for the op itself, so the oracle is a second, smaller
    # ONNX graph built directly from the same sliced weights and run
    # through onnxruntime, rather than hand-rolled conv math.
    Cin, C1, C2 = 3, 16, 8
    rng = np.random.default_rng(30)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    b1 = rng.standard_normal((C1,)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    model = _conv_pair_model(w1, w2, b1=b1)

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    keep = _oracle_keep_indices_conv(w1, C1 // 2)
    oracle = _conv_pair_model(w1[keep], w2[:, keep], b1=b1[keep])

    rng_x = np.random.default_rng(31)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def _auto_pad_conv_pair_model(w1, w2, spatial=10, activation="Relu"):
    # The auto_pad analogue of _conv_pair_model: both Convs keep their own
    # non-default auto_pad="SAME_UPPER" -- neither _match_conv_producer nor
    # _match_conv_consumer (the matchers apply_structured_pruning/
    # apply_structured_wanda_pruning's own producer/consumer chain-walk
    # uses) reads that attribute at all, so this is expected to be matched,
    # sliced, and re-run exactly like the plain _conv_pair_model chain
    # above -- see this module's own docstring for why: channel pruning
    # only ever indexes the weight tensor's own out_channels/in_channels
    # axes, never the spatial receptive-field math auto_pad changes.
    # auto_pad="SAME_UPPER" keeps each Conv's own output spatial size equal
    # to its input's, so the chain composes with no separate out_spatial
    # bookkeeping.
    Cin, C2 = w1.shape[1], w2.shape[0]
    return _model(
        f"""
        g (float[N,{Cin},{spatial},{spatial}] X) => (float[N,{C2},{spatial},{spatial}] Y)
        {{
          h = Conv<kernel_shape=[3,3], auto_pad="SAME_UPPER">(X, W1)
          a = {activation}(h)
          Y = Conv<kernel_shape=[3,3], auto_pad="SAME_UPPER">(a, W2)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2")],
    )


def _dilated_conv_pair_model(w1, w2, dilation=3, spatial=19, activation="Relu"):
    # The dilations analogue of _conv_pair_model, same reasoning as
    # _auto_pad_conv_pair_model above but for a non-unit dilation instead
    # (kept in a separate model/test from auto_pad -- not because
    # apply_structured_pruning/apply_structured_wanda_pruning need it kept
    # separate, but because onnxruntime's own CPU EP rejects a Conv node
    # combining a non-unit dilation with auto_pad SAME_UPPER/SAME_LOWER
    # ("Dilation not supported for AutoPadType::SAME_UPPER or
    # AutoPadType::SAME_LOWER"), discovered empirically while writing this
    # test -- an onnxruntime limitation on the *input* model this pass
    # would be asked to prune, not anything about onnxsim's own pruning
    # logic, but it means a real onnxruntime-executable adversarial model
    # can't combine both non-default attributes in one node). No explicit
    # padding either (`pads` defaults to all-zero, "VALID"-equivalent), so
    # each Conv's own output spatial size shrinks by the dilated kernel's
    # own effective extent.
    Cin, C2 = w1.shape[1], w2.shape[0]
    eff_k = (3 - 1) * dilation + 1
    mid_spatial = spatial - eff_k + 1
    out_spatial = mid_spatial - eff_k + 1
    d = f"dilations=[{dilation},{dilation}]"
    return _model(
        f"""
        g (float[N,{Cin},{spatial},{spatial}] X) => (float[N,{C2},{out_spatial},{out_spatial}] Y)
        {{
          h = Conv<kernel_shape=[3,3], {d}>(X, W1)
          a = {activation}(h)
          Y = Conv<kernel_shape=[3,3], {d}>(a, W2)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2")],
    )


def test_structured_pruning_conv_chain_with_auto_pad_matches_oracle_exactly():
    # Confirms empirically (not just "in principle") that a non-default
    # auto_pad Conv chain is *already* matched and pruned correctly by
    # apply_structured_pruning -- _match_conv_producer/_match_conv_consumer
    # never inspect that attribute at all (see this module's own docstring
    # and _auto_pad_conv_pair_model above), so there is no restriction here
    # to lift, only this regression test locking the already-correct
    # behavior in place. Same oracle bar as
    # test_structured_pruning_conv_chain_matches_manual_channel_deletion_exactly:
    # exact equivalence, via onnxruntime, to deleting the same output
    # filters by hand -- auto_pad itself must also survive onto both the
    # pruned producer and consumer unchanged for this to pass, since the
    # oracle model carries it too.
    Cin, C1, C2 = 3, 16, 8
    rng = np.random.default_rng(230)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    model = _auto_pad_conv_pair_model(w1, w2)

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    keep = _oracle_keep_indices_conv(w1, C1 // 2)
    oracle = _auto_pad_conv_pair_model(w1[keep], w2[:, keep])

    rng_x = np.random.default_rng(231)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_structured_pruning_conv_chain_with_dilation_matches_oracle_exactly():
    # The dilation analogue of the auto_pad test above -- same "no
    # restriction to lift, only a regression test locking already-correct
    # behavior in place" bar, this time for a non-unit dilations Conv
    # chain.
    Cin, C1, C2 = 3, 16, 8
    rng = np.random.default_rng(232)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    model = _dilated_conv_pair_model(w1, w2)

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    keep = _oracle_keep_indices_conv(w1, C1 // 2)
    oracle = _dilated_conv_pair_model(w1[keep], w2[:, keep])

    rng_x = np.random.default_rng(233)
    x = rng_x.standard_normal((2, Cin, 19, 19)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_structured_pruning_conv_only_chain_matches_oracle_no_bias():
    # No Conv bias at all, and a non-Relu activation -- a plain
    # Conv -> Sigmoid -> Conv chain.
    Cin, C1, C2 = 4, 12, 6
    rng = np.random.default_rng(32)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    model = _conv_pair_model(w1, w2, activation="Sigmoid")

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.25)
    onnx.checker.check_model(pruned)

    keep = _oracle_keep_indices_conv(w1, C1 - round(C1 * 0.25))
    oracle = _conv_pair_model(w1[keep], w2[:, keep], activation="Sigmoid")

    rng_x = np.random.default_rng(33)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_structured_pruning_skips_grouped_producer_conv():
    # A depthwise Conv (group == in_channels == out_channels) is never
    # itself matched as a producer -- it's only ever a transparent
    # pass-through hop the chain walk may cross between two real
    # producer/consumer boundaries (see the "depthwise_pass_through" tests
    # above). With nothing upstream of it here, there's no real producer to
    # anchor a chain at all, so both layers stay completely untouched, even
    # though the topology otherwise looks identical to a matched pair.
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
    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["W2"], w2)


def test_structured_pruning_skips_grouped_consumer_conv():
    # A depthwise Conv is likewise never matched as a *consumer* -- when
    # its own output feeds a graph output (as here) rather than a further
    # real Conv, crossing it as a pass-through hop simply runs out of chain
    # to walk (see "depthwise Conv ... last node before a graph output" in
    # this module's own docstring), so the walk finds no real consumer and
    # the whole chain -- producer included -- is left untouched.
    Cin, C1 = 3, 8
    rng = np.random.default_rng(35)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C1, 1, 3, 3)).astype(np.float32)  # depthwise consumer
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
    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["W2"], w2)


def _grouped_conv_pair_model(
    w1, w2, group1=1, group2=1, b1=None, spatial=10, activation="Relu"
):
    """`_conv_pair_model`, extended with an explicit ``group`` attribute on
    each Conv -- lets the same oracle-comparison-via-onnxruntime pattern
    used throughout this section cover a general grouped Conv producer
    (`group1`) and/or consumer (`group2`), not just the ``group=1`` case
    `_conv_pair_model` builds. `w1`'s shape must already be
    ``[C1, Cin/group1, kH, kW]`` and `w2`'s ``[C2, C1/group2, kH, kW]`` --
    same caller-responsibility convention `_conv_pair_model` already has for
    the two weights' shapes lining up.
    """
    Cin, C2 = w1.shape[1] * group1, w2.shape[0]
    initializer = [_f32(w1, "W1"), _f32(w2, "W2")]
    g1 = f", group={group1}" if group1 != 1 else ""
    g2 = f", group={group2}" if group2 != 1 else ""
    if b1 is not None:
        conv1 = f"h = Conv<kernel_shape=[3,3]{g1}>(X, W1, B1)"
        initializer.append(_f32(b1, "B1"))
    else:
        conv1 = f"h = Conv<kernel_shape=[3,3]{g1}>(X, W1)"
    out_spatial = spatial - 4  # two valid (no-pad) 3x3 convs
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


def _oracle_keep_indices_conv_grouped(w, group, sparsity):
    """The per-group analogue of `_oracle_keep_indices_conv`: ranks each of
    `group` equal-sized blocks of output filters (`w`'s axis 0) by L2 norm
    *independently*, keeping the same count from every block -- a from-
    scratch reimplementation of the production per-group selection in
    `_apply_chains`/`_chain_group`, not a call into it, so this stays a real
    check on the algorithm rather than the algorithm checking itself.
    """
    out_channels = w.shape[0]
    block = out_channels // group
    per_group_keep = max(1, round(block * (1.0 - sparsity)))
    parts = []
    for gi in range(group):
        lo, hi = gi * block, (gi + 1) * block
        parts.append(_oracle_keep_indices_conv(w[lo:hi], per_group_keep) + lo)
    return np.concatenate(parts)


def _oracle_slice_grouped_consumer_conv(w2, keep, group, n_channels):
    """The per-group analogue of the ordinary `w2[:, keep]` consumer slice:
    a grouped consumer's axis 1 is only `n_channels / group` wide and
    per-group-relative (weight column `j` on a filter in output-group `g`
    means global input channel `g * block + j`), so each output-filter
    group needs its own local slice of `keep` -- translated from global
    indices back to that group's own local ones -- rather than one global
    index set applied to the whole axis. A from-scratch reimplementation of
    `_slice_grouped_consumer_conv_weight`, for the same reason
    `_oracle_keep_indices_conv_grouped` reimplements the selection side.
    """
    out_channels = w2.shape[0]
    out_per_group = out_channels // group
    block = n_channels // group
    parts = []
    for gi in range(group):
        lo, hi = gi * block, (gi + 1) * block
        local_keep = keep[(keep >= lo) & (keep < hi)] - lo
        parts.append(w2[gi * out_per_group : (gi + 1) * out_per_group][:, local_keep])
    return np.concatenate(parts, axis=0)


def test_structured_pruning_general_grouped_producer_conv_prunes_per_group_independently():
    # A general grouped Conv producer (group=2, neither 1 nor its own
    # channel count) feeding an ordinary consumer. Per this module's own
    # docstring, a grouped Conv splits its output channels into `group`
    # equal blocks that must each be ranked and pruned *independently*,
    # keeping the same count per block (so `out_channels % group == 0`
    # survives). Engineer block 0's filters (indices 0-3) to all have a far
    # larger L2 norm than every filter in block 1 (indices 4-7): a *global*
    # top-k over all 8 filters would keep every one of block 0's filters and
    # none of block 1's, leaving block 1 with zero survivors -- violating
    # its own group structure. The correct per-group behavior instead keeps
    # each block's own top half, which -- given this engineered disparity
    # -- provably differs from the global top-k.
    Cin, C1, C2, group = 4, 8, 4, 2
    rng = np.random.default_rng(80)
    w1 = rng.standard_normal((C1, Cin // group, 3, 3)).astype(np.float32)
    w1[:4] *= 10.0  # block 0 dominates block 1
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    model = _grouped_conv_pair_model(w1, w2, group1=group)

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    keep_grouped = _oracle_keep_indices_conv_grouped(w1, group, 0.5)
    keep_global = _oracle_keep_indices_conv(w1, 4)
    assert list(keep_grouped) != list(keep_global)  # sanity: the engineered
    # disparity really does make per-group and global selection disagree
    assert sum(i < 4 for i in keep_grouped) == 2  # exactly half of block 0 ...
    assert sum(i >= 4 for i in keep_grouped) == 2  # ... and half of block 1

    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1[keep_grouped])
    dw_node = next(n for n in pruned.graph.node if "W1" in n.input)
    group_attr = next(a.i for a in dw_node.attribute if a.name == "group")
    assert group_attr == group  # the group count itself never shrinks

    oracle = _grouped_conv_pair_model(
        w1[keep_grouped], w2[:, keep_grouped], group1=group
    )
    rng_x = np.random.default_rng(81)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_structured_pruning_general_grouped_consumer_conv_matches_manual_channel_deletion_exactly():
    # A general grouped Conv consumer (group=2): the shared dimension being
    # pruned is the consumer's own *input* channel axis, but that axis is
    # per-group-relative in a grouped Conv's weight layout ([out_channels,
    # in_channels/group, kH, kW]) -- weight column j on an output filter in
    # group g means global input channel g*(in_channels/group) + j, not
    # global channel j the way an ordinary (group=1) consumer's flat axis
    # works. This is the part of grouped-Conv support that needs real new
    # slicing logic (_slice_grouped_consumer_conv_weight), exercised here
    # with block 0 given a far larger weight norm than block 1 so the two
    # blocks' independently-selected local keep sets aren't trivially
    # identical on both sides.
    Cin, C1, C2, group = 3, 8, 6, 2
    rng = np.random.default_rng(82)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    w1[:4] *= 8.0  # block 0 dominates block 1
    w2 = rng.standard_normal((C2, C1 // group, 3, 3)).astype(np.float32)
    model = _grouped_conv_pair_model(w1, w2, group2=group)

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
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


def test_structured_pruning_both_sides_grouped_matching_group_count_matches_oracle():
    # Both producer and consumer are general grouped Convs, sharing the
    # exact same `group` count -- the one cross-chain composition this pass
    # supports when *both* sides are grouped (see this module's own
    # docstring): the producer's own per-group keep selection (uniform
    # count per block, by construction) already lines up exactly with the
    # consumer's own group boundaries, since both partition the same
    # `n_channels` into the same number of equal contiguous blocks.
    Cin, C1, C2, group = 4, 8, 6, 2
    rng = np.random.default_rng(84)
    w1 = rng.standard_normal((C1, Cin // group, 3, 3)).astype(np.float32)
    w1[:4] *= 6.0
    w2 = rng.standard_normal((C2, C1 // group, 3, 3)).astype(np.float32)
    model = _grouped_conv_pair_model(w1, w2, group1=group, group2=group)

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    keep = _oracle_keep_indices_conv_grouped(w1, group, 0.5)
    w2_sliced = _oracle_slice_grouped_consumer_conv(w2, keep, group, C1)
    oracle = _grouped_conv_pair_model(w1[keep], w2_sliced, group1=group, group2=group)

    rng_x = np.random.default_rng(85)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_structured_pruning_skips_mismatched_grouped_producer_and_consumer():
    # Producer group=2, consumer group=4: both sides grouped, but with a
    # *different* group count. Per this module's own docstring, both sides
    # grouped is only supported when the group counts match -- otherwise
    # the two sides' block boundaries wouldn't generally align (a channel
    # surviving as "the 2nd of the producer's own group" has no
    # well-defined membership in any of the consumer's differently-sized
    # groups), so this composition is declined outright and the whole
    # chain -- both layers -- is left completely untouched, not partially
    # or incorrectly pruned.
    Cin, C1, C2, gp, gc = 4, 8, 8, 2, 4
    rng = np.random.default_rng(86)
    w1 = rng.standard_normal((C1, Cin // gp, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1 // gc, 3, 3)).astype(np.float32)
    model = _grouped_conv_pair_model(w1, w2, group1=gp, group2=gc)

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["W2"], w2)


def test_structured_pruning_conv_into_non_pass_through_op_is_left_untouched():
    # An ordinary CNN classifier tail: Conv -> GlobalAveragePool -> Flatten
    # -> MatMul head. Neither pooling nor flattening is a shape-preserving
    # elementwise op the chain walk recognizes, so the Conv producer is
    # left completely untouched rather than matched to the MatMul by
    # coincidence of a downstream reduction dimension.
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
    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["W2"], w2)


def test_structured_pruning_conv_chain_scale_between_convs_is_left_untouched():
    # A per-channel Mul (e.g. an un-fused BatchNormalization's scale, or a
    # standalone SE-style gate) between two Convs isn't recognized -- unlike
    # the MatMul/Gemm chain walk, which does allow Add/Mul against a
    # per-channel constant. See this module's own docstring for why Conv
    # chains restrict to unary activations only (a real Conv already
    # carries its own bias, and onnxsim's own default optimization fuses
    # BatchNormalization into the preceding Conv before this pass would
    # ever see it).
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
    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["W2"], w2)


def test_structured_wanda_pruning_conv_chain_matches_oracle_exactly():
    Cin, C1, C2 = 3, 16, 8
    rng = np.random.default_rng(40)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    model = _conv_pair_model(w1, w2)

    probe_model = onnx.ModelProto()
    probe_model.CopyFrom(model)
    probe_model.graph.output.append(
        onnx.helper.make_tensor_value_info("a", onnx.TensorProto.FLOAT, None)
    )

    rng_cal = np.random.default_rng(41)
    x_cal = rng_cal.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    calibration_data = [{"X": x_cal}]

    _, a_cal = _run(probe_model, {"X": x_cal})
    # Reduce over every axis but the channel one (axis 1 of NCHW) -- the
    # Conv analogue of the MatMul/Gemm oracle's last-axis reduction above.
    act_norm = np.sqrt(np.mean(np.square(a_cal.astype(np.float64)), axis=(0, 2, 3)))
    importance = np.linalg.norm(
        w1.reshape(C1, -1).astype(np.float64), axis=1
    ) * np.maximum(act_norm, 1e-8)
    keep = np.sort(np.argsort(-importance)[: C1 // 2])

    pruned = onnxsim.apply_structured_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned)

    oracle = _conv_pair_model(w1[keep], w2[:, keep])
    rng_x = np.random.default_rng(42)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def _structured_wanda_conv_chain_attrs_regression(
    model_fn, spatial, seed_w, seed_cal, seed_x
):
    # Shared body for the auto_pad-only and dilation-only structured Wanda
    # regression tests below: confirms empirically that
    # apply_structured_wanda_pruning's own calibration-activation capture
    # is unaffected by a non-default consumer-Conv attribute -- the probe
    # point is the *raw* activation feeding the chain's consumer (captured
    # before that consumer ever applies its own padding/dilation to it),
    # reduced over every axis but the channel one; auto_pad/dilations only
    # govern how the consumer computes *its own* output from that
    # already-captured activation, never which values the probe itself
    # reads or which channel axis they belong to. One input channel is
    # deliberately scaled far above the rest (the same protects-high-
    # activation-channel engineering the unstructured Wanda Conv test above
    # uses) so the resulting keep set is actually activation-driven,
    # verified below to differ from plain L2-norm-only ranking -- not
    # merely reproducing it by coincidence, the same "prove the metric is
    # doing something" bar test_wanda_pruning_conv_protects_high_activation_channel
    # already holds unstructured Wanda to.
    Cin, C1, C2 = 3, 16, 8
    salient_input_channel = 1
    rng = np.random.default_rng(seed_w)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32) * 0.5
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    model = model_fn(w1, w2)

    probe_model = onnx.ModelProto()
    probe_model.CopyFrom(model)
    probe_model.graph.output.append(
        onnx.helper.make_tensor_value_info("a", onnx.TensorProto.FLOAT, None)
    )

    rng_cal = np.random.default_rng(seed_cal)
    x_cal = rng_cal.standard_normal((2, Cin, spatial, spatial)).astype(np.float32)
    x_cal[:, salient_input_channel, :, :] *= 25.0
    calibration_data = [{"X": x_cal}]

    _, a_cal = _run(probe_model, {"X": x_cal})
    act_norm = np.sqrt(np.mean(np.square(a_cal.astype(np.float64)), axis=(0, 2, 3)))
    importance = np.linalg.norm(
        w1.reshape(C1, -1).astype(np.float64), axis=1
    ) * np.maximum(act_norm, 1e-8)
    keep = np.sort(np.argsort(-importance)[: C1 // 2])
    plain_keep = np.sort(
        np.argsort(-np.linalg.norm(w1.reshape(C1, -1).astype(np.float64), axis=1))[
            : C1 // 2
        ]
    )
    assert not np.array_equal(keep, plain_keep)  # the activation term matters here

    pruned = onnxsim.apply_structured_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned)

    oracle = model_fn(w1[keep], w2[:, keep])
    rng_x = np.random.default_rng(seed_x)
    x = rng_x.standard_normal((2, Cin, spatial, spatial)).astype(np.float32)
    x[:, salient_input_channel, :, :] *= 25.0
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_structured_wanda_pruning_conv_chain_with_auto_pad_matches_oracle_exactly():
    _structured_wanda_conv_chain_attrs_regression(
        _auto_pad_conv_pair_model, spatial=10, seed_w=240, seed_cal=241, seed_x=242
    )


def test_structured_wanda_pruning_conv_chain_with_dilation_matches_oracle_exactly():
    _structured_wanda_conv_chain_attrs_regression(
        _dilated_conv_pair_model, spatial=19, seed_w=243, seed_cal=244, seed_x=245
    )


def test_structured_wanda_pruning_depthwise_pass_through_matches_oracle_exactly():
    # Same oracle bar with a depthwise hop in the middle: the calibrated
    # activation norm is captured right where the chain feeds its *real*
    # consumer -- i.e. downstream of the (transparent) depthwise hop, not
    # at the real producer's own raw output -- since a depthwise Conv
    # contributes no importance of its own to the ranking.
    Cin, C1, C2 = 3, 16, 8
    rng = np.random.default_rng(70)
    w1 = rng.standard_normal((C1, Cin, 3, 3)).astype(np.float32)
    wd = rng.standard_normal((C1, 1, 3, 3)).astype(np.float32)
    bd = rng.standard_normal((C1,)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    model = _depthwise_pair_model(w1, [(wd, bd)], w2)

    probe_model = onnx.ModelProto()
    probe_model.CopyFrom(model)
    probe_model.graph.output.append(
        onnx.helper.make_tensor_value_info("ad0", onnx.TensorProto.FLOAT, None)
    )

    rng_cal = np.random.default_rng(71)
    x_cal = rng_cal.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    calibration_data = [{"X": x_cal}]

    _, ad0_cal = _run(probe_model, {"X": x_cal})
    act_norm = np.sqrt(np.mean(np.square(ad0_cal.astype(np.float64)), axis=(0, 2, 3)))
    importance = np.linalg.norm(
        w1.reshape(C1, -1).astype(np.float64), axis=1
    ) * np.maximum(act_norm, 1e-8)
    keep = np.sort(np.argsort(-importance)[: C1 // 2])

    pruned = onnxsim.apply_structured_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned)

    oracle = _depthwise_pair_model(w1[keep], [(wd[keep], bd[keep])], w2[:, keep])
    rng_x = np.random.default_rng(72)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


# --- Conv residual (Add-merge) structured pruning ------------------------
# See onnxsim/pruning.py's own "Conv residual (Add-merged) chains" section
# comment for the full reasoning: a bounded slice of general dependency-
# graph pruning, restricted to Conv chains merged by a channel-preserving
# `Add(a, b)` with two non-constant operands (every residual connection's
# shape), holding every hop *toward* a group's own producers to the same
# single-consumer safety bar as the rest of this pass, but propagating the
# group's own established `keep` set *forward* to every extra ordinary
# consumer a shared tensor has -- so a real multi-block ResNet stage's
# necessary fan-out (the post-block tensor read by both the next block's
# own first Conv and, unchanged, that block's own Add) is reached, not
# declined; what's still declined is covered case by case below.


def _residual_diamond_model(w_f, w_s, w_out, spatial=10):
    # y = Conv_out(Relu(Add(Conv_f(X), Conv_s(X)))) -- a "projection
    # shortcut" residual block: two entirely independent Conv producers
    # merge via Add and must therefore share one surviving channel-index
    # set, feeding one real consumer. The smallest instance
    # _find_conv_residual_chains recognizes (a union-find group of size
    # one -- no further Add to transitively union with).
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
    # Two Add merges chained transitively, sharing one spine channel count
    # ("many residual blocks share one spine") with *no* branch anywhere
    # along the chain: add1's own output feeds *only* into add2 (as a
    # second, entirely separate producer's merge partner), never reused
    # elsewhere -- a *simpler* shape than the interior-block fan-out case
    # (see test_structured_pruning_conv_residual_add_prunes_interior_block_fan_out
    # below), which needs no extra-branch resolution at all: the union-find
    # grouping in _find_conv_residual_chains extends across both Adds into
    # one group of three producers on its own.
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


def test_structured_pruning_conv_residual_add_shrinks_matched_layers():
    Cin, C, Cout = 3, 16, 8
    rng = np.random.default_rng(80)
    w_f = rng.standard_normal((C, Cin, 3, 3)).astype(np.float32)
    w_s = rng.standard_normal((C, Cin, 3, 3)).astype(np.float32)
    w_out = rng.standard_normal((Cout, C, 3, 3)).astype(np.float32)
    model = _residual_diamond_model(w_f, w_s, w_out)

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["WF"].dims) == [C // 2, Cin, 3, 3]
    assert list(inits["WS"].dims) == [C // 2, Cin, 3, 3]
    assert list(inits["WOUT"].dims) == [Cout, C // 2, 3, 3]


def test_structured_pruning_conv_residual_add_matches_oracle():
    # Correctness bar: exact equivalence to hand-slicing *both* independent
    # producers to the same combined-importance keep set, not just "the
    # checker doesn't complain".
    Cin, C, Cout = 3, 16, 8
    rng = np.random.default_rng(80)
    w_f = rng.standard_normal((C, Cin, 3, 3)).astype(np.float32)
    w_s = rng.standard_normal((C, Cin, 3, 3)).astype(np.float32)
    w_out = rng.standard_normal((Cout, C, 3, 3)).astype(np.float32)
    model = _residual_diamond_model(w_f, w_s, w_out)

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

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


def test_structured_pruning_conv_residual_add_transitive_chain_matches_oracle():
    # Two Add merges unioned transitively into one group of three
    # producers, all pruned to one shared index set -- see
    # _residual_transitive_model's own docstring above.
    Cin, C, Cz, Cout = 3, 16, 5, 8
    rng = np.random.default_rng(82)
    w_f1 = rng.standard_normal((C, Cin, 3, 3)).astype(np.float32)
    w_s1 = rng.standard_normal((C, Cin, 3, 3)).astype(np.float32)
    w_f2 = rng.standard_normal((C, Cz, 1, 1)).astype(np.float32)
    w_out = rng.standard_normal((Cout, C, 3, 3)).astype(np.float32)
    model = _residual_transitive_model(w_f1, w_s1, w_f2, w_out)

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
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


def test_structured_pruning_conv_residual_add_prunes_interior_block_fan_out():
    # The exact "interior ResNet block" shape this module's own docstring
    # now describes as reached (previously declined -- see this test's own
    # prior name/body, `..._declines_on_fan_out_branch`): `r` (add1's own
    # post-block tensor) is read *twice* -- once by the next block's own
    # first Conv (`nxt`), once unchanged as that next block's own Add
    # shortcut operand (`add2`) -- but both readers are safe, ordinary
    # continuations of the *same* already-established group (add1+add2
    # union transitively, per _find_conv_residual_chains's own union-find),
    # so the group's shared `keep` set is propagated to both rather than
    # declined. `WNEXT` ends up playing a genuine dual role within this
    # *one* chain: a leaf producer of the group (`nxt`'s own output feeds
    # add2, so its output channels are ranked alongside WF/WS's own),
    # *and* an ordinary ("extra branch") consumer of the group's spine
    # (its input channels are pruned to match `r`), exactly the "a weight
    # legitimately plays both roles" case _apply_chains already supports
    # across two different chains -- now exercised within a single one.
    Cin, C, Cout = 3, 16, 8
    rng = np.random.default_rng(84)
    w_f = rng.standard_normal((C, Cin, 3, 3)).astype(np.float32)
    w_s = rng.standard_normal((C, Cin, 3, 3)).astype(np.float32)
    w_next = rng.standard_normal((C, C, 1, 1)).astype(np.float32)
    w_out = rng.standard_normal((Cout, C, 1, 1)).astype(np.float32)

    def _interior_block_model(w_f, w_s, w_next, w_out):
        return _model(
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

    model = _interior_block_model(w_f, w_s, w_next, w_out)
    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    # Deliberately conflicting per-channel importance across WF/WS/WNEXT
    # (independent random weights) means a `keep` recomputed differently at
    # different points -- e.g. WF/WS's own combined ranking used for the
    # producer slice, but a *different* one silently used for WNEXT's own
    # input-axis consumer slice -- would show up immediately as a
    # dimension/index mismatch (caught by the checker/oracle below), not
    # just a subtly wrong number: proof the *same* propagated `keep` is
    # what every branch actually used.
    importance = np.sqrt(
        np.square(np.linalg.norm(w_f.reshape(C, -1).astype(np.float64), axis=1))
        + np.square(np.linalg.norm(w_s.reshape(C, -1).astype(np.float64), axis=1))
        + np.square(np.linalg.norm(w_next.reshape(C, -1).astype(np.float64), axis=1))
    )
    keep = np.sort(np.argsort(-importance)[: C // 2])

    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["WF"], w_f[keep])
    np.testing.assert_array_equal(inits["WS"], w_s[keep])
    np.testing.assert_array_equal(inits["WNEXT"], w_next[np.ix_(keep, keep)])
    np.testing.assert_array_equal(inits["WOUT"], w_out[:, keep])

    oracle = _interior_block_model(
        w_f[keep], w_s[keep], w_next[np.ix_(keep, keep)], w_out[:, keep]
    )
    rng_x = np.random.default_rng(841)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_structured_pruning_conv_residual_add_declines_on_conflicting_shared_weight():
    # The genuine danger fan-out could introduce: two *different* chains
    # both wanting to prune the exact same weight's same axis to two
    # *different*, independently-derived `keep` sets. This can't happen on
    # a shared *activation* tensor at all (ONNX gives every tensor exactly
    # one producer, so a tensor can only ever belong to the one chain whose
    # own walk reaches it) -- so it's engineered here the only way it can
    # actually arise: a tied/reused *weight*. `WNEXT` is reused as both the
    # interior-block group's own fan-out consumer (`nxt = Conv(r, WNEXT)`,
    # wanting the group's own combined-importance `keep`) *and*, completely
    # independently, an ordinary chain's own consumer (`Q = Conv(p,
    # WNEXT)`, wanting `WP`'s own unrelated `keep2`). `_find_conv_chains`
    # runs (and is applied) before `_find_conv_residual_chains` in
    # `apply_structured_pruning`'s own chain list, so the ordinary P->WNEXT
    # chain claims `WNEXT` as a consumer first; the interior-block group,
    # processed second, finds its own `consumer_weights` overlaps
    # `consumer_touched` and declines *entirely* -- WF/WS/WOUT left
    # byte-identical to their original values (no partial pruning), and
    # WNEXT touched *only* by the ordinary chain that actually won the
    # conflict, on its own axis (input/consumer), never on the axis
    # (output/producer) the declined group would have used.
    Cin, C, Cout = 3, 16, 8
    Cin2 = 5
    rng = np.random.default_rng(97)
    w_f = rng.standard_normal((C, Cin, 3, 3)).astype(np.float32)
    w_s = rng.standard_normal((C, Cin, 3, 3)).astype(np.float32)
    w_next = rng.standard_normal((C, C, 1, 1)).astype(np.float32)
    w_out = rng.standard_normal((Cout, C, 1, 1)).astype(np.float32)
    w_p = rng.standard_normal((C, Cin2, 3, 3)).astype(np.float32)
    model = _model(
        f"""
        g (float[N,{Cin},10,10] X, float[N,{Cin2},10,10] Z)
            => (float[N,{Cout},8,8] Y, float[N,{C},8,8] Q)
        {{
          f = Conv<kernel_shape=[3,3]>(X, WF)
          s = Conv<kernel_shape=[3,3]>(X, WS)
          add1 = Add(f, s)
          r = Relu(add1)
          nxt = Conv<kernel_shape=[1,1]>(r, WNEXT)
          add2 = Add(nxt, r)
          Y = Conv<kernel_shape=[1,1]>(add2, WOUT)
          p = Conv<kernel_shape=[3,3]>(Z, WP)
          Q = Conv<kernel_shape=[1,1]>(p, WNEXT)
        }}
        """,
        initializer=[
            _f32(w_f, "WF"),
            _f32(w_s, "WS"),
            _f32(w_next, "WNEXT"),
            _f32(w_out, "WOUT"),
            _f32(w_p, "WP"),
        ],
    )
    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["WF"], w_f)
    np.testing.assert_array_equal(inits["WS"], w_s)
    np.testing.assert_array_equal(inits["WOUT"], w_out)

    importance2 = np.linalg.norm(w_p.reshape(C, -1).astype(np.float64), axis=1)
    keep2 = np.sort(np.argsort(-importance2)[: C // 2])
    np.testing.assert_array_equal(inits["WP"], w_p[keep2])
    # WNEXT's *input* axis (consumer role) was pruned by the winning
    # ordinary chain's own keep2 -- its *output* axis (producer role) is
    # exactly its original, untouched width, since the declined group never
    # got to slice it.
    assert inits["WNEXT"].shape == (C, C // 2, 1, 1)
    np.testing.assert_array_equal(inits["WNEXT"], w_next[:, keep2])


def test_structured_pruning_conv_residual_add_declines_on_identity_shortcut():
    # y = Conv2(Relu(Add(Conv1(X), X))): a classic identity-shortcut
    # residual block with *no* Conv on the shortcut path at all. `X` has no
    # producer this pass owns at all -- it's a graph input, not a tensor any
    # node in this graph produces -- so the backward walk from `add1`'s `X`
    # operand fails outright (nothing to slice a graph input's own channel
    # count to match a pruned Conv1 with) and the whole block is left
    # untouched rather than guessed at. (`X` is also read twice -- by Conv1
    # and directly by Add -- but that alone no longer declines anything: see
    # `test_structured_pruning_conv_residual_add_prunes_interior_block_fan_out`.)
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
    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["W2"], w2)


def test_structured_pruning_conv_residual_add_prunes_grouped_conv_consumer():
    # Two independent (ordinary, group=1) Conv branches merge via Add, same
    # shape _residual_diamond_model uses, but the downstream consumer is a
    # *general grouped* Conv (group=2). Neither producer has any grouping
    # constraint of its own (an ordinary producer accepts any subset of
    # surviving output channels), so this is really the same composition
    # _find_conv_chains already supports for a single-producer chain
    # (an ordinary producer + a grouped consumer), just with the combined
    # (root-sum-square) importance of *two* producers ranked against the
    # consumer's own per-group block boundaries instead of one producer's.
    # See _find_conv_residual_chains's own group-count agreement check
    # (mirroring _find_conv_chains's "both sides grouped with a different
    # group count" decline, generalized to "no producer disagrees with the
    # consumer's group") and this module's own docstring.
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
    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    importance = np.sqrt(
        np.square(np.linalg.norm(w_f.reshape(C, -1).astype(np.float64), axis=1))
        + np.square(np.linalg.norm(w_s.reshape(C, -1).astype(np.float64), axis=1))
    )
    keep = _oracle_keep_indices_combined_grouped(importance, group, 0.5)
    w_out_sliced = _oracle_slice_grouped_consumer_conv(w_out, keep, group, C)

    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["WF"], w_f[keep])
    np.testing.assert_array_equal(inits["WS"], w_s[keep])
    np.testing.assert_array_equal(inits["WOUT"], w_out_sliced)

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
            _f32(w_out_sliced, "WOUT"),
        ],
    )
    rng_x = np.random.default_rng(90)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def _oracle_keep_indices_combined_grouped(importance, group, sparsity):
    """The multi-producer analogue of `_oracle_keep_indices_conv_grouped`:
    takes an already-combined (root-sum-square) per-channel importance
    vector directly, rather than computing it from one weight's own L2 norm
    -- everything else (independent per-`group`-block top-k, same count
    kept from every block) is identical.
    """
    n = importance.shape[0]
    block = n // group
    per_group_keep = max(1, round(block * (1.0 - sparsity)))
    parts = []
    for gi in range(group):
        lo, hi = gi * block, (gi + 1) * block
        local = importance[lo:hi]
        parts.append(np.sort(np.argsort(-local)[:per_group_keep]) + lo)
    return np.concatenate(parts)


def test_structured_pruning_conv_residual_add_prunes_grouped_producers_and_consumer_per_group():
    # The full composition: *two* general grouped Conv producers (group=4
    # each) merge via Add into a general grouped Conv consumer, also
    # group=4 -- every producer and the consumer share the exact same
    # `group` count, the one slice of this composition
    # _find_conv_residual_chains now supports (see this module's own
    # docstring). Adversarially engineered so a *global* (flat, whole-vector)
    # top-k over the combined importance would pick an entirely different --
    # and structurally invalid -- keep set than the correct independent
    # per-group top-k: block 0/1 (channels 0-7, driven almost entirely by
    # WF, WS left near-zero there) are given far larger magnitude than
    # block 2/3 (channels 8-15, driven almost entirely by WS, WF left
    # near-zero there), so a flat top-8 would keep *all* of blocks 0/1 and
    # *none* of blocks 2/3 -- violating both producers' own
    # `out_channels % group == 0` requirement and disagreeing with the
    # oracle built from the from-scratch, per-block reimplementation this
    # test uses. Real onnxruntime execution (not just weight-shape/value
    # comparison) is the actual bar, same as every other oracle test in
    # this module.
    Cin, C, Cout, group = 8, 16, 8, 4
    block = C // group  # 4
    rng = np.random.default_rng(91)
    # Tiny baseline everywhere (negligible contribution to the combined
    # root-sum-square importance -- effectively zero, but nonzero so
    # onnxruntime never sees an exact-zero row that could tie-break
    # differently on some backend).
    w_f = (rng.standard_normal((C, Cin // group, 1, 1)) * 1e-4).astype(np.float32)
    w_s = (rng.standard_normal((C, Cin // group, 1, 1)) * 1e-4).astype(np.float32)
    # Block-decreasing magnitude profile, alternating which producer
    # dominates each block -- see this test's own docstring.
    block_scale = [100.0, 10.0, 1.0, 0.1]
    for gi in range(group):
        lo, hi = gi * block, (gi + 1) * block
        driver = w_f if gi < 2 else w_s
        for j in range(block):
            driver[lo + j, 0, 0, 0] = block_scale[gi] * (block - j)
    w_out = rng.standard_normal((Cout, C // group, 1, 1)).astype(np.float32)

    def _grouped_residual_model(w_f, w_s, w_out):
        return _model(
            f"""
            g (float[N,{Cin},10,10] X) => (float[N,{Cout},10,10] Y)
            {{
              f = Conv<kernel_shape=[1,1],group={group}>(X, WF)
              s = Conv<kernel_shape=[1,1],group={group}>(X, WS)
              addr = Add(f, s)
              r = Relu(addr)
              Y = Conv<kernel_shape=[1,1],group={group}>(r, WOUT)
            }}
            """,
            initializer=[_f32(w_f, "WF"), _f32(w_s, "WS"), _f32(w_out, "WOUT")],
        )

    model = _grouped_residual_model(w_f, w_s, w_out)
    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    importance = np.sqrt(
        np.square(np.linalg.norm(w_f.reshape(C, -1).astype(np.float64), axis=1))
        + np.square(np.linalg.norm(w_s.reshape(C, -1).astype(np.float64), axis=1))
    )
    keep = _oracle_keep_indices_combined_grouped(importance, group, 0.5)
    keep_flat = np.sort(np.argsort(-importance)[:8])
    assert list(keep) != list(keep_flat)  # sanity: the engineered disparity
    # really does make per-group and flat/global selection disagree
    for gi in range(group):
        lo, hi = gi * block, (gi + 1) * block
        assert sum(lo <= i < hi for i in keep) == block // 2  # exactly half
        # of *every* block survives, not just the globally-largest blocks

    w_out_sliced = _oracle_slice_grouped_consumer_conv(w_out, keep, group, C)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["WF"], w_f[keep])
    np.testing.assert_array_equal(inits["WS"], w_s[keep])
    np.testing.assert_array_equal(inits["WOUT"], w_out_sliced)
    for name in ("WF", "WS", "WOUT"):
        node = next(n for n in pruned.graph.node if name in n.input)
        group_attr = next(a.i for a in node.attribute if a.name == "group")
        assert group_attr == group  # the group count itself never shrinks

    oracle = _grouped_residual_model(w_f[keep], w_s[keep], w_out_sliced)
    rng_x = np.random.default_rng(92)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_structured_pruning_conv_residual_add_declines_on_mismatched_producer_group_counts():
    # Two grouped-Conv producers merging via Add, but with *different*
    # group counts (2 vs 4) -- mirrors _find_conv_chains's own "both sides
    # grouped with a different group count" decline (see
    # test_structured_pruning_skips_mismatched_grouped_producer_and_consumer),
    # generalized to two producers instead of a producer/consumer pair: the
    # two producers' own block partitions of the same shared `n_channels`
    # don't generally align, so there's no single per-block top-k that
    # respects both simultaneously. Declined entirely -- every weight left
    # byte-identical to its original value.
    Cin, C, Cout, g1, g2 = 8, 16, 8, 2, 4
    rng = np.random.default_rng(93)
    w_f = rng.standard_normal((C, Cin // g1, 1, 1)).astype(np.float32)
    w_s = rng.standard_normal((C, Cin // g2, 1, 1)).astype(np.float32)
    w_out = rng.standard_normal((Cout, C, 1, 1)).astype(np.float32)
    model = _model(
        f"""
        g (float[N,{Cin},10,10] X) => (float[N,{Cout},10,10] Y)
        {{
          f = Conv<kernel_shape=[1,1],group={g1}>(X, WF)
          s = Conv<kernel_shape=[1,1],group={g2}>(X, WS)
          addr = Add(f, s)
          r = Relu(addr)
          Y = Conv<kernel_shape=[1,1]>(r, WOUT)
        }}
        """,
        initializer=[_f32(w_f, "WF"), _f32(w_s, "WS"), _f32(w_out, "WOUT")],
    )
    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["WF"], w_f)
    np.testing.assert_array_equal(inits["WS"], w_s)
    np.testing.assert_array_equal(inits["WOUT"], w_out)


def test_structured_pruning_conv_residual_add_declines_on_mismatched_producer_and_consumer_group_counts():
    # Both grouped-Conv producers agree with each other (group=2), but the
    # downstream consumer is grouped with a *different* count (group=4):
    # the consumer-side half of the same agreement check, still declined
    # for the same reason -- the producers' own group=2 block boundaries
    # and the consumer's own group=4 ones don't align.
    Cin, C, Cout, gp, gc = 8, 16, 8, 2, 4
    rng = np.random.default_rng(94)
    w_f = rng.standard_normal((C, Cin // gp, 1, 1)).astype(np.float32)
    w_s = rng.standard_normal((C, Cin // gp, 1, 1)).astype(np.float32)
    w_out = rng.standard_normal((Cout, C // gc, 1, 1)).astype(np.float32)
    model = _model(
        f"""
        g (float[N,{Cin},10,10] X) => (float[N,{Cout},10,10] Y)
        {{
          f = Conv<kernel_shape=[1,1],group={gp}>(X, WF)
          s = Conv<kernel_shape=[1,1],group={gp}>(X, WS)
          addr = Add(f, s)
          r = Relu(addr)
          Y = Conv<kernel_shape=[1,1],group={gc}>(r, WOUT)
        }}
        """,
        initializer=[_f32(w_f, "WF"), _f32(w_s, "WS"), _f32(w_out, "WOUT")],
    )
    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["WF"], w_f)
    np.testing.assert_array_equal(inits["WS"], w_s)
    np.testing.assert_array_equal(inits["WOUT"], w_out)


def test_structured_pruning_conv_residual_add_prunes_grouped_fan_out_branch():
    # The interior-block fan-out shape
    # (test_structured_pruning_conv_residual_add_prunes_interior_block_fan_out)
    # combined with a general grouped Conv: `r` (add1's own post-block
    # tensor) fans out to both the next block's own first Conv (`nxt`,
    # itself a *leaf producer* of the group via `add2`) and, unchanged, an
    # *extra* independent grouped-Conv consumer branch (`Y2`) -- exercising
    # _resolve_conv_fanout_branches's own carried `consumer_group` for an
    # extra branch (not just the primary consumer), still required to
    # agree with the rest of the group's shared `group` count.
    Cin, C, Cout, group = 8, 16, 8, 4
    rng = np.random.default_rng(95)
    w_f = rng.standard_normal((C, Cin // group, 1, 1)).astype(np.float32)
    w_s = rng.standard_normal((C, Cin // group, 1, 1)).astype(np.float32)
    w_next = rng.standard_normal((C, C // group, 1, 1)).astype(np.float32)
    w_out = rng.standard_normal((Cout, C, 1, 1)).astype(np.float32)
    w_out2 = rng.standard_normal((Cout, C // group, 1, 1)).astype(np.float32)

    def _model_fn(w_f, w_s, w_next, w_out, w_out2):
        return _model(
            f"""
            g (float[N,{Cin},10,10] X) => (float[N,{Cout},10,10] Y, float[N,{Cout},10,10] Y2)
            {{
              f = Conv<kernel_shape=[1,1],group={group}>(X, WF)
              s = Conv<kernel_shape=[1,1],group={group}>(X, WS)
              add1 = Add(f, s)
              r = Relu(add1)
              nxt = Conv<kernel_shape=[1,1],group={group}>(r, WNEXT)
              add2 = Add(nxt, r)
              Y = Conv<kernel_shape=[1,1]>(add2, WOUT)
              Y2 = Conv<kernel_shape=[1,1],group={group}>(r, WOUT2)
            }}
            """,
            initializer=[
                _f32(w_f, "WF"),
                _f32(w_s, "WS"),
                _f32(w_next, "WNEXT"),
                _f32(w_out, "WOUT"),
                _f32(w_out2, "WOUT2"),
            ],
        )

    model = _model_fn(w_f, w_s, w_next, w_out, w_out2)
    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    importance = np.sqrt(
        np.square(np.linalg.norm(w_f.reshape(C, -1).astype(np.float64), axis=1))
        + np.square(np.linalg.norm(w_s.reshape(C, -1).astype(np.float64), axis=1))
        + np.square(np.linalg.norm(w_next.reshape(C, -1).astype(np.float64), axis=1))
    )
    keep = _oracle_keep_indices_combined_grouped(importance, group, 0.5)
    w_next_sliced = _oracle_slice_grouped_consumer_conv(w_next, keep, group, C)[keep]
    w_out2_sliced = _oracle_slice_grouped_consumer_conv(w_out2, keep, group, C)

    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["WF"], w_f[keep])
    np.testing.assert_array_equal(inits["WS"], w_s[keep])
    np.testing.assert_array_equal(inits["WNEXT"], w_next_sliced)
    np.testing.assert_array_equal(inits["WOUT"], w_out[:, keep])
    np.testing.assert_array_equal(inits["WOUT2"], w_out2_sliced)

    oracle = _model_fn(
        w_f[keep], w_s[keep], w_next_sliced, w_out[:, keep], w_out2_sliced
    )
    rng_x = np.random.default_rng(96)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    y, y2 = _run(pruned, {"X": x})
    y_oracle, y2_oracle = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(y2, y2_oracle, rtol=1e-5, atol=1e-5)


def test_structured_wanda_pruning_conv_residual_add_matches_oracle():
    # apply_structured_wanda_pruning picks up residual grouping for free --
    # _find_conv_residual_chains is shared with apply_structured_pruning,
    # and the activation norm is captured at the same probe point
    # (`chain.consumer_node.input[0]`) any other Conv chain uses.
    Cin, C, Cout = 3, 16, 8
    rng = np.random.default_rng(86)
    w_f = rng.standard_normal((C, Cin, 3, 3)).astype(np.float32)
    w_s = rng.standard_normal((C, Cin, 3, 3)).astype(np.float32)
    w_out = rng.standard_normal((Cout, C, 3, 3)).astype(np.float32)
    model = _residual_diamond_model(w_f, w_s, w_out)

    probe_model = onnx.ModelProto()
    probe_model.CopyFrom(model)
    probe_model.graph.output.append(
        onnx.helper.make_tensor_value_info("r", onnx.TensorProto.FLOAT, None)
    )

    rng_cal = np.random.default_rng(87)
    x_cal = rng_cal.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    calibration_data = [{"X": x_cal}]

    _, r_cal = _run(probe_model, {"X": x_cal})
    act_norm = np.sqrt(np.mean(np.square(r_cal.astype(np.float64)), axis=(0, 2, 3)))
    base_importance = np.sqrt(
        np.square(np.linalg.norm(w_f.reshape(C, -1).astype(np.float64), axis=1))
        + np.square(np.linalg.norm(w_s.reshape(C, -1).astype(np.float64), axis=1))
    )
    importance = base_importance * np.maximum(act_norm, 1e-8)
    keep = np.sort(np.argsort(-importance)[: C // 2])

    pruned = onnxsim.apply_structured_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned)

    oracle = _residual_diamond_model(w_f[keep], w_s[keep], w_out[:, keep])
    rng_x = np.random.default_rng(88)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


# --- apply_structured_pruning: MatMul/Gemm residual (Add-merged) chains -----
#
# The MatMul/Gemm analogue of the Conv residual tests above -- see
# onnxsim.pruning's own module docstring and
# _find_matmul_residual_chains's own section comment for the shared
# reasoning. Mirrors the Conv residual test suite's own shape (diamond,
# transitive, interior-block fan-out, identity-shortcut decline,
# Wanda-for-free)
# plus the composition-safety cases specific to the wider MatMul/Gemm hop
# set: a per-channel bias Add hop, a transposed (`transB=1`) Gemm producer,
# a gated-FFN branch with no downstream projection, and a bare fused
# self-attention op shortcut.


def _matmul_residual_diamond_model(wf, ws, wout):
    # y = MatMul_out(Relu(Add(MatMul_f(X), MatMul_s(X)))) -- the MatMul/Gemm
    # analogue of _residual_diamond_model: two entirely independent MatMul
    # producers merge via Add and must therefore share one surviving
    # channel-index set, feeding one real consumer.
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
    # -- the MatMul/Gemm analogue of _residual_transitive_model. add1's own
    # output feeds *only* into add2 (as a second, entirely separate
    # producer's merge partner), never reused elsewhere, so the union-find
    # grouping in _find_matmul_residual_chains extends across both Adds
    # into one group of three producers.
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


def test_structured_pruning_matmul_residual_add_shrinks_matched_layers():
    K, C, Out = 8, 16, 4
    rng = np.random.default_rng(90)
    wf = rng.standard_normal((K, C)).astype(np.float32)
    ws = rng.standard_normal((K, C)).astype(np.float32)
    wout = rng.standard_normal((C, Out)).astype(np.float32)
    model = _matmul_residual_diamond_model(wf, ws, wout)

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["WF"].dims) == [K, C // 2]
    assert list(inits["WS"].dims) == [K, C // 2]
    assert list(inits["WOUT"].dims) == [C // 2, Out]


def test_structured_pruning_matmul_residual_add_matches_oracle():
    # Correctness bar: exact equivalence to hand-slicing *both* independent
    # producers to the same combined-importance keep set -- with weights
    # deliberately built so the two branches disagree about which channels
    # matter most (the first half of the columns dominate WF's own norm,
    # the second half dominate WS's own norm), so the correct combined-
    # importance keep set is neither branch's own individual top-k and a
    # bug that used only one branch's importance would be caught.
    K, C, Out = 8, 16, 4
    rng = np.random.default_rng(90)
    scale_f = np.where(np.arange(C) < C // 2, 3.0, 0.3).astype(np.float32)
    scale_s = np.where(np.arange(C) < C // 2, 0.3, 3.0).astype(np.float32)
    wf = rng.standard_normal((K, C)).astype(np.float32) * scale_f
    ws = rng.standard_normal((K, C)).astype(np.float32) * scale_s
    wout = rng.standard_normal((C, Out)).astype(np.float32)
    model = _matmul_residual_diamond_model(wf, ws, wout)

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    importance = np.sqrt(
        np.square(np.linalg.norm(wf.astype(np.float64), axis=0))
        + np.square(np.linalg.norm(ws.astype(np.float64), axis=0))
    )
    keep = np.sort(np.argsort(-importance)[: C // 2])
    # The conflicting-importance construction above is only doing its job
    # if the combined keep set actually straddles both halves.
    assert np.any(keep < C // 2) and np.any(keep >= C // 2)
    oracle = _matmul_residual_diamond_model(wf[:, keep], ws[:, keep], wout[keep, :])

    rng_x = np.random.default_rng(91)
    x = rng_x.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_structured_pruning_matmul_residual_add_transitive_chain_matches_oracle():
    K, C, Kz, Out = 8, 16, 5, 4
    rng = np.random.default_rng(92)
    wf1 = rng.standard_normal((K, C)).astype(np.float32)
    ws1 = rng.standard_normal((K, C)).astype(np.float32)
    wf2 = rng.standard_normal((Kz, C)).astype(np.float32)
    wout = rng.standard_normal((C, Out)).astype(np.float32)
    model = _matmul_residual_transitive_model(wf1, ws1, wf2, wout)

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
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


def test_structured_pruning_matmul_residual_add_prunes_interior_block_fan_out():
    # The MatMul/Gemm analogue of
    # test_structured_pruning_conv_residual_add_prunes_interior_block_fan_out
    # -- the exact transformer-stack "interior block" shape: `r` (add1's own
    # post-block tensor) is read *twice*, by the next block's own first
    # MatMul (`nxt`) and unchanged by that block's own Add shortcut
    # (`add2`), both safe continuations of the *same* union-find group, so
    # the group's shared `keep` set is propagated to both. `WNEXT` again
    # plays a genuine dual role in this one chain: a leaf producer (its own
    # output feeds add2, ranked alongside WF/WS) and an ordinary consumer of
    # the group's spine (its own reduction axis pruned to match `r`).
    K, C, Out = 8, 16, 4
    rng = np.random.default_rng(94)
    wf = rng.standard_normal((K, C)).astype(np.float32)
    ws = rng.standard_normal((K, C)).astype(np.float32)
    wnext = rng.standard_normal((C, C)).astype(np.float32)
    wout = rng.standard_normal((C, Out)).astype(np.float32)

    def _interior_block_model(wf, ws, wnext, wout):
        return _model(
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

    model = _interior_block_model(wf, ws, wnext, wout)
    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    # Same deliberately-conflicting-importance argument as the Conv version:
    # WF/WS/WNEXT are independent random weights, so a `keep` silently
    # recomputed differently at different points would show up as a
    # shape/index mismatch, not just a subtly wrong number.
    importance = np.sqrt(
        np.square(np.linalg.norm(wf.astype(np.float64), axis=0))
        + np.square(np.linalg.norm(ws.astype(np.float64), axis=0))
        + np.square(np.linalg.norm(wnext.astype(np.float64), axis=0))
    )
    keep = np.sort(np.argsort(-importance)[: C // 2])

    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["WF"], wf[:, keep])
    np.testing.assert_array_equal(inits["WS"], ws[:, keep])
    np.testing.assert_array_equal(inits["WNEXT"], wnext[np.ix_(keep, keep)])
    np.testing.assert_array_equal(inits["WOUT"], wout[keep, :])

    oracle = _interior_block_model(
        wf[:, keep], ws[:, keep], wnext[np.ix_(keep, keep)], wout[keep, :]
    )
    rng_x = np.random.default_rng(941)
    x = rng_x.standard_normal((2, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_structured_pruning_matmul_residual_add_declines_on_identity_shortcut():
    # y = MatMul2(Relu(Add(MatMul1(X), X))): the exact `x = x + f(x)`
    # transformer-residual identity-shortcut shape, no MatMul on the
    # shortcut path at all. `X` has no producer this pass owns at all --
    # it's a graph input, not a tensor any node in this graph produces -- so
    # the backward walk from `add1`'s `X` operand fails outright, and the
    # whole block is left untouched rather than guessed at. (`X` is also
    # read twice -- by MatMul1 and directly by Add -- but that alone no
    # longer declines anything: see
    # test_structured_pruning_matmul_residual_add_prunes_interior_block_fan_out.)
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
    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["W2"], w2)


def test_structured_pruning_matmul_residual_add_with_bias_hop_matches_oracle():
    # One branch has a per-channel bias Add (a separate node, not Gemm's own
    # bias input) between its producer and the residual merge -- exercises
    # _walk_matmul_producer_backward's wider MatMul/Gemm-only hop set (see
    # this module's own docstring's "and for MatMul/Gemm also a bias/scale
    # add/mul" phrase) and the self-consistent-then-revalidate check that
    # tells this per-channel bias Add apart from an eligible residual-merge
    # Add (one constant operand vs. two non-constant ones).
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

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
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


def test_structured_pruning_matmul_residual_add_transposed_gemm_producer_matches_oracle():
    # One branch is a Gemm with transB=1 (weight stored [N, K], the common
    # real-world PyTorch-exported layout) rather than a plain MatMul's
    # [K, N] -- a regression test for the same class of bug the Conv
    # residual feature's own development turned up (a field the backward
    # walk must carry through from the real producer, here
    # `weight_transposed`, silently dropped/defaulted would mis-slice this
    # producer's weight along the wrong axis).
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

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    # transB=1 storage: output channel is axis 0, so a correctly-sliced
    # producer keeps [N/2, K], not [N, K/2] (the bug a dropped
    # `weight_transposed` field would produce -- it would slice axis 1
    # instead, the wrong axis entirely, or crash on a mismatched shape).
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


def _gated_residual_no_projection_model(wg, wu, wp, wout):
    # y = MatMul_out(Relu(Add(MatMul_p(X), Sigmoid(MatMul_g(X)) *
    # MatMul_u(X)))) -- a gated (SwiGLU-style) combine feeding directly into
    # a residual Add, with no output-projection MatMul between the Mul and
    # the Add. WG/WU/WP must all three agree on one shared surviving
    # channel-index set: WG and WU because their outputs are multiplied
    # elementwise (the ordinary gated-pair constraint), and WP because it's
    # `h`'s own merge partner in `addr` (the ordinary residual constraint).
    K, C = wg.shape
    Out = wout.shape[1]
    return _model(
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


def test_structured_pruning_matmul_residual_add_prunes_gated_branch_with_no_projection():
    # Composition case: a gated (SwiGLU-style) combine feeding directly into
    # a residual Add, with no output-projection MatMul between the Mul and
    # the Add, is now resolved -- see _walk_matmul_producer_backward's own
    # section comment for the composition-safety argument (reusing
    # _find_gated_chains's own gate-branch tracer to resolve *both* Mul
    # operands, rather than picking one and dropping the other). Correctness
    # bar: exact equivalence, via real onnxruntime execution, to hand-slicing
    # all *three* independent producers (WG, WU, and WP, the Add's other
    # operand) to the one combined-importance keep set they must all share --
    # with weights deliberately built so each of the three branches
    # dominates a different third of the channels, so the correct combined
    # keep set is neither branch's own individual top-k and a bug that
    # dropped any one branch's contribution (the exact failure mode this
    # composition was previously declined to avoid) would be caught.
    K, C, Out = 9, 18, 4
    rng = np.random.default_rng(103)
    third = C // 3
    scale_g = np.where((np.arange(C) % 3) == 0, 3.0, 0.3).astype(np.float32)
    scale_u = np.where((np.arange(C) % 3) == 1, 3.0, 0.3).astype(np.float32)
    scale_p = np.where((np.arange(C) % 3) == 2, 3.0, 0.3).astype(np.float32)
    wg = rng.standard_normal((K, C)).astype(np.float32) * scale_g
    wu = rng.standard_normal((K, C)).astype(np.float32) * scale_u
    wp = rng.standard_normal((K, C)).astype(np.float32) * scale_p
    wout = rng.standard_normal((C, Out)).astype(np.float32)
    model = _gated_residual_no_projection_model(wg, wu, wp, wout)

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["WG"].dims) == [K, C // 2]
    assert list(inits["WU"].dims) == [K, C // 2]
    assert list(inits["WP"].dims) == [K, C // 2]
    assert list(inits["WOUT"].dims) == [C // 2, Out]

    importance = np.sqrt(
        np.square(np.linalg.norm(wg.astype(np.float64), axis=0))
        + np.square(np.linalg.norm(wu.astype(np.float64), axis=0))
        + np.square(np.linalg.norm(wp.astype(np.float64), axis=0))
    )
    keep = np.sort(np.argsort(-importance)[: C // 2])
    # The conflicting-importance construction above is only doing its job if
    # the combined keep set actually straddles all three thirds.
    assert np.any(keep % 3 == 0) and np.any(keep % 3 == 1) and np.any(keep % 3 == 2)
    oracle = _gated_residual_no_projection_model(
        wg[:, keep], wu[:, keep], wp[:, keep], wout[keep, :]
    )

    rng_x = np.random.default_rng(104)
    x = rng_x.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)
    assert third > 0  # sanity: the construction above assumes >=3 channels/group


def test_structured_pruning_matmul_residual_add_declines_on_gated_branch_with_extra_fanout():
    # A gate branch that fans out to a second, independent consumer besides
    # the Mul -- `gate_act` (the gate's own activation output) additionally
    # feeds a second graph output `Z` -- must still decline the *whole*
    # group, not just skip the gated branch: _trace_gate_producer_backward
    # (reused unchanged from _find_gated_chains) holds every tensor on a
    # gate/up path to an exact single-consumer bar, stricter than this
    # walk's own deferred bias/scale-hop tensors, so embedding it inside the
    # residual walk doesn't relax that bar. Proves the composition doesn't
    # silently narrow the existing gated-pair safety check.
    K, C, Out = 8, 16, 4
    rng = np.random.default_rng(105)
    wg = rng.standard_normal((K, C)).astype(np.float32)
    wu = rng.standard_normal((K, C)).astype(np.float32)
    wp = rng.standard_normal((K, C)).astype(np.float32)
    wout = rng.standard_normal((C, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y, float[batch,{C}] Z)
        {{
          gate = MatMul(X, WG)
          gate_act = Sigmoid(gate)
          up = MatMul(X, WU)
          h = Mul(gate_act, up)
          p = MatMul(X, WP)
          addr = Add(p, h)
          r = Relu(addr)
          Y = MatMul(r, WOUT)
          Z = Identity(gate_act)
        }}
        """,
        initializer=[
            _f32(wg, "WG"),
            _f32(wu, "WU"),
            _f32(wp, "WP"),
            _f32(wout, "WOUT"),
        ],
    )
    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["WG"], wg)
    np.testing.assert_array_equal(inits["WU"], wu)
    np.testing.assert_array_equal(inits["WP"], wp)
    np.testing.assert_array_equal(inits["WOUT"], wout)


def test_structured_pruning_matmul_residual_add_declines_on_gated_output_shared_with_second_merge():
    # The Mul's own combined output (`h`) feeds *two* independent residual
    # Adds directly -- addr1 and addr2, structurally unrelated merge points
    # (no shared operand unions them into one union-find group). This is the
    # gated analogue of a plain shared producer feeding two separate merges;
    # the existing fan-out machinery already declines that case (see
    # _resolve_matmul_fanout_branches's own forced_first_hop mechanism and
    # this module's own docstring on tie-breaks between conflicting keep
    # sets), and composing it with a gated Mul doesn't weaken that: `h` is
    # tracked as an ordinary backbone tensor of addr1's own group, so addr2
    # reading it too is resolved as an extra fan-out branch the same way any
    # other backbone tensor's extra reader would be -- and fails, since
    # addr2's own *other* operand (s2) is non-constant, not a valid
    # ordinary consumer shape -- declining the whole group rather than
    # silently pruning WG/WU to one group's keep set while the other
    # merge's own branch stays unpruned (which would corrupt addr2's shape).
    K, C, Out = 8, 16, 4
    rng = np.random.default_rng(106)
    wg = rng.standard_normal((K, C)).astype(np.float32)
    wu = rng.standard_normal((K, C)).astype(np.float32)
    wp = rng.standard_normal((K, C)).astype(np.float32)
    ws2 = rng.standard_normal((K, C)).astype(np.float32)
    wout = rng.standard_normal((C, Out)).astype(np.float32)
    wout2 = rng.standard_normal((C, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y, float[batch,{Out}] Y2)
        {{
          gate = MatMul(X, WG)
          gate_act = Sigmoid(gate)
          up = MatMul(X, WU)
          h = Mul(gate_act, up)
          p = MatMul(X, WP)
          addr1 = Add(p, h)
          r1 = Relu(addr1)
          Y = MatMul(r1, WOUT)
          s2 = MatMul(X, WS2)
          addr2 = Add(s2, h)
          r2 = Relu(addr2)
          Y2 = MatMul(r2, WOUT2)
        }}
        """,
        initializer=[
            _f32(wg, "WG"),
            _f32(wu, "WU"),
            _f32(wp, "WP"),
            _f32(ws2, "WS2"),
            _f32(wout, "WOUT"),
            _f32(wout2, "WOUT2"),
        ],
    )
    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["WG"], wg)
    np.testing.assert_array_equal(inits["WU"], wu)
    np.testing.assert_array_equal(inits["WP"], wp)
    np.testing.assert_array_equal(inits["WS2"], ws2)
    np.testing.assert_array_equal(inits["WOUT"], wout)
    np.testing.assert_array_equal(inits["WOUT2"], wout2)


def _swiglu_residual_no_projection_model(wg, wu, wp, wout):
    # y = MatMul_out(Relu(Add(MatMul_p(X), SwiGLU(MatMul_g(X), MatMul_u(X)))))
    # -- the native fused SwiGLU op (opset 28+) feeding directly into a
    # residual Add, with no output-projection MatMul between it and the Add.
    # Mirrors _gated_residual_no_projection_model above exactly, with the
    # separate Sigmoid/Mul pair collapsed into one SwiGLU node -- SwiGLU's
    # own swish lives entirely inside the op, so `gate`/`up` are wired
    # straight into it as its two raw operands.
    K, C = wg.shape
    Out = wout.shape[1]
    return _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          gate = MatMul(X, WG)
          up = MatMul(X, WU)
          h = SwiGLU(gate, up)
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
        opset=28,
    )


def test_structured_pruning_matmul_residual_add_prunes_swiglu_branch_with_no_projection():
    # Composition case, extended to the native fused SwiGLU op (opset 28+):
    # the same "resolve every real producer feeding a gated combine on a
    # residual branch" composition the plain-Mul case above already gets,
    # now also reached when the combine is SwiGLU rather than Mul (see
    # _walk_matmul_producer_backward's own section comment for the
    # composition-safety argument re-derived against SwiGLU's own shape).
    # opset 28 isn't yet implemented by the onnx/onnxruntime versions
    # installed in this environment (no registered SwiGLU schema, so
    # neither onnxruntime nor onnx's own reference evaluator can execute
    # it, and onnx.checker.check_model would reject the opset too) -- so,
    # like the module's own existing native-SwiGLU test
    # (test_structured_pruning_native_swiglu_node_prunes_both_producers_together),
    # this verifies the graph surgery directly via tensor values rather than
    # through actual execution. Same conflicting-importance construction as
    # the plain-Mul composition test above: WG/WU/WP built so each
    # dominates a different third of the channels, so the correct combined
    # keep set straddles all three and a bug that dropped any one branch's
    # contribution would be caught.
    K, C, Out = 9, 18, 4
    rng = np.random.default_rng(108)
    third = C // 3
    scale_g = np.where((np.arange(C) % 3) == 0, 3.0, 0.3).astype(np.float32)
    scale_u = np.where((np.arange(C) % 3) == 1, 3.0, 0.3).astype(np.float32)
    scale_p = np.where((np.arange(C) % 3) == 2, 3.0, 0.3).astype(np.float32)
    wg = rng.standard_normal((K, C)).astype(np.float32) * scale_g
    wu = rng.standard_normal((K, C)).astype(np.float32) * scale_u
    wp = rng.standard_normal((K, C)).astype(np.float32) * scale_p
    wout = rng.standard_normal((C, Out)).astype(np.float32)
    model = _swiglu_residual_no_projection_model(wg, wu, wp, wout)

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    assert inits["WG"].shape == (K, C // 2)
    assert inits["WU"].shape == (K, C // 2)
    assert inits["WP"].shape == (K, C // 2)
    assert inits["WOUT"].shape == (C // 2, Out)

    importance = np.sqrt(
        np.square(np.linalg.norm(wg.astype(np.float64), axis=0))
        + np.square(np.linalg.norm(wu.astype(np.float64), axis=0))
        + np.square(np.linalg.norm(wp.astype(np.float64), axis=0))
    )
    keep = np.sort(np.argsort(-importance)[: C // 2])
    # The conflicting-importance construction above is only doing its job if
    # the combined keep set actually straddles all three thirds.
    assert np.any(keep % 3 == 0) and np.any(keep % 3 == 1) and np.any(keep % 3 == 2)

    np.testing.assert_array_equal(inits["WG"], wg[:, keep])
    np.testing.assert_array_equal(inits["WU"], wu[:, keep])
    np.testing.assert_array_equal(inits["WP"], wp[:, keep])
    np.testing.assert_array_equal(inits["WOUT"], wout[keep, :])
    assert third > 0  # sanity: the construction above assumes >=3 channels/group


def test_structured_pruning_matmul_residual_add_declines_on_swiglu_branch_with_extra_fanout():
    # The SwiGLU analogue of
    # test_structured_pruning_matmul_residual_add_declines_on_gated_branch_with_extra_fanout
    # above: a gate branch that fans out to a second, independent consumer
    # besides the SwiGLU node itself (`gate`, the gate producer's own raw
    # output, additionally feeds a second graph output `Z`) must still
    # decline the *whole* group. SwiGLU's own operands are held to the exact
    # same single-consumer/not-a-graph-output bar _find_gated_chains's own
    # `_is_internal` applies to them for the non-residual case (see
    # _walk_matmul_producer_backward's own section comment), so embedding
    # that check inside the residual walk doesn't relax it -- a second
    # reader of `gate` means SwiGLU's own `a` operand fails that bar, this
    # branch is never resolved as a gated pair, and the whole group falls
    # through to "fail" and is left untouched.
    K, C, Out = 8, 16, 4
    rng = np.random.default_rng(109)
    wg = rng.standard_normal((K, C)).astype(np.float32)
    wu = rng.standard_normal((K, C)).astype(np.float32)
    wp = rng.standard_normal((K, C)).astype(np.float32)
    wout = rng.standard_normal((C, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y, float[batch,{C}] Z)
        {{
          gate = MatMul(X, WG)
          up = MatMul(X, WU)
          h = SwiGLU(gate, up)
          p = MatMul(X, WP)
          addr = Add(p, h)
          r = Relu(addr)
          Y = MatMul(r, WOUT)
          Z = Identity(gate)
        }}
        """,
        initializer=[
            _f32(wg, "WG"),
            _f32(wu, "WU"),
            _f32(wp, "WP"),
            _f32(wout, "WOUT"),
        ],
        opset=28,
    )
    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["WG"], wg)
    np.testing.assert_array_equal(inits["WU"], wu)
    np.testing.assert_array_equal(inits["WP"], wp)
    np.testing.assert_array_equal(inits["WOUT"], wout)


def test_structured_pruning_matmul_residual_add_declines_on_bare_gqa_shortcut():
    # A residual branch whose backward walk would need to cross a fused
    # self-attention op boundary to reach a real producer -- `ctx` (a
    # GroupQueryAttention node's own raw output) feeds directly into the
    # residual Add, with no output-projection MatMul in between. Neither
    # GroupQueryAttention nor its Q/K/V MatMul producers can be reached
    # through it: the walk starting from `ctx` finds an unrecognized node
    # (not a MatMul/Gemm, not an eligible Add, not a unary activation) on
    # its very first hop and fails immediately, without ever looking past
    # GroupQueryAttention at its own Q/K/V inputs. Left completely
    # untouched, the same conservative bar as the gated-branch case above.
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

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["Wq"], wq)
    np.testing.assert_array_equal(inits["Wk"], wk)
    np.testing.assert_array_equal(inits["Wv"], wv)
    np.testing.assert_array_equal(inits["Wp"], wp)
    np.testing.assert_array_equal(inits["Wout"], wout)


def test_structured_wanda_pruning_matmul_residual_add_matches_oracle():
    # apply_structured_wanda_pruning picks up MatMul/Gemm residual grouping
    # for free -- _find_matmul_residual_chains is shared with
    # apply_structured_pruning, and the activation norm is captured at the
    # same probe point (`chain.consumer_node.input[0]`) any other MatMul/Gemm
    # chain uses.
    K, C, Out = 8, 16, 4
    rng = np.random.default_rng(102)
    wf = rng.standard_normal((K, C)).astype(np.float32)
    ws = rng.standard_normal((K, C)).astype(np.float32)
    wout = rng.standard_normal((C, Out)).astype(np.float32)
    model = _matmul_residual_diamond_model(wf, ws, wout)

    probe_model = onnx.ModelProto()
    probe_model.CopyFrom(model)
    probe_model.graph.output.append(
        onnx.helper.make_tensor_value_info("r", onnx.TensorProto.FLOAT, None)
    )

    rng_cal = np.random.default_rng(103)
    x_cal = rng_cal.standard_normal((6, K)).astype(np.float32)
    calibration_data = [{"X": x_cal}]

    _, r_cal = _run(probe_model, {"X": x_cal})
    act_norm = np.sqrt(np.mean(np.square(r_cal.astype(np.float64)), axis=0))
    base_importance = np.sqrt(
        np.square(np.linalg.norm(wf.astype(np.float64), axis=0))
        + np.square(np.linalg.norm(ws.astype(np.float64), axis=0))
    )
    importance = base_importance * np.maximum(act_norm, 1e-8)
    keep = np.sort(np.argsort(-importance)[: C // 2])

    pruned = onnxsim.apply_structured_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned)

    oracle = _matmul_residual_diamond_model(wf[:, keep], ws[:, keep], wout[keep, :])
    rng_x = np.random.default_rng(104)
    x = rng_x.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


# --- apply_structured_pruning: MatMul/Gemm residual via SkipLayerNormalization ---
#
# The realistic shape the MatMul/Gemm residual tests above rarely see in
# practice: a transformer already run through onnxruntime's own
# transformer-optimizer tool, which fuses each residual `Add` (plus an
# optional per-channel bias `Add`) together with the *following*
# LayerNorm/RMSNorm into one `com.microsoft::SkipLayerNormalization`/
# `SkipSimplifiedLayerNormalization` node -- see
# `_match_matmul_residual_merge`'s own docstring and this module's "MatMul/
# Gemm residual" section comment for the exact fused arithmetic (confirmed
# against onnxruntime's own `skip_layer_norm.cc` kernel source and by
# direct execution before any of this was written).


def _skip_layer_norm_residual_diamond_model(
    wf, ws, wout, gamma, beta=None, bias=None, simplified=False, epsilon=1e-5
):
    # y = SkipLayerNormalization(MatMul_f(X), MatMul_s(X), gamma, beta?,
    # bias?) -- the SkipLayerNormalization/SkipSimplifiedLayerNormalization
    # analogue of _matmul_residual_diamond_model: two entirely independent
    # MatMul producers merge via the fused node instead of a bare `Add`, and
    # must therefore still share one surviving channel-index set, feeding
    # one real consumer. `beta`/`bias` are each included only if given --
    # `beta` absent but `bias` present (SkipLayerNormalization only; the
    # simplified/RMSNorm variant has no `beta` at all) uses the onnx text
    # format's positional-placeholder syntax (an empty operand) to reach
    # `bias`'s own input index with `beta` skipped, exactly the way this
    # file's own GroupQueryAttention model builders already skip an unused
    # optional input.
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
    # The conflicting-importance construction every test below uses is only
    # doing its job if the combined keep set actually straddles both halves.
    assert np.any(keep < C // 2) and np.any(keep >= C // 2)
    return keep


def _conflicting_wf_ws(seed, K, C):
    rng = np.random.default_rng(seed)
    scale_f = np.where(np.arange(C) < C // 2, 3.0, 0.3).astype(np.float32)
    scale_s = np.where(np.arange(C) < C // 2, 0.3, 3.0).astype(np.float32)
    wf = rng.standard_normal((K, C)).astype(np.float32) * scale_f
    ws = rng.standard_normal((K, C)).astype(np.float32) * scale_s
    return rng, wf, ws


def test_structured_pruning_skip_layer_norm_residual_shrinks_matched_layers():
    K, C, Out = 8, 16, 4
    rng = np.random.default_rng(110)
    wf = rng.standard_normal((K, C)).astype(np.float32)
    ws = rng.standard_normal((K, C)).astype(np.float32)
    wout = rng.standard_normal((C, Out)).astype(np.float32)
    gamma = rng.standard_normal((C,)).astype(np.float32)
    beta = rng.standard_normal((C,)).astype(np.float32)
    model = _skip_layer_norm_residual_diamond_model(wf, ws, wout, gamma, beta=beta)

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["WF"].dims) == [K, C // 2]
    assert list(inits["WS"].dims) == [K, C // 2]
    assert list(inits["WOUT"].dims) == [C // 2, Out]
    assert list(inits["Gamma"].dims) == [C // 2]
    assert list(inits["Beta"].dims) == [C // 2]


def test_structured_pruning_skip_layer_norm_residual_matches_oracle():
    # Correctness bar: exact equivalence to hand-slicing both independent
    # MatMul producers *and* Gamma/Beta to the same combined-importance keep
    # set -- with weights deliberately built so the two branches disagree
    # about which channels matter most, so a bug that used only one
    # branch's importance (or forgot to slice Gamma/Beta) would be caught,
    # including by onnx.checker (a missed Gamma/Beta slice is a shape
    # mismatch against the now-pruned MatMul outputs) and, more precisely,
    # by the numeric oracle comparison below (a missed slice that
    # onnxruntime's broadcasting rules happened to tolerate anyway would
    # still compute the wrong per-channel scale/shift).
    K, C, Out = 8, 16, 4
    rng, wf, ws = _conflicting_wf_ws(110, K, C)
    wout = rng.standard_normal((C, Out)).astype(np.float32)
    gamma = rng.standard_normal((C,)).astype(np.float32)
    beta = rng.standard_normal((C,)).astype(np.float32)
    model = _skip_layer_norm_residual_diamond_model(wf, ws, wout, gamma, beta=beta)

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    keep = _skip_layer_norm_keep(wf, ws, C)
    oracle = _skip_layer_norm_residual_diamond_model(
        wf[:, keep], ws[:, keep], wout[keep, :], gamma[keep], beta=beta[keep]
    )

    rng_x = np.random.default_rng(111)
    x = rng_x.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_structured_pruning_skip_simplified_layer_norm_residual_matches_oracle():
    # SkipSimplifiedLayerNormalization -- the RMSNorm variant LLaMA-style
    # models use -- drops `beta`/mean-centering entirely (see this module's
    # own docstring and the "MatMul/Gemm residual" section comment for the
    # exact RMSNorm arithmetic); this is the same oracle bar as the plain
    # SkipLayerNormalization test above, minus `beta`.
    K, C, Out = 8, 16, 4
    rng, wf, ws = _conflicting_wf_ws(112, K, C)
    wout = rng.standard_normal((C, Out)).astype(np.float32)
    gamma = rng.standard_normal((C,)).astype(np.float32)
    model = _skip_layer_norm_residual_diamond_model(
        wf, ws, wout, gamma, simplified=True
    )

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
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


def test_structured_pruning_skip_layer_norm_residual_with_bias_matches_oracle():
    # `bias` present (and, deliberately, `beta` absent -- SkipLayerNorm's
    # own optional inputs are independent of each other): exercises the
    # bias-idx-shift in `_skip_layer_norm_const_names` (bias lives at input
    # index 4 when beta is declared, but the model builder above still
    # reaches it via the parser's positional-placeholder syntax when beta
    # is skipped) and confirms `Bias` is sliced correctly alongside `Gamma`.
    K, C, Out = 8, 16, 4
    rng, wf, ws = _conflicting_wf_ws(114, K, C)
    wout = rng.standard_normal((C, Out)).astype(np.float32)
    gamma = rng.standard_normal((C,)).astype(np.float32)
    bias = rng.standard_normal((C,)).astype(np.float32)
    model = _skip_layer_norm_residual_diamond_model(wf, ws, wout, gamma, bias=bias)

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
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


def test_structured_pruning_skip_layer_norm_residual_declines_on_nonconstant_beta():
    # `Beta` is a graph input, not a constant initializer -- `gamma` (also
    # required) is fine, but a *present* non-constant `beta` still means
    # this pass can't slice it, so the whole chain is declined and the
    # model is left byte-identical, the same conservative bar a
    # non-constant Gemm bias already gets elsewhere in this module.
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

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    assert pruned.SerializeToString() == model.SerializeToString()


def test_structured_pruning_skip_layer_norm_residual_declines_on_consumed_mean_output():
    # The training-only `mean` output (index 1) is actually consumed here
    # (wired straight to a second graph output) -- onnxruntime's own CPU
    # kernel never actually populates it (see this module's own docstring),
    # and this pass has no basis for whether pruning keeps it meaningful for
    # whatever reads it, so the whole chain is declined outright rather
    # than guessed at, leaving the model byte-identical.
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

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    assert pruned.SerializeToString() == model.SerializeToString()


def test_structured_pruning_skip_layer_norm_residual_declines_on_consumed_sum_output():
    # The fourth output, `input_skip_bias_sum` (the raw, pre-normalization
    # `f + s`), is consumed directly here by a second graph output. Unlike
    # `mean`/`inv_std_var`, this pass never reads this output itself and its
    # *value* would still be correct post-pruning (it's a plain runtime sum
    # of two already-consistently-pruned tensors) -- but its *shape* shrinks
    # along with `f`/`s`, and this pass has no way to confirm the outside
    # consumer (here, the graph's own declared output shape) still expects
    # the new, narrower width. Declined outright, model left byte-identical
    # -- confirmed to actually matter: before this decline existed, pruning
    # produced a model whose `SumOut` graph output disagreed with its own
    # declared shape (a lenient-merge warning from onnxruntime, and a hard
    # failure for any stricter consumer of that output elsewhere).
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

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    assert pruned.SerializeToString() == model.SerializeToString()


def test_structured_wanda_pruning_skip_layer_norm_residual_matches_oracle():
    # apply_structured_wanda_pruning picks up SkipLayerNormalization-fused
    # residual grouping for free -- _find_matmul_residual_chains is shared
    # with apply_structured_pruning, and the activation norm is captured at
    # the same probe point (`chain.consumer_node.input[0]`) any other
    # MatMul/Gemm chain uses.
    K, C, Out = 8, 16, 4
    rng = np.random.default_rng(118)
    wf = rng.standard_normal((K, C)).astype(np.float32)
    ws = rng.standard_normal((K, C)).astype(np.float32)
    wout = rng.standard_normal((C, Out)).astype(np.float32)
    gamma = rng.standard_normal((C,)).astype(np.float32)
    beta = rng.standard_normal((C,)).astype(np.float32)
    model = _skip_layer_norm_residual_diamond_model(wf, ws, wout, gamma, beta=beta)

    probe_model = onnx.ModelProto()
    probe_model.CopyFrom(model)
    probe_model.graph.output.append(
        onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, None)
    )

    rng_cal = np.random.default_rng(119)
    x_cal = rng_cal.standard_normal((6, K)).astype(np.float32)
    calibration_data = [{"X": x_cal}]

    _, y_cal = _run(probe_model, {"X": x_cal})
    act_norm = np.sqrt(np.mean(np.square(y_cal.astype(np.float64)), axis=0))
    base_importance = np.sqrt(
        np.square(np.linalg.norm(wf.astype(np.float64), axis=0))
        + np.square(np.linalg.norm(ws.astype(np.float64), axis=0))
    )
    importance = base_importance * np.maximum(act_norm, 1e-8)
    keep = np.sort(np.argsort(-importance)[: C // 2])

    pruned = onnxsim.apply_structured_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned)

    oracle = _skip_layer_norm_residual_diamond_model(
        wf[:, keep], ws[:, keep], wout[keep, :], gamma[keep], beta=beta[keep]
    )
    rng_x = np.random.default_rng(120)
    x = rng_x.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_structured_pruning_mixed_add_and_skip_layer_norm_spine_matches_oracle():
    # A transitive spine of *two different* merge-node kinds sharing one
    # channel count: a bare `Add` (f1, s1) feeds forward as one operand of a
    # downstream `SkipLayerNormalization` merge (with f2 as the other) --
    # the "many residual blocks share one spine" case, but mixing an
    # ordinary transformer-block Add-residual with a fused post-LN block
    # right after it, exactly the shape a real model transitioning between
    # an un-fused and a fused block would take. `_walk_matmul_producer_backward`'s
    # own "resolves to another eligible merge node's raw output" case
    # doesn't care which kind of node it resolves to -- confirmed here with
    # a genuinely mixed pair rather than two of the same kind.
    K, C, Kz, Out = 8, 16, 5, 4
    rng = np.random.default_rng(121)
    wf1 = rng.standard_normal((K, C)).astype(np.float32)
    ws1 = rng.standard_normal((K, C)).astype(np.float32)
    wf2 = rng.standard_normal((Kz, C)).astype(np.float32)
    wout = rng.standard_normal((C, Out)).astype(np.float32)
    gamma = rng.standard_normal((C,)).astype(np.float32)

    def _build(wf1, ws1, wf2, wout, gamma):
        return _model(
            f"""
            g (float[batch,{wf1.shape[0]}] X, float[batch,{wf2.shape[0]}] Z) => (float[batch,{wout.shape[1]}] Y)
            {{
              f1 = MatMul(X, WF1)
              s1 = MatMul(X, WS1)
              add1 = Add(f1, s1)
              f2 = MatMul(Z, WF2)
              merged, mean, inv_std, sbs = com.microsoft.SkipSimplifiedLayerNormalization <epsilon=1e-5> (f2, add1, Gamma)
              Y = MatMul(merged, WOUT)
            }}
            """,
            initializer=[
                _f32(wf1, "WF1"),
                _f32(ws1, "WS1"),
                _f32(wf2, "WF2"),
                _f32(wout, "WOUT"),
                _f32(gamma, "Gamma"),
            ],
            opset=17,
        )

    model = _build(wf1, ws1, wf2, wout, gamma)
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    importance = np.sqrt(
        np.square(np.linalg.norm(wf1.astype(np.float64), axis=0))
        + np.square(np.linalg.norm(ws1.astype(np.float64), axis=0))
        + np.square(np.linalg.norm(wf2.astype(np.float64), axis=0))
    )
    keep = np.sort(np.argsort(-importance)[: C // 2])
    oracle = _build(
        wf1[:, keep], ws1[:, keep], wf2[:, keep], wout[keep, :], gamma[keep]
    )
    oracle.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))

    rng_x = np.random.default_rng(122)
    x = rng_x.standard_normal((5, K)).astype(np.float32)
    z = rng_x.standard_normal((5, Kz)).astype(np.float32)
    (y,) = _run(pruned, {"X": x, "Z": z})
    (y_oracle,) = _run(oracle, {"X": x, "Z": z})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


# --- apply_structured_pruning: Concat-merged (skip-connection) chains -------
#
# See onnxsim.pruning's own module docstring and the "Concat-merged
# (skip-connection) chains" section comment above
# _find_matmul_concat_chains/_find_conv_concat_chains for the full
# reasoning. Unlike every merge kind tested above (a gated pair, an Add
# residual, a SkipLayerNormalization-fused residual -- all of which force
# every branch onto one *shared* keep set), a Concat's branches are
# independent: each owns a fixed, disjoint slice of the merged channel
# range and is ranked/pruned entirely on its own. The tests below are
# deliberately built to catch a "treated it like a residual merge" bug --
# one branch's weights scaled far larger than another's, or branches with
# different channel counts -- since a shared-keep-set bug would silently
# starve or ignore one branch, while independent per-branch selection
# always keeps each branch's own top fraction regardless of the other
# branch's scale.


def _matmul_concat_model(weights, w_out, axis=-1):
    # merged = Concat(MatMul(X, W0), MatMul(X, W1), ..., axis=axis);
    # Y = MatMul(merged, WOUT) -- an arbitrary (2 or more) number of
    # independent MatMul producers merge via Concat, each keeping its own
    # disjoint slice of the merged channel range, feeding one real
    # consumer. `weights[i]` is `[K, Ci]`; `w_out` is `[sum(Ci), Out]`.
    K = weights[0].shape[0]
    Out = w_out.shape[1]
    initializer = []
    names = []
    lines = []
    for i, w in enumerate(weights):
        wname = f"W{i}"
        initializer.append(_f32(w, wname))
        lines.append(f"h{i} = MatMul(X, {wname})")
        names.append(f"h{i}")
    lines.append(f"merged = Concat<axis={axis}>({', '.join(names)})")
    lines.append("Y = MatMul(merged, WOUT)")
    initializer.append(_f32(w_out, "WOUT"))
    body = "\n          ".join(lines)
    return _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          {body}
        }}
        """,
        initializer=initializer,
    )


def _conv_concat_model(weights, w_out, axis=1, spatial=10):
    # The Conv analogue of _matmul_concat_model -- the U-Net-style
    # encoder/decoder merge this whole section exists for. `weights[i]` is
    # `[Ci, Cin, kH, kW]`; `w_out` is `[Cout, sum(Ci), kH, kW]`.
    Cin = weights[0].shape[1]
    Cout = w_out.shape[0]
    initializer = []
    names = []
    lines = []
    for i, w in enumerate(weights):
        wname = f"W{i}"
        initializer.append(_f32(w, wname))
        lines.append(f"h{i} = Conv<kernel_shape=[3,3]>(X, {wname})")
        names.append(f"h{i}")
    lines.append(f"merged = Concat<axis={axis}>({', '.join(names)})")
    lines.append("Y = Conv<kernel_shape=[3,3]>(merged, WOUT)")
    initializer.append(_f32(w_out, "WOUT"))
    out_spatial = spatial - 4
    body = "\n          ".join(lines)
    return _model(
        f"""
        g (float[N,{Cin},{spatial},{spatial}] X) => (float[N,{Cout},{out_spatial},{out_spatial}] Y)
        {{
          {body}
        }}
        """,
        initializer=initializer,
    )


def test_structured_pruning_matmul_concat_prunes_each_branch_to_its_own_count():
    # Ca != Cb, same sparsity fraction -- proves each branch is sized from
    # its *own* channel count (round(Ca*(1-s)) vs round(Cb*(1-s))), not one
    # shared count the way a gated/residual merge's shared keep set would
    # force. keep_a = 10 - round(10*0.5) = 5; keep_b = 6 - round(6*0.5) = 3.
    K, Ca, Cb, Out = 8, 10, 6, 4
    rng = np.random.default_rng(200)
    wa = rng.standard_normal((K, Ca)).astype(np.float32)
    wb = rng.standard_normal((K, Cb)).astype(np.float32)
    wout = rng.standard_normal((Ca + Cb, Out)).astype(np.float32)
    model = _matmul_concat_model([wa, wb], wout)

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W0"].dims) == [K, 5]
    assert list(inits["W1"].dims) == [K, 3]
    assert list(inits["WOUT"].dims) == [8, Out]


def test_structured_pruning_matmul_concat_matches_oracle_no_cross_branch_coupling():
    # Branch a's weights are scaled 10x branch b's -- a bug that (wrongly)
    # combined both branches into one shared importance ranking, the way
    # _find_matmul_residual_chains/_find_gated_chains do, would keep every
    # one of branch a's columns and none of branch b's (a's smallest column
    # still dwarfs b's largest). Correct independent per-branch ranking
    # keeps each branch's own top half regardless of the other branch's
    # scale -- confirmed both by an explicit non-empty/expected-set check on
    # branch b's own survivors and by the full onnxruntime oracle match.
    K, C, Out = 8, 8, 4
    rng = np.random.default_rng(201)
    wa = rng.standard_normal((K, C)).astype(np.float32) * 10.0
    wb = rng.standard_normal((K, C)).astype(np.float32)
    wout = rng.standard_normal((2 * C, Out)).astype(np.float32)
    model = _matmul_concat_model([wa, wb], wout)

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    keep_a = np.sort(
        np.argsort(-np.linalg.norm(wa.astype(np.float64), axis=0))[: C // 2]
    )
    keep_b = np.sort(
        np.argsort(-np.linalg.norm(wb.astype(np.float64), axis=0))[: C // 2]
    )
    assert len(keep_b) == C // 2  # branch b kept its own top half, not starved to 0
    global_keep = np.concatenate([keep_a, keep_b + C])

    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W0"], wa[:, keep_a])
    np.testing.assert_array_equal(inits["W1"], wb[:, keep_b])
    np.testing.assert_array_equal(inits["WOUT"], wout[global_keep, :])

    oracle = _matmul_concat_model([wa[:, keep_a], wb[:, keep_b]], wout[global_keep, :])
    rng_x = np.random.default_rng(202)
    x = rng_x.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_structured_pruning_matmul_concat_three_branches_matches_oracle():
    # N-ary Concat (three branches, three different channel counts) -- not
    # fixed at two operands the way an Add merge is.
    K, Ca, Cb, Cc, Out = 6, 8, 4, 6, 3
    rng = np.random.default_rng(203)
    wa = rng.standard_normal((K, Ca)).astype(np.float32)
    wb = rng.standard_normal((K, Cb)).astype(np.float32)
    wc = rng.standard_normal((K, Cc)).astype(np.float32)
    wout = rng.standard_normal((Ca + Cb + Cc, Out)).astype(np.float32)
    model = _matmul_concat_model([wa, wb, wc], wout)

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    keep_a = np.sort(np.argsort(-np.linalg.norm(wa.astype(np.float64), axis=0))[:4])
    keep_b = np.sort(np.argsort(-np.linalg.norm(wb.astype(np.float64), axis=0))[:2])
    keep_c = np.sort(np.argsort(-np.linalg.norm(wc.astype(np.float64), axis=0))[:3])
    global_keep = np.concatenate([keep_a, keep_b + Ca, keep_c + Ca + Cb])
    oracle = _matmul_concat_model(
        [wa[:, keep_a], wb[:, keep_b], wc[:, keep_c]], wout[global_keep, :]
    )

    rng_x = np.random.default_rng(204)
    x = rng_x.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_structured_pruning_matmul_concat_branch_activation_matches_oracle():
    # Each branch's own Relu (between its producer's raw output and the
    # Concat operand) is carried on that branch's own `pre_ops` -- exercised
    # here alongside a post-Concat Sigmoid (an ordinary _walk_to_consumer
    # hop, unrelated to the Concat machinery itself) to confirm both compose.
    K, Ca, Cb, Out = 8, 8, 6, 4
    rng = np.random.default_rng(205)
    wa = rng.standard_normal((K, Ca)).astype(np.float32)
    wb = rng.standard_normal((K, Cb)).astype(np.float32)
    wout = rng.standard_normal((Ca + Cb, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          ha = MatMul(X, WA)
          aa = Relu(ha)
          hb = MatMul(X, WB)
          ab = Relu(hb)
          merged = Concat<axis=-1>(aa, ab)
          s = Sigmoid(merged)
          Y = MatMul(s, WOUT)
        }}
        """,
        initializer=[_f32(wa, "WA"), _f32(wb, "WB"), _f32(wout, "WOUT")],
    )

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    keep_a = np.sort(
        np.argsort(-np.linalg.norm(wa.astype(np.float64), axis=0))[: Ca // 2]
    )
    keep_b = np.sort(
        np.argsort(-np.linalg.norm(wb.astype(np.float64), axis=0))[: Cb // 2]
    )
    global_keep = np.concatenate([keep_a, keep_b + Ca])
    oracle = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          ha = MatMul(X, WA)
          aa = Relu(ha)
          hb = MatMul(X, WB)
          ab = Relu(hb)
          merged = Concat<axis=-1>(aa, ab)
          s = Sigmoid(merged)
          Y = MatMul(s, WOUT)
        }}
        """,
        initializer=[
            _f32(wa[:, keep_a], "WA"),
            _f32(wb[:, keep_b], "WB"),
            _f32(wout[global_keep, :], "WOUT"),
        ],
    )

    rng_x = np.random.default_rng(206)
    x = rng_x.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_structured_wanda_pruning_matmul_concat_matches_oracle():
    # Confirms the exact ||W_row||_2 * ||X||_2 formula and probe point
    # (each branch's own operand feeding the Concat node, captured
    # independently -- not the shared downstream consumer, and not mixed
    # with the other branch's activation): weight columns are deliberately
    # scaled so the two branches' importances differ, and the pruned
    # result is checked bit-for-bit against a hand-computed oracle using
    # the real captured activations, the same correctness bar every other
    # "matches_oracle" test in this module holds to.
    K, Ca, Cb, Out = 6, 6, 6, 4
    rng = np.random.default_rng(207)
    wa = rng.standard_normal((K, Ca)).astype(np.float32)
    wb = rng.standard_normal((K, Cb)).astype(np.float32)
    wa[:, : Ca // 2] *= 3.0  # weight-only ranking favors the first half ...
    wb[:, : Cb // 2] *= 3.0
    wout = rng.standard_normal((Ca + Cb, Out)).astype(np.float32)
    model = _matmul_concat_model([wa, wb], wout)

    probe_model = onnx.ModelProto()
    probe_model.CopyFrom(model)
    probe_model.graph.output.append(
        onnx.helper.make_tensor_value_info("h0", onnx.TensorProto.FLOAT, None)
    )
    probe_model.graph.output.append(
        onnx.helper.make_tensor_value_info("h1", onnx.TensorProto.FLOAT, None)
    )

    rng_cal = np.random.default_rng(208)
    x_cal = rng_cal.standard_normal((16, K)).astype(np.float32)
    calibration_data = [{"X": x_cal}]
    _, h0_cal, h1_cal = _run(probe_model, {"X": x_cal})
    norm_a = np.sqrt(np.mean(np.square(h0_cal.astype(np.float64)), axis=0))
    norm_b = np.sqrt(np.mean(np.square(h1_cal.astype(np.float64)), axis=0))

    pruned = onnxsim.apply_structured_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned)

    importance_a = np.linalg.norm(wa.astype(np.float64), axis=0) * np.maximum(
        norm_a, 1e-8
    )
    importance_b = np.linalg.norm(wb.astype(np.float64), axis=0) * np.maximum(
        norm_b, 1e-8
    )
    keep_a = np.sort(np.argsort(-importance_a)[: Ca // 2])
    keep_b = np.sort(np.argsort(-importance_b)[: Cb // 2])
    global_keep = np.concatenate([keep_a, keep_b + Ca])
    oracle = _matmul_concat_model([wa[:, keep_a], wb[:, keep_b]], wout[global_keep, :])

    rng_x = np.random.default_rng(209)
    x = rng_x.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_structured_wanda_pruning_matmul_concat_protects_low_weight_high_activation_channel_per_branch():
    # The structured analogue of Wanda's own motivating scenario
    # (test_structured_wanda_pruning_protects_channels_with_small_weight_but_large_activation
    # above), replayed independently on *each* branch of a Concat: branch
    # a's own column 0 has a deliberately tiny weight (so plain L2-norm
    # pruning drops it) but responds only to input feature k0, which
    # calibration data makes huge -- and branch b's own column 0 is the
    # same construction against a *different* feature k1. Both must be
    # protected independently for this to pass: a bug that captured only
    # one branch's activation (or swapped the two, or fell back to probing
    # the shared downstream consumer) would protect at most one of them.
    K, Ca, Cb = 8, 6, 6
    k0, k1 = 0, 1
    small_scale = 0.4  # matches the single-branch test's own scale
    rng = np.random.default_rng(207)
    wa = rng.standard_normal((K, Ca)).astype(np.float32) * 0.5
    wb = rng.standard_normal((K, Cb)).astype(np.float32) * 0.5
    # Every *non*-salient column, on both branches, is barred from
    # responding to either amplified feature at all -- otherwise, since
    # both branches share the same input X, an ordinary column with a
    # random (uncontrolled) coefficient on k0/k1 would pick up the same
    # amplification and swamp the deliberately small salient column's own
    # importance, defeating the decoupling this test depends on. Branch a's
    # own salient column (0) is then zeroed and given a single small tap on
    # k0 alone; branch b's mirrors that against k1.
    wa[k1, :] = 0.0  # branch a never responds to b's own amplified feature
    wa[k0, 1:] = 0.0  # only branch a's own salient column responds to k0
    wa[:, 0] = 0.0
    wa[k0, 0] = small_scale
    wb[k0, :] = 0.0  # branch b never responds to a's own amplified feature
    wb[k1, 1:] = 0.0  # only branch b's own salient column responds to k1
    wb[:, 0] = 0.0
    wb[k1, 0] = small_scale
    wout = rng.standard_normal((Ca + Cb, 4)).astype(np.float32)
    model = _matmul_concat_model([wa, wb], wout)

    x = rng.standard_normal((64, K)).astype(np.float32)
    x[:, k0] *= 80.0
    x[:, k1] *= 80.0
    calibration_data = [{"X": x}]

    plain = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    wanda = onnxsim.apply_structured_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(plain)
    onnx.checker.check_model(wanda)

    assert 0 not in _kept_columns(plain, "W0", wa)
    assert 0 in _kept_columns(wanda, "W0", wa)
    assert 0 not in _kept_columns(plain, "W1", wb)
    assert 0 in _kept_columns(wanda, "W1", wb)


def test_structured_wanda_pruning_matmul_concat_composed_residual_branch_matches_oracle():
    # The Wanda-calibrated analogue of
    # test_structured_pruning_matmul_concat_composes_with_residual_merge_branch_matches_oracle:
    # confirms the composed branch's own Wanda probe point is exactly where
    # the residual group's own combined output (`addr`) feeds the `Concat`
    # node -- not the shared downstream consumer, and not either individual
    # producer's own raw output -- and that the base (weight-only) term
    # combines both leaf producers' own per-row norms via the same
    # root-sum-square formula _plain_structured_importance already uses for
    # an ordinary multi-producer chain, exactly as
    # _plain_branch_importance now does for a composed branch.
    K, C, Cb, Out = 8, 8, 6, 3
    rng = np.random.default_rng(221)
    wf = rng.standard_normal((K, C)).astype(np.float32)
    ws = rng.standard_normal((K, C)).astype(np.float32)
    wb = rng.standard_normal((K, Cb)).astype(np.float32)
    wout = rng.standard_normal((C + Cb, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          f = MatMul(X, WF)
          s = MatMul(X, WS)
          addr = Add(f, s)
          hb = MatMul(X, WB)
          merged = Concat<axis=-1>(addr, hb)
          Y = MatMul(merged, WOUT)
        }}
        """,
        initializer=[
            _f32(wf, "WF"),
            _f32(ws, "WS"),
            _f32(wb, "WB"),
            _f32(wout, "WOUT"),
        ],
    )

    probe_model = onnx.ModelProto()
    probe_model.CopyFrom(model)
    probe_model.graph.output.append(
        onnx.helper.make_tensor_value_info("addr", onnx.TensorProto.FLOAT, None)
    )
    probe_model.graph.output.append(
        onnx.helper.make_tensor_value_info("hb", onnx.TensorProto.FLOAT, None)
    )

    rng_cal = np.random.default_rng(222)
    x_cal = rng_cal.standard_normal((16, K)).astype(np.float32)
    calibration_data = [{"X": x_cal}]
    _, addr_cal, hb_cal = _run(probe_model, {"X": x_cal})
    norm_r = np.sqrt(np.mean(np.square(addr_cal.astype(np.float64)), axis=0))
    norm_b = np.sqrt(np.mean(np.square(hb_cal.astype(np.float64)), axis=0))

    pruned = onnxsim.apply_structured_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned)

    base_r = np.sqrt(
        np.square(np.linalg.norm(wf.astype(np.float64), axis=0))
        + np.square(np.linalg.norm(ws.astype(np.float64), axis=0))
    )
    importance_r = base_r * np.maximum(norm_r, 1e-8)
    importance_b = np.linalg.norm(wb.astype(np.float64), axis=0) * np.maximum(
        norm_b, 1e-8
    )
    keep_r = np.sort(np.argsort(-importance_r)[: C // 2])
    keep_b = np.sort(np.argsort(-importance_b)[: Cb // 2])
    global_keep = np.concatenate([keep_r, keep_b + C])

    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["WF"], wf[:, keep_r])
    np.testing.assert_array_equal(inits["WS"], ws[:, keep_r])
    np.testing.assert_array_equal(inits["WB"], wb[:, keep_b])
    np.testing.assert_array_equal(inits["WOUT"], wout[global_keep, :])

    oracle = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          f = MatMul(X, WF)
          s = MatMul(X, WS)
          addr = Add(f, s)
          hb = MatMul(X, WB)
          merged = Concat<axis=-1>(addr, hb)
          Y = MatMul(merged, WOUT)
        }}
        """,
        initializer=[
            _f32(wf[:, keep_r], "WF"),
            _f32(ws[:, keep_r], "WS"),
            _f32(wb[:, keep_b], "WB"),
            _f32(wout[global_keep, :], "WOUT"),
        ],
    )
    rng_x = np.random.default_rng(223)
    x = rng_x.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_structured_pruning_matmul_concat_declines_on_positive_axis_unknown_rank():
    # `axis=1` on a 2-D [batch, C] tensor is numerically the same as
    # `axis=-1`, but this bare hand-built graph carries no value_info at
    # all for the Concat operands (h0/h1 -- no shape-inference pass ever
    # ran over it), so their rank can't be confirmed and the positive axis
    # is declined rather than guessed at -- left completely untouched, even
    # though this particular instance would in fact have been safe. See
    # test_structured_pruning_matmul_concat_accepts_positive_last_axis_when_rank_known
    # for the same topology once the rank *is* confirmable.
    K, Ca, Cb, Out = 8, 6, 4, 3
    rng = np.random.default_rng(210)
    wa = rng.standard_normal((K, Ca)).astype(np.float32)
    wb = rng.standard_normal((K, Cb)).astype(np.float32)
    wout = rng.standard_normal((Ca + Cb, Out)).astype(np.float32)
    model = _matmul_concat_model([wa, wb], wout, axis=1)
    assert len(model.graph.value_info) == 0  # no rank annotation to piggyback on

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W0"], wa)
    np.testing.assert_array_equal(inits["W1"], wb)
    np.testing.assert_array_equal(inits["WOUT"], wout)


def test_structured_pruning_matmul_concat_declines_on_positive_axis_shape_unknown_rank():
    # Same as the unknown-rank test above, but this time the operands *do*
    # carry a value_info entry each (as a partially shape-inferred graph
    # might, e.g. a symbolic-rank input propagated through) -- just one
    # with no `shape` field at all, ONNX's own "rank not statically known"
    # spelling (see CLAUDE.md's own note on this pattern). _tensor_rank must
    # treat that exactly like no annotation at all, not crash on it or
    # (worse) treat a present-but-empty shape as rank 0.
    K, Ca, Cb, Out = 8, 6, 4, 3
    rng = np.random.default_rng(214)
    wa = rng.standard_normal((K, Ca)).astype(np.float32)
    wb = rng.standard_normal((K, Cb)).astype(np.float32)
    wout = rng.standard_normal((Ca + Cb, Out)).astype(np.float32)
    model = _matmul_concat_model([wa, wb], wout, axis=1)
    for name in ("h0", "h1"):
        vi = onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [1, 1])
        vi.type.tensor_type.ClearField("shape")
        model.graph.value_info.append(vi)

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W0"], wa)
    np.testing.assert_array_equal(inits["W1"], wb)
    np.testing.assert_array_equal(inits["WOUT"], wout)


def test_structured_pruning_matmul_concat_accepts_positive_last_axis_when_rank_known():
    # `axis=1` on 2-D [batch, C] tensors is numerically identical to
    # `axis=-1`, and once the graph carries value_info confirming each
    # operand's rank -- as it would after a real shape-inference pass (e.g.
    # onnxsim's own) ran earlier in the pipeline, the ordinary case
    # structured pruning is meant to run in -- this pass now recognizes and
    # prunes it exactly the same as the equivalent `axis=-1` graph: same
    # kept columns, same values, same runtime output. This is the
    # "genuinely-safe-but-unconfirmed" case
    # test_structured_pruning_matmul_concat_declines_on_positive_axis_unknown_rank
    # documents becoming confirmed once the rank is actually knowable.
    K, Ca, Cb, Out = 8, 6, 4, 3
    rng = np.random.default_rng(212)
    wa = rng.standard_normal((K, Ca)).astype(np.float32)
    wb = rng.standard_normal((K, Cb)).astype(np.float32)
    wout = rng.standard_normal((Ca + Cb, Out)).astype(np.float32)

    model_neg = _matmul_concat_model([wa, wb], wout, axis=-1)
    model_pos = _matmul_concat_model([wa, wb], wout, axis=1)
    model_pos = onnx.shape_inference.infer_shapes(model_pos)
    h0_vi = next(vi for vi in model_pos.graph.value_info if vi.name == "h0")
    assert len(h0_vi.type.tensor_type.shape.dim) == 2  # rank actually confirmed

    pruned_neg = onnxsim.apply_structured_pruning(model_neg, sparsity=0.5)
    pruned_pos = onnxsim.apply_structured_pruning(model_pos, sparsity=0.5)
    onnx.checker.check_model(pruned_neg)
    onnx.checker.check_model(pruned_pos)

    inits_neg = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned_neg.graph.initializer
    }
    inits_pos = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned_pos.graph.initializer
    }
    for name in ("W0", "W1", "WOUT"):
        np.testing.assert_array_equal(inits_pos[name], inits_neg[name])
    # Confirms it actually pruned (not merely "happened to match because
    # both were left untouched"): the branch's output channel count shrank.
    assert inits_pos["W0"].shape[1] < Ca

    rng_x = np.random.default_rng(213)
    x = rng_x.standard_normal((5, K)).astype(np.float32)
    (y_neg,) = _run(pruned_neg, {"X": x})
    (y_pos,) = _run(pruned_pos, {"X": x})
    np.testing.assert_allclose(y_pos, y_neg, rtol=1e-5, atol=1e-5)


def test_structured_pruning_matmul_concat_declines_on_positive_non_last_axis():
    # `axis=0` on rank-2 [batch, C] operands is confirmably *not* the last
    # axis (`rank - 1 == 1`) -- proves _concat_axis_is_last actually checks
    # the axis against the rank rather than accepting any positive axis
    # once a rank is confirmable. Rank is attached directly (rather than
    # via a real onnx.shape_inference.infer_shapes() pass, as the sibling
    # accept/decline tests above do) since `axis=0` here doesn't actually
    # describe a dimensionally-valid Concat (Ca != Cb along the
    # non-concat axis) -- irrelevant to what this test isolates, since the
    # axis check runs, and this whole group is declined, before any
    # producer/consumer walk ever inspects that.
    K, Ca, Cb, Out = 8, 6, 4, 3
    rng = np.random.default_rng(215)
    wa = rng.standard_normal((K, Ca)).astype(np.float32)
    wb = rng.standard_normal((K, Cb)).astype(np.float32)
    wout = rng.standard_normal((Ca + Cb, Out)).astype(np.float32)
    model = _matmul_concat_model([wa, wb], wout, axis=0)
    for name in ("h0", "h1"):
        model.graph.value_info.append(
            onnx.helper.make_tensor_value_info(
                name, onnx.TensorProto.FLOAT, [None, None]
            )
        )

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W0"], wa)
    np.testing.assert_array_equal(inits["W1"], wb)
    np.testing.assert_array_equal(inits["WOUT"], wout)


def test_structured_pruning_matmul_concat_declines_on_branch_fan_out():
    # Branch a's own raw output feeds both the Concat node *and* a second
    # graph output directly -- the same single-consumer safety bar every
    # other hop in this pass holds every intermediate tensor to. The whole
    # group is declined, branch b included, never partially pruned.
    K, Ca, Cb, Out = 8, 6, 4, 3
    rng = np.random.default_rng(211)
    wa = rng.standard_normal((K, Ca)).astype(np.float32)
    wb = rng.standard_normal((K, Cb)).astype(np.float32)
    wout = rng.standard_normal((Ca + Cb, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y, float[batch,{Ca}] Extra)
        {{
          ha = MatMul(X, WA)
          hb = MatMul(X, WB)
          merged = Concat<axis=-1>(ha, hb)
          Y = MatMul(merged, WOUT)
          Extra = Identity(ha)
        }}
        """,
        initializer=[_f32(wa, "WA"), _f32(wb, "WB"), _f32(wout, "WOUT")],
    )
    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["WA"], wa)
    np.testing.assert_array_equal(inits["WB"], wb)
    np.testing.assert_array_equal(inits["WOUT"], wout)


def test_structured_pruning_matmul_concat_declines_on_graph_input_branch():
    # One Concat operand is a graph input *directly* -- its only consumer is
    # the Concat node itself (so it passes the single-consumer check), but
    # it has no producing node at all, so the backward walk fails on that
    # operand and the whole group (including the *other*, otherwise-
    # prunable branch) is declined. X2 is a second, unrelated input so this
    # doesn't also (accidentally) exercise the fan-out decline path above.
    K, Cb, Out = 6, 4, 3
    rng = np.random.default_rng(212)
    wb = rng.standard_normal((K, Cb)).astype(np.float32)
    wout = rng.standard_normal((K + Cb, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X, float[batch,{K}] X2) => (float[batch,{Out}] Y)
        {{
          hb = MatMul(X2, WB)
          merged = Concat<axis=-1>(X, hb)
          Y = MatMul(merged, WOUT)
        }}
        """,
        initializer=[_f32(wb, "WB"), _f32(wout, "WOUT")],
    )
    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["WB"], wb)
    np.testing.assert_array_equal(inits["WOUT"], wout)


def test_structured_pruning_matmul_concat_composes_with_residual_merge_branch_matches_oracle():
    # One Concat operand is itself an eligible Add-residual merge point's
    # raw output, with no consumer anywhere else -- composed, not declined
    # (see this section's own comment on the `"add"` outcome): the merge's
    # own whole group (WF, WS) is resolved exactly as
    # _find_matmul_residual_chains would resolve it standalone, and the
    # group's own combined-importance keep set becomes this one branch's
    # own contribution, independent of the plain second branch (WB).
    # Deliberately adversarial on both axes at once: WF/WS individually
    # disagree about which half of the columns matter most (so only their
    # *combined* norm, not either producer's own, can be driving the
    # residual branch's keep set), and WB is scaled 10x larger so a bug that
    # (wrongly) mixed the two Concat branches' importances into one ranking
    # would starve WB's own columns entirely.
    K, C, Cb, Out = 8, 16, 6, 3
    rng = np.random.default_rng(213)
    scale_f = np.where(np.arange(C) < C // 2, 3.0, 0.3).astype(np.float32)
    scale_s = np.where(np.arange(C) < C // 2, 0.3, 3.0).astype(np.float32)
    wf = rng.standard_normal((K, C)).astype(np.float32) * scale_f
    ws = rng.standard_normal((K, C)).astype(np.float32) * scale_s
    wb = rng.standard_normal((K, Cb)).astype(np.float32) * 10.0
    wout = rng.standard_normal((C + Cb, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          f = MatMul(X, WF)
          s = MatMul(X, WS)
          addr = Add(f, s)
          hb = MatMul(X, WB)
          merged = Concat<axis=-1>(addr, hb)
          Y = MatMul(merged, WOUT)
        }}
        """,
        initializer=[
            _f32(wf, "WF"),
            _f32(ws, "WS"),
            _f32(wb, "WB"),
            _f32(wout, "WOUT"),
        ],
    )
    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    importance_r = np.sqrt(
        np.square(np.linalg.norm(wf.astype(np.float64), axis=0))
        + np.square(np.linalg.norm(ws.astype(np.float64), axis=0))
    )
    keep_r = np.sort(np.argsort(-importance_r)[: C // 2])
    # The conflicting-importance construction above is only doing its job
    # if the combined keep set actually straddles both halves.
    assert np.any(keep_r < C // 2) and np.any(keep_r >= C // 2)
    importance_b = np.linalg.norm(wb.astype(np.float64), axis=0)
    keep_b = np.sort(np.argsort(-importance_b)[: Cb // 2])
    assert len(keep_b) == Cb // 2  # branch b kept its own top half, not starved to 0
    global_keep = np.concatenate([keep_r, keep_b + C])

    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["WF"], wf[:, keep_r])
    np.testing.assert_array_equal(inits["WS"], ws[:, keep_r])
    np.testing.assert_array_equal(inits["WB"], wb[:, keep_b])
    np.testing.assert_array_equal(inits["WOUT"], wout[global_keep, :])

    oracle = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          f = MatMul(X, WF)
          s = MatMul(X, WS)
          addr = Add(f, s)
          hb = MatMul(X, WB)
          merged = Concat<axis=-1>(addr, hb)
          Y = MatMul(merged, WOUT)
        }}
        """,
        initializer=[
            _f32(wf[:, keep_r], "WF"),
            _f32(ws[:, keep_r], "WS"),
            _f32(wb[:, keep_b], "WB"),
            _f32(wout[global_keep, :], "WOUT"),
        ],
    )
    rng_x = np.random.default_rng(214)
    x = rng_x.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_structured_pruning_matmul_concat_composes_with_skip_layer_norm_branch_matches_oracle():
    # The composed branch's own merge point can be a fused
    # `SkipLayerNormalization` node, not just a bare `Add` -- its own
    # `gamma`/`beta` constants are folded into the composed branch's own
    # `pre_ops` (via `_match_matmul_residual_merge`'s own `extra_ops`,
    # reused unchanged inside `_resolve_matmul_residual_group_for_concat`)
    # and sliced by the branch's own local `keep`, exactly as they would be
    # for a standalone SkipLayerNormalization residual chain.
    K, C, Cb, Out = 8, 16, 6, 3
    rng, wf, ws = _conflicting_wf_ws(224, K, C)
    gamma = rng.standard_normal((C,)).astype(np.float32)
    beta = rng.standard_normal((C,)).astype(np.float32)
    wb = rng.standard_normal((K, Cb)).astype(np.float32) * 10.0
    wout = rng.standard_normal((C + Cb, Out)).astype(np.float32)

    def _build(wf_, ws_, gamma_, beta_, wb_, wout_):
        m = _model(
            f"""
            g (float[batch,{K}] X) => (float[batch,{Out}] Y)
            {{
              f = MatMul(X, WF)
              s = MatMul(X, WS)
              addr = com.microsoft.SkipLayerNormalization <epsilon=1e-5> (f, s, Gamma, Beta)
              hb = MatMul(X, WB)
              merged = Concat<axis=-1>(addr, hb)
              Y = MatMul(merged, WOUT)
            }}
            """,
            initializer=[
                _f32(wf_, "WF"),
                _f32(ws_, "WS"),
                _f32(gamma_, "Gamma"),
                _f32(beta_, "Beta"),
                _f32(wb_, "WB"),
                _f32(wout_, "WOUT"),
            ],
            opset=17,
        )
        m.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))
        return m

    model = _build(wf, ws, gamma, beta, wb, wout)
    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    keep_r = _skip_layer_norm_keep(wf, ws, C)
    importance_b = np.linalg.norm(wb.astype(np.float64), axis=0)
    keep_b = np.sort(np.argsort(-importance_b)[: Cb // 2])
    assert len(keep_b) == Cb // 2
    global_keep = np.concatenate([keep_r, keep_b + C])

    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["WF"], wf[:, keep_r])
    np.testing.assert_array_equal(inits["WS"], ws[:, keep_r])
    np.testing.assert_array_equal(inits["Gamma"], gamma[keep_r])
    np.testing.assert_array_equal(inits["Beta"], beta[keep_r])
    np.testing.assert_array_equal(inits["WB"], wb[:, keep_b])
    np.testing.assert_array_equal(inits["WOUT"], wout[global_keep, :])

    oracle = _build(
        wf[:, keep_r],
        ws[:, keep_r],
        gamma[keep_r],
        beta[keep_r],
        wb[:, keep_b],
        wout[global_keep, :],
    )
    rng_x = np.random.default_rng(225)
    x = rng_x.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-4, atol=1e-4)


def test_structured_pruning_matmul_concat_declines_on_residual_merge_branch_direct_fan_out():
    # Same shape as the composed test above, except `addr` (the residual
    # group's own sink, and the tensor directly feeding the `Concat`
    # operand) also feeds a second, ordinary consumer (`Z`) -- caught by the
    # same `_branch_walk_has_fanout` check an ordinary (non-composed) branch
    # is already held to: `addr`'s own consumer count is 2, not 1, so the
    # `"add"` outcome is declined before group resolution is even
    # attempted. Real fan-out this composition deliberately doesn't try to
    # reconcile with the `Concat` branch's own fixed-offset slice (see this
    # section's own comment on the `"add"` outcome): the whole `Concat`
    # group is declined, exactly as if the branch had failed to resolve at
    # all, so nothing here is touched -- not the residual pair, the plain
    # second branch, or `Z`'s own consumer.
    K, C, Cb, Cz, Out = 8, 6, 4, 3, 3
    rng = np.random.default_rng(215)
    wf = rng.standard_normal((K, C)).astype(np.float32)
    ws = rng.standard_normal((K, C)).astype(np.float32)
    wb = rng.standard_normal((K, Cb)).astype(np.float32)
    wz = rng.standard_normal((C, Cz)).astype(np.float32)
    wout = rng.standard_normal((C + Cb, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y, float[batch,{Cz}] Z)
        {{
          f = MatMul(X, WF)
          s = MatMul(X, WS)
          addr = Add(f, s)
          hb = MatMul(X, WB)
          merged = Concat<axis=-1>(addr, hb)
          Y = MatMul(merged, WOUT)
          Z = MatMul(addr, WZ)
        }}
        """,
        initializer=[
            _f32(wf, "WF"),
            _f32(ws, "WS"),
            _f32(wb, "WB"),
            _f32(wz, "WZ"),
            _f32(wout, "WOUT"),
        ],
    )
    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["WF"], wf)
    np.testing.assert_array_equal(inits["WS"], ws)
    np.testing.assert_array_equal(inits["WB"], wb)
    np.testing.assert_array_equal(inits["WZ"], wz)
    np.testing.assert_array_equal(inits["WOUT"], wout)


def test_structured_pruning_matmul_concat_declines_on_residual_merge_group_interior_fan_out():
    # A three-producer transitive group -- addr1 = Add(f, s), addr2 =
    # Add(addr1, t) -- whose *sink* (`addr2`) feeds the `Concat` cleanly
    # (its own only consumer), but whose *interior* tensor `addr1` also
    # feeds a second, ordinary consumer (`Z`) elsewhere. `addr2`'s own
    # direct walk to the `Concat` operand sees no fan-out at all (only
    # `addr1`, an entirely different tensor, is over-read) -- so this
    # exercises the deeper check, once the whole group is resolved:
    # `_resolve_matmul_fanout_branches` finds `Z` as a real, resolvable
    # extra consumer of the group's own internal wiring, and this
    # composition declines rather than trying to reconcile a `Concat`
    # branch's own fixed-offset slice with an ordinary chain sharing the
    # same weights. The whole `Concat` group is declined -- nothing here is
    # touched, not the three-producer group, the plain second branch, or
    # `Z`'s own consumer.
    K, C, Cb, Cz, Out = 8, 6, 4, 3, 3
    rng = np.random.default_rng(216)
    wf = rng.standard_normal((K, C)).astype(np.float32)
    ws = rng.standard_normal((K, C)).astype(np.float32)
    wt = rng.standard_normal((K, C)).astype(np.float32)
    wb = rng.standard_normal((K, Cb)).astype(np.float32)
    wz = rng.standard_normal((C, Cz)).astype(np.float32)
    wout = rng.standard_normal((C + Cb, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y, float[batch,{Cz}] Z)
        {{
          f = MatMul(X, WF)
          s = MatMul(X, WS)
          addr1 = Add(f, s)
          t = MatMul(X, WT)
          addr2 = Add(addr1, t)
          hb = MatMul(X, WB)
          merged = Concat<axis=-1>(addr2, hb)
          Y = MatMul(merged, WOUT)
          Z = MatMul(addr1, WZ)
        }}
        """,
        initializer=[
            _f32(wf, "WF"),
            _f32(ws, "WS"),
            _f32(wt, "WT"),
            _f32(wb, "WB"),
            _f32(wz, "WZ"),
            _f32(wout, "WOUT"),
        ],
    )
    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["WF"], wf)
    np.testing.assert_array_equal(inits["WS"], ws)
    np.testing.assert_array_equal(inits["WT"], wt)
    np.testing.assert_array_equal(inits["WB"], wb)
    np.testing.assert_array_equal(inits["WZ"], wz)
    np.testing.assert_array_equal(inits["WOUT"], wout)


def test_structured_pruning_matmul_concat_composes_with_gated_branch_matches_oracle():
    # A gated (SwiGLU-style) Mul of two non-constant operands feeds one
    # Concat operand directly -- no real producer's raw output in between,
    # and no Add/SkipLayerNormalization merge involved at all, distinct from
    # both the plain-producer and composed-residual-group Concat branch
    # shapes already covered above. Composed, not declined (see
    # _find_matmul_concat_chains's own docstring on the "gated" outcome):
    # WG and WU's two raw outputs each become this one branch's own
    # `producers`, ranked together by the same root-sum-square importance a
    # standalone gated pair already uses. Deliberately adversarial on both
    # axes at once, mirroring the residual-composition sibling test above:
    # WG/WU individually disagree about which half of the columns matter
    # most (so only their *combined* norm, not either producer's own, can be
    # driving the gated branch's keep set), and WB (the plain second Concat
    # branch) is scaled 10x larger so a bug that (wrongly) mixed the two
    # Concat branches' importances into one ranking would starve WB's own
    # columns entirely.
    K, C, Cb, Out = 8, 16, 6, 3
    rng, wg, wu = _conflicting_wf_ws(233, K, C)
    wb = rng.standard_normal((K, Cb)).astype(np.float32) * 10.0
    wout = rng.standard_normal((C + Cb, Out)).astype(np.float32)

    def _build(wg_, wu_, wb_, wout_):
        return _model(
            f"""
            g (float[batch,{K}] X) => (float[batch,{Out}] Y)
            {{
              gate = MatMul(X, WG)
              gate_act = Sigmoid(gate)
              up = MatMul(X, WU)
              h = Mul(gate_act, up)
              hb = MatMul(X, WB)
              merged = Concat<axis=-1>(h, hb)
              Y = MatMul(merged, WOUT)
            }}
            """,
            initializer=[
                _f32(wg_, "WG"),
                _f32(wu_, "WU"),
                _f32(wb_, "WB"),
                _f32(wout_, "WOUT"),
            ],
        )

    model = _build(wg, wu, wb, wout)
    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    keep_g = _skip_layer_norm_keep(wg, wu, C)
    importance_b = np.linalg.norm(wb.astype(np.float64), axis=0)
    keep_b = np.sort(np.argsort(-importance_b)[: Cb // 2])
    assert len(keep_b) == Cb // 2  # branch b kept its own top half, not starved to 0
    global_keep = np.concatenate([keep_g, keep_b + C])

    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["WG"], wg[:, keep_g])
    np.testing.assert_array_equal(inits["WU"], wu[:, keep_g])
    np.testing.assert_array_equal(inits["WB"], wb[:, keep_b])
    np.testing.assert_array_equal(inits["WOUT"], wout[global_keep, :])

    oracle = _build(wg[:, keep_g], wu[:, keep_g], wb[:, keep_b], wout[global_keep, :])
    rng_x = np.random.default_rng(234)
    x = rng_x.standard_normal((5, K)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_structured_pruning_matmul_concat_declines_on_gated_branch_with_extra_fanout():
    # Same gated-Mul-feeds-Concat-directly shape as the composed test above,
    # except `gate_act` (the gate's own activation output) additionally
    # feeds a second graph output `Z` -- must still decline the *whole*
    # Concat group, not just this one branch: _trace_gate_producer_backward
    # (reused unchanged from _find_gated_chains) holds every tensor on a
    # gate/up path to an exact single-consumer bar, and embedding it inside
    # _walk_matmul_producer_backward doesn't relax that bar here either --
    # mirrors test_structured_pruning_matmul_residual_add_declines_on_gated_branch_with_extra_fanout,
    # the same regression proven for the residual composition.
    K, C, Cb, Out = 8, 16, 6, 3
    rng, wg, wu = _conflicting_wf_ws(235, K, C)
    wb = rng.standard_normal((K, Cb)).astype(np.float32)
    wout = rng.standard_normal((C + Cb, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y, float[batch,{C}] Z)
        {{
          gate = MatMul(X, WG)
          gate_act = Sigmoid(gate)
          up = MatMul(X, WU)
          h = Mul(gate_act, up)
          hb = MatMul(X, WB)
          merged = Concat<axis=-1>(h, hb)
          Y = MatMul(merged, WOUT)
          Z = Identity(gate_act)
        }}
        """,
        initializer=[
            _f32(wg, "WG"),
            _f32(wu, "WU"),
            _f32(wb, "WB"),
            _f32(wout, "WOUT"),
        ],
    )
    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["WG"], wg)
    np.testing.assert_array_equal(inits["WU"], wu)
    np.testing.assert_array_equal(inits["WB"], wb)
    np.testing.assert_array_equal(inits["WOUT"], wout)


def test_structured_pruning_matmul_concat_declines_on_duplicate_operand():
    # Concat(h, h) -- the same tensor named twice as an operand of the same
    # Concat node -- is degenerate (not two independent branches at all) and
    # is declined outright.
    K, C, Out = 8, 6, 3
    rng = np.random.default_rng(214)
    w1 = rng.standard_normal((K, C)).astype(np.float32)
    wout = rng.standard_normal((2 * C, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          h = MatMul(X, W1)
          merged = Concat<axis=-1>(h, h)
          Y = MatMul(merged, WOUT)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(wout, "WOUT")],
    )
    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W1"], w1)
    np.testing.assert_array_equal(inits["WOUT"], wout)


def test_structured_pruning_conv_concat_prunes_each_branch_to_its_own_count():
    Cin, Ca, Cb, Cout = 3, 10, 6, 4
    rng = np.random.default_rng(215)
    wa = rng.standard_normal((Ca, Cin, 3, 3)).astype(np.float32)
    wb = rng.standard_normal((Cb, Cin, 3, 3)).astype(np.float32)
    wout = rng.standard_normal((Cout, Ca + Cb, 3, 3)).astype(np.float32)
    model = _conv_concat_model([wa, wb], wout)

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["W0"].dims) == [5, Cin, 3, 3]
    assert list(inits["W1"].dims) == [3, Cin, 3, 3]
    assert list(inits["WOUT"].dims) == [Cout, 8, 3, 3]


def test_structured_pruning_conv_concat_matches_oracle_no_cross_branch_coupling():
    # The Conv analogue of the MatMul "no cross-branch coupling" test above
    # -- branch a's filters scaled 10x branch b's; correct independent
    # per-branch ranking keeps each branch's own top half regardless.
    Cin, C, Cout = 3, 8, 4
    rng = np.random.default_rng(216)
    wa = rng.standard_normal((C, Cin, 3, 3)).astype(np.float32) * 10.0
    wb = rng.standard_normal((C, Cin, 3, 3)).astype(np.float32)
    wout = rng.standard_normal((Cout, 2 * C, 3, 3)).astype(np.float32)
    model = _conv_concat_model([wa, wb], wout)

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    keep_a = _oracle_keep_indices_conv(wa, C // 2)
    keep_b = _oracle_keep_indices_conv(wb, C // 2)
    assert len(keep_b) == C // 2  # branch b kept its own top half, not starved to 0
    global_keep = np.concatenate([keep_a, keep_b + C])

    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["W0"], wa[keep_a])
    np.testing.assert_array_equal(inits["W1"], wb[keep_b])
    np.testing.assert_array_equal(inits["WOUT"], wout[:, global_keep])

    oracle = _conv_concat_model([wa[keep_a], wb[keep_b]], wout[:, global_keep])
    rng_x = np.random.default_rng(217)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_structured_pruning_conv_concat_with_depthwise_pass_through_matches_oracle():
    # Branch a crosses a depthwise Conv hop (self-consistently matched by
    # _walk_conv_producer_backward, the exact same mechanism the Conv
    # residual section already verifies) before reaching the Concat node --
    # confirming the pass-through hop's own weight/bias/`group` slice by
    # branch a's own local `keep`, not the global one. The depthwise hop
    # uses a 1x1 kernel (spatial-preserving) so branch a's own spatial size
    # after its own two Convs (10 -> 8, unchanged by the 1x1 hop) still
    # lines up with branch b's single-Conv spatial size (10 -> 8) at the
    # point they Concat -- a 3x3 depthwise hop would shrink branch a's own
    # spatial size a second time and the two branches could no longer
    # Concat at all.
    Cin, Ca, Cb, Cout = 3, 8, 4, 5
    rng = np.random.default_rng(218)
    wa = rng.standard_normal((Ca, Cin, 3, 3)).astype(np.float32)
    wd = rng.standard_normal((Ca, 1, 1, 1)).astype(np.float32)
    bd = rng.standard_normal((Ca,)).astype(np.float32)
    wb = rng.standard_normal((Cb, Cin, 3, 3)).astype(np.float32)
    wout = rng.standard_normal((Cout, Ca + Cb, 3, 3)).astype(np.float32)
    model = _model(
        f"""
        g (float[N,{Cin},10,10] X) => (float[N,{Cout},6,6] Y)
        {{
          ha = Conv<kernel_shape=[3,3]>(X, WA)
          ra = Relu(ha)
          da = Conv<kernel_shape=[1,1], group={Ca}>(ra, WD, BD)
          hb = Conv<kernel_shape=[3,3]>(X, WB)
          merged = Concat<axis=1>(da, hb)
          Y = Conv<kernel_shape=[3,3]>(merged, WOUT)
        }}
        """,
        initializer=[
            _f32(wa, "WA"),
            _f32(wd, "WD"),
            _f32(bd, "BD"),
            _f32(wb, "WB"),
            _f32(wout, "WOUT"),
        ],
    )

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    keep_a = _oracle_keep_indices_conv(wa, Ca // 2)
    keep_b = _oracle_keep_indices_conv(wb, Cb // 2)
    global_keep = np.concatenate([keep_a, keep_b + Ca])

    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["WD"], wd[keep_a])
    np.testing.assert_array_equal(inits["BD"], bd[keep_a])
    dw_node = next(n for n in pruned.graph.node if "WD" in n.input)
    group_attr = next(a.i for a in dw_node.attribute if a.name == "group")
    assert group_attr == Ca // 2

    oracle = _model(
        f"""
        g (float[N,{Cin},10,10] X) => (float[N,{Cout},6,6] Y)
        {{
          ha = Conv<kernel_shape=[3,3]>(X, WA)
          ra = Relu(ha)
          da = Conv<kernel_shape=[1,1], group={Ca // 2}>(ra, WD, BD)
          hb = Conv<kernel_shape=[3,3]>(X, WB)
          merged = Concat<axis=1>(da, hb)
          Y = Conv<kernel_shape=[3,3]>(merged, WOUT)
        }}
        """,
        initializer=[
            _f32(wa[keep_a], "WA"),
            _f32(wd[keep_a], "WD"),
            _f32(bd[keep_a], "BD"),
            _f32(wb[keep_b], "WB"),
            _f32(wout[:, global_keep], "WOUT"),
        ],
    )
    rng_x = np.random.default_rng(219)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_structured_wanda_pruning_conv_concat_matches_oracle():
    Cin, Ca, Cb, Cout = 3, 6, 6, 4
    rng = np.random.default_rng(220)
    wa = rng.standard_normal((Ca, Cin, 3, 3)).astype(np.float32)
    wb = rng.standard_normal((Cb, Cin, 3, 3)).astype(np.float32)
    wa[: Ca // 2] *= 3.0
    wb[: Cb // 2] *= 3.0
    wout = rng.standard_normal((Cout, Ca + Cb, 3, 3)).astype(np.float32)
    model = _conv_concat_model([wa, wb], wout)

    probe_model = onnx.ModelProto()
    probe_model.CopyFrom(model)
    probe_model.graph.output.append(
        onnx.helper.make_tensor_value_info("h0", onnx.TensorProto.FLOAT, None)
    )
    probe_model.graph.output.append(
        onnx.helper.make_tensor_value_info("h1", onnx.TensorProto.FLOAT, None)
    )

    rng_cal = np.random.default_rng(221)
    x_cal = rng_cal.standard_normal((4, Cin, 10, 10)).astype(np.float32)
    calibration_data = [{"X": x_cal}]
    _, h0_cal, h1_cal = _run(probe_model, {"X": x_cal})
    norm_a = np.sqrt(np.mean(np.square(h0_cal.astype(np.float64)), axis=(0, 2, 3)))
    norm_b = np.sqrt(np.mean(np.square(h1_cal.astype(np.float64)), axis=(0, 2, 3)))

    pruned = onnxsim.apply_structured_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned)

    imp_a = np.linalg.norm(wa.reshape(Ca, -1).astype(np.float64), axis=1) * np.maximum(
        norm_a, 1e-8
    )
    imp_b = np.linalg.norm(wb.reshape(Cb, -1).astype(np.float64), axis=1) * np.maximum(
        norm_b, 1e-8
    )
    keep_a = np.sort(np.argsort(-imp_a)[: Ca // 2])
    keep_b = np.sort(np.argsort(-imp_b)[: Cb // 2])
    global_keep = np.concatenate([keep_a, keep_b + Ca])
    oracle = _conv_concat_model([wa[keep_a], wb[keep_b]], wout[:, global_keep])

    rng_x = np.random.default_rng(222)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_structured_pruning_conv_concat_composes_with_residual_merge_branch_matches_oracle():
    # The Conv analogue of
    # test_structured_pruning_matmul_concat_composes_with_residual_merge_branch_matches_oracle:
    # one Concat branch (`addr = Add(Conv_f(X), Conv_s(X))`) is itself a
    # residual-merge group with no consumer anywhere else, composed via
    # _resolve_conv_residual_group_for_concat exactly as
    # _find_conv_residual_chains would resolve it standalone; the other
    # branch (`hb`) is plain and ranked entirely independently. Same double
    # adversarial construction: WF/WS individually disagree about which
    # half of the *filters* matter most (so only the group's own combined
    # importance can be driving its keep set), and WB is scaled 10x larger
    # (so a bug mixing the two Concat branches' importances would starve
    # WB's own filters).
    Cin, C, Cb, Cout = 3, 16, 6, 5
    rng = np.random.default_rng(219)
    scale_f = (
        np.where(np.arange(C) < C // 2, 3.0, 0.3).astype(np.float32).reshape(C, 1, 1, 1)
    )
    scale_s = (
        np.where(np.arange(C) < C // 2, 0.3, 3.0).astype(np.float32).reshape(C, 1, 1, 1)
    )
    wf = rng.standard_normal((C, Cin, 3, 3)).astype(np.float32) * scale_f
    ws = rng.standard_normal((C, Cin, 3, 3)).astype(np.float32) * scale_s
    wb = rng.standard_normal((Cb, Cin, 3, 3)).astype(np.float32) * 10.0
    wout = rng.standard_normal((Cout, C + Cb, 3, 3)).astype(np.float32)
    model = _model(
        f"""
        g (float[N,{Cin},10,10] X) => (float[N,{Cout},6,6] Y)
        {{
          f = Conv<kernel_shape=[3,3]>(X, WF)
          s = Conv<kernel_shape=[3,3]>(X, WS)
          addr = Add(f, s)
          hb = Conv<kernel_shape=[3,3]>(X, WB)
          merged = Concat<axis=1>(addr, hb)
          Y = Conv<kernel_shape=[3,3]>(merged, WOUT)
        }}
        """,
        initializer=[
            _f32(wf, "WF"),
            _f32(ws, "WS"),
            _f32(wb, "WB"),
            _f32(wout, "WOUT"),
        ],
    )
    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    importance_r = np.sqrt(
        np.square(np.linalg.norm(wf.reshape(C, -1).astype(np.float64), axis=1))
        + np.square(np.linalg.norm(ws.reshape(C, -1).astype(np.float64), axis=1))
    )
    keep_r = np.sort(np.argsort(-importance_r)[: C // 2])
    assert np.any(keep_r < C // 2) and np.any(keep_r >= C // 2)
    keep_b = _oracle_keep_indices_conv(wb, Cb // 2)
    assert len(keep_b) == Cb // 2
    global_keep = np.concatenate([keep_r, keep_b + C])

    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["WF"], wf[keep_r])
    np.testing.assert_array_equal(inits["WS"], ws[keep_r])
    np.testing.assert_array_equal(inits["WB"], wb[keep_b])
    np.testing.assert_array_equal(inits["WOUT"], wout[:, global_keep])

    oracle = _model(
        f"""
        g (float[N,{Cin},10,10] X) => (float[N,{Cout},6,6] Y)
        {{
          f = Conv<kernel_shape=[3,3]>(X, WF)
          s = Conv<kernel_shape=[3,3]>(X, WS)
          addr = Add(f, s)
          hb = Conv<kernel_shape=[3,3]>(X, WB)
          merged = Concat<axis=1>(addr, hb)
          Y = Conv<kernel_shape=[3,3]>(merged, WOUT)
        }}
        """,
        initializer=[
            _f32(wf[keep_r], "WF"),
            _f32(ws[keep_r], "WS"),
            _f32(wb[keep_b], "WB"),
            _f32(wout[:, global_keep], "WOUT"),
        ],
    )
    rng_x = np.random.default_rng(220)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_structured_pruning_conv_concat_declines_on_wrong_axis():
    # axis=2 is a spatial axis, not the channel axis of [N, C, H, W] -- left
    # completely untouched, same conservative decline as a positive-axis
    # MatMul/Gemm Concat.
    Cin, Ca, Cb, Cout = 3, 4, 4, 4
    rng = np.random.default_rng(223)
    wa = rng.standard_normal((Ca, Cin, 3, 3)).astype(np.float32)
    wb = rng.standard_normal((Cb, Cin, 3, 3)).astype(np.float32)
    wout = rng.standard_normal((Cout, Ca, 3, 3)).astype(np.float32)
    model = _model(
        f"""
        g (float[N,{Cin},10,10] X) => (float[N,{Cout},14,6] Y)
        {{
          ha = Conv<kernel_shape=[3,3]>(X, WA)
          hb = Conv<kernel_shape=[3,3]>(X, WB)
          merged = Concat<axis=2>(ha, hb)
          Y = Conv<kernel_shape=[3,3]>(merged, WOUT)
        }}
        """,
        initializer=[_f32(wa, "WA"), _f32(wb, "WB"), _f32(wout, "WOUT")],
    )
    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["WA"], wa)
    np.testing.assert_array_equal(inits["WB"], wb)
    np.testing.assert_array_equal(inits["WOUT"], wout)


def test_structured_pruning_conv_concat_declines_on_straddling_grouped_conv_consumer():
    # The downstream consumer is a general grouped Conv (group=2, so
    # block=(Ca+Cb)//group=4), but branch B's own fixed offset (Ca=3) is
    # *not* a multiple of that block size: block 0 is channels [0, 4), and
    # branch A only owns [0, 3) of it -- column 3 belongs to branch B. That
    # one block therefore straddles both branches (see
    # _concat_branches_align_to_consumer_group's own docstring for exactly
    # this counter-example), so it's still declined outright, the whole
    # chain untouched -- unlike a block-aligned grouped consumer (see
    # test_structured_pruning_conv_concat_admits_block_aligned_grouped_conv_consumer
    # below), which *is* now admitted.
    Cin, Ca, Cb, Cout, group = 3, 3, 5, 8, 2
    rng = np.random.default_rng(224)
    wa = rng.standard_normal((Ca, Cin, 3, 3)).astype(np.float32)
    wb = rng.standard_normal((Cb, Cin, 3, 3)).astype(np.float32)
    wout = rng.standard_normal((Cout, (Ca + Cb) // group, 1, 1)).astype(np.float32)
    model = _model(
        f"""
        g (float[N,{Cin},10,10] X) => (float[N,{Cout},8,8] Y)
        {{
          ha = Conv<kernel_shape=[3,3]>(X, WA)
          hb = Conv<kernel_shape=[3,3]>(X, WB)
          merged = Concat<axis=1>(ha, hb)
          Y = Conv<kernel_shape=[1,1], group={group}>(merged, WOUT)
        }}
        """,
        initializer=[_f32(wa, "WA"), _f32(wb, "WB"), _f32(wout, "WOUT")],
    )
    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["WA"], wa)
    np.testing.assert_array_equal(inits["WB"], wb)
    np.testing.assert_array_equal(inits["WOUT"], wout)


def test_structured_pruning_conv_concat_admits_block_aligned_grouped_conv_consumer():
    # Ca == Cb == 4 and group=2 give block=(Ca+Cb)//group=4 -- each branch
    # owns exactly one of the consumer's two blocks (offsets 0 and 4, both
    # multiples of 4), the simplest block-aligned shape
    # _concat_branches_align_to_consumer_group admits: every block has
    # exactly one owning branch, so no cross-branch agreement is ever
    # needed and each branch's own ordinary top-k (one block == its own
    # whole n_channels here) already is that block's own per-block top-k.
    Cin, Ca, Cb, Cout, group = 3, 4, 4, 8, 2
    rng = np.random.default_rng(224)
    wa = rng.standard_normal((Ca, Cin, 3, 3)).astype(np.float32)
    wb = rng.standard_normal((Cb, Cin, 3, 3)).astype(np.float32)
    wout = rng.standard_normal((Cout, (Ca + Cb) // group, 1, 1)).astype(np.float32)
    model = _model(
        f"""
        g (float[N,{Cin},10,10] X) => (float[N,{Cout},8,8] Y)
        {{
          ha = Conv<kernel_shape=[3,3]>(X, WA)
          hb = Conv<kernel_shape=[3,3]>(X, WB)
          merged = Concat<axis=1>(ha, hb)
          Y = Conv<kernel_shape=[1,1], group={group}>(merged, WOUT)
        }}
        """,
        initializer=[_f32(wa, "WA"), _f32(wb, "WB"), _f32(wout, "WOUT")],
    )
    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    # Each branch is exactly one block wide here, so its own per-block
    # top-k reduces to an ordinary whole-branch top-k -- group=1 passed to
    # the existing per-group oracle helper below gives exactly that.
    keep_a = _oracle_keep_indices_conv_grouped(wa, 1, 0.5)
    keep_b = _oracle_keep_indices_conv_grouped(wb, 1, 0.5)
    assert len(keep_a) == 2 and len(keep_b) == 2
    global_keep = np.concatenate([keep_a, keep_b + Ca])
    wout_sliced = _oracle_slice_grouped_consumer_conv(wout, global_keep, group, Ca + Cb)

    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["WA"], wa[keep_a])
    np.testing.assert_array_equal(inits["WB"], wb[keep_b])
    np.testing.assert_array_equal(inits["WOUT"], wout_sliced)

    oracle = _model(
        f"""
        g (float[N,{Cin},10,10] X) => (float[N,{Cout},8,8] Y)
        {{
          ha = Conv<kernel_shape=[3,3]>(X, WA)
          hb = Conv<kernel_shape=[3,3]>(X, WB)
          merged = Concat<axis=1>(ha, hb)
          Y = Conv<kernel_shape=[1,1], group={group}>(merged, WOUT)
        }}
        """,
        initializer=[
            _f32(wa[keep_a], "WA"),
            _f32(wb[keep_b], "WB"),
            _f32(wout_sliced, "WOUT"),
        ],
    )
    rng_x = np.random.default_rng(225)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_structured_pruning_conv_concat_grouped_consumer_prunes_branches_independently_despite_conflicting_importance():
    # Deliberately adversarial: branch A's weights are scaled 100x larger
    # than branch B's, so by *any* cross-branch-comparable ranking branch B
    # would look catastrophically unimportant next to branch A -- if this
    # composition secretly needed the branches to agree (the way a
    # gated pair or residual merge's shared producers must), a buggy
    # implementation could plausibly starve branch B's own block down to
    # fewer survivors than its own 50% sparsity target, or even try to
    # "borrow" extra keep budget for branch A's block from branch B's.
    # Concat branches need no such agreement (this module's own docstring):
    # each branch is ranked and pruned purely against *itself*, so branch
    # B's own block keeps exactly its own per_group_keep=2 channels
    # regardless of branch A's own weight scale.
    Cin, Ca, Cb, Cout, group = 3, 4, 4, 8, 2
    rng = np.random.default_rng(226)
    wa = rng.standard_normal((Ca, Cin, 3, 3)).astype(np.float32) * 100.0
    wb = rng.standard_normal((Cb, Cin, 3, 3)).astype(np.float32) * 0.01
    wout = rng.standard_normal((Cout, (Ca + Cb) // group, 1, 1)).astype(np.float32)
    model = _model(
        f"""
        g (float[N,{Cin},10,10] X) => (float[N,{Cout},8,8] Y)
        {{
          ha = Conv<kernel_shape=[3,3]>(X, WA)
          hb = Conv<kernel_shape=[3,3]>(X, WB)
          merged = Concat<axis=1>(ha, hb)
          Y = Conv<kernel_shape=[1,1], group={group}>(merged, WOUT)
        }}
        """,
        initializer=[_f32(wa, "WA"), _f32(wb, "WB"), _f32(wout, "WOUT")],
    )
    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    keep_a = _oracle_keep_indices_conv_grouped(wa, 1, 0.5)
    keep_b = _oracle_keep_indices_conv_grouped(wb, 1, 0.5)
    # The crux of this test: branch B keeps exactly `per_group_keep` of its
    # own channels -- not zero, and not fewer than branch A's own count --
    # despite being 10,000x smaller in weight magnitude.
    assert len(keep_a) == 2 and len(keep_b) == 2
    global_keep = np.concatenate([keep_a, keep_b + Ca])
    wout_sliced = _oracle_slice_grouped_consumer_conv(wout, global_keep, group, Ca + Cb)

    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["WA"], wa[keep_a])
    np.testing.assert_array_equal(inits["WB"], wb[keep_b])
    np.testing.assert_array_equal(inits["WOUT"], wout_sliced)

    oracle = _model(
        f"""
        g (float[N,{Cin},10,10] X) => (float[N,{Cout},8,8] Y)
        {{
          ha = Conv<kernel_shape=[3,3]>(X, WA)
          hb = Conv<kernel_shape=[3,3]>(X, WB)
          merged = Concat<axis=1>(ha, hb)
          Y = Conv<kernel_shape=[1,1], group={group}>(merged, WOUT)
        }}
        """,
        initializer=[
            _f32(wa[keep_a], "WA"),
            _f32(wb[keep_b], "WB"),
            _f32(wout_sliced, "WOUT"),
        ],
    )
    rng_x = np.random.default_rng(227)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_structured_pruning_conv_concat_grouped_consumer_branch_spanning_multiple_blocks_matches_oracle():
    # group=3 gives block=(Ca+Cb)//group=4. Branch A (Ca=8) spans *two* of
    # the consumer's three blocks on its own (offsets 0 and 4, both
    # multiples of 4); branch B (Cb=4) owns the third block alone (offset
    # 8). This is the case _concat_branches_align_to_consumer_group's own
    # docstring singles out: a branch containing more than one block must
    # keep a *uniform* per_group_keep count from *each* of its own
    # contained blocks, not just an ordinary flat top-k over its whole
    # n_channels. Engineered so the two constructions disagree and are
    # separately checkable: branch A's first four filters (local block 0)
    # are scaled 100x larger than its last four (local block 1), so a
    # (wrong) flat top-4-of-8 over the whole branch would keep all of
    # block 0 and none of block 1 -- violating the grouped consumer's own
    # per-block-uniform-count requirement outright (and producing an empty
    # `local_keep` for the WOUT filter group belonging to block 1). The
    # correct per-block-aware selection keeps exactly 2 from each of
    # branch A's own two blocks regardless of the magnitude gap.
    Cin, Ca, Cb, Cout, group = 3, 8, 4, 9, 3
    n_channels = Ca + Cb
    block = n_channels // group
    rng = np.random.default_rng(228)
    wa = rng.standard_normal((Ca, Cin, 3, 3)).astype(np.float32)
    wa[:4] *= 100.0  # local block 0 -- overwhelmingly "more important"
    wa[4:] *= 0.01  # local block 1 -- overwhelmingly "less important"
    wb = rng.standard_normal((Cb, Cin, 3, 3)).astype(np.float32)
    wout = rng.standard_normal((Cout, n_channels // group, 1, 1)).astype(np.float32)
    model = _model(
        f"""
        g (float[N,{Cin},10,10] X) => (float[N,{Cout},8,8] Y)
        {{
          ha = Conv<kernel_shape=[3,3]>(X, WA)
          hb = Conv<kernel_shape=[3,3]>(X, WB)
          merged = Concat<axis=1>(ha, hb)
          Y = Conv<kernel_shape=[1,1], group={group}>(merged, WOUT)
        }}
        """,
        initializer=[_f32(wa, "WA"), _f32(wb, "WB"), _f32(wout, "WOUT")],
    )
    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    # Branch A spans Ca // block == 2 of the consumer's own blocks.
    keep_a = _oracle_keep_indices_conv_grouped(wa, Ca // block, 0.5)
    keep_b = _oracle_keep_indices_conv_grouped(wb, Cb // block, 0.5)
    # The crux of this test: exactly 2 survivors from *each* of branch A's
    # own two local blocks, not 4-and-0 (what a flat whole-branch top-k
    # would produce given the 100x/0.01x magnitude gap above).
    local_block0 = keep_a[keep_a < 4]
    local_block1 = keep_a[keep_a >= 4]
    assert len(local_block0) == 2
    assert len(local_block1) == 2
    assert len(keep_b) == 2

    global_keep = np.concatenate([keep_a, keep_b + Ca])
    wout_sliced = _oracle_slice_grouped_consumer_conv(
        wout, global_keep, group, n_channels
    )

    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["WA"], wa[keep_a])
    np.testing.assert_array_equal(inits["WB"], wb[keep_b])
    np.testing.assert_array_equal(inits["WOUT"], wout_sliced)

    oracle = _model(
        f"""
        g (float[N,{Cin},10,10] X) => (float[N,{Cout},8,8] Y)
        {{
          ha = Conv<kernel_shape=[3,3]>(X, WA)
          hb = Conv<kernel_shape=[3,3]>(X, WB)
          merged = Concat<axis=1>(ha, hb)
          Y = Conv<kernel_shape=[1,1], group={group}>(merged, WOUT)
        }}
        """,
        initializer=[
            _f32(wa[keep_a], "WA"),
            _f32(wb[keep_b], "WB"),
            _f32(wout_sliced, "WOUT"),
        ],
    )
    rng_x = np.random.default_rng(229)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


def test_structured_wanda_pruning_conv_concat_admits_block_aligned_grouped_conv_consumer():
    # The Wanda analogue of
    # test_structured_pruning_conv_concat_admits_block_aligned_grouped_conv_consumer:
    # confirms the group-aware per-block branch selection composes
    # correctly with _wanda_branch_importance's own activation-weighted
    # metric (captured at each branch's own Concat operand, exactly as an
    # ungrouped Concat consumer already is), not just the plain L2 path.
    Cin, Ca, Cb, Cout, group = 3, 4, 4, 8, 2
    rng = np.random.default_rng(230)
    wa = rng.standard_normal((Ca, Cin, 3, 3)).astype(np.float32)
    wb = rng.standard_normal((Cb, Cin, 3, 3)).astype(np.float32)
    wout = rng.standard_normal((Cout, (Ca + Cb) // group, 1, 1)).astype(np.float32)
    model = _model(
        f"""
        g (float[N,{Cin},10,10] X) => (float[N,{Cout},8,8] Y)
        {{
          ha = Conv<kernel_shape=[3,3]>(X, WA)
          hb = Conv<kernel_shape=[3,3]>(X, WB)
          merged = Concat<axis=1>(ha, hb)
          Y = Conv<kernel_shape=[1,1], group={group}>(merged, WOUT)
        }}
        """,
        initializer=[_f32(wa, "WA"), _f32(wb, "WB"), _f32(wout, "WOUT")],
    )

    probe_model = onnx.ModelProto()
    probe_model.CopyFrom(model)
    probe_model.graph.output.append(
        onnx.helper.make_tensor_value_info("ha", onnx.TensorProto.FLOAT, None)
    )
    probe_model.graph.output.append(
        onnx.helper.make_tensor_value_info("hb", onnx.TensorProto.FLOAT, None)
    )
    rng_cal = np.random.default_rng(231)
    x_cal = rng_cal.standard_normal((4, Cin, 10, 10)).astype(np.float32)
    calibration_data = [{"X": x_cal}]
    _, ha_cal, hb_cal = _run(probe_model, {"X": x_cal})
    norm_a = np.sqrt(np.mean(np.square(ha_cal.astype(np.float64)), axis=(0, 2, 3)))
    norm_b = np.sqrt(np.mean(np.square(hb_cal.astype(np.float64)), axis=(0, 2, 3)))

    pruned = onnxsim.apply_structured_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned)

    imp_a = np.linalg.norm(wa.reshape(Ca, -1).astype(np.float64), axis=1) * np.maximum(
        norm_a, 1e-8
    )
    imp_b = np.linalg.norm(wb.reshape(Cb, -1).astype(np.float64), axis=1) * np.maximum(
        norm_b, 1e-8
    )
    # Each branch is exactly one block wide, so its own per-block top-k is
    # an ordinary whole-branch top-k over this Wanda-weighted importance.
    keep_a = np.sort(np.argsort(-imp_a)[:2])
    keep_b = np.sort(np.argsort(-imp_b)[:2])
    global_keep = np.concatenate([keep_a, keep_b + Ca])
    wout_sliced = _oracle_slice_grouped_consumer_conv(wout, global_keep, group, Ca + Cb)

    oracle = _model(
        f"""
        g (float[N,{Cin},10,10] X) => (float[N,{Cout},8,8] Y)
        {{
          ha = Conv<kernel_shape=[3,3]>(X, WA)
          hb = Conv<kernel_shape=[3,3]>(X, WB)
          merged = Concat<axis=1>(ha, hb)
          Y = Conv<kernel_shape=[1,1], group={group}>(merged, WOUT)
        }}
        """,
        initializer=[
            _f32(wa[keep_a], "WA"),
            _f32(wb[keep_b], "WB"),
            _f32(wout_sliced, "WOUT"),
        ],
    )
    rng_x = np.random.default_rng(232)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y, y_oracle, rtol=1e-5, atol=1e-5)


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

    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          h = MatMul(X, W1)
          a = Relu(h)
          Y = MatMul(a, W2)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2")],
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
    model = _model(
        f"""
        g (float[batch,seq,{K}] X) => (float[batch,seq,{Out}] Y)
        {{
          h = MatMul(X, W1)
          a = Relu(h)
          Y = MatMul(a, W2)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2")],
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


# --- apply_structured_wanda_pruning: global_sparsity ---------------------


def _two_input_mlp_model(K=8, H=16, Out=4, seed=0):
    # Two independent, ordinary MLP chains fed by *separate* graph inputs
    # (mirroring `_two_input_matmul_model`'s unstructured analogue) so
    # calibration data can give each chain's own consumer-side activation a
    # different magnitude while both chains' weights stay the same scale.
    rng = np.random.default_rng(seed)
    w1a = rng.standard_normal((K, H)).astype(np.float32)
    w2a = rng.standard_normal((H, Out)).astype(np.float32)
    w1b = rng.standard_normal((K, H)).astype(np.float32)
    w2b = rng.standard_normal((H, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] Xa, float[batch,{K}] Xb) => (float[batch,{Out}] Ya, float[batch,{Out}] Yb)
        {{
          ha = MatMul(Xa, W1a)
          aa = Relu(ha)
          Ya = MatMul(aa, W2a)
          hb = MatMul(Xb, W1b)
          ab = Relu(hb)
          Yb = MatMul(ab, W2b)
        }}
        """,
        initializer=[
            _f32(w1a, "W1a"),
            _f32(w2a, "W2a"),
            _f32(w1b, "W1b"),
            _f32(w2b, "W2b"),
        ],
    )
    return model, w1a, w2a, w1b, w2b


def test_structured_wanda_pruning_global_sparsity_redistributes_and_matches_oracle():
    K, H, Out = 8, 16, 4
    sparsity = 0.5
    model, w1a, w2a, w1b, w2b = _two_input_mlp_model(K=K, H=H, Out=Out, seed=17)
    rng = np.random.default_rng(19)
    xa_cal = (rng.standard_normal((32, K)) * 40.0).astype(np.float32)
    xb_cal = (rng.standard_normal((32, K)) * 1.0).astype(np.float32)
    calibration_data = [{"Xa": xa_cal, "Xb": xb_cal}]

    pruned = onnxsim.apply_structured_wanda_pruning(
        model,
        calibration_data=calibration_data,
        sparsity=sparsity,
        global_sparsity=True,
    )
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}

    # Chain "a" sees a 40x larger activation despite the same weight scale
    # -- global Wanda structured importance must protect it accordingly.
    a_kept = inits["W1a"].dims[1]
    b_kept = inits["W1b"].dims[1]
    assert a_kept > H // 2 > b_kept

    # Hand-built oracle: the activation norm captured right where each
    # chain feeds its consumer (post-Relu, computed directly in numpy --
    # deterministic and exactly what the internal onnxruntime probe would
    # see for a plain MatMul->Relu->MatMul chain), combined with each
    # chain's own L2-norm weight importance, then pooled and selected via
    # the same algorithm _apply_chains_global implements.
    aa_cal = np.maximum(xa_cal @ w1a, 0)
    ab_cal = np.maximum(xb_cal @ w1b, 0)
    norm_a = np.sqrt(np.mean(np.square(aa_cal), axis=0))
    norm_b = np.sqrt(np.mean(np.square(ab_cal), axis=0))
    importance_a = np.linalg.norm(w1a.T, axis=1) * np.maximum(norm_a, 1e-8)
    importance_b = np.linalg.norm(w1b.T, axis=1) * np.maximum(norm_b, 1e-8)
    keep_a_mask, keep_b_mask = _oracle_global_structured_keep(
        [importance_a, importance_b], sparsity
    )
    keep_a = np.flatnonzero(keep_a_mask)
    keep_b = np.flatnonzero(keep_b_mask)
    assert a_kept == keep_a.size
    assert b_kept == keep_b.size

    x_a = rng.standard_normal((5, K)).astype(np.float32)
    x_b = rng.standard_normal((5, K)).astype(np.float32)
    y_a, y_b = _run(pruned, {"Xa": x_a, "Xb": x_b})
    ya_oracle = np.maximum(x_a @ w1a[:, keep_a], 0) @ w2a[keep_a, :]
    yb_oracle = np.maximum(x_b @ w1b[:, keep_b], 0) @ w2b[keep_b, :]
    np.testing.assert_allclose(y_a, ya_oracle, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(y_b, yb_oracle, rtol=1e-4, atol=1e-4)


def test_structured_wanda_pruning_global_sparsity_invalid_sparsity_raises():
    model = _mlp_model(K=8, H=16, Out=4)
    with pytest.raises(ValueError):
        onnxsim.apply_structured_wanda_pruning(
            model, sparsity=1.0, global_sparsity=True
        )
    with pytest.raises(ValueError):
        onnxsim.apply_structured_wanda_pruning(
            model, sparsity=-0.1, global_sparsity=True
        )


# --- apply_attention_head_pruning ---------------------------------------


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
    attention_bias=None,  # constant attention_bias array, or None (unconnected)
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
    operands = ["X", "Wqkv"]
    if bias:
        initializer.append(_f32(bqkv, "Bqkv"))
        operands.append("Bqkv")
    else:
        operands.append("")

    # `attention_bias` (index 5) sits behind `mask_index` (3) and `past`
    # (4), both always left unconnected here (see this module's own
    # "Attention-head pruning" section comment for why `mask_index`'s own
    # several documented shapes never carry a `num_heads` axis at all) --
    # threaded through as empty positional placeholders to reach index 5.
    if attention_bias is not None:
        operands += ["", ""]
        initializer.append(_f32(np.asarray(attention_bias), "AttentionBias"))
        operands.append("AttentionBias")

    # Trailing optional inputs may simply be omitted rather than spelled
    # out as empty placeholders.
    while operands and operands[-1] == "":
        operands.pop()
    qkv_inputs = ", ".join(operands)

    if with_reshape:
        shape = np.array([batch, seq, Nv], dtype=np.int64)
        initializer.append(onnx.numpy_helper.from_array(shape, "Shape"))
        tail = "ctx2 = Reshape(ctx, Shape)\n          Y = MatMul(ctx2, Wout)"
    else:
        tail = "Y = MatMul(ctx, Wout)"

    body = f"""
        g ({K} X) => ({Out} Y)
        {{
          ctx = com.microsoft.Attention <num_heads={heads}, qkv_hidden_sizes=[{Nq},{Nk},{Nv}]> ({qkv_inputs})
          {tail}
        }}
        """
    # Substitute the actual rank-3 shapes by hand -- `_model`'s own f-string
    # convention assumes a 2-D-only [batch, dim] input/output signature.
    body = body.replace(f"({K} X)", f"(float[batch,seq,{K}] X)")
    body = body.replace(f"({Out} Y)", f"(float[batch,seq,{Out}] Y)")

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
        D=D,
        Out=Out,
        Nq=Nq,
        Nk=Nk,
        Nv=Nv,
        wqkv=wqkv,
        bqkv=bqkv,
        wout=wout,
        attention_bias=attention_bias,
    )


def _attention_node(model):
    return next(n for n in model.graph.node if n.op_type == "Attention")


def _attention_attrs(node):
    num_heads = next(a.i for a in node.attribute if a.name == "num_heads")
    qkv = next(list(a.ints) for a in node.attribute if a.name == "qkv_hidden_sizes")
    return num_heads, qkv


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


def _head_idx(keep_heads, d):
    return np.concatenate([np.arange(h * d, (h + 1) * d) for h in keep_heads])


def test_attention_head_pruning_shrinks_matched_block():
    model, cfg = _attention_model(K=8, H=4, D=4, Out=6)
    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    node = _attention_node(pruned)
    num_heads, qkv = _attention_attrs(node)
    assert num_heads == 2
    assert qkv == [8, 8, 8]

    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["Wqkv"].dims) == [8, 24]
    assert list(inits["Bqkv"].dims) == [24]
    assert list(inits["Wout"].dims) == [8, 6]


def test_attention_head_pruning_matches_manual_head_deletion_exactly():
    model, cfg = _attention_model(K=8, H=4, D=4, Out=6, seed=1)
    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
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


def test_attention_head_pruning_attention_bias_is_sliced_and_matches_oracle():
    # `com.microsoft::Attention`'s own contrib-op schema gives it a second,
    # unrelated optional mask-shaped input beyond `mask_index`:
    # `attention_bias` (index 5), documented shape `(batch_size or 1,
    # num_heads or 1, sequence_length, total_sequence_length)` -- confirmed
    # to have a real, non-ignored numeric effect via actual onnxruntime
    # execution (unlike `mask_index`, whose several documented shapes never
    # carry a `num_heads` axis at all -- see this module's own
    # "Attention-head pruning" section comment). An earlier version of this
    # matcher never inspected `attention_bias` at all, so pruning would
    # have silently left a now-wrong-head-count bias connected to a
    # pruned-head-count node -- a genuine correctness bug, not just an
    # overly conservative decline. This test would fail against that
    # earlier version: a stale full-width bias raises a broadcast shape
    # error at `onnxruntime.InferenceSession` run time (neither the new
    # `num_heads` nor 1), not just a numeric mismatch.
    rng = np.random.default_rng(30)
    H, D, seq = 4, 4, 5
    bias = (rng.standard_normal((1, H, seq, seq)) * 1000.0).astype(np.float32)
    model, cfg = _attention_model(K=8, H=H, D=D, Out=6, seed=30, attention_bias=bias)
    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    node = _attention_node(pruned)
    num_heads, _ = _attention_attrs(node)
    assert num_heads == 2  # round(4 - 4*0.5)

    keep = _oracle_keep_heads(cfg["wqkv"], cfg["Nq"], cfg["Nk"], cfg["Nv"], cfg["H"], 2)
    d = cfg["D"]
    qi, ki, vi = (
        _head_idx(keep, d),
        _head_idx(keep, d) + cfg["Nq"],
        _head_idx(keep, d) + cfg["Nq"] + cfg["Nk"],
    )
    all_idx = np.concatenate([qi, ki, vi])

    pruned_inits = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer
    }
    np.testing.assert_array_equal(pruned_inits["AttentionBias"], bias[:, keep])

    oracle, _ = _attention_model(
        K=cfg["K"],
        H=2,
        D=d,
        Out=cfg["Out"],
        seed=30,
        wqkv=cfg["wqkv"][:, all_idx],
        bqkv=cfg["bqkv"][all_idx],
        wout=cfg["wout"][_head_idx(keep, d), :],
        num_heads=2,
        attention_bias=bias[:, keep],
    )

    rng = np.random.default_rng(31)
    x = rng.standard_normal((2, 5, cfg["K"])).astype(np.float32)
    (y_pruned,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y_pruned, y_oracle, rtol=1e-4, atol=1e-4)


def test_attention_head_pruning_broadcast_attention_bias_is_left_untouched():
    # A `(1, 1, seq, seq)` `attention_bias` has its own `num_heads`-aligned
    # axis (axis 1) present but size 1 -- an ordinary broadcast, no
    # per-head values at all -- so it needs no slicing and must be left
    # byte-identical even though the rest of the block is pruned.
    rng = np.random.default_rng(32)
    H, D, seq = 4, 4, 5
    bias = (rng.standard_normal((1, 1, seq, seq)) * 1000.0).astype(np.float32)
    model, cfg = _attention_model(K=8, H=H, D=D, Out=6, seed=32, attention_bias=bias)
    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    node = _attention_node(pruned)
    num_heads, _ = _attention_attrs(node)
    assert num_heads == 2

    pruned_inits = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer
    }
    np.testing.assert_array_equal(pruned_inits["AttentionBias"], bias)


def test_attention_head_pruning_ambiguous_attention_bias_shape_is_declined():
    # A `(1, 3, seq, seq)` `attention_bias` -- axis 1 neither `num_heads`
    # (4) nor 1 -- doesn't cleanly resolve to either "genuinely per-head" or
    # "broadcast" against this op's own broadcasting rule, so the whole
    # match must be declined rather than guessed at.
    rng = np.random.default_rng(33)
    H, D, seq = 4, 4, 5
    bias = (rng.standard_normal((1, 3, seq, seq)) * 1000.0).astype(np.float32)
    model, cfg = _attention_model(K=8, H=H, D=D, Out=6, seed=33, attention_bias=bias)
    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)

    inits_before = {
        t.name: onnx.numpy_helper.to_array(t) for t in model.graph.initializer
    }
    inits_after = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer
    }
    for name in inits_before:
        np.testing.assert_array_equal(inits_before[name], inits_after[name])


def test_attention_head_pruning_reshape_hop_is_recognized_and_shape_updated():
    model, cfg = _attention_model(K=8, H=4, D=4, Out=6, seed=3, with_reshape=True)
    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.25)
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


def test_attention_head_pruning_group_query_attention_missing_required_inputs_is_left_untouched():
    # GroupQueryAttention is supported (see the "-- GroupQueryAttention"
    # section below), but its schema requires seqlens_k/
    # total_sequence_length even for a plain forward pass -- a node missing
    # them (as here, only q/k/v given) isn't a complete/safe-to-act-on GQA
    # node and must not be mistaken for one, nor for a plain `Attention`
    # node (whose merged-QKV-weight shape this one doesn't have either).
    K, H, D, Out = 8, 4, 4, 6
    Nqkv = H * D
    rng = np.random.default_rng(5)
    wq = rng.standard_normal((K, Nqkv)).astype(np.float32)
    wk = rng.standard_normal((K, Nqkv)).astype(np.float32)
    wv = rng.standard_normal((K, Nqkv)).astype(np.float32)
    wout = rng.standard_normal((Nqkv, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[2,5,{K}] X) => (float[2,5,{Out}] Y)
        {{
          q = MatMul(X, Wq)
          k = MatMul(X, Wk)
          v = MatMul(X, Wv)
          ctx, present_k, present_v = com.microsoft.GroupQueryAttention <num_heads={H}, kv_num_heads={H}> (q, k, v)
          Y = MatMul(ctx, Wout)
        }}
        """,
        initializer=[
            _f32(wq, "Wq"),
            _f32(wk, "Wk"),
            _f32(wv, "Wv"),
            _f32(wout, "Wout"),
        ],
        opset=17,
    )
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))

    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    inits_before = {
        t.name: onnx.numpy_helper.to_array(t) for t in model.graph.initializer
    }
    inits_after = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer
    }
    for name in inits_before:
        np.testing.assert_array_equal(inits_before[name], inits_after[name])


def test_attention_head_pruning_mismatched_consumer_reduction_dim_is_left_untouched():
    K, H, D, Out = 8, 4, 4, 6
    Nqkv = H * D
    rng = np.random.default_rng(6)
    wqkv = rng.standard_normal((K, 3 * Nqkv)).astype(np.float32)
    wout_wrong = rng.standard_normal((Nqkv + 1, Out)).astype(np.float32)  # off-by-one
    model = _model(
        f"""
        g (float[2,5,{K}] X) => (float[2,5,{Out}] Y)
        {{
          ctx = com.microsoft.Attention <num_heads={H}, qkv_hidden_sizes=[{Nqkv},{Nqkv},{Nqkv}]> (X, Wqkv)
          padded = Pad <pads = [0,0,0,0,0,1]> (ctx)
          Y = MatMul(padded, Wout)
        }}
        """,
        initializer=[_f32(wqkv, "Wqkv"), _f32(wout_wrong, "Wout")],
        opset=17,
    )
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))

    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    inits_before = {
        t.name: onnx.numpy_helper.to_array(t) for t in model.graph.initializer
    }
    inits_after = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer
    }
    for name in inits_before:
        np.testing.assert_array_equal(inits_before[name], inits_after[name])


def test_attention_head_pruning_zero_sparsity_is_a_no_op():
    model, cfg = _attention_model(K=8, H=4, D=4, Out=6, seed=7)
    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.0)
    node = _attention_node(pruned)
    num_heads, qkv = _attention_attrs(node)
    assert num_heads == cfg["H"]
    assert qkv == [cfg["Nq"], cfg["Nk"], cfg["Nv"]]
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["Wqkv"], cfg["wqkv"])


def test_attention_head_pruning_invalid_sparsity_raises():
    model, _ = _attention_model(K=8, H=4, D=4, Out=6)
    with pytest.raises(ValueError):
        onnxsim.apply_attention_head_pruning(model, sparsity=1.0)
    with pytest.raises(ValueError):
        onnxsim.apply_attention_head_pruning(model, sparsity=-0.1)


# --- apply_attention_head_wanda_pruning ---------------------------------


def test_attention_head_wanda_pruning_matches_oracle_exactly():
    model, cfg = _attention_model(K=8, H=4, D=4, Out=6, seed=8)

    rng = np.random.default_rng(9)
    x_cal = rng.standard_normal((3, 6, cfg["K"])).astype(np.float32)
    calibration_data = [{"X": x_cal}]

    pruned = onnxsim.apply_attention_head_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned)

    # Reproduce the calibrated importance from scratch: probe `ctx` (the
    # Attention node's own raw output, exactly what the consumer MatMul
    # reads here since there is no Reshape hop), reduce over every axis but
    # the channel one, combine per-head via root-sum-square, and multiply
    # into the plain Frobenius-norm weight importance.
    probe_model = onnx.ModelProto()
    probe_model.CopyFrom(model)
    probe_model.graph.output.append(onnx.ValueInfoProto(name="ctx"))
    (_, ctx_cal) = _run(probe_model, {"X": x_cal})
    act_norm = np.sqrt(np.mean(np.square(ctx_cal.astype(np.float64)), axis=(0, 1)))

    d = cfg["D"]
    wq, wk, wv = (
        cfg["wqkv"][:, : cfg["Nq"]],
        cfg["wqkv"][:, cfg["Nq"] : cfg["Nq"] + cfg["Nk"]],
        cfg["wqkv"][:, cfg["Nq"] + cfg["Nk"] :],
    )
    importance = np.zeros(cfg["H"])
    for h in range(cfg["H"]):
        block = np.concatenate(
            [
                wq[:, h * d : (h + 1) * d],
                wk[:, h * d : (h + 1) * d],
                wv[:, h * d : (h + 1) * d],
            ],
            axis=1,
        )
        act_head = np.linalg.norm(act_norm[h * d : (h + 1) * d])
        importance[h] = np.linalg.norm(block) * max(act_head, 1e-8)
    keep = np.sort(np.argsort(-importance)[:2])

    qi, ki, vi = (
        _head_idx(keep, d),
        _head_idx(keep, d) + cfg["Nq"],
        _head_idx(keep, d) + cfg["Nq"] + cfg["Nk"],
    )
    all_idx = np.concatenate([qi, ki, vi])
    oracle, _ = _attention_model(
        K=cfg["K"],
        H=2,
        D=d,
        Out=cfg["Out"],
        seed=8,
        wqkv=cfg["wqkv"][:, all_idx],
        bqkv=cfg["bqkv"][all_idx],
        wout=cfg["wout"][_head_idx(keep, d), :],
        num_heads=2,
    )

    x = rng.standard_normal((2, 5, cfg["K"])).astype(np.float32)
    (y_pruned,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y_pruned, y_oracle, rtol=1e-4, atol=1e-4)


def test_attention_head_wanda_pruning_falls_back_to_plain_with_no_calibration_batches():
    model, cfg = _attention_model(K=8, H=4, D=4, Out=6, seed=10)
    plain = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    wanda = onnxsim.apply_attention_head_wanda_pruning(
        model, calibration_data=[], sparsity=0.5
    )
    inits_plain = {
        t.name: onnx.numpy_helper.to_array(t) for t in plain.graph.initializer
    }
    inits_wanda = {
        t.name: onnx.numpy_helper.to_array(t) for t in wanda.graph.initializer
    }
    for name in inits_plain:
        np.testing.assert_array_equal(inits_plain[name], inits_wanda[name])


# --- apply_attention_head_pruning / _wanda_pruning -- GroupQueryAttention --


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
    past_key=None,  # explicit override for past_kv="nonempty" (else random)
    past_value=None,  # explicit override for past_kv="nonempty" (else random)
    past_kv_dtype=np.float32,  # dtype of PastKey/PastValue -- quantized when non-float
    k_scale=None,  # constant k_scale (PER_TENSOR/PER_CHANNEL array) or None (unconnected)
    v_scale=None,  # constant v_scale, same convention as k_scale
    k_quant_type=None,  # GQA node's own k_quant_type attribute, e.g. "PER_TENSOR"
    v_quant_type=None,  # GQA node's own v_quant_type attribute
    kv_cache_bit_width=None,  # GQA node's own kv_cache_bit_width attribute (8 or 4)
    attention_bias=None,  # constant attention_bias array, or None (unconnected)
    head_sink=None,  # constant head_sink array (shape (H,)), or None (unconnected)
):
    # Real ONNX Runtime CPU kernels for GroupQueryAttention require
    # head_size to be a multiple of 8 (verified empirically -- a smaller
    # head_size segfaults/errors at run time the same way a 2-input
    # com.microsoft::Attention does elsewhere in this file), so D defaults
    # to 8 rather than mirroring _attention_model's smaller default.
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

    # seqlens_k/total_sequence_length: mandatory KV-cache bookkeeping inputs
    # GroupQueryAttention's schema requires even for a plain, no-cache
    # forward pass (see fuse_gqa.h's own top comment) -- `total_seq - 1` per
    # batch row and `total_seq`, where `total_seq = seq + past_seq_len`
    # (`past_seq_len` is the hardcoded past_key/past_value sequence length of
    # 1 below when past_kv is connected at all, constant or dynamic, else 0)
    # -- `S-1`/`S` exactly when there's no past context, what fuse_attn.h
    # itself synthesizes for that no-cache case.
    past_seq_len = 1 if past_kv in ("nonempty", "dynamic") else 0
    total_seq = seq + past_seq_len
    initializer.append(
        onnx.numpy_helper.from_array(
            np.full((batch,), total_seq - 1, dtype=np.int32), "SeqLensK"
        )
    )
    initializer.append(
        onnx.numpy_helper.from_array(np.array(total_seq, dtype=np.int32), "TotalSeq")
    )

    def _random_past_kv_default():
        # `np.iinfo` covers `int8`/`uint8`; `float8e4m3fn` is neither
        # `np.floating` nor an integer dtype `np.iinfo` recognizes (its
        # numpy `kind` is opaque, `'V'`), so it gets its own small-range
        # float draw instead -- well inside e4m3's own representable range
        # (~448 magnitude) with plenty of margin for its coarser mantissa.
        if np.issubdtype(past_kv_dtype, np.floating):
            return rng.standard_normal((batch, KVH, 1, D))
        try:
            info = np.iinfo(past_kv_dtype)
        except (ValueError, TypeError):
            return rng.uniform(-2.0, 2.0, size=(batch, KVH, 1, D))
        return rng.integers(
            max(info.min, -64), min(info.max, 64) + 1, size=(batch, KVH, 1, D)
        )

    operands = ["q", "k", "v"]
    extra_graph_inputs = ""
    if past_kv == "nonempty":
        if past_key is None:
            past_key = _random_past_kv_default()
        if past_value is None:
            past_value = _random_past_kv_default()
        # Cast once and reuse the *quantized* (post-cast) array both for the
        # initializer and the returned `cfg` -- `past_kv_dtype`'s own cast
        # can be lossy (`float8e4m3fn`'s coarse mantissa, most notably), so
        # a caller comparing `cfg["past_key"]` against the model's own
        # initializer (e.g. after slicing) needs the already-quantized
        # values, not the pre-cast ones.
        past_key = np.asarray(past_key).astype(past_kv_dtype)
        past_value = np.asarray(past_value).astype(past_kv_dtype)
        initializer += [
            onnx.numpy_helper.from_array(past_key, "PastKey"),
            onnx.numpy_helper.from_array(past_value, "PastValue"),
        ]
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

    # `attention_bias`/`head_sink`/`k_scale`/`v_scale` (GQA input indices
    # 10/11/12/13) sit behind three other optional inputs
    # (cos_cache/sin_cache/position_ids, indices 7-9, always left
    # unconnected here -- none of this module's own matching/slicing
    # touches them) that must be threaded through as empty positional
    # placeholders for the text format's positional-input convention to
    # reach index 10 at all.
    if (
        attention_bias is not None
        or head_sink is not None
        or k_scale is not None
        or v_scale is not None
    ):
        operands += [""] * 3  # cos_cache, sin_cache, position_ids
        if attention_bias is not None:
            initializer.append(_f32(np.asarray(attention_bias), "AttentionBias"))
            operands.append("AttentionBias")
        else:
            operands.append("")
        if head_sink is not None:
            initializer.append(_f32(np.asarray(head_sink), "HeadSink"))
            operands.append("HeadSink")
        else:
            operands.append("")
        if k_scale is not None:
            initializer.append(_f32(np.asarray(k_scale), "KScale"))
            operands.append("KScale")
        else:
            operands.append("")
        if v_scale is not None:
            initializer.append(_f32(np.asarray(v_scale), "VScale"))
            operands.append("VScale")
        else:
            operands.append("")

    if with_reshape:
        shape = np.array([batch, seq, Nq], dtype=np.int64)
        initializer.append(onnx.numpy_helper.from_array(shape, "Shape"))
        tail = "ctx2 = Reshape(ctx, Shape)\n          Y = MatMul(ctx2, Wout)"
    else:
        tail = "Y = MatMul(ctx, Wout)"

    attrs = f"num_heads={H}, kv_num_heads={KVH}"
    if k_quant_type is not None:
        attrs += f', k_quant_type = "{k_quant_type}"'
    if v_quant_type is not None:
        attrs += f', v_quant_type = "{v_quant_type}"'
    if kv_cache_bit_width is not None:
        attrs += f", kv_cache_bit_width={kv_cache_bit_width}"

    body = f"""
        g (float[{batch},{seq},{K}] X{extra_graph_inputs}) => (float[{batch},{seq},{Out}] Y)
        {{
          q = {q_op}
          k = {k_op}
          v = {v_op}
          ctx, pk, pv = com.microsoft.GroupQueryAttention <{attrs}> ({", ".join(operands)})
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
        past_key=past_key,
        past_value=past_value,
        k_scale=k_scale,
        v_scale=v_scale,
        attention_bias=attention_bias,
        head_sink=head_sink,
    )


def _gqa_node(model):
    return next(n for n in model.graph.node if n.op_type == "GroupQueryAttention")


def _gqa_attrs(node):
    num_heads = next(a.i for a in node.attribute if a.name == "num_heads")
    kv_num_heads = next(a.i for a in node.attribute if a.name == "kv_num_heads")
    return num_heads, kv_num_heads


def _oracle_keep_groups(
    wq, wk, wv, num_heads, kv_num_heads, head_size, keep_count, v_head_size=None
):
    # `v_head_size` (V's own per-head column stride into `wv`) defaults to
    # `head_size` (Q's/K's shared one) -- the uniform case every caller but
    # the plain-ai.onnx-Attention "diff V head size" tests wants; those pass
    # a genuinely different `v_head_size` explicitly.
    if v_head_size is None:
        v_head_size = head_size
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
        v_block = wv[:, kv * v_head_size : (kv + 1) * v_head_size]
        importance[kv] = np.linalg.norm(
            np.concatenate([q_block, k_block, v_block], axis=1)
        )
    return np.sort(np.argsort(-importance)[:keep_count])


def _group_q_heads(keep_groups, group_size):
    return np.concatenate(
        [np.arange(g * group_size, (g + 1) * group_size) for g in keep_groups]
    )


def test_gqa_pruning_shrinks_matched_block():
    model, cfg = _gqa_model(K=8, H=4, KVH=4, D=8, Out=6)
    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    node = _gqa_node(pruned)
    num_heads, kv_num_heads = _gqa_attrs(node)
    assert num_heads == 2  # round(4 - 4*0.5) query heads ...
    assert kv_num_heads == 2  # ... and KV heads alike, since group_size == 1 here

    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["Wq"].dims) == [8, 16]
    assert list(inits["Wk"].dims) == [8, 16]
    assert list(inits["Wv"].dims) == [8, 16]
    assert list(inits["Wout"].dims) == [16, 6]


def test_gqa_pruning_zero_sparsity_is_a_no_op():
    model, cfg = _gqa_model(K=8, H=8, KVH=2, D=8, Out=6, seed=7)
    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.0)
    node = _gqa_node(pruned)
    num_heads, kv_num_heads = _gqa_attrs(node)
    assert num_heads == cfg["H"]
    assert kv_num_heads == cfg["KVH"]
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["Wq"], cfg["wq"])
    np.testing.assert_array_equal(inits["Wk"], cfg["wk"])
    np.testing.assert_array_equal(inits["Wv"], cfg["wv"])


def test_gqa_pruning_unequal_heads_drops_whole_groups_and_preserves_ratio():
    # 8 query heads sharing 4 KV heads (2 query heads per KV head); at
    # sparsity=0.5 two of the four *groups* must be dropped, never an
    # individual query head in isolation -- confirmed here by checking the
    # surviving Wq columns are exactly the two kept groups' own contiguous
    # 2-head blocks, matching the kept Wk/Wv columns' own group indices.
    model, cfg = _gqa_model(K=8, H=8, KVH=4, D=8, Out=6, seed=11)
    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    node = _gqa_node(pruned)
    num_heads, kv_num_heads = _gqa_attrs(node)
    assert kv_num_heads == 2
    assert num_heads == 4
    assert num_heads // kv_num_heads == cfg["H"] // cfg["KVH"]  # ratio preserved

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


def test_gqa_pruning_matches_oracle_exactly():
    model, cfg = _gqa_model(K=8, H=8, KVH=2, D=8, Out=6, seed=1)
    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    node = _gqa_node(pruned)
    num_heads, kv_num_heads = _gqa_attrs(node)
    group_size = cfg["H"] // cfg["KVH"]
    assert kv_num_heads == 1  # max(1, 2 - round(2*0.5))
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


def test_gqa_pruning_slices_bias_when_producer_has_one():
    # Gemm's own ONNX spec requires a rank-2 input, so a bias-carrying Gemm
    # producer can't sit directly ahead of GroupQueryAttention's rank-3
    # query/key/value inputs in a graph meant to actually run through
    # onnxruntime -- this exercises the bias-slicing path itself (shared
    # with every other producer match in this module via `_match_producer`)
    # directly against the initializers instead.
    model, cfg = _gqa_model(K=8, H=4, KVH=2, D=8, Out=6, seed=14, bias=True)
    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)

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


def test_gqa_pruning_reshape_hop_is_recognized_and_shape_updated():
    model, cfg = _gqa_model(K=8, H=4, KVH=2, D=8, Out=6, seed=3, with_reshape=True)
    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
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
    assert kv_num_heads == 1  # max(1, 2 - round(2*0.5)) == 1
    assert num_heads == 2  # group_size(2) * kv_num_heads(1)

    shape = onnx.numpy_helper.to_array(
        next(t for t in pruned.graph.initializer if t.name == "Shape")
    )
    assert shape[-1] == num_heads * cfg["D"]  # updated to the new (post-prune) Nq

    rng = np.random.default_rng(4)
    x = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    assert y.shape == (cfg["batch"], cfg["seq"], cfg["Out"])


def test_gqa_pruning_nonempty_past_kv_constant_matches_oracle_exactly():
    # A non-empty constant past_key/past_value holds real per-KV-head cache
    # data laid out along the kv_num_heads axis (BNSH format, confirmed via
    # `onnxruntime.capi.onnxruntime_pybind11_state.get_all_operator_schema()`
    # -- see `_past_kv_constants_are_sliceable`'s own docstring) -- sliced
    # along that same axis by the identical `keep_groups` index set K's/V's
    # own producer weights are sliced by, rather than declined. `batch=1`:
    # onnxruntime's own GroupQueryAttention CPU kernel requires
    # `batch_size == 1` whenever `sequence_length > 1` and a past context is
    # supplied (verified empirically, see this module's own "Attention-head
    # pruning" section comment).
    model, cfg = _gqa_model(
        K=8, H=8, KVH=2, D=8, Out=6, seed=12, batch=1, past_kv="nonempty"
    )
    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    node = _gqa_node(pruned)
    num_heads, kv_num_heads = _gqa_attrs(node)
    group_size = cfg["H"] // cfg["KVH"]
    assert kv_num_heads == 1  # max(1, 2 - round(2*0.5))
    assert num_heads == kv_num_heads * group_size

    keep_groups = _oracle_keep_groups(
        cfg["wq"], cfg["wk"], cfg["wv"], cfg["H"], cfg["KVH"], cfg["D"], kv_num_heads
    )
    keep_q_heads = _group_q_heads(keep_groups, group_size)
    d = cfg["D"]
    q_idx, kv_idx = _head_idx(keep_q_heads, d), _head_idx(keep_groups, d)

    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(
        inits["PastKey"], cfg["past_key"][:, keep_groups, :, :]
    )
    np.testing.assert_array_equal(
        inits["PastValue"], cfg["past_value"][:, keep_groups, :, :]
    )

    oracle, _ = _gqa_model(
        K=cfg["K"],
        H=num_heads,
        KVH=kv_num_heads,
        D=d,
        Out=cfg["Out"],
        seed=12,
        batch=cfg["batch"],
        seq=cfg["seq"],
        wq=cfg["wq"][:, q_idx],
        wk=cfg["wk"][:, kv_idx],
        wv=cfg["wv"][:, kv_idx],
        wout=cfg["wout"][q_idx, :],
        past_kv="nonempty",
        past_key=cfg["past_key"][:, keep_groups, :, :],
        past_value=cfg["past_value"][:, keep_groups, :, :],
    )
    onnx.checker.check_model(oracle)

    rng = np.random.default_rng(20)
    x = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(np.float32)
    (y_pruned,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y_pruned, y_oracle, rtol=1e-4, atol=1e-4)


def test_gqa_pruning_dynamic_past_kv_input_is_still_pruned():
    # A *dynamic* (non-constant) past_key/past_value -- an ordinary graph
    # input here, standing in for real runtime KV-cache data -- is not a
    # weight this module could corrupt by leaving it untouched, so it must
    # not block the match the way a non-empty constant does.
    model, cfg = _gqa_model(K=8, H=8, KVH=2, D=8, Out=6, seed=13, past_kv="dynamic")
    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    node = _gqa_node(pruned)
    num_heads, kv_num_heads = _gqa_attrs(node)
    assert kv_num_heads == 1
    assert num_heads == 4
    assert list(node.input[3:5]) == ["PastKeyIn", "PastValueIn"]

    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["Wq"].dims) == [8, 4 * cfg["D"]]
    assert list(inits["Wk"].dims) == [8, 1 * cfg["D"]]
    assert list(inits["Wv"].dims) == [8, 1 * cfg["D"]]


def test_gqa_pruning_quantized_past_kv_constant_with_no_scale_is_left_untouched():
    # A non-FLOAT constant past_key/past_value (standing in for a quantized
    # KV cache -- GroupQueryAttention's own schema allows `past_key`/
    # `past_value` to be `float8e4m3fn`/`uint8`/`int8` when quantized, per
    # `onnxruntime.capi.onnxruntime_pybind11_state.get_all_operator_schema()`'s
    # own "Quantization" section) with *no* `k_scale`/`v_scale` wired up at
    # all is declined outright by `_past_kv_constants_are_sliceable` rather
    # than sliced as if it were an ordinary float BNSH tensor: that schema
    # section states the corresponding scale "must be provided" whenever
    # quantization is enabled, so a quantized cache with no scale connected
    # is off-schema and not a shape this module can prove safe to touch --
    # unlike the case with a real, schema-conforming scale connected (see
    # `test_gqa_pruning_quantized_int8_per_tensor_scale_matches_oracle_exactly`
    # and `test_gqa_pruning_quantized_int8_per_channel_scale_matches_oracle_exactly`
    # below), which this module *does* now slice consistently.
    model, cfg = _gqa_model(K=8, H=8, KVH=2, D=8, Out=6, seed=25, past_kv="nonempty")
    inits_map = {t.name: t for t in model.graph.initializer}
    quantized_past_key = onnx.numpy_helper.from_array(
        onnx.numpy_helper.to_array(inits_map["PastKey"]).astype(np.uint8), "PastKey"
    )
    inits_map["PastKey"].CopyFrom(quantized_past_key)

    assert onnxsim.pruning._find_gqa_chains(model.graph) == []

    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    inits_before = {
        t.name: onnx.numpy_helper.to_array(t) for t in model.graph.initializer
    }
    inits_after = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer
    }
    for name in inits_before:
        np.testing.assert_array_equal(inits_before[name], inits_after[name])


def test_gqa_pruning_quantized_int8_per_tensor_scale_matches_oracle_exactly():
    # A quantized (`int8`) constant past_key/past_value with a real,
    # schema-conforming `"PER_TENSOR"` `k_scale`/`v_scale` (a single
    # broadcast scalar, shape `[1]` -- GroupQueryAttention's own
    # "Quantization Modes" doc section) is now matched and pruned: the cache
    # is sliced along its `kv_num_heads` axis by the same `keep_groups` used
    # for K's/V's own producer weights, exactly like an unquantized cache,
    # while the scale -- having no per-head axis at all -- is left
    # completely untouched. Verified by *actual execution*: real
    # onnxruntime (confirmed in this environment to run a quantized
    # `GroupQueryAttention` node on CPU, `int8` cache, unlike `uint8` which
    # this environment's onnxruntime CPU kernel errors on for any input --
    # a pre-existing runtime limitation unrelated to this module, not
    # exercised via execution here) on the pruned model matches a
    # from-scratch oracle model built with every tensor (weights, cache,
    # scale) pre-sliced by hand to the identical indices.
    K, H, KVH, D, Out = 8, 8, 2, 8, 6
    rng = np.random.default_rng(50)
    k_scale = np.array([0.05], dtype=np.float32)
    v_scale = np.array([0.03], dtype=np.float32)
    model, cfg = _gqa_model(
        K=K,
        H=H,
        KVH=KVH,
        D=D,
        Out=Out,
        seed=50,
        batch=1,
        past_kv="nonempty",
        past_kv_dtype=np.int8,
        k_scale=k_scale,
        v_scale=v_scale,
        k_quant_type="PER_TENSOR",
        v_quant_type="PER_TENSOR",
        kv_cache_bit_width=8,
    )
    onnx.checker.check_model(model)
    assert len(onnxsim.pruning._find_gqa_chains(model.graph)) == 1

    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    node = _gqa_node(pruned)
    num_heads, kv_num_heads = _gqa_attrs(node)
    assert kv_num_heads == 1  # max(1, 2 - round(2*0.5))
    group_size = H // KVH

    keep_groups = _oracle_keep_groups(
        cfg["wq"], cfg["wk"], cfg["wv"], H, KVH, D, kv_num_heads
    )
    keep_q_heads = _group_q_heads(keep_groups, group_size)
    q_idx, kv_idx = _head_idx(keep_q_heads, D), _head_idx(keep_groups, D)

    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(
        inits["PastKey"], cfg["past_key"][:, keep_groups, :, :]
    )
    np.testing.assert_array_equal(
        inits["PastValue"], cfg["past_value"][:, keep_groups, :, :]
    )
    # PER_TENSOR: no per-head axis -- the scale is untouched, byte-for-byte.
    np.testing.assert_array_equal(inits["KScale"], k_scale)
    np.testing.assert_array_equal(inits["VScale"], v_scale)

    oracle, _ = _gqa_model(
        K=K,
        H=num_heads,
        KVH=kv_num_heads,
        D=D,
        Out=Out,
        seed=50,
        batch=cfg["batch"],
        seq=cfg["seq"],
        wq=cfg["wq"][:, q_idx],
        wk=cfg["wk"][:, kv_idx],
        wv=cfg["wv"][:, kv_idx],
        wout=cfg["wout"][q_idx, :],
        past_kv="nonempty",
        past_key=cfg["past_key"][:, keep_groups, :, :],
        past_value=cfg["past_value"][:, keep_groups, :, :],
        past_kv_dtype=np.int8,
        k_scale=k_scale,
        v_scale=v_scale,
        k_quant_type="PER_TENSOR",
        v_quant_type="PER_TENSOR",
        kv_cache_bit_width=8,
    )
    onnx.checker.check_model(oracle)

    x = rng.standard_normal((cfg["batch"], cfg["seq"], K)).astype(np.float32)
    (y_pruned,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y_pruned, y_oracle, rtol=1e-4, atol=1e-4)


def test_gqa_pruning_quantized_int8_per_channel_scale_matches_oracle_exactly():
    # Same as the `"PER_TENSOR"` case above, but with a real
    # `"PER_CHANNEL"` `k_scale`/`v_scale` (`[1, kv_num_heads, 1, head_size]`
    # -- the *same* axis-1 `kv_num_heads` layout as the cache tensor itself,
    # per the schema's own doc section) holding per-KV-head scale data: the
    # scale is now sliced along axis 1 by the identical `keep_groups` index
    # set the cache tensor and K's/V's own producer weights are sliced by,
    # not merely left alone. Verified the same way, via real onnxruntime
    # execution against a from-scratch, hand-sliced oracle model.
    K, H, KVH, D, Out = 8, 8, 2, 8, 6
    rng = np.random.default_rng(51)
    k_scale = rng.uniform(0.01, 0.1, size=(1, KVH, 1, D)).astype(np.float32)
    v_scale = rng.uniform(0.01, 0.1, size=(1, KVH, 1, D)).astype(np.float32)
    model, cfg = _gqa_model(
        K=K,
        H=H,
        KVH=KVH,
        D=D,
        Out=Out,
        seed=51,
        batch=1,
        past_kv="nonempty",
        past_kv_dtype=np.int8,
        k_scale=k_scale,
        v_scale=v_scale,
        k_quant_type="PER_CHANNEL",
        v_quant_type="PER_CHANNEL",
        kv_cache_bit_width=8,
    )
    onnx.checker.check_model(model)
    assert len(onnxsim.pruning._find_gqa_chains(model.graph)) == 1

    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    node = _gqa_node(pruned)
    num_heads, kv_num_heads = _gqa_attrs(node)
    assert kv_num_heads == 1
    group_size = H // KVH

    keep_groups = _oracle_keep_groups(
        cfg["wq"], cfg["wk"], cfg["wv"], H, KVH, D, kv_num_heads
    )
    keep_q_heads = _group_q_heads(keep_groups, group_size)
    q_idx, kv_idx = _head_idx(keep_q_heads, D), _head_idx(keep_groups, D)

    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(
        inits["PastKey"], cfg["past_key"][:, keep_groups, :, :]
    )
    np.testing.assert_array_equal(
        inits["PastValue"], cfg["past_value"][:, keep_groups, :, :]
    )
    # PER_CHANNEL: sliced along axis 1 by the identical `keep_groups`.
    np.testing.assert_array_equal(inits["KScale"], k_scale[:, keep_groups, :, :])
    np.testing.assert_array_equal(inits["VScale"], v_scale[:, keep_groups, :, :])

    oracle, _ = _gqa_model(
        K=K,
        H=num_heads,
        KVH=kv_num_heads,
        D=D,
        Out=Out,
        seed=51,
        batch=cfg["batch"],
        seq=cfg["seq"],
        wq=cfg["wq"][:, q_idx],
        wk=cfg["wk"][:, kv_idx],
        wv=cfg["wv"][:, kv_idx],
        wout=cfg["wout"][q_idx, :],
        past_kv="nonempty",
        past_key=cfg["past_key"][:, keep_groups, :, :],
        past_value=cfg["past_value"][:, keep_groups, :, :],
        past_kv_dtype=np.int8,
        k_scale=k_scale[:, keep_groups, :, :],
        v_scale=v_scale[:, keep_groups, :, :],
        k_quant_type="PER_CHANNEL",
        v_quant_type="PER_CHANNEL",
        kv_cache_bit_width=8,
    )
    onnx.checker.check_model(oracle)

    x = rng.standard_normal((cfg["batch"], cfg["seq"], K)).astype(np.float32)
    (y_pruned,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y_pruned, y_oracle, rtol=1e-4, atol=1e-4)


def test_gqa_pruning_quantized_cache_malformed_scale_shape_is_declined():
    # A `k_scale`/`v_scale` constant that is neither the `"PER_TENSOR"`
    # single-broadcast-scalar shape nor the `"PER_CHANNEL"`
    # `[1, kv_num_heads, 1, head_size]` layout (here: a plain rank-2
    # `[kv_num_heads, head_size]` tensor, a shape neither this module nor
    # the schema's own two named quantization modes account for) declines
    # the whole match outright -- guessing at a slicing axis for a shape the
    # schema itself doesn't document would risk silently producing a
    # mismatched scale.
    K, H, KVH, D, Out = 8, 8, 2, 8, 6
    bad_scale = (
        np.random.default_rng(52).uniform(0.01, 0.1, size=(KVH, D)).astype(np.float32)
    )
    model, cfg = _gqa_model(
        K=K,
        H=H,
        KVH=KVH,
        D=D,
        Out=Out,
        seed=52,
        batch=1,
        past_kv="nonempty",
        past_kv_dtype=np.int8,
        k_scale=bad_scale,
        v_scale=bad_scale,
        k_quant_type="PER_CHANNEL",
        v_quant_type="PER_CHANNEL",
        kv_cache_bit_width=8,
    )
    assert onnxsim.pruning._find_gqa_chains(model.graph) == []

    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    inits_before = {
        t.name: onnx.numpy_helper.to_array(t) for t in model.graph.initializer
    }
    inits_after = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer
    }
    for name in inits_before:
        np.testing.assert_array_equal(inits_before[name], inits_after[name])


def test_gqa_pruning_quantized_cache_dynamic_scale_does_not_block_match():
    # A *dynamic* (non-constant, an ordinary graph input) `k_scale`/`v_scale`
    # -- even one shaped like a real `"PER_CHANNEL"` scale -- is left alone
    # exactly like a dynamic past_key/past_value: it is the caller's own
    # runtime data, not a weight this rewrite could corrupt by leaving
    # untouched, so it must not block the match. The quantized cache itself
    # is still sliced along its own `kv_num_heads` axis as usual.
    K, H, KVH, D, Out = 8, 8, 2, 8, 6
    rng = np.random.default_rng(53)
    wq = rng.standard_normal((K, H * D)).astype(np.float32)
    wk = rng.standard_normal((K, KVH * D)).astype(np.float32)
    wv = rng.standard_normal((K, KVH * D)).astype(np.float32)
    wout = rng.standard_normal((H * D, Out)).astype(np.float32)
    past_key = rng.integers(-64, 64, size=(1, KVH, 1, D)).astype(np.int8)
    past_value = rng.integers(-64, 64, size=(1, KVH, 1, D)).astype(np.int8)

    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 17, "com.microsoft": 1]
        >
        g (float[1,1,{K}] X, float[1,{KVH},1,{D}] KScaleIn, float[1,{KVH},1,{D}] VScaleIn)
        => (float[1,1,{Out}] Y)
        {{
          q = MatMul(X, Wq)
          k = MatMul(X, Wk)
          v = MatMul(X, Wv)
          ctx, pk, pv = com.microsoft.GroupQueryAttention
              <num_heads={H}, kv_num_heads={KVH}, k_quant_type = "PER_CHANNEL",
               v_quant_type = "PER_CHANNEL", kv_cache_bit_width=8>
              (q, k, v, PastKey, PastValue, SeqLensK, TotalSeq,
               "", "", "", "", "", KScaleIn, VScaleIn)
          Y = MatMul(ctx, Wout)
        }}
        """
    )
    model.graph.initializer.extend(
        [
            _f32(wq, "Wq"),
            _f32(wk, "Wk"),
            _f32(wv, "Wv"),
            _f32(wout, "Wout"),
            onnx.numpy_helper.from_array(past_key, "PastKey"),
            onnx.numpy_helper.from_array(past_value, "PastValue"),
            onnx.numpy_helper.from_array(np.array([0], dtype=np.int32), "SeqLensK"),
            onnx.numpy_helper.from_array(np.array(1, dtype=np.int32), "TotalSeq"),
        ]
    )
    onnx.checker.check_model(model)
    assert len(onnxsim.pruning._find_gqa_chains(model.graph)) == 1

    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    node = _gqa_node(pruned)
    num_heads, kv_num_heads = _gqa_attrs(node)
    assert kv_num_heads == 1
    # The scale inputs stay wired to the same (untouched) graph inputs.
    assert list(node.input[12:14]) == ["KScaleIn", "VScaleIn"]

    keep_groups = _oracle_keep_groups(wq, wk, wv, H, KVH, D, kv_num_heads)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["PastKey"], past_key[:, keep_groups, :, :])
    np.testing.assert_array_equal(inits["PastValue"], past_value[:, keep_groups, :, :])


def test_gqa_pruning_quantized_cache_accepts_every_schema_quantized_dtype():
    # `GroupQueryAttention`'s own `T_CACHE` type constraint allows exactly
    # three quantized dtypes for `past_key`/`past_value`:
    # `float8e4m3fn`/`uint8`/`int8` (confirmed via
    # `onnxruntime.capi.onnxruntime_pybind11_state.get_all_operator_schema()`).
    # `int8` is exercised end-to-end via real onnxruntime execution in
    # `test_gqa_pruning_quantized_int8_per_tensor_scale_matches_oracle_exactly`/
    # `..._per_channel_scale_matches_oracle_exactly` above; `uint8` cannot be
    # (this environment's onnxruntime CPU `GroupQueryAttention` kernel
    # errors -- "Tensor type mismatch" -- on *any* `uint8` cache, even an
    # unpruned, un-modified one, a pre-existing runtime limitation unrelated
    # to this module) and `float8e4m3fn` is not attempted via execution
    # either, so both are instead verified structurally here: the match
    # succeeds and the cache/scale tensors are sliced to the exact expected
    # values (direct tensor inspection), the fallback this task's own
    # instructions call for when execution isn't available.
    K, H, KVH, D, Out = 8, 8, 2, 8, 6
    for dtype in (np.uint8, ml_dtypes.float8_e4m3fn):
        k_scale = np.array([0.05], dtype=np.float32)
        v_scale = np.array([0.03], dtype=np.float32)
        model, cfg = _gqa_model(
            K=K,
            H=H,
            KVH=KVH,
            D=D,
            Out=Out,
            seed=54,
            batch=1,
            past_kv="nonempty",
            past_kv_dtype=dtype,
            k_scale=k_scale,
            v_scale=v_scale,
            k_quant_type="PER_TENSOR",
            v_quant_type="PER_TENSOR",
            kv_cache_bit_width=8,
        )
        onnx.checker.check_model(model)
        assert len(onnxsim.pruning._find_gqa_chains(model.graph)) == 1, dtype

        pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
        onnx.checker.check_model(pruned)
        node = _gqa_node(pruned)
        _, kv_num_heads = _gqa_attrs(node)
        keep_groups = _oracle_keep_groups(
            cfg["wq"], cfg["wk"], cfg["wv"], H, KVH, D, kv_num_heads
        )
        inits = {
            t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer
        }
        np.testing.assert_array_equal(
            inits["PastKey"], cfg["past_key"][:, keep_groups, :, :]
        )
        np.testing.assert_array_equal(
            inits["PastValue"], cfg["past_value"][:, keep_groups, :, :]
        )
        np.testing.assert_array_equal(inits["KScale"], k_scale)
        np.testing.assert_array_equal(inits["VScale"], v_scale)


def test_gqa_pruning_attention_bias_and_head_sink_are_sliced_and_match_oracle():
    # `GroupQueryAttention`'s own schema gives it two more optional inputs
    # beyond `past_key`/`past_value`/`k_scale`/`v_scale` that carry a
    # genuine per-*query*-head axis -- `attention_bias` (index 10, shape
    # `(batch_size or 1, num_heads or 1, sequence_length,
    # total_sequence_length)`, added after GQA's own internal KV-repeat, so
    # addressed per query head like Q's own producer weight) and
    # `head_sink` (index 11, shape `(num_heads,)`, a genuine
    # one-scalar-per-query-head softmax-smoothing constant) -- both
    # confirmed to have a real, non-ignored numeric effect via actual
    # onnxruntime execution, and both previously completely unhandled by
    # this matcher: an earlier version would have silently left
    # now-wrong-head-count `attention_bias`/`head_sink` tensors connected
    # to a pruned-head-count node -- a genuine correctness bug. This test
    # would fail against that earlier version: a stale full-width
    # `attention_bias`/`head_sink` raises a broadcast/shape error at
    # `onnxruntime.InferenceSession` run time, not just a numeric mismatch.
    K, H, KVH, D, Out = 8, 8, 2, 8, 6
    rng = np.random.default_rng(55)
    seq = 5
    attention_bias = (rng.standard_normal((1, H, seq, seq)) * 1000.0).astype(np.float32)
    head_sink = (rng.standard_normal((H,)) * 1000.0).astype(np.float32)
    model, cfg = _gqa_model(
        K=K,
        H=H,
        KVH=KVH,
        D=D,
        Out=Out,
        seed=55,
        attention_bias=attention_bias,
        head_sink=head_sink,
    )
    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    node = _gqa_node(pruned)
    num_heads, kv_num_heads = _gqa_attrs(node)
    group_size = H // KVH
    assert kv_num_heads == 1
    assert num_heads == kv_num_heads * group_size

    keep_groups = _oracle_keep_groups(cfg["wq"], cfg["wk"], cfg["wv"], H, KVH, D, 1)
    keep_q_heads = _group_q_heads(keep_groups, group_size)

    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(
        inits["AttentionBias"], attention_bias[:, keep_q_heads]
    )
    np.testing.assert_array_equal(inits["HeadSink"], head_sink[keep_q_heads])

    d = D
    q_idx, kv_idx = _head_idx(keep_q_heads, d), _head_idx(keep_groups, d)
    oracle, _ = _gqa_model(
        K=K,
        H=len(keep_q_heads),
        KVH=len(keep_groups),
        D=d,
        Out=Out,
        seed=55,
        wq=cfg["wq"][:, q_idx],
        wk=cfg["wk"][:, kv_idx],
        wv=cfg["wv"][:, kv_idx],
        wout=cfg["wout"][q_idx, :],
        batch=cfg["batch"],
        seq=seq,
        attention_bias=attention_bias[:, keep_q_heads],
        head_sink=head_sink[keep_q_heads],
    )

    rng2 = np.random.default_rng(56)
    x = rng2.standard_normal((cfg["batch"], seq, K)).astype(np.float32)
    (y_pruned,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y_pruned, y_oracle, rtol=1e-4, atol=1e-4)


def test_gqa_pruning_broadcast_attention_bias_is_left_untouched():
    # A `(1, 1, seq, seq)` `attention_bias` has its own `num_heads`-aligned
    # axis present but size 1 -- a broadcast, no per-head values -- so it
    # needs no slicing and must be left byte-identical.
    K, H, KVH, D, Out = 8, 8, 2, 8, 6
    rng = np.random.default_rng(57)
    seq = 5
    attention_bias = (rng.standard_normal((1, 1, seq, seq)) * 1000.0).astype(np.float32)
    model, cfg = _gqa_model(
        K=K, H=H, KVH=KVH, D=D, Out=Out, seed=57, attention_bias=attention_bias
    )
    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    node = _gqa_node(pruned)
    num_heads, kv_num_heads = _gqa_attrs(node)
    assert kv_num_heads == 1
    assert num_heads == 4

    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["AttentionBias"], attention_bias)


def test_gqa_pruning_ambiguous_attention_bias_shape_is_declined():
    # Axis 1 of this `attention_bias` is neither `num_heads` (8) nor 1 --
    # doesn't cleanly resolve either way, so the whole match must be
    # declined rather than guessed at.
    K, H, KVH, D, Out = 8, 8, 2, 8, 6
    rng = np.random.default_rng(58)
    seq = 5
    attention_bias = (rng.standard_normal((1, 3, seq, seq)) * 1000.0).astype(np.float32)
    model, cfg = _gqa_model(
        K=K, H=H, KVH=KVH, D=D, Out=Out, seed=58, attention_bias=attention_bias
    )
    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)

    inits_before = {
        t.name: onnx.numpy_helper.to_array(t) for t in model.graph.initializer
    }
    inits_after = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer
    }
    for name in inits_before:
        np.testing.assert_array_equal(inits_before[name], inits_after[name])


def test_gqa_pruning_malformed_head_sink_shape_is_declined():
    # `head_sink`'s own schema shape is exactly `(num_heads,)` -- any other
    # constant shape is declined rather than guessed at, the same
    # conservative treatment an unrecognized `past_key`/`past_value` shape
    # already gets.
    K, H, KVH, D, Out = 8, 8, 2, 8, 6
    rng = np.random.default_rng(59)
    head_sink = (rng.standard_normal((H + 1,)) * 1000.0).astype(np.float32)
    model, cfg = _gqa_model(
        K=K, H=H, KVH=KVH, D=D, Out=Out, seed=59, head_sink=head_sink
    )
    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)

    inits_before = {
        t.name: onnx.numpy_helper.to_array(t) for t in model.graph.initializer
    }
    inits_after = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer
    }
    for name in inits_before:
        np.testing.assert_array_equal(inits_before[name], inits_after[name])


def test_gqa_wanda_pruning_matches_oracle_exactly():
    # Calibration and eval data must share the model's own fixed batch/seq
    # here (unlike the plain-Attention wanda test, which uses symbolic
    # batch/seq dims): seqlens_k/total_sequence_length are baked-in
    # constants tied to a specific batch/seq (see _gqa_model), a real
    # constraint of GroupQueryAttention's own KV-cache-bookkeeping inputs,
    # not a limitation of this pass.
    model, cfg = _gqa_model(K=8, H=8, KVH=2, D=8, Out=6, seed=8)

    rng = np.random.default_rng(9)
    x_cal = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(np.float32)
    calibration_data = [{"X": x_cal}]

    pruned = onnxsim.apply_attention_head_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned)

    probe_model = onnx.ModelProto()
    probe_model.CopyFrom(model)
    probe_model.graph.output.append(onnx.ValueInfoProto(name="ctx"))
    (_, ctx_cal) = _run(probe_model, {"X": x_cal})
    act_norm = np.sqrt(np.mean(np.square(ctx_cal.astype(np.float64)), axis=(0, 1)))

    d = cfg["D"]
    group_size = cfg["H"] // cfg["KVH"]
    importance = np.zeros(cfg["KVH"])
    for kv in range(cfg["KVH"]):
        q_block = np.concatenate(
            [
                cfg["wq"][:, h * d : (h + 1) * d]
                for h in range(kv * group_size, (kv + 1) * group_size)
            ],
            axis=1,
        )
        k_block = cfg["wk"][:, kv * d : (kv + 1) * d]
        v_block = cfg["wv"][:, kv * d : (kv + 1) * d]
        base = np.linalg.norm(np.concatenate([q_block, k_block, v_block], axis=1))
        act_group = np.linalg.norm(
            act_norm[kv * group_size * d : (kv + 1) * group_size * d]
        )
        importance[kv] = base * max(act_group, 1e-8)
    keep_groups = np.sort(np.argsort(-importance)[:1])  # max(1, 2 - round(2*0.5)) == 1

    keep_q_heads = _group_q_heads(keep_groups, group_size)
    q_idx, kv_idx = _head_idx(keep_q_heads, d), _head_idx(keep_groups, d)

    oracle, _ = _gqa_model(
        K=cfg["K"],
        H=len(keep_q_heads),
        KVH=len(keep_groups),
        D=d,
        Out=cfg["Out"],
        seed=8,
        wq=cfg["wq"][:, q_idx],
        wk=cfg["wk"][:, kv_idx],
        wv=cfg["wv"][:, kv_idx],
        wout=cfg["wout"][q_idx, :],
        batch=cfg["batch"],
        seq=cfg["seq"],
    )

    x = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(np.float32)
    (y_pruned,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y_pruned, y_oracle, rtol=1e-4, atol=1e-4)


def test_gqa_wanda_pruning_falls_back_to_plain_with_no_calibration_batches():
    model, cfg = _gqa_model(K=8, H=8, KVH=2, D=8, Out=6, seed=10)
    plain = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    wanda = onnxsim.apply_attention_head_wanda_pruning(
        model, calibration_data=[], sparsity=0.5
    )
    inits_plain = {
        t.name: onnx.numpy_helper.to_array(t) for t in plain.graph.initializer
    }
    inits_wanda = {
        t.name: onnx.numpy_helper.to_array(t) for t in wanda.graph.initializer
    }
    for name in inits_plain:
        np.testing.assert_array_equal(inits_plain[name], inits_wanda[name])


# --- apply_attention_head_pruning / _wanda_pruning -- GroupQueryAttention,
# cross-attention (Q and K/V from genuinely different source tensors) ------
#
# See onnxsim/pruning.py's own "Attention-head pruning" section comment for
# the full investigation this proves: GroupQueryAttention's matchers never
# tie Q's own producer to K/V's own, so a real encoder/decoder pair (Q from
# one graph input, K/V from a different one, with its own different feature
# dimension) matches and prunes correctly -- oracle-verified here exactly
# like every self-attention GroupQueryAttention test above. Sequence length
# is kept equal between the two source tensors (a real onnxruntime
# GroupQueryAttention CPU-kernel restriction confirmed empirically, not a
# limitation this module's own matching/pruning logic adds -- see that same
# section comment); the plain ai.onnx Attention cross-attention section
# further below uses genuinely different sequence lengths too, since that
# op's own kernel has no such restriction.


def _gqa_cross_model(
    K_dec=8,
    K_enc=6,
    H=4,
    KVH=2,
    D=8,
    Out=6,
    seed=0,
    batch=2,
    seq=5,
    wq=None,
    wk=None,
    wv=None,
    wout=None,
):
    rng = np.random.default_rng(seed)
    Nq, Nkv = H * D, KVH * D
    if wq is None:
        wq = rng.standard_normal((K_dec, Nq)).astype(np.float32)
    if wk is None:
        wk = rng.standard_normal((K_enc, Nkv)).astype(np.float32)
    if wv is None:
        wv = rng.standard_normal((K_enc, Nkv)).astype(np.float32)
    if wout is None:
        wout = rng.standard_normal((Nq, Out)).astype(np.float32)

    initializer = [
        _f32(wq, "Wq"),
        _f32(wk, "Wk"),
        _f32(wv, "Wv"),
        _f32(wout, "Wout"),
        onnx.numpy_helper.from_array(
            np.full((batch,), seq - 1, dtype=np.int32), "SeqLensK"
        ),
        onnx.numpy_helper.from_array(np.array(seq, dtype=np.int32), "TotalSeq"),
    ]

    body = f"""
        g (float[{batch},{seq},{K_dec}] Xdec, float[{batch},{seq},{K_enc}] Xenc) => (float[{batch},{seq},{Out}] Y)
        {{
          q = MatMul(Xdec, Wq)
          k = MatMul(Xenc, Wk)
          v = MatMul(Xenc, Wv)
          ctx, pk, pv = com.microsoft.GroupQueryAttention <num_heads={H}, kv_num_heads={KVH}> (q, k, v, , , SeqLensK, TotalSeq)
          Y = MatMul(ctx, Wout)
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
        K_dec=K_dec,
        K_enc=K_enc,
        H=H,
        KVH=KVH,
        D=D,
        Out=Out,
        Nq=Nq,
        Nkv=Nkv,
        wq=wq,
        wk=wk,
        wv=wv,
        wout=wout,
        batch=batch,
        seq=seq,
    )


def _oracle_keep_groups_cross(
    wq, wk, wv, num_heads, kv_num_heads, head_size, keep_count
):
    # Like `_oracle_keep_groups` above, but combines each KV group's Q/K/V
    # block importance via sqrt(sum of squared per-block Frobenius norms)
    # rather than norm(concatenate(...)) -- required once wq's own row count
    # (Q's source tensor's own feature dimension) differs from wk's/wv's own
    # (K/V's source tensor's own feature dimension), exactly the shape this
    # helper is for. See `_gqa_group_importance`'s own updated comment in
    # onnxsim/pruning.py for why the two formulas agree whenever
    # concatenation is legal, and why only this one stays well-defined when
    # it isn't.
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
        importance[kv] = np.sqrt(
            np.linalg.norm(q_block) ** 2
            + np.linalg.norm(k_block) ** 2
            + np.linalg.norm(v_block) ** 2
        )
    return np.sort(np.argsort(-importance)[:keep_count])


def test_gqa_pruning_cross_attention_matches_oracle_exactly():
    # Without the `_gqa_group_importance` fix (concatenate-then-norm, which
    # requires wq's row count to equal wk's/wv's own), this raises a bare
    # numpy ValueError instead of reaching the assertions below -- K_dec=8 !=
    # K_enc=6 here is deliberate.
    model, cfg = _gqa_cross_model(K_dec=8, K_enc=6, H=8, KVH=2, D=8, Out=6, seed=20)
    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    node = _gqa_node(pruned)
    num_heads, kv_num_heads = _gqa_attrs(node)
    group_size = cfg["H"] // cfg["KVH"]
    assert kv_num_heads == 1  # max(1, 2 - round(2*0.5))
    assert num_heads == kv_num_heads * group_size

    keep_groups = _oracle_keep_groups_cross(
        cfg["wq"], cfg["wk"], cfg["wv"], cfg["H"], cfg["KVH"], cfg["D"], kv_num_heads
    )
    keep_q_heads = _group_q_heads(keep_groups, group_size)
    d = cfg["D"]
    q_idx, kv_idx = _head_idx(keep_q_heads, d), _head_idx(keep_groups, d)

    oracle, _ = _gqa_cross_model(
        K_dec=cfg["K_dec"],
        K_enc=cfg["K_enc"],
        H=num_heads,
        KVH=kv_num_heads,
        D=d,
        Out=cfg["Out"],
        seed=20,
        wq=cfg["wq"][:, q_idx],
        wk=cfg["wk"][:, kv_idx],
        wv=cfg["wv"][:, kv_idx],
        wout=cfg["wout"][q_idx, :],
        batch=cfg["batch"],
        seq=cfg["seq"],
    )

    rng = np.random.default_rng(21)
    xdec = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K_dec"])).astype(
        np.float32
    )
    xenc = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K_enc"])).astype(
        np.float32
    )
    (y_pruned,) = _run(pruned, {"Xdec": xdec, "Xenc": xenc})
    (y_oracle,) = _run(oracle, {"Xdec": xdec, "Xenc": xenc})
    np.testing.assert_allclose(y_pruned, y_oracle, rtol=1e-4, atol=1e-4)

    # Sanity: Q and K/V really are independently sourced, not accidentally
    # both reading the same tensor -- perturbing Xenc alone (Xdec held
    # fixed) must still change the output.
    (y_pruned2,) = _run(pruned, {"Xdec": xdec, "Xenc": xenc + 1.0})
    assert not np.allclose(y_pruned, y_pruned2)


def test_gqa_wanda_pruning_cross_attention_matches_oracle_exactly():
    model, cfg = _gqa_cross_model(K_dec=8, K_enc=6, H=8, KVH=2, D=8, Out=6, seed=22)

    rng = np.random.default_rng(23)
    xdec_cal = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K_dec"])).astype(
        np.float32
    )
    xenc_cal = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K_enc"])).astype(
        np.float32
    )
    calibration_data = [{"Xdec": xdec_cal, "Xenc": xenc_cal}]

    pruned = onnxsim.apply_attention_head_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned)

    probe_model = onnx.ModelProto()
    probe_model.CopyFrom(model)
    probe_model.graph.output.append(onnx.ValueInfoProto(name="ctx"))
    (_, ctx_cal) = _run(probe_model, {"Xdec": xdec_cal, "Xenc": xenc_cal})
    act_norm = np.sqrt(np.mean(np.square(ctx_cal.astype(np.float64)), axis=(0, 1)))

    d = cfg["D"]
    group_size = cfg["H"] // cfg["KVH"]
    importance = np.zeros(cfg["KVH"])
    for kv in range(cfg["KVH"]):
        q_block = np.concatenate(
            [
                cfg["wq"][:, h * d : (h + 1) * d]
                for h in range(kv * group_size, (kv + 1) * group_size)
            ],
            axis=1,
        )
        k_block = cfg["wk"][:, kv * d : (kv + 1) * d]
        v_block = cfg["wv"][:, kv * d : (kv + 1) * d]
        base = np.sqrt(
            np.linalg.norm(q_block) ** 2
            + np.linalg.norm(k_block) ** 2
            + np.linalg.norm(v_block) ** 2
        )
        act_group = np.linalg.norm(
            act_norm[kv * group_size * d : (kv + 1) * group_size * d]
        )
        importance[kv] = base * max(act_group, 1e-8)
    keep_groups = np.sort(np.argsort(-importance)[:1])  # max(1, 2 - round(2*0.5)) == 1

    keep_q_heads = _group_q_heads(keep_groups, group_size)
    q_idx, kv_idx = _head_idx(keep_q_heads, d), _head_idx(keep_groups, d)

    oracle, _ = _gqa_cross_model(
        K_dec=cfg["K_dec"],
        K_enc=cfg["K_enc"],
        H=len(keep_q_heads),
        KVH=len(keep_groups),
        D=d,
        Out=cfg["Out"],
        seed=22,
        wq=cfg["wq"][:, q_idx],
        wk=cfg["wk"][:, kv_idx],
        wv=cfg["wv"][:, kv_idx],
        wout=cfg["wout"][q_idx, :],
        batch=cfg["batch"],
        seq=cfg["seq"],
    )

    xdec = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K_dec"])).astype(
        np.float32
    )
    xenc = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K_enc"])).astype(
        np.float32
    )
    (y_pruned,) = _run(pruned, {"Xdec": xdec, "Xenc": xenc})
    (y_oracle,) = _run(oracle, {"Xdec": xdec, "Xenc": xenc})
    np.testing.assert_allclose(y_pruned, y_oracle, rtol=1e-4, atol=1e-4)


def test_attention_head_pruning_handles_attention_and_gqa_in_one_model():
    # Regression check for _apply_attention_chains's per-chain-type
    # dispatch: a plain `Attention` block and a `GroupQueryAttention` block
    # in the same graph, sharing no tensors, must each be pruned correctly
    # and independently -- one chain family must not disturb the other.
    K, H, D, Out1 = 8, 4, 4, 6
    Nqkv = H * D
    rng = np.random.default_rng(30)
    wqkv = rng.standard_normal((K, 3 * Nqkv)).astype(np.float32)
    # A real onnxruntime CPU build's `Attention` kernel can segfault given
    # only 2 inputs (no bias) -- see this file's own `_attention_model`
    # default and the other plain-Attention tests above, all of which
    # always give it one.
    bqkv = rng.standard_normal((3 * Nqkv,)).astype(np.float32)
    wout1 = rng.standard_normal((Nqkv, Out1)).astype(np.float32)

    GH, GKVH, GD, Out2 = 8, 2, 8, 5
    Nq2, Nkv2 = GH * GD, GKVH * GD
    wq = rng.standard_normal((K, Nq2)).astype(np.float32)
    wk = rng.standard_normal((K, Nkv2)).astype(np.float32)
    wv = rng.standard_normal((K, Nkv2)).astype(np.float32)
    wout2 = rng.standard_normal((Nq2, Out2)).astype(np.float32)

    batch, seq = 2, 5
    seqlens_k = np.full((batch,), seq - 1, dtype=np.int32)
    total_seq = np.array(seq, dtype=np.int32)

    model = _model(
        f"""
        g (float[{batch},{seq},{K}] X1, float[{batch},{seq},{K}] X2) => (float[{batch},{seq},{Out1}] Y1, float[{batch},{seq},{Out2}] Y2)
        {{
          ctx1 = com.microsoft.Attention <num_heads={H}, qkv_hidden_sizes=[{Nqkv},{Nqkv},{Nqkv}]> (X1, Wqkv, Bqkv)
          Y1 = MatMul(ctx1, Wout1)
          q = MatMul(X2, Wq)
          k = MatMul(X2, Wk)
          v = MatMul(X2, Wv)
          ctx2, pk, pv = com.microsoft.GroupQueryAttention <num_heads={GH}, kv_num_heads={GKVH}> (q, k, v, , , SeqLensK, TotalSeq)
          Y2 = MatMul(ctx2, Wout2)
        }}
        """,
        initializer=[
            _f32(wqkv, "Wqkv"),
            _f32(bqkv, "Bqkv"),
            _f32(wout1, "Wout1"),
            _f32(wq, "Wq"),
            _f32(wk, "Wk"),
            _f32(wv, "Wv"),
            _f32(wout2, "Wout2"),
            onnx.numpy_helper.from_array(seqlens_k, "SeqLensK"),
            onnx.numpy_helper.from_array(total_seq, "TotalSeq"),
        ],
        opset=17,
    )
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))

    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    attn_node = next(n for n in pruned.graph.node if n.op_type == "Attention")
    gqa_node = next(n for n in pruned.graph.node if n.op_type == "GroupQueryAttention")
    attn_heads, _ = _attention_attrs(attn_node)
    gqa_heads, gqa_kv_heads = _gqa_attrs(gqa_node)
    assert attn_heads == 2
    assert gqa_kv_heads == 1
    assert gqa_heads == 4

    rng2 = np.random.default_rng(31)
    x1 = rng2.standard_normal((batch, seq, K)).astype(np.float32)
    x2 = rng2.standard_normal((batch, seq, K)).astype(np.float32)
    y1, y2 = _run(pruned, {"X1": x1, "X2": x2})
    assert y1.shape == (batch, seq, Out1)
    assert y2.shape == (batch, seq, Out2)


# --- magnitude/Wanda/SparseGPT pruning: Attention/GQA weights -----------
#
# The value-only (unstructured/N:M) pruning functions above -- as opposed
# to this file's own attention *head* pruning section, which removes whole
# heads -- reuse `_candidates()`/`_prune_weight()` unchanged for these
# layer types; see onnxsim/pruning.py's own module docstring and
# `_candidates`'s own docstring for the full reasoning. `_attention_model`/
# `_gqa_model` (defined above, in the head-pruning section) are reused here
# too -- same fused-attention topology, just pruned along a different axis.


def test_magnitude_pruning_attention_merged_weight_reaches_target_sparsity():
    model, cfg = _attention_model(K=8, H=4, D=4, Out=6, seed=20)
    pruned = onnxsim.apply_magnitude_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    inits = {t.name: t for t in pruned.graph.initializer}
    wqkv_pruned = onnx.numpy_helper.to_array(inits["Wqkv"])
    assert wqkv_pruned.shape == cfg["wqkv"].shape
    zeros = np.count_nonzero(wqkv_pruned == 0)
    assert zeros / wqkv_pruned.size == pytest.approx(0.5, abs=1e-9)

    # Bias is never touched by this pass -- only the matched weight is.
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(inits["Bqkv"]), cfg["bqkv"]
    )

    # num_heads/qkv_hidden_sizes describe the merged weight's *column
    # layout*, not any zeroed-vs-nonzero split within it -- since this pass
    # only zeros values and never reshapes the weight, the whole node (not
    # just these two attributes) must come out byte-identical.
    node_before = _attention_node(model)
    node_after = _attention_node(pruned)
    assert node_after.SerializeToString() == node_before.SerializeToString()
    num_heads, qkv = _attention_attrs(node_after)
    assert num_heads == cfg["H"]
    assert qkv == [cfg["Nq"], cfg["Nk"], cfg["Nv"]]


def test_magnitude_pruning_attention_merged_weight_keeps_the_largest_entries_per_column():
    model, cfg = _attention_model(K=8, H=4, D=4, Out=6, seed=21)
    pruned = onnxsim.apply_magnitude_pruning(model, sparsity=0.75)

    inits = {t.name: t for t in pruned.graph.initializer}
    w = cfg["wqkv"].astype(np.float64)  # [K, N]
    w_pruned = onnx.numpy_helper.to_array(inits["Wqkv"]).astype(np.float64)
    keep_count = round(cfg["K"] * 0.25)
    for col in range(w.shape[1]):
        kept = np.flatnonzero(w_pruned[:, col] != 0)
        assert len(kept) == keep_count
        threshold = np.abs(w[:, col])[kept].min()
        dropped_max = np.abs(w[:, col])[np.flatnonzero(w_pruned[:, col] == 0)].max()
        assert dropped_max <= threshold


def test_magnitude_pruning_attention_merged_weight_nm_pattern():
    model, cfg = _attention_model(K=16, H=4, D=4, Out=6, seed=22)
    pruned = onnxsim.apply_magnitude_pruning(model, n=2, m=4)
    onnx.checker.check_model(pruned)

    inits = {t.name: t for t in pruned.graph.initializer}
    w_pruned = onnx.numpy_helper.to_array(inits["Wqkv"]).T  # [N, K]
    for row in w_pruned:
        for start in range(0, len(row), 4):
            group = row[start : start + 4]
            assert np.count_nonzero(group) <= 2


def test_wanda_pruning_attention_merged_weight_uses_calibrated_activation_norm():
    # Unlike an ordinary MatMul/Gemm layer with a rank-3 activation (see
    # test_wanda_pruning_falls_back_to_magnitude_without_matching_activation,
    # just above, which must keep falling back to plain magnitude
    # unmodified), `Attention`'s merged QKV weight gets its own,
    # separately-accumulated activation statistic that reduces `X`
    # ([batch, seq, hidden]) over every leading axis (mirroring
    # apply_sparsegpt_pruning's own x.reshape(-1, x.shape[-1])) -- see
    # apply_wanda_pruning's own docstring. A handful of input (K-axis)
    # features carry deliberately inflated activation magnitude but
    # otherwise-ordinary weight magnitude -- Wanda's own motivating scenario,
    # exactly mirroring test_wanda_pruning_protects_high_activation_channels's
    # plain-MatMul version of this same test.
    K, H, D, Out = 8, 4, 4, 6
    salient = (0, 3, 5)
    model, cfg = _attention_model(K=K, H=H, D=D, Out=Out, seed=23)
    rng = np.random.default_rng(24)
    x = rng.standard_normal((2, 5, K)).astype(np.float32)
    for c in salient:
        x[:, :, c] *= 25.0
    calibration_data = [{"X": x}]

    magnitude_pruned = onnxsim.apply_magnitude_pruning(model, sparsity=0.5)
    wanda_pruned = onnxsim.apply_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(wanda_pruned)

    inits_m = {t.name: t for t in magnitude_pruned.graph.initializer}
    inits_w = {t.name: t for t in wanda_pruned.graph.initializer}
    w_magnitude = onnx.numpy_helper.to_array(inits_m["Wqkv"])  # [K, N]
    w_wanda = onnx.numpy_helper.to_array(inits_w["Wqkv"])

    # The calibration signal must actually be used, not just tolerated
    # without error -- Wanda's result must differ from plain magnitude
    # pruning of the same weight under the same skewed activation profile
    # that left them identical before this fix (see the docstring above).
    assert not np.array_equal(w_magnitude, w_wanda)

    # The salient K-axis rows must keep strictly more nonzero entries under
    # Wanda than under plain magnitude -- the same protection
    # test_wanda_pruning_protects_high_activation_channels checks for a
    # plain MatMul layer, now proven for Attention's merged weight too.
    salient_kept_magnitude = np.count_nonzero(w_magnitude[list(salient), :])
    salient_kept_wanda = np.count_nonzero(w_wanda[list(salient), :])
    assert salient_kept_wanda > salient_kept_magnitude

    # Cross-check against a hand-rolled oracle that reimplements the exact
    # documented metric independently of onnxsim's own internals: reduce X
    # over every leading axis (reshape(-1, K), SparseGPT's own convention),
    # take the per-feature L2 norm, and per output column (this weight's
    # per-row Wanda comparison group, one column at [K, N] per output
    # channel) rank |W_ij| * ||X_j||_2 -- exactly
    # test_sparsegpt_pruning_attention_merged_weight_matches_reference_transliteration's
    # own transliteration-oracle style, just for Wanda's own metric instead
    # of SparseGPT's Hessian-corrected one.
    x_flat = x.reshape(-1, K).astype(np.float64)
    act_norm = np.sqrt(np.square(x_flat).sum(axis=0) / x_flat.shape[0])
    w = cfg["wqkv"].astype(np.float64)  # [K, N]
    w_nk = w.T  # [N, K], output channel first -- Wanda's own comparison axis
    importance = np.abs(w_nk) * np.maximum(act_norm[np.newaxis, :], 1e-8)
    keep = round(K * 0.5)
    order = np.argsort(importance, axis=1)
    drop = order[:, : K - keep]
    mask = np.ones_like(w_nk, dtype=bool)
    np.put_along_axis(mask, drop, False, axis=1)
    expected = np.where(mask, w_nk, 0.0).T  # back to [K, N]
    np.testing.assert_array_equal(expected, w_wanda)


def test_sparsegpt_pruning_attention_merged_weight_matches_reference_transliteration():
    # Unlike Conv, `Attention`'s merged QKV weight has a plain `[*, K]`
    # activation as its own input (no im2col unfolding), so it is
    # deliberately included in SparseGPT's candidate list -- see
    # apply_sparsegpt_pruning's own docstring. `H = X^T X` is computed the
    # same way as any other MatMul/Gemm layer, reduced over every leading
    # axis of the rank-3 `X`.
    K = 8
    model, cfg = _attention_model(K=K, H=4, D=4, Out=6, seed=25)
    rng = np.random.default_rng(26)
    x_cal = rng.standard_normal((3, 6, K)).astype(np.float32)  # [batch, seq, K]

    pruned = onnxsim.apply_sparsegpt_pruning(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5, proc_block_size=6
    )
    onnx.checker.check_model(pruned)

    x_flat = x_cal.reshape(-1, K).astype(np.float64)
    w_nk = cfg["wqkv"].T.astype(np.float64)  # [N, K]
    h = x_flat.T @ x_flat
    expected_nk = _reference_sparsegpt(
        w_nk, h, sparsity=0.5, n=None, m=None, percdamp=0.01, blocksize=6
    )

    inits = {t.name: t for t in pruned.graph.initializer}
    w_pruned = onnx.numpy_helper.to_array(inits["Wqkv"]).T.astype(np.float64)
    np.testing.assert_allclose(w_pruned, expected_nk, rtol=1e-6, atol=1e-6)

    node_before = _attention_node(model)
    node_after = _attention_node(pruned)
    assert node_after.SerializeToString() == node_before.SerializeToString()


def test_magnitude_pruning_gqa_qkv_weights_already_matched():
    # Not new behavior: Wq/Wk/Wv are ordinary MatMul producers feeding
    # GroupQueryAttention, not weights the op itself owns (see
    # _match_gqa_producer's own docstring) -- `_candidates` already matched
    # them as plain MatMul/Gemm layers before this task added any
    # Attention-specific matching at all. This test exists to prove that
    # explicitly rather than leaving it implicit.
    model, cfg = _gqa_model(K=8, H=4, KVH=2, D=8, Out=6, seed=27)
    pruned = onnxsim.apply_magnitude_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    inits = {t.name: t for t in pruned.graph.initializer}
    for name, original in (("Wq", cfg["wq"]), ("Wk", cfg["wk"]), ("Wv", cfg["wv"])):
        w = onnx.numpy_helper.to_array(inits[name])
        assert w.shape == original.shape
        zeros = np.count_nonzero(w == 0)
        assert zeros / w.size == pytest.approx(0.5, abs=1e-9)
        assert not np.array_equal(w, original)

    # num_heads/kv_num_heads are untouched -- this is a value-only rewrite
    # of Wq/Wk/Wv/Wout, not the structural head-pruning this file's own
    # `apply_attention_head_pruning` performs on the same node type.
    node_before = _gqa_node(model)
    node_after = _gqa_node(pruned)
    assert node_after.SerializeToString() == node_before.SerializeToString()

    rng = np.random.default_rng(28)
    x = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    assert y.shape == (cfg["batch"], cfg["seq"], cfg["Out"])


# --- apply_attention_head_pruning / _wanda_pruning -- plain ai.onnx Attention --
#
# ``onnx.defs.get_schema("Attention", domain="")`` against this environment's
# installed ``onnx==1.22.0`` confirms: `since_version=24`, inputs `Q, K, V,
# attn_mask?, past_key?, past_value?, nonpad_kv_seqlen?`, attributes
# `q_num_heads`/`kv_num_heads` (both schema-optional, required here -- see
# `_match_onnx_attention_producer`'s own docstring), and -- per the op's own
# backend-test suite (`onnx/backend/test/case/node/attention.py`) -- V may
# carry its own, independent head_size, a shape this pass declines rather
# than mis-slices (see `test_onnx_attention_pruning_diff_v_head_size_...`
# below). onnxruntime 1.29.0 in this environment executes the op directly
# (confirmed empirically), so every test below runs the real oracle-vs-
# onnxruntime comparison the rest of this file uses, the same as the
# `com.microsoft::GroupQueryAttention` section above -- no structural-only
# fallback was needed for this op.


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
    attn_mask_array=None,  # explicit constant array for attn_mask="nonempty" (else zeros((seq, seq)))
    past_kv=None,  # None (omitted) | "nonempty" (constant) | "dynamic" (graph input)
    past_key=None,  # explicit override for past_kv="nonempty" (else random)
    past_value=None,  # explicit override for past_kv="nonempty" (else random)
    Dv=None,  # V's own head_size, if it should genuinely differ from D
):
    # Unlike `com.microsoft::GroupQueryAttention` (see `_gqa_model`'s own
    # comment), this op's real onnxruntime CPU kernel has no observed
    # head_size-multiple-of-8 requirement -- verified empirically above --
    # so D defaults to a small 4, mirroring `_attention_model`'s own default.
    #
    # `Dv` (V's own head_size, defaulting to `D` -- the uniform case every
    # other caller of this helper wants) is independent of Q/K's `D`: unlike
    # `_gqa_model`'s `GroupQueryAttention`, this op's own schema genuinely
    # allows the two to differ (see this module's own "Attention-head
    # pruning" section comment), and the raw output/output-projection's own
    # reduction dim is sized off `Dv`, not `D` -- `H * Dv`, not `H * D`.
    if Dv is None:
        Dv = D
    rng = np.random.default_rng(seed)
    Nq, Nk, Nv = H * D, KVH * D, KVH * Dv
    if wq is None:
        wq = rng.standard_normal((K, Nq)).astype(np.float32)
    if wk is None:
        wk = rng.standard_normal((K, Nk)).astype(np.float32)
    if wv is None:
        wv = rng.standard_normal((K, Nv)).astype(np.float32)
    if wout is None:
        wout = rng.standard_normal((H * Dv, Out)).astype(np.float32)

    initializer = [_f32(wq, "Wq"), _f32(wk, "Wk"), _f32(wv, "Wv"), _f32(wout, "Wout")]
    q_op, k_op, v_op = "MatMul(X, Wq)", "MatMul(X, Wk)", "MatMul(X, Wv)"
    if bias:
        if bq is None:
            bq = rng.standard_normal((Nq,)).astype(np.float32)
        if bk is None:
            bk = rng.standard_normal((Nk,)).astype(np.float32)
        if bv is None:
            bv = rng.standard_normal((Nv,)).astype(np.float32)
        initializer += [_f32(bq, "Bq"), _f32(bk, "Bk"), _f32(bv, "Bv")]
        q_op, k_op, v_op = "Gemm(X, Wq, Bq)", "Gemm(X, Wk, Bk)", "Gemm(X, Wv, Bv)"

    operands = ["q", "k", "v"]
    extra_graph_inputs = ""

    if attn_mask == "nonempty":
        mask = (
            np.zeros((seq, seq), dtype=np.float32)
            if attn_mask_array is None
            else np.asarray(attn_mask_array, dtype=np.float32)
        )
        initializer.append(_f32(mask, "AttnMask"))
        operands.append("AttnMask")
    elif attn_mask == "dynamic":
        operands.append("AttnMaskIn")
        extra_graph_inputs += f", float[{seq},{seq}] AttnMaskIn"
    else:
        operands.append("")

    if past_kv == "nonempty":
        if past_key is None:
            past_key = rng.standard_normal((batch, KVH, 1, D)).astype(np.float32)
        if past_value is None:
            past_value = rng.standard_normal((batch, KVH, 1, Dv)).astype(np.float32)
        initializer += [_f32(past_key, "PastKey"), _f32(past_value, "PastValue")]
        operands += ["PastKey", "PastValue"]
    elif past_kv == "dynamic":
        operands += ["PastKeyIn", "PastValueIn"]
        extra_graph_inputs += (
            f", float[{batch},{KVH},1,{D}] PastKeyIn"
            f", float[{batch},{KVH},1,{Dv}] PastValueIn"
        )
    else:
        operands += ["", ""]

    # Trailing optional inputs may simply be omitted from the node's own
    # input list rather than spelled out as empty placeholders.
    while operands and operands[-1] == "":
        operands.pop()

    if with_reshape:
        shape = np.array([batch, seq, H * Dv], dtype=np.int64)
        initializer.append(onnx.numpy_helper.from_array(shape, "Shape"))
        tail = "ctx2 = Reshape(ctx, Shape)\n          Y = MatMul(ctx2, Wout)"
    else:
        tail = "Y = MatMul(ctx, Wout)"

    # onnxruntime's own kernel for this op requires `present_key`/
    # `present_value` (indices 1/2) declared as node outputs whenever
    # `past_key`/`past_value` are connected at all (verified empirically:
    # "The implementation does not support past_key provided and
    # present_key being null") -- unused beyond the node itself, just like
    # any other output no downstream node consumes.
    ctx_outputs = "ctx, present_key, present_value" if past_kv else "ctx"
    body = f"""
        g (float[{batch},{seq},{K}] X{extra_graph_inputs}) => (float[{batch},{seq},{Out}] Y)
        {{
          q = {q_op}
          k = {k_op}
          v = {v_op}
          {ctx_outputs} = Attention <q_num_heads={H}, kv_num_heads={KVH}> ({", ".join(operands)})
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
        Dv=Dv,
        Out=Out,
        Nq=Nq,
        Nkv=Nk,
        Nk=Nk,
        Nv=Nv,
        wq=wq,
        wk=wk,
        wv=wv,
        bq=bq,
        bk=bk,
        bv=bv,
        wout=wout,
        batch=batch,
        seq=seq,
        past_key=past_key,
        past_value=past_value,
        attn_mask=mask if attn_mask == "nonempty" else None,
    )


def _onnx_attention_node(model):
    # Filters on `.domain` too, not just `.op_type == "Attention"` (unlike
    # this file's own `_attention_node`) -- necessary once a graph can
    # contain both this op and `com.microsoft::Attention` side by side (see
    # `test_attention_head_pruning_handles_all_three_attention_op_types_...`
    # below), which share the bare op_type "Attention" but not the domain.
    return next(
        n for n in model.graph.node if n.op_type == "Attention" and n.domain == ""
    )


def _onnx_attention_attrs(node):
    q_num_heads = next(a.i for a in node.attribute if a.name == "q_num_heads")
    kv_num_heads = next(a.i for a in node.attribute if a.name == "kv_num_heads")
    return q_num_heads, kv_num_heads


def test_onnx_attention_pruning_shrinks_matched_block():
    model, cfg = _onnx_attention_model(K=8, H=4, KVH=4, D=4, Out=6)
    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    node = _onnx_attention_node(pruned)
    q_num_heads, kv_num_heads = _onnx_attention_attrs(node)
    assert q_num_heads == 2  # round(4 - 4*0.5) query heads ...
    assert kv_num_heads == 2  # ... and KV heads alike, since group_size == 1 here

    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["Wq"].dims) == [8, 8]
    assert list(inits["Wk"].dims) == [8, 8]
    assert list(inits["Wv"].dims) == [8, 8]
    assert list(inits["Wout"].dims) == [8, 6]

    rng = np.random.default_rng(1)
    x = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    assert y.shape == (cfg["batch"], cfg["seq"], cfg["Out"])


def test_weight_sparsity_includes_attention_and_gqa_weights():
    model, cfg = _attention_model(K=8, H=4, D=4, Out=6, seed=29)
    assert onnxsim.weight_sparsity(model) == 0.0
    pruned = onnxsim.apply_magnitude_pruning(model, sparsity=0.5)
    # Both Wqkv (newly matched here) and Wout (already matched before this
    # task) are now exactly half-zero, so the whole model's aggregate
    # sparsity is still 0.5 -- automatic, since weight_sparsity shares
    # `_candidates` with every apply_*_pruning function.
    assert onnxsim.weight_sparsity(pruned) == pytest.approx(0.5, abs=1e-9)

    gqa_model, _ = _gqa_model(K=8, H=4, KVH=2, D=8, Out=6, seed=30)
    assert onnxsim.weight_sparsity(gqa_model) == 0.0
    gqa_pruned = onnxsim.apply_magnitude_pruning(gqa_model, sparsity=0.5)
    assert onnxsim.weight_sparsity(gqa_pruned) == pytest.approx(0.5, abs=1e-9)


def test_onnx_attention_pruning_zero_sparsity_is_a_no_op():
    model, cfg = _onnx_attention_model(K=8, H=8, KVH=2, D=4, Out=6, seed=7)
    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.0)
    node = _onnx_attention_node(pruned)
    q_num_heads, kv_num_heads = _onnx_attention_attrs(node)
    assert q_num_heads == cfg["H"]
    assert kv_num_heads == cfg["KVH"]
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["Wq"], cfg["wq"])
    np.testing.assert_array_equal(inits["Wk"], cfg["wk"])
    np.testing.assert_array_equal(inits["Wv"], cfg["wv"])


def test_onnx_attention_pruning_unequal_heads_drops_whole_groups_and_preserves_ratio():
    # 8 query heads sharing 4 KV heads (2 query heads per KV head); at
    # sparsity=0.5 two of the four *groups* must be dropped, never an
    # individual query head in isolation -- exactly the same GQA-style
    # granularity `com.microsoft::GroupQueryAttention` enforces (see
    # `test_gqa_pruning_unequal_heads_drops_whole_groups_and_preserves_ratio`),
    # since this pass reuses that op's own `_apply_one_gqa_chain` unmodified.
    model, cfg = _onnx_attention_model(K=8, H=8, KVH=4, D=4, Out=6, seed=11)
    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    node = _onnx_attention_node(pruned)
    q_num_heads, kv_num_heads = _onnx_attention_attrs(node)
    assert kv_num_heads == 2
    assert q_num_heads == 4
    assert q_num_heads // kv_num_heads == cfg["H"] // cfg["KVH"]  # ratio preserved

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


def test_onnx_attention_pruning_matches_oracle_exactly():
    model, cfg = _onnx_attention_model(K=8, H=8, KVH=2, D=4, Out=6, seed=1)
    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    node = _onnx_attention_node(pruned)
    q_num_heads, kv_num_heads = _onnx_attention_attrs(node)
    group_size = cfg["H"] // cfg["KVH"]
    assert kv_num_heads == 1  # max(1, 2 - round(2*0.5))
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


def test_onnx_attention_pruning_slices_bias_when_producer_has_one():
    # Gemm's own ONNX spec requires a rank-2 input, so a bias-carrying Gemm
    # producer can't sit directly ahead of this op's rank-3 query/key/value
    # inputs in a graph meant to actually run through onnxruntime (see
    # `test_gqa_pruning_slices_bias_when_producer_has_one`'s own comment) --
    # this exercises the bias-slicing path (shared with every other producer
    # match in this module via `_match_producer`) directly against the
    # initializers instead.
    model, cfg = _onnx_attention_model(K=8, H=4, KVH=2, D=4, Out=6, seed=14, bias=True)
    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)

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


def test_onnx_attention_pruning_reshape_hop_is_recognized_and_shape_updated():
    model, cfg = _onnx_attention_model(
        K=8, H=4, KVH=2, D=4, Out=6, seed=3, with_reshape=True
    )
    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    assert [n.op_type for n in pruned.graph.node] == [
        "MatMul",
        "MatMul",
        "MatMul",
        "Attention",
        "Reshape",
        "MatMul",
    ]

    node = _onnx_attention_node(pruned)
    q_num_heads, kv_num_heads = _onnx_attention_attrs(node)
    assert kv_num_heads == 1  # max(1, 2 - round(2*0.5)) == 1
    assert q_num_heads == 2  # group_size(2) * kv_num_heads(1)

    shape = onnx.numpy_helper.to_array(
        next(t for t in pruned.graph.initializer if t.name == "Shape")
    )
    assert shape[-1] == q_num_heads * cfg["D"]  # updated to the new (post-prune) Nq

    rng = np.random.default_rng(4)
    x = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(np.float32)
    (y,) = _run(pruned, {"X": x})
    assert y.shape == (cfg["batch"], cfg["seq"], cfg["Out"])


def test_onnx_attention_pruning_nonempty_past_kv_constant_matches_oracle_exactly():
    # A non-empty constant past_key/past_value holds real per-KV-head cache
    # data laid out along the kv_num_heads axis (BNSH format, confirmed via
    # `onnx.defs.get_schema("Attention", domain="")` -- see
    # `_past_kv_constants_are_sliceable`'s own docstring) -- sliced along
    # that same axis by the identical `keep_groups` index set K's/V's own
    # producer weights are sliced by, mirroring
    # `test_gqa_pruning_nonempty_past_kv_constant_matches_oracle_exactly`
    # (`_match_onnx_attention_producer` shares the identical safety gate and
    # `_apply_one_gqa_chain` the identical slicing).
    model, cfg = _onnx_attention_model(
        K=8, H=8, KVH=2, D=4, Out=6, seed=12, past_kv="nonempty"
    )
    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    node = _onnx_attention_node(pruned)
    q_num_heads, kv_num_heads = _onnx_attention_attrs(node)
    group_size = cfg["H"] // cfg["KVH"]
    assert kv_num_heads == 1  # max(1, 2 - round(2*0.5))
    assert q_num_heads == kv_num_heads * group_size

    keep_groups = _oracle_keep_groups(
        cfg["wq"], cfg["wk"], cfg["wv"], cfg["H"], cfg["KVH"], cfg["D"], kv_num_heads
    )
    keep_q_heads = _group_q_heads(keep_groups, group_size)
    d = cfg["D"]
    q_idx, kv_idx = _head_idx(keep_q_heads, d), _head_idx(keep_groups, d)

    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(
        inits["PastKey"], cfg["past_key"][:, keep_groups, :, :]
    )
    np.testing.assert_array_equal(
        inits["PastValue"], cfg["past_value"][:, keep_groups, :, :]
    )

    oracle, _ = _onnx_attention_model(
        K=cfg["K"],
        H=q_num_heads,
        KVH=kv_num_heads,
        D=d,
        Out=cfg["Out"],
        seed=12,
        batch=cfg["batch"],
        seq=cfg["seq"],
        wq=cfg["wq"][:, q_idx],
        wk=cfg["wk"][:, kv_idx],
        wv=cfg["wv"][:, kv_idx],
        wout=cfg["wout"][q_idx, :],
        past_kv="nonempty",
        past_key=cfg["past_key"][:, keep_groups, :, :],
        past_value=cfg["past_value"][:, keep_groups, :, :],
    )
    onnx.checker.check_model(oracle)

    rng = np.random.default_rng(21)
    x = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(np.float32)
    (y_pruned,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y_pruned, y_oracle, rtol=1e-4, atol=1e-4)


def test_onnx_attention_pruning_dynamic_past_kv_input_is_still_pruned():
    # A *dynamic* (non-constant) past_key/past_value -- an ordinary graph
    # input here -- is not a weight this module could corrupt by leaving it
    # untouched, so it must not block the match the way a non-empty constant
    # does (mirrors `test_gqa_pruning_dynamic_past_kv_input_is_still_pruned`).
    model, cfg = _onnx_attention_model(
        K=8, H=8, KVH=2, D=4, Out=6, seed=13, past_kv="dynamic"
    )
    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    node = _onnx_attention_node(pruned)
    q_num_heads, kv_num_heads = _onnx_attention_attrs(node)
    assert kv_num_heads == 1
    assert q_num_heads == 4
    assert list(node.input[4:6]) == ["PastKeyIn", "PastValueIn"]

    inits = {t.name: t for t in pruned.graph.initializer}
    assert list(inits["Wq"].dims) == [8, 4 * cfg["D"]]
    assert list(inits["Wk"].dims) == [8, 1 * cfg["D"]]
    assert list(inits["Wv"].dims) == [8, 1 * cfg["D"]]


def test_onnx_attention_pruning_nonempty_2d_attn_mask_constant_is_pruned():
    # `attn_mask` is an optional input this op has that `GroupQueryAttention`
    # does not. A rank-2 `(seq, seq)` constant -- the shape this file's own
    # `_onnx_attention_model` helper generates for `attn_mask="nonempty"` --
    # broadcasts against the op's own `(batch, q_num_heads, q_seq, kv_seq)`
    # attention-score tensor with *no* axis ever landing on the
    # `q_num_heads` slot at all (rank 2 < 3, see `_head_bias_axis`'s own
    # docstring for the right-alignment reasoning): it is unconditionally
    # head-count-independent, so this must now be pruned exactly like the
    # no-mask case -- an earlier version of this matcher declined *any*
    # non-empty constant mask here regardless of shape, which was overly
    # conservative for exactly this common case (see
    # `test_onnx_attention_pruning_rank4_attn_mask_head_axis_is_sliced_and_matches_oracle`
    # below for the genuinely-per-head case this matcher still must, and
    # now does, slice correctly).
    model, cfg = _onnx_attention_model(
        K=8, H=8, KVH=2, D=4, Out=6, seed=15, attn_mask="nonempty"
    )
    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    node = _onnx_attention_node(pruned)
    q_num_heads, kv_num_heads = _onnx_attention_attrs(node)
    assert kv_num_heads == 1
    assert q_num_heads == 4

    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    # The rank-2 mask itself carries no per-head axis at all, so it must
    # come through completely unchanged even though the rest of the block
    # was pruned.
    np.testing.assert_array_equal(inits["AttnMask"], cfg["attn_mask"])
    assert inits["Wq"].shape == (cfg["K"], q_num_heads * cfg["D"])

    rng = np.random.default_rng(150)
    x = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(np.float32)
    (y_pruned,) = _run(pruned, {"X": x})
    assert y_pruned.shape == (cfg["batch"], cfg["seq"], cfg["Out"])


def test_onnx_attention_pruning_dynamic_attn_mask_input_is_still_pruned():
    # The dynamic-input counterpart of the test above: an ordinary graph
    # input standing in for real runtime mask data is left alone and must
    # not block the match.
    model, cfg = _onnx_attention_model(
        K=8, H=8, KVH=2, D=4, Out=6, seed=16, attn_mask="dynamic"
    )
    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    node = _onnx_attention_node(pruned)
    q_num_heads, kv_num_heads = _onnx_attention_attrs(node)
    assert kv_num_heads == 1
    assert q_num_heads == 4
    assert node.input[3] == "AttnMaskIn"

    rng = np.random.default_rng(17)
    x = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(np.float32)
    mask = np.zeros((cfg["seq"], cfg["seq"]), dtype=np.float32)
    (y,) = _run(pruned, {"X": x, "AttnMaskIn": mask})
    assert y.shape == (cfg["batch"], cfg["seq"], cfg["Out"])


def test_onnx_attention_pruning_rank4_attn_mask_head_axis_is_sliced_and_matches_oracle():
    # A rank-4 `(1, q_num_heads, seq, seq)` constant `attn_mask` genuinely
    # varies per query head (the op's own doc names this shape explicitly;
    # `onnx.reference.ops.op_attention` adds it against the
    # `(batch, q_num_heads, q_seq, kv_seq)` attention-score tensor via a
    # plain ``+``, i.e. ordinary broadcasting) -- this is the real gap this
    # matcher used to have: an earlier version declined the whole chain
    # outright rather than slice it, and *before that check existed at
    # all* (this op's own `attention_bias`-analogue -- see the
    # `com.microsoft::Attention`/`GroupQueryAttention` sections above) would
    # have silently left a now-wrong-head-count mask connected to a
    # pruned-head-count node. This test would fail against either of those:
    # a stale full-width mask either blocks the match (wrong shape check)
    # or raises a broadcast shape error at `onnxruntime.InferenceSession`
    # run time (neither `q_num_heads` nor 1), not just a numeric mismatch.
    rng = np.random.default_rng(18)
    H, KVH, D, seq = 8, 2, 4, 5
    mask = (rng.standard_normal((1, H, seq, seq)) * 1000.0).astype(np.float32)
    model, cfg = _onnx_attention_model(
        K=8,
        H=H,
        KVH=KVH,
        D=D,
        Out=6,
        seed=18,
        attn_mask="nonempty",
        attn_mask_array=mask,
    )
    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
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

    # The mask actually connected to the pruned model must be exactly this
    # slice -- checked directly, not just indirectly through numerics.
    pruned_inits = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer
    }
    np.testing.assert_array_equal(pruned_inits["AttnMask"], mask[:, keep_q_heads])

    oracle, _ = _onnx_attention_model(
        K=cfg["K"],
        H=len(keep_q_heads),
        KVH=len(keep_groups),
        D=d,
        Out=cfg["Out"],
        seed=18,
        wq=cfg["wq"][:, q_idx],
        wk=cfg["wk"][:, kv_idx],
        wv=cfg["wv"][:, kv_idx],
        wout=cfg["wout"][q_idx, :],
        batch=cfg["batch"],
        seq=cfg["seq"],
        attn_mask="nonempty",
        attn_mask_array=mask[:, keep_q_heads],
    )

    rng = np.random.default_rng(19)
    x = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(np.float32)
    (y_pruned,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y_pruned, y_oracle, rtol=1e-4, atol=1e-4)


def test_onnx_attention_pruning_rank3_attn_mask_head_axis_is_sliced():
    # A rank-3 `(q_num_heads, seq, seq)` mask -- omitting the batch axis
    # entirely, relying on broadcast -- lands its own axis *0* on the
    # `q_num_heads` slot once right-aligned against the op's rank-4
    # `(batch, q_num_heads, q_seq, kv_seq)` attention-score tensor, not
    # axis 1 the way a rank-4 mask does (see `_head_bias_axis`'s own
    # docstring; confirmed directly against onnxruntime, not assumed from
    # the broadcasting rule alone). This is the case most likely to be
    # mis-handled by an implementation that only ever checks axis 1.
    rng = np.random.default_rng(20)
    H, KVH, D, seq = 8, 2, 4, 5
    mask = (rng.standard_normal((H, seq, seq)) * 1000.0).astype(np.float32)
    model, cfg = _onnx_attention_model(
        K=8,
        H=H,
        KVH=KVH,
        D=D,
        Out=6,
        seed=20,
        attn_mask="nonempty",
        attn_mask_array=mask,
    )
    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    node = _onnx_attention_node(pruned)
    q_num_heads, kv_num_heads = _onnx_attention_attrs(node)
    assert kv_num_heads == 1
    assert q_num_heads == 4
    group_size = cfg["H"] // cfg["KVH"]

    keep_groups = _oracle_keep_groups(
        cfg["wq"], cfg["wk"], cfg["wv"], cfg["H"], cfg["KVH"], cfg["D"], kv_num_heads
    )
    keep_q_heads = _group_q_heads(keep_groups, group_size)

    pruned_inits = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer
    }
    np.testing.assert_array_equal(pruned_inits["AttnMask"], mask[keep_q_heads])


def test_onnx_attention_pruning_broadcast_head_axis_attn_mask_is_left_untouched():
    # A rank-4 `(1, 1, seq, seq)` mask has a `q_num_heads`-aligned axis
    # (axis 1) present but size 1 -- an ordinary broadcast, no per-head
    # values at all, so it needs no slicing and must be left byte-identical
    # even though the rest of the block is pruned (unlike the rank4/rank3
    # tests above, whose masks *do* carry real per-head data).
    rng = np.random.default_rng(21)
    H, KVH, D, seq = 8, 2, 4, 5
    mask = (rng.standard_normal((1, 1, seq, seq)) * 1000.0).astype(np.float32)
    model, cfg = _onnx_attention_model(
        K=8,
        H=H,
        KVH=KVH,
        D=D,
        Out=6,
        seed=21,
        attn_mask="nonempty",
        attn_mask_array=mask,
    )
    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    node = _onnx_attention_node(pruned)
    q_num_heads, kv_num_heads = _onnx_attention_attrs(node)
    assert kv_num_heads == 1
    assert q_num_heads == 4

    pruned_inits = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer
    }
    np.testing.assert_array_equal(pruned_inits["AttnMask"], mask)


def test_onnx_attention_pruning_ambiguous_attn_mask_shape_is_declined():
    # A rank-4 mask whose second axis is neither `q_num_heads` (8) nor 1 --
    # here 3, an arbitrary value that doesn't cleanly resolve to either
    # "genuinely per-head" or "broadcast" against this op's own
    # broadcasting rule -- must decline the whole match rather than guess,
    # exactly like an unrecognized `past_key`/`past_value` shape already
    # does.
    rng = np.random.default_rng(22)
    H, KVH, D, seq = 8, 2, 4, 5
    mask = (rng.standard_normal((1, 3, seq, seq)) * 1000.0).astype(np.float32)
    model, cfg = _onnx_attention_model(
        K=8,
        H=H,
        KVH=KVH,
        D=D,
        Out=6,
        seed=22,
        attn_mask="nonempty",
        attn_mask_array=mask,
    )
    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)

    inits_before = {
        t.name: onnx.numpy_helper.to_array(t) for t in model.graph.initializer
    }
    inits_after = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer
    }
    for name in inits_before:
        np.testing.assert_array_equal(inits_before[name], inits_after[name])


def test_onnx_attention_pruning_missing_kv_num_heads_attribute_is_left_untouched():
    # Both `q_num_heads`/`kv_num_heads` are schema-*optional* (inferable
    # from a rank-4 Q/K input this pass never produces/matches -- see
    # `_match_onnx_attention_producer`'s own docstring), so a node giving
    # only one of the two isn't a topology this pass tracks and must be
    # left alone rather than guessed at.
    K, H, D, Out = 8, 4, 4, 6
    N = H * D
    rng = np.random.default_rng(18)
    wq = rng.standard_normal((K, N)).astype(np.float32)
    wk = rng.standard_normal((K, N)).astype(np.float32)
    wv = rng.standard_normal((K, N)).astype(np.float32)
    wout = rng.standard_normal((N, Out)).astype(np.float32)
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
          ctx = Attention <q_num_heads={H}> (q, k, v)
          Y = MatMul(ctx, Wout)
        }}
        """
    )
    model.graph.initializer.extend(
        [_f32(wq, "Wq"), _f32(wk, "Wk"), _f32(wv, "Wv"), _f32(wout, "Wout")]
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    inits_before = {
        t.name: onnx.numpy_helper.to_array(t) for t in model.graph.initializer
    }
    inits_after = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer
    }
    for name in inits_before:
        np.testing.assert_array_equal(inits_before[name], inits_after[name])


def test_onnx_attention_pruning_diff_v_head_size_matches_oracle_exactly():
    # This op's real schema (unlike `com.microsoft::GroupQueryAttention`,
    # which `fuse_gqa.h` always emits with equal Q/K/V head_size) genuinely
    # allows V its own, independent head_size -- confirmed via the op's own
    # backend-test suite (`test_attention_3d_diff_heads_sizes` and friends
    # in `onnx/backend/test/case/node/attention.py`) and via actual
    # onnxruntime execution above. `_GQAChain` now carries Q's/K's shared
    # `head_size` and V's own (possibly different) `v_head_size` as separate
    # fields, and `_apply_one_gqa_chain` slices Q's/K's own producer weight
    # at `head_size` while V's own producer weight -- and the output
    # projection's own reduction dim, and the raw output's own width -- at
    # `v_head_size` (see this module's own "Attention-head pruning" section
    # comment). Each KV group's own Q+K+V block is scaled by a distinct,
    # well-separated factor (not left to natural per-head random variance)
    # so which 2 of 4 groups the importance ranking keeps is unambiguous,
    # genuinely exercising the ranking logic rather than merely running
    # without error.
    K, H, KVH, D, Dv, Out = 8, 8, 4, 4, 6, 5
    group_size = H // KVH
    rng = np.random.default_rng(19)
    wq = rng.standard_normal((K, H * D)).astype(np.float32)
    wk = rng.standard_normal((K, KVH * D)).astype(np.float32)
    wv = rng.standard_normal((K, KVH * Dv)).astype(np.float32)
    wout = rng.standard_normal((H * Dv, Out)).astype(np.float32)

    # Group 0 and 2 are scaled far above group 1 and 3 -- the top-2 by
    # importance (kv_num_heads == 2 at sparsity=0.5) are therefore exactly
    # {0, 2}, regardless of the underlying random weights' own natural norm
    # variance (a 20x-60x separation swamps it).
    scales = [3.0, 0.1, 2.0, 0.05]
    for kv, scale in enumerate(scales):
        for h in range(kv * group_size, (kv + 1) * group_size):
            wq[:, h * D : (h + 1) * D] *= scale
        wk[:, kv * D : (kv + 1) * D] *= scale
        wv[:, kv * Dv : (kv + 1) * Dv] *= scale

    model, cfg = _onnx_attention_model(
        K=K,
        H=H,
        KVH=KVH,
        D=D,
        Dv=Dv,
        Out=Out,
        seed=19,
        wq=wq,
        wk=wk,
        wv=wv,
        wout=wout,
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    node = _onnx_attention_node(pruned)
    q_num_heads, kv_num_heads = _onnx_attention_attrs(node)
    assert kv_num_heads == 2  # max(1, 4 - round(4*0.5))
    assert q_num_heads == kv_num_heads * group_size

    keep_groups = _oracle_keep_groups(
        wq, wk, wv, H, KVH, D, kv_num_heads, v_head_size=Dv
    )
    assert list(keep_groups) == [0, 2]

    keep_q_heads = _group_q_heads(keep_groups, group_size)
    q_idx = _head_idx(keep_q_heads, D)  # Q's own producer weight columns
    kv_idx = _head_idx(keep_groups, D)  # K's own producer weight columns
    v_idx = _head_idx(keep_groups, Dv)  # V's own producer weight columns
    y_idx = _head_idx(keep_q_heads, Dv)  # output/consumer-side columns

    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["Wq"], wq[:, q_idx])
    np.testing.assert_array_equal(inits["Wk"], wk[:, kv_idx])
    np.testing.assert_array_equal(inits["Wv"], wv[:, v_idx])
    np.testing.assert_array_equal(inits["Wout"], wout[y_idx, :])

    oracle, _ = _onnx_attention_model(
        K=K,
        H=q_num_heads,
        KVH=kv_num_heads,
        D=D,
        Dv=Dv,
        Out=Out,
        seed=19,
        wq=wq[:, q_idx],
        wk=wk[:, kv_idx],
        wv=wv[:, v_idx],
        wout=wout[y_idx, :],
        batch=cfg["batch"],
        seq=cfg["seq"],
    )
    onnx.checker.check_model(oracle)

    rng2 = np.random.default_rng(22)
    x = rng2.standard_normal((cfg["batch"], cfg["seq"], K)).astype(np.float32)
    (y_pruned,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y_pruned, y_oracle, rtol=1e-4, atol=1e-4)


def test_onnx_attention_wanda_pruning_matches_oracle_exactly():
    model, cfg = _onnx_attention_model(K=8, H=8, KVH=2, D=4, Out=6, seed=8)

    rng = np.random.default_rng(9)
    x_cal = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(np.float32)
    calibration_data = [{"X": x_cal}]

    pruned = onnxsim.apply_attention_head_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned)

    probe_model = onnx.ModelProto()
    probe_model.CopyFrom(model)
    probe_model.graph.output.append(onnx.ValueInfoProto(name="ctx"))
    (_, ctx_cal) = _run(probe_model, {"X": x_cal})
    act_norm = np.sqrt(np.mean(np.square(ctx_cal.astype(np.float64)), axis=(0, 1)))

    d = cfg["D"]
    group_size = cfg["H"] // cfg["KVH"]
    importance = np.zeros(cfg["KVH"])
    for kv in range(cfg["KVH"]):
        q_block = np.concatenate(
            [
                cfg["wq"][:, h * d : (h + 1) * d]
                for h in range(kv * group_size, (kv + 1) * group_size)
            ],
            axis=1,
        )
        k_block = cfg["wk"][:, kv * d : (kv + 1) * d]
        v_block = cfg["wv"][:, kv * d : (kv + 1) * d]
        base = np.linalg.norm(np.concatenate([q_block, k_block, v_block], axis=1))
        act_group = np.linalg.norm(
            act_norm[kv * group_size * d : (kv + 1) * group_size * d]
        )
        importance[kv] = base * max(act_group, 1e-8)
    keep_groups = np.sort(np.argsort(-importance)[:1])  # max(1, 2 - round(2*0.5)) == 1

    keep_q_heads = _group_q_heads(keep_groups, group_size)
    q_idx, kv_idx = _head_idx(keep_q_heads, d), _head_idx(keep_groups, d)

    oracle, _ = _onnx_attention_model(
        K=cfg["K"],
        H=len(keep_q_heads),
        KVH=len(keep_groups),
        D=d,
        Out=cfg["Out"],
        seed=8,
        wq=cfg["wq"][:, q_idx],
        wk=cfg["wk"][:, kv_idx],
        wv=cfg["wv"][:, kv_idx],
        wout=cfg["wout"][q_idx, :],
        batch=cfg["batch"],
        seq=cfg["seq"],
    )

    x = rng.standard_normal((cfg["batch"], cfg["seq"], cfg["K"])).astype(np.float32)
    (y_pruned,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y_pruned, y_oracle, rtol=1e-4, atol=1e-4)


def test_onnx_attention_wanda_pruning_diff_v_head_size_matches_oracle_exactly():
    # The Wanda-calibrated counterpart of
    # `test_onnx_attention_pruning_diff_v_head_size_matches_oracle_exactly`:
    # `_wanda_gqa_group_importance`'s own activation probe sits on the
    # consumer's input (the attention output), laid out per query head at
    # V's own `v_head_size` -- not Q's/K's `head_size` -- so both its own
    # `width` check and its per-group activation window must stride by
    # `v_head_size`, exactly like the weight-only path's `y_idx` does.
    K, H, KVH, D, Dv, Out = 8, 8, 2, 4, 6, 5
    group_size = H // KVH
    model, cfg = _onnx_attention_model(K=K, H=H, KVH=KVH, D=D, Dv=Dv, Out=Out, seed=23)

    rng = np.random.default_rng(24)
    x_cal = rng.standard_normal((cfg["batch"], cfg["seq"], K)).astype(np.float32)
    calibration_data = [{"X": x_cal}]

    pruned = onnxsim.apply_attention_head_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned)

    probe_model = onnx.ModelProto()
    probe_model.CopyFrom(model)
    probe_model.graph.output.append(onnx.ValueInfoProto(name="ctx"))
    (_, ctx_cal) = _run(probe_model, {"X": x_cal})
    act_norm = np.sqrt(np.mean(np.square(ctx_cal.astype(np.float64)), axis=(0, 1)))
    assert act_norm.shape == (H * Dv,)  # sanity: laid out per Q head at Dv, not D

    importance = np.zeros(KVH)
    for kv in range(KVH):
        q_block = np.concatenate(
            [
                cfg["wq"][:, h * D : (h + 1) * D]
                for h in range(kv * group_size, (kv + 1) * group_size)
            ],
            axis=1,
        )
        k_block = cfg["wk"][:, kv * D : (kv + 1) * D]
        v_block = cfg["wv"][:, kv * Dv : (kv + 1) * Dv]
        base = np.linalg.norm(np.concatenate([q_block, k_block, v_block], axis=1))
        act_group = np.linalg.norm(
            act_norm[kv * group_size * Dv : (kv + 1) * group_size * Dv]
        )
        importance[kv] = base * max(act_group, 1e-8)
    keep_groups = np.sort(np.argsort(-importance)[:1])  # max(1, 2 - round(2*0.5))

    keep_q_heads = _group_q_heads(keep_groups, group_size)
    q_idx = _head_idx(keep_q_heads, D)
    kv_idx = _head_idx(keep_groups, D)
    v_idx = _head_idx(keep_groups, Dv)
    y_idx = _head_idx(keep_q_heads, Dv)

    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["Wq"], cfg["wq"][:, q_idx])
    np.testing.assert_array_equal(inits["Wk"], cfg["wk"][:, kv_idx])
    np.testing.assert_array_equal(inits["Wv"], cfg["wv"][:, v_idx])
    np.testing.assert_array_equal(inits["Wout"], cfg["wout"][y_idx, :])

    oracle, _ = _onnx_attention_model(
        K=K,
        H=len(keep_q_heads),
        KVH=len(keep_groups),
        D=D,
        Dv=Dv,
        Out=Out,
        seed=23,
        wq=cfg["wq"][:, q_idx],
        wk=cfg["wk"][:, kv_idx],
        wv=cfg["wv"][:, v_idx],
        wout=cfg["wout"][y_idx, :],
        batch=cfg["batch"],
        seq=cfg["seq"],
    )
    onnx.checker.check_model(oracle)

    x = rng.standard_normal((cfg["batch"], cfg["seq"], K)).astype(np.float32)
    (y_pruned,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y_pruned, y_oracle, rtol=1e-4, atol=1e-4)


def test_onnx_attention_wanda_pruning_falls_back_to_plain_with_no_calibration_batches():
    model, cfg = _onnx_attention_model(K=8, H=8, KVH=2, D=4, Out=6, seed=10)
    plain = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    wanda = onnxsim.apply_attention_head_wanda_pruning(
        model, calibration_data=[], sparsity=0.5
    )
    inits_plain = {
        t.name: onnx.numpy_helper.to_array(t) for t in plain.graph.initializer
    }
    inits_wanda = {
        t.name: onnx.numpy_helper.to_array(t) for t in wanda.graph.initializer
    }
    for name in inits_plain:
        np.testing.assert_array_equal(inits_plain[name], inits_wanda[name])


# --- apply_attention_head_pruning / _wanda_pruning -- plain ai.onnx
# Attention, cross-attention (Q and K/V from genuinely different source
# tensors, at genuinely different sequence lengths) ------------------------
#
# Unlike GroupQueryAttention's own cross-attention section above (equal
# sequence length only, an onnxruntime CPU-kernel restriction on that op --
# see onnxsim/pruning.py's own "Attention-head pruning" section comment),
# this op's own onnxruntime kernel has no such restriction (confirmed
# empirically): `seq_q` and `seq_kv` genuinely differ below, exercising this
# op's own schema doc ("For cross attention, query and key might have
# different lengths") at full strength, oracle-verified via onnxruntime like
# every other function in this module.


def _onnx_attention_cross_model(
    K_dec=8,
    K_enc=6,
    H=4,
    KVH=2,
    D=4,
    Out=6,
    seed=0,
    batch=2,
    seq_q=5,
    seq_kv=7,
    wq=None,
    wk=None,
    wv=None,
    wout=None,
):
    rng = np.random.default_rng(seed)
    Nq, Nkv = H * D, KVH * D
    if wq is None:
        wq = rng.standard_normal((K_dec, Nq)).astype(np.float32)
    if wk is None:
        wk = rng.standard_normal((K_enc, Nkv)).astype(np.float32)
    if wv is None:
        wv = rng.standard_normal((K_enc, Nkv)).astype(np.float32)
    if wout is None:
        wout = rng.standard_normal((Nq, Out)).astype(np.float32)

    initializer = [_f32(wq, "Wq"), _f32(wk, "Wk"), _f32(wv, "Wv"), _f32(wout, "Wout")]

    body = f"""
        g (float[{batch},{seq_q},{K_dec}] Xdec, float[{batch},{seq_kv},{K_enc}] Xenc) => (float[{batch},{seq_q},{Out}] Y)
        {{
          q = MatMul(Xdec, Wq)
          k = MatMul(Xenc, Wk)
          v = MatMul(Xenc, Wv)
          ctx = Attention <q_num_heads={H}, kv_num_heads={KVH}> (q, k, v)
          Y = MatMul(ctx, Wout)
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
        K_dec=K_dec,
        K_enc=K_enc,
        H=H,
        KVH=KVH,
        D=D,
        Out=Out,
        Nq=Nq,
        Nkv=Nkv,
        wq=wq,
        wk=wk,
        wv=wv,
        wout=wout,
        batch=batch,
        seq_q=seq_q,
        seq_kv=seq_kv,
    )


def test_onnx_attention_pruning_cross_attention_matches_oracle_exactly():
    # Without the `_gqa_group_importance` fix this shares with the
    # GroupQueryAttention cross-attention test above, this raises a bare
    # numpy ValueError instead of reaching the assertions below --
    # K_dec=8 != K_enc=6 here is deliberate, and (unlike that test)
    # seq_q=5 != seq_kv=7 too, since this op's own kernel allows it.
    model, cfg = _onnx_attention_cross_model(
        K_dec=8, K_enc=6, H=8, KVH=2, D=4, Out=6, seed=24
    )
    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    node = _onnx_attention_node(pruned)
    q_num_heads, kv_num_heads = _onnx_attention_attrs(node)
    group_size = cfg["H"] // cfg["KVH"]
    assert kv_num_heads == 1  # max(1, 2 - round(2*0.5))
    assert q_num_heads == kv_num_heads * group_size

    keep_groups = _oracle_keep_groups_cross(
        cfg["wq"], cfg["wk"], cfg["wv"], cfg["H"], cfg["KVH"], cfg["D"], kv_num_heads
    )
    keep_q_heads = _group_q_heads(keep_groups, group_size)
    d = cfg["D"]
    q_idx, kv_idx = _head_idx(keep_q_heads, d), _head_idx(keep_groups, d)

    oracle, _ = _onnx_attention_cross_model(
        K_dec=cfg["K_dec"],
        K_enc=cfg["K_enc"],
        H=q_num_heads,
        KVH=kv_num_heads,
        D=d,
        Out=cfg["Out"],
        seed=24,
        wq=cfg["wq"][:, q_idx],
        wk=cfg["wk"][:, kv_idx],
        wv=cfg["wv"][:, kv_idx],
        wout=cfg["wout"][q_idx, :],
        batch=cfg["batch"],
        seq_q=cfg["seq_q"],
        seq_kv=cfg["seq_kv"],
    )

    rng = np.random.default_rng(25)
    xdec = rng.standard_normal((cfg["batch"], cfg["seq_q"], cfg["K_dec"])).astype(
        np.float32
    )
    xenc = rng.standard_normal((cfg["batch"], cfg["seq_kv"], cfg["K_enc"])).astype(
        np.float32
    )
    (y_pruned,) = _run(pruned, {"Xdec": xdec, "Xenc": xenc})
    (y_oracle,) = _run(oracle, {"Xdec": xdec, "Xenc": xenc})
    np.testing.assert_allclose(y_pruned, y_oracle, rtol=1e-4, atol=1e-4)

    # Sanity: Q and K/V really are independently sourced, at genuinely
    # different sequence lengths -- perturbing Xenc alone (Xdec held fixed)
    # must still change the output.
    (y_pruned2,) = _run(pruned, {"Xdec": xdec, "Xenc": xenc + 1.0})
    assert not np.allclose(y_pruned, y_pruned2)


def test_onnx_attention_wanda_pruning_cross_attention_matches_oracle_exactly():
    model, cfg = _onnx_attention_cross_model(
        K_dec=8, K_enc=6, H=8, KVH=2, D=4, Out=6, seed=26
    )

    rng = np.random.default_rng(27)
    xdec_cal = rng.standard_normal((cfg["batch"], cfg["seq_q"], cfg["K_dec"])).astype(
        np.float32
    )
    xenc_cal = rng.standard_normal((cfg["batch"], cfg["seq_kv"], cfg["K_enc"])).astype(
        np.float32
    )
    calibration_data = [{"Xdec": xdec_cal, "Xenc": xenc_cal}]

    pruned = onnxsim.apply_attention_head_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned)

    probe_model = onnx.ModelProto()
    probe_model.CopyFrom(model)
    probe_model.graph.output.append(onnx.ValueInfoProto(name="ctx"))
    (_, ctx_cal) = _run(probe_model, {"Xdec": xdec_cal, "Xenc": xenc_cal})
    act_norm = np.sqrt(np.mean(np.square(ctx_cal.astype(np.float64)), axis=(0, 1)))

    d = cfg["D"]
    group_size = cfg["H"] // cfg["KVH"]
    importance = np.zeros(cfg["KVH"])
    for kv in range(cfg["KVH"]):
        q_block = np.concatenate(
            [
                cfg["wq"][:, h * d : (h + 1) * d]
                for h in range(kv * group_size, (kv + 1) * group_size)
            ],
            axis=1,
        )
        k_block = cfg["wk"][:, kv * d : (kv + 1) * d]
        v_block = cfg["wv"][:, kv * d : (kv + 1) * d]
        base = np.sqrt(
            np.linalg.norm(q_block) ** 2
            + np.linalg.norm(k_block) ** 2
            + np.linalg.norm(v_block) ** 2
        )
        act_group = np.linalg.norm(
            act_norm[kv * group_size * d : (kv + 1) * group_size * d]
        )
        importance[kv] = base * max(act_group, 1e-8)
    keep_groups = np.sort(np.argsort(-importance)[:1])  # max(1, 2 - round(2*0.5)) == 1

    keep_q_heads = _group_q_heads(keep_groups, group_size)
    q_idx, kv_idx = _head_idx(keep_q_heads, d), _head_idx(keep_groups, d)

    oracle, _ = _onnx_attention_cross_model(
        K_dec=cfg["K_dec"],
        K_enc=cfg["K_enc"],
        H=len(keep_q_heads),
        KVH=len(keep_groups),
        D=d,
        Out=cfg["Out"],
        seed=26,
        wq=cfg["wq"][:, q_idx],
        wk=cfg["wk"][:, kv_idx],
        wv=cfg["wv"][:, kv_idx],
        wout=cfg["wout"][q_idx, :],
        batch=cfg["batch"],
        seq_q=cfg["seq_q"],
        seq_kv=cfg["seq_kv"],
    )

    xdec = rng.standard_normal((cfg["batch"], cfg["seq_q"], cfg["K_dec"])).astype(
        np.float32
    )
    xenc = rng.standard_normal((cfg["batch"], cfg["seq_kv"], cfg["K_enc"])).astype(
        np.float32
    )
    (y_pruned,) = _run(pruned, {"Xdec": xdec, "Xenc": xenc})
    (y_oracle,) = _run(oracle, {"Xdec": xdec, "Xenc": xenc})
    np.testing.assert_allclose(y_pruned, y_oracle, rtol=1e-4, atol=1e-4)


def test_attention_head_pruning_handles_all_three_attention_op_types_in_one_model():
    # Regression check for `_apply_attention_chains`'s per-chain-type
    # dispatch once a third node type shares it: a plain
    # `com.microsoft::Attention` block, a `GroupQueryAttention` block, and a
    # plain `ai.onnx::Attention` block in the same graph, sharing no
    # tensors, must each be pruned correctly and independently -- no chain
    # family may disturb another, extending
    # `test_attention_head_pruning_handles_attention_and_gqa_in_one_model`'s
    # own two-type check to all three now-matched op types.
    K, H, D, Out1 = 8, 4, 4, 6
    Nqkv = H * D
    rng = np.random.default_rng(40)
    wqkv = rng.standard_normal((K, 3 * Nqkv)).astype(np.float32)
    bqkv = rng.standard_normal((3 * Nqkv,)).astype(np.float32)
    wout1 = rng.standard_normal((Nqkv, Out1)).astype(np.float32)

    GH, GKVH, GD, Out2 = 8, 2, 8, 5
    Nq2, Nkv2 = GH * GD, GKVH * GD
    wq2 = rng.standard_normal((K, Nq2)).astype(np.float32)
    wk2 = rng.standard_normal((K, Nkv2)).astype(np.float32)
    wv2 = rng.standard_normal((K, Nkv2)).astype(np.float32)
    wout2 = rng.standard_normal((Nq2, Out2)).astype(np.float32)

    AH, AKVH, AD, Out3 = 8, 2, 4, 7
    Nq3, Nkv3 = AH * AD, AKVH * AD
    wq3 = rng.standard_normal((K, Nq3)).astype(np.float32)
    wk3 = rng.standard_normal((K, Nkv3)).astype(np.float32)
    wv3 = rng.standard_normal((K, Nkv3)).astype(np.float32)
    wout3 = rng.standard_normal((Nq3, Out3)).astype(np.float32)

    batch, seq = 2, 5
    seqlens_k = np.full((batch,), seq - 1, dtype=np.int32)
    total_seq = np.array(seq, dtype=np.int32)

    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 24, "com.microsoft": 1]
        >
        g (float[{batch},{seq},{K}] X1, float[{batch},{seq},{K}] X2, float[{batch},{seq},{K}] X3) => (float[{batch},{seq},{Out1}] Y1, float[{batch},{seq},{Out2}] Y2, float[{batch},{seq},{Out3}] Y3)
        {{
          ctx1 = com.microsoft.Attention <num_heads={H}, qkv_hidden_sizes=[{Nqkv},{Nqkv},{Nqkv}]> (X1, Wqkv, Bqkv)
          Y1 = MatMul(ctx1, Wout1)
          q2 = MatMul(X2, Wq2)
          k2 = MatMul(X2, Wk2)
          v2 = MatMul(X2, Wv2)
          ctx2, pk, pv = com.microsoft.GroupQueryAttention <num_heads={GH}, kv_num_heads={GKVH}> (q2, k2, v2, , , SeqLensK, TotalSeq)
          Y2 = MatMul(ctx2, Wout2)
          q3 = MatMul(X3, Wq3)
          k3 = MatMul(X3, Wk3)
          v3 = MatMul(X3, Wv3)
          ctx3 = Attention <q_num_heads={AH}, kv_num_heads={AKVH}> (q3, k3, v3)
          Y3 = MatMul(ctx3, Wout3)
        }}
        """
    )
    model.graph.initializer.extend(
        [
            _f32(wqkv, "Wqkv"),
            _f32(bqkv, "Bqkv"),
            _f32(wout1, "Wout1"),
            _f32(wq2, "Wq2"),
            _f32(wk2, "Wk2"),
            _f32(wv2, "Wv2"),
            _f32(wout2, "Wout2"),
            onnx.numpy_helper.from_array(seqlens_k, "SeqLensK"),
            onnx.numpy_helper.from_array(total_seq, "TotalSeq"),
            _f32(wq3, "Wq3"),
            _f32(wk3, "Wk3"),
            _f32(wv3, "Wv3"),
            _f32(wout3, "Wout3"),
        ]
    )

    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    attn_node = next(
        n
        for n in pruned.graph.node
        if n.op_type == "Attention" and n.domain == "com.microsoft"
    )
    gqa_node = next(n for n in pruned.graph.node if n.op_type == "GroupQueryAttention")
    onnx_attn_node = _onnx_attention_node(pruned)
    attn_heads, _ = _attention_attrs(attn_node)
    gqa_heads, gqa_kv_heads = _gqa_attrs(gqa_node)
    onnx_attn_heads, onnx_attn_kv_heads = _onnx_attention_attrs(onnx_attn_node)
    assert attn_heads == 2
    assert gqa_kv_heads == 1
    assert gqa_heads == 4
    assert onnx_attn_kv_heads == 1
    assert onnx_attn_heads == 4

    rng2 = np.random.default_rng(41)
    x1 = rng2.standard_normal((batch, seq, K)).astype(np.float32)
    x2 = rng2.standard_normal((batch, seq, K)).astype(np.float32)
    x3 = rng2.standard_normal((batch, seq, K)).astype(np.float32)
    y1, y2, y3 = _run(pruned, {"X1": x1, "X2": x2, "X3": x3})
    assert y1.shape == (batch, seq, Out1)
    assert y2.shape == (batch, seq, Out2)
    assert y3.shape == (batch, seq, Out3)


# --- importance_norm ("l1" vs "l2") ---------------------------------------
#
# Every importance ranking above defaults to Li et al.'s L2-norm criterion,
# unchanged from this module's own behavior before `importance_norm`
# existed at all -- see the explicit "l2 is the unchanged default" checks
# below (byte-identical serialized output with vs. without the parameter),
# plus the entire pre-existing suite above this section, which this change
# leaves passing untouched. What follows instead targets the genuinely new
# "l1" path: adversarial weight layouts where a channel/head/group with a
# few large entries (high L2, lower total L1) and one with many medium
# entries (lower L2, higher total L1) trade rank depending on which norm is
# asked for -- so a bug that silently keeps computing L2 under the hood
# even when "l1" is requested shows up as the *wrong unit surviving*, not
# merely a slightly-different score that happens to keep the same one.


def test_structured_pruning_l1_norm_favors_total_magnitude_single_producer():
    # Column "concentrated": one entry of magnitude 8, the other 15 entries
    # exactly zero -- L2 == L1 == 8 (a single nonzero entry has no L1/L2 gap
    # at all). Column "spread": all 16 entries equal to 1 -- L2 = sqrt(16) =
    # 4, L1 = 16. So L2 ranks "concentrated" (8) above "spread" (4), while
    # L1 ranks "spread" (16) above "concentrated" (8) -- a genuine
    # disagreement a correct L1 implementation must reproduce. A
    # "filler_high"/"filler_low" pair (dominant/negligible under either
    # norm) pins the other surviving slot so the test turns on only this
    # one comparison.
    K, H, Out = 16, 4, 3
    w1 = np.zeros((K, H), dtype=np.float32)
    w1[0, 0] = 8.0  # "concentrated"
    w1[:, 1] = 1.0  # "spread"
    w1[2, 2] = 1000.0  # "filler_high"
    w1[3, 3] = 0.001  # "filler_low"
    rng = np.random.default_rng(90)
    w2 = rng.standard_normal((H, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          h = MatMul(X, W1)
          a = Relu(h)
          Y = MatMul(a, W2)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2")],
    )

    pruned_l2 = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    pruned_l1 = onnxsim.apply_structured_pruning(
        model, sparsity=0.5, importance_norm="l1"
    )
    onnx.checker.check_model(pruned_l2)
    onnx.checker.check_model(pruned_l1)

    keep_l2 = np.array([0, 2])  # concentrated, filler_high
    keep_l1 = np.array([1, 2])  # spread, filler_high
    np.testing.assert_array_equal(_kept_columns(pruned_l2, "W1", w1), keep_l2)
    np.testing.assert_array_equal(_kept_columns(pruned_l1, "W1", w1), keep_l1)

    rng2 = np.random.default_rng(91)
    x = rng2.standard_normal((5, K)).astype(np.float32)
    (y_l2,) = _run(pruned_l2, {"X": x})
    (y_l1,) = _run(pruned_l1, {"X": x})

    def _oracle(keep):
        h = np.maximum(x @ w1[:, keep], 0)
        return h @ w2[keep, :]

    np.testing.assert_allclose(y_l2, _oracle(keep_l2), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(y_l1, _oracle(keep_l1), rtol=1e-5, atol=1e-5)


def test_structured_pruning_l1_norm_favors_total_magnitude_conv_residual_merge():
    # Same L1-vs-L2 disagreement as the single-producer case above, but now
    # spanning *two* independent Conv producers merged by a residual Add --
    # exercising the multi-producer combination this module documents as a
    # plain *sum* of per-producer L1 norms for "l1" (never any square/sqrt),
    # rather than L2's own root-sum-square. Channel "A" concentrates its
    # magnitude entirely in producer f (vf=10, vs=0): combined L2 = 10,
    # combined L1 = 10 (a single nonzero producer has no L1/L2 gap either,
    # same as the single-entry-column case above). Channel "B" splits evenly
    # across both producers (vf=vs=7): combined L2 = sqrt(98) ~= 9.9,
    # combined L1 = 14. So L2 (10 > 9.9) keeps "A", while L1 (14 > 10) keeps
    # "B" -- exactly the flip a still-secretly-root-sum-square "l1"
    # implementation would fail to reproduce.
    Cin, C, Cout = 1, 4, 3
    w_f = np.zeros((C, Cin, 3, 3), dtype=np.float32)
    w_s = np.zeros((C, Cin, 3, 3), dtype=np.float32)
    # index 0: "A" (concentrated in f), 1: "B" (split evenly f/s), 2:
    # filler_high (dominates either norm), 3: filler_low (negligible either
    # norm).
    w_f[0, 0, 0, 0] = 10.0
    w_f[1, 0, 0, 0] = 7.0
    w_f[2, 0, 0, 0] = 100.0
    w_f[3, 0, 0, 0] = 0.001
    w_s[1, 0, 0, 0] = 7.0
    w_s[2, 0, 0, 0] = 100.0
    w_s[3, 0, 0, 0] = 0.001
    rng = np.random.default_rng(92)
    w_out = rng.standard_normal((Cout, C, 3, 3)).astype(np.float32)
    model = _residual_diamond_model(w_f, w_s, w_out)

    pruned_l2 = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    pruned_l1 = onnxsim.apply_structured_pruning(
        model, sparsity=0.5, importance_norm="l1"
    )
    onnx.checker.check_model(pruned_l2)
    onnx.checker.check_model(pruned_l1)

    keep_l2 = np.array([0, 2])  # A, filler_high
    keep_l1 = np.array([1, 2])  # B, filler_high

    inits_l2 = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned_l2.graph.initializer
    }
    inits_l1 = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned_l1.graph.initializer
    }
    np.testing.assert_array_equal(inits_l2["WF"], w_f[keep_l2])
    np.testing.assert_array_equal(inits_l2["WS"], w_s[keep_l2])
    np.testing.assert_array_equal(inits_l1["WF"], w_f[keep_l1])
    np.testing.assert_array_equal(inits_l1["WS"], w_s[keep_l1])

    oracle_l2 = _residual_diamond_model(w_f[keep_l2], w_s[keep_l2], w_out[:, keep_l2])
    oracle_l1 = _residual_diamond_model(w_f[keep_l1], w_s[keep_l1], w_out[:, keep_l1])

    rng_x = np.random.default_rng(93)
    x = rng_x.standard_normal((2, Cin, 10, 10)).astype(np.float32)
    (y_l2,) = _run(pruned_l2, {"X": x})
    (y_oracle_l2,) = _run(oracle_l2, {"X": x})
    (y_l1,) = _run(pruned_l1, {"X": x})
    (y_oracle_l1,) = _run(oracle_l1, {"X": x})
    np.testing.assert_allclose(y_l2, y_oracle_l2, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(y_l1, y_oracle_l1, rtol=1e-5, atol=1e-5)


def test_structured_wanda_pruning_l1_norm_uses_l1_weight_term_but_l2_activation_term():
    # Confirms two things at once: (1) importance_norm="l1" changes only the
    # *weight*-magnitude half of Wanda's ||W_row|| * ||X||_2 metric, not the
    # activation-norm half, which stays L2 regardless (per Wanda's own
    # definition, see this module's docstring); (2) the combination is a
    # plain product of that L1 weight term and the L2 activation term. As in
    # the plain-L1 test above, column "concentrated" (L2 == L1 == 8) times
    # its own activation norm (8, engineered via the calibration row below)
    # gives combined 64 under either norm choice. Column "spread" (L2 = 4,
    # L1 = 16) times its own activation norm (10) gives combined 40 under L2
    # (still less than "concentrated"'s 64 -- L2-Wanda keeps
    # "concentrated") but 160 under L1 (now more than "concentrated"'s 64 --
    # L1-Wanda keeps "spread" instead).
    K, H, Out = 16, 4, 3
    w1 = np.zeros((K, H), dtype=np.float32)
    w1[0, 0] = 8.0  # "concentrated"
    w1[:, 1] = 1.0  # "spread"
    w1[2, 2] = 1000.0  # "filler_high"
    w1[3, 3] = 0.001  # "filler_low"
    rng = np.random.default_rng(94)
    w2 = rng.standard_normal((H, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          h = MatMul(X, W1)
          a = Relu(h)
          Y = MatMul(a, W2)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2")],
    )

    # A single calibration row, engineered so the resulting activation norm
    # (computed identically regardless of `importance_norm` -- rms of `a`
    # over calibration samples) comes out to 8/10/1000/~0 for the four
    # columns respectively: only row 0 feeds "concentrated" (act = 8*x0),
    # *every* row feeds "spread" (act = sum of all 16 entries), only row 2
    # feeds "filler_high", only row 3 feeds "filler_low".
    x = np.zeros((1, K), dtype=np.float32)
    x[0, 0] = 1.0  # concentrated: h = 8*1 = 8
    x[0, 1] = 8.0  # spread: h = sum(x) = 1(row0) + 8(row1) + 1(row2) = 10
    x[0, 2] = 1.0  # filler_high: h = 1000*1 = 1000
    x[0, 3] = 0.0  # filler_low: h = 0.001*0 = 0
    calibration_data = [{"X": x}]

    pruned_l2 = onnxsim.apply_structured_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    pruned_l1 = onnxsim.apply_structured_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.5, importance_norm="l1"
    )
    onnx.checker.check_model(pruned_l2)
    onnx.checker.check_model(pruned_l1)

    keep_l2 = np.array([0, 2])  # concentrated, filler_high
    keep_l1 = np.array([1, 2])  # spread, filler_high
    np.testing.assert_array_equal(_kept_columns(pruned_l2, "W1", w1), keep_l2)
    np.testing.assert_array_equal(_kept_columns(pruned_l1, "W1", w1), keep_l1)

    rng2 = np.random.default_rng(95)
    x_test = rng2.standard_normal((5, K)).astype(np.float32)
    (y_l2,) = _run(pruned_l2, {"X": x_test})
    (y_l1,) = _run(pruned_l1, {"X": x_test})

    def _oracle(keep):
        h = np.maximum(x_test @ w1[:, keep], 0)
        return h @ w2[keep, :]

    np.testing.assert_allclose(y_l2, _oracle(keep_l2), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(y_l1, _oracle(keep_l1), rtol=1e-5, atol=1e-5)


def test_attention_head_pruning_l1_norm_favors_total_magnitude():
    # The attention-head analogue of the single-producer test above: each
    # head's combined Q+K+V weight block is ranked by Frobenius (L2) norm by
    # default, by that same block's own entrywise abs-sum (L1) norm instead
    # under `importance_norm="l1"`. Q and K carry small fixed noise, equal
    # across every head (so it shifts every head's block norm by the same
    # amount either way -- both `sqrt(c + v^2)` vs. L2 and `c + |v|` vs. L1
    # are monotonic in `v` for a shared constant `c`, so it can't flip any
    # ranking, only avoids an all-zero QK^T block); only V's own per-head
    # column block carries the actual signal. Head "concentrated" (one
    # nonzero V entry, 16, within its own [K, D] block) has L2 == L1 == 16;
    # head "spread" (every entry of its own [K, D] block == 1, 64 entries)
    # has L2 = sqrt(64) = 8, L1 = 64. Two filler heads (single huge/tiny V
    # entries of their own) pin the other surviving slot the same way as the
    # plain-L1 test above.
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
    # A bias-free `com.microsoft::Attention` node crashes this environment's
    # onnxruntime CPU kernel outright (confirmed with plain random weights,
    # unrelated to anything about this test's own values) -- an all-zero
    # bias is mathematically a no-op and sidesteps that entirely.
    bqkv = np.zeros((Nq + Nk + Nv,), dtype=np.float32)

    model, cfg = _attention_model(
        K=K, H=H, D=D, Out=Out, seed=50, bias=True, wqkv=wqkv, bqkv=bqkv
    )

    pruned_l2 = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    pruned_l1 = onnxsim.apply_attention_head_pruning(
        model, sparsity=0.5, importance_norm="l1"
    )
    onnx.checker.check_model(pruned_l2)
    onnx.checker.check_model(pruned_l1)

    keep_l2 = np.array([0, 2])  # concentrated, filler_high
    keep_l1 = np.array([1, 2])  # spread, filler_high

    def _oracle_model(keep):
        qi, ki, vi = (
            _head_idx(keep, D),
            _head_idx(keep, D) + Nq,
            _head_idx(keep, D) + Nq + Nk,
        )
        idx = np.concatenate([qi, ki, vi])
        return _attention_model(
            K=K,
            H=2,
            D=D,
            Out=Out,
            seed=50,
            bias=True,
            wqkv=wqkv[:, idx],
            bqkv=bqkv[idx],
            wout=cfg["wout"][_head_idx(keep, D), :],
            num_heads=2,
        )[0]

    oracle_l2 = _oracle_model(keep_l2)
    oracle_l1 = _oracle_model(keep_l1)

    rng = np.random.default_rng(51)
    x = rng.standard_normal((2, 5, K)).astype(np.float32)
    (y_l2,) = _run(pruned_l2, {"X": x})
    (y_oracle_l2,) = _run(oracle_l2, {"X": x})
    (y_l1,) = _run(pruned_l1, {"X": x})
    (y_oracle_l1,) = _run(oracle_l1, {"X": x})
    np.testing.assert_allclose(y_l2, y_oracle_l2, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(y_l1, y_oracle_l1, rtol=1e-4, atol=1e-4)


def test_gqa_pruning_l1_norm_favors_total_magnitude():
    # The GQA analogue: each KV group's combined Q+K+V weight block is
    # ranked by Frobenius norm by default, by entrywise abs-sum (L1) norm
    # instead under `importance_norm="l1"`. Q and K stay entirely zero (see
    # the plain-Attention L1 test above for why that's still a valid,
    # NaN-free block) so only V carries weight: KV group 0's own V slice
    # ([K, D] = [8, 8], 64 entries) concentrates all its magnitude in a
    # single entry (16) -- Frobenius == L1 == 16, no gap, same reasoning as
    # every other single-nonzero-entry case in this file. KV group 1's own V
    # slice spreads magnitude 1 evenly across all 64 entries -- Frobenius =
    # sqrt(64) = 8, L1 = 64. With exactly two KV groups and keep_count = 1,
    # L2 (16 > 8) keeps group 0, L1 (64 > 16) keeps group 1 -- no filler
    # groups needed at all.
    K, H, KVH, D, Out = 8, 4, 2, 8, 3
    Nq, Nkv = H * D, KVH * D
    wq = np.zeros((K, Nq), dtype=np.float32)
    wk = np.zeros((K, Nkv), dtype=np.float32)
    wv = np.zeros((K, Nkv), dtype=np.float32)
    wv[0, 0] = 16.0  # KV group 0's own V slice (columns 0:D) -- concentrated
    wv[:, D : 2 * D] = 1.0  # KV group 1's own V slice (columns D:2D) -- spread

    model, cfg = _gqa_model(
        K=K, H=H, KVH=KVH, D=D, Out=Out, seed=60, wq=wq, wk=wk, wv=wv
    )

    pruned_l2 = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    pruned_l1 = onnxsim.apply_attention_head_pruning(
        model, sparsity=0.5, importance_norm="l1"
    )
    onnx.checker.check_model(pruned_l2)
    onnx.checker.check_model(pruned_l1)

    group_size = H // KVH  # 2 query heads per KV group

    def _oracle_model(keep_group):
        q_idx = np.arange(
            keep_group * group_size * D, (keep_group + 1) * group_size * D
        )
        kv_idx = np.arange(keep_group * D, (keep_group + 1) * D)
        return _gqa_model(
            K=K,
            H=group_size,
            KVH=1,
            D=D,
            Out=Out,
            seed=60,
            wq=wq[:, q_idx],
            wk=wk[:, kv_idx],
            wv=wv[:, kv_idx],
            wout=cfg["wout"][q_idx, :],
        )[0]

    oracle_l2 = _oracle_model(0)
    oracle_l1 = _oracle_model(1)

    node_l2 = _gqa_node(pruned_l2)
    node_l1 = _gqa_node(pruned_l1)
    assert _gqa_attrs(node_l2) == (group_size, 1)
    assert _gqa_attrs(node_l1) == (group_size, 1)

    rng = np.random.default_rng(61)
    x = rng.standard_normal((cfg["batch"], cfg["seq"], K)).astype(np.float32)
    (y_l2,) = _run(pruned_l2, {"X": x})
    (y_oracle_l2,) = _run(oracle_l2, {"X": x})
    (y_l1,) = _run(pruned_l1, {"X": x})
    (y_oracle_l1,) = _run(oracle_l1, {"X": x})
    np.testing.assert_allclose(y_l2, y_oracle_l2, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(y_l1, y_oracle_l1, rtol=1e-4, atol=1e-4)


# --- apply_attention_head_pruning / _wanda_pruning -- GroupQueryAttention,
# packed-QKV-then-Split --
#
# A single packed MatMul/Gemm projection feeding a `Split` whose three
# outputs are wired directly into `GroupQueryAttention`'s own three
# separate, still non-empty, query/key/value inputs -- confirmed a real
# onnxruntime-genai model-builder export shape (its fused Q/K-norm GQA
# path), not merely a hypothetical one -- see
# :func:`onnxsim.pruning._match_packed_qkv_split`'s own docstring for the
# exact topology and the export code path that produces it, and this
# module's "Attention-head pruning" section comment for how it differs from
# `GroupQueryAttention`'s own unrelated, still-declined, schema-level
# packed-`query`-input convention.


def _gqa_packed_model(
    K=8,
    H=4,
    KVH=2,
    D=8,
    Out=6,
    seed=0,
    batch=2,
    seq=5,
    bias=False,
    wqkv=None,
    bqkv=None,
    wout=None,
    split_sizes=None,
    split_axis=-1,
    split_outputs=("q", "k", "v"),
    gqa_inputs=None,
):
    # Mirrors `_gqa_model`'s own scaffolding (SeqLensK/TotalSeq bookkeeping
    # inputs; no past_kv/scale support -- not needed by any packed-QKV test
    # below) but replaces its three independent Wq/Wk/Wv MatMul/Gemm
    # producers with one packed MatMul/Gemm producer feeding a `Split`
    # node -- see this section's own comment above.
    rng = np.random.default_rng(seed)
    Nq, Nkv = H * D, KVH * D
    if wqkv is None:
        wqkv = rng.standard_normal((K, Nq + 2 * Nkv)).astype(np.float32)
    if wout is None:
        wout = rng.standard_normal((Nq, Out)).astype(np.float32)

    initializer = [_f32(wqkv, "Wqkv"), _f32(wout, "Wout")]
    qkv_op = "MatMul(X, Wqkv)"
    if bias:
        if bqkv is None:
            bqkv = rng.standard_normal((Nq + 2 * Nkv,)).astype(np.float32)
        initializer.append(_f32(bqkv, "Bqkv"))
        qkv_op = "Gemm(X, Wqkv, Bqkv)"

    if split_sizes is None:
        split_sizes = [Nq, Nkv, Nkv]
    initializer.append(
        onnx.numpy_helper.from_array(
            np.array(split_sizes, dtype=np.int64), "SplitSizes"
        )
    )

    # No past_kv connected here, so `total_seq == seq` -- the same no-cache
    # case `_gqa_model`'s own default (`past_kv=None`) synthesizes.
    initializer.append(
        onnx.numpy_helper.from_array(
            np.full((batch,), seq - 1, dtype=np.int32), "SeqLensK"
        )
    )
    initializer.append(
        onnx.numpy_helper.from_array(np.array(seq, dtype=np.int32), "TotalSeq")
    )

    if gqa_inputs is None:
        gqa_inputs = split_outputs
    split_out = ", ".join(split_outputs)
    gqa_q, gqa_k, gqa_v = gqa_inputs

    body = f"""
        g (float[{batch},{seq},{K}] X) => (float[{batch},{seq},{Out}] Y)
        {{
          qkv = {qkv_op}
          {split_out} = Split <axis = {split_axis}> (qkv, SplitSizes)
          ctx, pk, pv = com.microsoft.GroupQueryAttention <num_heads={H}, kv_num_heads={KVH}> ({gqa_q}, {gqa_k}, {gqa_v}, , , SeqLensK, TotalSeq)
          Y = MatMul(ctx, Wout)
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
        wqkv=wqkv,
        bqkv=bqkv,
        wout=wout,
        batch=batch,
        seq=seq,
    )


def test_gqa_packed_qkv_split_pruning_matches_oracle_exactly():
    # Q's/K's/V's "own weight" is really one shared packed tensor here,
    # sliced once by a combined index set, with the `Split`'s own
    # split-sizes constant shrunk to match -- verified both directly
    # against a hand-sliced expectation and end-to-end against a real
    # onnxruntime execution of an independently-built oracle model (the
    # ordinary three-separate-producer `_gqa_model`, the same ground truth
    # `test_gqa_pruning_matches_oracle_exactly` uses above).
    K, H, KVH, D, Out = 8, 8, 2, 8, 6
    model, cfg = _gqa_packed_model(K=K, H=H, KVH=KVH, D=D, Out=Out, seed=1)
    assert len(onnxsim.pruning._find_gqa_chains(model.graph)) == 1

    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    node = _gqa_node(pruned)
    num_heads, kv_num_heads = _gqa_attrs(node)
    group_size = H // KVH
    assert kv_num_heads == 1  # max(1, 2 - round(2*0.5))
    assert num_heads == kv_num_heads * group_size

    Nq, Nkv = cfg["Nq"], cfg["Nkv"]
    wqkv = cfg["wqkv"]
    wq, wk, wv = wqkv[:, :Nq], wqkv[:, Nq : Nq + Nkv], wqkv[:, Nq + Nkv :]

    keep_groups = _oracle_keep_groups(wq, wk, wv, H, KVH, D, kv_num_heads)
    keep_q_heads = _group_q_heads(keep_groups, group_size)
    q_idx, kv_idx = _head_idx(keep_q_heads, D), _head_idx(keep_groups, D)

    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    expected_wqkv = np.concatenate([wq[:, q_idx], wk[:, kv_idx], wv[:, kv_idx]], axis=1)
    np.testing.assert_array_equal(inits["Wqkv"], expected_wqkv)
    np.testing.assert_array_equal(
        inits["SplitSizes"],
        np.array([len(q_idx), len(kv_idx), len(kv_idx)], dtype=np.int64),
    )

    oracle, _ = _gqa_model(
        K=K,
        H=num_heads,
        KVH=kv_num_heads,
        D=D,
        Out=Out,
        seed=1,
        wq=wq[:, q_idx],
        wk=wk[:, kv_idx],
        wv=wv[:, kv_idx],
        wout=cfg["wout"][q_idx, :],
        batch=cfg["batch"],
        seq=cfg["seq"],
    )

    rng = np.random.default_rng(2)
    x = rng.standard_normal((cfg["batch"], cfg["seq"], K)).astype(np.float32)
    (y_pruned,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y_pruned, y_oracle, rtol=1e-4, atol=1e-4)


def test_gqa_packed_qkv_split_pruning_zero_sparsity_is_a_no_op():
    model, cfg = _gqa_packed_model(K=8, H=8, KVH=2, D=8, Out=6, seed=7)
    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.0)
    node = _gqa_node(pruned)
    num_heads, kv_num_heads = _gqa_attrs(node)
    assert num_heads == cfg["H"]
    assert kv_num_heads == cfg["KVH"]
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["Wqkv"], cfg["wqkv"])
    np.testing.assert_array_equal(
        inits["SplitSizes"],
        np.array([cfg["Nq"], cfg["Nkv"], cfg["Nkv"]], dtype=np.int64),
    )


def test_gqa_packed_qkv_split_pruning_slices_packed_bias():
    # Same reasoning as `test_gqa_pruning_slices_bias_when_producer_has_one`:
    # a bias-carrying Gemm producer -- here, the single packed Gemm feeding
    # the `Split` -- can't sit directly ahead of a rank-3 input in a graph
    # meant to actually run through onnxruntime (Gemm's own schema requires
    # a rank-2 `A`), so this exercises the packed-bias-slicing path
    # directly against the initializers instead.
    K, H, KVH, D, Out = 8, 4, 2, 8, 6
    model, cfg = _gqa_packed_model(K=K, H=H, KVH=KVH, D=D, Out=Out, seed=14, bias=True)
    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)

    node = _gqa_node(pruned)
    num_heads, kv_num_heads = _gqa_attrs(node)
    group_size = H // KVH
    assert kv_num_heads == 1
    assert num_heads == group_size

    Nq, Nkv = cfg["Nq"], cfg["Nkv"]
    wqkv, bqkv = cfg["wqkv"], cfg["bqkv"]
    wq, wk, wv = wqkv[:, :Nq], wqkv[:, Nq : Nq + Nkv], wqkv[:, Nq + Nkv :]
    bq, bk, bv = bqkv[:Nq], bqkv[Nq : Nq + Nkv], bqkv[Nq + Nkv :]

    keep_groups = _oracle_keep_groups(wq, wk, wv, H, KVH, D, kv_num_heads)
    keep_q_heads = _group_q_heads(keep_groups, group_size)
    q_idx, kv_idx = _head_idx(keep_q_heads, D), _head_idx(keep_groups, D)

    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    expected_wqkv = np.concatenate([wq[:, q_idx], wk[:, kv_idx], wv[:, kv_idx]], axis=1)
    expected_bqkv = np.concatenate([bq[q_idx], bk[kv_idx], bv[kv_idx]])
    np.testing.assert_array_equal(inits["Wqkv"], expected_wqkv)
    np.testing.assert_array_equal(inits["Bqkv"], expected_bqkv)
    np.testing.assert_array_equal(
        inits["SplitSizes"],
        np.array([len(q_idx), len(kv_idx), len(kv_idx)], dtype=np.int64),
    )


def test_gqa_wanda_packed_qkv_split_pruning_matches_oracle_exactly():
    K, H, KVH, D, Out = 8, 8, 2, 8, 6
    model, cfg = _gqa_packed_model(K=K, H=H, KVH=KVH, D=D, Out=Out, seed=8)

    rng = np.random.default_rng(9)
    x_cal = rng.standard_normal((cfg["batch"], cfg["seq"], K)).astype(np.float32)
    calibration_data = [{"X": x_cal}]

    pruned = onnxsim.apply_attention_head_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(pruned)

    probe_model = onnx.ModelProto()
    probe_model.CopyFrom(model)
    probe_model.graph.output.append(onnx.ValueInfoProto(name="ctx"))
    (_, ctx_cal) = _run(probe_model, {"X": x_cal})
    act_norm = np.sqrt(np.mean(np.square(ctx_cal.astype(np.float64)), axis=(0, 1)))

    Nq, Nkv = cfg["Nq"], cfg["Nkv"]
    wqkv = cfg["wqkv"]
    wq, wk, wv = wqkv[:, :Nq], wqkv[:, Nq : Nq + Nkv], wqkv[:, Nq + Nkv :]

    group_size = H // KVH
    importance = np.zeros(KVH)
    for kv in range(KVH):
        q_block = np.concatenate(
            [
                wq[:, h * D : (h + 1) * D]
                for h in range(kv * group_size, (kv + 1) * group_size)
            ],
            axis=1,
        )
        k_block = wk[:, kv * D : (kv + 1) * D]
        v_block = wv[:, kv * D : (kv + 1) * D]
        base = np.linalg.norm(np.concatenate([q_block, k_block, v_block], axis=1))
        act_group = np.linalg.norm(
            act_norm[kv * group_size * D : (kv + 1) * group_size * D]
        )
        importance[kv] = base * max(act_group, 1e-8)
    keep_groups = np.sort(np.argsort(-importance)[:1])  # max(1, 2 - round(2*0.5)) == 1

    keep_q_heads = _group_q_heads(keep_groups, group_size)
    q_idx, kv_idx = _head_idx(keep_q_heads, D), _head_idx(keep_groups, D)

    oracle, _ = _gqa_model(
        K=K,
        H=len(keep_q_heads),
        KVH=len(keep_groups),
        D=D,
        Out=Out,
        seed=8,
        wq=wq[:, q_idx],
        wk=wk[:, kv_idx],
        wv=wv[:, kv_idx],
        wout=cfg["wout"][q_idx, :],
        batch=cfg["batch"],
        seq=cfg["seq"],
    )

    x = rng.standard_normal((cfg["batch"], cfg["seq"], K)).astype(np.float32)
    (y_pruned,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    np.testing.assert_allclose(y_pruned, y_oracle, rtol=1e-4, atol=1e-4)


def test_gqa_packed_qkv_split_wrong_output_order_is_declined():
    # Adversarial: a `Split` whose own three OUTPUTS aren't literally in
    # Q-then-K-then-V order (here: K's own range comes out of the `Split`
    # first) is declined outright by `_match_packed_qkv_split`'s own
    # ``list(node.output) == [q_name, k_name, v_name]`` check, rather than
    # a naive offset-by-position implementation (assume output 0 is always
    # Q) mis-slicing K's own column range into Q's or vice versa.
    # `GroupQueryAttention` itself is fed `(q_out, k_out, v_out)` --
    # `Split`'s own *second* output as `query`, its *first* as `key` -- so
    # semantically this is a valid (if unusual) GQA graph with Q and K
    # simply swapped, not a malformed one; this pass just can't safely
    # rewrite it, and leaves it completely untouched rather than guessing.
    K, H, KVH, D, Out = 8, 4, 2, 8, 6
    model, cfg = _gqa_packed_model(
        K=K,
        H=H,
        KVH=KVH,
        D=D,
        Out=Out,
        seed=21,
        split_outputs=("k_out", "q_out", "v_out"),
        gqa_inputs=("q_out", "k_out", "v_out"),
    )

    assert onnxsim.pruning._find_gqa_chains(model.graph) == []

    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    assert pruned.SerializeToString() == model.SerializeToString()


def test_gqa_packed_qkv_split_mismatched_total_width_is_declined():
    # The `Split`'s own `SplitSizes` sums to one column short of the packed
    # projection's own real output width -- `_match_packed_qkv_split`'s own
    # ``n_channels != nq + nk + nv`` check declines this rather than
    # pruning against a split-sizes total that doesn't actually describe
    # the packed weight's own shape.
    K, H, KVH, D, Out = 8, 4, 2, 8, 6
    model, cfg = _gqa_packed_model(
        K=K,
        H=H,
        KVH=KVH,
        D=D,
        Out=Out,
        seed=23,
        split_sizes=[H * D, KVH * D, KVH * D - 1],
    )

    assert onnxsim.pruning._find_gqa_chains(model.graph) == []

    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    assert pruned.SerializeToString() == model.SerializeToString()


def test_gqa_native_packed_query_convention_still_declined():
    # `GroupQueryAttention`'s own SCHEMA (not this pass) has its own,
    # different packed-input convention: the whole packed Q/K/V tensor
    # passed as `query` itself, with `key`/`value` left empty -- confirmed
    # via live schema introspection
    # (``onnxruntime.capi.onnxruntime_pybind11_state.get_all_operator_schema()``,
    # `query`'s own doc string: "Query with shape (batch_size,
    # sequence_length, hidden_size), or packed QKV with shape (batch_size,
    # sequence_length, d)"; see this module's own "Attention-head pruning"
    # section comment). This is a different tensor layout from the
    # MatMul-then-Split-into-three-separate-inputs shape
    # `_match_packed_qkv_split` matches above (this model has no `Split`
    # node at all), and `_match_gqa_producer` has always declined it
    # outright, via its own ``not (node.input[0] and node.input[1] and
    # node.input[2])`` check -- confirmed here still holds, unmodified by
    # this module's new packed-`Split` support.
    K, H, KVH, D, Out = 8, 4, 2, 8, 6
    Nq, Nkv = H * D, KVH * D
    rng = np.random.default_rng(22)
    wqkv = rng.standard_normal((K, Nq + 2 * Nkv)).astype(np.float32)
    wout = rng.standard_normal((Nq, Out)).astype(np.float32)
    batch, seq = 2, 5

    initializer = [_f32(wqkv, "Wqkv"), _f32(wout, "Wout")]
    initializer.append(
        onnx.numpy_helper.from_array(
            np.full((batch,), seq - 1, dtype=np.int32), "SeqLensK"
        )
    )
    initializer.append(
        onnx.numpy_helper.from_array(np.array(seq, dtype=np.int32), "TotalSeq")
    )

    body = f"""
        g (float[{batch},{seq},{K}] X) => (float[{batch},{seq},{Out}] Y)
        {{
          qkv = MatMul(X, Wqkv)
          ctx, pk, pv = com.microsoft.GroupQueryAttention <num_heads={H}, kv_num_heads={KVH}> (qkv, , , , , SeqLensK, TotalSeq)
          Y = MatMul(ctx, Wout)
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

    assert onnxsim.pruning._find_gqa_chains(model.graph) == []

    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    assert pruned.SerializeToString() == model.SerializeToString()


def test_structured_pruning_importance_norm_l2_is_the_unchanged_default():
    model, wg, wu, wd = _swiglu_mlp_model(K=8, H=16, Out=4, seed=30)
    default = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    explicit = onnxsim.apply_structured_pruning(
        model, sparsity=0.5, importance_norm="l2"
    )
    assert default.SerializeToString() == explicit.SerializeToString()


def test_structured_wanda_pruning_importance_norm_l2_is_the_unchanged_default():
    K, H, Out = 8, 24, 4
    rng = np.random.default_rng(31)
    w1 = rng.standard_normal((K, H)).astype(np.float32)
    w2 = rng.standard_normal((H, Out)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          h = MatMul(X, W1)
          a = Relu(h)
          Y = MatMul(a, W2)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2")],
    )
    rng_cal = np.random.default_rng(32)
    calibration_data = [{"X": rng_cal.standard_normal((4, K)).astype(np.float32)}]
    default = onnxsim.apply_structured_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    explicit = onnxsim.apply_structured_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.5, importance_norm="l2"
    )
    assert default.SerializeToString() == explicit.SerializeToString()


def test_attention_head_pruning_importance_norm_l2_is_the_unchanged_default():
    model, _ = _attention_model(K=8, H=4, D=4, Out=6, seed=40)
    default = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    explicit = onnxsim.apply_attention_head_pruning(
        model, sparsity=0.5, importance_norm="l2"
    )
    assert default.SerializeToString() == explicit.SerializeToString()


def test_attention_head_wanda_pruning_importance_norm_l2_is_the_unchanged_default():
    model, _ = _attention_model(K=8, H=4, D=4, Out=6, seed=41)
    rng_cal = np.random.default_rng(33)
    calibration_data = [{"X": rng_cal.standard_normal((2, 5, 8)).astype(np.float32)}]
    default = onnxsim.apply_attention_head_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    explicit = onnxsim.apply_attention_head_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.5, importance_norm="l2"
    )
    assert default.SerializeToString() == explicit.SerializeToString()


def test_structured_pruning_invalid_importance_norm_raises():
    model = _mlp_model(K=8, H=16, Out=4)
    with pytest.raises(ValueError):
        onnxsim.apply_structured_pruning(model, importance_norm="l3")


def test_structured_wanda_pruning_invalid_importance_norm_raises():
    model = _mlp_model(K=8, H=16, Out=4)
    with pytest.raises(ValueError):
        onnxsim.apply_structured_wanda_pruning(
            model, calibration_data=[], importance_norm="l3"
        )


def test_attention_head_pruning_invalid_importance_norm_raises():
    model, _ = _attention_model(K=8, H=4, D=4, Out=6)
    with pytest.raises(ValueError):
        onnxsim.apply_attention_head_pruning(model, importance_norm="bogus")


def test_attention_head_wanda_pruning_invalid_importance_norm_raises():
    model, _ = _attention_model(K=8, H=4, D=4, Out=6)
    with pytest.raises(ValueError):
        onnxsim.apply_attention_head_wanda_pruning(
            model, calibration_data=[], importance_norm="bogus"
        )


# --- MoE expert-intermediate-channel pruning --------------------------------
#
# See ``onnxsim/pruning.py``'s own "MoE expert-intermediate-channel pruning"
# section comment for the full safety argument and exactly which
# ``com.microsoft::MoE`` shapes :func:`onnxsim.apply_moe_expert_channel_pruning`
# matches. Every test below that actually prunes something runs the result
# through a real onnxruntime CPU session -- confirmed, empirically, to be the
# one real oracle available for this op in this environment (``fc3`` and
# ``swiglu`` are not; see the "declines fc3"/"declines swiglu" tests, which
# instead confirm the *node* is left untouched rather than trying to execute
# an unsupported combination).


def _moe_model(
    fc1_w,
    fc2_w,
    fc1_b=None,
    fc2_b=None,
    fc3_w=None,
    activation="relu",
    swiglu_fusion=0,
    k=2,
    tokens=6,
):
    num_experts, inter, hidden = fc1_w.shape
    fc1_b_arg = "FC1B" if fc1_b is not None else ""
    fc2_b_arg = "FC2B" if fc2_b is not None else ""
    fc3_w_arg = "FC3W" if fc3_w is not None else ""
    model = _model(
        f"""
        g (float[{tokens},{hidden}] X, float[{tokens},{num_experts}] R) => (float[{tokens},{hidden}] Y)
        {{
          Y = com.microsoft.MoE <k={k}, activation_type="{activation}", swiglu_fusion={swiglu_fusion}> (X, R, FC1W, {fc1_b_arg}, FC2W, {fc2_b_arg}, {fc3_w_arg})
        }}
        """,
        opset=18,
    )
    inits = [_f32(fc1_w, "FC1W"), _f32(fc2_w, "FC2W")]
    if fc1_b is not None:
        inits.append(_f32(fc1_b, "FC1B"))
    if fc2_b is not None:
        inits.append(_f32(fc2_b, "FC2B"))
    if fc3_w is not None:
        inits.append(_f32(fc3_w, "FC3W"))
    model.graph.initializer.extend(inits)
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))
    return model


def _moe_inits(model):
    return {t.name: onnx.numpy_helper.to_array(t) for t in model.graph.initializer}


def test_moe_expert_channel_pruning_matches_ort_masking_oracle():
    # Physically removing the lowest-importance `inter_size` channels must be
    # numerically identical to *zeroing* those same channels (fc1's own row,
    # fc1_experts_bias's own entry, fc2's own column) in a same-shape model:
    # a dropped channel's fc1 output is then exactly 0 for relu (bias and
    # weight both zeroed, so pre-activation is 0 and relu(0) == 0) and
    # contributes nothing through fc2's own zeroed column either way. This is
    # the real onnxruntime CPU-execution oracle this pass's own safety
    # argument rests on.
    E, hidden, inter = 5, 10, 12
    rng = np.random.default_rng(3)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.5).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.5).astype(np.float32)
    fc1_b = rng.standard_normal((E, inter)).astype(np.float32)
    fc2_b = rng.standard_normal((E, hidden)).astype(np.float32)
    model = _moe_model(fc1_w, fc2_w, fc1_b=fc1_b, fc2_b=fc2_b)
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_moe_expert_channel_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = _moe_inits(pruned)
    assert inits["FC1W"].shape == (E, 6, hidden)
    assert inits["FC2W"].shape == (E, hidden, 6)
    assert inits["FC1B"].shape == (E, 6)
    np.testing.assert_array_equal(
        inits["FC2B"], fc2_b
    )  # indexes hidden_size, untouched

    sq = (
        np.sum(fc1_w**2, axis=(0, 2))
        + np.sum(fc2_w**2, axis=(0, 1))
        + np.sum(fc1_b**2, axis=0)
    )
    keep = np.sort(np.argsort(-np.sqrt(sq))[:6])
    drop = np.setdiff1d(np.arange(inter), keep)
    np.testing.assert_allclose(inits["FC1W"], fc1_w[:, keep, :])
    np.testing.assert_allclose(inits["FC2W"], fc2_w[:, :, keep])
    np.testing.assert_allclose(inits["FC1B"], fc1_b[:, keep])

    fc1_w_masked = fc1_w.copy()
    fc1_w_masked[:, drop, :] = 0
    fc1_b_masked = fc1_b.copy()
    fc1_b_masked[:, drop] = 0
    fc2_w_masked = fc2_w.copy()
    fc2_w_masked[:, :, drop] = 0
    masked = _moe_model(fc1_w_masked, fc2_w_masked, fc1_b=fc1_b_masked, fc2_b=fc2_b)

    rng2 = np.random.default_rng(7)
    tokens = 6
    feeds = {
        "X": rng2.standard_normal((tokens, hidden)).astype(np.float32),
        "R": rng2.standard_normal((tokens, E)).astype(np.float32),
    }
    (out_pruned,) = _run(pruned, feeds)
    (out_masked,) = _run(masked, feeds)
    np.testing.assert_allclose(out_pruned, out_masked, rtol=1e-5, atol=1e-5)


def test_moe_expert_channel_pruning_adversarial_conflicting_fc1_fc2_importance():
    # Deliberately conflicting per-channel importance: channel A has a large
    # fc1 row but a tiny fc2 column, channel B the reverse, and channel C is
    # tiny on both. A bug that ranked by only one of fc1/fc2 (instead of the
    # documented combined root-sum-square of both) would keep A or B, not
    # both -- this catches that by making the *combined* score of A and B
    # comparably large (each large on one side) while C is small on both, so
    # only C should be the one channel dropped at sparsity=1/3.
    E, hidden, inter = 3, 4, 3
    rng = np.random.default_rng(11)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.01).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.01).astype(np.float32)
    fc1_w[:, 0, :] = 5.0  # channel 0 (A): large fc1, tiny fc2 (left as noise)
    fc2_w[:, :, 1] = 5.0  # channel 1 (B): tiny fc1 (noise), large fc2
    # channel 2 (C) stays small noise on both fc1 and fc2 -- the one channel
    # this test expects to be dropped.
    model = _moe_model(fc1_w, fc2_w)

    pruned = onnxsim.apply_moe_expert_channel_pruning(model, sparsity=1.0 / 3.0)
    onnx.checker.check_model(pruned)
    inits = _moe_inits(pruned)
    assert inits["FC1W"].shape == (E, 2, hidden)
    np.testing.assert_allclose(inits["FC1W"], fc1_w[:, [0, 1], :])
    np.testing.assert_allclose(inits["FC2W"], fc2_w[:, :, [0, 1]])


def test_moe_expert_channel_pruning_zero_sparsity_is_a_no_op():
    E, hidden, inter = 3, 6, 5
    rng = np.random.default_rng(13)
    fc1_w = rng.standard_normal((E, inter, hidden)).astype(np.float32)
    fc2_w = rng.standard_normal((E, hidden, inter)).astype(np.float32)
    model = _moe_model(fc1_w, fc2_w)
    pruned = onnxsim.apply_moe_expert_channel_pruning(model, sparsity=0.0)
    inits = _moe_inits(pruned)
    np.testing.assert_array_equal(inits["FC1W"], fc1_w)
    np.testing.assert_array_equal(inits["FC2W"], fc2_w)


def test_moe_expert_channel_pruning_declines_fc3():
    # com.microsoft::MoE's own CPU execution provider, in this environment,
    # raises "FC3 is not implemented for CPU MoE" for any activation_type --
    # confirmed empirically, see this module's own section comment -- so a
    # node with fc3_experts_weights present is left completely untouched
    # rather than pruned against a shape this environment has no real
    # runtime to validate.
    E, hidden, inter = 3, 6, 5
    rng = np.random.default_rng(17)
    fc1_w = rng.standard_normal((E, inter, hidden)).astype(np.float32)
    fc2_w = rng.standard_normal((E, hidden, inter)).astype(np.float32)
    fc3_w = rng.standard_normal((E, inter, hidden)).astype(np.float32)
    model = _moe_model(fc1_w, fc2_w, fc3_w=fc3_w, activation="silu")
    onnx.checker.check_model(model)
    pruned = onnxsim.apply_moe_expert_channel_pruning(model, sparsity=0.5)
    inits = _moe_inits(pruned)
    np.testing.assert_array_equal(inits["FC1W"], fc1_w)
    np.testing.assert_array_equal(inits["FC2W"], fc2_w)
    np.testing.assert_array_equal(inits["FC3W"], fc3_w)


def test_moe_expert_channel_pruning_declines_swiglu_activation():
    E, hidden, inter = 3, 6, 5
    rng = np.random.default_rng(19)
    fc1_w = rng.standard_normal((E, inter, hidden)).astype(np.float32)
    fc2_w = rng.standard_normal((E, hidden, inter)).astype(np.float32)
    model = _moe_model(fc1_w, fc2_w, activation="swiglu")
    pruned = onnxsim.apply_moe_expert_channel_pruning(model, sparsity=0.5)
    inits = _moe_inits(pruned)
    np.testing.assert_array_equal(inits["FC1W"], fc1_w)
    np.testing.assert_array_equal(inits["FC2W"], fc2_w)


def test_moe_expert_channel_pruning_declines_nonzero_swiglu_fusion():
    E, hidden, inter = 3, 6, 5
    rng = np.random.default_rng(23)
    fc1_w = rng.standard_normal((E, inter, hidden)).astype(np.float32)
    fc2_w = rng.standard_normal((E, hidden, inter)).astype(np.float32)
    model = _moe_model(fc1_w, fc2_w, swiglu_fusion=1)
    pruned = onnxsim.apply_moe_expert_channel_pruning(model, sparsity=0.5)
    inits = _moe_inits(pruned)
    np.testing.assert_array_equal(inits["FC1W"], fc1_w)
    np.testing.assert_array_equal(inits["FC2W"], fc2_w)


def test_moe_expert_channel_pruning_declines_fused_swiglu_shape():
    # A real fused-swiglu fc1 doubles its own row count (fusion_size=2) --
    # fc1's axis-1 size then never equals fc2's own axis-2 size, so this
    # declines via the shape-consistency check alone, without even needing
    # to read swiglu_fusion/activation_type (see this module's own section
    # comment for why that's a deliberate, attribute-free safety net).
    E, hidden, inter = 3, 6, 5
    rng = np.random.default_rng(29)
    fc1_w = rng.standard_normal((E, 2 * inter, hidden)).astype(np.float32)
    fc2_w = rng.standard_normal((E, hidden, inter)).astype(np.float32)
    model = _moe_model(fc1_w, fc2_w, activation="swiglu", swiglu_fusion=1)
    pruned = onnxsim.apply_moe_expert_channel_pruning(model, sparsity=0.5)
    inits = _moe_inits(pruned)
    np.testing.assert_array_equal(inits["FC1W"], fc1_w)
    np.testing.assert_array_equal(inits["FC2W"], fc2_w)


def test_moe_expert_channel_pruning_declines_tied_fc1_weight():
    # fc1_experts_weights reused by a second node -- an in-place resize
    # would corrupt that other consumer, so this is declined outright, the
    # same tied-weight guard every other chain-matcher in this module
    # applies via `consumers_of`.
    E, hidden, inter = 3, 6, 5
    rng = np.random.default_rng(31)
    fc1_w = rng.standard_normal((E, inter, hidden)).astype(np.float32)
    fc2_w = rng.standard_normal((E, hidden, inter)).astype(np.float32)
    model = _model(
        f"""
        g (float[6,{hidden}] X, float[6,{E}] R) => (float[6,{hidden}] Y, float[{E},{inter},{hidden}] Z)
        {{
          Y = com.microsoft.MoE <k=2, activation_type="relu"> (X, R, FC1W, , FC2W)
          Z = Identity(FC1W)
        }}
        """,
        initializer=[_f32(fc1_w, "FC1W"), _f32(fc2_w, "FC2W")],
        opset=18,
    )
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))
    pruned = onnxsim.apply_moe_expert_channel_pruning(model, sparsity=0.5)
    inits = _moe_inits(pruned)
    np.testing.assert_array_equal(inits["FC1W"], fc1_w)
    np.testing.assert_array_equal(inits["FC2W"], fc2_w)


def test_moe_expert_channel_pruning_declines_non_constant_weight():
    # fc1_experts_weights fed by a graph input (not an initializer) --
    # there's nothing to slice in place, so the node is left untouched
    # rather than raising.
    E, hidden, inter = 3, 6, 5
    rng = np.random.default_rng(37)
    fc2_w = rng.standard_normal((E, hidden, inter)).astype(np.float32)
    model = _model(
        f"""
        g (float[6,{hidden}] X, float[6,{E}] R, float[{E},{inter},{hidden}] FC1W) => (float[6,{hidden}] Y)
        {{
          Y = com.microsoft.MoE <k=2, activation_type="relu"> (X, R, FC1W, , FC2W)
        }}
        """,
        initializer=[_f32(fc2_w, "FC2W")],
        opset=18,
    )
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))
    pruned = onnxsim.apply_moe_expert_channel_pruning(model, sparsity=0.5)
    inits = _moe_inits(pruned)
    np.testing.assert_array_equal(inits["FC2W"], fc2_w)


def test_moe_expert_channel_pruning_declines_mismatched_hidden_size():
    E, hidden, inter = 3, 6, 5
    rng = np.random.default_rng(41)
    fc1_w = rng.standard_normal((E, inter, hidden)).astype(np.float32)
    fc2_w = rng.standard_normal((E, hidden + 1, inter)).astype(np.float32)  # mismatched
    model = _model(
        f"""
        g (float[6,{hidden}] X, float[6,{E}] R) => (float[6,{hidden + 1}] Y)
        {{
          Y = com.microsoft.MoE <k=2, activation_type="relu"> (X, R, FC1W, , FC2W)
        }}
        """,
        initializer=[_f32(fc1_w, "FC1W"), _f32(fc2_w, "FC2W")],
        opset=18,
    )
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))
    pruned = onnxsim.apply_moe_expert_channel_pruning(model, sparsity=0.5)
    inits = _moe_inits(pruned)
    np.testing.assert_array_equal(inits["FC1W"], fc1_w)
    np.testing.assert_array_equal(inits["FC2W"], fc2_w)


def test_moe_expert_channel_pruning_no_bias_matches_ort_masking_oracle():
    E, hidden, inter = 4, 8, 10
    rng = np.random.default_rng(43)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.5).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.5).astype(np.float32)
    model = _moe_model(fc1_w, fc2_w, activation="gelu")
    pruned = onnxsim.apply_moe_expert_channel_pruning(model, sparsity=0.4)
    onnx.checker.check_model(pruned)
    inits = _moe_inits(pruned)
    keep_count = inits["FC1W"].shape[1]
    assert keep_count == 6

    sq = np.sum(fc1_w**2, axis=(0, 2)) + np.sum(fc2_w**2, axis=(0, 1))
    keep = np.sort(np.argsort(-np.sqrt(sq))[:keep_count])
    drop = np.setdiff1d(np.arange(inter), keep)
    fc1_w_masked = fc1_w.copy()
    fc1_w_masked[:, drop, :] = 0
    fc2_w_masked = fc2_w.copy()
    fc2_w_masked[:, :, drop] = 0
    masked = _moe_model(fc1_w_masked, fc2_w_masked, activation="gelu")

    rng2 = np.random.default_rng(47)
    tokens = 6
    feeds = {
        "X": rng2.standard_normal((tokens, hidden)).astype(np.float32),
        "R": rng2.standard_normal((tokens, E)).astype(np.float32),
    }
    (out_pruned,) = _run(pruned, feeds)
    (out_masked,) = _run(masked, feeds)
    np.testing.assert_allclose(out_pruned, out_masked, rtol=1e-4, atol=1e-4)


def test_moe_expert_channel_pruning_multiple_nodes_pruned_independently():
    E1, hidden1, inter1 = 3, 4, 6
    E2, hidden2, inter2 = 2, 5, 8
    rng = np.random.default_rng(53)
    fc1_w1 = rng.standard_normal((E1, inter1, hidden1)).astype(np.float32)
    fc2_w1 = rng.standard_normal((E1, hidden1, inter1)).astype(np.float32)
    fc1_w2 = rng.standard_normal((E2, inter2, hidden2)).astype(np.float32)
    fc2_w2 = rng.standard_normal((E2, hidden2, inter2)).astype(np.float32)
    model = _model(
        f"""
        g (float[6,{hidden1}] X1, float[6,{E1}] R1, float[6,{hidden2}] X2, float[6,{E2}] R2)
            => (float[6,{hidden1}] Y1, float[6,{hidden2}] Y2)
        {{
          Y1 = com.microsoft.MoE <k=1, activation_type="relu"> (X1, R1, FC1W1, , FC2W1)
          Y2 = com.microsoft.MoE <k=1, activation_type="relu"> (X2, R2, FC1W2, , FC2W2)
        }}
        """,
        initializer=[
            _f32(fc1_w1, "FC1W1"),
            _f32(fc2_w1, "FC2W1"),
            _f32(fc1_w2, "FC1W2"),
            _f32(fc2_w2, "FC2W2"),
        ],
        opset=18,
    )
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))
    pruned = onnxsim.apply_moe_expert_channel_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = _moe_inits(pruned)
    assert inits["FC1W1"].shape == (E1, 3, hidden1)
    assert inits["FC1W2"].shape == (E2, 4, hidden2)


# --- MoE whole-expert pruning -------------------------------------------------
#
# See ``onnxsim/pruning.py``'s own "MoE whole-expert pruning" section comment
# for the full safety argument: shrinking `router_probs`' own width is exactly
# equivalent (confirmed to 0.0 max-abs-diff against a real onnxruntime CPU
# session) to forcing the dropped experts' routing logits to `-inf` in a
# same-shape model, which is the oracle every execution test below uses.


def _moe_router_model(
    fc1_w,
    fc2_w,
    router_w,
    router_b=None,
    fc1_b=None,
    fc2_b=None,
    fc3_w=None,
    activation="relu",
    swiglu_fusion=0,
    use_sparse_mixer=0,
    k=2,
    tokens=6,
    extra_router_consumer=False,
):
    num_experts, inter, hidden = fc1_w.shape
    fc1_b_arg = "FC1B" if fc1_b is not None else ""
    fc2_b_arg = "FC2B" if fc2_b is not None else ""
    fc3_w_arg = "FC3W" if fc3_w is not None else ""
    router_call = "Gemm(X, RW, RB)" if router_b is not None else "Gemm(X, RW)"
    extra_out = f", float[{tokens},{num_experts}] R2" if extra_router_consumer else ""
    extra_node = "R2 = Identity(R)" if extra_router_consumer else ""
    model = _model(
        f"""
        g (float[{tokens},{hidden}] X) => (float[{tokens},{hidden}] Y{extra_out})
        {{
          R = {router_call}
          Y = com.microsoft.MoE <k={k}, activation_type="{activation}", swiglu_fusion={swiglu_fusion}, use_sparse_mixer={use_sparse_mixer}> (X, R, FC1W, {fc1_b_arg}, FC2W, {fc2_b_arg}, {fc3_w_arg})
          {extra_node}
        }}
        """,
        opset=18,
    )
    inits = [_f32(fc1_w, "FC1W"), _f32(fc2_w, "FC2W"), _f32(router_w, "RW")]
    if router_b is not None:
        inits.append(_f32(router_b, "RB"))
    if fc1_b is not None:
        inits.append(_f32(fc1_b, "FC1B"))
    if fc2_b is not None:
        inits.append(_f32(fc2_b, "FC2B"))
    if fc3_w is not None:
        inits.append(_f32(fc3_w, "FC3W"))
    model.graph.initializer.extend(inits)
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))
    return model


def _moe_router_masking_oracle(
    fc1_w,
    fc2_w,
    router_w,
    router_b,
    dropped,
    k,
    fc1_b=None,
    activation="relu",
    tokens=6,
):
    # Same-shape model with every `dropped` expert's routing logit forced to
    # -1e9 (so Softmax assigns it exactly 0 probability, dropping it from both
    # top-k selection and any `normalize_routing_weights` renormalization)
    # and its own fc1/fc2 (+fc1_b) rows zeroed -- see this section's own
    # comment above.
    fc1_w_masked = fc1_w.copy()
    fc2_w_masked = fc2_w.copy()
    fc1_b_masked = fc1_b.copy() if fc1_b is not None else None
    router_b_masked = (
        router_b.copy()
        if router_b is not None
        else np.zeros(fc1_w.shape[0], np.float32)
    )
    for e in dropped:
        fc1_w_masked[e] = 0
        fc2_w_masked[e] = 0
        if fc1_b_masked is not None:
            fc1_b_masked[e] = 0
        router_b_masked[e] = -1e9
    return _moe_router_model(
        fc1_w_masked,
        fc2_w_masked,
        router_w,
        router_b=router_b_masked,
        fc1_b=fc1_b_masked,
        activation=activation,
        k=k,
        tokens=tokens,
    )


def test_moe_whole_expert_pruning_matches_ort_masking_oracle():
    E, hidden, inter, tokens = 5, 8, 6, 10
    rng = np.random.default_rng(101)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.4).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.4).astype(np.float32)
    fc1_b = rng.standard_normal((E, inter)).astype(np.float32)
    router_w = (rng.standard_normal((hidden, E)) * 0.2).astype(np.float32)
    router_b = rng.standard_normal(E).astype(np.float32)
    k = 2
    model = _moe_router_model(
        fc1_w, fc2_w, router_w, router_b=router_b, fc1_b=fc1_b, k=k, tokens=tokens
    )
    onnx.checker.check_model(model)

    calib_rng = np.random.default_rng(103)
    calibration_data = [
        {"X": calib_rng.standard_normal((tokens, hidden)).astype(np.float32)}
        for _ in range(4)
    ]
    pruned = onnxsim.apply_moe_whole_expert_pruning(
        model,
        calibration_data=calibration_data,
        sparsity=0.4,  # keep 3 of 5
    )
    onnx.checker.check_model(pruned)
    inits = _moe_inits(pruned)
    assert inits["FC1W"].shape == (3, inter, hidden)
    assert inits["RW"].shape == (hidden, 3)
    assert inits["FC1B"].shape == (3, inter)

    kept_router_w = inits["RW"]
    dropped = [
        e
        for e in range(E)
        if not any(np.allclose(router_w[:, e], kept_router_w[:, i]) for i in range(3))
    ]
    assert len(dropped) == 2
    masked = _moe_router_masking_oracle(
        fc1_w, fc2_w, router_w, router_b, dropped, k, fc1_b=fc1_b, tokens=tokens
    )

    feed_rng = np.random.default_rng(107)
    feeds = {"X": feed_rng.standard_normal((tokens, hidden)).astype(np.float32)}
    (out_pruned,) = _run(pruned, feeds)
    (out_masked,) = _run(masked, feeds)
    np.testing.assert_allclose(out_pruned, out_masked, rtol=1e-4, atol=1e-4)


def test_moe_whole_expert_pruning_adversarial_low_usage_expert_dropped():
    # Deliberately conflicting usage: expert 0's router bias is large and
    # positive (dominant -- selected with high gate weight on almost every
    # token), expert (E-1)'s is large and negative (rarely/never selected,
    # near-zero mean gate weight), the rest are mid-range noise. At
    # sparsity=1/E (drop exactly one expert), the correct (low-usage) expert
    # must be the one dropped, not the dominant one -- catches a ranking bug
    # that inverted the comparison or picked the highest-usage expert instead.
    E, hidden, inter, tokens = 4, 6, 5, 8
    rng = np.random.default_rng(109)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float32)
    router_w = (rng.standard_normal((hidden, E)) * 0.05).astype(np.float32)
    router_b = np.zeros(E, dtype=np.float32)
    router_b[0] = 8.0  # dominant
    router_b[E - 1] = -8.0  # rarely used -- expected to be dropped
    k = 1
    model = _moe_router_model(
        fc1_w, fc2_w, router_w, router_b=router_b, k=k, tokens=tokens
    )
    onnx.checker.check_model(model)

    calib_rng = np.random.default_rng(113)
    calibration_data = [
        {"X": calib_rng.standard_normal((tokens, hidden)).astype(np.float32)}
        for _ in range(4)
    ]
    pruned = onnxsim.apply_moe_whole_expert_pruning(
        model, calibration_data=calibration_data, sparsity=1.0 / E
    )
    inits = _moe_inits(pruned)
    assert inits["FC1W"].shape == (E - 1, inter, hidden)
    kept_router_w = inits["RW"]
    dropped = [
        e
        for e in range(E)
        if not any(
            np.allclose(router_w[:, e], kept_router_w[:, i]) for i in range(E - 1)
        )
    ]
    assert dropped == [E - 1], f"expected the rarely-used expert dropped, got {dropped}"


def test_moe_whole_expert_pruning_k_is_floored_not_exceeded():
    # k=2 must never be pruned below -- confirmed empirically that ONNX
    # Runtime's own CPU MoE kernel fails execution outright with `k` >
    # `num_experts` (see this module's own section comment). Requesting
    # sparsity that would remove more than num_experts - k experts is
    # silently floored instead.
    E, hidden, inter, tokens = 5, 6, 4, 6
    rng = np.random.default_rng(127)
    fc1_w = rng.standard_normal((E, inter, hidden)).astype(np.float32)
    fc2_w = rng.standard_normal((E, hidden, inter)).astype(np.float32)
    router_w = rng.standard_normal((hidden, E)).astype(np.float32)
    k = 2
    model = _moe_router_model(fc1_w, fc2_w, router_w, k=k, tokens=tokens)
    pruned = onnxsim.apply_moe_whole_expert_pruning(
        model, calibration_data=[], sparsity=0.9
    )
    onnx.checker.check_model(pruned)
    inits = _moe_inits(pruned)
    assert inits["FC1W"].shape == (k, inter, hidden)
    assert inits["RW"].shape == (hidden, k)
    feed_rng = np.random.default_rng(131)
    feeds = {"X": feed_rng.standard_normal((tokens, hidden)).astype(np.float32)}
    _run(pruned, feeds)  # must execute without error


def test_moe_whole_expert_pruning_zero_sparsity_is_a_no_op():
    E, hidden, inter, tokens = 4, 6, 5, 6
    rng = np.random.default_rng(137)
    fc1_w = rng.standard_normal((E, inter, hidden)).astype(np.float32)
    fc2_w = rng.standard_normal((E, hidden, inter)).astype(np.float32)
    router_w = rng.standard_normal((hidden, E)).astype(np.float32)
    model = _moe_router_model(fc1_w, fc2_w, router_w, tokens=tokens)
    pruned = onnxsim.apply_moe_whole_expert_pruning(
        model, calibration_data=[], sparsity=0.0
    )
    inits = _moe_inits(pruned)
    np.testing.assert_array_equal(inits["FC1W"], fc1_w)
    np.testing.assert_array_equal(inits["FC2W"], fc2_w)
    np.testing.assert_array_equal(inits["RW"], router_w)


def test_moe_whole_expert_pruning_invalid_sparsity_raises():
    E, hidden, inter, tokens = 3, 4, 3, 4
    rng = np.random.default_rng(139)
    fc1_w = rng.standard_normal((E, inter, hidden)).astype(np.float32)
    fc2_w = rng.standard_normal((E, hidden, inter)).astype(np.float32)
    router_w = rng.standard_normal((hidden, E)).astype(np.float32)
    model = _moe_router_model(fc1_w, fc2_w, router_w, tokens=tokens)
    with pytest.raises(ValueError):
        onnxsim.apply_moe_whole_expert_pruning(model, sparsity=1.0)
    with pytest.raises(ValueError):
        onnxsim.apply_moe_whole_expert_pruning(model, sparsity=-0.1)


def test_moe_whole_expert_pruning_empty_calibration_falls_back_to_weight_norm():
    # No calibration data observed for this chain's router_probs -> falls
    # back to each expert's own combined fc1/fc2 L2 weight norm (the same
    # "no matching activation observed" fallback
    # apply_structured_wanda_pruning already uses). Deliberately conflicting
    # per-expert weight magnitude: expert 0 tiny on both fc1/fc2 (expected
    # dropped), the rest large.
    E, hidden, inter, tokens = 3, 4, 3, 4
    rng = np.random.default_rng(149)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 5.0).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 5.0).astype(np.float32)
    fc1_w[0] *= 0.001
    fc2_w[0] *= 0.001
    router_w = rng.standard_normal((hidden, E)).astype(np.float32)
    model = _moe_router_model(fc1_w, fc2_w, router_w, k=1, tokens=tokens)
    pruned = onnxsim.apply_moe_whole_expert_pruning(
        model, calibration_data=[], sparsity=1.0 / E
    )
    inits = _moe_inits(pruned)
    assert inits["FC1W"].shape == (E - 1, inter, hidden)
    kept_router_w = inits["RW"]
    dropped = [
        e
        for e in range(E)
        if not any(
            np.allclose(router_w[:, e], kept_router_w[:, i]) for i in range(E - 1)
        )
    ]
    assert dropped == [0]


def test_moe_whole_expert_pruning_declines_fc3():
    E, hidden, inter, tokens = 3, 6, 5, 6
    rng = np.random.default_rng(151)
    fc1_w = rng.standard_normal((E, inter, hidden)).astype(np.float32)
    fc2_w = rng.standard_normal((E, hidden, inter)).astype(np.float32)
    fc3_w = rng.standard_normal((E, inter, hidden)).astype(np.float32)
    router_w = rng.standard_normal((hidden, E)).astype(np.float32)
    model = _moe_router_model(
        fc1_w, fc2_w, router_w, fc3_w=fc3_w, activation="silu", tokens=tokens
    )
    pruned = onnxsim.apply_moe_whole_expert_pruning(
        model, calibration_data=[], sparsity=0.5
    )
    inits = _moe_inits(pruned)
    np.testing.assert_array_equal(inits["FC1W"], fc1_w)
    np.testing.assert_array_equal(inits["RW"], router_w)


def test_moe_whole_expert_pruning_declines_swiglu_activation():
    E, hidden, inter, tokens = 3, 6, 5, 6
    rng = np.random.default_rng(157)
    fc1_w = rng.standard_normal((E, inter, hidden)).astype(np.float32)
    fc2_w = rng.standard_normal((E, hidden, inter)).astype(np.float32)
    router_w = rng.standard_normal((hidden, E)).astype(np.float32)
    model = _moe_router_model(
        fc1_w, fc2_w, router_w, activation="swiglu", tokens=tokens
    )
    pruned = onnxsim.apply_moe_whole_expert_pruning(
        model, calibration_data=[], sparsity=0.5
    )
    inits = _moe_inits(pruned)
    np.testing.assert_array_equal(inits["FC1W"], fc1_w)
    np.testing.assert_array_equal(inits["RW"], router_w)


def test_moe_whole_expert_pruning_declines_use_sparse_mixer():
    # use_sparse_mixer=1 hard-requires k == 2 (confirmed empirically,
    # moe_base_cpu.h's own "Sparse mixer only supports k=2" check) and
    # engages a different, jitter-named top-2 routing path this pass's own
    # `-inf`-masking oracle was never independently re-checked against -- so
    # it's declined outright rather than assumed safe.
    E, hidden, inter, tokens = 4, 6, 5, 6
    rng = np.random.default_rng(163)
    fc1_w = rng.standard_normal((E, inter, hidden)).astype(np.float32)
    fc2_w = rng.standard_normal((E, hidden, inter)).astype(np.float32)
    router_w = rng.standard_normal((hidden, E)).astype(np.float32)
    model = _moe_router_model(
        fc1_w, fc2_w, router_w, k=2, use_sparse_mixer=1, tokens=tokens
    )
    onnx.checker.check_model(model)
    pruned = onnxsim.apply_moe_whole_expert_pruning(
        model, calibration_data=[], sparsity=0.5
    )
    inits = _moe_inits(pruned)
    np.testing.assert_array_equal(inits["FC1W"], fc1_w)
    np.testing.assert_array_equal(inits["RW"], router_w)


def test_moe_whole_expert_pruning_declines_router_with_extra_consumer():
    # router_probs (the router projection's own output) feeding anything
    # besides this one MoE node would silently see a now-differently-shaped
    # tensor if pruned -- declined outright.
    E, hidden, inter, tokens = 3, 6, 5, 6
    rng = np.random.default_rng(167)
    fc1_w = rng.standard_normal((E, inter, hidden)).astype(np.float32)
    fc2_w = rng.standard_normal((E, hidden, inter)).astype(np.float32)
    router_w = rng.standard_normal((hidden, E)).astype(np.float32)
    model = _moe_router_model(
        fc1_w, fc2_w, router_w, tokens=tokens, extra_router_consumer=True
    )
    onnx.checker.check_model(model)
    pruned = onnxsim.apply_moe_whole_expert_pruning(
        model, calibration_data=[], sparsity=0.5
    )
    inits = _moe_inits(pruned)
    np.testing.assert_array_equal(inits["FC1W"], fc1_w)
    np.testing.assert_array_equal(inits["RW"], router_w)


def test_moe_whole_expert_pruning_declines_tied_router_weight():
    # The router weight reused by a second node -- an in-place resize would
    # corrupt that other consumer, the same tied-weight guard
    # apply_moe_expert_channel_pruning already applies to fc1/fc2.
    E, hidden, inter, tokens = 3, 6, 5, 6
    rng = np.random.default_rng(173)
    fc1_w = rng.standard_normal((E, inter, hidden)).astype(np.float32)
    fc2_w = rng.standard_normal((E, hidden, inter)).astype(np.float32)
    router_w = rng.standard_normal((hidden, E)).astype(np.float32)
    model = _model(
        f"""
        g (float[{tokens},{hidden}] X) => (float[{tokens},{hidden}] Y, float[{hidden},{E}] Z)
        {{
          R = MatMul(X, RW)
          Y = com.microsoft.MoE <k=1, activation_type="relu"> (X, R, FC1W, , FC2W)
          Z = Identity(RW)
        }}
        """,
        initializer=[_f32(fc1_w, "FC1W"), _f32(fc2_w, "FC2W"), _f32(router_w, "RW")],
        opset=18,
    )
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))
    pruned = onnxsim.apply_moe_whole_expert_pruning(
        model, calibration_data=[], sparsity=0.5
    )
    inits = _moe_inits(pruned)
    np.testing.assert_array_equal(inits["FC1W"], fc1_w)
    np.testing.assert_array_equal(inits["RW"], router_w)


def test_moe_whole_expert_pruning_declines_non_matmul_router_producer():
    # router_probs fed directly as a graph input (no producer node at all --
    # or, equally, any producer that isn't a plain MatMul/Gemm) can't be
    # safely resized: there's no weight to slice, so the node is left
    # untouched rather than raising.
    E, hidden, inter, tokens = 3, 6, 5, 6
    rng = np.random.default_rng(179)
    fc1_w = rng.standard_normal((E, inter, hidden)).astype(np.float32)
    fc2_w = rng.standard_normal((E, hidden, inter)).astype(np.float32)
    model = _model(
        f"""
        g (float[{tokens},{hidden}] X, float[{tokens},{E}] R) => (float[{tokens},{hidden}] Y)
        {{
          Y = com.microsoft.MoE <k=1, activation_type="relu"> (X, R, FC1W, , FC2W)
        }}
        """,
        initializer=[_f32(fc1_w, "FC1W"), _f32(fc2_w, "FC2W")],
        opset=18,
    )
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))
    pruned = onnxsim.apply_moe_whole_expert_pruning(
        model, calibration_data=[], sparsity=0.5
    )
    inits = _moe_inits(pruned)
    np.testing.assert_array_equal(inits["FC1W"], fc1_w)
    np.testing.assert_array_equal(inits["FC2W"], fc2_w)


def test_moe_whole_expert_pruning_declines_k_equals_num_experts():
    # Nothing can be pruned without violating the k floor -- a no-op, not an
    # error.
    E, hidden, inter, tokens = 3, 5, 4, 5
    rng = np.random.default_rng(181)
    fc1_w = rng.standard_normal((E, inter, hidden)).astype(np.float32)
    fc2_w = rng.standard_normal((E, hidden, inter)).astype(np.float32)
    router_w = rng.standard_normal((hidden, E)).astype(np.float32)
    model = _moe_router_model(fc1_w, fc2_w, router_w, k=E, tokens=tokens)
    pruned = onnxsim.apply_moe_whole_expert_pruning(
        model, calibration_data=[], sparsity=0.9
    )
    inits = _moe_inits(pruned)
    np.testing.assert_array_equal(inits["FC1W"], fc1_w)
    np.testing.assert_array_equal(inits["RW"], router_w)


def test_moe_whole_expert_pruning_multiple_nodes_pruned_independently():
    E1, hidden1, inter1, tokens = 4, 4, 5, 6
    E2, hidden2, inter2 = 3, 5, 4
    rng = np.random.default_rng(191)
    fc1_w1 = rng.standard_normal((E1, inter1, hidden1)).astype(np.float32)
    fc2_w1 = rng.standard_normal((E1, hidden1, inter1)).astype(np.float32)
    router_w1 = rng.standard_normal((hidden1, E1)).astype(np.float32)
    fc1_w2 = rng.standard_normal((E2, inter2, hidden2)).astype(np.float32)
    fc2_w2 = rng.standard_normal((E2, hidden2, inter2)).astype(np.float32)
    router_w2 = rng.standard_normal((hidden2, E2)).astype(np.float32)
    model = _model(
        f"""
        g (float[{tokens},{hidden1}] X1, float[{tokens},{hidden2}] X2)
            => (float[{tokens},{hidden1}] Y1, float[{tokens},{hidden2}] Y2)
        {{
          R1 = MatMul(X1, RW1)
          Y1 = com.microsoft.MoE <k=1, activation_type="relu"> (X1, R1, FC1W1, , FC2W1)
          R2 = MatMul(X2, RW2)
          Y2 = com.microsoft.MoE <k=1, activation_type="relu"> (X2, R2, FC1W2, , FC2W2)
        }}
        """,
        initializer=[
            _f32(fc1_w1, "FC1W1"),
            _f32(fc2_w1, "FC2W1"),
            _f32(router_w1, "RW1"),
            _f32(fc1_w2, "FC1W2"),
            _f32(fc2_w2, "FC2W2"),
            _f32(router_w2, "RW2"),
        ],
        opset=18,
    )
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))
    onnx.checker.check_model(model)
    pruned = onnxsim.apply_moe_whole_expert_pruning(
        model, calibration_data=[], sparsity=0.5
    )
    onnx.checker.check_model(pruned)
    inits = _moe_inits(pruned)
    assert inits["FC1W1"].shape == (2, inter1, hidden1)  # 4 - round(4*0.5) = 2
    assert inits["RW1"].shape == (hidden1, 2)
    assert inits["FC1W2"].shape == (1, inter2, hidden2)  # 3 - round(3*0.5) = 1
    assert inits["RW2"].shape == (hidden2, 1)


# --- FP16/BFloat16 weight support -----------------------------------------
#
# Every matcher in this module used to hard-require ``onnx.TensorProto.
# FLOAT``, silently declining any layer whose weight was stored as
# FLOAT16 or BFLOAT16 (the common case for an exported inference-ready
# LLM/CNN graph) -- see ``onnxsim/pruning.py``'s own "FP16/BFloat16 weight
# support" section comment for the read-upcast/write-downcast pattern that
# now handles both. The tests below independently verify, for at least one
# representative function of every major algorithm family this module
# offers (magnitude, Wanda, SparseGPT, structured/channel, attention-head,
# MoE):
#
#   1. FLOAT16 execution correctness against a *real* onnxruntime CPU
#      session (confirmed, separately from this test suite, that
#      onnxruntime's CPU provider actually executes MatMul/Conv/Attention/
#      MoE with genuine FLOAT16 weights -- not just tolerates the dtype
#      tag) -- not merely a numpy-side oracle, since a bug could plausibly
#      pass a same-process numpy check while still producing an
#      unrunnable or wrongly-typed graph.
#   2. The pruned model's tensors keep their original FLOAT16 dtype
#      (checked via ``TensorProto.data_type``, not just value), rather
#      than silently upcasting to float32.
#   3. For every pass that only *masks or slices* (never recomputes a
#      surviving value): every surviving entry reproduces the *exact*
#      original fp16 bit pattern (compared via ``.view(np.uint16)``, not
#      ``assert_allclose``) -- verifying the "upcast-to-float64-then-
#      downcast-with-no-intervening-arithmetic is bit-exact" claim
#      empirically, not merely asserting it. SparseGPT is the deliberate
#      exception: its Hessian-compensated update genuinely recomputes
#      every kept entry's own value, so its own test checks reconstruction
#      quality instead (mirroring
#      ``test_sparsegpt_pruning_reconstructs_better_than_a_same_mask_style_baseline``),
#      not bit-exactness.
#
# BFLOAT16 has no onnxruntime CPU execution support in this environment at
# all -- confirmed separately: a plain BFLOAT16 MatMul model raises
# ``NOT_IMPLEMENTED`` ("Could not find an implementation for MatMul(13)
# node...") the moment a session is created, for every op type tried, not
# just this module's own output. BFLOAT16 tests below therefore check
# correctness at the array level (dtype preservation, exact per-element
# decode via ``ml_dtypes.bfloat16``, matching the same masking/slicing
# math every other test in this module already verifies numpy-side) rather
# than via a real session run -- an honest adjustment forced by that
# environment fact, not a shortcut taken for convenience.
#
# Per CLAUDE.md's own guidance on when to fall back off the ``onnx.parser``
# text format: the text format has no fp16/bfloat16 tensor-literal syntax
# (confirmed: ``<float16 W = {1.5}>`` raises a ``ParseError``), though it
# *does* support ``float16[...]``/``bfloat16[...]`` in a graph's own
# input/output type signature. So every test below still builds its
# graph/signature via ``_model``'s own ``onnx.parser`` convention, and only
# attaches the fp16/bf16 initializer tensors themselves programmatically,
# via ``onnx.numpy_helper.from_array`` on an already-fp16/bf16 numpy array
# (``_f16``/``_bf16`` below).


def _f16(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float16), name)


def _bf16(array, name):
    return onnx.numpy_helper.from_array(array.astype(ml_dtypes.bfloat16), name)


def test_magnitude_pruning_fp16_matches_ort_execution_and_preserves_exact_bits():
    # magnitude pruning family, representative function: apply_magnitude_pruning.
    K, N = 16, 8
    rng = np.random.default_rng(101)
    w = (rng.standard_normal((K, N)) * 0.5).astype(np.float16)
    model = _model(
        f"""
        g (float16[batch,{K}] X) => (float16[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f16(w, "W")],
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_magnitude_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    w_init = pruned.graph.initializer[0]
    assert w_init.data_type == onnx.TensorProto.FLOAT16  # dtype preserved, not upcast
    w_pruned = onnx.numpy_helper.to_array(w_init)
    assert w_pruned.dtype == np.float16
    assert onnxsim.weight_sparsity(pruned) == pytest.approx(0.5, abs=1e-9)

    # Masking never recomputes a kept value -- only zeros dropped ones -- so
    # every surviving entry must reproduce the exact original fp16 bit
    # pattern (compared as raw uint16, not by value).
    survivors = w_pruned != 0
    np.testing.assert_array_equal(
        w_pruned[survivors].view(np.uint16), w[survivors].view(np.uint16)
    )

    rng2 = np.random.default_rng(102)
    x = rng2.standard_normal((3, K)).astype(np.float16)
    (y,) = _run(pruned, {"X": x})
    assert y.dtype == np.float16
    y_oracle = x.astype(np.float64) @ w_pruned.astype(np.float64)
    np.testing.assert_allclose(y.astype(np.float64), y_oracle, rtol=1e-2, atol=1e-2)


def test_magnitude_pruning_fp16_uses_genuine_decode_not_raw_float32_reinterpretation():
    # Adversarial check that this module's fp16 support is genuinely
    # decoding FLOAT16 storage (as onnx.numpy_helper does: each entry a
    # real IEEE-754 half-precision value) rather than, say, treating the
    # tensor's raw little-endian byte buffer as if it already held float32
    # values (4 bytes/entry) -- a plausible implementation bug this test is
    # specifically built to catch, since it wouldn't crash: it would
    # silently produce a *different*, still-numeric result (half as many
    # "entries" per row, each an arbitrary float32 bit-pattern unrelated to
    # the true fp16 values) and therefore a detectably wrong pruning
    # decision.
    K, N = 4, 4
    # apply_magnitude_pruning ranks per *output channel* -- each row of
    # W^T ([N, K], _weight_to_nk's own convention), i.e. each *column* of
    # this [K, N] MatMul weight -- so every column here is given the same
    # strictly-decreasing-by-row-index true fp16 |W| order (8, 4, 2, 1) top
    # to bottom, keeping row indices 0/1 (8, 4) and dropping 2/3 (2, 1) for
    # every output channel alike.
    # A slight per-column offset (8.0/8.1/8.2/8.3, ...) keeps every column's
    # own row-wise magnitude order strictly decreasing (needed for a clean
    # "every output channel keeps the same two rows" assertion below) while
    # avoiding the repeated-identical-value degenerate case, where pairing
    # up two copies of the same fp16 bit pattern as one float32 could
    # coincidentally land back near a true value.
    w16 = np.array(
        [
            [8.0, 8.1, 8.2, 8.3],
            [4.0, 4.1, 4.2, 4.3],
            [2.0, 2.1, 2.2, 2.3],
            [1.0, 1.1, 1.2, 1.3],
        ],
        dtype=np.float16,
    )
    # Confirm the "wrong decode" this test guards against really is garbage
    # relative to the true values (magnitude range ~1.0-8.3) -- not a
    # coincidentally similar reinterpretation this test would fail to
    # actually distinguish. Checked via overall magnitude *range* (a
    # single stray close value doesn't itself prove a genuinely wrong
    # decode; a several-orders-of-magnitude-wider spread does) rather than
    # any single value, which is independently verified once, empirically,
    # before writing this test, to span from ~0.014 to ~170272 -- roughly
    # seven orders of magnitude wider than the true ~1.0-8.3 range.
    wrong_as_float32 = np.frombuffer(w16.tobytes(), dtype="<f4")
    true_range = w16.astype(np.float64).max() / w16.astype(np.float64).min()
    wrong_range = np.abs(wrong_as_float32).max() / np.abs(wrong_as_float32).min()
    assert wrong_range > true_range * 1000

    model = _model(
        f"""
        g (float16[batch,{K}] X) => (float16[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f16(w16, "W")],
    )
    pruned = onnxsim.apply_magnitude_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    w_pruned = onnx.numpy_helper.to_array(pruned.graph.initializer[0])
    assert w_pruned.dtype == np.float16

    # Correct fp16 decode keeps rows 0/1 (values 8, 4) entirely nonzero and
    # rows 2/3 (values 2, 1) entirely zero, for every output channel. A
    # "wrong dtype" bug reading garbage values would not reliably produce
    # this exact, uniform-across-columns pattern.
    np.testing.assert_array_equal(w_pruned[0, :], w16[0, :])
    np.testing.assert_array_equal(w_pruned[1, :], w16[1, :])
    np.testing.assert_array_equal(w_pruned[2, :], np.zeros(N, dtype=np.float16))
    np.testing.assert_array_equal(w_pruned[3, :], np.zeros(N, dtype=np.float16))

    x = np.ones((1, K), dtype=np.float16)
    (y,) = _run(pruned, {"X": x})
    y_oracle = (x.astype(np.float64) @ w_pruned.astype(np.float64)).astype(np.float64)
    np.testing.assert_allclose(y.astype(np.float64), y_oracle, rtol=1e-3, atol=1e-3)


def test_wanda_pruning_fp16_protects_high_activation_channels_matches_ort():
    # Wanda family, representative function: apply_wanda_pruning. fp16
    # analogue of test_wanda_pruning_protects_high_activation_channels,
    # with real onnxruntime execution as the correctness oracle.
    K, N = 32, 8
    salient = (2, 5, 20)
    rng = np.random.default_rng(103)
    w = (rng.standard_normal((K, N)) * 0.5).astype(np.float16)
    model = _model(
        f"""
        g (float16[batch,{K}] X) => (float16[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f16(w, "W")],
    )
    onnx.checker.check_model(model)

    x = rng.standard_normal((32, K)).astype(np.float16)
    for c in salient:
        x[:, c] *= 20.0
    calibration_data = [{"X": x}]

    magnitude_pruned = onnxsim.apply_magnitude_pruning(model, sparsity=0.5)
    wanda_pruned = onnxsim.apply_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    onnx.checker.check_model(wanda_pruned)
    assert wanda_pruned.graph.initializer[0].data_type == onnx.TensorProto.FLOAT16
    assert onnxsim.weight_sparsity(wanda_pruned) == pytest.approx(0.5, abs=1e-9)

    w_magnitude = onnx.numpy_helper.to_array(magnitude_pruned.graph.initializer[0])
    w_wanda = onnx.numpy_helper.to_array(wanda_pruned.graph.initializer[0])
    salient_kept_magnitude = np.count_nonzero(w_magnitude[list(salient), :])
    salient_kept_wanda = np.count_nonzero(w_wanda[list(salient), :])
    assert salient_kept_wanda > salient_kept_magnitude

    (float_y,) = _run(model, {"X": x})
    (magnitude_y,) = _run(magnitude_pruned, {"X": x})
    (wanda_y,) = _run(wanda_pruned, {"X": x})
    magnitude_err = np.linalg.norm(
        float_y.astype(np.float64) - magnitude_y.astype(np.float64)
    )
    wanda_err = np.linalg.norm(float_y.astype(np.float64) - wanda_y.astype(np.float64))
    assert wanda_err < magnitude_err


def test_sparsegpt_pruning_fp16_reconstructs_better_than_a_same_mask_style_baseline():
    # SparseGPT family, representative function: apply_sparsegpt_pruning.
    # fp16 analogue of
    # test_sparsegpt_pruning_reconstructs_better_than_a_same_mask_style_baseline,
    # checked both numpy-side and via real onnxruntime execution. SparseGPT
    # genuinely recomputes every kept entry's value (Hessian-compensated),
    # so -- unlike magnitude/Wanda above -- this does not check bit-exact
    # surviving entries, only that the technique still improves
    # reconstruction quality over naive same-mask zeroing once weights are
    # fp16.
    K, N = 48, 12
    rng = np.random.default_rng(104)
    w = (rng.standard_normal((K, N)) * 0.5).astype(np.float16)
    model = _model(
        f"""
        g (float16[batch,{K}] X) => (float16[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f16(w, "W")],
    )
    onnx.checker.check_model(model)
    x_cal = (rng.standard_normal((512, K)) * 0.3).astype(np.float16)  # well-cond. H

    pruned = onnxsim.apply_sparsegpt_pruning(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5
    )
    onnx.checker.check_model(pruned)
    w_init = pruned.graph.initializer[0]
    assert w_init.data_type == onnx.TensorProto.FLOAT16
    w_sparsegpt = onnx.numpy_helper.to_array(w_init).astype(np.float64)

    w64 = w.astype(np.float64)
    score = np.abs(w64)
    thresh = np.sort(score.flatten())[int(score.size * 0.5)]
    w_naive = np.where(score <= thresh, 0.0, w64)

    x64 = x_cal.astype(np.float64)
    y_orig = x64 @ w64
    err_sparsegpt = np.sum((y_orig - x64 @ w_sparsegpt) ** 2)
    err_naive = np.sum((y_orig - x64 @ w_naive) ** 2)
    assert err_sparsegpt <= err_naive

    # And a real onnxruntime run of the pruned (fp16) model must still
    # reconstruct the original layer's output at least as well as the
    # naive same-mask baseline, evaluated through the same fp16 runtime
    # execution path a real deployment would use.
    naive_model = _model(
        f"""
        g (float16[batch,{K}] X) => (float16[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f16(w_naive.astype(np.float32), "W")],
    )
    (y_pruned_ort,) = _run(pruned, {"X": x_cal})
    (y_naive_ort,) = _run(naive_model, {"X": x_cal})
    (y_orig_ort,) = _run(model, {"X": x_cal})
    err_sparsegpt_ort = np.sum(
        (y_orig_ort.astype(np.float64) - y_pruned_ort.astype(np.float64)) ** 2
    )
    err_naive_ort = np.sum(
        (y_orig_ort.astype(np.float64) - y_naive_ort.astype(np.float64)) ** 2
    )
    assert err_sparsegpt_ort <= err_naive_ort * 1.05  # small fp16-rounding slack


def test_structured_pruning_fp16_matmul_chain_matches_ort_oracle():
    # structured/channel family, representative function:
    # apply_structured_pruning. fp16 analogue of
    # test_structured_pruning_matmul_only_chain_matches_oracle.
    K, H, Out = 8, 24, 4
    rng = np.random.default_rng(105)
    w1 = (rng.standard_normal((K, H)) * 0.4).astype(np.float16)
    w2 = (rng.standard_normal((H, Out)) * 0.4).astype(np.float16)
    model = _model(
        f"""
        g (float16[batch,{K}] X) => (float16[batch,{Out}] Y)
        {{
          h = MatMul(X, W1)
          a = Sigmoid(h)
          Y = MatMul(a, W2)
        }}
        """,
        initializer=[_f16(w1, "W1"), _f16(w2, "W2")],
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.25)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert inits["W1"].data_type == onnx.TensorProto.FLOAT16
    assert inits["W2"].data_type == onnx.TensorProto.FLOAT16
    assert list(inits["W1"].dims) == [K, H - round(H * 0.25)]

    keep = _oracle_keep_indices(w1.astype(np.float64), H - round(H * 0.25))

    rng2 = np.random.default_rng(106)
    x = rng2.standard_normal((5, K)).astype(np.float16)
    (y,) = _run(pruned, {"X": x})
    assert y.dtype == np.float16

    h = x.astype(np.float64) @ w1.astype(np.float64)[:, keep]
    a = 1.0 / (1.0 + np.exp(-h))
    y_oracle = a @ w2.astype(np.float64)[keep, :]
    np.testing.assert_allclose(y.astype(np.float64), y_oracle, rtol=5e-2, atol=5e-2)


def test_attention_head_pruning_fp16_matches_manual_head_deletion_oracle():
    # attention-head family, representative function:
    # apply_attention_head_pruning. fp16 analogue of
    # test_attention_head_pruning_matches_manual_head_deletion_exactly.
    K, H, D, Out, batch, seq = 8, 4, 4, 6, 2, 5
    Nq = Nk = Nv = H * D
    rng = np.random.default_rng(107)
    wqkv = (rng.standard_normal((K, Nq + Nk + Nv)) * 0.3).astype(np.float16)
    bqkv = (rng.standard_normal((Nq + Nk + Nv,)) * 0.3).astype(np.float16)
    wout = (rng.standard_normal((Nv, Out)) * 0.3).astype(np.float16)

    def _fp16_attention_model(h, wqkv_, bqkv_, wout_):
        m = parser.parse_model(
            f"""
            <
              ir_version: 10,
              opset_import: ["": 17, "com.microsoft": 1]
            >
            g (float16[batch,seq,{K}] X) => (float16[batch,seq,{Out}] Y)
            {{
              ctx = com.microsoft.Attention <num_heads={h}, qkv_hidden_sizes=[{wqkv_.shape[1] // 3},{wqkv_.shape[1] // 3},{wqkv_.shape[1] // 3}]> (X, Wqkv, Bqkv)
              Y = MatMul(ctx, Wout)
            }}
            """
        )
        m.graph.initializer.extend(
            [_f16(wqkv_, "Wqkv"), _f16(bqkv_, "Bqkv"), _f16(wout_, "Wout")]
        )
        return m

    model = _fp16_attention_model(H, wqkv, bqkv, wout)
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    pruned_inits = {t.name: t for t in pruned.graph.initializer}
    assert pruned_inits["Wqkv"].data_type == onnx.TensorProto.FLOAT16
    node = next(n for n in pruned.graph.node if n.op_type == "Attention")
    num_heads = next(a.i for a in node.attribute if a.name == "num_heads")
    assert num_heads == 2

    keep = _oracle_keep_heads(
        wqkv.astype(np.float64), Nq, Nk, Nv, H, 2
    )  # d64-precision ranking, mirroring the float32 oracle's own convention
    qi, ki, vi = (
        _head_idx(keep, D),
        _head_idx(keep, D) + Nq,
        _head_idx(keep, D) + Nq + Nk,
    )
    all_idx = np.concatenate([qi, ki, vi])
    oracle_wqkv = wqkv[:, all_idx]
    oracle_bqkv = bqkv[all_idx]
    oracle_wout = wout[_head_idx(keep, D), :]
    oracle = _fp16_attention_model(2, oracle_wqkv, oracle_bqkv, oracle_wout)

    rng2 = np.random.default_rng(108)
    x = rng2.standard_normal((batch, seq, K)).astype(np.float16)
    (y_pruned,) = _run(pruned, {"X": x})
    (y_oracle,) = _run(oracle, {"X": x})
    assert y_pruned.dtype == np.float16
    np.testing.assert_allclose(
        y_pruned.astype(np.float64), y_oracle.astype(np.float64), rtol=5e-2, atol=5e-2
    )


def test_moe_expert_channel_pruning_fp16_matches_ort_masking_oracle():
    # MoE family, representative function: apply_moe_expert_channel_pruning.
    # fp16 analogue of test_moe_expert_channel_pruning_matches_ort_masking_oracle.
    E, hidden, inter, tokens, k = 5, 10, 12, 6, 2
    rng = np.random.default_rng(109)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float16)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float16)

    model = _model(
        f"""
        g (float16[{tokens},{hidden}] X, float16[{tokens},{E}] R) => (float16[{tokens},{hidden}] Y)
        {{
          Y = com.microsoft.MoE <k={k}, activation_type="relu"> (X, R, FC1W, , FC2W)
        }}
        """,
        initializer=[_f16(fc1_w, "FC1W"), _f16(fc2_w, "FC2W")],
        opset=18,
    )
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_moe_expert_channel_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert inits["FC1W"].data_type == onnx.TensorProto.FLOAT16
    assert list(inits["FC1W"].dims) == [E, 6, hidden]

    fc1_w64 = fc1_w.astype(np.float64)
    fc2_w64 = fc2_w.astype(np.float64)
    sq = np.sum(fc1_w64**2, axis=(0, 2)) + np.sum(fc2_w64**2, axis=(0, 1))
    keep = np.sort(np.argsort(-np.sqrt(sq))[:6])

    fc1_w_pruned = onnx.numpy_helper.to_array(inits["FC1W"])
    fc2_w_pruned = onnx.numpy_helper.to_array(inits["FC2W"])
    np.testing.assert_array_equal(
        fc1_w_pruned.view(np.uint16), fc1_w[:, keep, :].view(np.uint16)
    )
    np.testing.assert_array_equal(
        fc2_w_pruned.view(np.uint16), fc2_w[:, :, keep].view(np.uint16)
    )

    rng2 = np.random.default_rng(110)
    x = rng2.standard_normal((tokens, hidden)).astype(np.float16)
    r = rng2.standard_normal((tokens, E)).astype(np.float16)
    (y_pruned,) = _run(pruned, {"X": x, "R": r})
    assert y_pruned.dtype == np.float16

    # Same-shape ORT masking oracle: zero the dropped inter_size channels
    # instead of physically removing them -- must be numerically identical
    # for relu (dropped fc1 row and bias both zero => pre-activation 0 =>
    # relu(0) == 0, contributing nothing through fc2's own zeroed column).
    drop = np.setdiff1d(np.arange(inter), keep)
    fc1_masked = fc1_w64.copy()
    fc1_masked[:, drop, :] = 0.0
    fc2_masked = fc2_w64.copy()
    fc2_masked[:, :, drop] = 0.0
    masked_model = _model(
        f"""
        g (float16[{tokens},{hidden}] X, float16[{tokens},{E}] R) => (float16[{tokens},{hidden}] Y)
        {{
          Y = com.microsoft.MoE <k={k}, activation_type="relu"> (X, R, FC1W, , FC2W)
        }}
        """,
        initializer=[
            _f16(fc1_masked.astype(np.float32), "FC1W"),
            _f16(fc2_masked.astype(np.float32), "FC2W"),
        ],
        opset=18,
    )
    masked_model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))
    (y_masked,) = _run(masked_model, {"X": x, "R": r})
    np.testing.assert_allclose(
        y_pruned.astype(np.float64), y_masked.astype(np.float64), rtol=5e-2, atol=5e-2
    )


def test_magnitude_pruning_bfloat16_preserves_dtype_and_matches_array_oracle():
    # onnxruntime has no BFLOAT16 CPU execution support in this environment
    # (confirmed separately: a plain BFLOAT16 MatMul session raises
    # NOT_IMPLEMENTED at session-creation time) -- so this checks
    # correctness at the array level (dtype preservation, exact per-element
    # bfloat16 decode) rather than via a real session run, unlike the
    # FLOAT16 tests above.
    K, N = 16, 8
    rng = np.random.default_rng(111)
    w = (rng.standard_normal((K, N)) * 0.5).astype(ml_dtypes.bfloat16)
    model = _model(
        f"""
        g (bfloat16[batch,{K}] X) => (bfloat16[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_bf16(w, "W")],
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_magnitude_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    w_init = pruned.graph.initializer[0]
    assert w_init.data_type == onnx.TensorProto.BFLOAT16
    w_pruned = onnx.numpy_helper.to_array(w_init)
    assert w_pruned.dtype == ml_dtypes.bfloat16
    assert onnxsim.weight_sparsity(pruned) == pytest.approx(0.5, abs=1e-9)

    survivors = w_pruned != ml_dtypes.bfloat16(0.0)
    np.testing.assert_array_equal(
        w_pruned[survivors].view(np.uint16), w[survivors].view(np.uint16)
    )

    # Every output column's surviving entries must be exactly the
    # top-(1 - sparsity) fraction by magnitude, computed in float64 off the
    # correctly-decoded bfloat16 values -- the same check
    # test_magnitude_pruning_keeps_the_largest_entries_per_row makes for
    # float32.
    w64 = w.astype(np.float64)
    w_pruned64 = w_pruned.astype(np.float64)
    for col in range(N):
        kept = np.flatnonzero(w_pruned64[:, col] != 0)
        assert len(kept) == 8  # round(16 * 0.5)
        threshold = np.abs(w64[:, col])[kept].min()
        dropped_max = np.abs(w64[:, col])[np.flatnonzero(w_pruned64[:, col] == 0)].max()
        assert dropped_max <= threshold


def test_structured_pruning_bfloat16_matches_array_oracle():
    # structured/channel family, BFLOAT16: confirms the matcher/slicer path
    # (not just the unstructured masking path above) round-trips BFLOAT16
    # correctly. No onnxruntime execution -- see the module comment above
    # for why BFLOAT16 has no CPU kernel support here.
    K, H, Out = 8, 16, 4
    rng = np.random.default_rng(112)
    w1 = (rng.standard_normal((K, H)) * 0.4).astype(ml_dtypes.bfloat16)
    w2 = (rng.standard_normal((H, Out)) * 0.4).astype(ml_dtypes.bfloat16)
    model = _model(
        f"""
        g (bfloat16[batch,{K}] X) => (bfloat16[batch,{Out}] Y)
        {{
          h = MatMul(X, W1)
          a = Relu(h)
          Y = MatMul(a, W2)
        }}
        """,
        initializer=[_bf16(w1, "W1"), _bf16(w2, "W2")],
    )
    onnx.checker.check_model(model)

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.25)
    onnx.checker.check_model(pruned)
    inits = {t.name: t for t in pruned.graph.initializer}
    assert inits["W1"].data_type == onnx.TensorProto.BFLOAT16
    assert inits["W2"].data_type == onnx.TensorProto.BFLOAT16

    keep = _oracle_keep_indices(w1.astype(np.float64), H - round(H * 0.25))
    w1_pruned = onnx.numpy_helper.to_array(inits["W1"])
    w2_pruned = onnx.numpy_helper.to_array(inits["W2"])
    # Pure slicing -- exact bfloat16 bit-for-bit match against the manually
    # sliced original array, not just a numeric closeness check.
    np.testing.assert_array_equal(
        w1_pruned.view(np.uint16), w1[:, keep].view(np.uint16)
    )
    np.testing.assert_array_equal(
        w2_pruned.view(np.uint16), w2[keep, :].view(np.uint16)
    )


# --- analyze_pruning_sensitivity (dry-run sensitivity report) -------------
#
# The correctness bar here is different from the rest of this file: rather
# than an independent from-scratch oracle, the report's own predicted
# would-drop counts and matched chains are checked directly against what
# the corresponding *real*, mutating `apply_*` call actually does when
# given the exact same arguments -- see this section's own
# ``_report_matches_real_call`` docstring for why that's a strong,
# mechanical check despite not being a second independent implementation.


def _initializer_dims(model):
    return {t.name: tuple(t.dims) for t in model.graph.initializer}


def test_analyze_pruning_sensitivity_never_mutates_input_model():
    # The entire point of this tool: building the report must not touch
    # `model` at all, for every supported family -- checked here via exact
    # byte-for-byte serialization equality, not just "the values I later
    # read still look right".
    mlp, _, _, _ = _swiglu_mlp_model(K=8, H=16, Out=4)
    attention, _ = _attention_model(K=8, H=4, D=4, Out=6)
    gqa, _ = _gqa_model()
    x = np.random.default_rng(0).standard_normal((4, 8)).astype(np.float32)
    calibration_data = [{"X": x}]

    for model, apply_fn, kwargs in [
        (_matmul_model(K=64, N=16), onnxsim.apply_magnitude_pruning, {}),
        (
            _matmul_model(K=64, N=16),
            onnxsim.apply_wanda_pruning,
            {"calibration_data": [{"X": np.zeros((1, 64), dtype=np.float32)}]},
        ),
        (mlp, onnxsim.apply_structured_pruning, {}),
        (
            mlp,
            onnxsim.apply_structured_wanda_pruning,
            {"calibration_data": calibration_data},
        ),
        (attention, onnxsim.apply_attention_head_pruning, {}),
        (gqa, onnxsim.apply_attention_head_pruning, {}),
    ]:
        before_bytes = model.SerializeToString()
        report = onnxsim.analyze_pruning_sensitivity(
            model, apply_fn, sparsity=0.5, **kwargs
        )
        assert model.SerializeToString() == before_bytes, (
            f"{apply_fn.__name__} dry run mutated the input model"
        )
        assert isinstance(report, onnxsim.PruningSensitivityReport)


def test_analyze_magnitude_pruning_matches_real_call():
    K, N = 64, 16
    model = _matmul_model(K=K, N=N)
    report = onnxsim.analyze_pruning_sensitivity(
        model, onnxsim.apply_magnitude_pruning, sparsity=0.5
    )
    assert len(report.layers) == 1
    layer = report.layers[0]
    assert layer.family == "matmul"
    assert layer.total == K * N
    assert report.not_eligible == []

    pruned = onnxsim.apply_magnitude_pruning(model, sparsity=0.5)
    w = onnx.numpy_helper.to_array(pruned.graph.initializer[0])
    assert layer.would_drop == int((w == 0).sum())
    assert layer.would_drop == pytest.approx(K * N * 0.5, abs=N)  # per-row rounding


def test_analyze_magnitude_pruning_nm_pattern_matches_real_call():
    K, N = 32, 8
    model = _matmul_model(K=K, N=N)
    report = onnxsim.analyze_pruning_sensitivity(
        model, onnxsim.apply_magnitude_pruning, n=2, m=4
    )
    pruned = onnxsim.apply_magnitude_pruning(model, n=2, m=4)
    w = onnx.numpy_helper.to_array(pruned.graph.initializer[0])
    assert report.layers[0].would_drop == int((w == 0).sum())
    assert report.layers[0].would_drop == K * N // 2  # exactly half for 2:4


def test_analyze_magnitude_pruning_global_sparsity_matches_real_call():
    model, _, _ = _two_scale_matmul_model()
    report = onnxsim.analyze_pruning_sensitivity(
        model,
        onnxsim.apply_magnitude_pruning,
        sparsity=0.5,
        global_sparsity=True,
    )
    pruned = onnxsim.apply_magnitude_pruning(model, sparsity=0.5, global_sparsity=True)
    inits = {t.name: t for t in pruned.graph.initializer}
    # `report.layers`' own `label` is each producer node's own output name
    # (MatMul nodes here are unnamed in the parsed text) -- "Y1"/"Y2" --
    # not the weight initializer's own name ("Wbig"/"Wsmall"); map between
    # them via each node's own single weight input.
    weight_by_output = {node.output[0]: node.input[1] for node in model.graph.node}
    by_weight = {weight_by_output[layer.label]: layer for layer in report.layers}
    assert set(by_weight) == set(inits)
    for name, init in inits.items():
        w = onnx.numpy_helper.to_array(init)
        assert by_weight[name].would_drop == int((w == 0).sum())


def test_analyze_wanda_pruning_matches_real_call():
    K, N = 64, 16
    salient = (3, 7, 40)
    rng = np.random.default_rng(0)
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f32(weight, "W")],
    )
    x = rng.standard_normal((32, K)).astype(np.float32)
    for c in salient:
        x[:, c] *= 20.0
    calibration_data = [{"X": x}]

    report = onnxsim.analyze_pruning_sensitivity(
        model,
        onnxsim.apply_wanda_pruning,
        calibration_data=calibration_data,
        sparsity=0.5,
    )
    pruned = onnxsim.apply_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    w = onnx.numpy_helper.to_array(pruned.graph.initializer[0])
    assert report.layers[0].would_drop == int((w == 0).sum())
    assert report.layers[0].family == "matmul"


def test_analyze_structured_pruning_plain_chain_matches_real_call():
    K, H, Out = 8, 32, 4
    model = _mlp_model(K=K, H=H, Out=Out)
    report = onnxsim.analyze_pruning_sensitivity(
        model, onnxsim.apply_structured_pruning, sparsity=0.5
    )
    assert len(report.layers) == 1
    layer = report.layers[0]
    assert layer.family == "matmul_plain"
    assert layer.total == H
    assert report.not_eligible == []

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    dims_after = _initializer_dims(pruned)
    assert dims_after["W1"][1] == H - layer.would_drop
    assert dims_after["W2"][0] == H - layer.would_drop


def test_analyze_structured_pruning_gated_pair_matches_real_call():
    K, H, Out = 8, 16, 4
    model, wg, wu, wd = _swiglu_mlp_model(K=K, H=H, Out=Out)
    report = onnxsim.analyze_pruning_sensitivity(
        model, onnxsim.apply_structured_pruning, sparsity=0.5
    )
    assert len(report.layers) == 1
    layer = report.layers[0]
    assert layer.family == "matmul_gated"
    assert layer.total == H

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    dims_after = _initializer_dims(pruned)
    kept = H - layer.would_drop
    assert dims_after["Wg"][1] == kept
    assert dims_after["Wu"][1] == kept
    assert dims_after["Wd"][0] == kept


def test_analyze_structured_pruning_grouped_conv_matches_real_call():
    Cin, C1, C2, group1 = 4, 8, 6, 2
    rng = np.random.default_rng(0)
    w1 = rng.standard_normal((C1, Cin // group1, 3, 3)).astype(np.float32)
    w2 = rng.standard_normal((C2, C1, 3, 3)).astype(np.float32)
    model = _grouped_conv_pair_model(w1, w2, group1=group1, group2=1)

    report = onnxsim.analyze_pruning_sensitivity(
        model, onnxsim.apply_structured_pruning, sparsity=0.5
    )
    assert len(report.layers) == 1
    layer = report.layers[0]
    assert layer.family == "conv_plain"
    assert layer.total == C1

    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    dims_after = _initializer_dims(pruned)
    kept = C1 - layer.would_drop
    assert dims_after["W1"][0] == kept
    assert dims_after["W2"][1] == kept


def test_analyze_structured_wanda_pruning_matches_real_call():
    K, H, Out = 8, 32, 4
    model = _mlp_model(K=K, H=H, Out=Out, bias=False)
    rng = np.random.default_rng(1)
    x = rng.standard_normal((16, K)).astype(np.float32)
    calibration_data = [{"X": x}]

    report = onnxsim.analyze_pruning_sensitivity(
        model,
        onnxsim.apply_structured_wanda_pruning,
        calibration_data=calibration_data,
        sparsity=0.5,
    )
    assert len(report.layers) == 1
    layer = report.layers[0]
    assert layer.family == "matmul_plain"

    pruned = onnxsim.apply_structured_wanda_pruning(
        model, calibration_data=calibration_data, sparsity=0.5
    )
    dims_after = _initializer_dims(pruned)
    kept = H - layer.would_drop
    assert dims_after["W1"][1] == kept
    assert dims_after["W2"][0] == kept


def test_analyze_attention_head_pruning_plain_attention_matches_real_call():
    K, H, D, Out = 8, 4, 4, 6
    model, info = _attention_model(K=K, H=H, D=D, Out=Out)
    report = onnxsim.analyze_pruning_sensitivity(
        model, onnxsim.apply_attention_head_pruning, sparsity=0.5
    )
    assert len(report.layers) == 1
    layer = report.layers[0]
    assert layer.family == "attention_head"
    assert layer.total == H
    assert report.not_eligible == []

    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    dims_after = _initializer_dims(pruned)
    kept = H - layer.would_drop
    assert dims_after["Wqkv"][1] == kept * 3 * D  # Nq==Nk==Nv==H*D here


def test_analyze_attention_head_pruning_gqa_matches_real_call():
    H, KVH, D = 4, 2, 8
    model, info = _gqa_model(H=H, KVH=KVH, D=D)
    report = onnxsim.analyze_pruning_sensitivity(
        model, onnxsim.apply_attention_head_pruning, sparsity=0.5
    )
    assert len(report.layers) == 1
    layer = report.layers[0]
    assert layer.family == "attention_gqa_group"
    assert layer.total == KVH

    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    dims_after = _initializer_dims(pruned)
    kept_groups = KVH - layer.would_drop
    group_size = H // KVH
    assert dims_after["Wq"][1] == kept_groups * group_size * D
    assert dims_after["Wk"][1] == kept_groups * D
    assert dims_after["Wv"][1] == kept_groups * D


def test_analyze_pruning_sensitivity_not_eligible_lists_unmatched_nodes():
    # A second MatMul whose weight is a graph input, not a constant
    # initializer, can never be matched by any of this module's own
    # structured-pruning finders -- it must show up in `not_eligible`, not
    # silently be dropped from the report entirely.
    K, H, Out = 8, 16, 4
    mlp = _mlp_model(K=K, H=H, Out=Out, bias=False)
    model = _model(
        f"""
        g (float[batch,{K}] X, float[{Out},{Out}] Wdyn) => (float[batch,{Out}] Z)
        {{
          h = MatMul(X, W1)
          a = Relu(h)
          Y = MatMul(a, W2)
          Z = MatMul(Y, Wdyn)
        }}
        """,
        initializer=list(mlp.graph.initializer),
    )
    report = onnxsim.analyze_pruning_sensitivity(
        model, onnxsim.apply_structured_pruning, sparsity=0.5
    )
    assert len(report.layers) == 1  # only the W1->W2 chain is matched
    assert report.not_eligible == ["MatMul 'Z'"]
    matched_labels = {layer.label for layer in report.layers}
    assert "Z" not in matched_labels


def test_analyze_pruning_sensitivity_rejects_unsupported_apply_fn():
    model = _matmul_model(K=16, N=8)
    with pytest.raises(ValueError, match="apply_sparsegpt_pruning"):
        onnxsim.analyze_pruning_sensitivity(
            model, onnxsim.apply_sparsegpt_pruning, sparsity=0.5
        )


def test_analyze_structured_pruning_rejects_global_sparsity():
    model = _mlp_model(K=8, H=16, Out=4)
    with pytest.raises(NotImplementedError, match="global_sparsity"):
        onnxsim.analyze_pruning_sensitivity(
            model,
            onnxsim.apply_structured_pruning,
            sparsity=0.5,
            global_sparsity=True,
        )


def test_analyze_structured_pruning_rejects_concat_merged_models():
    rng = np.random.default_rng(0)
    weights = [
        rng.standard_normal((8, 3)).astype(np.float32),
        rng.standard_normal((8, 5)).astype(np.float32),
    ]
    w_out = rng.standard_normal((8, 4)).astype(np.float32)
    model = _matmul_concat_model(weights, w_out)
    with pytest.raises(NotImplementedError, match="Concat"):
        onnxsim.analyze_pruning_sensitivity(
            model, onnxsim.apply_structured_pruning, sparsity=0.5
        )


def test_analyze_pruning_sensitivity_margin_reflects_importance_distribution():
    # Adversarial check that `margin` is a real, distribution-sensitive
    # signal, not a constant regardless of what's actually being ranked:
    # layer A has a huge, unambiguous gap between its "big" and "small"
    # column blocks (margin should be strongly positive, near the top of
    # its possible range); layer B's entries are all *exactly* equal (no
    # meaningful cut exists at all -- margin must come out exactly 0.0).
    # A normalization bug that always reports ~0, or always reports the
    # same value regardless of the underlying weights, would fail this.
    # `_sparsity_mask` ranks each output channel's own row of `w_nk`
    # ([N, K], `_weight_to_nk`'s own convention) independently -- so the
    # big/small split has to be *within* every row (along `K`), not across
    # rows, for a plain (non-transposed) MatMul weight of shape [K, N]:
    # `w_nk == W.T`, so `W` itself is built as this test's own desired
    # `w_nk` array, transposed.
    N, K = 4, 8
    big = np.full((N, K // 2), 10.0, dtype=np.float32)
    small = np.full((N, K // 2), 1e-3, dtype=np.float32)
    w_nk_a = np.concatenate([big, small], axis=1)  # [N, K]: every row big|small
    w_nk_b = np.full((N, K), 3.0, dtype=np.float32)  # [N, K]: every entry tied

    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Ya, float[batch,{N}] Yb)
        {{
          Ya = MatMul(X, Wa)
          Yb = MatMul(X, Wb)
        }}
        """,
        initializer=[_f32(w_nk_a.T, "Wa"), _f32(w_nk_b.T, "Wb")],
    )

    report = onnxsim.analyze_pruning_sensitivity(
        model, onnxsim.apply_magnitude_pruning, sparsity=0.5
    )
    by_label = {layer.label: layer for layer in report.layers}
    margin_a = by_label["Ya"].margin
    margin_b = by_label["Yb"].margin

    assert margin_b == 0.0  # every entry exactly tied -- no meaningful cut
    assert margin_a is not None and margin_a > 0.9  # wide, unambiguous gap
    assert margin_a != margin_b


# --- Embedding / lm_head vocabulary pruning --------------------------------
#
# See ``onnxsim/pruning.py``'s own "Embedding / lm_head vocabulary pruning"
# section comment for the full matching/safety bar and the id-remapping
# contract these tests exercise. Every positive-match test below runs the
# pruned model through a real onnxruntime session and compares against an
# independently-constructed oracle: the *original* (unpruned) model's own
# onnxruntime output, sliced down to the kept-vocabulary columns -- never a
# hand-computed expected array, and never just a shape/structural check.


def test_embedding_vocab_pruning_untied_matches_oracle_and_renumbers_contiguously():
    # (a) untied embedding + lm_head: both weights sliced to the same kept
    # set, model still executes correctly for every kept token id, and the
    # output column ordering matches the new contiguous renumbering.
    V, H = 12, 8
    rng = np.random.default_rng(1)
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

    keep_token_ids = [9, 1, 4, 7, 3, 11]  # deliberately unsorted, with dups below
    result = onnxsim.apply_embedding_vocab_pruning(
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


def test_embedding_vocab_pruning_untied_lm_head_gemm_bias_is_sliced():
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
    result = onnxsim.apply_embedding_vocab_pruning(model, keep_token_ids=keep_token_ids)
    onnx.checker.check_model(result.model)
    assert result.matched and result.lm_head_pruned

    input_ids = np.array([0, 5, 9, 2, 8], dtype=np.int64)
    remapped = np.array([result.id_map[i] for i in input_ids], dtype=np.int64)

    orig_out = _run(model, {"input_ids": input_ids})[0]
    pruned_out = _run(result.model, {"input_ids": remapped})[0]
    expected = orig_out[..., result.kept_token_ids]
    np.testing.assert_allclose(pruned_out, expected, atol=1e-5, rtol=1e-5)


def test_embedding_vocab_pruning_tied_via_transpose_matches_oracle():
    # (b) tied (weight-shared) embedding/lm_head, the "Transpose then
    # MatMul" sub-shape (the realistic pattern for a 3-D hidden-state
    # model, since Gemm has no batch-dim support): confirms the pass
    # recognizes the sharing and doesn't corrupt it -- the single shared
    # initializer is sliced exactly once, never independently twice.
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
    assert len(model.graph.initializer) == 1  # the one shared weight

    keep_token_ids = [1, 2, 4, 6, 8, 10, 11]
    result = onnxsim.apply_embedding_vocab_pruning(model, keep_token_ids=keep_token_ids)
    onnx.checker.check_model(result.model)

    assert result.matched
    assert result.lm_head_pruned
    # Still exactly one weight initializer -- the tied weight was sliced
    # once, not duplicated into two independently-sliced copies.
    assert len(result.model.graph.initializer) == 1
    emb_init = result.model.graph.initializer[0]
    assert list(emb_init.dims) == [len(keep_token_ids), H]

    input_ids = np.array([[1, 4, 8], [10, 2, 11]], dtype=np.int64)
    remapped = np.vectorize(result.id_map.get)(input_ids).astype(np.int64)

    orig_out = _run(model, {"input_ids": input_ids})[0]
    pruned_out = _run(result.model, {"input_ids": remapped})[0]
    expected = orig_out[..., result.kept_token_ids]
    np.testing.assert_allclose(pruned_out, expected, atol=1e-5, rtol=1e-5)


def test_embedding_vocab_pruning_tied_direct_gemm_matches_oracle():
    # The other tied sub-shape: a direct Gemm(transB=1) reusing the
    # embedding table as its own [vocab, hidden] weight with no Transpose
    # node at all (e.g. GPT-2's own ONNX export pattern).
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
    result = onnxsim.apply_embedding_vocab_pruning(model, drop_token_ids=[2, 4, 7])
    onnx.checker.check_model(result.model)
    assert result.matched and result.lm_head_pruned
    assert result.kept_token_ids == keep_token_ids
    assert len(result.model.graph.initializer) == 1

    input_ids = np.array([0, 3, 6, 8, 1], dtype=np.int64)
    remapped = np.array([result.id_map[i] for i in input_ids], dtype=np.int64)
    orig_out = _run(model, {"input_ids": input_ids})[0]
    pruned_out = _run(result.model, {"input_ids": remapped})[0]
    expected = orig_out[..., result.kept_token_ids]
    np.testing.assert_allclose(pruned_out, expected, atol=1e-5, rtol=1e-5)


def test_embedding_vocab_pruning_cast_hop_indices_still_matched():
    # A `Cast` between the raw graph input and `Gather`'s own `indices`
    # operand (a common real-export dtype-conversion hop) is the one
    # bounded pass-through this matcher allows.
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
    result = onnxsim.apply_embedding_vocab_pruning(model, keep_token_ids=keep_token_ids)
    onnx.checker.check_model(result.model)
    assert result.matched
    assert not result.lm_head_pruned  # no lm_head in this graph at all

    input_ids = np.array([0, 5, 2, 7], dtype=np.int32)
    remapped = np.array([result.id_map[i] for i in input_ids], dtype=np.int32)
    orig_out = _run(model, {"input_ids": input_ids})[0]
    pruned_out = _run(result.model, {"input_ids": remapped})[0]
    np.testing.assert_allclose(pruned_out, orig_out, atol=1e-6, rtol=1e-6)


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


def test_embedding_vocab_pruning_declines_when_gather_is_ambiguous():
    # (c) decline path: two structurally-identical Gather-embedding
    # candidates (a token embedding and a positional embedding, both
    # reading a genuine graph input) with no `input_name` to disambiguate
    # -- the whole call must decline, model left byte-for-byte untouched.
    model = _ambiguous_embedding_model()
    result = onnxsim.apply_embedding_vocab_pruning(model, keep_token_ids=[0, 1, 2])
    assert result.matched is False
    assert result.kept_token_ids is None
    assert result.id_map is None
    assert result.model.SerializeToString() == model.SerializeToString()


def test_embedding_vocab_pruning_input_name_disambiguates_correctly():
    model = _ambiguous_embedding_model(V_tok=10, V_pos=6)
    result = onnxsim.apply_embedding_vocab_pruning(
        model, drop_token_ids=[4, 5], input_name="input_ids"
    )
    assert result.matched
    assert result.kept_token_ids == [0, 1, 2, 3, 6, 7, 8, 9]
    # The positional-embedding table must be left completely untouched.
    w_pos = {t.name: t for t in result.model.graph.initializer}["W_pos"]
    assert list(w_pos.dims) == [6, 4]


def test_embedding_vocab_pruning_unknown_input_name_raises():
    model = _ambiguous_embedding_model()
    with pytest.raises(ValueError, match="not_a_real_input"):
        onnxsim.apply_embedding_vocab_pruning(
            model, keep_token_ids=[0, 1], input_name="not_a_real_input"
        )


def test_embedding_vocab_pruning_declines_non_zero_gather_axis():
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
    result = onnxsim.apply_embedding_vocab_pruning(model, keep_token_ids=[0, 1])
    assert result.matched is False
    assert result.model.SerializeToString() == model.SerializeToString()


def test_embedding_vocab_pruning_declines_unexpected_shared_consumer():
    # (c) decline path: the embedding weight is read by a second consumer
    # that isn't one of the two recognized tied `lm_head` shapes -- the
    # whole chain must decline rather than prune the embedding and
    # silently corrupt that unexplained second reader.
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
    result = onnxsim.apply_embedding_vocab_pruning(model, keep_token_ids=[0, 1, 2])
    assert result.matched is False
    assert result.model.SerializeToString() == model.SerializeToString()


def test_embedding_vocab_pruning_untied_lm_head_matmul_add_bias_hop_matches_oracle():
    # An untied MatMul lm_head whose output feeds exactly one following
    # per-channel bias Add (rather than a Gemm's own built-in bias) is the
    # common real-export shape for a biased linear layer on a 3-D hidden
    # state (Gemm has no batch-dim support). Confirms it is correctly
    # identified as the lm_head and its bias sliced right alongside the
    # weight.
    V, H, M = 10, 6, 4
    rng = np.random.default_rng(9)
    w_emb = rng.standard_normal((V, H)).astype(np.float32)
    w_lm = rng.standard_normal((H, V)).astype(np.float32)
    b_lm = rng.standard_normal(V).astype(np.float32)
    model = _model(
        f"""
        g (int64[{M}] input_ids) => (float[{M},{V}] logits)
        {{
          hidden = Gather<axis=0>(W_emb, input_ids)
          raw = MatMul(hidden, W_lm)
          logits = Add(raw, B_lm)
        }}
        """,
        initializer=[_f32(w_emb, "W_emb"), _f32(w_lm, "W_lm"), _f32(b_lm, "B_lm")],
    )
    onnx.checker.check_model(model)

    keep_token_ids = [0, 1, 2, 3, 4, 5]
    result = onnxsim.apply_embedding_vocab_pruning(model, keep_token_ids=keep_token_ids)
    onnx.checker.check_model(result.model)
    assert result.matched
    assert result.lm_head_pruned

    inits = {t.name: t for t in result.model.graph.initializer}
    assert list(inits["W_lm"].dims) == [H, len(keep_token_ids)]
    assert list(inits["B_lm"].dims) == [len(keep_token_ids)]

    input_ids = np.array([0, 3, 5, 1], dtype=np.int64)
    remapped = np.array([result.id_map[i] for i in input_ids], dtype=np.int64)
    orig_out = _run(model, {"input_ids": input_ids})[0]
    pruned_out = _run(result.model, {"input_ids": remapped})[0]
    expected = orig_out[..., result.kept_token_ids]
    np.testing.assert_allclose(pruned_out, expected, atol=1e-5, rtol=1e-5)


def test_embedding_vocab_pruning_declines_lm_head_with_mismatched_bias_shape():
    # The Add's other operand doesn't look like a per-channel vocab-width
    # bias at all (wrong trailing width) -- this pass must decline that
    # lm_head match outright rather than slice the MatMul weight and leave
    # a now-mismatched bias in place. The embedding itself is still safe
    # to prune on its own.
    V, H, M = 10, 6, 4
    rng = np.random.default_rng(90)
    w_emb = rng.standard_normal((V, H)).astype(np.float32)
    w_lm = rng.standard_normal((H, V)).astype(np.float32)
    bogus_bias = rng.standard_normal(H).astype(np.float32)  # wrong width -- not V
    model = _model(
        f"""
        g (int64[{M}] input_ids) => (float[{M},{V}] logits)
        {{
          hidden = Gather<axis=0>(W_emb, input_ids)
          raw = MatMul(hidden, W_lm)
          logits = Add(raw, Bogus)
        }}
        """,
        initializer=[
            _f32(w_emb, "W_emb"),
            _f32(w_lm, "W_lm"),
            _f32(bogus_bias, "Bogus"),
        ],
    )

    keep_token_ids = [0, 1, 2, 3, 4, 5]
    result = onnxsim.apply_embedding_vocab_pruning(model, keep_token_ids=keep_token_ids)
    assert result.matched
    assert not result.lm_head_pruned  # declined -- lm_head left fully untouched
    w_lm_out = {t.name: t for t in result.model.graph.initializer}["W_lm"]
    assert list(w_lm_out.dims) == [H, V]  # unchanged width


def test_embedding_vocab_pruning_fp16_matches_ort_execution():
    V, H = 10, 6
    rng = np.random.default_rng(10)
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
    result = onnxsim.apply_embedding_vocab_pruning(model, keep_token_ids=keep_token_ids)
    onnx.checker.check_model(result.model)
    assert result.matched and result.lm_head_pruned

    emb_init = {t.name: t for t in result.model.graph.initializer}["W_emb"]
    lm_init = {t.name: t for t in result.model.graph.initializer}["W_lm"]
    assert emb_init.data_type == onnx.TensorProto.FLOAT16
    assert lm_init.data_type == onnx.TensorProto.FLOAT16
    # Value-preserving slice -- every surviving row/column must reproduce
    # the exact original fp16 bit pattern, not a re-rounded one.
    emb_new = onnx.numpy_helper.to_array(emb_init)
    np.testing.assert_array_equal(
        emb_new.view(np.uint16), w_emb[keep_token_ids].view(np.uint16)
    )

    input_ids = np.array([0, 8, 4, 9], dtype=np.int64)
    remapped = np.array([result.id_map[i] for i in input_ids], dtype=np.int64)
    orig_out = _run(model, {"input_ids": input_ids})[0]
    pruned_out = _run(result.model, {"input_ids": remapped})[0]
    expected = orig_out[..., result.kept_token_ids]
    np.testing.assert_allclose(
        pruned_out.astype(np.float32), expected.astype(np.float32), atol=1e-2, rtol=1e-2
    )


def test_embedding_vocab_pruning_validates_keep_and_drop_arguments():
    model = _matmul_model(K=4, N=4)  # any model -- validation runs before matching
    with pytest.raises(ValueError, match="exactly one"):
        onnxsim.apply_embedding_vocab_pruning(model)
    with pytest.raises(ValueError, match="exactly one"):
        onnxsim.apply_embedding_vocab_pruning(
            model, keep_token_ids=[0], drop_token_ids=[1]
        )


def test_embedding_vocab_pruning_rejects_out_of_range_ids():
    V, H = 8, 4
    rng = np.random.default_rng(11)
    w_emb = rng.standard_normal((V, H)).astype(np.float32)
    model = _model(
        f"""
        g (int64[M] input_ids) => (float[M,{H}] hidden)
        {{
          hidden = Gather<axis=0>(W_emb, input_ids)
        }}
        """,
        initializer=[_f32(w_emb, "W_emb")],
    )
    with pytest.raises(ValueError, match="out of range"):
        onnxsim.apply_embedding_vocab_pruning(model, keep_token_ids=[0, 100])
    with pytest.raises(ValueError, match="out of range"):
        onnxsim.apply_embedding_vocab_pruning(model, drop_token_ids=[0, -1])
    with pytest.raises(ValueError, match="empty"):
        onnxsim.apply_embedding_vocab_pruning(model, drop_token_ids=list(range(V)))


def test_embedding_vocab_magnitude_pruning_drops_lowest_norm_rows_and_protects():
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

    result = onnxsim.apply_embedding_vocab_magnitude_pruning(model, sparsity=0.3)
    onnx.checker.check_model(result.model)
    assert result.matched
    assert not result.lm_head_pruned
    assert len(result.kept_token_ids) == round(V * 0.7) == 7
    assert 2 not in result.kept_token_ids
    assert 7 not in result.kept_token_ids

    protected = onnxsim.apply_embedding_vocab_magnitude_pruning(
        model, sparsity=0.3, protect_token_ids=[2]
    )
    assert 2 in protected.kept_token_ids


def test_embedding_vocab_magnitude_pruning_matches_oracle_for_kept_ids():
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

    result = onnxsim.apply_embedding_vocab_magnitude_pruning(model, sparsity=0.4)
    onnx.checker.check_model(result.model)
    assert result.matched and result.lm_head_pruned
    assert len(result.kept_token_ids) == round(V * 0.6)
    # Rows were scaled by a strictly decreasing factor -- the kept set
    # must be exactly the lowest-index (highest-norm) rows.
    assert result.kept_token_ids == list(range(len(result.kept_token_ids)))

    input_ids = np.array([0, 1, 2, 3], dtype=np.int64)
    remapped = np.array([result.id_map[i] for i in input_ids], dtype=np.int64)
    orig_out = _run(model, {"input_ids": input_ids})[0]
    pruned_out = _run(result.model, {"input_ids": remapped})[0]
    expected = orig_out[..., result.kept_token_ids]
    np.testing.assert_allclose(pruned_out, expected, atol=1e-5, rtol=1e-5)


def test_embedding_vocab_magnitude_pruning_rejects_bad_sparsity():
    model = _matmul_model(K=4, N=4)
    with pytest.raises(ValueError, match="sparsity"):
        onnxsim.apply_embedding_vocab_magnitude_pruning(model, sparsity=1.0)


def test_embedding_vocab_pruning_declines_when_no_embedding_pattern_exists():
    model = _matmul_model(K=8, N=4)
    result = onnxsim.apply_embedding_vocab_pruning(model, keep_token_ids=[0])
    assert result.matched is False
    assert result.model.SerializeToString() == model.SerializeToString()
