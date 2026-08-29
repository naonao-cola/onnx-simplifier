"""Tests for ``onnxsim.pruning`` -- magnitude pruning (data-free baseline),
Wanda pruning (calibrated on activation norms), and structured (channel)
pruning, see ``onnxsim/pruning.py``.
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


def test_wanda_pruning_conv_falls_back_to_magnitude_for_auto_pad():
    # auto_pad SAME_UPPER's padding depends on the input's own spatial
    # size, not something fixed per node -- :func:`_conv_spatial_attrs`
    # declines it, so Wanda must fall back to plain magnitude for this
    # layer rather than guessing at the padding.
    Cin, Cout, spatial = 3, 6, 10
    rng = np.random.default_rng(74)
    w = rng.standard_normal((Cout, Cin, 3, 3)).astype(np.float32)
    model = _single_conv_model(
        w, spatial=spatial, extra_attrs='auto_pad="SAME_UPPER"', out_spatial=spatial
    )
    x = rng.standard_normal((2, Cin, spatial, spatial)).astype(np.float32)

    magnitude_pruned = onnxsim.apply_magnitude_pruning(model, sparsity=0.5)
    wanda_pruned = onnxsim.apply_wanda_pruning(
        model, calibration_data=[{"X": x}], sparsity=0.5
    )
    onnx.checker.check_model(wanda_pruned)
    np.testing.assert_array_equal(
        _conv_weight(wanda_pruned), _conv_weight(magnitude_pruned)
    )


def test_wanda_pruning_conv_falls_back_to_magnitude_for_dilated_conv():
    # A dilated receptive field's (kh, kw) offsets aren't evenly spaced in
    # the padded input the way sliding_window_view assumes --
    # :func:`_conv_spatial_attrs` declines non-all-ones dilations, so
    # Wanda must fall back to plain magnitude for this layer too.
    Cin, Cout, spatial = 3, 6, 10
    rng = np.random.default_rng(75)
    w = rng.standard_normal((Cout, Cin, 3, 3)).astype(np.float32)
    out_spatial = spatial - 2 * (3 - 1)  # dilation=2, kernel=3, no padding
    model = _single_conv_model(
        w, spatial=spatial, extra_attrs="dilations=[2,2]", out_spatial=out_spatial
    )
    x = rng.standard_normal((2, Cin, spatial, spatial)).astype(np.float32)

    magnitude_pruned = onnxsim.apply_magnitude_pruning(model, sparsity=0.5)
    wanda_pruned = onnxsim.apply_wanda_pruning(
        model, calibration_data=[{"X": x}], sparsity=0.5
    )
    onnx.checker.check_model(wanda_pruned)
    np.testing.assert_array_equal(
        _conv_weight(wanda_pruned), _conv_weight(magnitude_pruned)
    )


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


def test_sparsegpt_pruning_conv_skips_grouped_conv():
    # A depthwise Conv (group == in_channels == out_channels) is left
    # completely untouched -- unlike apply_magnitude_pruning/
    # apply_wanda_pruning (which now do match depthwise/general grouped
    # Conv, see test_magnitude_pruning_conv_depthwise_reaches_target_
    # sparsity_and_matches_oracle and friends above),
    # apply_sparsegpt_pruning deliberately calls
    # _candidates(graph, allow_grouped_conv=False) -- its Conv Hessian
    # machinery is group=1-only, see this module's own docstring.
    C = 8
    rng = np.random.default_rng(76)
    w = rng.standard_normal((C, 1, 3, 3)).astype(np.float32)
    model = _single_conv_model(w, spatial=10, group=C)
    x_cal = rng.standard_normal((4, C, 10, 10)).astype(np.float32)

    pruned = onnxsim.apply_sparsegpt_pruning(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5
    )
    np.testing.assert_array_equal(_conv_weight(pruned), w)


def test_sparsegpt_pruning_conv_skips_general_grouped_conv():
    # Same restriction as the depthwise case above, for a general grouped
    # Conv (1 < group < in_channels) instead: apply_sparsegpt_pruning's
    # allow_grouped_conv=False excludes it too, even though
    # apply_magnitude_pruning/apply_wanda_pruning now match it.
    Cin, Cout, group = 8, 8, 2
    rng = np.random.default_rng(77)
    w = rng.standard_normal((Cout, Cin // group, 3, 3)).astype(np.float32)
    model = _single_conv_model(w, spatial=10, group=group)
    x_cal = rng.standard_normal((4, Cin, 10, 10)).astype(np.float32)

    pruned = onnxsim.apply_sparsegpt_pruning(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5
    )
    np.testing.assert_array_equal(_conv_weight(pruned), w)


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


def test_sparsegpt_pruning_conv_declines_auto_pad():
    # auto_pad SAME_UPPER's padding depends on the input's own spatial
    # size -- _conv_spatial_attrs declines it, so (unlike Wanda, which
    # falls back to plain magnitude) SparseGPT must leave the layer
    # completely untouched: there is no data-free fallback here.
    Cin, Cout, spatial = 3, 6, 10
    rng = np.random.default_rng(98)
    w = rng.standard_normal((Cout, Cin, 3, 3)).astype(np.float32)
    model = _single_conv_model(
        w, spatial=spatial, extra_attrs='auto_pad="SAME_UPPER"', out_spatial=spatial
    )
    x_cal = rng.standard_normal((4, Cin, spatial, spatial)).astype(np.float32)

    pruned = onnxsim.apply_sparsegpt_pruning(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5
    )
    np.testing.assert_array_equal(_conv_weight(pruned), w)


def test_sparsegpt_pruning_conv_declines_dilated_conv():
    # A dilated receptive field's (kh, kw) offsets aren't evenly spaced in
    # the padded input the way sliding_window_view assumes --
    # _conv_spatial_attrs declines non-all-ones dilations, so this layer
    # is left completely untouched too.
    Cin, Cout, spatial = 3, 6, 10
    rng = np.random.default_rng(99)
    w = rng.standard_normal((Cout, Cin, 3, 3)).astype(np.float32)
    out_spatial = spatial - 2 * (3 - 1)  # dilation=2, kernel=3, no padding
    model = _single_conv_model(
        w, spatial=spatial, extra_attrs="dilations=[2,2]", out_spatial=out_spatial
    )
    x_cal = rng.standard_normal((4, Cin, spatial, spatial)).astype(np.float32)

    pruned = onnxsim.apply_sparsegpt_pruning(
        model, calibration_data=[{"X": x_cal}], sparsity=0.5
    )
    np.testing.assert_array_equal(_conv_weight(pruned), w)


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
# shape), still holding every branch to the same single-consumer safety bar
# as the rest of this pass -- so a real multi-block ResNet's necessary
# fan-out (the post-block tensor read by both the next block's own first
# Conv and, unchanged, that block's own Add) is declined, not guessed at.


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
    # elsewhere -- unlike a real ResNet stage's interior block boundary
    # (see test_structured_pruning_conv_residual_add_declines_on_fan_out_branch
    # below), so the union-find grouping in _find_conv_residual_chains
    # extends across both Adds into one group of three producers.
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


def test_structured_pruning_conv_residual_add_declines_on_fan_out_branch():
    # A realistic multi-block residual stage's interior boundary: `r`
    # (add1's own post-block tensor) is read *twice* -- once by the next
    # block's own first Conv (`nxt`), once unchanged as that next block's
    # own Add shortcut operand (`add2`) -- exactly the fan-out
    # _walk_conv_producer_backward's single-consumer bar declines. Left
    # completely untouched (no partial/incorrect pruning), the same
    # guarantee every other branch case in this module holds to.
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
    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["WF"], w_f)
    np.testing.assert_array_equal(inits["WS"], w_s)
    np.testing.assert_array_equal(inits["WNEXT"], w_next)
    np.testing.assert_array_equal(inits["WOUT"], w_out)


def test_structured_pruning_conv_residual_add_declines_on_identity_shortcut():
    # y = Conv2(Relu(Add(Conv1(X), X))): a classic identity-shortcut
    # residual block with *no* Conv on the shortcut path at all. `X` has no
    # producer this pass owns (it's a graph input) *and* is itself read
    # twice (by Conv1 and directly by Add) -- either alone is enough to
    # decline: nothing can slice a graph input's own channel count to match
    # a pruned Conv1, so the whole block is left untouched rather than
    # guessed at.
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


def test_structured_pruning_conv_residual_add_declines_on_grouped_conv_consumer():
    # Two independent Conv branches merge via Add, same shape
    # _residual_diamond_model uses, but the downstream consumer is a
    # *general grouped* Conv (group=2) -- the composition _find_conv_chains
    # handles for a single-producer chain (see
    # test_structured_pruning_grouped_conv_consumer_only, if present) isn't
    # verified for a multi-producer residual chain: _chain_group's
    # per-group top-k assumes each producer feeds the consumer's full
    # channel range, which a residual group's combined-importance ranking
    # doesn't establish. _walk_to_conv_consumer still matches this grouped
    # Conv (it's an ordinary forward consumer match, unrelated to the
    # residual grouping above it), so this exercises
    # _find_conv_residual_chains's own explicit decline of a non-1
    # `consumer_group` rather than accidentally exiting earlier for some
    # unrelated reason.
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
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["WF"], w_f)
    np.testing.assert_array_equal(inits["WS"], w_s)
    np.testing.assert_array_equal(inits["WOUT"], w_out)


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
# transitive, fan-out decline, identity-shortcut decline, Wanda-for-free)
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


def test_structured_pruning_matmul_residual_add_declines_on_fan_out_branch():
    # A realistic multi-block residual stack's interior boundary: `r`
    # (add1's own post-block tensor) is read *twice* -- once by the next
    # block's own first MatMul (`nxt`), once unchanged as that next block's
    # own Add shortcut operand (`add2`) -- exactly the fan-out
    # _walk_matmul_producer_backward's single-consumer bar declines. Left
    # completely untouched, the same guarantee the Conv version holds to.
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
    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer}
    np.testing.assert_array_equal(inits["WF"], wf)
    np.testing.assert_array_equal(inits["WS"], ws)
    np.testing.assert_array_equal(inits["WNEXT"], wnext)
    np.testing.assert_array_equal(inits["WOUT"], wout)


def test_structured_pruning_matmul_residual_add_declines_on_identity_shortcut():
    # y = MatMul2(Relu(Add(MatMul1(X), X))): the exact `x = x + f(x)`
    # transformer-residual identity-shortcut shape, no MatMul on the
    # shortcut path at all. `X` has no producer this pass owns (it's a
    # graph input) *and* is itself read twice (by MatMul1 and directly by
    # Add) -- either alone is enough to decline: nothing can slice a graph
    # input's own channel count to match a pruned MatMul1, so the whole
    # block is left untouched rather than guessed at.
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


def test_structured_pruning_matmul_residual_add_declines_on_gated_branch_with_no_projection():
    # A gated (SwiGLU-style) combine feeding directly into a residual Add,
    # with no output-projection MatMul between the Mul and the Add -- see
    # _find_matmul_residual_chains's own section comment for why this
    # composition is declined rather than guessed at (the backward walk
    # would have no principled way to pick "the" one branch through a Mul
    # of two non-constant operands without silently dropping the other's
    # contribution). Left completely untouched; the far more common shape
    # -- the gated FFN's own down-projection feeding the residual Add --
    # is exercised by every other test in this section (WD below plays
    # exactly that role in every oracle-verified case).
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
        K=K, H=H, D=D, Out=Out, Nq=Nq, Nk=Nk, Nv=Nv, wqkv=wqkv, bqkv=bqkv, wout=wout
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
    # forward pass (see fuse_gqa.h's own top comment) -- `S-1` per batch row
    # and `S`, exactly what fuse_gqa.h itself synthesizes.
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


def test_gqa_pruning_nonempty_past_kv_constant_is_left_untouched():
    # A non-empty constant past_key/past_value holds real per-KV-head cache
    # data along the kv_num_heads axis that this module would need to slice
    # but doesn't attempt to -- declined outright rather than corrupted.
    model, cfg = _gqa_model(K=8, H=8, KVH=2, D=8, Out=6, seed=12, past_kv="nonempty")
    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    inits_before = {
        t.name: onnx.numpy_helper.to_array(t) for t in model.graph.initializer
    }
    inits_after = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer
    }
    for name in inits_before:
        np.testing.assert_array_equal(inits_before[name], inits_after[name])


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


def test_wanda_pruning_attention_merged_weight_falls_back_to_magnitude():
    # `Attention`'s own `X` input is rank-3 ([batch, seq, hidden]), not the
    # plain 2-D tensor Wanda's calibrated metric requires at the probe
    # point -- the documented fallback every other MatMul/Gemm layer with a
    # non-2-D activation already gets (see apply_wanda_pruning's own
    # docstring), not a crash and not an untouched layer.
    model, cfg = _attention_model(K=8, H=4, D=4, Out=6, seed=23)
    rng = np.random.default_rng(24)
    x = rng.standard_normal((2, 5, cfg["K"])).astype(np.float32)

    magnitude_pruned = onnxsim.apply_magnitude_pruning(model, sparsity=0.5)
    wanda_pruned = onnxsim.apply_wanda_pruning(
        model, calibration_data=[{"X": x}], sparsity=0.5
    )
    onnx.checker.check_model(wanda_pruned)

    inits_m = {t.name: t for t in magnitude_pruned.graph.initializer}
    inits_w = {t.name: t for t in wanda_pruned.graph.initializer}
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(inits_m["Wqkv"]),
        onnx.numpy_helper.to_array(inits_w["Wqkv"]),
    )


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
    past_kv=None,  # None (omitted) | "nonempty" (constant) | "dynamic" (graph input)
):
    # Unlike `com.microsoft::GroupQueryAttention` (see `_gqa_model`'s own
    # comment), this op's real onnxruntime CPU kernel has no observed
    # head_size-multiple-of-8 requirement -- verified empirically above --
    # so D defaults to a small 4, mirroring `_attention_model`'s own default.
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

    # Trailing optional inputs may simply be omitted from the node's own
    # input list rather than spelled out as empty placeholders.
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


def test_onnx_attention_pruning_nonempty_past_kv_constant_is_left_untouched():
    # A non-empty constant past_key/past_value holds real per-KV-head cache
    # data along the kv_num_heads axis this module would need to slice but
    # doesn't attempt to -- declined outright, mirroring
    # `test_gqa_pruning_nonempty_past_kv_constant_is_left_untouched` exactly
    # (`_match_onnx_attention_producer` applies the identical safety check).
    model, cfg = _onnx_attention_model(
        K=8, H=8, KVH=2, D=4, Out=6, seed=12, past_kv="nonempty"
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


def test_onnx_attention_pruning_nonempty_attn_mask_constant_is_left_untouched():
    # `attn_mask` is an optional input this op has that `GroupQueryAttention`
    # does not; a non-empty constant one is given the identical
    # decline-outright treatment `_match_gqa_producer` already applies to
    # `past_key`/`past_value` (see `_match_onnx_attention_producer`'s own
    # docstring) -- this exercises that specific input, not just the two
    # `past_key`/`past_value` inputs the two ops share the same shape of.
    model, cfg = _onnx_attention_model(
        K=8, H=8, KVH=2, D=4, Out=6, seed=15, attn_mask="nonempty"
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


def test_onnx_attention_pruning_diff_v_head_size_is_left_untouched():
    # This op's real schema (unlike `com.microsoft::GroupQueryAttention`,
    # which `fuse_gqa.h` always emits with equal Q/K/V head_size) genuinely
    # allows V its own, independent head_size -- confirmed via the op's own
    # backend-test suite (`test_attention_3d_diff_heads_sizes` and friends
    # in `onnx/backend/test/case/node/attention.py`). This pass reuses
    # `_apply_one_gqa_chain`'s shared, uniform-head_size slicing body
    # unmodified rather than a parallel implementation (see this module's
    # own "Attention-head pruning" section comment), so a node whose V head
    # size actually differs from Q/K's is declined here rather than
    # mis-sliced -- checked directly against `_find_onnx_attention_chains`
    # too, since nothing downstream needs exercising once the node fails to
    # match at all.
    K, H, KVH, D, Dv, Out = 8, 4, 4, 4, 6, 5
    Nq, Nk, Nv = H * D, KVH * D, KVH * Dv
    rng = np.random.default_rng(19)
    wq = rng.standard_normal((K, Nq)).astype(np.float32)
    wk = rng.standard_normal((K, Nk)).astype(np.float32)
    wv = rng.standard_normal((K, Nv)).astype(np.float32)
    wout = rng.standard_normal((Nq, Out)).astype(np.float32)
    # V's own head_size (`Dv`) differs from Q/K's (`D`) -- a shape this pass
    # declines rather than mis-slices.
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

    assert onnxsim.pruning._find_onnx_attention_chains(model.graph) == []

    pruned = onnxsim.apply_attention_head_pruning(model, sparsity=0.5)
    inits_before = {
        t.name: onnx.numpy_helper.to_array(t) for t in model.graph.initializer
    }
    inits_after = {
        t.name: onnx.numpy_helper.to_array(t) for t in pruned.graph.initializer
    }
    for name in inits_before:
        np.testing.assert_array_equal(inits_before[name], inits_after[name])


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
