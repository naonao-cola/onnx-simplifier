"""Tests for ``onnxsim.cross_layer_equalize`` (the ``cross_layer_equalization``
C++ pass) -- the data-free weight-equalization preprocessing technique from
"Data-Free Quantization Through Weight Equalization and Bias Correction"
(Nagel et al., 2019), also shipped as part of Qualcomm's AIMET toolkit.

Unlike this repo's quantize_* passes, CLE is supposed to be *exact*: it only
reparameterizes a Conv1 -> [activation] -> Conv2 pair's weights, never
changes the composed function. So these tests check near-bit-exact numeric
equivalence (``np.testing.assert_allclose`` with a tight tolerance), not the
lossy relative-L2 comparison this repo's quantize_* tests use, and directly
check that channel ranges actually rebalance (not just "the output didn't
change" -- a pass that declined to do anything would pass that check too).
"""

import collections

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper
import pytest

import onnxsim

# A bare ``import onnxruntime`` would fail collection (not skip the test) on
# platforms onnxruntime doesn't ship wheels for.
ort = pytest.importorskip("onnxruntime")


def _vi(name, shape):
    return onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)


def _f32(array, name):
    return onnx.numpy_helper.from_array(np.asarray(array, dtype=np.float32), name)


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _op_counts(model):
    return collections.Counter(n.op_type for n in model.graph.node)


def _weight_by_name(model, name):
    for t in model.graph.initializer:
        if t.name == name:
            return onnx.numpy_helper.to_array(t)
    raise KeyError(name)


def _conv_weight(model, output_name):
    """The weight initializer feeding the Conv node whose output is
    `output_name` -- resolved by name since cross_layer_equalize replaces
    the initializer object (new name) but not the Conv node itself."""
    for n in model.graph.node:
        if n.op_type == "Conv" and n.output[0] == output_name:
            return _weight_by_name(model, n.input[1])
    raise KeyError(output_name)


def _conv_chain_model(
    activation="relu",
    group2=1,
    branch=False,
    c1_out=8,
    c1_in=4,
    c2_out=4,
    seed=0,
    outlier_channels=(0,),
    outlier_scale=30.0,
    opset=13,
):
    """Conv1(c1_in -> c1_out) -> [activation] -> Conv2(c1_out -> c2_out),
    with `outlier_channels` of Conv1's weight scaled up by `outlier_scale`
    to create a channel-range imbalance for CLE to fix. `activation` is
    "relu", "prelu", "clip" (NOT scale-invariant -- CLE must decline),
    or None (direct Conv1 -> Conv2, still valid for CLE)."""
    rng = np.random.default_rng(seed)
    w1 = rng.standard_normal((c1_out, c1_in, 3, 3)).astype(np.float32) * 0.1
    for c in outlier_channels:
        w1[c] *= outlier_scale
    b1 = rng.standard_normal(c1_out).astype(np.float32) * 0.01
    w2 = rng.standard_normal((c2_out, c1_out // group2, 3, 3)).astype(np.float32) * 0.1
    b2 = rng.standard_normal(c2_out).astype(np.float32) * 0.01

    nodes = [
        onnx.helper.make_node(
            "Conv", ["x", "w1", "b1"], ["c1"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]
        )
    ]
    inits = [_f32(w1, "w1"), _f32(b1, "b1"), _f32(w2, "w2"), _f32(b2, "b2")]
    feed = "c1"
    if activation == "relu":
        nodes.append(onnx.helper.make_node("Relu", ["c1"], ["act"]))
        feed = "act"
    elif activation == "prelu":
        slope = np.full(c1_out, 0.1, dtype=np.float32)
        inits.append(_f32(slope, "slope"))
        nodes.append(onnx.helper.make_node("PRelu", ["c1", "slope"], ["act"]))
        feed = "act"
    elif activation == "clip":
        inits.append(_f32(np.float32(0.0), "clip_min"))
        inits.append(_f32(np.float32(6.0), "clip_max"))
        nodes.append(
            onnx.helper.make_node("Clip", ["c1", "clip_min", "clip_max"], ["act"])
        )
        feed = "act"
    elif activation is not None:
        raise ValueError(activation)

    conv2_kwargs = {"kernel_shape": [3, 3], "pads": [1, 1, 1, 1]}
    if group2 != 1:
        conv2_kwargs["group"] = group2
    nodes.append(
        onnx.helper.make_node("Conv", [feed, "w2", "b2"], ["y"], **conv2_kwargs)
    )
    outputs = [_vi("y", [1, c2_out, 8, 8])]
    if branch:
        nodes.append(onnx.helper.make_node("Identity", [feed], ["branch_out"]))
        outputs.append(_vi("branch_out", [1, c1_out, 8, 8]))

    graph = onnx.helper.make_graph(
        nodes, "g", [_vi("x", [1, c1_in, 8, 8])], outputs, inits
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", opset)], ir_version=8
    )
    return model, w1


def test_equalize_rebalances_channel_ranges_through_relu():
    model, w1 = _conv_chain_model(activation="relu")
    equalized = onnxsim.cross_layer_equalize(model)
    onnx.checker.check_model(equalized)

    new_w1 = _conv_weight(equalized, "c1")
    new_w2 = _conv_weight(equalized, "y")
    assert not np.allclose(w1, new_w1)  # actually rescaled, not a no-op

    c1_out = w1.shape[0]
    r1 = np.abs(new_w1).reshape(c1_out, -1).max(axis=1)
    r2 = np.abs(new_w2).transpose(1, 0, 2, 3).reshape(c1_out, -1).max(axis=1)
    np.testing.assert_allclose(r1, r2, rtol=1e-3)


def test_equalize_preserves_model_output_through_relu():
    model, _ = _conv_chain_model(activation="relu", seed=1)
    equalized = onnxsim.cross_layer_equalize(model)
    onnx.checker.check_model(equalized)

    rng = np.random.default_rng(7)
    x = rng.standard_normal((1, 4, 8, 8)).astype(np.float32)
    (out_before,) = _run(model, {"x": x})
    (out_after,) = _run(equalized, {"x": x})
    np.testing.assert_allclose(out_before, out_after, rtol=1e-4, atol=1e-4)


def test_equalize_preserves_model_output_through_prelu():
    model, _ = _conv_chain_model(activation="prelu", seed=2)
    equalized = onnxsim.cross_layer_equalize(model)
    onnx.checker.check_model(equalized)

    rng = np.random.default_rng(8)
    x = rng.standard_normal((1, 4, 8, 8)).astype(np.float32)
    (out_before,) = _run(model, {"x": x})
    (out_after,) = _run(equalized, {"x": x})
    np.testing.assert_allclose(out_before, out_after, rtol=1e-4, atol=1e-4)


def test_equalize_handles_conv_directly_feeding_conv():
    # No activation at all between the two convs is also a valid (trivially
    # scale-invariant, "identity activation") case for CLE.
    model, w1 = _conv_chain_model(activation=None, seed=3)
    equalized = onnxsim.cross_layer_equalize(model)
    onnx.checker.check_model(equalized)

    new_w1 = _conv_weight(equalized, "c1")
    assert not np.allclose(w1, new_w1)

    rng = np.random.default_rng(9)
    x = rng.standard_normal((1, 4, 8, 8)).astype(np.float32)
    (out_before,) = _run(model, {"x": x})
    (out_after,) = _run(equalized, {"x": x})
    np.testing.assert_allclose(out_before, out_after, rtol=1e-4, atol=1e-4)


def test_equalize_declines_clip_activation():
    # Clip(0, 6) is not positive-homogeneous (the upper bound doesn't scale
    # with the input) -- rescaling through it would change the function, so
    # the pass must leave this pair untouched.
    model, w1 = _conv_chain_model(activation="clip", seed=4)
    equalized = onnxsim.cross_layer_equalize(model)
    onnx.checker.check_model(equalized)
    np.testing.assert_array_equal(w1, _conv_weight(equalized, "c1"))


def test_equalize_declines_grouped_conv2():
    model, w1 = _conv_chain_model(activation="relu", group2=2, seed=5)
    equalized = onnxsim.cross_layer_equalize(model)
    onnx.checker.check_model(equalized)
    np.testing.assert_array_equal(w1, _conv_weight(equalized, "c1"))


def test_equalize_declines_when_conv1_output_branches():
    # Conv1's (post-activation) output also feeds something other than
    # Conv2 -- rescaling it would change that other consumer's input too.
    model, w1 = _conv_chain_model(activation="relu", branch=True, seed=6)
    equalized = onnxsim.cross_layer_equalize(model)
    onnx.checker.check_model(equalized)
    np.testing.assert_array_equal(w1, _conv_weight(equalized, "c1"))


def test_equalize_is_a_noop_on_a_model_with_no_matching_pattern():
    x = _vi("x", [2, 4])
    y = _vi("y", [2, 4])
    w = _f32(np.random.default_rng(0).standard_normal((4, 4)), "w")
    node = onnx.helper.make_node("MatMul", ["x", "w"], ["y"])
    graph = onnx.helper.make_graph([node], "g", [x], [y], [w])
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 13)], ir_version=8
    )

    equalized = onnxsim.cross_layer_equalize(model)
    assert _op_counts(equalized) == {"MatMul": 1}
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(model.graph.initializer[0]),
        onnx.numpy_helper.to_array(equalized.graph.initializer[0]),
    )


def test_equalize_propagates_across_a_three_conv_chain():
    # Fixed-point convergence across more than one adjacent pair: equalizing
    # (Conv1, Conv2) changes Conv2's own weight, which the same pass call
    # then re-equalizes against Conv3 in the same fixed-point sweep -- this
    # is what lets a single cross_layer_equalize() call balance a whole
    # chain of layers, not just the one pair it happens to match first. See
    # cross_layer_equalization.h's own top comment for why no explicit outer
    # iteration loop is needed here.
    rng = np.random.default_rng(10)
    channels = [4, 8, 8, 4]
    weights, biases = [], []
    for i in range(3):
        w = (
            rng.standard_normal((channels[i + 1], channels[i], 3, 3)).astype(np.float32)
            * 0.1
        )
        w[0] *= 25.0
        weights.append(w)
        biases.append(rng.standard_normal(channels[i + 1]).astype(np.float32) * 0.01)

    nodes = []
    inits = []
    feed = "x"
    for i in range(3):
        inits += [_f32(weights[i], f"w{i}"), _f32(biases[i], f"b{i}")]
        nodes.append(
            onnx.helper.make_node(
                "Conv",
                [feed, f"w{i}", f"b{i}"],
                [f"c{i}"],
                kernel_shape=[3, 3],
                pads=[1, 1, 1, 1],
            )
        )
        feed = f"c{i}"
        if i < 2:
            nodes.append(onnx.helper.make_node("Relu", [feed], [f"r{i}"]))
            feed = f"r{i}"
    nodes.append(onnx.helper.make_node("Identity", [feed], ["y"]))

    graph = onnx.helper.make_graph(
        nodes,
        "g",
        [_vi("x", [1, channels[0], 8, 8])],
        [_vi("y", [1, channels[3], 8, 8])],
        inits,
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 13)], ir_version=8
    )
    onnx.checker.check_model(model)

    equalized = onnxsim.cross_layer_equalize(model)
    onnx.checker.check_model(equalized)

    # Conv1/Conv2's shared channel ranges (not just Conv0/Conv1's) should
    # have moved from the pass also equalizing the (Conv1, Conv2) pair.
    w1_before = weights[1]
    w1_after = _conv_weight(equalized, "c1")
    assert not np.allclose(w1_before, w1_after)

    rng2 = np.random.default_rng(11)
    x = rng2.standard_normal((1, channels[0], 8, 8)).astype(np.float32)
    (out_before,) = _run(model, {"x": x})
    (out_after,) = _run(equalized, {"x": x})
    np.testing.assert_allclose(out_before, out_after, rtol=1e-4, atol=1e-4)


def test_equalize_is_idempotent():
    model, _ = _conv_chain_model(activation="relu", seed=12)
    once = onnxsim.cross_layer_equalize(model)
    twice = onnxsim.cross_layer_equalize(once)
    np.testing.assert_allclose(
        _conv_weight(once, "c1"), _conv_weight(twice, "c1"), rtol=1e-4, atol=1e-6
    )
