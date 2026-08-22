"""Tests for ``onnxsim.quantize_weight_only_int8_block`` (the
``weight_only_quantize_int8_block_matmul``/
``weight_only_quantize_int8_block_conv`` C++ passes).

Structurally identical to ``test_weight_only_quantize_int4.py``'s block-shape
coverage (scale-shape check, block-size-mismatch skip, pointwise vs. spatial
Conv) and ``test_weight_only_quantize_int16.py``'s real-execution/precision
comparisons -- just checking INT8's wider code range at INT4's block-wise
granularity instead. Needs opset 21 for DequantizeLinear's block_size
attribute (INT8 itself needs only opset 13, but the blocked form shares
INT4's opset floor).
"""

import collections

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper
import pytest

import onnxsim

# A bare ``import onnxruntime`` would fail collection (not skip the test) on
# platforms onnxruntime doesn't ship wheels for (e.g. s390x).
ort = pytest.importorskip("onnxruntime")


def _model(nodes, inputs, outputs, initializer, opset=21):
    graph = onnx.helper.make_graph(nodes, "g", inputs, outputs, initializer)
    return onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", opset)], ir_version=10
    )


def _vi(name, shape):
    return onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _op_counts(model):
    return collections.Counter(n.op_type for n in model.graph.node)


def _assert_close(float_outputs, quant_outputs, rel_l2_tol=0.1):
    for f, q in zip(float_outputs, quant_outputs):
        f = np.asarray(f, dtype=np.float64).ravel()
        q = np.asarray(q, dtype=np.float64).ravel()
        assert np.all(np.isfinite(q))
        rel_l2 = np.linalg.norm(f - q) / max(np.linalg.norm(f), 1e-6)
        assert rel_l2 < rel_l2_tol, f"relative L2 error too large: {rel_l2:.4f}"


def test_quantize_matmul():
    rng = np.random.default_rng(0)
    K, N = 64, 16
    weight = _f32(rng.standard_normal((K, N)) * 0.5, "W")
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    model = _model(nodes, [_vi("X", [4, K])], [_vi("Y", [4, N])], [weight])

    quant = onnxsim.quantize_weight_only_int8_block(model)
    onnx.checker.check_model(quant)
    ops = _op_counts(quant)
    assert ops["MatMul"] == 1
    assert ops["DequantizeLinear"] == 1

    w_init = next(t for t in quant.graph.initializer if t.name != "W")
    assert w_init.data_type == onnx.TensorProto.INT8

    x = rng.standard_normal((4, K)).astype(np.float32)
    _assert_close(_run(model, {"X": x}), _run(quant, {"X": x}))


def test_quantize_more_precise_than_per_channel_int8():
    # A weight with one dominant-outlier row should resolve noticeably
    # better under block-wise scales (each block's own scale, not one scale
    # for the whole channel dragged up by the outlier) than
    # quantize_weight_only's single per-channel scale.
    rng = np.random.default_rng(8)
    K, N = 64, 8
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.1
    w[0, :] = 50.0  # one outlier row blows a per-channel scale
    weight = _f32(w, "W")
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    model = _model(nodes, [_vi("X", [4, K])], [_vi("Y", [4, N])], [weight])

    quant_per_channel = onnxsim.quantize_weight_only(model)
    quant_block = onnxsim.quantize_weight_only_int8_block(model)

    x = rng.standard_normal((4, K)).astype(np.float32)
    float_out = _run(model, {"X": x})[0]
    out_per_channel = _run(quant_per_channel, {"X": x})[0]
    out_block = _run(quant_block, {"X": x})[0]

    err_per_channel = np.linalg.norm(float_out - out_per_channel)
    err_block = np.linalg.norm(float_out - out_block)
    assert err_block < err_per_channel


def test_quantize_gemm_transb_with_bias():
    # PyTorch's nn.Linear layout: weight is [out_features, in_features], i.e.
    # [N, K], exported as Gemm(X, W, B, transB=1) -- the common real-world case.
    rng = np.random.default_rng(1)
    K, N = 96, 12
    weight = _f32(rng.standard_normal((N, K)) * 0.5, "W")
    bias = _f32(rng.standard_normal(N), "B")
    nodes = [onnx.helper.make_node("Gemm", ["X", "W", "B"], ["Y"], transB=1)]
    model = _model(nodes, [_vi("X", [3, K])], [_vi("Y", [3, N])], [weight, bias])

    quant = onnxsim.quantize_weight_only_int8_block(model)
    onnx.checker.check_model(quant)
    ops = _op_counts(quant)
    assert ops["Gemm"] == 1
    assert ops["DequantizeLinear"] == 1

    x = rng.standard_normal((3, K)).astype(np.float32)
    _assert_close(_run(model, {"X": x}), _run(quant, {"X": x}))


def test_quantize_scale_shape_matches_block_count():
    # K=64 with the pass's block_size=32 gives 2 blocks; the scale
    # initializer must be [2, N] (block axis 0, matching MatMul's own [K, N]
    # weight layout).
    rng = np.random.default_rng(2)
    K, N = 64, 8
    weight = _f32(rng.standard_normal((K, N)) * 0.5, "W")
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    model = _model(nodes, [_vi("X", [1, K])], [_vi("Y", [1, N])], [weight])

    quant = onnxsim.quantize_weight_only_int8_block(model)
    scale_init = next(
        t
        for t in quant.graph.initializer
        if t.name != "W" and t.data_type == onnx.TensorProto.FLOAT
    )
    assert list(scale_init.dims) == [2, N]


def test_quantize_skips_k_not_divisible_by_block_size():
    # K=48 is not a multiple of the pass's block_size=32.
    rng = np.random.default_rng(3)
    K, N = 48, 8
    weight = _f32(rng.standard_normal((K, N)) * 0.5, "W")
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    model = _model(nodes, [_vi("X", [1, K])], [_vi("Y", [1, N])], [weight])

    quant = onnxsim.quantize_weight_only_int8_block(model)
    assert _op_counts(quant)["MatMul"] == 1
    assert _op_counts(quant)["DequantizeLinear"] == 0


def test_quantize_conv_pointwise():
    # A 1x1 (pointwise) Conv: inner = Cin/groups * kH * kW = Cin * 1 * 1, so
    # Cin=32 gives exactly one block -- the simplest Conv case, structurally
    # equivalent to a per-pixel MatMul.
    rng = np.random.default_rng(5)
    cout, cin = 16, 32
    weight = _f32(rng.standard_normal((cout, cin, 1, 1)) * 0.5, "W")
    nodes = [onnx.helper.make_node("Conv", ["X", "W"], ["Y"], kernel_shape=[1, 1])]
    model = _model(
        nodes, [_vi("X", [1, cin, 8, 8])], [_vi("Y", [1, cout, 8, 8])], [weight]
    )

    quant = onnxsim.quantize_weight_only_int8_block(model)
    onnx.checker.check_model(quant)
    ops = _op_counts(quant)
    assert ops["Conv"] == 1
    assert ops["DequantizeLinear"] == 1
    assert ops["Reshape"] == 1

    w_init = next(
        t for t in quant.graph.initializer if t.data_type == onnx.TensorProto.INT8
    )
    assert list(w_init.dims) == [cout, cin]  # flattened [Cout, inner]

    x = rng.standard_normal((1, cin, 8, 8)).astype(np.float32)
    _assert_close(_run(model, {"X": x}), _run(quant, {"X": x}))


def test_quantize_conv_spatial_kernel_with_bias():
    # kernel_shape=[2, 2], Cin=8 -> inner = 8 * 2 * 2 = 32: the flattening
    # spans both the channel and spatial dims, unlike the pointwise case.
    rng = np.random.default_rng(6)
    cout, cin = 4, 8
    weight = _f32(rng.standard_normal((cout, cin, 2, 2)) * 0.5, "W")
    bias = _f32(rng.standard_normal(cout), "B")
    nodes = [onnx.helper.make_node("Conv", ["X", "W", "B"], ["Y"], kernel_shape=[2, 2])]
    model = _model(
        nodes, [_vi("X", [1, cin, 8, 8])], [_vi("Y", [1, cout, 7, 7])], [weight, bias]
    )

    quant = onnxsim.quantize_weight_only_int8_block(model)
    onnx.checker.check_model(quant)
    ops = _op_counts(quant)
    assert ops["Conv"] == 1
    assert ops["DequantizeLinear"] == 1
    assert ops["Reshape"] == 1

    x = rng.standard_normal((1, cin, 8, 8)).astype(np.float32)
    _assert_close(_run(model, {"X": x}), _run(quant, {"X": x}))


def test_quantize_conv_skips_inner_not_divisible_by_block_size():
    # inner = Cin * kH * kW = 4 * 3 * 3 = 36, not a multiple of 32.
    rng = np.random.default_rng(7)
    cout, cin = 4, 4
    weight = _f32(rng.standard_normal((cout, cin, 3, 3)) * 0.5, "W")
    nodes = [onnx.helper.make_node("Conv", ["X", "W"], ["Y"], kernel_shape=[3, 3])]
    model = _model(
        nodes,
        [_vi("X", [1, cin, 8, 8])],
        [_vi("Y", [1, cout, 6, 6])],
        [weight],
    )

    quant = onnxsim.quantize_weight_only_int8_block(model)
    assert _op_counts(quant)["Conv"] == 1
    assert _op_counts(quant)["DequantizeLinear"] == 0


def test_quantize_skips_non_constant_weight():
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    model = _model(
        nodes, [_vi("X", [4, 64]), _vi("W", [64, 4])], [_vi("Y", [4, 4])], []
    )
    quant = onnxsim.quantize_weight_only_int8_block(model)
    assert _op_counts(quant)["MatMul"] == 1


def test_quantize_skips_old_opset():
    # DequantizeLinear's block_size attribute needs opset >= 21.
    weight = _f32(np.random.default_rng(4).standard_normal((64, 4)), "W")
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    model = _model(nodes, [_vi("X", [4, 64])], [_vi("Y", [4, 4])], [weight], opset=20)
    quant = onnxsim.quantize_weight_only_int8_block(model)
    assert _op_counts(quant)["MatMul"] == 1
    assert _op_counts(quant)["DequantizeLinear"] == 0
