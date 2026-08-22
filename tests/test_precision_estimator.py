"""Tests for ``onnxsim.estimate_quantization_precision`` (onnxsim/precision_estimator.py).

Pure Python, read-only analysis -- no C++ extension involved -- so these
models are built directly with ``onnx.helper`` and never run through an
inference engine.
"""

import math

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

import onnxsim
from onnxsim.precision_estimator import (
    MAX_EXACT_FLOAT32_REDUCTION_DEPTH,
    MAX_SAFE_INT32_REDUCTION_DEPTH,
    AttentionPrecisionEstimate,
    ConvPrecisionEstimate,
    MatMulGemmPrecisionEstimate,
)


def _vi(name, shape):
    return onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)


def _f32(array, name):
    return onnx.numpy_helper.from_array(np.asarray(array, dtype=np.float32), name)


def _model(nodes, inputs, outputs, initializer, opset=17):
    graph = onnx.helper.make_graph(nodes, "g", inputs, outputs, initializer)
    return onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", opset)]
    )


def test_matmul_small_reduction_depth_is_safe_and_exact():
    rng = np.random.default_rng(0)
    K, N = 16, 4
    weight = _f32(rng.standard_normal((K, N)) * 0.1, "W")
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    model = _model(nodes, [_vi("X", [1, K])], [_vi("Y", [1, N])], [weight])

    estimates = onnxsim.estimate_quantization_precision(model)
    assert len(estimates) == 1
    est = estimates[0]
    assert isinstance(est, MatMulGemmPrecisionEstimate)
    assert est.op_type == "MatMul"
    assert est.reduction_depth == K
    assert est.num_channels == N
    assert est.int32_accumulator_safe
    assert est.float32_cast_exact
    assert not est.outlier_risk


def test_matmul_reduction_depth_past_int32_bound_is_unsafe():
    k = MAX_SAFE_INT32_REDUCTION_DEPTH + 1
    weight = _f32(np.random.default_rng(1).standard_normal((k, 1)) * 0.01, "W")
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    model = _model(nodes, [_vi("X", [1, k])], [_vi("Y", [1, 1])], [weight])

    (est,) = onnxsim.estimate_quantization_precision(model)
    assert not est.int32_accumulator_safe
    assert "int32-safe bound" in est.recommendation


def test_matmul_reduction_depth_past_float32_exact_bound_but_int32_safe():
    k = MAX_EXACT_FLOAT32_REDUCTION_DEPTH + 1
    assert k <= MAX_SAFE_INT32_REDUCTION_DEPTH  # sanity: still int32-safe
    weight = _f32(np.random.default_rng(2).standard_normal((k, 1)) * 0.01, "W")
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    model = _model(nodes, [_vi("X", [1, k])], [_vi("Y", [1, 1])], [weight])

    (est,) = onnxsim.estimate_quantization_precision(model)
    assert est.int32_accumulator_safe
    assert not est.float32_cast_exact
    assert "not bit-exact" in est.recommendation


def test_matmul_outlier_channel_is_flagged():
    K, N = 32, 2
    rng = np.random.default_rng(3)
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.05
    weight[:, 1] = 0.01  # channel 1: uniform small weights ...
    weight[0, 1] = 10.0  # ... plus one extreme outlier
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    model = _model(nodes, [_vi("X", [1, K])], [_vi("Y", [1, N])], [_f32(weight, "W")])

    (est,) = onnxsim.estimate_quantization_precision(model)
    assert est.outlier_risk
    assert est.max_outlier_ratio > 127.0


def test_gemm_transb_reduction_depth_uses_transposed_layout():
    # PyTorch nn.Linear layout: weight [out_features, in_features] = [N, K].
    N, K = 4, 20
    weight = _f32(np.random.default_rng(4).standard_normal((N, K)) * 0.1, "W")
    nodes = [onnx.helper.make_node("Gemm", ["X", "W"], ["Y"], transB=1)]
    model = _model(nodes, [_vi("X", [1, K])], [_vi("Y", [1, N])], [weight])

    (est,) = onnxsim.estimate_quantization_precision(model)
    assert est.op_type == "Gemm"
    assert est.reduction_depth == K
    assert est.num_channels == N


def test_conv_reduction_depth_is_cin_times_kernel_volume():
    cout, cin, kh, kw = 6, 3, 5, 5
    weight = _f32(
        np.random.default_rng(5).standard_normal((cout, cin, kh, kw)) * 0.1, "W"
    )
    x = _vi("X", [1, cin, 16, 16])
    y = _vi("Y", [1, cout, 12, 12])
    nodes = [onnx.helper.make_node("Conv", ["X", "W"], ["Y"])]
    model = _model(nodes, [x], [y], [weight])

    (est,) = onnxsim.estimate_quantization_precision(model)
    assert isinstance(est, ConvPrecisionEstimate)
    assert est.reduction_depth == cin * kh * kw
    assert est.num_channels == cout


def test_attention_reports_head_dim_and_flags_scale_mismatch():
    q_heads, kv_heads, head_dim, sq, skv = 4, 4, 8, 6, 6
    q = _vi("Q", [1, sq, q_heads * head_dim])
    k = _vi("K", [1, skv, kv_heads * head_dim])
    v = _vi("V", [1, skv, kv_heads * head_dim])
    out = _vi("O", [1, sq, q_heads * head_dim])
    node = onnx.helper.make_node(
        "Attention",
        ["Q", "K", "V"],
        ["O"],
        q_num_heads=q_heads,
        kv_num_heads=kv_heads,
        scale=1.0,  # deliberately not 1/sqrt(head_dim)
    )
    model = _model([node], [q, k, v], [out], [], opset=23)

    (est,) = onnxsim.estimate_quantization_precision(model)
    assert isinstance(est, AttentionPrecisionEstimate)
    assert est.head_dim == head_dim
    assert est.num_query_heads == q_heads
    assert est.num_kv_heads == kv_heads
    assert math.isclose(est.default_scale, 1.0 / math.sqrt(head_dim))
    assert est.scale_matches_default is False
    assert "does not match" in est.recommendation


def test_attention_scale_matching_default_is_not_flagged():
    q_heads, kv_heads, head_dim, sq, skv = 2, 2, 16, 4, 4
    q = _vi("Q", [1, sq, q_heads * head_dim])
    k = _vi("K", [1, skv, kv_heads * head_dim])
    v = _vi("V", [1, skv, kv_heads * head_dim])
    out = _vi("O", [1, sq, q_heads * head_dim])
    node = onnx.helper.make_node(
        "Attention",
        ["Q", "K", "V"],
        ["O"],
        q_num_heads=q_heads,
        kv_num_heads=kv_heads,
    )
    model = _model([node], [q, k, v], [out], [], opset=23)

    (est,) = onnxsim.estimate_quantization_precision(model)
    assert est.actual_scale is None
    assert est.scale_matches_default is None
    assert "no int8 accumulator applies" in est.recommendation


def test_matmul_fed_by_softmax_reports_known_activation_range():
    K, N = 32, 4
    weight = _f32(np.random.default_rng(7).standard_normal((K, N)) * 0.1, "W")
    softmax = onnx.helper.make_node("Softmax", ["X"], ["S"], axis=-1)
    matmul = onnx.helper.make_node("MatMul", ["S", "W"], ["Y"])
    model = _model([softmax, matmul], [_vi("X", [1, K])], [_vi("Y", [1, N])], [weight])

    (est,) = onnxsim.estimate_quantization_precision(model)
    assert est.activation_producer_op == "Softmax"
    assert est.activation_range == (0.0, 1.0)
    # Known range does NOT change the overflow/exactness verdicts -- it's a
    # separate, calibration-free-static-quantization claim (see the module
    # docstring's point 4).
    assert est.int32_accumulator_safe
    assert "fixed static scale would quantize it exactly" in est.recommendation


def test_conv_fed_by_clip_with_constant_bounds_reports_range():
    cout, cin, kh, kw = 3, 2, 3, 3
    weight = _f32(
        np.random.default_rng(8).standard_normal((cout, cin, kh, kw)) * 0.1, "W"
    )
    lo = _f32(np.array(0.0), "lo")
    hi = _f32(np.array(6.0), "hi")
    clip = onnx.helper.make_node("Clip", ["X", "lo", "hi"], ["C"])
    conv = onnx.helper.make_node("Conv", ["C", "W"], ["Y"])
    x = _vi("X", [1, cin, 8, 8])
    y = _vi("Y", [1, cout, 6, 6])
    model = _model([clip, conv], [x], [y], [weight, lo, hi])

    (est,) = onnxsim.estimate_quantization_precision(model)
    assert est.activation_producer_op == "Clip"
    assert est.activation_range == (0.0, 6.0)


def test_clip_with_non_constant_bound_is_not_reported_as_known_range():
    # max ("hi") is a runtime activation, not a constant -- the range isn't
    # analytically fixed, so this must NOT be reported as known.
    weight = _f32(np.random.default_rng(9).standard_normal((16, 4)) * 0.1, "W")
    clip = onnx.helper.make_node("Clip", ["X", "lo", "hi"], ["C"])
    matmul = onnx.helper.make_node("MatMul", ["C", "W"], ["Y"])
    lo = _f32(np.array(0.0), "lo")
    model = _model(
        [clip, matmul],
        [_vi("X", [1, 16]), _vi("hi", [])],
        [_vi("Y", [1, 4])],
        [weight, lo],
    )

    (est,) = onnxsim.estimate_quantization_precision(model)
    assert est.activation_producer_op is None
    assert est.activation_range is None


def test_matmul_fed_by_relu_has_no_known_fixed_range():
    # Relu is only bounded on one side (non-negative, unbounded above), so it
    # doesn't qualify for a fixed (lo, hi) static-quantization range.
    weight = _f32(np.random.default_rng(10).standard_normal((16, 4)) * 0.1, "W")
    relu = onnx.helper.make_node("Relu", ["X"], ["R"])
    matmul = onnx.helper.make_node("MatMul", ["R", "W"], ["Y"])
    model = _model([relu, matmul], [_vi("X", [1, 16])], [_vi("Y", [1, 4])], [weight])

    (est,) = onnxsim.estimate_quantization_precision(model)
    assert est.activation_producer_op is None
    assert est.activation_range is None


def test_non_constant_weight_and_unrelated_ops_are_skipped():
    nodes = [
        onnx.helper.make_node("MatMul", ["X", "W"], ["Y"]),  # W not a constant
        onnx.helper.make_node("Relu", ["Y"], ["Z"]),
    ]
    model = _model(nodes, [_vi("X", [1, 8]), _vi("W", [8, 4])], [_vi("Z", [1, 4])], [])
    assert onnxsim.estimate_quantization_precision(model) == []


def test_never_modifies_the_model():
    rng = np.random.default_rng(6)
    weight = _f32(rng.standard_normal((8, 4)) * 0.1, "W")
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    model = _model(nodes, [_vi("X", [1, 8])], [_vi("Y", [1, 4])], [weight])
    before = model.SerializeToString()
    onnxsim.estimate_quantization_precision(model)
    assert model.SerializeToString() == before
