"""Tests for ``onnxsim.estimate_quantization_precision`` (onnxsim/precision_estimator.py).

Pure Python, read-only analysis -- no C++ extension involved -- so these
models are built directly with the ONNX text format parser (``onnx.parser``)
and never run through an inference engine.
"""

import math

import numpy as np
import onnx
import onnx.numpy_helper
from onnx import parser

import onnxsim
from onnxsim.precision_estimator import (
    MAX_EXACT_FLOAT32_REDUCTION_DEPTH,
    MAX_SAFE_INT32_REDUCTION_DEPTH,
    OUTLIER_RATIO_THRESHOLD,
    AttentionPrecisionEstimate,
    ConvPrecisionEstimate,
    MatMulGemmPrecisionEstimate,
)


def _f32(array, name):
    return onnx.numpy_helper.from_array(np.asarray(array, dtype=np.float32), name)


def _model(body, initializer=(), opset=17):
    model = parser.parse_model(
        f"""
        <
          opset_import: ["": {opset}]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    # Left at the default (not pinned), matching the original onnx.helper.make_model
    # calls here, which never passed ir_version either.
    model.ir_version = onnx.IR_VERSION
    return model


def test_matmul_small_reduction_depth_is_safe_and_exact():
    rng = np.random.default_rng(0)
    K, N = 16, 4
    weight = _f32(rng.standard_normal((K, N)) * 0.1, "W")
    model = _model(
        f"""
        g (float[1,{K}] X) => (float[1,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[weight],
    )

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
    model = _model(
        f"""
        g (float[1,{k}] X) => (float[1,1] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[weight],
    )

    (est,) = onnxsim.estimate_quantization_precision(model)
    assert not est.int32_accumulator_safe
    assert "int32-safe bound" in est.recommendation


def test_matmul_reduction_depth_past_float32_exact_bound_but_int32_safe():
    k = MAX_EXACT_FLOAT32_REDUCTION_DEPTH + 1
    assert k <= MAX_SAFE_INT32_REDUCTION_DEPTH  # sanity: still int32-safe
    weight = _f32(np.random.default_rng(2).standard_normal((k, 1)) * 0.01, "W")
    model = _model(
        f"""
        g (float[1,{k}] X) => (float[1,1] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[weight],
    )

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
    model = _model(
        f"""
        g (float[1,{K}] X) => (float[1,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f32(weight, "W")],
    )

    (est,) = onnxsim.estimate_quantization_precision(model)
    assert est.outlier_risk
    assert est.max_outlier_ratio > 127.0


def test_gemm_transb_reduction_depth_uses_transposed_layout():
    # PyTorch nn.Linear layout: weight [out_features, in_features] = [N, K].
    N, K = 4, 20
    weight = _f32(np.random.default_rng(4).standard_normal((N, K)) * 0.1, "W")
    model = _model(
        f"""
        g (float[1,{K}] X) => (float[1,{N}] Y)
        {{
          Y = Gemm<transB = 1>(X, W)
        }}
        """,
        initializer=[weight],
    )

    (est,) = onnxsim.estimate_quantization_precision(model)
    assert est.op_type == "Gemm"
    assert est.reduction_depth == K
    assert est.num_channels == N


def test_conv_reduction_depth_is_cin_times_kernel_volume():
    cout, cin, kh, kw = 6, 3, 5, 5
    weight = _f32(
        np.random.default_rng(5).standard_normal((cout, cin, kh, kw)) * 0.1, "W"
    )
    model = _model(
        f"""
        g (float[1,{cin},16,16] X) => (float[1,{cout},12,12] Y)
        {{
          Y = Conv(X, W)
        }}
        """,
        initializer=[weight],
    )

    (est,) = onnxsim.estimate_quantization_precision(model)
    assert isinstance(est, ConvPrecisionEstimate)
    assert est.reduction_depth == cin * kh * kw
    assert est.num_channels == cout


def test_attention_reports_head_dim_and_flags_scale_mismatch():
    q_heads, kv_heads, head_dim, sq, skv = 4, 4, 8, 6, 6
    qh, kvh = q_heads * head_dim, kv_heads * head_dim
    # scale=1.0 is deliberately not 1/sqrt(head_dim).
    model = _model(
        f"""
        g (float[1,{sq},{qh}] Q, float[1,{skv},{kvh}] K, float[1,{skv},{kvh}] V) => (float[1,{sq},{qh}] O)
        {{
          O = Attention<q_num_heads = {q_heads}, kv_num_heads = {kv_heads}, scale = 1.0>(Q, K, V)
        }}
        """,
        opset=23,
    )

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
    qh, kvh = q_heads * head_dim, kv_heads * head_dim
    model = _model(
        f"""
        g (float[1,{sq},{qh}] Q, float[1,{skv},{kvh}] K, float[1,{skv},{kvh}] V) => (float[1,{sq},{qh}] O)
        {{
          O = Attention<q_num_heads = {q_heads}, kv_num_heads = {kv_heads}>(Q, K, V)
        }}
        """,
        opset=23,
    )

    (est,) = onnxsim.estimate_quantization_precision(model)
    assert est.actual_scale is None
    assert est.scale_matches_default is None
    assert "no int8 accumulator applies" in est.recommendation


def test_matmul_fed_by_softmax_reports_known_activation_range():
    K, N = 32, 4
    weight = _f32(np.random.default_rng(7).standard_normal((K, N)) * 0.1, "W")
    model = _model(
        f"""
        g (float[1,{K}] X) => (float[1,{N}] Y)
        {{
          S = Softmax<axis = -1>(X)
          Y = MatMul(S, W)
        }}
        """,
        initializer=[weight],
    )

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
    model = _model(
        f"""
        g (float[1,{cin},8,8] X) => (float[1,{cout},6,6] Y)
        <float lo = {{0.0}}, float hi = {{6.0}}>
        {{
          C = Clip(X, lo, hi)
          Y = Conv(C, W)
        }}
        """,
        initializer=[weight],
    )

    (est,) = onnxsim.estimate_quantization_precision(model)
    assert est.activation_producer_op == "Clip"
    assert est.activation_range == (0.0, 6.0)


def test_clip_with_non_constant_bound_is_not_reported_as_known_range():
    # max ("hi") is a runtime activation, not a constant -- the range isn't
    # analytically fixed, so this must NOT be reported as known.
    weight = _f32(np.random.default_rng(9).standard_normal((16, 4)) * 0.1, "W")
    model = _model(
        """
        g (float[1,16] X, float hi) => (float[1,4] Y)
        <float lo = {0.0}>
        {
          C = Clip(X, lo, hi)
          Y = MatMul(C, W)
        }
        """,
        initializer=[weight],
    )

    (est,) = onnxsim.estimate_quantization_precision(model)
    assert est.activation_producer_op is None
    assert est.activation_range is None


def test_matmul_fed_by_relu_has_no_known_fixed_range():
    # Relu is only bounded on one side (non-negative, unbounded above), so it
    # doesn't qualify for a fixed (lo, hi) static-quantization range.
    weight = _f32(np.random.default_rng(10).standard_normal((16, 4)) * 0.1, "W")
    model = _model(
        """
        g (float[1,16] X) => (float[1,4] Y)
        {
          R = Relu(X)
          Y = MatMul(R, W)
        }
        """,
        initializer=[weight],
    )

    (est,) = onnxsim.estimate_quantization_precision(model)
    assert est.activation_producer_op is None
    assert est.activation_range is None


def test_non_constant_weight_and_unrelated_ops_are_skipped():
    model = _model(
        """
        g (float[1,8] X, float[8,4] W) => (float[1,4] Z)
        {
          Y = MatMul(X, W)
          Z = Relu(Y)
        }
        """
    )
    assert onnxsim.estimate_quantization_precision(model) == []


def test_never_modifies_the_model():
    rng = np.random.default_rng(6)
    weight = _f32(rng.standard_normal((8, 4)) * 0.1, "W")
    model = _model(
        """
        g (float[1,8] X) => (float[1,4] Y)
        {
          Y = MatMul(X, W)
        }
        """,
        initializer=[weight],
    )
    before = model.SerializeToString()
    onnxsim.estimate_quantization_precision(model)
    assert model.SerializeToString() == before


# --------------------------------------------------------------------------- #
# estimate_model_quantization_drop -- whole-model aggregate
# --------------------------------------------------------------------------- #
def test_model_drop_safe_with_no_outliers():
    # Every weight has the exact same magnitude (alternating sign), so each
    # channel's max(|w|)/median(|w|) outlier ratio is exactly 1 -- the
    # uniform-quantizer-noise baseline case, with no outlier-driven penalty.
    K, N = 16, 4
    weight = np.tile([0.1, -0.1], (K // 2, N)).astype(np.float32)
    model = _model(
        f"""
        g (float[1,{K}] X) => (float[1,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f32(weight, "W")],
    )

    est = onnxsim.estimate_model_quantization_drop(model)
    assert est.total_nodes_analyzed == 1
    assert est.unsafe_nodes == []
    assert est.outlier_risk_nodes == []
    assert est.risk_level == "safe"
    assert math.isclose(est.worst_outlier_ratio, 1.0, rel_tol=1e-9)
    # The uniform-quantizer-noise baseline (ratio=1): a single analyzed
    # node's relative error is exactly 1 / (127 * sqrt(12)).
    assert math.isclose(
        est.estimated_relative_error, 1.0 / (127.0 * math.sqrt(12.0)), rel_tol=1e-9
    )


def test_model_drop_degraded_when_outlier_channel_present():
    K, N = 32, 2
    rng = np.random.default_rng(21)
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.05
    weight[:, 1] = 0.01
    weight[0, 1] = 10.0  # channel 1: a single extreme outlier
    model = _model(
        f"""
        g (float[1,{K}] X) => (float[1,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f32(weight, "W")],
    )

    est = onnxsim.estimate_model_quantization_drop(model)
    assert est.risk_level == "degraded"
    assert est.outlier_risk_nodes == [est.per_node[0].node_name]
    assert est.unsafe_nodes == []
    assert est.worst_outlier_ratio > OUTLIER_RATIO_THRESHOLD
    # A node with a big outlier ratio must dominate the aggregate error over
    # the safe/no-outlier baseline case.
    assert est.estimated_relative_error > 1.0 / (127.0 * math.sqrt(12.0))


def test_model_drop_unsafe_reports_nan_error_and_lists_the_node():
    k = MAX_SAFE_INT32_REDUCTION_DEPTH + 1
    weight = _f32(np.random.default_rng(22).standard_normal((k, 1)) * 0.01, "W")
    model = _model(
        f"""
        g (float[1,{k}] X) => (float[1,1] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[weight],
    )

    est = onnxsim.estimate_model_quantization_drop(model)
    assert est.risk_level == "unsafe"
    assert est.unsafe_nodes == [est.per_node[0].node_name]
    assert math.isnan(est.estimated_relative_error)


def test_model_drop_more_unsafe_nodes_widen_the_aggregate_error():
    # Two safe nodes (independent noise, root-sum-square) must report a
    # bigger aggregate error than either alone.
    rng = np.random.default_rng(23)
    K, N = 16, 4
    w1 = _f32(rng.standard_normal((K, N)) * 0.1, "W1")
    w2 = _f32(rng.standard_normal((K, N)) * 0.1, "W2")
    model = _model(
        f"""
        g (float[1,{K}] X) => (float[1,{N}] Y1, float[1,{N}] Y2)
        {{
          Y1 = MatMul(X, W1)
          Y2 = MatMul(X, W2)
        }}
        """,
        initializer=[w1, w2],
    )
    one_node_model = _model(
        f"""
        g (float[1,{K}] X) => (float[1,{N}] Y1)
        {{
          Y1 = MatMul(X, W1)
        }}
        """,
        initializer=[w1],
    )

    est_two = onnxsim.estimate_model_quantization_drop(model)
    est_one = onnxsim.estimate_model_quantization_drop(one_node_model)
    assert est_two.total_nodes_analyzed == 2
    assert est_two.estimated_relative_error > est_one.estimated_relative_error


def test_model_drop_no_analyzable_nodes_is_safe_with_zero_error():
    model = _model(
        """
        g (float[1,4] X) => (float[1,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    est = onnxsim.estimate_model_quantization_drop(model)
    assert est.total_nodes_analyzed == 0
    assert est.risk_level == "safe"
    assert est.estimated_relative_error == 0.0
