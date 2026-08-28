"""Tests for ``onnxsim.quantize_dynamic`` (the ``dynamic_quantize_matmul`` C++
pass).

Each model is built directly with the ONNX text format (no torch dependency),
quantized, and then actually run through ONNX Runtime -- both before and after
quantization -- so these tests double as a minimal end-to-end
simplify/quantize/deploy check: the quantized graph must load and execute
under a real inference engine, and its outputs must stay close to the float
baseline.
"""

import collections

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim

# A bare ``import onnxruntime`` would fail collection (not skip the test) on
# platforms onnxruntime doesn't ship wheels for (e.g. s390x).
ort = pytest.importorskip("onnxruntime")


def _model(body, initializer=(), opset=13, ir_version=10):
    # Pin a low IR version so the model loads under older onnxruntime builds
    # (which cap at IR version 11), matching test_fusion_patterns.py.
    model = parser.parse_model(
        f"""
        <
          ir_version: {ir_version},
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


def _op_counts(model):
    return collections.Counter(n.op_type for n in model.graph.node)


def _assert_close(float_outputs, quant_outputs):
    # INT8/uint8 dynamic quantization is lossy by design, and rounding is a
    # discontinuous function of its input: a value that lands right on a
    # rounding boundary can flip to the adjacent integer from a last-bit
    # floating-point difference in DynamicQuantizeLinear/MatMulInteger's own
    # onnxruntime kernels (which can round a hair differently across
    # onnxruntime versions/CPUs), producing a full quantization-step-sized
    # error on that one output element even though the implementation is
    # correct -- observed in practice: CI's Linux runners consistently landed
    # 1-4 elements out of 36-64 outside a tight per-element band that this
    # machine satisfied exactly. So this checks the *aggregate* relL2 error
    # across the whole output instead of demanding every single element sit
    # within a tight per-element band; a real bug would blow up every
    # element, not just one near a boundary.
    for f, q in zip(float_outputs, quant_outputs):
        f = np.asarray(f, dtype=np.float64).ravel()
        q = np.asarray(q, dtype=np.float64).ravel()
        assert np.all(np.isfinite(q))
        rel_l2 = np.linalg.norm(f - q) / max(np.linalg.norm(f), 1e-6)
        assert rel_l2 < 0.1, f"relative L2 error too large: {rel_l2:.4f}"


def test_quantize_matmul():
    rng = np.random.default_rng(0)
    K, N = 32, 16
    weight = _f32(rng.standard_normal((K, N)) * 0.5, "W")
    model = _model(
        f"""
        g (float[4,{K}] X) => (float[4,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [weight],
    )

    quant = onnxsim.quantize_dynamic(model)
    onnx.checker.check_model(quant)
    ops = _op_counts(quant)
    assert ops["MatMul"] == 0
    assert ops["DynamicQuantizeLinear"] == 1
    assert ops["MatMulInteger"] == 1
    assert ops["Cast"] == 1
    assert ops["Mul"] == 2

    x = rng.standard_normal((4, K)).astype(np.float32)
    _assert_close(_run(model, {"X": x}), _run(quant, {"X": x}))


def test_quantize_gemm_transb_with_bias():
    # PyTorch's nn.Linear layout: weight is [out_features, in_features], i.e.
    # [N, K], exported as Gemm(X, W, B, transB=1) -- the common real-world case.
    rng = np.random.default_rng(1)
    K, N = 24, 12
    weight = _f32(rng.standard_normal((N, K)) * 0.5, "W")
    bias = _f32(rng.standard_normal(N), "B")
    model = _model(
        f"""
        g (float[3,{K}] X) => (float[3,{N}] Y)
        {{
          Y = Gemm<transB = 1>(X, W, B)
        }}
        """,
        [weight, bias],
    )

    quant = onnxsim.quantize_dynamic(model)
    onnx.checker.check_model(quant)
    ops = _op_counts(quant)
    assert ops["Gemm"] == 0
    assert ops["DynamicQuantizeLinear"] == 1
    assert ops["MatMulInteger"] == 1
    # The bias is added back, unquantized, after dequantization.
    assert ops["Add"] == 1

    x = rng.standard_normal((3, K)).astype(np.float32)
    _assert_close(_run(model, {"X": x}), _run(quant, {"X": x}))


def test_quantize_skips_non_constant_weight():
    # Both MatMul operands are graph inputs (neither is a constant), so there
    # is nothing to quantize ahead of time.
    model = _model(
        """
        g (float[4,8] X, float[8,4] W) => (float[4,4] Y)
        {
          Y = MatMul(X, W)
        }
        """
    )
    quant = onnxsim.quantize_dynamic(model)
    assert _op_counts(quant)["MatMul"] == 1


def test_quantize_skips_non_default_gemm_attrs():
    # alpha != 1 falls outside the "vanilla" Gemm shape this pass handles.
    weight = _f32(np.random.randn(8, 4).astype(np.float32), "W")
    model = _model(
        """
        g (float[4,8] X) => (float[4,4] Y)
        {
          Y = Gemm<alpha = 2.0>(X, W)
        }
        """,
        [weight],
    )
    quant = onnxsim.quantize_dynamic(model)
    assert _op_counts(quant)["Gemm"] == 1


def test_quantize_skips_reduction_depth_that_would_overflow_int32():
    # A worst-case int32 accumulator (every term at its extreme quantized
    # value, K * 127 * 255) can wrap around once K exceeds
    # MAX_SAFE_INT32_REDUCTION_DEPTH -- see
    # onnxsim/passes/quantize_matmul_common.h's IsSafeInt32ReductionDepth.
    # This pass must leave such a node in float rather than silently
    # producing a quantization that can overflow.
    from onnxsim.precision_estimator import MAX_SAFE_INT32_REDUCTION_DEPTH

    k = MAX_SAFE_INT32_REDUCTION_DEPTH + 1
    weight = _f32(np.random.randn(k, 1) * 0.01, "W")
    model = _model(
        f"""
        g (float[1,{k}] X) => (float[1,1] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [weight],
    )

    quant = onnxsim.quantize_dynamic(model)
    onnx.checker.check_model(quant)
    ops = _op_counts(quant)
    assert ops["MatMul"] == 1
    assert ops["MatMulInteger"] == 0


def test_quantize_still_applies_at_the_safe_reduction_depth_boundary():
    # One less than the overflow test above: still safe, so the node is
    # quantized as usual.
    from onnxsim.precision_estimator import MAX_SAFE_INT32_REDUCTION_DEPTH

    k = MAX_SAFE_INT32_REDUCTION_DEPTH
    weight = _f32(np.random.randn(k, 1) * 0.01, "W")
    model = _model(
        f"""
        g (float[1,{k}] X) => (float[1,1] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [weight],
    )

    quant = onnxsim.quantize_dynamic(model)
    onnx.checker.check_model(quant)
    ops = _op_counts(quant)
    assert ops["MatMul"] == 0
    assert ops["MatMulInteger"] == 1
