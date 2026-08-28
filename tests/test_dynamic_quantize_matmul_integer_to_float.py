"""Tests for ``onnxsim.quantize_dynamic_matmul_integer_to_float`` (the
``dynamic_quantize_matmul_integer_to_float`` C++ pass) -- the same dynamic
quantization scheme ``test_dynamic_quantize_matmul.py`` covers, but using
ONNX Runtime's "com.microsoft" contrib op ``MatMulIntegerToFloat`` to fuse
the dequantize (and optional bias-add) step into a single node instead of a
MatMulInteger+Cast+Mul(+Add) chain.
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
    # See test_dynamic_quantize_matmul.py's identical helper: INT8 dynamic
    # quantization rounding is a discontinuous function of its input, so a
    # value near a rounding boundary can land in the adjacent bucket from a
    # last-bit floating-point difference across platforms/onnxruntime
    # versions. Checking the aggregate relative L2 error (not a tight
    # per-element band) avoids exactly the CI flakiness that produced.
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
        initializer=[weight],
    )

    quant = onnxsim.quantize_dynamic_matmul_integer_to_float(model)
    onnx.checker.check_model(quant)
    ops = _op_counts(quant)
    assert ops["MatMul"] == 0
    assert ops["DynamicQuantizeLinear"] == 1
    assert ops["MatMulIntegerToFloat"] == 1
    # No separate MatMulInteger/Cast/Mul/Add chain -- MatMulIntegerToFloat
    # fuses all of it into one node.
    assert ops["MatMulInteger"] == 0
    assert ops["Cast"] == 0
    assert ops["Mul"] == 0
    domains = {o.domain for o in quant.opset_import}
    assert "com.microsoft" in domains

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
        initializer=[weight, bias],
    )

    quant = onnxsim.quantize_dynamic_matmul_integer_to_float(model)
    onnx.checker.check_model(quant)
    ops = _op_counts(quant)
    assert ops["Gemm"] == 0
    assert ops["DynamicQuantizeLinear"] == 1
    assert ops["MatMulIntegerToFloat"] == 1
    # The bias is passed directly as MatMulIntegerToFloat's 7th input, not a
    # separate Add node.
    assert ops["Add"] == 0

    mmitf = next(n for n in quant.graph.node if n.op_type == "MatMulIntegerToFloat")
    assert len(mmitf.input) == 7
    assert mmitf.input[6] == "B"

    x = rng.standard_normal((3, K)).astype(np.float32)
    _assert_close(_run(model, {"X": x}), _run(quant, {"X": x}))


def test_quantize_matmul_no_bias_uses_empty_placeholder():
    rng = np.random.default_rng(2)
    K, N = 16, 8
    weight = _f32(rng.standard_normal((K, N)) * 0.5, "W")
    model = _model(
        f"""
        g (float[2,{K}] X) => (float[2,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[weight],
    )

    quant = onnxsim.quantize_dynamic_matmul_integer_to_float(model)
    onnx.checker.check_model(quant)
    mmitf = next(n for n in quant.graph.node if n.op_type == "MatMulIntegerToFloat")
    # b_zero_point (index 5) omitted as the standard empty-string
    # placeholder; no bias, so only 6 inputs total (no trailing 7th).
    assert mmitf.input[5] == ""
    assert len(mmitf.input) == 6


def test_quantize_skips_non_constant_weight():
    model = _model(
        """
        g (float[4,8] X, float[8,4] W) => (float[4,4] Y)
        {
          Y = MatMul(X, W)
        }
        """
    )
    quant = onnxsim.quantize_dynamic_matmul_integer_to_float(model)
    assert _op_counts(quant)["MatMul"] == 1


def test_quantize_skips_non_default_gemm_attrs():
    weight = _f32(np.random.randn(8, 4).astype(np.float32), "W")
    model = _model(
        """
        g (float[4,8] X) => (float[4,4] Y)
        {
          Y = Gemm<alpha = 2.0>(X, W)
        }
        """,
        initializer=[weight],
    )
    quant = onnxsim.quantize_dynamic_matmul_integer_to_float(model)
    assert _op_counts(quant)["Gemm"] == 1


def test_quantize_skips_reduction_depth_that_would_overflow_int32():
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
        initializer=[weight],
    )

    quant = onnxsim.quantize_dynamic_matmul_integer_to_float(model)
    onnx.checker.check_model(quant)
    ops = _op_counts(quant)
    assert ops["MatMul"] == 1
    assert ops["MatMulIntegerToFloat"] == 0
