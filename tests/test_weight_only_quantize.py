"""Tests for ``onnxsim.quantize_weight_only`` (the
``weight_only_quantize_matmul``/``weight_only_quantize_conv`` C++ passes).

Each model is built directly with ``onnx.parser`` (no torch dependency),
quantized, and then actually run through ONNX Runtime -- both before and after
quantization -- so these tests double as a minimal end-to-end
simplify/quantize/deploy check: the quantized graph must load and execute
under a real inference engine, and its outputs must stay close to the float
baseline. Unlike ``quantize_dynamic``/``quantize_static``, no calibration data
is involved at all: only the weight changes.
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
    # Pin a low IR version by default so the model loads under older
    # onnxruntime builds (which cap at IR version 11), matching
    # test_fusion_patterns.py.
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
    # INT8 weight-only quantization is lossy by design; see
    # test_dynamic_quantize_matmul.py's identically-named helper for why this
    # checks aggregate relative L2 error rather than a tight per-element bound.
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

    quant = onnxsim.quantize_weight_only(model)
    onnx.checker.check_model(quant)
    ops = _op_counts(quant)
    # Unlike quantize_dynamic, MatMul itself is untouched (QDQ-on-weight-only
    # format) and no activation quantize/dequantize nodes are added.
    assert ops["MatMul"] == 1
    assert ops["DequantizeLinear"] == 1
    assert ops["DynamicQuantizeLinear"] == 0
    assert ops["QuantizeLinear"] == 0

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

    quant = onnxsim.quantize_weight_only(model)
    onnx.checker.check_model(quant)
    ops = _op_counts(quant)
    assert ops["Gemm"] == 1
    assert ops["DequantizeLinear"] == 1

    x = rng.standard_normal((3, K)).astype(np.float32)
    _assert_close(_run(model, {"X": x}), _run(quant, {"X": x}))


def test_quantize_conv():
    rng = np.random.default_rng(2)
    cout, cin = 8, 3
    weight = _f32(rng.standard_normal((cout, cin, 3, 3)) * 0.5, "W")
    model = _model(
        f"""
        g (float[1,{cin},16,16] X) => (float[1,{cout},16,16] Y)
        {{
          Y = Conv<kernel_shape = [3, 3], pads = [1, 1, 1, 1]>(X, W)
        }}
        """,
        initializer=[weight],
    )

    quant = onnxsim.quantize_weight_only(model)
    onnx.checker.check_model(quant)
    ops = _op_counts(quant)
    assert ops["Conv"] == 1
    assert ops["DequantizeLinear"] == 1
    assert ops["QuantizeLinear"] == 0

    x = rng.standard_normal((1, cin, 16, 16)).astype(np.float32)
    _assert_close(_run(model, {"X": x}), _run(quant, {"X": x}))


def test_quantize_conv_with_bias():
    rng = np.random.default_rng(3)
    cout, cin = 4, 2
    weight = _f32(rng.standard_normal((cout, cin, 3, 3)) * 0.5, "W")
    bias = _f32(rng.standard_normal(cout), "B")
    model = _model(
        f"""
        g (float[2,{cin},8,8] X) => (float[2,{cout},8,8] Y)
        {{
          Y = Conv<kernel_shape = [3, 3], pads = [1, 1, 1, 1]>(X, W, B)
        }}
        """,
        initializer=[weight, bias],
    )

    quant = onnxsim.quantize_weight_only(model)
    onnx.checker.check_model(quant)
    ops = _op_counts(quant)
    assert ops["Conv"] == 1
    assert ops["DequantizeLinear"] == 1
    # The bias is left in float, untouched, as Conv's third input.
    assert ops["Add"] == 0

    x = rng.standard_normal((2, cin, 8, 8)).astype(np.float32)
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
    quant = onnxsim.quantize_weight_only(model)
    assert _op_counts(quant)["MatMul"] == 1
    assert _op_counts(quant)["DequantizeLinear"] == 0


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
        initializer=[weight],
    )
    quant = onnxsim.quantize_weight_only(model)
    assert _op_counts(quant)["Gemm"] == 1
    assert _op_counts(quant)["DequantizeLinear"] == 0


def test_quantize_skips_old_opset():
    # DequantizeLinear's per-channel `axis` attribute needs opset >= 13.
    weight = _f32(np.random.randn(8, 4).astype(np.float32), "W")
    model = _model(
        """
        g (float[4,8] X) => (float[4,4] Y)
        {
          Y = MatMul(X, W)
        }
        """,
        initializer=[weight],
        opset=12,
    )
    quant = onnxsim.quantize_weight_only(model)
    assert _op_counts(quant)["MatMul"] == 1
    assert _op_counts(quant)["DequantizeLinear"] == 0
