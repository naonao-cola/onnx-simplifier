"""Tests for ``onnxsim.quantize_static_int16`` (the
``static_quantize_int16_matmul``/``static_quantize_int16_conv`` C++ passes)
and their reuse of ``onnxsim.calibration``'s helpers.

Structurally identical to ``test_static_quantize_matmul.py``'s coverage --
same models, same real ``onnxruntime.InferenceSession`` execution -- just
checking the "W8A16" scheme's uint16 activation quantization (and its
opset >= 21 requirement, unlike ``quantize_static``'s opset >= 13) instead.
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
    K, N = 32, 16
    weight = _f32(rng.standard_normal((K, N)) * 0.5, "W")
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    model = _model(nodes, [_vi("X", [4, K])], [_vi("Y", [4, N])], [weight])

    quant = onnxsim.quantize_static_int16(model, num_calibration_samples=16, seed=0)
    onnx.checker.check_model(quant)
    ops = _op_counts(quant)
    assert ops["MatMul"] == 1  # the MatMul node itself is kept (QDQ format)
    assert ops["QuantizeLinear"] == 1
    assert ops["DequantizeLinear"] == 2  # one for X, one for W

    x = rng.standard_normal((4, K)).astype(np.float32)
    _assert_close(_run(model, {"X": x}), _run(quant, {"X": x}))


def test_quantize_more_precise_than_uint8():
    # A uint16 QDQ round trip should track the float baseline noticeably
    # more closely than quantize_static's uint8 scheme, for the same
    # calibrated activation range.
    rng = np.random.default_rng(4)
    K, N = 32, 16
    weight = _f32(rng.standard_normal((K, N)) * 0.5, "W")
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    model = _model(nodes, [_vi("X", [4, K])], [_vi("Y", [4, N])], [weight], opset=13)

    quant8 = onnxsim.quantize_static(model, num_calibration_samples=16, seed=0)
    model21 = _model(nodes, [_vi("X", [4, K])], [_vi("Y", [4, N])], [weight])
    quant16 = onnxsim.quantize_static_int16(model21, num_calibration_samples=16, seed=0)

    x = rng.standard_normal((4, K)).astype(np.float32)
    float_out = _run(model, {"X": x})[0]
    out8 = _run(quant8, {"X": x})[0]
    out16 = _run(quant16, {"X": x})[0]

    err8 = np.linalg.norm(float_out - out8)
    err16 = np.linalg.norm(float_out - out16)
    assert err16 < err8


def test_quantize_gemm_transb_with_bias():
    # PyTorch's nn.Linear layout: weight is [out_features, in_features], i.e.
    # [N, K], exported as Gemm(X, W, B, transB=1) -- the common real-world case.
    rng = np.random.default_rng(1)
    K, N = 24, 12
    weight = _f32(rng.standard_normal((N, K)) * 0.5, "W")
    bias = _f32(rng.standard_normal(N), "B")
    nodes = [onnx.helper.make_node("Gemm", ["X", "W", "B"], ["Y"], transB=1)]
    model = _model(nodes, [_vi("X", [3, K])], [_vi("Y", [3, N])], [weight, bias])

    quant = onnxsim.quantize_static_int16(model, num_calibration_samples=16, seed=1)
    onnx.checker.check_model(quant)
    ops = _op_counts(quant)
    assert ops["Gemm"] == 1
    assert ops["QuantizeLinear"] == 1
    assert ops["DequantizeLinear"] == 2

    x = rng.standard_normal((3, K)).astype(np.float32)
    _assert_close(_run(model, {"X": x}), _run(quant, {"X": x}))


def test_quantize_conv():
    rng = np.random.default_rng(2)
    cout, cin = 8, 3
    weight = _f32(rng.standard_normal((cout, cin, 3, 3)) * 0.5, "W")
    nodes = [
        onnx.helper.make_node(
            "Conv", ["X", "W"], ["Y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]
        )
    ]
    model = _model(
        nodes, [_vi("X", [1, cin, 16, 16])], [_vi("Y", [1, cout, 16, 16])], [weight]
    )

    quant = onnxsim.quantize_static_int16(model, num_calibration_samples=16, seed=2)
    onnx.checker.check_model(quant)
    ops = _op_counts(quant)
    assert ops["Conv"] == 1
    assert ops["QuantizeLinear"] == 1
    assert ops["DequantizeLinear"] == 2

    x = rng.standard_normal((1, cin, 16, 16)).astype(np.float32)
    _assert_close(_run(model, {"X": x}), _run(quant, {"X": x}))


def test_quantize_skips_non_constant_weight():
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    model = _model(nodes, [_vi("X", [4, 8]), _vi("W", [8, 4])], [_vi("Y", [4, 4])], [])
    quant = onnxsim.quantize_static_int16(model)
    assert _op_counts(quant)["MatMul"] == 1


def test_quantize_skips_non_default_gemm_attrs():
    # alpha != 1 falls outside the "vanilla" Gemm shape this pass handles.
    weight = _f32(np.random.randn(8, 4).astype(np.float32), "W")
    nodes = [onnx.helper.make_node("Gemm", ["X", "W"], ["Y"], alpha=2.0)]
    model = _model(nodes, [_vi("X", [4, 8])], [_vi("Y", [4, 4])], [weight])
    quant = onnxsim.quantize_static_int16(model)
    assert _op_counts(quant)["Gemm"] == 1


def test_quantize_skips_old_opset():
    # uint16 QuantizeLinear/DequantizeLinear needs opset >= 21 -- unlike
    # quantize_static's opset >= 13, an opset in [13, 21) is old for this
    # pass even though it would be fine for the uint8 one.
    weight = _f32(np.random.randn(8, 4).astype(np.float32), "W")
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    model = _model(nodes, [_vi("X", [4, 8])], [_vi("Y", [4, 4])], [weight], opset=13)
    quant = onnxsim.quantize_static_int16(model)
    assert _op_counts(quant)["MatMul"] == 1
    assert _op_counts(quant)["QuantizeLinear"] == 0
