"""Tests for ``onnxsim.quantize_weight_only_int16`` (the
``weight_only_quantize_int16_matmul``/``weight_only_quantize_int16_conv``
C++ passes).

Structurally identical to ``test_weight_only_quantize.py``'s INT8 tests --
same models, same real ``onnxruntime.InferenceSession`` execution -- just
checking the INT16 scheme's finer resolution (and its opset >= 21
requirement, unlike INT8's opset >= 13) instead.
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


def _initializer_by_name(model, name):
    for init in model.graph.initializer:
        if init.name == name:
            return init
    raise KeyError(name)


def _quantized_weight_initializer(model):
    # weight_only_quantize_{matmul,conv}'s rewrite inserts
    # Wdq = DequantizeLinear(Wq, Ws, ...) and rewires the consuming
    # MatMul/Gemm/Conv to take Wdq (not Wq) as input -- so the quantized
    # weight itself is DequantizeLinear's own first input, not the
    # consuming node's.
    dql = next(n for n in model.graph.node if n.op_type == "DequantizeLinear")
    return _initializer_by_name(model, dql.input[0])


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

    quant = onnxsim.quantize_weight_only_int16(model)
    onnx.checker.check_model(quant)
    ops = _op_counts(quant)
    assert ops["MatMul"] == 1
    assert ops["DequantizeLinear"] == 1
    assert ops["QuantizeLinear"] == 0

    w_init = _quantized_weight_initializer(quant)
    assert w_init.data_type == onnx.TensorProto.INT16

    x = rng.standard_normal((4, K)).astype(np.float32)
    _assert_close(_run(model, {"X": x}), _run(quant, {"X": x}))


def test_quantize_matmul_more_precise_than_int8():
    # INT16's ~8x finer step should track the float baseline noticeably more
    # closely than quantize_weight_only's INT8 scheme, for the same weight.
    rng = np.random.default_rng(4)
    K, N = 32, 16
    weight = _f32(rng.standard_normal((K, N)) * 0.5, "W")
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    model = _model(nodes, [_vi("X", [4, K])], [_vi("Y", [4, N])], [weight], opset=13)

    quant8 = onnxsim.quantize_weight_only(model)
    quant16 = onnxsim.quantize_weight_only_int16(model)

    x = rng.standard_normal((4, K)).astype(np.float32)
    float_out = _run(model, {"X": x})[0]
    out8 = _run(quant8, {"X": x})[0]
    out16 = _run(quant16, {"X": x})[0]

    err8 = np.linalg.norm(float_out - out8)
    err16 = np.linalg.norm(float_out - out16)
    assert err16 < err8


def test_quantize_gemm_transb_with_bias():
    rng = np.random.default_rng(1)
    K, N = 24, 12
    weight = _f32(rng.standard_normal((N, K)) * 0.5, "W")
    bias = _f32(rng.standard_normal(N), "B")
    nodes = [onnx.helper.make_node("Gemm", ["X", "W", "B"], ["Y"], transB=1)]
    model = _model(nodes, [_vi("X", [3, K])], [_vi("Y", [3, N])], [weight, bias])

    quant = onnxsim.quantize_weight_only_int16(model)
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
    nodes = [
        onnx.helper.make_node(
            "Conv", ["X", "W"], ["Y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]
        )
    ]
    model = _model(
        nodes, [_vi("X", [1, cin, 16, 16])], [_vi("Y", [1, cout, 16, 16])], [weight]
    )

    quant = onnxsim.quantize_weight_only_int16(model)
    onnx.checker.check_model(quant)
    ops = _op_counts(quant)
    assert ops["Conv"] == 1
    assert ops["DequantizeLinear"] == 1
    assert ops["QuantizeLinear"] == 0

    w_init = _quantized_weight_initializer(quant)
    assert w_init.data_type == onnx.TensorProto.INT16

    x = rng.standard_normal((1, cin, 16, 16)).astype(np.float32)
    _assert_close(_run(model, {"X": x}), _run(quant, {"X": x}))


def test_quantize_conv_with_bias():
    rng = np.random.default_rng(3)
    cout, cin = 4, 2
    weight = _f32(rng.standard_normal((cout, cin, 3, 3)) * 0.5, "W")
    bias = _f32(rng.standard_normal(cout), "B")
    nodes = [
        onnx.helper.make_node(
            "Conv", ["X", "W", "B"], ["Y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]
        )
    ]
    model = _model(
        nodes,
        [_vi("X", [2, cin, 8, 8])],
        [_vi("Y", [2, cout, 8, 8])],
        [weight, bias],
    )

    quant = onnxsim.quantize_weight_only_int16(model)
    onnx.checker.check_model(quant)
    ops = _op_counts(quant)
    assert ops["Conv"] == 1
    assert ops["DequantizeLinear"] == 1
    assert ops["Add"] == 0

    x = rng.standard_normal((2, cin, 8, 8)).astype(np.float32)
    _assert_close(_run(model, {"X": x}), _run(quant, {"X": x}))


def test_quantize_skips_non_constant_weight():
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    model = _model(nodes, [_vi("X", [4, 8]), _vi("W", [8, 4])], [_vi("Y", [4, 4])], [])
    quant = onnxsim.quantize_weight_only_int16(model)
    assert _op_counts(quant)["MatMul"] == 1
    assert _op_counts(quant)["DequantizeLinear"] == 0


def test_quantize_skips_non_default_gemm_attrs():
    weight = _f32(np.random.randn(8, 4).astype(np.float32), "W")
    nodes = [onnx.helper.make_node("Gemm", ["X", "W"], ["Y"], alpha=2.0)]
    model = _model(nodes, [_vi("X", [4, 8])], [_vi("Y", [4, 4])], [weight])
    quant = onnxsim.quantize_weight_only_int16(model)
    assert _op_counts(quant)["Gemm"] == 1
    assert _op_counts(quant)["DequantizeLinear"] == 0


def test_quantize_skips_old_opset():
    # INT16 QuantizeLinear/DequantizeLinear needs opset >= 21 -- unlike INT8's
    # opset >= 13, an opset in [13, 21) is old for this pass even though it
    # would be fine for quantize_weight_only.
    weight = _f32(np.random.randn(8, 4).astype(np.float32), "W")
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    model = _model(nodes, [_vi("X", [4, 8])], [_vi("Y", [4, 4])], [weight], opset=13)
    quant = onnxsim.quantize_weight_only_int16(model)
    assert _op_counts(quant)["MatMul"] == 1
    assert _op_counts(quant)["DequantizeLinear"] == 0
