"""Tests for ``onnxsim.quantize_weight_only_int4`` (the
``weight_only_quantize_int4_matmul`` C++ pass).

Each model is built directly with ``onnx.helper`` (no torch dependency),
quantized, and then actually run through ONNX Runtime -- both before and
after quantization -- so these tests double as a minimal end-to-end
simplify/quantize/deploy check: the quantized graph must load and execute
under a real inference engine, and its outputs must stay close to the float
baseline. Needs opset 21 for INT4 tensors and DequantizeLinear's block_size
attribute.
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
    # Pin a low-ish IR version so the model loads under older onnxruntime
    # builds, matching test_fusion_patterns.py -- IR version 10 supports
    # opset 21 fine (IR version only gates the *envelope*, not individual op
    # opsets).
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
    # INT4 weight-only quantization is considerably lossier than INT8 (16
    # levels per block instead of 255), so this needs more headroom than
    # test_weight_only_quantize.py's INT8 tests -- see that file's
    # identically-named helper for why aggregate relative L2 error is used
    # instead of a tight per-element bound.
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

    quant = onnxsim.quantize_weight_only_int4(model)
    onnx.checker.check_model(quant)
    ops = _op_counts(quant)
    assert ops["MatMul"] == 1
    assert ops["DequantizeLinear"] == 1

    w_init = next(t for t in quant.graph.initializer if t.name != "W")
    assert w_init.data_type == onnx.TensorProto.INT4

    x = rng.standard_normal((4, K)).astype(np.float32)
    _assert_close(_run(model, {"X": x}), _run(quant, {"X": x}))


def test_quantize_gemm_transb_with_bias():
    # PyTorch's nn.Linear layout: weight is [out_features, in_features], i.e.
    # [N, K], exported as Gemm(X, W, B, transB=1) -- the common real-world case.
    rng = np.random.default_rng(1)
    K, N = 96, 12
    weight = _f32(rng.standard_normal((N, K)) * 0.5, "W")
    bias = _f32(rng.standard_normal(N), "B")
    nodes = [onnx.helper.make_node("Gemm", ["X", "W", "B"], ["Y"], transB=1)]
    model = _model(nodes, [_vi("X", [3, K])], [_vi("Y", [3, N])], [weight, bias])

    quant = onnxsim.quantize_weight_only_int4(model)
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

    quant = onnxsim.quantize_weight_only_int4(model)
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

    quant = onnxsim.quantize_weight_only_int4(model)
    assert _op_counts(quant)["MatMul"] == 1
    assert _op_counts(quant)["DequantizeLinear"] == 0


def test_quantize_skips_non_constant_weight():
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    model = _model(
        nodes, [_vi("X", [4, 64]), _vi("W", [64, 4])], [_vi("Y", [4, 4])], []
    )
    quant = onnxsim.quantize_weight_only_int4(model)
    assert _op_counts(quant)["MatMul"] == 1


def test_quantize_skips_old_opset():
    # INT4 tensors and DequantizeLinear's block_size both need opset >= 21.
    weight = _f32(np.random.default_rng(4).standard_normal((64, 4)), "W")
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    model = _model(nodes, [_vi("X", [4, 64])], [_vi("Y", [4, 4])], [weight], opset=20)
    quant = onnxsim.quantize_weight_only_int4(model)
    assert _op_counts(quant)["MatMul"] == 1
    assert _op_counts(quant)["DequantizeLinear"] == 0
