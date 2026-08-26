"""Tests for ``onnxsim.apply_adaround`` (AIMET's Adaptive Rounding, see
``onnxsim/adaround.py``) -- optimizes each INT4-quantized MatMul/Gemm
layer's own per-element rounding decision (floor vs. ceil) to minimize that
layer's real reconstruction error, instead of the round-to-nearest every
``quantize_weight_only_int4`` layer starts out with.
"""

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper
import pytest

import onnxsim

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


def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-6)


def _matmul_matmul_int4_models(K=64, N=16, batch=4, seed=0, opset=21):
    rng = np.random.default_rng(seed)
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    float_model = _model(
        nodes,
        [_vi("X", [batch, K])],
        [_vi("Y", [batch, N])],
        [_f32(weight, "W")],
        opset=opset,
    )
    quant_model = onnxsim.quantize_weight_only_int4(float_model)
    return float_model, quant_model


def _dequantize_int4(quant_model):
    """Decodes the DequantizeLinear(Wq, Ws)-fed MatMul/Gemm's weight in
    ``quant_model`` back to a dense float array, using onnxruntime itself
    (so this stays independent of adaround.py's own internal math)."""
    # Feed a zero/one probe matrix through the model's own DequantizeLinear
    # node isn't directly possible without extracting it into its own
    # session, so instead this decodes by hand from the initializer bytes.
    wq = next(
        t for t in quant_model.graph.initializer if t.data_type == onnx.TensorProto.INT4
    )
    ws = next(
        t
        for t in quant_model.graph.initializer
        if t.data_type == onnx.TensorProto.FLOAT
    )
    dq_node = next(n for n in quant_model.graph.node if n.op_type == "DequantizeLinear")
    block_size = next(a.i for a in dq_node.attribute if a.name == "block_size")
    axis = next((a.i for a in dq_node.attribute if a.name == "axis"), 1)

    dims = list(wq.dims)
    numel = int(np.prod(dims))
    raw = np.frombuffer(wq.raw_data, dtype=np.uint8)
    lo = (raw & 0x0F).astype(np.int8)
    hi = ((raw >> 4) & 0x0F).astype(np.int8)
    lo = np.where(lo >= 8, lo - 16, lo)
    hi = np.where(hi >= 8, hi - 16, hi)
    codes = np.empty(numel, dtype=np.int8)
    codes[0::2] = lo[: (numel + 1) // 2]
    codes[1::2] = hi[: numel // 2]
    codes = codes.reshape(dims).astype(np.float64)

    scale = onnx.numpy_helper.to_array(ws).astype(np.float64)
    scale_full = np.repeat(scale, block_size, axis=axis)
    slicer = [slice(None)] * codes.ndim
    slicer[axis] = slice(0, codes.shape[axis])
    scale_full = scale_full[tuple(slicer)]
    return codes * scale_full


def test_adaround_reduces_reconstruction_error_vs_round_to_nearest():
    float_model, quant_model = _matmul_matmul_int4_models(K=64, N=16, batch=32, seed=1)
    rng = np.random.default_rng(2)
    x = rng.standard_normal((32, 64)).astype(np.float32)
    calibration_data = [{"X": x}]

    w_float = onnx.numpy_helper.to_array(float_model.graph.initializer[0]).astype(
        np.float64
    )
    w_rtn = _dequantize_int4(quant_model)
    y_float = x.astype(np.float64) @ w_float
    y_rtn = x.astype(np.float64) @ w_rtn
    rtn_err = np.linalg.norm(y_float - y_rtn)

    adaround_model = onnxsim.apply_adaround(
        float_model,
        quant_model,
        calibration_data=calibration_data,
        num_iterations=200,
    )
    w_ada = _dequantize_int4(adaround_model)
    y_ada = x.astype(np.float64) @ w_ada
    ada_err = np.linalg.norm(y_float - y_ada)

    assert ada_err < rtn_err


def test_adaround_output_stays_close_to_float_via_onnxruntime():
    float_model, quant_model = _matmul_matmul_int4_models(K=64, N=16, batch=16, seed=3)
    rng = np.random.default_rng(4)
    x = rng.standard_normal((16, 64)).astype(np.float32)
    calibration_data = [{"X": x}]

    adaround_model = onnxsim.apply_adaround(
        float_model, quant_model, calibration_data=calibration_data, num_iterations=200
    )
    onnx.checker.check_model(adaround_model)

    (float_y,) = _run(float_model, {"X": x})
    (ada_y,) = _run(adaround_model, {"X": x})
    assert np.all(np.isfinite(ada_y))
    assert _rel_l2(float_y, ada_y) < 0.25


def test_adaround_preserves_scale_and_shape():
    float_model, quant_model = _matmul_matmul_int4_models(K=32, N=8, seed=5)
    rng = np.random.default_rng(6)
    calibration_data = [{"X": rng.standard_normal((4, 32)).astype(np.float32)}]

    before_scale = onnx.numpy_helper.to_array(
        next(
            t
            for t in quant_model.graph.initializer
            if t.data_type == onnx.TensorProto.FLOAT
        )
    )
    adaround_model = onnxsim.apply_adaround(
        float_model, quant_model, calibration_data=calibration_data, num_iterations=50
    )
    after_scale = onnx.numpy_helper.to_array(
        next(
            t
            for t in adaround_model.graph.initializer
            if t.data_type == onnx.TensorProto.FLOAT
        )
    )
    np.testing.assert_array_equal(before_scale, after_scale)

    wq = next(
        t
        for t in adaround_model.graph.initializer
        if t.data_type == onnx.TensorProto.INT4
    )
    assert list(wq.dims) == [32, 8]


def test_adaround_codes_stay_in_range():
    float_model, quant_model = _matmul_matmul_int4_models(K=32, N=8, seed=7)
    rng = np.random.default_rng(8)
    calibration_data = [{"X": rng.standard_normal((4, 32)).astype(np.float32) * 3}]

    adaround_model = onnxsim.apply_adaround(
        float_model, quant_model, calibration_data=calibration_data, num_iterations=100
    )
    wq = next(
        t
        for t in adaround_model.graph.initializer
        if t.data_type == onnx.TensorProto.INT4
    )
    numel = int(np.prod(list(wq.dims)))
    raw = np.frombuffer(wq.raw_data, dtype=np.uint8)
    lo = (raw & 0x0F).astype(np.int8)
    hi = ((raw >> 4) & 0x0F).astype(np.int8)
    lo = np.where(lo >= 8, lo - 16, lo)
    hi = np.where(hi >= 8, hi - 16, hi)
    codes = np.empty(numel, dtype=np.int8)
    codes[0::2] = lo[: (numel + 1) // 2]
    codes[1::2] = hi[: numel // 2]
    assert np.all(codes >= -7) and np.all(codes <= 7)


def test_adaround_gemm_transb_with_bias():
    rng = np.random.default_rng(9)
    K, N = 96, 12
    weight = rng.standard_normal((N, K)).astype(np.float32) * 0.5
    bias = rng.standard_normal(N).astype(np.float32)
    nodes = [onnx.helper.make_node("Gemm", ["X", "W", "B"], ["Y"], transB=1)]
    float_model = _model(
        nodes,
        [_vi("X", [8, K])],
        [_vi("Y", [8, N])],
        [_f32(weight, "W"), _f32(bias, "B")],
    )
    quant_model = onnxsim.quantize_weight_only_int4(float_model)
    onnx.checker.check_model(quant_model)

    x = rng.standard_normal((8, K)).astype(np.float32)
    calibration_data = [{"X": x}]

    adaround_model = onnxsim.apply_adaround(
        float_model, quant_model, calibration_data=calibration_data, num_iterations=150
    )
    onnx.checker.check_model(adaround_model)

    (float_y,) = _run(float_model, {"X": x})
    (ada_y,) = _run(adaround_model, {"X": x})
    assert _rel_l2(float_y, ada_y) < 0.25


def test_adaround_noop_when_no_int4_matmul_present():
    nodes = [onnx.helper.make_node("Relu", ["X"], ["Y"])]
    model = _model(nodes, [_vi("X", [4, 4])], [_vi("Y", [4, 4])], [])
    result = onnxsim.apply_adaround(
        model, model, calibration_data=[{"X": np.zeros((4, 4), dtype=np.float32)}]
    )
    assert result.SerializeToString() == model.SerializeToString()
