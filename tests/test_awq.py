"""Tests for ``onnxsim.apply_awq`` (AWQ -- Activation-aware Weight
Quantization, see ``onnxsim/awq.py``) -- rescales each INT4-quantized
MatMul/Gemm layer's weight columns in proportion to their own input
channel's average activation magnitude before re-quantizing, protecting
"salient" channels' relative precision, with a compensating ``Mul`` inserted
on the activation so the transformation is exact pre-quantization.
"""

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper
import pytest

import onnxsim

ort = pytest.importorskip("onnxruntime")


def _vi(name, shape):
    return onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _model(nodes, inputs, outputs, initializer, opset=21):
    graph = onnx.helper.make_graph(nodes, "g", inputs, outputs, initializer)
    return onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", opset)], ir_version=10
    )


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-6)


def _matmul_model(K=64, N=16, seed=0):
    rng = np.random.default_rng(seed)
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    return _model(
        nodes, [_vi("X", ["batch", K])], [_vi("Y", ["batch", N])], [_f32(weight, "W")]
    )


def _salient_channel_calibration(K=64, num_samples=32, salient_channels=(3, 7), seed=1):
    # A calibration set with a handful of channels carrying much larger
    # activation magnitude than the rest -- AWQ's own motivating scenario:
    # quantization error on those channels' weight columns costs the
    # output disproportionately, so protecting them should measurably help.
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((num_samples, K)).astype(np.float32)
    for c in salient_channels:
        x[:, c] *= 20.0
    return x


def _dequantize_int4(model):
    dq_node = next(n for n in model.graph.node if n.op_type == "DequantizeLinear")
    # Fetch Wq/Ws by the DequantizeLinear node's own input names, not by
    # scanning for "some tensor of this dtype": quantize_weight_only_int4
    # never prunes the original (now-dead) float32 weight initializer, so a
    # dtype-only scan can silently grab that instead of the real scale.
    wq = next(t for t in model.graph.initializer if t.name == dq_node.input[0])
    ws = next(t for t in model.graph.initializer if t.name == dq_node.input[1])
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
    return codes * scale_full[tuple(slicer)]


def test_awq_reduces_reconstruction_error_with_salient_channels():
    model = _matmul_model(K=64, N=16, seed=0)
    x = _salient_channel_calibration(K=64, num_samples=64, seed=1)
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_weight_only_int4(model)
    w_float = onnx.numpy_helper.to_array(model.graph.initializer[0]).astype(np.float64)
    w_rtn = _dequantize_int4(quant)
    y_float = x.astype(np.float64) @ w_float
    y_rtn = x.astype(np.float64) @ w_rtn
    rtn_err = np.linalg.norm(y_float - y_rtn)

    awq_model = onnxsim.apply_awq(model, quant, calibration_data=calibration_data)
    onnx.checker.check_model(awq_model)

    # AWQ must have actually inserted its compensating Mul -- otherwise
    # this is just re-testing plain RTN.
    assert any(n.op_type == "Mul" for n in awq_model.graph.node)

    (float_y,) = _run(model, {"X": x.astype(np.float32)})
    (awq_y,) = _run(awq_model, {"X": x.astype(np.float32)})
    awq_err = np.linalg.norm(y_float - awq_y.astype(np.float64))
    assert awq_err < rtn_err


def test_awq_output_stays_close_to_float_via_onnxruntime():
    model = _matmul_model(K=64, N=16, seed=2)
    x = _salient_channel_calibration(K=64, num_samples=32, seed=3)
    calibration_data = [{"X": x}]

    awq_model = onnxsim.apply_awq(
        model,
        onnxsim.quantize_weight_only_int4(model),
        calibration_data=calibration_data,
    )
    onnx.checker.check_model(awq_model)

    (float_y,) = _run(model, {"X": x})
    (awq_y,) = _run(awq_model, {"X": x})
    assert np.all(np.isfinite(awq_y))
    assert _rel_l2(float_y, awq_y) < 0.25


def test_awq_preserves_function_when_alpha_zero_is_best():
    # Uniform activation magnitude across channels -- no channel is more
    # "salient" than any other, so AWQ's own alpha=0 grid point (plain RTN,
    # no rescaling) should win, and the model should come back completely
    # untouched (no Mul inserted, same INT4 codes as plain RTN).
    model = _matmul_model(K=32, N=8, seed=4)
    rng = np.random.default_rng(5)
    x = rng.standard_normal((16, 32)).astype(np.float32)  # no salient channels
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_weight_only_int4(model)
    awq_model = onnxsim.apply_awq(model, quant, calibration_data=calibration_data)

    assert awq_model.SerializeToString() == quant.SerializeToString()


def test_awq_gemm_transb():
    rng = np.random.default_rng(6)
    K, N = 96, 12
    weight = rng.standard_normal((N, K)).astype(np.float32) * 0.5
    nodes = [onnx.helper.make_node("Gemm", ["X", "W"], ["Y"], transB=1)]
    model = _model(
        nodes, [_vi("X", ["batch", K])], [_vi("Y", ["batch", N])], [_f32(weight, "W")]
    )
    quant = onnxsim.quantize_weight_only_int4(model)
    onnx.checker.check_model(quant)

    x = _salient_channel_calibration(
        K=K, num_samples=32, salient_channels=(10, 50), seed=7
    )
    calibration_data = [{"X": x}]

    awq_model = onnxsim.apply_awq(model, quant, calibration_data=calibration_data)
    onnx.checker.check_model(awq_model)

    (float_y,) = _run(model, {"X": x})
    (awq_y,) = _run(awq_model, {"X": x})
    assert _rel_l2(float_y, awq_y) < 0.25


def test_awq_codes_stay_in_range():
    model = _matmul_model(K=32, N=8, seed=8)
    x = _salient_channel_calibration(
        K=32, num_samples=16, salient_channels=(1,), seed=9
    )
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_weight_only_int4(model)
    awq_model = onnxsim.apply_awq(model, quant, calibration_data=calibration_data)
    wq = next(
        t for t in awq_model.graph.initializer if t.data_type == onnx.TensorProto.INT4
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


def test_awq_noop_when_no_int4_matmul_present():
    nodes = [onnx.helper.make_node("Relu", ["X"], ["Y"])]
    model = _model(nodes, [_vi("X", [4, 4])], [_vi("Y", [4, 4])], [])
    result = onnxsim.apply_awq(
        model, model, calibration_data=[{"X": np.zeros((4, 4), dtype=np.float32)}]
    )
    assert result.SerializeToString() == model.SerializeToString()
