"""Tests for ``onnxsim.apply_llm_int8`` (LLM.int8(), see
``onnxsim/llm_int8.py``) -- decomposes a MatMul/Gemm into an outlier float32
part (input channels whose activation magnitude exceeds a threshold
anywhere in calibration data) plus a vector-wise INT8 part (everything
else, quantized via ``MatMulInteger`` with a runtime per-row activation
scale and an offline per-output-channel weight scale).
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


def _model(nodes, inputs, outputs, initializer, opset=18):
    graph = onnx.helper.make_graph(nodes, "g", inputs, outputs, initializer)
    return onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", opset)], ir_version=9
    )


def _matmul_model(K=64, N=16, weight=None, seed=0, opset=18):
    if weight is None:
        rng = np.random.default_rng(seed)
        weight = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    return _model(
        nodes,
        [_vi("X", ["batch", K])],
        [_vi("Y", ["batch", N])],
        [_f32(weight, "W")],
        opset=opset,
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


def _outlier_channel_calibration(K=64, num_samples=64, outlier_channels=(3, 7), seed=1):
    # LLM.int8()'s own motivating scenario: a few activation channels
    # consistently far above the paper's own default threshold (6.0), the
    # rest comfortably below it.
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((num_samples, K)).astype(np.float32) * 0.5
    for c in outlier_channels:
        x[:, c] = rng.standard_normal(num_samples).astype(np.float32) * 3.0 + 10.0
    return x


def test_llm_int8_decomposes_and_stays_close_to_float():
    model = _matmul_model(K=64, N=16, seed=0)
    x = _outlier_channel_calibration(
        K=64, num_samples=64, outlier_channels=(3, 7), seed=1
    )
    calibration_data = [{"X": x}]

    q_model = onnxsim.apply_llm_int8(model, calibration_data=calibration_data)
    onnx.checker.check_model(q_model)

    op_types = {n.op_type for n in q_model.graph.node}
    assert "MatMulInteger" in op_types
    assert "Gather" in op_types  # the outlier/regular channel split

    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q_model, {"X": x})
    assert np.all(np.isfinite(q_y))
    # INT8 is coarse, but excluding the outlier channels entirely from
    # quantization should still keep this reasonably close.
    assert _rel_l2(float_y, q_y) < 0.15


def test_llm_int8_output_name_and_node_count_unchanged_elsewhere():
    # The rewritten layer's own output tensor name must be preserved (same
    # convention as every onnxsim quantize_*/apply_* pass), and an
    # unrelated node before/after it should be left completely alone.
    K, N = 32, 8
    rng = np.random.default_rng(2)
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    nodes = [
        onnx.helper.make_node("Relu", ["X0"], ["X"]),
        onnx.helper.make_node("MatMul", ["X", "W"], ["Y0"]),
        onnx.helper.make_node("Relu", ["Y0"], ["Y"]),
    ]
    model = _model(
        nodes,
        [_vi("X0", ["batch", K])],
        [_vi("Y", ["batch", N])],
        [_f32(weight, "W")],
    )
    x0 = _outlier_channel_calibration(
        K=K, num_samples=32, outlier_channels=(5,), seed=3
    )
    x0 = np.abs(x0)  # keep positive so Relu doesn't zero out the signal
    calibration_data = [{"X0": x0}]

    q_model = onnxsim.apply_llm_int8(model, calibration_data=calibration_data)
    onnx.checker.check_model(q_model)

    relu_nodes = [n for n in q_model.graph.node if n.op_type == "Relu"]
    assert [n.output[0] for n in relu_nodes] == ["X", "Y"]
    assert any(n.op_type == "Add" and n.output[0] == "Y0" for n in q_model.graph.node)


def test_llm_int8_gemm_with_bias():
    rng = np.random.default_rng(4)
    K, N = 48, 12
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    bias = rng.standard_normal(N).astype(np.float32)
    nodes = [onnx.helper.make_node("Gemm", ["X", "W", "B"], ["Y"])]
    model = _model(
        nodes,
        [_vi("X", ["batch", K])],
        [_vi("Y", ["batch", N])],
        [_f32(weight, "W"), _f32(bias, "B")],
    )
    x = _outlier_channel_calibration(
        K=K, num_samples=32, outlier_channels=(2, 30), seed=5
    )
    calibration_data = [{"X": x}]

    q_model = onnxsim.apply_llm_int8(model, calibration_data=calibration_data)
    onnx.checker.check_model(q_model)

    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q_model, {"X": x})
    assert _rel_l2(float_y, q_y) < 0.15


def test_llm_int8_gemm_transb():
    rng = np.random.default_rng(6)
    K, N = 80, 10
    weight = rng.standard_normal((N, K)).astype(np.float32) * 0.5
    nodes = [onnx.helper.make_node("Gemm", ["X", "W"], ["Y"], transB=1)]
    model = _model(
        nodes, [_vi("X", ["batch", K])], [_vi("Y", ["batch", N])], [_f32(weight, "W")]
    )
    x = _outlier_channel_calibration(
        K=K, num_samples=32, outlier_channels=(15, 60), seed=7
    )
    calibration_data = [{"X": x}]

    q_model = onnxsim.apply_llm_int8(model, calibration_data=calibration_data)
    onnx.checker.check_model(q_model)

    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q_model, {"X": x})
    assert _rel_l2(float_y, q_y) < 0.15


def test_llm_int8_noop_when_no_outlier_channels():
    # Every channel comfortably below the threshold -- nothing to
    # decompose, so the model should come back completely untouched.
    model = _matmul_model(K=32, N=8, seed=8)
    rng = np.random.default_rng(9)
    x = rng.standard_normal((16, 32)).astype(np.float32) * 0.1
    calibration_data = [{"X": x}]

    q_model = onnxsim.apply_llm_int8(model, calibration_data=calibration_data)
    assert q_model.SerializeToString() == model.SerializeToString()


def test_llm_int8_noop_when_every_channel_is_an_outlier():
    model = _matmul_model(K=8, N=4, seed=10)
    rng = np.random.default_rng(11)
    x = rng.standard_normal((16, 8)).astype(np.float32) * 20.0  # all channels "hot"
    calibration_data = [{"X": x}]

    q_model = onnxsim.apply_llm_int8(model, calibration_data=calibration_data)
    assert q_model.SerializeToString() == model.SerializeToString()


def test_llm_int8_noop_when_no_matmul_present():
    nodes = [onnx.helper.make_node("Relu", ["X"], ["Y"])]
    model = _model(nodes, [_vi("X", [4, 4])], [_vi("Y", [4, 4])], [])
    result = onnxsim.apply_llm_int8(
        model, calibration_data=[{"X": np.zeros((4, 4), dtype=np.float32)}]
    )
    assert result.SerializeToString() == model.SerializeToString()


def test_llm_int8_noop_on_old_opset():
    # ReduceMax's axes-as-input form needs opset >= 18.
    model = _matmul_model(K=32, N=8, seed=12, opset=13)
    x = _outlier_channel_calibration(
        K=32, num_samples=16, outlier_channels=(1,), seed=13
    )
    result = onnxsim.apply_llm_int8(model, calibration_data=[{"X": x}])
    assert result.SerializeToString() == model.SerializeToString()


def test_llm_int8_skips_non_constant_weight():
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    model = _model(
        nodes, [_vi("X", [4, 64]), _vi("W", [64, 4])], [_vi("Y", [4, 4])], []
    )
    result = onnxsim.apply_llm_int8(
        model, calibration_data=[{"X": np.zeros((4, 64), dtype=np.float32)}]
    )
    assert result.SerializeToString() == model.SerializeToString()
