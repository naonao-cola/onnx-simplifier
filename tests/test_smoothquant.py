"""Tests for ``onnxsim.apply_smoothquant`` (SmoothQuant, see
``onnxsim/smoothquant.py``) -- migrates per-channel quantization difficulty
between a MatMul/Gemm's activation and its weight (``s_j = max(|X_j|)**alpha
/ max(|W_j|)**(1-alpha)``, dividing the activation and multiplying the
weight by the same ``s``), a lossless pre-conditioning transform meant to run
ahead of a separate W8A8 quantizer.
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


def _model(nodes, inputs, outputs, initializer, opset=13):
    graph = onnx.helper.make_graph(nodes, "g", inputs, outputs, initializer)
    return onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", opset)], ir_version=8
    )


def _matmul_model(K=64, N=16, weight=None, seed=0):
    if weight is None:
        rng = np.random.default_rng(seed)
        weight = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    return _model(
        nodes, [_vi("X", ["batch", K])], [_vi("Y", ["batch", N])], [_f32(weight, "W")]
    )


def _run(model, feeds, output_names=None):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    names = output_names or [o.name for o in sess.get_outputs()]
    return dict(zip(names, sess.run(names, feeds)))


def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-6)


def _outlier_channel_calibration(K=64, num_samples=64, outlier_channels=(3, 7), seed=1):
    # SmoothQuant's own motivating scenario: a few activation channels
    # consistently much larger than the rest.
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((num_samples, K)).astype(np.float32)
    for i, c in enumerate(outlier_channels):
        x[:, c] *= 20.0 + 10.0 * i
    return x


def _with_extra_output(model, name):
    out = onnx.ModelProto()
    out.CopyFrom(model)
    existing = {o.name for o in out.graph.output}
    if name not in existing:
        out.graph.output.append(onnx.ValueInfoProto(name=name))
    return out


def test_smoothquant_output_matches_float_almost_exactly():
    # Unlike every other onnxsim PTQ technique, SmoothQuant does not
    # quantize anything itself -- the migration is exact real-number math,
    # so its own output should match the float model far more tightly than
    # any lossy INT4/INT8 scheme's tolerance.
    model = _matmul_model(K=64, N=16, seed=0)
    x = _outlier_channel_calibration(K=64, num_samples=64, seed=1)
    calibration_data = [{"X": x}]

    sq_model = onnxsim.apply_smoothquant(model, calibration_data=calibration_data)
    onnx.checker.check_model(sq_model)
    assert any(n.op_type == "Mul" for n in sq_model.graph.node)

    float_out = _run(model, {"X": x})
    sq_out = _run(sq_model, {"X": x}, output_names=["Y"])
    assert np.all(np.isfinite(sq_out["Y"]))
    assert _rel_l2(float_out["Y"], sq_out["Y"]) < 1e-4


def test_smoothquant_alpha_one_normalizes_activation_channels():
    # alpha=1 sets s_j = max(|X_j|) exactly, so the smoothed activation
    # (the inserted Mul's own output) should have per-channel max-abs == 1.
    K = 32
    model = _matmul_model(K=K, N=8, seed=2)
    x = _outlier_channel_calibration(
        K=K, num_samples=32, outlier_channels=(1, 5, 20), seed=3
    )
    calibration_data = [{"X": x}]

    sq_model = onnxsim.apply_smoothquant(
        model, calibration_data=calibration_data, alpha=1.0
    )
    mul_node = next(n for n in sq_model.graph.node if n.op_type == "Mul")
    probe_model = _with_extra_output(sq_model, mul_node.output[0])
    result = _run(probe_model, {"X": x}, output_names=[mul_node.output[0]])
    smoothed = result[mul_node.output[0]]
    assert np.allclose(np.abs(smoothed).max(axis=0), 1.0, atol=1e-3)


def test_smoothquant_alpha_zero_normalizes_weight_columns():
    # alpha=0 sets s_j = 1 / max(|W_j|) exactly, so every rescaled weight
    # column's own max-abs should become 1.
    K = 32
    rng = np.random.default_rng(4)
    weight = rng.standard_normal((K, 8)).astype(np.float32) * rng.uniform(
        0.1, 5.0, size=(1, 8)
    ).astype(np.float32)
    model = _matmul_model(K=K, N=8, weight=weight)
    x = _outlier_channel_calibration(K=K, num_samples=16, outlier_channels=(2,), seed=5)
    calibration_data = [{"X": x}]

    sq_model = onnxsim.apply_smoothquant(
        model, calibration_data=calibration_data, alpha=0.0
    )
    w_new = onnx.numpy_helper.to_array(sq_model.graph.initializer[0])
    # Weight is stored [K, N] (plain MatMul, not transposed): the alpha=0
    # invariant holds per input channel K (axis 0), reduced over N (axis 1).
    assert np.allclose(np.abs(w_new).max(axis=1), 1.0, atol=1e-3)


def test_smoothquant_reduces_activation_dynamic_range_on_outlier_channels():
    K = 64
    model = _matmul_model(K=K, N=16, seed=6)
    x = _outlier_channel_calibration(
        K=K, num_samples=64, outlier_channels=(3, 7), seed=7
    )
    calibration_data = [{"X": x}]

    sq_model = onnxsim.apply_smoothquant(model, calibration_data=calibration_data)
    mul_node = next(n for n in sq_model.graph.node if n.op_type == "Mul")
    probe_model = _with_extra_output(sq_model, mul_node.output[0])
    result = _run(probe_model, {"X": x}, output_names=[mul_node.output[0]])
    smoothed = result[mul_node.output[0]]

    float_range = np.abs(x.astype(np.float64)).max(axis=0)
    smoothed_range = np.abs(smoothed).max(axis=0)
    # The overall spread between the largest and smallest channel range
    # should shrink -- the entire point of migrating the outlier channels'
    # difficulty into the weight.
    assert (smoothed_range.max() / smoothed_range.min()) < (
        float_range.max() / float_range.min()
    )


def test_smoothquant_gemm_transb():
    rng = np.random.default_rng(8)
    K, N = 96, 12
    weight = rng.standard_normal((N, K)).astype(np.float32) * 0.5
    nodes = [onnx.helper.make_node("Gemm", ["X", "W"], ["Y"], transB=1)]
    model = _model(
        nodes, [_vi("X", ["batch", K])], [_vi("Y", ["batch", N])], [_f32(weight, "W")]
    )
    x = _outlier_channel_calibration(
        K=K, num_samples=32, outlier_channels=(10, 50), seed=9
    )
    calibration_data = [{"X": x}]

    sq_model = onnxsim.apply_smoothquant(model, calibration_data=calibration_data)
    onnx.checker.check_model(sq_model)

    float_out = _run(model, {"X": x})
    sq_out = _run(sq_model, {"X": x}, output_names=["Y"])
    assert _rel_l2(float_out["Y"], sq_out["Y"]) < 1e-4


def test_smoothquant_noop_when_no_matmul_present():
    nodes = [onnx.helper.make_node("Relu", ["X"], ["Y"])]
    model = _model(nodes, [_vi("X", [4, 4])], [_vi("Y", [4, 4])], [])
    result = onnxsim.apply_smoothquant(
        model, calibration_data=[{"X": np.zeros((4, 4), dtype=np.float32)}]
    )
    assert result.SerializeToString() == model.SerializeToString()


def test_smoothquant_skips_non_constant_weight():
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    model = _model(
        nodes, [_vi("X", [4, 64]), _vi("W", [64, 4])], [_vi("Y", [4, 4])], []
    )
    result = onnxsim.apply_smoothquant(
        model, calibration_data=[{"X": np.zeros((4, 64), dtype=np.float32)}]
    )
    assert result.SerializeToString() == model.SerializeToString()


def test_smoothquant_skips_non_2d_activation():
    # A 3-D activation (e.g. [batch, seq, hidden], typical of an
    # ONNX-exported transformer) isn't a plain 2-D tensor -- matches this
    # module's own documented scope, same as onnxsim.apply_awq.
    K, N = 16, 8
    rng = np.random.default_rng(10)
    weight = rng.standard_normal((K, N)).astype(np.float32)
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    model = _model(
        nodes,
        [_vi("X", ["batch", "seq", K])],
        [_vi("Y", ["batch", "seq", N])],
        [_f32(weight, "W")],
    )
    x = rng.standard_normal((2, 3, K)).astype(np.float32)
    result = onnxsim.apply_smoothquant(model, calibration_data=[{"X": x}])
    assert result.SerializeToString() == model.SerializeToString()
