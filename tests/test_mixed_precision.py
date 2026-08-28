"""Tests for ``onnxsim.apply_mixed_precision_quantization`` -- see
``onnxsim/mixed_precision.py`` for the technique (calibration-driven
per-layer choice between block-wise INT8 and block-wise INT4).
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


def _two_layer_model(K=32, H=16, N=8, seed=0, opset=21):
    rng = np.random.default_rng(seed)
    w1 = rng.standard_normal((K, H)).astype(np.float32) * 0.5
    # w2's rows get planted, large-magnitude outliers, making it far more
    # sensitive to INT4 quantization than w1 -- so a good sensitivity
    # ranking should pick THIS layer for the INT8 tier.
    w2 = rng.standard_normal((H, N)).astype(np.float32) * 0.05
    w2[0, :] = 20.0
    nodes = [
        onnx.helper.make_node("MatMul", ["X", "W1"], ["H1"]),
        onnx.helper.make_node("MatMul", ["H1", "W2"], ["Y"]),
    ]
    return _model(
        nodes,
        [_vi("X", ["batch", K])],
        [_vi("Y", ["batch", N])],
        [_f32(w1, "W1"), _f32(w2, "W2")],
        opset=opset,
    )


def test_mixed_precision_output_stays_close_to_float_via_onnxruntime():
    model = _two_layer_model(K=32, H=16, N=8, seed=0)
    q = onnxsim.apply_mixed_precision_quantization(
        model, block_size=8, high_bits_fraction=0.5, num_samples=16, seed=1
    )
    onnx.checker.check_model(q)

    rng = np.random.default_rng(2)
    x = rng.standard_normal((8, 32)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.3


def test_mixed_precision_picks_the_more_sensitive_layer_for_int8():
    model = _two_layer_model(K=32, H=16, N=8, seed=3)
    q = onnxsim.apply_mixed_precision_quantization(
        model, block_size=8, high_bits_fraction=0.5, num_samples=32, seed=4
    )
    codes_by_prefix = {
        t.name: t for t in q.graph.initializer if t.name.endswith("_codes")
    }
    w2_codes = next(t for name, t in codes_by_prefix.items() if name.startswith("W2_"))
    w1_codes = next(t for name, t in codes_by_prefix.items() if name.startswith("W1_"))
    # W2 has the planted outlier row -- it must be the INT8 (more precise)
    # tier, while the ordinary W1 stays at INT4.
    assert w2_codes.data_type == onnx.TensorProto.INT8
    assert w1_codes.data_type == onnx.TensorProto.INT4


def test_mixed_precision_zero_fraction_matches_all_int4():
    model = _two_layer_model(K=32, H=16, N=8, seed=5)
    q = onnxsim.apply_mixed_precision_quantization(
        model, block_size=8, high_bits_fraction=0.0, num_samples=16, seed=6
    )
    codes = [t for t in q.graph.initializer if t.name.endswith("_codes")]
    assert len(codes) == 2
    assert all(t.data_type == onnx.TensorProto.INT4 for t in codes)


def test_mixed_precision_one_fraction_matches_all_int8():
    model = _two_layer_model(K=32, H=16, N=8, seed=7)
    q = onnxsim.apply_mixed_precision_quantization(
        model, block_size=8, high_bits_fraction=1.0, num_samples=16, seed=8
    )
    codes = [t for t in q.graph.initializer if t.name.endswith("_codes")]
    assert len(codes) == 2
    assert all(t.data_type == onnx.TensorProto.INT8 for t in codes)


def test_mixed_precision_declines_when_k_not_divisible_by_block_size():
    rng = np.random.default_rng(9)
    weight = rng.standard_normal((20, 4)).astype(np.float32)  # 20 not a multiple of 8
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    model = _model(
        nodes, [_vi("X", ["batch", 20])], [_vi("Y", ["batch", 4])], [_f32(weight, "W")]
    )
    q = onnxsim.apply_mixed_precision_quantization(model, block_size=8)
    assert q.SerializeToString() == model.SerializeToString()


def test_mixed_precision_declines_non_constant_weight():
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    model = _model(
        nodes, [_vi("X", [4, 32]), _vi("W", [32, 4])], [_vi("Y", [4, 4])], []
    )
    q = onnxsim.apply_mixed_precision_quantization(model)
    assert q.SerializeToString() == model.SerializeToString()


def test_mixed_precision_noop_when_no_matmul_present():
    nodes = [onnx.helper.make_node("Relu", ["X"], ["Y"])]
    model = _model(nodes, [_vi("X", [4, 4])], [_vi("Y", [4, 4])], [])
    result = onnxsim.apply_mixed_precision_quantization(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_mixed_precision_declines_below_opset21():
    model = _two_layer_model(K=32, H=16, N=8, opset=13)
    result = onnxsim.apply_mixed_precision_quantization(model)
    assert result.SerializeToString() == model.SerializeToString()
