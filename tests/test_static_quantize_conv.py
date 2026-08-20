"""Tests for ``onnxsim.quantize_static``'s Conv coverage (the
``static_quantize_conv`` C++ pass).

Mirrors ``test_static_quantize_matmul.py``: each model is built directly with
``onnx.helper``, calibrated with random data, quantized, and then actually
run through ONNX Runtime -- both before and after quantization -- so the
quantized graph must load and execute under a real inference engine, and its
outputs must stay close to the float baseline.
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


def _model(nodes, inputs, outputs, initializer, opset=13):
    graph = onnx.helper.make_graph(nodes, "g", inputs, outputs, initializer)
    # Pin a low IR version so the model loads under older onnxruntime builds
    # (which cap at IR version 11), matching test_fusion_patterns.py.
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


def _assert_close(float_outputs, quant_outputs):
    # INT8/uint8 quantization is lossy by design; see
    # test_dynamic_quantize_matmul.py's identically-named helper for why this
    # checks aggregate relative L2 error rather than a tight per-element bound.
    for f, q in zip(float_outputs, quant_outputs):
        f = np.asarray(f, dtype=np.float64).ravel()
        q = np.asarray(q, dtype=np.float64).ravel()
        assert np.all(np.isfinite(q))
        rel_l2 = np.linalg.norm(f - q) / max(np.linalg.norm(f), 1e-6)
        assert rel_l2 < 0.1, f"relative L2 error too large: {rel_l2:.4f}"


def test_quantize_conv():
    rng = np.random.default_rng(0)
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

    quant = onnxsim.quantize_static(model, num_calibration_samples=16, seed=0)
    onnx.checker.check_model(quant)
    ops = _op_counts(quant)
    assert ops["Conv"] == 1  # the Conv node itself is kept (QDQ format)
    assert ops["QuantizeLinear"] == 1
    assert ops["DequantizeLinear"] == 2  # one for X, one for W

    x = rng.standard_normal((1, cin, 16, 16)).astype(np.float32)
    _assert_close(_run(model, {"X": x}), _run(quant, {"X": x}))


def test_quantize_conv_with_bias():
    rng = np.random.default_rng(1)
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

    quant = onnxsim.quantize_static(model, num_calibration_samples=16, seed=1)
    onnx.checker.check_model(quant)
    ops = _op_counts(quant)
    # The Conv node itself (bias included) is kept; only its X/W inputs are
    # rerouted through QDQ pairs.
    assert ops["Conv"] == 1
    assert ops["QuantizeLinear"] == 1
    assert ops["DequantizeLinear"] == 2

    x = rng.standard_normal((2, cin, 8, 8)).astype(np.float32)
    _assert_close(_run(model, {"X": x}), _run(quant, {"X": x}))


def test_quantize_skips_non_constant_conv_weight():
    # Both Conv operands are graph inputs (neither is a constant), so there
    # is nothing to quantize ahead of time.
    nodes = [onnx.helper.make_node("Conv", ["X", "W"], ["Y"], kernel_shape=[3, 3])]
    model = _model(
        nodes,
        [_vi("X", [1, 3, 8, 8]), _vi("W", [4, 3, 3, 3])],
        [_vi("Y", [1, 4, 6, 6])],
        [],
    )
    quant = onnxsim.quantize_static(model)
    assert _op_counts(quant)["Conv"] == 1
    assert _op_counts(quant)["QuantizeLinear"] == 0


def test_quantize_skips_old_opset_conv():
    # DequantizeLinear's per-channel `axis` attribute needs opset >= 13.
    weight = _f32(np.random.randn(4, 3, 3, 3).astype(np.float32), "W")
    nodes = [onnx.helper.make_node("Conv", ["X", "W"], ["Y"], kernel_shape=[3, 3])]
    model = _model(
        nodes, [_vi("X", [1, 3, 8, 8])], [_vi("Y", [1, 4, 6, 6])], [weight], opset=12
    )
    quant = onnxsim.quantize_static(model)
    assert _op_counts(quant)["Conv"] == 1
    assert _op_counts(quant)["QuantizeLinear"] == 0


def test_calibrate_includes_conv_activation():
    weight = _f32(np.random.randn(4, 3, 3, 3).astype(np.float32), "W")
    nodes = [onnx.helper.make_node("Conv", ["X", "W"], ["Y"], kernel_shape=[3, 3])]
    model = _model(nodes, [_vi("X", [1, 3, 8, 8])], [_vi("Y", [1, 4, 6, 6])], [weight])
    data = onnxsim.generate_random_calibration_data(model, num_samples=4, seed=0)
    ranges = onnxsim.calibrate(model, data)
    assert set(ranges.keys()) == {"X"}
    lo, hi = ranges["X"]
    assert lo < hi
