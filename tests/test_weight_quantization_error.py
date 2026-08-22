"""Tests for ``onnxsim.model_info.weight_quantization_error``: the before/after
weight accuracy metrics computed by rematerializing a ``quantize_static``-quantized
weight through its ``DequantizeLinear`` and diffing it against the original float
weight it was quantized from.
"""

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper
import pytest

import onnxsim
from onnxsim.model_info import (
    WeightQuantizationError,
    print_weight_quantization_error,
    weight_quantization_error,
)

# quantize_static's default (random) calibration data needs onnxruntime to run
# the float model; skip (not fail collection) where onnxruntime has no wheel.
ort = pytest.importorskip("onnxruntime")


def _model(nodes, inputs, outputs, initializer, opset=13):
    graph = onnx.helper.make_graph(nodes, "g", inputs, outputs, initializer)
    return onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", opset)], ir_version=10
    )


def _vi(name, shape):
    return onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _matmul_model(K=32, N=16, seed=0):
    rng = np.random.default_rng(seed)
    weight = _f32(rng.standard_normal((K, N)) * 0.5, "W")
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    return _model(nodes, [_vi("X", [4, K])], [_vi("Y", [4, N])], [weight])


def _conv_model(seed=0):
    rng = np.random.default_rng(seed)
    weight = _f32(rng.standard_normal((6, 3, 3, 3)), "W")
    nodes = [onnx.helper.make_node("Conv", ["X", "W"], ["Y"], kernel_shape=[3, 3])]
    return _model(nodes, [_vi("X", [1, 3, 8, 8])], [_vi("Y", [1, 6, 6, 6])], [weight])


def test_matmul_weight_quantization_error():
    model = _matmul_model(K=32, N=16, seed=0)
    quant = onnxsim.quantize_static(model, num_calibration_samples=16, seed=0)

    results = weight_quantization_error(model, quant)
    assert len(results) == 1
    r = results[0]
    assert isinstance(r, WeightQuantizationError)
    assert r.weight_name == "W"
    assert r.shape == (32, 16)
    assert r.axis == 1  # MatMul's un-transposed weight: output channel is axis 1
    assert len(r.per_channel_relative_l2) == 16

    # INT8 per-channel symmetric quantization of ~N(0, 0.5) data: a real bug
    # (wrong scale, wrong axis, transposed reconstruction, ...) would blow this
    # up far past a quantization step's worth of noise.
    assert 0.0 <= r.relative_l2 < 0.05
    assert r.cosine_similarity > 0.999
    assert r.sqnr_db > 20.0
    assert all(0.0 <= e < 0.2 for e in r.per_channel_relative_l2)

    # ENOB is just SQNR re-expressed in bits -- must track it exactly.
    assert r.enob_bits == pytest.approx((r.sqnr_db - 1.76) / 6.02)
    # PSNR is peak- rather than energy-normalized, so it never falls below SQNR.
    assert r.psnr_db >= r.sqnr_db
    # A close element-wise reconstruction should also leave the value
    # histogram's shape close to unchanged.
    assert 0.0 <= r.histogram_js_divergence < 0.05


def test_gemm_transb_weight_quantization_error():
    # PyTorch's nn.Linear layout: weight stored [N, K], transB=1 -- the output
    # channel is then axis 0 of W's own (untransposed) storage.
    rng = np.random.default_rng(1)
    K, N = 24, 12
    weight = _f32(rng.standard_normal((N, K)) * 0.5, "W")
    bias = _f32(rng.standard_normal(N), "B")
    nodes = [onnx.helper.make_node("Gemm", ["X", "W", "B"], ["Y"], transB=1)]
    model = _model(nodes, [_vi("X", [3, K])], [_vi("Y", [3, N])], [weight, bias])
    quant = onnxsim.quantize_static(model, num_calibration_samples=16, seed=1)

    results = weight_quantization_error(model, quant)
    assert len(results) == 1
    r = results[0]
    assert r.shape == (N, K)
    assert r.axis == 0
    assert len(r.per_channel_relative_l2) == N
    assert r.relative_l2 < 0.05


def test_conv_weight_quantization_error():
    model = _conv_model()
    quant = onnxsim.quantize_static(model, num_calibration_samples=16, seed=2)

    results = weight_quantization_error(model, quant)
    assert len(results) == 1
    r = results[0]
    assert r.shape == (6, 3, 3, 3)
    assert r.axis == 0  # Conv's output channel is always axis 0
    assert len(r.per_channel_relative_l2) == 6
    assert r.relative_l2 < 0.1


def test_exact_reconstruction_is_zero_error():
    # A weight whose per-channel max already sits on a clean INT8 step
    # reconstructs exactly: mse/relative_l2 should be exactly 0 and cosine
    # similarity exactly 1, not just "close".
    K, N = 4, 2
    w = np.zeros((K, N), dtype=np.float32)
    w[:, 0] = np.array([127, -127, 64, -64], dtype=np.float32)  # scale=1
    w[:, 1] = np.array([254, -254, 128, -128], dtype=np.float32)  # scale=2
    weight = _f32(w, "W")
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    model = _model(nodes, [_vi("X", [1, K])], [_vi("Y", [1, N])], [weight])
    quant = onnxsim.quantize_static(model, num_calibration_samples=4, seed=0)

    results = weight_quantization_error(model, quant)
    assert len(results) == 1
    r = results[0]
    assert r.mse == 0.0
    assert r.relative_l2 == 0.0
    assert r.cosine_similarity == pytest.approx(1.0)
    assert r.sqnr_db == float("inf")
    assert r.enob_bits == float("inf")
    assert r.psnr_db == float("inf")
    assert r.histogram_js_divergence == 0.0
    assert all(e == 0.0 for e in r.per_channel_relative_l2)


def test_outlier_channel_increases_histogram_js_divergence():
    # A single large per-channel outlier forces a coarse scale (scale =
    # max|w| / 127), crushing the rest of the channel's continuous-looking
    # values down into just a handful of discrete INT8 levels -- a
    # distributional collapse the histogram JS divergence should flag even
    # though the element-wise relative L2 error stays fairly small.
    rng = np.random.default_rng(3)
    K, N = 300, 1
    w = (rng.standard_normal((K, N)) * 0.01).astype(np.float32)
    w[0, 0] = 1.0  # the outlier
    weight = _f32(w, "W")
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    model = _model(nodes, [_vi("X", [1, K])], [_vi("Y", [1, N])], [weight])
    quant = onnxsim.quantize_static(model, num_calibration_samples=4, seed=3)

    results = weight_quantization_error(model, quant)
    assert len(results) == 1
    r = results[0]
    assert r.relative_l2 < 0.1  # element-wise error alone looks unremarkable
    assert r.histogram_js_divergence > 0.1  # but the distribution shape moved


def test_no_results_when_weight_not_constant():
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    model = _model(nodes, [_vi("X", [4, 8]), _vi("W", [8, 4])], [_vi("Y", [4, 4])], [])
    quant = onnxsim.quantize_static(model)
    assert weight_quantization_error(model, quant) == []


def test_no_results_on_unquantized_model():
    # model_after == model_before: no DequantizeLinear on any weight input, so
    # nothing is reported rather than a spurious zero-error match.
    model = _matmul_model()
    assert weight_quantization_error(model, model) == []


def test_dynamic_quantization_yields_no_results():
    # quantize_dynamic replaces MatMul/Gemm with MatMulInteger rather than
    # dequantizing the weight back to float in the graph, so there is no
    # DequantizeLinear for this function to walk -- see its docstring.
    model = _matmul_model()
    quant = onnxsim.quantize_dynamic(model)
    assert weight_quantization_error(model, quant) == []


def test_print_weight_quantization_error_does_not_raise(capsys):
    model = _matmul_model()
    quant = onnxsim.quantize_static(model, num_calibration_samples=16, seed=0)
    results = weight_quantization_error(model, quant)
    print_weight_quantization_error(results)
    captured = capsys.readouterr()
    assert "W" in captured.out


def test_print_weight_quantization_error_caps_output(capsys):
    results = [
        WeightQuantizationError(
            node=f"n{i}",
            weight_name=f"W{i}",
            shape=(2, 2),
            axis=1,
            mse=0.0,
            max_abs_error=0.0,
            relative_l2=float(i),
            cosine_similarity=1.0,
            sqnr_db=float("inf"),
            enob_bits=float("inf"),
            psnr_db=float("inf"),
            histogram_js_divergence=0.0,
            per_channel_relative_l2=[0.0, 0.0],
        )
        for i in range(5)
    ]
    print_weight_quantization_error(results, limit=2)
    captured = capsys.readouterr()
    assert "... and 3 more" in captured.out
