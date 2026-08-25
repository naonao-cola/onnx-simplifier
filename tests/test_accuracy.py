"""Tests for ``onnxsim.accuracy`` -- the unified ``QuantizationConfig``/
``quantize()`` dispatcher and the empirical ``measure_accuracy_drop`` tool.

Models are built directly with ``onnx.helper``. ``measure_accuracy_drop``
and the ``quantize()``-dispatched calibration-based schemes execute the
model (through ``onnxsim.backend``, onnxruntime when installed), so this
mirrors ``test_dynamic_quantize_matmul_integer_to_float.py``'s
``pytest.importorskip("onnxruntime")`` guard -- a bare ``import
onnxruntime`` would fail *collection* (not skip the test) on a platform
onnxruntime doesn't ship wheels for.
"""

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper
import pytest

import onnxsim
from onnxsim.accuracy import AccuracyDropReport, OutputAccuracyStats

ort = pytest.importorskip("onnxruntime")


def _vi(name, shape):
    return onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)


def _f32(array, name):
    return onnx.numpy_helper.from_array(np.asarray(array, dtype=np.float32), name)


def _linear_model(K=64, N=32, opset=21, seed=0):
    rng = np.random.default_rng(seed)
    w = _f32(rng.standard_normal((K, N)) * 0.3, "W")
    b = _f32(rng.standard_normal(N) * 0.1, "B")
    nodes = [
        onnx.helper.make_node("MatMul", ["X", "W"], ["mm"]),
        onnx.helper.make_node("Add", ["mm", "B"], ["Y"]),
    ]
    graph = onnx.helper.make_graph(
        nodes, "g", [_vi("X", [4, K])], [_vi("Y", [4, N])], [w, b]
    )
    return onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", opset)], ir_version=10
    )


# --------------------------------------------------------------------------- #
# QuantizationConfig / quantize() dispatcher
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "config,expected_new_op",
    [
        (onnxsim.QuantizationConfig(scheme="dynamic"), "DynamicQuantizeLinear"),
        (
            onnxsim.QuantizationConfig(scheme="dynamic_fused"),
            "MatMulIntegerToFloat",
        ),
        (
            onnxsim.QuantizationConfig(scheme="weight_only", dtype="int8"),
            "DequantizeLinear",
        ),
        (
            onnxsim.QuantizationConfig(
                scheme="weight_only", dtype="int8", granularity="per_block"
            ),
            "DequantizeLinear",
        ),
        (
            onnxsim.QuantizationConfig(scheme="weight_only", dtype="int16"),
            "DequantizeLinear",
        ),
        (
            onnxsim.QuantizationConfig(scheme="weight_only", dtype="int4"),
            "DequantizeLinear",
        ),
        (onnxsim.QuantizationConfig(scheme="static"), "QuantizeLinear"),
        (
            onnxsim.QuantizationConfig(scheme="static_int16", dtype="int16"),
            "QuantizeLinear",
        ),
        (onnxsim.QuantizationConfig(scheme="qoperator"), "QLinearMatMul"),
        (onnxsim.QuantizationConfig(scheme="float", dtype="float16"), "Cast"),
        (onnxsim.QuantizationConfig(scheme="float", dtype="bfloat16"), "Cast"),
        (onnxsim.QuantizationConfig(scheme="float", dtype="float8_e4m3"), "Cast"),
        (onnxsim.QuantizationConfig(scheme="float", dtype="float8_e5m2"), "Cast"),
    ],
)
def test_quantize_dispatches_every_scheme(config, expected_new_op):
    # weight_only/int4 needs a reduction depth divisible by 32; every other
    # scheme is happy with the same K=64.
    model = _linear_model(K=64, N=32)
    quantized = onnxsim.quantize(model, config)
    onnx.checker.check_model(quantized)
    ops = {n.op_type for n in quantized.graph.node}
    assert expected_new_op in ops


def test_quantize_ternary_scheme_dispatches_and_is_a_noop_on_non_ternary_weight():
    model = _linear_model(K=64, N=32)
    quantized = onnxsim.quantize(model, onnxsim.QuantizationConfig(scheme="ternary"))
    onnx.checker.check_model(quantized)
    # The weight isn't structurally ternary, so the pass declines -- still a
    # valid dispatch, just a no-op rewrite.
    assert {n.op_type for n in quantized.graph.node} == {"MatMul", "Add"}


def test_quantize_unknown_scheme_raises_value_error():
    model = _linear_model()
    with pytest.raises(ValueError, match="unknown QuantizationConfig.scheme"):
        onnxsim.quantize(model, onnxsim.QuantizationConfig(scheme="not-a-scheme"))


def test_quantize_invalid_dtype_for_scheme_raises_value_error():
    model = _linear_model()
    with pytest.raises(ValueError, match="does not support dtype"):
        onnxsim.quantize(
            model, onnxsim.QuantizationConfig(scheme="dynamic", dtype="int16")
        )


def test_quantize_invalid_granularity_raises_value_error():
    model = _linear_model()
    with pytest.raises(ValueError, match="granularity"):
        onnxsim.quantize(
            model,
            onnxsim.QuantizationConfig(
                scheme="weight_only", dtype="int8", granularity="bogus"
            ),
        )


def test_quantize_static_passes_through_calibration_settings():
    model = _linear_model(K=8, N=4)
    rng = np.random.default_rng(5)
    calibration_data = [{"X": rng.standard_normal((4, 8)).astype(np.float32)}]
    quantized = onnxsim.quantize(
        model,
        onnxsim.QuantizationConfig(
            scheme="static",
            calibration_data=calibration_data,
            calibration_method="minmax",
        ),
    )
    onnx.checker.check_model(quantized)
    assert "QuantizeLinear" in {n.op_type for n in quantized.graph.node}


# --------------------------------------------------------------------------- #
# measure_accuracy_drop
# --------------------------------------------------------------------------- #
def test_measure_accuracy_drop_reports_small_but_nonzero_error_for_int8():
    model = _linear_model(K=64, N=32)
    quantized = onnxsim.quantize_dynamic(model)

    report = onnxsim.measure_accuracy_drop(model, quantized, num_samples=16, seed=1)
    assert isinstance(report, AccuracyDropReport)
    assert report.num_samples == 16
    assert set(report.per_output) == {"Y"}
    stats = report.per_output["Y"]
    assert isinstance(stats, OutputAccuracyStats)
    # INT8 quantization is lossy but should stay well within a coarse bound
    # for a small, well-conditioned random matrix.
    assert 0.0 < report.worst_relative_l2 < 0.2
    assert report.worst_cosine_distance < 0.1
    assert report.all_finite


def test_measure_accuracy_drop_identity_quantization_is_exact():
    # "Quantizing" with fp16 keep_io_types=True and then immediately casting
    # back is not exact, but comparing a model against *itself* must report
    # exactly zero error -- a basic sanity check on the metric plumbing.
    model = _linear_model(K=16, N=4)
    report = onnxsim.measure_accuracy_drop(model, model, num_samples=4, seed=2)
    assert report.worst_relative_l2 == 0.0
    assert report.worst_cosine_distance == 0.0
    assert report.all_finite


def test_measure_accuracy_drop_casts_inputs_for_keep_io_types_false():
    # fp16 with keep_io_types=False redeclares the graph's own inputs as
    # float16 -- the float32 calibration data must be auto-cast to match, or
    # the quantized model's session would reject it outright.
    model = _linear_model(K=16, N=4)
    quantized = onnxsim.quantize_fp16(model, keep_io_types=False)
    assert quantized.graph.input[0].type.tensor_type.elem_type == (
        onnx.TensorProto.FLOAT16
    )

    report = onnxsim.measure_accuracy_drop(model, quantized, num_samples=4, seed=3)
    assert report.worst_relative_l2 < 0.01
    assert report.all_finite


def test_measure_accuracy_drop_uses_supplied_calibration_data():
    model = _linear_model(K=8, N=4)
    quantized = onnxsim.quantize_dynamic(model)
    calibration_data = [{"X": np.ones((4, 8), dtype=np.float32)}]

    report = onnxsim.measure_accuracy_drop(
        model, quantized, calibration_data=calibration_data
    )
    assert report.num_samples == 1
