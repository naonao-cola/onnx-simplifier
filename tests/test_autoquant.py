"""Tests for ``onnxsim.auto_quantize_int4`` (AIMET's AutoQuant, see
``onnxsim/autoquant.py``) -- the escalating baseline -> CLE -> AdaRound ->
Bias Correction pipeline built on top of the three previously-ported AIMET
techniques and ``onnxsim.quantize_weight_only_int4``.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import onnx.shape_inference
import pytest
from onnx import parser

import onnxsim

ort = pytest.importorskip("onnxruntime")


def _f32(array, name):
    return onnx.numpy_helper.from_array(np.asarray(array, dtype=np.float32), name)


def _model(body, initializer=(), opset=21, ir_version=10):
    model = parser.parse_model(
        f"""
        <
          ir_version: {ir_version},
          opset_import: ["": {opset}]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    # weight_only_quantize_int4_matmul.h declines a MatMul/Gemm whose
    # activation input's elem_type isn't known -- true for an intermediate
    # tensor (e.g. Flatten's output) with no declared value_info, which the
    # parser never infers on its own.
    return onnx.shape_inference.infer_shapes(model)


def _gemm_model(K=64, N=16, batch=8, seed=0):
    rng = np.random.default_rng(seed)
    weight = rng.standard_normal((N, K)).astype(np.float32) * 0.5
    bias = rng.standard_normal(N).astype(np.float32) * 0.1
    return _model(
        f"""
        g (float[{batch},{K}] X) => (float[{batch},{N}] Y)
        {{
          Y = Gemm<transB = 1>(X, W, B)
        }}
        """,
        initializer=[_f32(weight, "W"), _f32(bias, "B")],
    )


def _conv_and_gemm_model(seed=0):
    # Conv1 -> Relu -> Conv2 (CLE has real work to do: an outlier channel
    # unbalances Conv1/Conv2's per-channel weight ranges), then flattened
    # into a Gemm (AdaRound has real work to do: only MatMul/Gemm are
    # AdaRound candidates, see adaround.py).
    rng = np.random.default_rng(seed)
    w1 = rng.standard_normal((8, 4, 3, 3)).astype(np.float32) * 0.1
    w1[0] *= 30.0  # outlier channel, matching test_cross_layer_equalization.py
    b1 = rng.standard_normal(8).astype(np.float32) * 0.01
    w2 = rng.standard_normal((4, 8, 3, 3)).astype(np.float32) * 0.1
    b2 = rng.standard_normal(4).astype(np.float32) * 0.01
    wg = rng.standard_normal((4, 4 * 8 * 8)).astype(np.float32) * 0.05
    bg = rng.standard_normal(4).astype(np.float32) * 0.01

    return _model(
        """
        g (float[2,4,8,8] X) => (float[2,4] Y)
        {
          C1 = Conv<kernel_shape = [3, 3], pads = [1, 1, 1, 1]>(X, W1, B1)
          R1 = Relu(C1)
          C2 = Conv<kernel_shape = [3, 3], pads = [1, 1, 1, 1]>(R1, W2, B2)
          F = Flatten(C2)
          Y = Gemm<transB = 1>(F, WG, BG)
        }
        """,
        initializer=[
            _f32(w1, "W1"),
            _f32(b1, "B1"),
            _f32(w2, "W2"),
            _f32(b2, "B2"),
            _f32(wg, "WG"),
            _f32(bg, "BG"),
        ],
    )


def test_auto_quantize_returns_baseline_immediately_when_budget_is_loose():
    model = _gemm_model(seed=1)
    rng = np.random.default_rng(2)
    calibration_data = [{"X": rng.standard_normal((8, 64)).astype(np.float32)}]

    result = onnxsim.auto_quantize_int4(
        model, accuracy_budget=1.0, calibration_data=calibration_data
    )
    assert result.techniques_applied == []
    assert result.meets_budget
    onnx.checker.check_model(result.quantized_model)


def test_auto_quantize_exhausts_every_stage_when_budget_is_unreachable():
    model = _conv_and_gemm_model(seed=3)
    rng = np.random.default_rng(4)
    calibration_data = [{"X": rng.standard_normal((2, 4, 8, 8)).astype(np.float32)}]

    result = onnxsim.auto_quantize_int4(
        model,
        accuracy_budget=0.0,  # unreachable -- every stage must be tried
        calibration_data=calibration_data,
        num_adaround_iterations=100,
    )
    # accuracy_budget=0.0 means no stage can ever *meet* budget, so all four
    # are attempted -- but the returned result is the best *reached* across
    # them (never regressing versus an earlier stage, matching
    # recommend_quantization's own least-lossy-fallback convention), which
    # need not be the final stage if an earlier one already did best.
    valid_prefixes = [
        [],
        ["cross_layer_equalization"],
        ["cross_layer_equalization", "adaround"],
        ["cross_layer_equalization", "adaround", "bias_correction"],
    ]
    assert result.techniques_applied in valid_prefixes
    # For this model/seed, AdaRound and Bias Correction each measurably
    # improve on the stage before them (verified independently), so the
    # pipeline should reach further than the untouched baseline.
    assert result.techniques_applied != []
    assert not result.meets_budget
    onnx.checker.check_model(result.quantized_model)

    sess = ort.InferenceSession(
        result.quantized_model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (y,) = sess.run(None, calibration_data[0])
    assert np.all(np.isfinite(y))


def test_auto_quantize_escalates_to_meet_a_budget_baseline_misses():
    model = _gemm_model(K=64, N=16, batch=32, seed=5)
    rng = np.random.default_rng(6)
    calibration_data = [{"X": rng.standard_normal((32, 64)).astype(np.float32)}]

    # Measure the baseline's own error first, then pick a budget the
    # baseline provably misses but that the full pipeline (whose final
    # stage's error this test does not otherwise assume the exact value of)
    # has a real chance to reach -- avoids a hand-tuned magic threshold.
    baseline = onnxsim.quantize_weight_only_int4(model)
    baseline_report = onnxsim.measure_accuracy_drop(
        model, baseline, calibration_data=calibration_data
    )
    budget = baseline_report.worst_relative_l2 * 0.98

    result = onnxsim.auto_quantize_int4(
        model,
        accuracy_budget=budget,
        calibration_data=calibration_data,
        num_adaround_iterations=200,
    )
    assert result.techniques_applied != []
    assert result.report.worst_relative_l2 <= baseline_report.worst_relative_l2


def test_auto_quantize_report_matches_returned_model():
    model = _gemm_model(seed=7)
    rng = np.random.default_rng(8)
    calibration_data = [{"X": rng.standard_normal((8, 64)).astype(np.float32)}]

    result = onnxsim.auto_quantize_int4(
        model,
        accuracy_budget=0.0,
        calibration_data=calibration_data,
        num_adaround_iterations=50,
    )
    recomputed = onnxsim.measure_accuracy_drop(
        model, result.quantized_model, calibration_data=calibration_data
    )
    assert recomputed.worst_relative_l2 == pytest.approx(
        result.report.worst_relative_l2
    )


def test_auto_quantize_never_regresses_versus_baseline():
    model = _conv_and_gemm_model(seed=9)
    rng = np.random.default_rng(10)
    calibration_data = [{"X": rng.standard_normal((2, 4, 8, 8)).astype(np.float32)}]

    baseline = onnxsim.quantize_weight_only_int4(model)
    baseline_report = onnxsim.measure_accuracy_drop(
        model, baseline, calibration_data=calibration_data
    )

    result = onnxsim.auto_quantize_int4(
        model,
        accuracy_budget=0.0,
        calibration_data=calibration_data,
        num_adaround_iterations=100,
    )
    assert result.report.worst_relative_l2 <= baseline_report.worst_relative_l2 + 1e-9
