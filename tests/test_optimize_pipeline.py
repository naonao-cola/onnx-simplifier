"""Tests for ``onnxsim.apply_optimization_pipeline`` (see
``onnxsim/optimize_pipeline.py``) -- the escalating
simplify -> prune -> CLE -> quantize -> refine -> compress pipeline chaining
onnxsim's recently added features together.
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
    # weight-only/quarot passes decline a MatMul/Gemm whose activation
    # input's elem_type isn't known -- true for an intermediate tensor with
    # no declared value_info, which the parser never infers on its own.
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
    # Conv1 -> Relu -> Conv2 (CLE has real work to do), then flattened into
    # a Gemm (AdaRound/pruning have real work to do).
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


def test_pipeline_returns_baseline_immediately_when_budget_is_loose():
    model = _gemm_model(seed=1)
    rng = np.random.default_rng(2)
    calibration_data = [{"X": rng.standard_normal((8, 64)).astype(np.float32)}]

    result = onnxsim.apply_optimization_pipeline(
        model, accuracy_budget=1.0, calibration_data=calibration_data
    )
    assert result.meets_budget
    assert "adaround" not in result.stages_applied
    assert "bias_correction" not in result.stages_applied
    assert "cross_layer_equalization" in result.stages_applied
    assert "quantize_weight_only_int4" in result.stages_applied
    onnx.checker.check_model(result.optimized_model)


def test_pipeline_exhausts_refinement_stages_when_budget_is_unreachable():
    model = _conv_and_gemm_model(seed=3)
    rng = np.random.default_rng(4)
    calibration_data = [{"X": rng.standard_normal((2, 4, 8, 8)).astype(np.float32)}]

    result = onnxsim.apply_optimization_pipeline(
        model,
        accuracy_budget=0.0,  # unreachable -- every stage must be tried
        calibration_data=calibration_data,
        num_adaround_iterations=100,
    )
    assert not result.meets_budget
    assert "adaround" in result.stages_applied
    onnx.checker.check_model(result.optimized_model)

    sess = ort.InferenceSession(
        result.optimized_model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (y,) = sess.run(None, calibration_data[0])
    assert np.all(np.isfinite(y))


def test_pipeline_with_pruning_applies_structured_pruning_stage():
    model = _conv_and_gemm_model(seed=11)
    rng = np.random.default_rng(12)
    calibration_data = [{"X": rng.standard_normal((2, 4, 8, 8)).astype(np.float32)}]

    result = onnxsim.apply_optimization_pipeline(
        model,
        accuracy_budget=1.0,
        calibration_data=calibration_data,
        sparsity=0.25,
    )
    assert result.stages_applied[0] == "structured_pruning"
    onnx.checker.check_model(result.optimized_model)


def test_pipeline_without_pruning_skips_structured_pruning_stage():
    model = _gemm_model(seed=13)
    rng = np.random.default_rng(14)
    calibration_data = [{"X": rng.standard_normal((8, 64)).astype(np.float32)}]

    result = onnxsim.apply_optimization_pipeline(
        model, accuracy_budget=1.0, calibration_data=calibration_data
    )
    assert "structured_pruning" not in result.stages_applied


def test_pipeline_mixed_precision_bit_selection():
    model = _gemm_model(seed=15)
    rng = np.random.default_rng(16)
    calibration_data = [{"X": rng.standard_normal((8, 64)).astype(np.float32)}]

    result = onnxsim.apply_optimization_pipeline(
        model,
        accuracy_budget=1.0,
        calibration_data=calibration_data,
        bit_selection="mixed_precision",
    )
    assert "apply_mixed_precision_quantization" in result.stages_applied
    assert "quantize_weight_only_int4" not in result.stages_applied
    onnx.checker.check_model(result.optimized_model)


def test_pipeline_rotation_upgrade_skips_adaround_refinement():
    model = _gemm_model(K=64, N=16, seed=17)
    rng = np.random.default_rng(18)
    calibration_data = [{"X": rng.standard_normal((8, 64)).astype(np.float32)}]

    result = onnxsim.apply_optimization_pipeline(
        model,
        accuracy_budget=0.0,  # unreachable -- would otherwise trigger refinement
        calibration_data=calibration_data,
        use_rotation=True,
    )
    assert "quarot" in result.stages_applied
    assert "adaround" not in result.stages_applied
    assert "bias_correction" not in result.stages_applied
    onnx.checker.check_model(result.optimized_model)


def test_pipeline_rejects_invalid_bit_selection():
    model = _gemm_model(seed=19)
    with pytest.raises(ValueError):
        onnxsim.apply_optimization_pipeline(model, bit_selection="bogus")


def test_pipeline_report_matches_returned_model():
    model = _gemm_model(seed=20)
    rng = np.random.default_rng(21)
    calibration_data = [{"X": rng.standard_normal((8, 64)).astype(np.float32)}]

    result = onnxsim.apply_optimization_pipeline(
        model,
        accuracy_budget=0.0,
        calibration_data=calibration_data,
        num_adaround_iterations=50,
    )
    simplified, _ = onnxsim.simplify(model)
    recomputed = onnxsim.measure_accuracy_drop(
        simplified, result.optimized_model, calibration_data=calibration_data
    )
    assert recomputed.worst_relative_l2 == pytest.approx(
        result.report.worst_relative_l2
    )


def test_pipeline_is_deterministic_for_a_given_seed():
    model = _gemm_model(seed=22)
    rng = np.random.default_rng(23)
    calibration_data = [{"X": rng.standard_normal((8, 64)).astype(np.float32)}]

    result1 = onnxsim.apply_optimization_pipeline(
        model, accuracy_budget=1.0, calibration_data=calibration_data, seed=5
    )
    result2 = onnxsim.apply_optimization_pipeline(
        model, accuracy_budget=1.0, calibration_data=calibration_data, seed=5
    )
    assert result1.stages_applied == result2.stages_applied
    assert result1.optimized_model.SerializeToString() == (
        result2.optimized_model.SerializeToString()
    )
