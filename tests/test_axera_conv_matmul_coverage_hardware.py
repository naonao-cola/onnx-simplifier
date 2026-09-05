"""Real-hardware follow-up to tests/test_axera_conv_matmul_coverage.py:
that file documents what Pulsar2's *static* AX650 coverage heuristic can
and can't tell apart, explicitly caveated as never verified against a real
compiler (no Docker/device access in that session). This file compiles
every variant it flagged as unverified through a real `pulsar2 build`.

Needs a loaded `pulsar2:*` Docker image -- skip-guarded like
tests/test_pulsar2_hf_to_axmodel.py. See scripts/axera/README.md's
"Confirmed on real hardware: which of these actually build" section for
the narrative writeup of these results.
"""

import json
import os
import sys

import numpy as np
import onnx
import pytest
from onnx import numpy_helper, parser

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import pulsar2_docker  # noqa: E402

pytestmark = pytest.mark.skipif(
    not pulsar2_docker.docker_image_available(),
    reason=f"pulsar2 Docker image not loaded: {pulsar2_docker.DEFAULT_IMAGE}",
)


def _model(body, weights=(), opset=13, ir_version=10):
    model = parser.parse_model(
        f'<ir_version: {ir_version}, opset_import: ["": {opset}]> {body}'
    )
    model.graph.initializer.extend(weights)
    onnx.checker.check_model(model)
    return model


def _rand_weight(name, shape, seed=0):
    rng = np.random.default_rng(seed)
    arr = (rng.standard_normal(shape).astype(np.float32) * 0.1).astype("<f4")
    return numpy_helper.from_array(arr, name=name)


def _rand_uint8_weight(name, shape, seed=0):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 4, size=shape).astype(np.uint8)
    return numpy_helper.from_array(arr, name=name)


def _generic_quant_config(input_specs, calibration_size=4):
    return {
        "model_type": "ONNX",
        "npu_mode": "NPU1",
        "quant": {
            "input_configs": [
                {
                    "tensor_name": name,
                    "calibration_dataset": f"./dataset/calib_{name}.tar",
                    "calibration_format": "Numpy",
                    "calibration_size": calibration_size,
                }
                for name, _shape, _dtype in input_specs
            ],
            "calibration_method": "MinMax",
            "precision_analysis": False,
        },
        "compiler": {"check": 0},
    }


def _build_and_run(tmp_path, name, model, input_specs, calibration_size=4, seed=0):
    work_dir = os.path.join(str(tmp_path), name)
    os.makedirs(work_dir, exist_ok=True)
    onnx.save(model, os.path.join(work_dir, "model.onnx"))

    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(work_dir, "dataset"), exist_ok=True)
    for tname, shape, dtype in input_specs:
        if np.issubdtype(dtype, np.floating):
            samples = [
                rng.standard_normal(shape).astype(dtype)
                for _ in range(calibration_size)
            ]
        else:
            samples = [
                rng.integers(0, 4, size=shape).astype(dtype)
                for _ in range(calibration_size)
            ]
        pulsar2_docker.make_numpy_calibration_tar(
            os.path.join(work_dir, "dataset", f"calib_{tname}.tar"), samples
        )

    os.makedirs(os.path.join(work_dir, "config"), exist_ok=True)
    with open(os.path.join(work_dir, "config", "cfg.json"), "w") as f:
        json.dump(_generic_quant_config(input_specs, calibration_size), f)

    return pulsar2_docker.build(
        work_dir, "model.onnx", "output", config_path="config/cfg.json"
    )


@pytest.mark.parametrize(
    "name,body,weight_shape,input_spec",
    [
        (
            "conv_auto_pad_same_upper",
            "g (float[1,4,10,10] X) => (float[1,4,10,10] Y) "
            '{ Y = Conv<auto_pad="SAME_UPPER", kernel_shape=[3,3]>(X, W) }',
            (4, 4, 3, 3),
            ("X", (1, 4, 10, 10), np.float32),
        ),
        (
            "conv_1d",
            "g (float[1,4,10] X) => (float[1,4,8] Y) "
            "{ Y = Conv<kernel_shape=[3]>(X, W) }",
            (4, 4, 3),
            ("X", (1, 4, 10), np.float32),
        ),
        (
            "conv_3d",
            "g (float[1,4,10,10,10] X) => (float[1,4,8,8,8] Y) "
            "{ Y = Conv<kernel_shape=[3,3,3]>(X, W) }",
            (4, 4, 3, 3, 3),
            ("X", (1, 4, 10, 10, 10), np.float32),
        ),
    ],
)
def test_conv_variant_compiles(tmp_path, name, body, weight_shape, input_spec):
    model = _model(body, weights=[_rand_weight("W", weight_shape)])
    result = _build_and_run(tmp_path, name, model, [input_spec])
    assert result.success, result.error


def test_matmul_batched_broadcast_compiles(tmp_path):
    model = _model(
        "g (float[2,4,8] A) => (float[2,4,16] Y) { Y = MatMul(A, B) }",
        weights=[_rand_weight("B", (8, 16))],
    )
    result = _build_and_run(
        tmp_path, "matmul_batched_broadcast", model, [("A", (2, 4, 8), np.float32)]
    )
    assert result.success, result.error


def test_gemm_transb_compiles(tmp_path):
    model = _model(
        "g (float[4,8] A) => (float[4,16] Y) { Y = Gemm<transB=1>(A, B) }",
        weights=[_rand_weight("B", (16, 8))],
    )
    result = _build_and_run(tmp_path, "gemm_transb", model, [("A", (4, 8), np.float32)])
    assert result.success, result.error


def test_gemm_nondefault_alpha_beta_fails(tmp_path):
    """Confirmed real, see scripts/axera/README.md: distinct from a
    'not on the supported-op list' failure -- Gemm with non-default
    alpha/beta lowers to a different, entirely unimplemented
    AxQuantizedGemm op; default alpha=1.0/beta=1.0 (test_gemm_transb_compiles
    above; test_axera_conv_matmul_coverage.py's gemm_transb) lowers through
    the same path as MatMul + bias-add instead and compiles fine."""
    model = _model(
        "g (float[4,8] A) => (float[4,16] Y) "
        "{ Y = Gemm<alpha=2.0, beta=0.5>(A, B, C) }",
        weights=[_rand_weight("B", (8, 16)), _rand_weight("C", (16,))],
    )
    result = _build_and_run(
        tmp_path, "gemm_nondefault_alpha_beta", model, [("A", (4, 8), np.float32)]
    )
    assert not result.success
    assert "AxQuantizedGemm" in result.error


@pytest.mark.parametrize(
    "name,body,weight_shape,input_spec",
    [
        (
            "conv_integer",
            "g (uint8[1,4,10,10] X) => (int32[1,4,8,8] Y) { Y = ConvInteger(X, W) }",
            (4, 4, 3, 3),
            ("X", (1, 4, 10, 10), np.uint8),
        ),
        (
            "matmul_integer",
            "g (uint8[4,8] A) => (int32[4,16] Y) { Y = MatMulInteger(A, B) }",
            (8, 16),
            ("A", (4, 8), np.uint8),
        ),
    ],
)
def test_standard_quantized_op_fails_as_unimplemented(
    tmp_path, name, body, weight_shape, input_spec
):
    op_type = body.split(" = ")[1].split("(")[0]
    model = _model(
        body, weights=[_rand_uint8_weight("W" if "W" in body else "B", weight_shape)]
    )
    result = _build_and_run(tmp_path, name, model, [input_spec])
    assert not result.success
    assert f"dont support {op_type} opr" in result.error


def test_qlinear_conv_fails_as_unimplemented(tmp_path):
    model = _model(
        "g (uint8[1,4,10,10] X) => (uint8[1,4,8,8] Y) "
        "<float x_scale = {0.5}, uint8 x_zero_point = {0}, "
        "float w_scale = {0.5}, uint8 w_zero_point = {0}, "
        "float y_scale = {0.5}, uint8 y_zero_point = {0}> "
        "{ Y = QLinearConv(X, x_scale, x_zero_point, w, w_scale, w_zero_point, "
        "y_scale, y_zero_point) }",
        weights=[_rand_uint8_weight("w", (4, 4, 3, 3))],
    )
    result = _build_and_run(
        tmp_path, "qlinear_conv", model, [("X", (1, 4, 10, 10), np.uint8)]
    )
    assert not result.success
    assert "dont support QLinearConv opr" in result.error


def test_qlinear_matmul_fails_as_unimplemented(tmp_path):
    model = _model(
        "g (uint8[4,8] A) => (uint8[4,16] Y) "
        "<float a_scale = {0.5}, uint8 a_zero_point = {0}, "
        "float b_scale = {0.5}, uint8 b_zero_point = {0}, "
        "float y_scale = {0.5}, uint8 y_zero_point = {0}> "
        "{ Y = QLinearMatMul(A, a_scale, a_zero_point, b, b_scale, b_zero_point, "
        "y_scale, y_zero_point) }",
        weights=[_rand_uint8_weight("b", (8, 16))],
    )
    result = _build_and_run(
        tmp_path, "qlinear_matmul", model, [("A", (4, 8), np.uint8)]
    )
    assert not result.success
    assert "dont support QLinearMatMul opr" in result.error


def test_conv_transpose_fails_with_a_different_error(tmp_path):
    """Confirmed real, distinct failure mode: ConvTranspose IS present in
    AX650_SUPPORTED_OPS (test_axera_conv_matmul_coverage.py confirms
    partition() reports full coverage for it), but a real build of this
    plain shape fails during quantization, not with the
    'dont support ... opr' pattern the genuinely-unimplemented ops above
    hit -- a real gap in the static heuristic's "supported" claim."""
    model = _model(
        "g (float[1,4,8,8] X) => (float[1,4,10,10] Y) "
        "{ Y = ConvTranspose<kernel_shape=[3,3]>(X, W) }",
        weights=[_rand_weight("W", (4, 4, 3, 3))],
    )
    result = _build_and_run(
        tmp_path, "conv_transpose", model, [("X", (1, 4, 8, 8), np.float32)]
    )
    assert not result.success
    assert "dont support" not in result.error
