"""Real-hardware differential analysis of onnxsim/pulsar2_docker.py's
compiled `.axmodel` `neu mode` format: how Add/Sub/Mul/Div constants and
Conv bias terms get encoded in the `npu_params` initializer blob.

See scripts/axera/README.md's "Differential analysis: how elementwise ops
and Conv bias get encoded" section for the full narrative writeup this
locks in -- including the parts of that investigation (a still-undecoded
4-byte field in Add/Sub's non-trivial encoding) this file does NOT attempt
to test, since their exact bit-level format was never identified.

Needs a loaded `pulsar2:*` Docker image -- skip-guarded like
tests/test_pulsar2_hf_to_axmodel.py.
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


def _model(op, const_val, opset=17, ir_version=10):
    model = parser.parse_model(
        f'<ir_version: {ir_version}, opset_import: ["": {opset}]> '
        f"agraph (float[1,4] x) => (float[1,4] y) "
        f"{{ y = {op}(x, c) }}"
    )
    arr = np.full((4,), const_val, dtype=np.float32)
    model.graph.initializer.append(numpy_helper.from_array(arr, name="c"))
    onnx.checker.check_model(model)
    return model


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


def _build(tmp_path, name, model, input_specs, calibration_size=4, seed=0):
    work_dir = os.path.join(str(tmp_path), name)
    os.makedirs(work_dir, exist_ok=True)
    onnx.save(model, os.path.join(work_dir, "model.onnx"))

    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(work_dir, "dataset"), exist_ok=True)
    for tname, shape, dtype in input_specs:
        samples = [
            rng.standard_normal(shape).astype(dtype) for _ in range(calibration_size)
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


def _npu_params_bytes(axmodel_path):
    compiled = onnx.load(axmodel_path)
    neu_node = next(n for n in compiled.graph.node if n.op_type == "neu mode")
    inits = {i.name: i for i in compiled.graph.initializer}
    info = None
    for attr in neu_node.attribute:
        if attr.name == "npu_graph_info":
            info = json.loads(attr.s.decode())
    params_key = info["dotneus"][0]["extra_inputs"][0]["const_data_key"]
    return bytes(inits[params_key].raw_data)


def test_mul_and_div_by_one_crash_the_real_build(tmp_path):
    """Confirmed real, reproducible compiler bug: multiplying or dividing
    by the identity constant 1.0 gets eliminated by Pulsar2's own frontend
    graph optimizer (x*1=x, x/1=x) before quantization runs, leaving the
    declared graph output with no producing node."""
    for op in ("Mul", "Div"):
        model = _model(op, 1.0)
        result = _build(
            tmp_path, f"{op.lower()}_by_one", model, [("x", (1, 4), np.float32)]
        )
        assert not result.success
        assert "Seems config of input(y) doesn't exist" in result.error


@pytest.mark.parametrize(
    "value,expect_trivial",
    [(0.0, True), (1.0, True), (2.0, True), (3.0, False), (-1.0, False), (0.5, False)],
)
def test_add_trivial_set_is_exactly_0_1_2(tmp_path, value, expect_trivial):
    model = _model("Add", value)
    result = _build(tmp_path, f"add_{value}", model, [("x", (1, 4), np.float32)])
    assert result.success, result.error
    params = _npu_params_bytes(result.axmodel_path)
    assert (len(params) == 44) == expect_trivial, (value, len(params))


@pytest.mark.parametrize(
    "value,expect_trivial", [(0.0, True), (1.0, True), (2.0, False), (3.0, False)]
)
def test_sub_trivial_set_is_only_0_1(tmp_path, value, expect_trivial):
    """Confirmed narrower than Add's {0,1,2} -- Sub(x, 2.0) is not trivial,
    unlike Add(x, 2.0)."""
    model = _model("Sub", value)
    result = _build(tmp_path, f"sub_{value}", model, [("x", (1, 4), np.float32)])
    assert result.success, result.error
    params = _npu_params_bytes(result.axmodel_path)
    assert (len(params) == 44) == expect_trivial, (value, len(params))


def test_div_triviality_depends_on_the_reciprocal(tmp_path):
    """Div(x, 0.5) (reciprocal 2.0, an integer) is trivial; Div(x, 2.0)
    and Div(x, 3.0) (reciprocals 0.5, 0.333..., non-integer) saturate to
    0xff -- consistent with Div(x, c) compiling as Mul(x, 1/c)."""
    model_half = _model("Div", 0.5)
    result_half = _build(tmp_path, "div_half", model_half, [("x", (1, 4), np.float32)])
    assert result_half.success, result_half.error
    params_half = _npu_params_bytes(result_half.axmodel_path)
    assert params_half[:4] == bytes([2, 2, 2, 2])

    for value in (2.0, 3.0):
        model = _model("Div", value)
        result = _build(tmp_path, f"div_{value}", model, [("x", (1, 4), np.float32)])
        assert result.success, result.error
        params = _npu_params_bytes(result.axmodel_path)
        assert params[:4] == bytes([0xFF] * 4), (value, params[:4].hex())


def test_div_by_zero_stores_literal_float32_infinity(tmp_path):
    model = _model("Div", 0.0)
    result = _build(tmp_path, "div_zero", model, [("x", (1, 4), np.float32)])
    assert result.success, result.error
    params = _npu_params_bytes(result.axmodel_path)
    inf_bytes = np.float32(np.inf).tobytes()
    assert params[:16] == inf_bytes * 4


def test_mixed_constant_is_never_trivial_even_if_every_element_qualifies(tmp_path):
    """Add's trivial fast path needs the whole tensor to be a uniform
    single-value broadcast, not just small per-element values: [1,1,2,2]
    (every element individually in {0,1,2}) is still non-trivial, but the
    exact literal integer values are still preserved per element."""
    model = parser.parse_model(
        '<ir_version: 10, opset_import: ["": 17]> '
        "agraph (float[1,4] x) => (float[1,4] y) { y = Add(x, c) }"
    )
    model.graph.initializer.append(
        numpy_helper.from_array(np.array([1, 1, 2, 2], dtype=np.float32), name="c")
    )
    onnx.checker.check_model(model)

    result = _build(tmp_path, "add_mixed_1122", model, [("x", (1, 4), np.float32)])
    assert result.success, result.error
    params = _npu_params_bytes(result.axmodel_path)
    assert len(params) != 44
    assert params[:4] == bytes([1, 1, 2, 2])
