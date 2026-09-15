"""Device numerics for the single-LeakyRelu mcode probe streams.

Fifteenth op in the hardware-verified set, closing the loop on a
compile-time gotcha this project's README already documents: `LeakyRelu`
(along with `Elu` and `TopK`) failed to compile on the first attempt with
a confusing internal error (`RuntimeError("... convert error: 'alpha'")`)
because Pulsar2's frontend does not resolve an unset optional attribute
to its ONNX-schema default -- it reads the missing attribute as `None`
and chokes. Setting `alpha` explicitly (the same default value the
schema would have applied) fixed the *compile*. This probe closes the
other half of that story: does the device *compute* LeakyRelu correctly
once it compiles, not just accept the graph? `alpha` is set explicitly
here for the same reason the README's fix does it.

Measured (2026-09-16): base 1.11, wide 1.11, positive 0.46 output-LSBs --
confirms the device computes LeakyRelu correctly (not just accepts the
graph) once the compile-time attribute-defaulting gotcha is worked
around. Notably higher on the two symmetric-range cases than on the
positive-only one, the same shape as Relu's own numbers (see that
probe's docstring) but starting from a higher floor -- unlike Relu,
LeakyRelu got a full int8 quantize/dequantize path with its own
independently-calibrated output scale (`AxQuantizedLeakyRelu`, not
Relu's scale-reusing `AxClip`), so it doesn't get Relu's "free" precision
from skipping a second calibration step.

Needs a loaded `pulsar2:*` Docker image to (re)build and an AX650N card
to run -- skip-guarded on both, like the rest of the hardware suite.
"""

import json
import os
import sys

import numpy as np
import onnx
import pytest
from onnx import parser

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import pulsar2_docker  # noqa: E402

_IMAGE = "pulsar2:7.0-lite"
pytestmark = pytest.mark.skipif(
    not pulsar2_docker.docker_image_available(_IMAGE),
    reason=f"pulsar2 Docker image not loaded: {_IMAGE}",
)

CASES = {
    "base": (-1.0, 1.0),
    "wide": (-10.0, 10.0),
    "positive": (0.5, 2.0),
}

ALPHA = 0.01  # ONNX schema default, set explicitly -- see module docstring.

LSB_BOUND = 1.5


def _build(tmp_path, tag, lo, hi):
    work_dir = os.path.join(str(tmp_path), f"leakyrelu_{tag}")
    os.makedirs(os.path.join(work_dir, "dataset"), exist_ok=True)
    model = parser.parse_model(
        '<ir_version: 8, opset_import: ["": 17]> '
        "agraph (float[1,8] x) => (float[1,8] y) "
        f"{{ y = LeakyRelu<alpha={ALPHA}> (x) }}"
    )
    onnx.save(model, os.path.join(work_dir, "model.onnx"))
    rng = np.random.default_rng(0)
    pulsar2_docker.make_numpy_calibration_tar(
        os.path.join(work_dir, "dataset", "x.tar"),
        [(rng.uniform(lo, hi, (1, 8))).astype(np.float32) for _ in range(8)],
    )
    cfg = {
        "model_type": "ONNX",
        "npu_mode": "NPU1",
        "quant": {
            "input_configs": [
                {
                    "tensor_name": "x",
                    "calibration_dataset": "./dataset/x.tar",
                    "calibration_format": "Numpy",
                    "calibration_size": 8,
                }
            ],
            "calibration_method": "MinMax",
            "precision_analysis": False,
            "finetune_epochs": 0,
        },
        "compiler": {"check": 0},
    }
    with open(os.path.join(work_dir, "config.json"), "w") as f:
        json.dump(cfg, f)
    result = pulsar2_docker.build(
        work_dir,
        "model.onnx",
        "out",
        config_path="config.json",
        image=_IMAGE,
        target_hardware="AX650",
        timeout=1200,
    )
    assert result.success, result.error
    return os.path.join(work_dir, "out")


def _run_retry_once(axmodel_path, feeds):
    """`run_on_device_with_inputs`, retried once on a `0x8030070C` fault --
    see test_axera_mul_probe_hardware.py's identical helper for why."""
    dev = pulsar2_docker.run_on_device_with_inputs(axmodel_path, feeds)
    if dev.error and "0x8030070C" in dev.error:
        dev = pulsar2_docker.run_on_device_with_inputs(axmodel_path, feeds)
    return dev


def _output_scale(out_dir):
    """y's dequant scale, read from the quant graph's own
    `AxDequantizeLinear` node -- see test_axera_relu_probe_hardware.py's
    module docstring for why this project's probes stopped assuming
    `replay.load_scales()` generalizes to every op."""
    m = onnx.load(os.path.join(out_dir, "quant", "quant_axmodel.onnx"))
    for n in m.graph.node:
        if n.op_type == "AxDequantizeLinear" and "y" in n.output:
            for a in n.attribute:
                if a.name == "input_scales":
                    return float(onnx.helper.get_attribute_value(a)[0])
    raise RuntimeError("no AxDequantizeLinear producing y found")


@pytest.mark.parametrize("tag", sorted(CASES))
def test_leakyrelu_probe_matches_numpy_reference(tmp_path, tag):
    lo, hi = CASES[tag]
    out_dir = _build(tmp_path, tag, lo, hi)
    if not pulsar2_docker.axcl_available():
        pytest.skip("no AXCL device")
    rng = np.random.default_rng(7)
    x = (rng.uniform(lo, hi, (1, 8))).astype(np.float32)
    dev = _run_retry_once(os.path.join(out_dir, "compiled.axmodel"), {"x": x.tobytes()})
    assert not dev.error, dev.error
    y = np.frombuffer(dev.outputs[0], dtype=np.float32).reshape(1, 8)
    y_scale = _output_scale(out_dir)
    ref = np.where(x >= 0, x, ALPHA * x)
    max_diff = float(np.abs(y - ref).max())
    assert max_diff <= LSB_BOUND * y_scale, (tag, max_diff, y_scale)
