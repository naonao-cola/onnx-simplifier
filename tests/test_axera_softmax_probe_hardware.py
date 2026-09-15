"""Device numerics for the single-Softmax mcode probe streams.

Sixteenth op in the hardware-verified set, and the most compound one
probed: `Softmax` is `exp` (nonlinear, LUT-based like Sigmoid/Tanh) over
a `ReduceSum`-normalized division, three of the op classes already
probed individually composed into one. Feed ranges are kept modest
(unlike the arithmetic probes' `[-10, 10]`): softmax saturates toward a
one-hot output for widely-separated inputs (the largest element's
`exp(x)` dominates the sum), which would mostly test saturation
behavior rather than accuracy, the same reasoning behind Sigmoid/Tanh's
narrower `wide` ranges.

Measured (2026-09-16): base 0.63, wide 1.16, positive 0.43 output-LSBs --
compiles to a single `AxQuantizedSoftmax` node (confirmed by inspecting
the quant graph: one op, not the exp/reduce/div decomposition the module
docstring describes as the *conceptual* three-op composition -- Pulsar2
fuses the whole thing into one hardware primitive, it does not literally
chain Sigmoid/Tanh-style building blocks). `wide`'s figure is the
highest of the three, consistent with softmax's steepest, most
code-sensitive region (near the largest input) shifting further from
the bulk of the calibrated range as the input range widens.

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
    "wide": (-3.0, 3.0),
    "positive": (0.0, 2.0),
}

LSB_BOUND = 1.5


def _build(tmp_path, tag, lo, hi):
    work_dir = os.path.join(str(tmp_path), f"softmax_{tag}")
    os.makedirs(os.path.join(work_dir, "dataset"), exist_ok=True)
    model = parser.parse_model(
        '<ir_version: 8, opset_import: ["": 17]> '
        "agraph (float[1,8] x) => (float[1,8] y) "
        "{ y = Softmax<axis=1> (x) }"
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


def _softmax(x, axis):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


@pytest.mark.parametrize("tag", sorted(CASES))
def test_softmax_probe_matches_numpy_reference(tmp_path, tag):
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
    max_diff = float(np.abs(y - _softmax(x, axis=1)).max())
    assert max_diff <= LSB_BOUND * y_scale, (tag, max_diff, y_scale)
