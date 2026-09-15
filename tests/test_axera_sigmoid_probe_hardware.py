"""Device numerics for the single-Sigmoid mcode probe streams.

Fourth op in the hardware-verified set alongside Mul, Neg and Add (see
their own probe files), and the first genuinely *nonlinear* one: Sigmoid
is LUT-based on this hardware, not a scale/requant computation, so it is
a useful contrast point for whether accuracy patterns seen in the
arithmetic ops (Mul/Neg/Add) generalize to a different compute class.
Three `Sigmoid[1,8]` builds (`pulsar2:7.0-lite`, MinMax Numpy
calibration, in-calib-range feeds) all return the sigmoid at
quantization-noise level against a numpy reference.

Measured (2026-09-16):

- base (x in [-1, 1]): max|diff| 0.40 output-LSBs
- wide (x in [-6, 6]): 0.70 LSBs
- positive (x in [0.5, 2.0]): 0.48 LSBs

In the same 0.4-0.7 LSB band as the arithmetic ops (Add: 0.40-0.67, Neg:
0.46 flat, Mul: 0.48-1.28) -- accuracy at this level does not appear to
be specific to linear/multiplicative quant math; LUT-based nonlinear ops
land in the same range.

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
from replay import load_scales  # noqa: E402

_IMAGE = "pulsar2:7.0-lite"
pytestmark = pytest.mark.skipif(
    not pulsar2_docker.docker_image_available(_IMAGE),
    reason=f"pulsar2 Docker image not loaded: {_IMAGE}",
)

# tag -> (lo, hi): calibration and feed range. Sigmoid saturates fast, so
# "wide" here is much narrower than the arithmetic probes' 10.0 -- a
# [-10, 10] calibration range would spend almost the entire 256-code
# budget on an essentially-flat 0/1 region, which tests LUT saturation
# behavior, not accuracy; [-6, 6] keeps the useful (non-flat) part of the
# curve inside the calibration range instead.
CASES = {
    "base": (-1.0, 1.0),
    "wide": (-6.0, 6.0),
    "positive": (0.5, 2.0),
}

LSB_BOUND = 1.5


def _build(tmp_path, tag, lo, hi):
    work_dir = os.path.join(str(tmp_path), f"sigmoid_{tag}")
    os.makedirs(os.path.join(work_dir, "dataset"), exist_ok=True)
    model = parser.parse_model(
        '<ir_version: 8, opset_import: ["": 17]> '
        "agraph (float[1,8] x) => (float[1,8] y) "
        "{ y = Sigmoid (x) }"
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


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@pytest.mark.parametrize("tag", sorted(CASES))
def test_sigmoid_probe_matches_numpy_reference(tmp_path, tag):
    lo, hi = CASES[tag]
    out_dir = _build(tmp_path, tag, lo, hi)
    if not pulsar2_docker.axcl_available():
        pytest.skip("no AXCL device")
    rng = np.random.default_rng(7)
    x = (rng.uniform(lo, hi, (1, 8))).astype(np.float32)
    dev = _run_retry_once(os.path.join(out_dir, "compiled.axmodel"), {"x": x.tobytes()})
    assert not dev.error, dev.error
    y = np.frombuffer(dev.outputs[0], dtype=np.float32).reshape(1, 8)
    scales = load_scales(out_dir)
    y_scale = float(np.asarray(scales["y"][1]).flat[0])
    max_diff = float(np.abs(y - _sigmoid(x)).max())
    assert max_diff <= LSB_BOUND * y_scale, (tag, max_diff, y_scale)
