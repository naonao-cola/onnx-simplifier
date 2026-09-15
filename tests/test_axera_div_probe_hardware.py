"""Device numerics for the single-Div mcode probe streams.

Seventh op in the hardware-verified set (Mul, Neg, Add, Sigmoid, Relu,
Sub -- see their own probe files). Interesting contrast to Mul: the
handoff notes record `Reciprocal` (the ONNX op computing `1/x` alone)
failing at Pulsar2's quantization stage (`Quant doesn't support
Reciprocal operation`) -- this checks whether `Div` (`x / y`, a binary
op) shares that fate or compiles and computes correctly as its own op,
not via a Reciprocal decomposition. Three `Div[1,8]` builds
(`pulsar2:7.0-lite`, MinMax Numpy calibration, in-calib-range feeds),
with y's calibration kept away from 0 in every case to avoid a
division-by-near-zero blowup that would be a stimulus problem, not a
hardware one.

Measured (2026-09-16): base 0.52, x10y01 0.52, wide_divisor 0.77
output-LSBs -- **Div compiles and computes correctly as its own op**,
unlike the standalone `Reciprocal` op the handoff notes record failing
at quantization. In the same 0.5-0.8 LSB band as the other binary ops
probed (Add 0.40-0.67, Sub 0.60-0.65, Mul 0.48-1.28), so whatever made
`Reciprocal` alone unsupported doesn't carry over to Div's own quant
lowering (division doesn't decompose through a separate, individually-
quantized reciprocal step the way the failing standalone op does).

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

# tag -> (x_lo, x_hi, y_lo, y_hi): calibration/feed ranges. y is always
# kept strictly positive and away from 0 -- unlike the other binary-op
# probes' symmetric ranges, a y range straddling 0 would let the divisor
# land arbitrarily close to 0 and blow up the reference, which tests
# stimulus sanity, not hardware accuracy.
CASES = {
    "base": (-1.0, 1.0, 0.5, 2.0),
    "x10y01": (-10.0, 10.0, 0.5, 2.0),
    "wide_divisor": (-1.0, 1.0, 2.0, 20.0),
}

LSB_BOUND = 1.5


def _build(tmp_path, tag, ranges):
    xlo, xhi, ylo, yhi = ranges
    work_dir = os.path.join(str(tmp_path), f"div_{tag}")
    os.makedirs(os.path.join(work_dir, "dataset"), exist_ok=True)
    model = parser.parse_model(
        '<ir_version: 8, opset_import: ["": 17]> '
        "agraph (float[1,8] x, float[1,8] y) => (float[1,8] z) "
        "{ z = Div (x, y) }"
    )
    onnx.save(model, os.path.join(work_dir, "model.onnx"))
    rng = np.random.default_rng(0)
    pulsar2_docker.make_numpy_calibration_tar(
        os.path.join(work_dir, "dataset", "x.tar"),
        [(rng.uniform(xlo, xhi, (1, 8))).astype(np.float32) for _ in range(8)],
    )
    pulsar2_docker.make_numpy_calibration_tar(
        os.path.join(work_dir, "dataset", "y.tar"),
        [(rng.uniform(ylo, yhi, (1, 8))).astype(np.float32) for _ in range(8)],
    )
    cfg = {
        "model_type": "ONNX",
        "npu_mode": "NPU1",
        "quant": {
            "input_configs": [
                {
                    "tensor_name": t,
                    "calibration_dataset": f"./dataset/{t}.tar",
                    "calibration_format": "Numpy",
                    "calibration_size": 8,
                }
                for t in ("x", "y")
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


@pytest.mark.parametrize("tag", sorted(CASES))
def test_div_probe_matches_numpy_reference(tmp_path, tag):
    out_dir = _build(tmp_path, tag, CASES[tag])
    if not pulsar2_docker.axcl_available():
        pytest.skip("no AXCL device")
    xlo, xhi, ylo, yhi = CASES[tag]
    rng = np.random.default_rng(7)
    x = (rng.uniform(xlo, xhi, (1, 8))).astype(np.float32)
    y = (rng.uniform(ylo, yhi, (1, 8))).astype(np.float32)
    dev = _run_retry_once(
        os.path.join(out_dir, "compiled.axmodel"),
        {"x": x.tobytes(), "y": y.tobytes()},
    )
    assert not dev.error, dev.error
    z = np.frombuffer(dev.outputs[0], dtype=np.float32).reshape(1, 8)
    scales = load_scales(out_dir)
    z_scale = float(np.asarray(scales["z"][1]).flat[0])
    max_diff = float(np.abs(z - (x / y)).max())
    assert max_diff <= LSB_BOUND * z_scale, (tag, max_diff, z_scale)
