"""Device numerics for the single-Relu mcode probe streams.

Fifth op in the hardware-verified set (Mul, Neg, Add, Sigmoid -- see
their own probe files), and a second nonlinear one, but piecewise-linear
rather than LUT-based: `max(x, 0)` is a comparison-and-select, not a
lookup table the way Sigmoid's curve is. Three `Relu[1,8]` builds
(`pulsar2:7.0-lite`, MinMax Numpy calibration, in-calib-range feeds) all
return the clamp at quantization-noise level against a numpy reference.

Genuinely different quant lowering from the other four ops, discovered
while writing this test: Relu compiles to `AxClip(min=zp_x, max=255)`
applied directly to x's own quantized code (`AxQuantizeLinear(x)` ->
`AxClip` -> `AxDequantizeLinear`), reusing x's scale and zero point
exactly rather than getting an independently-calibrated output scale the
way Mul/Add/Sigmoid's outputs do -- makes sense, since `max(x, 0)`'s
output range is a strict subset of x's own, so no new scale is needed.
Practical fallout: `replay.load_scales()` (used by the other probe
files) doesn't surface a "y" entry for a graph like this at all, fused
or not -- there is no dedicated quant-table entry to find, since y's
hash resolves to the same one x already has. This file reads the output
scale from the quant graph's `AxDequantizeLinear` node directly instead
(`_output_scale` below) rather than assuming every probe file's
load_scales() pattern generalizes.

Measured (2026-09-16):

- base (x in [-1, 1]): max|diff| 0.37 output-LSBs
- wide (x in [-10, 10]): 0.37 LSBs
- positive (x in [0.5, 2.0]): 0.46 LSBs

The lowest max-LSB figure of the five ops probed so far (Mul 0.48-1.28,
Neg flat 0.46, Add 0.40-0.67, Sigmoid 0.40-0.70) -- consistent with
reusing x's own scale/zero point exactly (no independent output
calibration to add its own rounding) rather than being a general
property of piecewise-linear ops.

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

# tag -> (lo, hi): calibration and feed range. Symmetric ranges straddling
# 0 exercise the clamp itself (half the calibration budget spent on the
# always-zero side); "positive" checks the all-positive, clamp-never-
# fires case behaves as a no-op, not just as "some values clamped".
CASES = {
    "base": (-1.0, 1.0),
    "wide": (-10.0, 10.0),
    "positive": (0.5, 2.0),
}

LSB_BOUND = 1.5


def _build(tmp_path, tag, lo, hi):
    work_dir = os.path.join(str(tmp_path), f"relu_{tag}")
    os.makedirs(os.path.join(work_dir, "dataset"), exist_ok=True)
    model = parser.parse_model(
        '<ir_version: 8, opset_import: ["": 17]> '
        "agraph (float[1,8] x) => (float[1,8] y) "
        "{ y = Relu (x) }"
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
    `AxDequantizeLinear` node -- see the module docstring for why
    `replay.load_scales()` doesn't have an entry for it here."""
    m = onnx.load(os.path.join(out_dir, "quant", "quant_axmodel.onnx"))
    for n in m.graph.node:
        if n.op_type == "AxDequantizeLinear" and "y" in n.output:
            for a in n.attribute:
                if a.name == "input_scales":
                    return float(onnx.helper.get_attribute_value(a)[0])
    raise RuntimeError("no AxDequantizeLinear producing y found")


@pytest.mark.parametrize("tag", sorted(CASES))
def test_relu_probe_matches_numpy_reference(tmp_path, tag):
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
    max_diff = float(np.abs(y - np.maximum(x, 0.0)).max())
    assert max_diff <= LSB_BOUND * y_scale, (tag, max_diff, y_scale)
