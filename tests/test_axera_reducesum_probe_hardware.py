"""Device numerics for the single-ReduceSum mcode probe streams.

Eighth op in the hardware-verified set (Mul, Neg, Add, Sub, Div,
Sigmoid, Relu -- see their own probe files), and the first reduction:
`ReduceSum[1,8] -> [1,1]` collapses eight quantized codes into one,
genuinely different from every elementwise op probed so far (output
shape differs from input shape; the quant math accumulates eight terms
instead of combining one or two). At opset 17, `ReduceSum`'s axes are a
second graph *input* (an int64 tensor), not an attribute -- unlike
`ReduceMean` at the same opset (see that probe's own docstring) -- so
this graph carries `axes` as a constant initializer rather than a node
attribute.

Measured (2026-09-16): base 0.41, wide 0.41, positive 0.44 output-LSBs --
notably tight and consistent (unlike the binary ops' wider spreads), and
(see ReduceMean's own probe) numerically identical to ReduceMean's
figures for the matching ranges, consistent with MinMax calibration
scaling the mean's range down by exactly the same factor (8, the
reduced axis length) as the sum-vs-mean relationship itself, leaving the
noise-as-a-fraction-of-scale unchanged between the two ops.

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

# tag -> (lo, hi): calibration and feed range. A sum over 8 terms can
# land well outside any single term's own range, so the output scale
# (calibrated from the *sum*, not the input) absorbs that automatically --
# these ranges are chosen the same way as the other single-input probes,
# not specially widened for the reduction.
CASES = {
    "base": (-1.0, 1.0),
    "wide": (-10.0, 10.0),
    "positive": (0.5, 2.0),
}

LSB_BOUND = 1.5


def _build(tmp_path, tag, lo, hi):
    work_dir = os.path.join(str(tmp_path), f"reducesum_{tag}")
    os.makedirs(os.path.join(work_dir, "dataset"), exist_ok=True)
    model = parser.parse_model(
        '<ir_version: 8, opset_import: ["": 17]> '
        "agraph (float[1,8] x) => (float[1,1] y) "
        "<int64[1] axes = {1}> "
        "{ y = ReduceSum (x, axes) }"
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


@pytest.mark.parametrize("tag", sorted(CASES))
def test_reducesum_probe_matches_numpy_reference(tmp_path, tag):
    lo, hi = CASES[tag]
    out_dir = _build(tmp_path, tag, lo, hi)
    if not pulsar2_docker.axcl_available():
        pytest.skip("no AXCL device")
    rng = np.random.default_rng(7)
    x = (rng.uniform(lo, hi, (1, 8))).astype(np.float32)
    dev = _run_retry_once(os.path.join(out_dir, "compiled.axmodel"), {"x": x.tobytes()})
    assert not dev.error, dev.error
    y = np.frombuffer(dev.outputs[0], dtype=np.float32).reshape(1, 1)
    scales = load_scales(out_dir)
    y_scale = float(np.asarray(scales["y"][1]).flat[0])
    max_diff = float(np.abs(y - x.sum(axis=1, keepdims=True)).max())
    assert max_diff <= LSB_BOUND * y_scale, (tag, max_diff, y_scale)
