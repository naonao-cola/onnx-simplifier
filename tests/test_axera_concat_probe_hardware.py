"""Device numerics for the single-Concat mcode probe streams.

Eleventh op in the hardware-verified set, and the only one so far that
isn't quantized at all: `Concat[1,8]+[1,8] -> [1,16]` along axis 1
compiles to `AxConcat(x, y) -> z` with `output_dtype=FP32` and an
*empty* quant table (no `tensor_configs` entry, confirmed by inspecting
`quant_axmodel.json` directly -- there is no int8 calibration for `z` to
even try reading, unlike every other probe in this set including Relu's
scale-reusing `AxClip`, which at least has a real int8 output). Every
other op probed compiles through an `AxQuantizeLinear`/
`AxDequantizeLinear` pair; Concat's inputs and output here are plain
float32 straight through.

Confirmed on the actual device, not just inferred from the quant graph:
feeding real (differently-ranged, per `CASES` below) x/y and comparing
against `np.concatenate([x, y], axis=1)` gives `np.array_equal(...) ==
True` -- a bit-exact match, not merely within quantization noise, since
there is no quantization step to introduce noise. This test asserts
exact equality, not an LSB-bounded tolerance the way every other probe
in this set does (there is no output scale to express a tolerance in
terms of).

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

# tag -> (x_lo, x_hi, y_lo, y_hi): deliberately different ranges for the
# two inputs in every case (unlike a same-range sanity case) -- the
# point of this probe is exactly the "two different scales feeding one
# concat" path.
CASES = {
    "base": (-1.0, 1.0, -1.0, 1.0),
    "x10y01": (-10.0, 10.0, -0.1, 0.1),
    "x01y10": (-0.1, 0.1, -10.0, 10.0),
}


def _build(tmp_path, tag, ranges):
    xlo, xhi, ylo, yhi = ranges
    work_dir = os.path.join(str(tmp_path), f"concat_{tag}")
    os.makedirs(os.path.join(work_dir, "dataset"), exist_ok=True)
    model = parser.parse_model(
        '<ir_version: 8, opset_import: ["": 17]> '
        "agraph (float[1,8] x, float[1,8] y) => (float[1,16] z) "
        "{ z = Concat<axis=1> (x, y) }"
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
def test_concat_probe_matches_numpy_reference(tmp_path, tag):
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
    z = np.frombuffer(dev.outputs[0], dtype=np.float32).reshape(1, 16)
    ref = np.concatenate([x, y], axis=1)
    assert np.array_equal(z, ref), (tag, z, ref)
