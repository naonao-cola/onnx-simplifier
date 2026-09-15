"""Device numerics for the single-Clip mcode probe streams.

Twelfth op in the hardware-verified set, and a real surprise once
checked: it's the second op found (after Concat) that isn't quantized at
all, and a direct *contrast* to Relu's own finding rather than a
confirmation of it. Relu's `max(x, 0)` compiled through a real int8
path: `AxQuantizeLinear(x) -> AxClip(min=zp_x, max=255) ->
AxDequantizeLinear`, reusing x's own scale/zero point exactly (see that
probe's docstring). `Clip(x, min_v, max_v)` with finite, non-zero bounds
(`[-0.5, 0.5]`, not `[0, +inf)`) compiles completely differently: a bare
`AxClip(x) -> y` node with **no quantize/dequantize wrapper at all** and
float `min`/`max` attributes (not int8 codes) -- the same
not-quantized-at-all shape Concat's `AxConcat` has, not Relu's
scale-reusing one. Confirmed on the actual device, not just inferred
from the quant graph: feeding real x and comparing against
`np.clip(x, -0.5, 0.5)` gives `np.array_equal(...) == True`, bit-exact.

So among the two clamp-shaped ops probed so far, only the one whose
bounds are literally `[0, +inf)` (Relu) gets int8 treatment; a clamp
with arbitrary finite bounds does not. Whether that's because Pulsar2
specifically recognizes the `max(x, 0)` shape as "free to fold into an
existing int8 range" versus treating a general two-sided clamp as
something else entirely, or some other reason, isn't established here --
only the two data points (Relu quantized, general Clip not) are.

At opset 11+, `min`/`max` are optional graph *inputs* (constant
initializers here), not attributes -- confirmed against the real ONNX
opset-17 schema before writing this. This test asserts exact equality,
not an LSB-bounded tolerance the way every quantized probe in this set
does (there is no output scale to express a tolerance in terms of).

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

# tag -> (lo, hi): calibration/feed range. clip_min/clip_max (-0.5, 0.5)
# sit inside every one of these ranges, so every case genuinely exercises
# both the clamped and unclamped regions, not just one.
CASES = {
    "base": (-1.0, 1.0),
    "wide": (-10.0, 10.0),
    "positive": (0.0, 2.0),
}

CLIP_MIN, CLIP_MAX = -0.5, 0.5


def _build(tmp_path, tag, lo, hi):
    work_dir = os.path.join(str(tmp_path), f"clip_{tag}")
    os.makedirs(os.path.join(work_dir, "dataset"), exist_ok=True)
    model = parser.parse_model(
        '<ir_version: 8, opset_import: ["": 17]> '
        "agraph (float[1,8] x) => (float[1,8] y) "
        f"<float min_v = {{{CLIP_MIN}}}, float max_v = {{{CLIP_MAX}}}> "
        "{ y = Clip (x, min_v, max_v) }"
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
def test_clip_probe_matches_numpy_reference(tmp_path, tag):
    lo, hi = CASES[tag]
    out_dir = _build(tmp_path, tag, lo, hi)
    if not pulsar2_docker.axcl_available():
        pytest.skip("no AXCL device")
    rng = np.random.default_rng(7)
    x = (rng.uniform(lo, hi, (1, 8))).astype(np.float32)
    dev = _run_retry_once(os.path.join(out_dir, "compiled.axmodel"), {"x": x.tobytes()})
    assert not dev.error, dev.error
    y = np.frombuffer(dev.outputs[0], dtype=np.float32).reshape(1, 8)
    ref = np.clip(x, CLIP_MIN, CLIP_MAX)
    assert np.array_equal(y, ref), (tag, y, ref)
