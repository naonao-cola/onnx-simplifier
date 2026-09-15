"""Device numerics for a patched (not rebuilt) Mul mcode stream.

Builds two real Mul[1,8] axmodels that differ only in x's calibration
range -- chosen so both land x's zero point in the one form this project
has actually decoded (the literal `02 10 1b <zp_x> 83 36` unit, see
tests/test_axera_mcode_reciprocal.py's TestZpXLiteralByteWhenPresent).
Patches the first build's mcode (scales via tiny_emit.patch_mul_scales,
the output quad via patch_mul_output_quad, x's zero point via
patch_mul_zp_x -- 17 bytes total) to carry the second build's values,
writes the patched bytes back into the first build's own `.axmodel`
container, and runs both the patched artifact and the real second build
on the AX650N with the same input.

Result (2026-09-16, one build pair, x shifted 0.31 vs 0.32 giving zp_x
33 vs 35): the patched stream's device output is bit-exact identical to
the real rebuild's -- not just within quantization noise, `==` on every
element. Neither zp_y (both builds keep it 0 by construction) nor zp_z
(left unpatched; the two builds' output zero points differ, 30 vs 32) are
touched, and their difference does not show up in the dequantized float32
output either build returns -- consistent with zero points affecting
only the internal int8 arithmetic, not the two things patched here.

This is one data point, not a general proof: it says patching these
three families is *sufficient* when both source and target land in the
literal zp_x form and only scales/zp_x differ, not that it is sufficient
whenever the opaque zp_x forms are involved (untested -- patch_mul_zp_x
raises rather than guessing there) or that zp_y ever needs patching (also
untested -- no known encoding for it, see the module docstring).

Needs a loaded `pulsar2:*` Docker image to (re)build and an AX650N card
to run -- skip-guarded on both, like the rest of the hardware suite.
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

import mcode  # noqa: E402
import pulsar2_docker  # noqa: E402
import tiny_emit  # noqa: E402

# Checked against "pulsar2:7.0-lite" specifically, not
# pulsar2_docker.DEFAULT_IMAGE ("pulsar2:6.0-lite") -- this test (like
# test_axera_mul_probe_hardware.py) builds with 7.0-lite explicitly below;
# a host with only 7.0-lite loaded would otherwise skip for the wrong
# reason.
_IMAGE = "pulsar2:7.0-lite"
pytestmark = pytest.mark.skipif(
    not pulsar2_docker.docker_image_available(_IMAGE),
    reason=f"pulsar2 Docker image not loaded: {_IMAGE}",
)

# Both give x_scale ~0.0076048 (span-invariant to the shift) with zp_x in
# the literal form -- confirmed offline against the committed
# mul_1x8_zp33sweep/zp35sweep fixtures before spending a device run on it.
_SHIFT_REF = 0.31  # -> zp_x = 33
_SHIFT_TGT = 0.32  # -> zp_x = 35
_X_LO, _X_HI = 0.05, 2.0
_Y_LO, _Y_HI = 0.05, 2.0


def _build(tmp_path, tag, x_shift):
    work_dir = os.path.join(str(tmp_path), f"mul_{tag}")
    os.makedirs(os.path.join(work_dir, "dataset"), exist_ok=True)
    model = parser.parse_model(
        '<ir_version: 8, opset_import: ["": 17]> '
        "agraph (float[1,8] x, float[1,8] y) => (float[1,8] z) "
        "{ z = Mul (x, y) }"
    )
    onnx.save(model, os.path.join(work_dir, "model.onnx"))
    # One continuous rng stream, x drawn before y -- matches the scratch
    # recipe (mul_poscalib.py) this test's shift constants were verified
    # against offline; drawing y from a separately-seeded stream would
    # change its calibration values and invalidate the "0.31 -> zp_x=33,
    # 0.32 -> zp_x=35" facts asserted below.
    rng = np.random.default_rng(0)
    x_cal = [
        (rng.uniform(_X_LO, _X_HI, (1, 8)) - x_shift).astype(np.float32)
        for _ in range(8)
    ]
    y_cal = [(rng.uniform(_Y_LO, _Y_HI, (1, 8))).astype(np.float32) for _ in range(8)]
    pulsar2_docker.make_numpy_calibration_tar(
        os.path.join(work_dir, "dataset", "x.tar"), x_cal
    )
    pulsar2_docker.make_numpy_calibration_tar(
        os.path.join(work_dir, "dataset", "y.tar"), y_cal
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
        image="pulsar2:7.0-lite",
        target_hardware="AX650",
        timeout=1200,
    )
    assert result.success, result.error
    return os.path.join(work_dir, "out")


def _quant_params(quant_axmodel_path):
    m = onnx.load(quant_axmodel_path)
    for n in m.graph.node:
        if n.op_type == "AxQuantizedMul":
            params = {}
            for a in n.attribute:
                v = onnx.helper.get_attribute_value(a)
                if a.name == "input_scales":
                    params["xs"], params["ys"] = list(v)
                if a.name == "input_zeropoints":
                    params["zpx"], params["zpy"] = list(v)
                if a.name == "output_scales":
                    params["zs"] = list(v)[0]
                if a.name == "output_zeropoints":
                    params["zpz"] = list(v)[0]
            return params
    raise RuntimeError("no AxQuantizedMul in quant_axmodel.onnx")


def _write_patched_axmodel(ref_axmodel_path, patched_mcode, out_path):
    model = onnx.load(ref_axmodel_path)
    node = next(n for n in model.graph.node if n.op_type == "neu mode")
    info = json.loads(
        next(a for a in node.attribute if a.name == "npu_graph_info").s.decode()
    )
    key = info["dotneus"][0]["neu_key"]
    init = next(i for i in model.graph.initializer if i.name == key)
    orig = numpy_helper.to_array(init)
    assert orig.nbytes == len(patched_mcode), (orig.nbytes, len(patched_mcode))
    new_arr = np.frombuffer(patched_mcode, dtype=orig.dtype).reshape(orig.shape)
    init.CopyFrom(numpy_helper.from_array(new_arr.copy(), key))
    onnx.save(model, out_path)


def _run_retry_once(axmodel_path, feeds):
    dev = pulsar2_docker.run_on_device_with_inputs(axmodel_path, feeds)
    if dev.error and "0x8030070C" in dev.error:
        dev = pulsar2_docker.run_on_device_with_inputs(axmodel_path, feeds)
    return dev


def test_patched_stream_matches_real_rebuild_bit_exactly(tmp_path):
    ref_dir = _build(tmp_path, "ref", _SHIFT_REF)
    tgt_dir = _build(tmp_path, "tgt", _SHIFT_TGT)

    ref_q = _quant_params(os.path.join(ref_dir, "quant", "quant_axmodel.onnx"))
    tgt_q = _quant_params(os.path.join(tgt_dir, "quant", "quant_axmodel.onnx"))
    assert ref_q["zpy"] == 0 and tgt_q["zpy"] == 0  # this test doesn't cover zp_y
    # Confirmed offline against the committed sweep fixtures before ever
    # spending a device run: if this drifts, the calibration recipe above
    # no longer matches what it was verified against, and everything
    # below needs re-checking before trusting it.
    assert (ref_q["zpx"], tgt_q["zpx"]) == (33, 35), (ref_q["zpx"], tgt_q["zpx"])

    ref_ax = os.path.join(ref_dir, "compiled.axmodel")
    tgt_ax = os.path.join(tgt_dir, "compiled.axmodel")

    ref_mcode = mcode.mcodes_of(ref_ax)[0][1]
    patched = tiny_emit.patch_mul_scales(
        ref_mcode,
        (ref_q["xs"], ref_q["ys"], ref_q["zs"]),
        (tgt_q["xs"], tgt_q["ys"], tgt_q["zs"]),
    )
    patched = tiny_emit.patch_mul_output_quad(patched, ref_q["zs"], tgt_q["zs"])
    patched = tiny_emit.patch_mul_zp_x(patched, ref_q["zpx"], tgt_q["zpx"])
    assert mcode.check(patched) == []

    patched_ax = os.path.join(str(tmp_path), "patched.axmodel")
    _write_patched_axmodel(ref_ax, patched, patched_ax)

    if not pulsar2_docker.axcl_available():
        pytest.skip("no AXCL device")

    rng = np.random.default_rng(3)
    x = (rng.uniform(-0.2, 1.6, (1, 8))).astype(np.float32)
    y = (rng.uniform(0.1, 1.8, (1, 8))).astype(np.float32)
    feeds = {"x": x.tobytes(), "y": y.tobytes()}

    dev_tgt = _run_retry_once(tgt_ax, feeds)
    dev_patched = _run_retry_once(patched_ax, feeds)
    assert not dev_tgt.error, dev_tgt.error
    assert not dev_patched.error, dev_patched.error

    z_tgt = np.frombuffer(dev_tgt.outputs[0], dtype=np.float32)
    z_patched = np.frombuffer(dev_patched.outputs[0], dtype=np.float32)
    assert np.array_equal(z_patched, z_tgt), (z_patched, z_tgt)

    # And both are within quantization noise of the float reference, so
    # "bit-exact vs target" isn't bit-exact against a broken target.
    z_numpy = x * y
    max_lsb = float(np.abs(z_tgt - z_numpy.reshape(-1)).max() / tgt_q["zs"])
    assert max_lsb <= 1.5, max_lsb
