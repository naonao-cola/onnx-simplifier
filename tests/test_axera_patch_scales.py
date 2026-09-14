"""Patching recalibrated scales into a compiled `.axmodel`, without Docker.

`scripts/axera/patch_scales.py` rewrites a compiled artifact's scale slots
in place instead of rebuilding it. These tests build a synthetic model and
quant table (no Pulsar2, no card) and check the slot-finding, the patch,
and the audit trail.
"""

import json
import os
import struct
import sys

import numpy as np
import onnx
import onnx.parser
from onnx import numpy_helper

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import patch_scales  # noqa: E402


def _f32(x):
    return struct.pack("<f", np.float32(x))


def _bf16(x):
    return struct.pack("<H", patch_scales.bf16_trunc(x))


def _write_table(path, scales):
    """A minimal quant table: `{tensor: scale}` with asymmetric policy."""
    os.makedirs(os.path.join(path, "quant"), exist_ok=True)
    configs, values = {}, {}
    for i, (tensor, scale) in enumerate(scales.items()):
        h = str(5000 + i)
        values[h] = {"scale": [scale], "zero_point": [0.0]}
        configs.setdefault("op", {})[tensor] = {
            "bit_width": 8,
            "policy": {"SYMMETRICAL": False, "ASYMMETRICAL": True},
            "hash": int(h),
            "quant_min": 0,
            "quant_max": 255,
        }
    with open(os.path.join(path, "quant", "quant_axmodel.json"), "w") as f:
        json.dump({"tensor_configs": configs, "values": values}, f)
    return os.path.join(path, "quant", "quant_axmodel.json")


def _write_model(path, blob):
    """A minimal `.axmodel`: one `neu mode` node holding `blob`."""
    model = onnx.parser.parse_model(
        '<ir_version: 10, opset_import: ["": 17]>'
        " g (float[4] x) => (float[4] y) { y = Identity (x) }"
    )
    info = json.dumps({"dotneus": [{"neu_key": "mcode_blob"}]})
    node = onnx.helper.make_node("neu mode", ["x"], ["y"])
    node.attribute.append(onnx.helper.make_attribute("npu_graph_info", info.encode()))
    model.graph.node.append(node)
    arr = np.frombuffer(bytes(blob), dtype=np.uint16).copy()
    model.graph.initializer.append(numpy_helper.from_array(arr, "mcode_blob"))
    # pad the blob to a whole number of uint16 without changing its bytes
    assert len(blob) % 2 == 0
    p = os.path.join(path, "model.axmodel")
    onnx.save(model, p)
    return p


def test_bf16_trunc_keeps_the_top_half():
    assert patch_scales.bf16_trunc(1.0) == 0x3F80
    assert patch_scales.bf16_trunc(0.5) == 0x3F00
    # truncation, not rounding: 1.0039 rounds up in float32, truncates down
    assert patch_scales.bf16_trunc(1.0039) == 0x3F80


def test_find_scale_slots_matches_all_encodings(tmp_path):
    blob = (
        b"\xff" * 16
        + _f32(0.5)  # f32 direct
        + b"\xff" * 16
        + _bf16(2.0)  # bf16 direct
        + b"\xff" * 16
        + _bf16(1 / 0.25)  # bf16 reciprocal
        + b"\xff" * 16
    )
    old = {"a": 0.5, "b": 2.0, "c": 0.25}
    new = {"a": 1.0, "b": 4.0, "c": 0.5}
    slots = patch_scales.find_scale_slots(blob, old, new)
    assert [(s[0], s[1], s[2]) for s in slots] == [
        (16, "a", "f32"),
        # offset 36 matches both b-direct and a-reciprocal encodings of the
        # same value 2.0 -- ambiguous across tensors, so dropped, not guessed.
        (54, "c", "bf16recip"),
    ]
    # unchanged scales never match, so constants are safe.
    assert patch_scales.find_scale_slots(blob, old, dict(old)) == []


def test_patch_model_rewrites_and_reports(tmp_path):
    blob = b"\x11" * 8 + _f32(0.5) + b"\x22" * 8
    model = _write_model(str(tmp_path), blob)
    old_t = _write_table(str(tmp_path / "old"), {"a": 0.5})
    new_t = _write_table(str(tmp_path / "new"), {"a": 1.0})
    out = str(tmp_path / "patched.axmodel")
    slots = patch_scales.patch_model(model, old_t, new_t, out)
    assert len(slots) == 1 and slots[0][:3] == (8, "a", "f32")
    patched, _ = patch_scales.mcode_bytes(onnx.load(out))
    assert patched[8:12] == _f32(1.0)
    assert patched[:8] == b"\x11" * 8 and patched[12:] == b"\x22" * 8


def test_patch_model_refuses_when_nothing_changed(tmp_path):
    blob = b"\x00" * 16 + _f32(0.5) + b"\x00" * 16
    model = _write_model(str(tmp_path), blob)
    table = _write_table(str(tmp_path / "same"), {"a": 0.5})
    try:
        patch_scales.patch_model(model, table, table, str(tmp_path / "o.axmodel"))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for identical tables")


def test_nested_matches_keep_the_widest():
    """A float32 slot contains its own bfloat16 truncation as nested bytes;
    patching both would corrupt the float. Widest wins -- this is the
    `qk`-scale shape from a real attention build, where `[9f 8f]` reads as
    a bare pair but belongs to the float."""
    blob = b"\xff" * 8 + _f32(0.108672) + b"\xff" * 8
    slots = patch_scales.find_scale_slots(blob, {"x": 0.108672}, {"x": 0.217345})
    assert [(s[0], s[2]) for s in slots] == [(8, "f32")]
    patched = bytearray(blob)
    for offset, _, _, old_b, new_b in slots:
        patched[offset : offset + len(old_b)] = new_b
    assert bytes(patched[8:12]) == _f32(0.217345)
