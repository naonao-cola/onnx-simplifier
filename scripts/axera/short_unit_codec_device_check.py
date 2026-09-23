"""Device check for ``short_unit_codec``: do re-encoded MCode segments run on
the AX8850 exactly like the native ones?

``short_unit_codec`` decompresses every MCode segment into plain 8-byte
register records and re-encodes them. Its tests prove the codec round-trips
in software. This module asks the hardware question, using only committed
fixtures, with no Pulsar2 builds. Every variant is compared against a native
model's device output, bit for bit:

* ``roundtrip``: every compressed segment is re-encoded from its own decoded
  bytes. Only segments whose new stream differs from the native bytes make
  this a real test.
* ``scale_edit``: one scale record in a Relu's dequantize block is scaled by a
  known factor. The predicted effect is that exactly one output lane changes,
  by exactly that factor.
* ``transplant``: the records that differ between two native builds of the
  same graph are copied from one into the other's decompressed segments,
  which are then re-encoded. The result must behave exactly like the second
  native build. This covers zero-point changes (a Relu zero-point sweep) and
  a step recalibration (``step_recalib/toyf_A1`` -> ``toyf_Dmom100``).

Findings are in ``docs/axera-short-unit-codec-device.md``.

Usage::

    short_unit_codec_device_check.py device OUT.json   # needs axcl-vm + the card
"""

from __future__ import annotations

import gzip
import json
import os
import struct
import sys
import tempfile

import numpy as np
import onnx

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import short_unit_codec as codec  # noqa: E402

FIX = os.path.join(_HERE, "fixtures")
OWN = os.path.join(FIX, "short_unit_codec_device")
_LOCK = "/tmp/axcl-device.lock"


def load(path: str) -> onnx.ModelProto:
    data = gzip.open(path).read() if path.endswith(".gz") else open(path, "rb").read()
    return onnx.load_model_from_string(data)


def mcode_name(model: onnx.ModelProto) -> str:
    return next(i.name for i in model.graph.initializer if i.name.endswith("_neu"))


def get_mcode(model: onnx.ModelProto) -> bytes:
    name = mcode_name(model)
    return next(bytes(i.raw_data) for i in model.graph.initializer if i.name == name)


def with_mcode(model: onnx.ModelProto, mc: bytes) -> onnx.ModelProto:
    out = onnx.ModelProto()
    out.CopyFrom(model)
    name = mcode_name(out)
    for init in out.graph.initializer:
        if init.name == name:
            init.raw_data = mc
    return out


def reencode_all(mc: bytes) -> tuple[bytes, list[int]]:
    """Re-encode every compressed segment from its own decoded bytes.

    Returns the new MCode and the indices of segments whose new stream differs
    from the native one (the ones that actually test the encoder)."""
    changed = []
    raws = codec.decode_segments(mc)
    for i, (start, size, _, comp) in enumerate(codec.segment_streams(mc)):
        if not comp:
            continue
        new = codec.replace_segment(mc, i, raws[i])
        s2, n2, _, _ = codec.segment_streams(new)[i]
        if new[s2 : s2 + n2] != mc[start : start + size]:
            changed.append(i)
        mc = new
    return mc, changed


def diff_records(a: bytes, b: bytes) -> list[int]:
    """Record indices at which two equal-length decoded segments differ."""
    if len(a) != len(b):
        raise codec.CodecError(f"decoded lengths differ: {len(a)} vs {len(b)}")
    return [r // 8 for r in range(0, len(a), 8) if a[r : r + 8] != b[r : r + 8]]


def transplant(
    mc_from: bytes, mc_to: bytes, segments: list[int] | None = None
) -> tuple[bytes, dict]:
    """Copy every differing record of ``mc_to``'s decoded segments into
    ``mc_from``'s, then re-encode those segments in ``mc_from``'s slots.

    ``segments`` limits which segments are transplanted (default: all
    compressed segments that differ). Raises ``CodecError`` when a decoded
    segment's length differs (not a record edit) or a new stream outgrows its
    padded slot. Returns the new MCode and ``{segment: n_records}``."""
    ra, rb = codec.decode_segments(mc_from), codec.decode_segments(mc_to)
    streams = codec.segment_streams(mc_from)
    moved = {}
    out = mc_from
    for i, (_, _, _, comp) in enumerate(streams):
        if segments is not None and i not in segments:
            continue
        recs = diff_records(ra[i], rb[i])
        if not recs:
            continue
        if not comp:
            raise codec.CodecError(f"segment {i} differs but is not compressed")
        out = codec.replace_segment(out, i, rb[i])
        moved[i] = len(recs)
    return out, moved


def scale_edit(mc: bytes, segment: int, reg: int, occurrence: int, factor: float):
    """Scale the float32 value of the ``occurrence``-th record writing ``reg``
    in ``segment``. Returns ``(new mcode, old value, new value)``."""
    raw = bytearray(codec.decode_segments(mc)[segment])
    hits = [r for r in codec.records(bytes(raw)) if r["reg"] == reg]
    rec = hits[occurrence]
    at = rec["index"] * 8 + 4
    old = struct.unpack_from("<f", raw, at)[0]
    new = np.float32(old * factor).item()
    struct.pack_into("<f", raw, at, new)
    return codec.replace_segment(mc, segment, bytes(raw)), old, new


def inputs_for(model: onnx.ModelProto, seed: int = 7) -> dict:
    """Deterministic in-range float32 inputs for every graph input."""
    rng = np.random.RandomState(seed)
    feeds = {}
    for v in model.graph.input:
        shape = [d.dim_value for d in v.type.tensor_type.shape.dim]
        if v.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
            raise ValueError(f"input {v.name} is not float32")
        if v.name in ("lr",):
            arr = np.full(shape, 0.01, np.float32)
        elif v.name in ("m_correction", "v_correction"):
            arr = np.ones(shape, np.float32)
        elif v.name == "batch_size":
            arr = np.full(shape, 32.0, np.float32)
        else:
            arr = rng.uniform(-0.8, 0.8, shape).astype(np.float32)
        feeds[v.name] = arr
    return feeds


def run(model: onnx.ModelProto, feeds: dict) -> dict:
    """One device run, holding the shared device lock for this run only."""
    import fcntl

    from teng2_fault_injection import _run_on_device_with_inputs

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "m.axmodel")
        onnx.save(model, path)
        fd = os.open(_LOCK, os.O_CREAT | os.O_RDWR, 0o666)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            return _run_on_device_with_inputs(
                path, {k: v.tobytes() for k, v in feeds.items()}, timeout=180
            )
        finally:
            os.close(fd)


def _compare(a: list | None, b: list | None) -> dict:
    if a is None or b is None:
        return {"equal": False, "note": "a run produced no outputs"}
    if len(a) != len(b):
        return {"equal": False, "note": "output count differs"}
    diffs = [
        float(np.abs(np.frombuffer(x, np.float32) - np.frombuffer(y, np.float32)).max())
        for x, y in zip(a, b)
    ]
    return {"equal": a == b, "max_abs_diff": max(diffs) if diffs else 0.0}


def lane_effect(control: bytes, patched: bytes, lanes: int = 4) -> dict:
    """Per output lane (flat index mod ``lanes``): how many elements changed,
    and the patched/control ratio over changed nonzero elements."""
    c = np.frombuffer(control, np.float32)
    p = np.frombuffer(patched, np.float32)
    res = {}
    for lane in range(lanes):
        cl, pl = c[lane::lanes], p[lane::lanes]
        changed = cl != pl
        nz = changed & (cl != 0)
        ratio = pl[nz] / cl[nz] if nz.any() else np.array([])
        res[lane] = {
            "changed": int(changed.sum()),
            "n": int(cl.size),
            "ratio_min": float(ratio.min()) if ratio.size else None,
            "ratio_max": float(ratio.max()) if ratio.size else None,
        }
    return res


# (name, template fixture, target fixture) for transplant checks.
RELU = os.path.join(FIX, "dma_tiles", "relu_1x64x56x56.axmodel.gz")
TRANSPLANTS = [
    ("relu_zp127_to_zp100", RELU, os.path.join(OWN, "relu_1x64x56x56_z100.axmodel.gz")),
    ("relu_zp127_to_zp64", RELU, os.path.join(OWN, "relu_1x64x56x56_z64.axmodel.gz")),
    ("relu_zp127_to_zp200", RELU, os.path.join(OWN, "relu_1x64x56x56_z200.axmodel.gz")),
    (
        "step_A1_to_Dmom100",
        os.path.join(FIX, "step_recalib", "toyf_A1.axmodel.gz"),
        os.path.join(FIX, "step_recalib", "toyf_Dmom100.axmodel.gz"),
    ),
]
ROUNDTRIPS = [
    os.path.join(FIX, "binary_op_registers", "add_c2.axmodel.gz"),
    os.path.join(FIX, "binary_op_registers", "sub_c3.axmodel.gz"),
    os.path.join(FIX, "step_recalib", "toyf_A1.axmodel.gz"),
]


def device(out_path: str) -> None:
    results: dict = {"roundtrip": [], "scale_edit": [], "transplant": []}

    def save():
        with open(out_path, "w") as f:
            json.dump(results, f, indent=1)

    def control_and_health(model, feeds, label):
        ctl = run(model, feeds)
        if ctl["outputs"] is None:
            raise RuntimeError(f"control run failed for {label}: {ctl['error']}")
        return ctl["outputs"]

    for path in ROUNDTRIPS:
        model = load(path)
        feeds = inputs_for(model)
        ctl = control_and_health(model, feeds, path)
        new, changed = reencode_all(get_mcode(model))
        res = run(with_mcode(model, new), feeds)
        health = run(model, feeds)["outputs"] == ctl
        results["roundtrip"].append(
            {
                "fixture": os.path.relpath(path, FIX),
                "reencoded_segments_that_differ": changed,
                **_compare(ctl, res["outputs"]),
                "error": res["error"],
                "health_after": health,
            }
        )
        save()

    model = load(RELU)
    feeds = inputs_for(model)
    ctl = control_and_health(model, feeds, "relu")
    for reg in (0x0F60, 0x0F70, 0x0F80):
        # occurrence 1 = the dequantize block's copy (s_y); never 0x0f50.
        new, old, newv = scale_edit(get_mcode(model), 2, reg, 1, 2.0)
        res = run(with_mcode(model, new), feeds)
        health = run(model, feeds)["outputs"] == ctl
        entry = {"reg": f"0x{reg:04x}", "old": old, "new": newv, "error": res["error"]}
        if res["outputs"] is not None:
            entry["lanes"] = lane_effect(ctl[0], res["outputs"][0])
        entry["health_after"] = health
        results["scale_edit"].append(entry)
        save()

    for name, src, dst in TRANSPLANTS:
        a, b = load(src), load(dst)
        feeds = inputs_for(a)
        native_a = control_and_health(a, feeds, name)
        native_b = run(b, feeds)["outputs"]
        new, moved = transplant(get_mcode(a), get_mcode(b))
        res = run(with_mcode(a, new), feeds)
        health = run(a, feeds)["outputs"] == native_a
        results["transplant"].append(
            {
                "name": name,
                "moved_records": {str(k): v for k, v in moved.items()},
                "native_a_vs_native_b": _compare(native_a, native_b),
                **_compare(native_b, res["outputs"]),
                "error": res["error"],
                "health_after": health,
            }
        )
        save()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "device":
        device(sys.argv[2])
    else:
        print(__doc__)
        sys.exit(2)
