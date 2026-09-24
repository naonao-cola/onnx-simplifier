"""Recalibrate a compiled AX650 step without rebuilding it, including the
case where a re-encoded MCode segment no longer fits its padded slot.

``short_unit_codec.replace_segment`` rewrites a segment in place and gives up
when the new token stream needs a different 32-byte-padded size. A real
recalibration hits that case: ``toyf_A1`` -> ``toyf_Bx2`` changes 570 scale
records in segment 2, and the recompressed stream grows by 108 bytes, which
is 128 bytes once padded. ``relayout`` handles it. It splices the new stream
in, shifts everything after it, and fixes every field that records a
position or size. The rules below were read off native Pulsar2 builds of the
same graph at two sizes: ``toyf_A1``/``toyf_Bx2`` (0-byte gap before the
tail) and ``relu ... x0_y0``/``x128_y128`` (4-byte gap). ``relayout``
reproduces both larger builds byte for byte from the smaller ones, and the
reverse.

Inside the blob:

* the grown segment's tail table: key 2 (padded size in 8-byte words) and
  key 5 (exact stream length);
* every later segment's tail table: key 3 (word offset from the header end);
* the ``[uint64]`` vector length prefix just before segment 0 (total words);
* header words that are FlatBuffers references across the moved boundary.
  A uoffset whose target moved gets ``+delta``. A negative soffset whose
  vtable moved gets ``-delta``. One of these is the header's pointer to the
  tail table vector;
* soffsets in moved tail tables whose vtable did not move get ``+delta``.

Nothing inside the decompressed register records holds a blob offset, so no
other segment changes. Every segment starts and ends on a 32-byte boundary,
so the 0/4-byte gap before the tail stays the same.

In the ``.axmodel``: the ``*_neu`` initializer's ``dims`` and the length of
its ``value_info`` (``with_mcode``).

``recalibrate`` also covers ``npu_params``, because calibration rewrites its
requantization blocks as well as the MCode scale records. There is still no
scale-to-record map, so both come from a *reference build* at the target
calibration: that is, a Pulsar2 build of the same graph with the target
scales and zero points. ``recalibrate`` copies every differing register
record and ``npu_params`` byte from that build into the step. It refuses
anything that is not a same-shape value edit. See
``docs/axera-step-recalibrate-relayout.md``.

Usage::

    step_recalibrate.py STEP.axmodel REFERENCE.axmodel OUT.axmodel
    step_recalibrate.py device OUT.json   # A1 -> Bx2 on the AX8850 (axcl-vm)
"""

from __future__ import annotations

import json
import os
import struct
import sys

import onnx

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import mcode as mcode_mod  # noqa: E402
import short_unit_codec as codec  # noqa: E402
from short_unit_codec_device_check import (  # noqa: E402
    get_mcode,
    load,
    mcode_name,
)

PAD = 32
ZERO_POINT_REGS = (0x1B10, 0x1A90)
PARAMS = "npu_params"


class RelayoutError(ValueError):
    pass


def _u32(b, o):
    return struct.unpack_from("<I", b, o)[0]


def _i32(b, o):
    return struct.unpack_from("<i", b, o)[0]


def _u16(b, o):
    return struct.unpack_from("<H", b, o)[0]


def _tail_table_positions(mc: bytes) -> list[tuple[int, int]]:
    """``(table pos, vtable pos)`` for each tail table, in table order."""
    vec = mcode_mod.tail_vector(mc)
    res = []
    for k in range(_u32(mc, vec)):
        p = vec + 4 + 4 * k
        tpos = p + _u32(mc, p)
        res.append((tpos, tpos - _i32(mc, tpos)))
    return res


def _put_field(out: bytearray, mc: bytes, table: int, field: int, value: int):
    at, width = codec._table_field_offset(mc, table, field)
    if value >= 1 << (8 * width):
        raise RelayoutError(
            f"table {table} key {field}: {value} overflows {width} bytes"
        )
    out[at : at + width] = value.to_bytes(width, "little")


def _is_vtable(b: bytes, vt: int) -> bool:
    if vt % 2 or vt < 0 or vt + 4 > len(b):
        return False
    vsz, tsz = _u16(b, vt), _u16(b, vt + 2)
    if vsz < 4 or vsz % 2 or tsz < 4 or vt + vsz > len(b):
        return False
    return all(_u16(b, vt + 4 + 2 * f) < tsz for f in range((vsz - 4) // 2))


def _is_object(b: bytes, t: int) -> bool:
    """``t`` holds a table, or a string/vector whose length fits the blob."""
    if t % 4 or t + 4 > len(b):
        return False
    return t + 4 + _u32(b, t) <= len(b) or _is_vtable(b, t - _i32(b, t))


def header_refs(mc: bytes, s0: int) -> list[tuple[int, str]]:
    """FlatBuffers references held in the header (before segment 0 at
    ``s0``): ``(offset, "s")`` for each header table's soffset and
    ``(offset, "u")`` for each 4-byte table field that points at an object.

    Tables are found by walking from the root through fields that point at
    header tables. A blind scan of every header word also finds words that
    only look like references: string bytes (four in ``toyf_A1`` whose value,
    read as a uoffset, lands past segment 2) and scalar fields (three in
    ``gather_1024x28224_axis1_n112896``). The walk skips them. Over all 328
    fixture blobs, it finds every reference that a blind scan with target
    validation does, apart from those three scalars."""
    refs, todo, seen = [], [_u32(mc, 0)], set()
    while todo:
        t = todo.pop()
        if t in seen or not 0 <= t < s0 or t % 4 or not _is_vtable(mc, t - _i32(mc, t)):
            continue
        seen.add(t)
        refs.append((t, "s"))
        vt = t - _i32(mc, t)
        vsz, tsz = _u16(mc, vt), _u16(mc, vt + 2)
        offs = sorted({_u16(mc, vt + 4 + 2 * f) for f in range((vsz - 4) // 2)} - {0})
        for off, end in zip(offs, offs[1:] + [tsz]):
            p = t + off
            if end - off < 4 or p % 4 or p + 4 > s0:
                continue
            target = p + _u32(mc, p)
            if _is_object(mc, target):
                refs.append((p, "u"))
                if target < s0:
                    todo.append(target)
    return sorted(refs)


def relayout(mc: bytes, index: int, stream: bytes) -> bytes:
    """Return ``mc`` with segment ``index`` (stream order) replaced by the
    token stream ``stream``, re-laid out if the padded size changes."""
    streams = codec.segment_streams(mc)
    n = len(streams)
    start, _, table, comp = streams[index]
    if not comp:
        raise RelayoutError(f"segment {index} is not compressed")
    old_room = 8 * table[2]
    new_room = -(-len(stream) // PAD) * PAD
    delta = new_room - old_room
    boundary = start + old_room  # everything at or past this moves
    s0 = streams[0][0]
    tail_tables = _tail_table_positions(mc)

    out = bytearray(mc[:start])
    out += stream + bytes(new_room - len(stream))
    out += mc[boundary:]
    if delta == 0:
        _put_field(out, mc, n - 1 - index, 5, len(stream))
        return bytes(out)

    def moved(pos):
        return boundary <= pos <= len(mc)

    # Header references that cross the boundary: fields of the header's
    # FlatBuffers tables, found by walking from the root (``header_refs``).
    for o, kind in header_refs(mc, s0):
        if kind == "u" and moved(o + _u32(mc, o)):
            struct.pack_into("<I", out, o, _u32(mc, o) + delta)
        elif kind == "s" and moved(o - _i32(mc, o)):
            struct.pack_into("<i", out, o, _i32(mc, o) - delta)
    words = _u32(mc, s0 - 4)
    if words * 8 != sum(8 * t[2] for _, _, t, _ in streams):
        raise RelayoutError(f"word at {s0 - 4} is not the segment length prefix")
    struct.pack_into("<I", out, s0 - 4, words + delta // 8)

    # Tail tables whose vtable stayed behind (none in the fixtures, but a
    # FlatBuffers builder may share a vtable with a header table).
    for tpos, vt in tail_tables:
        if moved(tpos) and not moved(vt):
            struct.pack_into("<i", out, tpos + delta, _i32(mc, tpos) + delta)

    # Tail keys. ``_table_field_offset`` walks ``mc``; the fields sit
    # ``delta`` further on in ``out`` since every tail table moved.
    def put(table_index, field, value):
        at, width = codec._table_field_offset(mc, table_index, field)
        if not moved(at):
            raise RelayoutError(f"table {table_index} key {field} did not move")
        if value >= 1 << (8 * width):
            raise RelayoutError(f"table {table_index} key {field} overflows")
        out[at + delta : at + delta + width] = value.to_bytes(width, "little")

    put(n - 1 - index, 2, new_room // 8)
    put(n - 1 - index, 5, len(stream))
    for later in range(index + 1, n):
        t = streams[later][2]
        put(n - 1 - later, 3, t[3] + delta // 8)
    return bytes(out)


def replace_segments(mc: bytes, raws: dict[int, bytes]) -> bytes:
    """Re-encode each ``{segment: decompressed bytes}`` and re-lay out."""
    for i in sorted(raws):
        mc = relayout(mc, i, codec.encode(raws[i]))
    return mc


def check_relayout(mc: bytes, raws: list[bytes]) -> None:
    """The re-laid-out blob must tile, pass ``mcode.check`` and decode to
    exactly ``raws``."""
    got = codec.decode_segments(mc)
    if got != list(raws):
        bad = [i for i, (a, b) in enumerate(zip(got, raws)) if a != b]
        raise RelayoutError(f"segments {bad} do not decode to the intended bytes")
    mcode_mod.check(mc)


def with_mcode(model: onnx.ModelProto, mc: bytes) -> onnx.ModelProto:
    """Copy of ``model`` carrying MCode ``mc``, with every size that records
    the MCode length (initializer dims, value_info) updated."""
    out = onnx.ModelProto()
    out.CopyFrom(model)
    name = mcode_name(out)
    for init in out.graph.initializer:
        if init.name == name:
            init.raw_data = mc
            del init.dims[:]
            init.dims.append(len(mc))
    for vi in list(out.graph.value_info) + list(out.graph.input):
        if vi.name == name:
            dims = vi.type.tensor_type.shape.dim
            if len(dims) != 1:
                raise RelayoutError(f"value_info {name} is not 1-D")
            dims[0].dim_value = len(mc)
    return out


def _init(model: onnx.ModelProto, name: str) -> bytes:
    return next(bytes(i.raw_data) for i in model.graph.initializer if i.name == name)


def _zero_point_kinds(recs: list[dict]) -> list[bool]:
    return [r["value"] != 0 for r in recs if r["reg"] in ZERO_POINT_REGS]


def recalibrate(
    step: onnx.ModelProto, reference: onnx.ModelProto
) -> tuple[onnx.ModelProto, dict]:
    """Move ``step`` to ``reference``'s calibration.

    ``reference`` is a build of the same graph at the target scales and zero
    points. Every register record that differs is copied, the changed
    segments are re-encoded and re-laid out, and ``npu_params`` is copied.
    Raises ``RelayoutError`` for anything that is not a same-shape value
    edit, which includes a zero point that goes between zero and nonzero
    (Pulsar2 then emits a different number of records)."""
    a, b = get_mcode(step), get_mcode(reference)
    ra, rb = codec.decode_segments(a), codec.decode_segments(b)
    if len(ra) != len(rb):
        raise RelayoutError(f"segment count differs: {len(ra)} vs {len(rb)}")
    raws, report = {}, {"records": {}}
    for i, (x, y) in enumerate(zip(ra, rb)):
        if len(x) != len(y):
            raise RelayoutError(
                f"segment {i}: record count differs ({len(x) // 8} vs {len(y) // 8}); "
                "a zero point changed between zero and nonzero, or the graph differs"
            )
        rx, ry = codec.records(x), codec.records(y)
        if [r["index"] for r in rx] != [r["index"] for r in ry] or [
            (r["verb"], r["reg"]) for r in rx
        ] != [(r["verb"], r["reg"]) for r in ry]:
            raise RelayoutError(
                f"segment {i}: records differ in verb/register, not value"
            )
        if _zero_point_kinds(rx) != _zero_point_kinds(ry):
            raise RelayoutError(
                f"segment {i}: a zero point changes between zero and nonzero"
            )
        n = sum(1 for p, q in zip(rx, ry) if p["raw"] != q["raw"])
        if n:
            raws[i] = y
            report["records"][i] = n
    pa, pb = _init(step, PARAMS), _init(reference, PARAMS)
    if len(pa) != len(pb):
        raise RelayoutError(f"{PARAMS} length differs: {len(pa)} vs {len(pb)}")
    report["params_bytes"] = sum(1 for p, q in zip(pa, pb) if p != q)

    new = replace_segments(a, raws)
    check_relayout(new, rb)
    report["mcode_bytes"] = [len(a), len(new)]
    out = with_mcode(step, new)
    for init in out.graph.initializer:
        if init.name == PARAMS:
            init.raw_data = pb
    return out, report


FIX = os.path.join(_HERE, "fixtures", "step_recalib")


def device(out_path: str) -> None:
    """``toyf_A1`` recalibrated to ``toyf_Bx2``, against native ``toyf_Bx2``.

    Order: native A1 (control), native Bx2, the recalibrated model, the
    recalibrated MCode with A1's ``npu_params``, native A1 again (health). Each run takes the device lock for that run
    only. On a fault, collect the card log with ``axcl-smi log`` and read it
    with ``npu_fault_log.py``."""
    from short_unit_codec_device_check import _compare, inputs_for, run

    a = load(os.path.join(FIX, "toyf_A1.axmodel.gz"))
    b = load(os.path.join(FIX, "toyf_Bx2.axmodel.gz"))
    new, report = recalibrate(a, b)
    feeds = inputs_for(a)
    res: dict = {
        "report": {
            **report,
            "records": {str(k): v for k, v in report["records"].items()},
        }
    }

    def save():
        with open(out_path, "w") as f:
            json.dump(res, f, indent=1)

    ctl = run(a, feeds)
    res["control_error"] = ctl["error"]
    save()
    if ctl["outputs"] is None:
        raise RuntimeError(f"control run failed: {ctl['error']}")
    nb = run(b, feeds)
    res["native_a_vs_native_b"] = _compare(ctl["outputs"], nb["outputs"])
    save()
    got = run(new, feeds)
    res["recalibrated_vs_native_b"] = {
        **_compare(nb["outputs"], got["outputs"]),
        "error": got["error"],
    }
    res["recalibrated_vs_native_a"] = _compare(ctl["outputs"], got["outputs"])
    save()
    # The same MCode with A1's npu_params: does the requant block matter?
    mcode_only = onnx.ModelProto()
    mcode_only.CopyFrom(new)
    for init in mcode_only.graph.initializer:
        if init.name == PARAMS:
            init.raw_data = _init(a, PARAMS)
    got2 = run(mcode_only, feeds)
    res["mcode_only_vs_native_b"] = {
        **_compare(nb["outputs"], got2["outputs"]),
        "error": got2["error"],
    }
    save()
    health = run(a, feeds)
    res["health_after"] = health["outputs"] == ctl["outputs"]
    save()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "device":
        device(sys.argv[2])
    elif len(sys.argv) == 4:
        model, rep = recalibrate(load(sys.argv[1]), load(sys.argv[2]))
        onnx.save(model, sys.argv[3])
        print(
            json.dumps(
                {**rep, "records": {str(k): v for k, v in rep["records"].items()}}
            )
        )
    else:
        print(__doc__)
        sys.exit(2)
