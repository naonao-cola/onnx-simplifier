"""Record-level emitter for non-fused AX650 ``Reshape -> Relu`` weight folds.

Earlier work (``docs/axera-reshape.md``, ``docs/axera-reshape-dma.md``,
``docs/axera-reshape-calibration-isolation.md``) diffed the *compressed* MCode
and found no closed form: some shapes seemed to change layout for no reason.
Decompressed with ``short_unit_codec`` the segments are 8-byte register
records, and for a family of builds that share one record structure (the same
``(verb, register)`` sequence) every record value is an exact polynomial of
degree <= 2 in the varying channel count. The old "layout changes" were the
compressor choosing different matches, not different programs
(``docs/axera-reshape-decompressed.md``).

This module fits those polynomials from committed compiler-built fixtures,
predicts the decompressed segments at a new size, re-encodes them with the
native-compatible encoder, rewrites the tensor-size words in the tail tables,
and relabels the graph dims. Nothing is hard-coded per register: a fit that
does not reproduce every fixture exactly (with at least one fixture to spare
beyond the polynomial's degree) is an error.

It refuses sizes outside the measured record-structure group of each family.
The calibration scale ``s`` lives in 16 scale-lane records (registers
0x0f50..0x0fc0: eight lanes of ``1/s``, then eight of ``s``). Each family's
fixtures share one calibration, so those records fit as constants and the
emitted program carries the fixtures' calibration unless ``scale=`` gives
another ``s``.
"""

from __future__ import annotations

import gzip
import json
import os
import struct
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction

import onnx

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import binary_op_scale_emit as bose  # noqa: E402
import short_unit_codec as suc  # noqa: E402

_FIXTURES = os.path.join(_HERE, "fixtures")
RECORD = suc.RECORD
MAX_DEGREE = 2
SCALE_REGS = frozenset((0x0F50 + 0x10 * i).to_bytes(2, "little") for i in range(8))


@dataclass(frozen=True)
class Family:
    """One record-structure group of a ``Reshape -> Relu`` family."""

    in_shape: Callable[[int], list[int]]
    out_shape: Callable[[int], list[int]]
    # (param, fixture path relative to scripts/axera/fixtures)
    fixtures: tuple[tuple[int, str], ...]
    # Params whose compiler-built program has this group's record structure.
    measured: frozenset[int]


FAMILIES: dict[str, Family] = {
    # [C,C,3,3] -> [1,C,C,9]: the ResNet18 step's [64,64,3,3] weight fold is
    # C=64. Fixtures share one fixed calibration (reshape_calib_isolation).
    "square_weight_fold": Family(
        in_shape=lambda c: [c, c, 3, 3],
        out_shape=lambda c: [1, c, c, 9],
        fixtures=tuple(
            (c, f"reshape_calib_isolation/fixed_C{c}.axmodel.gz")
            for c in (40, 44, 56, 60, 64)
        ),
        measured=frozenset({40, 44, 48, 52, 56, 60, 64}),
    ),
    # [Co,8,3,3] -> [1,Co,8,9]. Co=36 and 60 are in the group: their
    # "layout change" in docs/axera-reshape-dma.md was compression only.
    "cin8_weight_fold": Family(
        in_shape=lambda c: [c, 8, 3, 3],
        out_shape=lambda c: [1, c, 8, 9],
        fixtures=(
            (34, "reshape_dma/w1_co34_oracle.axmodel.gz"),
            (40, "reshape_dma/w1_co40.axmodel.gz"),
            (42, "reshape_dma/w1_co42_oracle.axmodel.gz"),
            (58, "reshape_dma/w1_co58_oracle.axmodel.gz"),
            (62, "reshape_dma/w1_co62_oracle.axmodel.gz"),
        ),
        measured=frozenset(
            {16, 20, 24, 28, 32, 34, 36, 38, 64, 66, 68, 70, 72, 80}
            | set(range(40, 63))
        ),
    ),
}


def load_axmodel(path: str) -> onnx.ModelProto:
    with open(path, "rb") as f:
        data = f.read()
    if path.endswith(".gz"):
        data = gzip.decompress(data)
    return onnx.load_model_from_string(data)


def _neu(model: onnx.ModelProto):
    matches = [i for i in model.graph.initializer if i.name.endswith("_neu")]
    if len(matches) != 1:
        raise ValueError(f"expected one *_neu initializer, found {len(matches)}")
    return matches[0]


def mcode_of(model: onnx.ModelProto) -> bytes:
    return bytes(_neu(model).raw_data)


def fit_poly(xs: Sequence[int], ys: Sequence[int]) -> list[Fraction] | None:
    """Lowest-degree exact polynomial (coefficients low to high) through all
    points, with at least one point beyond the degree; ``None`` if none."""
    for deg in range(MAX_DEGREE + 1):
        if len(xs) < deg + 2 and deg > 0:
            return None
        n = deg + 1
        m = [
            [Fraction(x) ** k for k in range(n)] + [Fraction(y)]
            for x, y in zip(xs[:n], ys[:n])
        ]
        singular = False
        for c in range(n):
            piv = next((r for r in range(c, n) if m[r][c] != 0), None)
            if piv is None:
                singular = True
                break
            m[c], m[piv] = m[piv], m[c]
            for r in range(n):
                if r != c and m[r][c] != 0:
                    f = m[r][c] / m[c][c]
                    m[r] = [a - f * b for a, b in zip(m[r], m[c])]
        if singular:
            continue
        coef = [m[k][n] / m[k][k] for k in range(n)]
        if all(evaluate(coef, x) == y for x, y in zip(xs, ys)):
            return coef
    return None


def evaluate(coef: Sequence[Fraction], x: int) -> Fraction:
    return sum((c * Fraction(x) ** k for k, c in enumerate(coef)), Fraction(0))


def _is_rotation(seg: int, slot: bytes) -> bool:
    """Segment-0 sync records whose value is ``(n << 20) | low`` with n
    rotating between rebuilds of the same graph: rebuild noise."""
    return (
        seg == 0
        and slot[0] == 0xA2
        and 1 <= (int.from_bytes(slot[4:], "little") >> 20) <= 8
    )


def _layout(mc: bytes) -> list[tuple[int, int, bool]]:
    return [(s, t.get(2, size), comp) for s, size, t, comp in suc.segment_streams(mc)]


def _rooms(mc: bytes) -> list[tuple[int, int]]:
    return [
        (s, 8 * t[2] if comp else size) for s, size, t, comp in suc.segment_streams(mc)
    ]


def _key5_words(mc: bytes) -> set[int]:
    n = len(suc.segment_streams(mc))
    words = set()
    for i, (_, _, _, comp) in enumerate(suc.segment_streams(mc)):
        if comp:
            at, _ = suc._table_field_offset(mc, n - 1 - i, 5)
            words.add(at - at % 4)
    return words


@dataclass
class Rules:
    """Fitted per-record and per-tail-word polynomials of one family."""

    params: list[int]
    template: dict[int, bytes]  # param -> native MCode
    # (segment, slot) -> coefficients, only for slots whose value varies
    records: dict[tuple[int, int], list[Fraction]]
    # aligned tail offset -> coefficients (uint32 words outside every segment)
    tail: dict[int, list[Fraction]]


def fit_rules(fixtures: Sequence[tuple[int, bytes]]) -> Rules:
    """Fit a record-level rule from native MCode of one record-structure group.

    Raises ``ValueError`` when the builds do not share one layout and record
    structure, or when a varying record or tail word is not an exact
    polynomial."""
    params = [p for p, _ in fixtures]
    mcs = [mc for _, mc in fixtures]
    if len({len(mc) for mc in mcs}) != 1 or len({repr(_layout(mc)) for mc in mcs}) != 1:
        raise ValueError("fixtures do not share one blob layout")
    raws = [suc.decode_segments(mc) for mc in mcs]
    comp = [c for _, _, c in _layout(mcs[0])]
    records: dict[tuple[int, int], list[Fraction]] = {}
    for s in range(len(raws[0])):
        if len({len(r[s]) for r in raws}) != 1:
            raise ValueError(f"segment {s}: decompressed lengths differ")
        for k in range(0, len(raws[0][s]) // RECORD):
            slots = [r[s][k * RECORD : (k + 1) * RECORD] for r in raws]
            if len({sl[:4] for sl in slots}) != 1:
                raise ValueError(f"segment {s} slot {k}: record structure differs")
            values = [int.from_bytes(sl[4:], "little") for sl in slots]
            if len(set(values)) == 1 or any(_is_rotation(s, sl) for sl in slots):
                continue
            if not comp[s]:
                raise ValueError(f"segment {s} is stored raw but varies")
            coef = fit_poly(params, values)
            if coef is None:
                raise ValueError(
                    f"segment {s} slot {k}: not a polynomial {dict(zip(params, values))}"
                )
            records[(s, k)] = coef
    covered = bytearray(len(mcs[0]))
    for start, room in _rooms(mcs[0]):
        covered[start : start + room] = b"\1" * room
    skip = _key5_words(mcs[0])
    tail: dict[int, list[Fraction]] = {}
    for at in range(0, len(mcs[0]) - 3, 4):
        if at in skip or any(covered[at : at + 4]):
            continue
        values = [int.from_bytes(mc[at : at + 4], "little") for mc in mcs]
        if len(set(values)) == 1:
            continue
        coef = fit_poly(params, values)
        if coef is None:
            raise ValueError(
                f"tail word {at}: not a polynomial {dict(zip(params, values))}"
            )
        tail[at] = coef
    return Rules(params, dict(zip(params, mcs)), records, tail)


def _value(coef: Sequence[Fraction], x: int, what: str) -> int:
    v = evaluate(coef, x)
    if v.denominator != 1 or not 0 <= v < 1 << 32:
        raise ValueError(f"{what}: predicted {v} is not a uint32 at {x}")
    return int(v)


def _f32(x: float) -> bytes:
    return struct.pack("<f", x)


def scale_lanes(raw: bytes) -> list[int]:
    """Slots of the 16 scale-lane records (registers 0x0f50..0x0fc0) of one
    decompressed segment: eight lanes of ``1/s`` followed by eight of ``s``.
    Empty when the segment has none."""
    slots = [
        k
        for k in range(len(raw) // RECORD)
        if raw[k * RECORD] == 0xA1
        and raw[k * RECORD + 2 : k * RECORD + 4] in SCALE_REGS
    ]
    if not slots:
        return []
    vals = [raw[k * RECORD + 4 : (k + 1) * RECORD] for k in slots]
    if len(slots) != 16 or len(set(vals[:8])) != 1 or len(set(vals[8:])) != 1:
        raise ValueError(f"unexpected scale-lane records at slots {slots}")
    s = struct.unpack("<f", vals[8])[0]
    if vals[0] != _f32(1.0 / s):
        raise ValueError("scale lanes are not (1/s, s)")
    return slots


def predict_mcode(rules: Rules, param: int, scale: float | None = None) -> bytes:
    """MCode at ``param`` from the fixture nearest to it.

    ``scale`` overrides the calibration scale ``s`` in the scale lanes (the
    fixtures' own calibration is kept otherwise)."""
    base = min(rules.params, key=lambda p: (abs(p - param), p))
    mc = rules.template[base]
    raws = suc.decode_segments(mc)
    out = mc
    for s, raw in enumerate(raws):
        new = bytearray(raw)
        for (seg, k), coef in rules.records.items():
            if seg == s:
                v = _value(coef, param, f"segment {s} slot {k}")
                new[k * RECORD + 4 : (k + 1) * RECORD] = v.to_bytes(4, "little")
        if scale is not None:
            s32 = struct.unpack("<f", _f32(scale))[0]
            for i, k in enumerate(scale_lanes(raw)):
                new[k * RECORD + 4 : (k + 1) * RECORD] = (
                    _f32(1.0 / s32) if i < 8 else _f32(s32)
                )
        if bytes(new) != raw:
            out = suc.replace_segment(out, s, bytes(new))
    buf = bytearray(out)
    for at, coef in rules.tail.items():
        buf[at : at + 4] = _value(coef, param, f"tail word {at}").to_bytes(4, "little")
    return bytes(buf)


ZP_REGS = frozenset(r.to_bytes(2, "little") for r in (0x1B10, 0x1EB0, 0x1A90))
"""Registers holding the uint8 zero point (MinMax: ``round(-lo / s)``); a
non-fused Reshape -> Relu writes each once with the calibrated value."""
GATHER_ZP_REGS = frozenset(r.to_bytes(2, "little") for r in (0x1B10, 0x1A90))
"""A standalone last-axis Gather (``memory_emit``) writes its one shared zero
point to these two only."""
IDENTITY_ZP_REGS = GATHER_ZP_REGS
"""A ``Reshape -> Identity`` build writes its zero point to 0x1b10/0x1a90 only
(the Relu form's 0x1eb0 write is the Relu's own)."""


def _lane_values(raw: bytes) -> dict[int, bytes]:
    """Scale-lane slots (registers 0x0f50..0x0fc0, verb 0xa1) -> value."""
    return {
        k: raw[k * RECORD + 4 : (k + 1) * RECORD]
        for k in range(len(raw) // RECORD)
        if raw[k * RECORD] == 0xA1
        and raw[k * RECORD + 2 : k * RECORD + 4] in SCALE_REGS
    }


def retarget_scale(
    mc: bytes,
    scale: float,
    zero_point: int | None = None,
    zp_regs: frozenset = ZP_REGS,
) -> bytes:
    """``mc`` with its calibration moved to ``(scale, zero_point)``.

    Every scale-lane record holds ``1/s`` or ``s`` of the one calibration: an
    untiled program writes one group of eight each; a tiled program repeats
    the groups per tile. The first lane is ``1/s``. The zero-point records
    are the nonzero writes to 0x1b10/0x1eb0/0x1a90 (the registers are also
    written 0 around the Relu's own records); a tiled program repeats them.

    A zero point of 0 compiles to a different program (the zero-point write
    is elided and the stream changes), so moving to 0 is refused. Raises
    ``ValueError`` when the lanes or zero points are not of that shape.
    ``zp_regs`` is the set of registers the program writes its zero point to
    (a last-axis Gather writes only 0x1b10/0x1a90: ``GATHER_ZP_REGS``)."""
    s32 = struct.unpack("<f", _f32(scale))[0]
    out, seen = mc, False
    for s, raw in enumerate(suc.decode_segments(mc)):
        lanes = _lane_values(raw)
        if not lanes:
            continue
        seen = True
        inv = lanes[min(lanes)]
        # the other lane value is s; it need not be exactly f32(1 / inv)
        # (a Gather template stores 1/s = 141.6667 next to s = 0.00705882)
        if len(set(lanes.values()) - {inv}) > 1 or len(lanes) % 8:
            raise ValueError(f"segment {s}: scale lanes are not one (1/s, s) pair")
        new = bytearray(raw)
        for k, v in lanes.items():
            new[k * RECORD + 4 : (k + 1) * RECORD] = (
                _f32(1.0 / s32) if v == inv else _f32(s32)
            )
        if zero_point is not None:
            zps = [
                k
                for k in range(len(raw) // RECORD)
                if raw[k * RECORD] == 0xA1
                and raw[k * RECORD + 2 : k * RECORD + 4] in zp_regs
                and raw[k * RECORD + 4 : (k + 1) * RECORD] != bytes(4)
            ]
            regs = {raw[k * RECORD + 2 : k * RECORD + 4] for k in zps}
            old = {
                int.from_bytes(raw[k * RECORD + 4 : (k + 1) * RECORD], "little")
                for k in zps
            }
            if regs != zp_regs or len(old) != 1:
                raise ValueError(f"segment {s}: zero-point records {zps} {sorted(old)}")
            if not 0 < zero_point < 256:
                raise ValueError(
                    f"zero point {zero_point} vs template {sorted(old)}: only nonzero "
                    "uint8 zero points share the template's program"
                )
            for k in zps:
                new[k * RECORD + 4 : (k + 1) * RECORD] = zero_point.to_bytes(
                    4, "little"
                )
        if bytes(new) != raw:
            # the re-encoded stream can pad to another multiple of 32 bytes
            # (seen on a 1024x28224 Identity build): relayout the blob as
            # Pulsar2 does rather than refuse
            out = bose.relayout_segment(out, s, bytes(new))
    if not seen:
        raise ValueError("no scale-lane records to retarget")
    return out


def family_rules(name: str, exclude: Sequence[int] = ()) -> Rules:
    fam = FAMILIES[name]
    fixtures = [
        (p, mcode_of(load_axmodel(os.path.join(_FIXTURES, rel))))
        for p, rel in fam.fixtures
        if p not in exclude
    ]
    return fit_rules(fixtures)


def _set_dims(value_info, shape: Sequence[int]) -> None:
    del value_info.type.tensor_type.shape.dim[:]
    for size in shape:
        value_info.type.tensor_type.shape.dim.add().dim_value = size


def emit_axmodel(
    name: str, param: int, output_path: str | None = None, *, scale: float | None = None
) -> onnx.ModelProto:
    """Compiled ``Reshape -> Relu`` model of family ``name`` at ``param``.

    ``scale`` is the calibration scale ``s`` to write into the scale lanes;
    by default the fixtures' calibration is kept."""
    if name not in FAMILIES:
        raise ValueError(f"unknown family {name!r}; known: {sorted(FAMILIES)}")
    fam = FAMILIES[name]
    if not isinstance(param, int) or isinstance(param, bool):
        raise ValueError("param must be an integer")
    if param not in fam.measured:
        raise ValueError(
            f"{name}: {param} is outside the measured record-structure group "
            f"{sorted(fam.measured)}"
        )
    rules = family_rules(name)
    base = min(rules.params, key=lambda p: (abs(p - param), p))
    model = load_axmodel(os.path.join(_FIXTURES, dict(fam.fixtures)[base]))
    if len(model.graph.node) != 1 or model.graph.node[0].op_type != "neu mode":
        raise ValueError("template must contain exactly one 'neu mode' node")
    _neu(model).raw_data = predict_mcode(rules, param, scale)
    in_name, out_name = model.graph.input[0].name, model.graph.output[0].name
    in_shape, out_shape = fam.in_shape(param), fam.out_shape(param)
    _set_dims(model.graph.input[0], in_shape)
    _set_dims(model.graph.output[0], out_shape)
    for vi in model.graph.value_info:
        if vi.name == in_name:
            _set_dims(vi, in_shape)
        elif vi.name == out_name:
            _set_dims(vi, out_shape)
    for attr in model.graph.node[0].attribute:
        if attr.name == "inputs_info":
            attr.s = json.dumps({in_name: ["FP32", in_shape]}).encode()
        elif attr.name == "outputs_info":
            attr.s = json.dumps({out_name: ["FP32", out_shape]}).encode()
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        onnx.save(model, output_path)
    return model


STEP_TEMPLATE_DIR = os.path.join(_FIXTURES, "reshape_step_templates")
"""One ``Reshape -> Relu`` build per distinct non-fused ResNet18 step Reshape
(``in -> out`` shape pair), at a symmetric calibration range, plus a second
build of the same shape at another range. ``manifest.json`` lists the pairs
whose records differ only in the scale lanes and whose scale lanes retarget
onto each other exactly (``docs/axera-reshape-step-templates.md``)."""


def _shape_key(in_shape: Sequence[int], out_shape: Sequence[int]) -> str:
    return "x".join(map(str, in_shape)) + "->" + "x".join(map(str, out_shape))


def step_manifest() -> dict:
    path = os.path.join(STEP_TEMPLATE_DIR, "manifest.json")
    if not os.path.exists(path):
        return {"templates": {}}
    with open(path) as f:
        return json.load(f)


def step_template(
    in_shape: Sequence[int], out_shape: Sequence[int], manifest: dict | None = None
) -> dict:
    """Manifest entry for the step Reshape ``in_shape -> out_shape``, with its
    ``axmodel`` path made absolute. Raises ``ValueError`` if no validated
    template serves that shape pair."""
    m = manifest if manifest is not None else step_manifest()
    entry = m["templates"].get(_shape_key(in_shape, out_shape))
    if entry is None:
        raise ValueError(
            f"no validated step Reshape template for {_shape_key(in_shape, out_shape)}"
        )
    return dict(entry, axmodel=os.path.join(STEP_TEMPLATE_DIR, entry["axmodel"]))


def step_template_zp0(
    in_shape: Sequence[int], out_shape: Sequence[int], manifest: dict | None = None
) -> dict:
    """Like ``step_template``, for a zero point of 0 (a nonnegative input:
    the zero-point writes are elided, so it is its own program). Only the
    scale moves; ``docs/axera-step-real-calibration.md``."""
    m = manifest if manifest is not None else step_manifest()
    key = _shape_key(in_shape, out_shape)
    entry = m.get("zero_point_0_templates", {}).get(key)
    if entry is None:
        raise ValueError(f"no validated zero-point-0 step Reshape template for {key}")
    return dict(entry, axmodel=os.path.join(STEP_TEMPLATE_DIR, entry["axmodel"]))


def step_template_identity(
    in_shape: Sequence[int], out_shape: Sequence[int], manifest: dict | None = None
) -> dict:
    """Like ``step_template``, for the ``Reshape -> Identity`` build of the
    shape pair: a pure requantizing copy with no Relu, so it keeps a signed
    input (``docs/axera-reshape-signed-templates.md``). Raises ``ValueError``
    if none was built for that shape pair."""
    m = manifest if manifest is not None else step_manifest()
    key = _shape_key(in_shape, out_shape)
    entry = m.get("identity_templates", {}).get(key)
    if entry is None:
        raise ValueError(f"no validated Reshape -> Identity step template for {key}")
    return dict(entry, axmodel=os.path.join(STEP_TEMPLATE_DIR, entry["axmodel"]))


def emit_step_reshape(
    in_shape: Sequence[int],
    out_shape: Sequence[int],
    scale: float,
    zero_point: int,
    output_path: str | None = None,
) -> onnx.ModelProto:
    """The step template for ``in_shape -> out_shape`` retargeted to the
    calibration ``(scale, zero_point)``.

    A nonzero zero point means the input range goes negative, so it takes the
    ``Reshape -> Identity`` template (``step_template_identity``): the
    ``Reshape -> Relu`` templates of ``step_template`` clip the negative half
    and are never emitted for it. A zero point of 0 (a nonnegative input, where
    the Relu is the identity) takes its own template (``step_template_zp0``; a
    different program, see ``retarget_scale``)."""
    import step_recalibrate

    if zero_point == 0:
        entry = step_template_zp0(in_shape, out_shape)
        model = load_axmodel(entry["axmodel"])
        mc = retarget_scale(mcode_of(model), scale)
    else:
        entry = step_template_identity(in_shape, out_shape)
        model = load_axmodel(entry["axmodel"])
        mc = retarget_scale(mcode_of(model), scale, zero_point, IDENTITY_ZP_REGS)
    # re-encoding can change the blob length; the runtime reads it from dims
    model = step_recalibrate.with_mcode(model, mc)
    if output_path:
        onnx.save(model, output_path)
    return model


def compare_mcode(got: bytes, want: bytes) -> dict:
    """Record-level comparison, ignoring segment-0 rotation noise."""
    a, b = suc.decode_segments(got), suc.decode_segments(want)
    diff = []
    for s, (x, y) in enumerate(zip(a, b)):
        for k in range(min(len(x), len(y)) // RECORD):
            sx, sy = x[k * RECORD : (k + 1) * RECORD], y[k * RECORD : (k + 1) * RECORD]
            if sx != sy and not (_is_rotation(s, sx) and _is_rotation(s, sy)):
                diff.append((s, k, sy[2] | sy[3] << 8))
    rooms = _rooms(want)
    outside = [
        i
        for i in range(min(len(got), len(want)))
        if got[i] != want[i] and not any(st <= i < st + r for st, r in rooms)
    ]
    return {
        "same_length": len(got) == len(want),
        "segments": [len(x) == len(y) for x, y in zip(a, b)] + [len(a) == len(b)],
        "record_diffs": diff,
        "tail_byte_diffs": outside,
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("family", choices=sorted(FAMILIES))
    ap.add_argument("param", type=int)
    ap.add_argument("output")
    ap.add_argument(
        "--scale", type=float, help="calibration scale s for the scale lanes"
    )
    ap.add_argument("--oracle", help="native compiled.axmodel to compare against")
    args = ap.parse_args()
    m = emit_axmodel(args.family, args.param, args.output, scale=args.scale)
    if args.oracle:
        print(
            json.dumps(compare_mcode(mcode_of(m), mcode_of(load_axmodel(args.oracle))))
        )
