"""Regression coverage for ``scripts/axera/short_unit_codec.py``, the codec for
AX650 MCode's segment compression, whose tokens are the "short units". No
Docker or device is required: it decodes committed Pulsar2 builds and checks
three things. The output must be 8-byte register records, the calibration's
own scales and zero points must appear there as whole values, and
re-encoding must round-trip, within the native encoder's envelope. See
``docs/axera-short-unit-encoding.md``."""

import json
import os
import struct
import sys

import pytest

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import short_unit_codec as codec  # noqa: E402
from teng_register_census import load_axmodel, mcode_of  # noqa: E402

_FIX = os.path.join(_AXERA_DIR, "fixtures")
_CENSUS = os.path.join(_FIX, "teng_register_census")
_BINARY = os.path.join(_FIX, "binary_op_registers")
_RELU = os.path.join(_FIX, "elementwise_scale_emit")
_CENSUS_SCALES = json.load(open(os.path.join(_CENSUS, "scales.json")))
_BINARY_SCALES = json.load(open(os.path.join(_BINARY, "scales.json")))
_LANES = tuple(range(0x0F50, 0x0FD0, 0x10))  # the eight per-lane float registers
TENG = 2  # TENG (EU9) program segment, in stream order


def _mcode(path):
    return mcode_of(load_axmodel(path))


def _f32(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _lane_values(records, value):
    """Lane registers that hold ``value`` as float32, within 1 ulp."""
    want = _f32(value)
    return {
        r["reg"] for r in records if r["reg"] in _LANES and abs(r["value"] - want) <= 1
    }


# -- token grammar on hand-built streams ---------------------------------------


def test_literal_run_and_copy():
    # "abcd" as a literal run, then copy 4 bytes from offset 4.
    assert codec.decode(bytes([0x03]) + b"abcd" + bytes([0x81, 0x01])) == b"abcdabcd"


def test_offset_low_bits_come_from_the_token():
    # t = 0x80 | (off & 3) << 5 | (len - 3); b = off >> 2. Offset 5, length 3.
    lit = bytes([0x04]) + b"vwxyz"
    assert codec.decode(lit + bytes([0x80 | 1 << 5 | 0, 0x01])) == b"vwxyzvwx"


def test_length_extension_byte():
    lit = bytes([0x00]) + b"z"
    # len = 0x1f + 3 + 5 = 39, offset 1 (an overlapping run-length copy).
    stream = lit + bytes([0x80 | 1 << 5 | 0x1F, 0x00, 0x05])
    assert codec.decode(stream) == b"z" * 40


def test_offset_beyond_history_raises():
    with pytest.raises(codec.CodecError):
        codec.decode(bytes([0x00]) + b"z" + bytes([0x81, 0x01]))


def test_census_short_unit_is_two_copies_around_a_literal():
    # docs/axera-teng-register-census.md's zero-point reflow unit
    # ``84 22 01 10 1b 83 4a`` is not a register write with an encoded value.
    # It is a copy, then the literal register address 0x1b10, then another copy.
    toks = codec.tokens(bytes.fromhex("8422 01101b 834a"))
    assert toks == [
        ("M", 0, 7, 0x22 << 2, 2),
        ("L", 2, b"\x10\x1b"),
        ("M", 5, 6, 0x4A << 2, 2),
    ]


# -- whole-segment decoding of native builds -----------------------------------


@pytest.mark.parametrize("name", sorted(n for n, s in _CENSUS_SCALES.items() if s))
def test_census_segments_decode_to_records(name):
    mc = _mcode(os.path.join(_CENSUS, f"{name}.axmodel.gz"))
    for (start, size, table, comp), data in zip(
        codec.segment_streams(mc), codec.decode_segments(mc)
    ):
        assert data[:3] == b"\xa7\x00\x00"  # every segment opens with its header
        assert len(data) % 8 == 0
        if comp:
            assert size == table[5]
            # Stream padded to a multiple of 32 bytes, zero-filled.
            assert 8 * table[2] == -(-size // 32) * 32
        for r in codec.records(data):
            assert r["verb"] >= 0xA0 and r["raw"][1] == 0


@pytest.mark.parametrize("name", ["div_s1", "div_r3"])
def test_div_scales_are_whole_float_records(name):
    s = {t: v[0] for t, v in _CENSUS_SCALES[name].items()}
    recs = codec.register_values(_mcode(os.path.join(_CENSUS, f"{name}.axmodel.gz")))
    teng = recs[TENG]
    # 1/s_x, 1/s_z, s_y and s_x/(s_y*s_z), each on all eight lanes -- s_y was
    # a "short unit" in the census.
    for value in (1 / s["x"], 1 / s["z"], s["y"], s["x"] / (s["y"] * s["z"])):
        assert _lane_values(teng, value) == set(_LANES), value


@pytest.mark.parametrize("name", ["div_s1", "div_r3", "div_asym"])
def test_div_zero_points_are_whole_records(name):
    zp = {t: v[1] for t, v in _CENSUS_SCALES[name].items()}
    teng = codec.register_values(_mcode(os.path.join(_CENSUS, f"{name}.axmodel.gz")))[
        TENG
    ]
    written = [r["value"] for r in teng if r["reg"] == 0x1B10]
    assert zp["x"] in written and zp["y"] in written


def test_relu_zero_point_register():
    def zp_writes(tag):
        path = os.path.join(_RELU, f"relu_16x256x14x14_{tag}.axmodel.gz")
        teng = codec.register_values(_mcode(path))[TENG]
        return [r["value"] for r in teng if r["reg"] == 0x1B10]

    # The first write of 0x1b10 carries zp_x (a later pass writes it again).
    assert zp_writes("x0_y0")[0] == 0
    assert zp_writes("x128_y128")[0] == 128


def test_binary_op_scale_ratio_field_is_a_whole_float():
    # docs/axera-elementwise-scale-emit.md refused Add because f32(1/z) did not
    # appear once x and z had different scales. After decompression it is an
    # ordinary eight-lane float, and add_c2 and add_c3 (same structure)
    # differ in their TENG program only through 1/s_x, 1/s_z, s_y and a few
    # small fields.
    for name in ("add_c2", "add_c3"):
        s = {t: v["scale"] for t, v in _BINARY_SCALES[name].items()}
        teng = codec.register_values(
            _mcode(os.path.join(_BINARY, f"{name}.axmodel.gz"))
        )[TENG]
        for value in (1 / s["x"], 1 / s["z"], s["y"]):
            assert _lane_values(teng, value) == set(_LANES), (name, value)
    a, b = (
        codec.decode_segments(_mcode(os.path.join(_BINARY, f"{n}.axmodel.gz")))[TENG]
        for n in ("add_c2", "add_c3")
    )
    assert len(a) == len(b)
    changed = {
        a[r + 2] | a[r + 3] << 8
        for r in range(0, len(a), 8)
        if a[r : r + 8] != b[r : r + 8]
    }
    assert set(_LANES) <= changed
    assert len(changed - set(_LANES)) <= 3  # 0x03d0 / 0x02b0 lane counts


# -- encoding ------------------------------------------------------------------


def _census_streams():
    for name in sorted(n for n, s in _CENSUS_SCALES.items() if s):
        mc = _mcode(os.path.join(_CENSUS, f"{name}.axmodel.gz"))
        for i, (start, size, table, comp) in enumerate(codec.segment_streams(mc)):
            if comp:
                yield name, i, mc, mc[start : start + size]


def test_encode_round_trips_inside_the_native_envelope():
    identical = total = 0
    for name, _, _, native in _census_streams():
        raw = codec.decode(native)
        stream = codec.encode(raw)
        assert codec.decode(stream) == raw, name
        assert len(stream) == len(native), name
        toks = codec.tokens(stream)
        for t in toks:
            if t[0] == "M":
                _, _, length, off, _ = t
                assert off >= codec.NATIVE_MIN_OFFSET and length <= off
                assert length >= codec.NATIVE_MIN_MATCH
        assert toks[-1][0] == "L" and len(toks[-1][2]) >= codec.NATIVE_TAIL_LITERALS
        total += 1
        identical += stream == native
    # Byte-identical to Pulsar2's own output for 12 of 18 segments. The other
    # six have the same length and differ only in which of several
    # equal-length earlier copies the last back-reference points at.
    assert (total, identical) == (18, 12)


def test_native_streams_stay_in_the_envelope():
    for name, _, _, native in _census_streams():
        toks = codec.tokens(native)
        for t in toks:
            if t[0] == "M":
                assert t[3] >= codec.NATIVE_MIN_OFFSET and t[2] <= t[3], name
                assert t[2] >= codec.NATIVE_MIN_MATCH, name
        assert toks[-1][0] == "L" and len(toks[-1][2]) >= codec.NATIVE_TAIL_LITERALS


def test_replace_segment_patches_a_value_and_key5():
    mc = _mcode(os.path.join(_CENSUS, "div_s1.axmodel.gz"))
    raw = bytearray(codec.decode_segments(mc)[TENG])
    # Retarget zp_x (0x1b10's first write, 128) to 100: a value that also
    # occurs elsewhere, the case that previously broke in-place patching.
    at = next(r for r in range(0, len(raw), 8) if raw[r + 2 : r + 4] == b"\x10\x1b")
    assert raw[at + 4] == 128
    raw[at + 4] = 100
    new = codec.replace_segment(mc, TENG, bytes(raw))
    assert len(new) == len(mc)
    start, size, table, _ = codec.segment_streams(new)[TENG]
    assert codec.decode_segments(new)[TENG] == bytes(raw)
    assert size == table[5] and size != codec.segment_streams(mc)[TENG][1]
    # Nothing outside this segment's slot and its table changes.
    old_start = codec.segment_streams(mc)[TENG][0]
    assert new[:old_start] == mc[:old_start]
    assert (
        new[old_start + 8 * table[2] : old_start + 8 * table[2] + 64]
        == (mc[old_start + 8 * table[2] : old_start + 8 * table[2] + 64])
    )


def test_replace_segment_is_identity_when_the_encoder_matches():
    mc = _mcode(os.path.join(_CENSUS, "div_s1.axmodel.gz"))
    for i, (_, _, _, comp) in enumerate(codec.segment_streams(mc)):
        if comp:
            assert codec.replace_segment(mc, i, codec.decode_segments(mc)[i]) == mc
