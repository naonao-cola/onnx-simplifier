"""`mcode.check` decompresses every compressed segment
(docs/axera-mcode-check-decompress.md).

MCode segments with tail-table key 5 are LZ77 token streams
(docs/axera-short-unit-encoding.md). The older checks read those tokens as-is,
so a patch that overwrote a token byte still passed them
(docs/axera-mcode-segments-fix.md). `mcode.codec_violations`, called from
`mcode.check`, now decodes each stream and checks the framing measured on
every native fixture.
"""

import gzip
import os
import sys

import onnx
import pytest

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import mcode  # noqa: E402
import short_unit_codec as codec  # noqa: E402

_FIXTURES = os.path.join(_AXERA_DIR, "fixtures")


def _neu(rel):
    with gzip.open(os.path.join(_FIXTURES, rel), "rb") as f:
        data = f.read()
    if rel.endswith(".mcode.gz"):
        return data
    model = onnx.load_model_from_string(data)
    return next(
        bytes(i.raw_data) for i in model.graph.initializer if i.name.endswith("_neu")
    )


_NATIVE = [
    "dma_tiles/relu_1x64x56x56.axmodel.gz",
    "teng_register_census/div_s1.axmodel.gz",
    "step_recalib/toyf_A1.axmodel.gz",
    "conv_learn_256to512/c1x1_reference.axmodel.gz",
    "conv_learn_256to512/c1x1_holdout_native.axmodel.gz",
    "adam_update_fp32.mcode.gz",
]

_CORRUPT = [
    "conv_learn_256to512/c1x1_holdout_emitted.axmodel.gz",
    "conv_256to512_tiled_fix/c1x1_emitted_scaled.axmodel.gz",
]


def _first_compressed(mc):
    for i, (start, size, table, comp) in enumerate(codec.segment_streams(mc)):
        if comp:
            return i, start, size, table
    raise AssertionError("no compressed segment")


@pytest.mark.parametrize("rel", _NATIVE)
def test_native_blobs_decompress_cleanly(rel):
    mc = _neu(rel)
    assert mcode.codec_violations(mc) == []


@pytest.mark.parametrize("rel", _CORRUPT)
def test_patch_scales_token_overwrites_are_reported(rel):
    problems = mcode.codec_violations(_neu(rel))
    assert len(problems) == 1, problems
    assert "not records" in problems[0] and "LZ77 token" in problems[0]
    assert problems[0] in mcode.check(_neu(rel))


def test_an_edit_through_the_codec_stays_clean():
    mc = _neu("teng_register_census/div_s1.axmodel.gz")
    teng = 2
    raw = bytearray(codec.decode_segments(mc)[teng])
    at = next(r for r in range(0, len(raw), 8) if raw[r + 2 : r + 4] == b"\x10\x1b")
    raw[at + 4] = 100  # zp_x 128 -> 100
    new = codec.replace_segment(mc, teng, bytes(raw))
    assert new != mc
    assert mcode.codec_violations(new) == []


def test_a_bad_back_reference_offset_is_reported():
    mc = bytearray(_neu("dma_tiles/relu_1x64x56x56.axmodel.gz"))
    _, start, size, _ = _first_compressed(bytes(mc))
    first_copy = next(
        t for t in codec.tokens(bytes(mc[start : start + size])) if t[0] == "M"
    )
    # The offset byte of the stream's first copy now points 1020+ bytes back,
    # past everything decoded so far.
    mc[start + first_copy[1] + 1] = 0xFF
    problems = mcode.codec_violations(bytes(mc))
    assert any("does not decompress" in p for p in problems), problems


def test_non_zero_padding_is_reported():
    mc = bytearray(_neu("dma_tiles/relu_1x64x56x56.axmodel.gz"))
    slot = next(
        (start, size, 8 * table[2])
        for start, size, table, comp in codec.segment_streams(bytes(mc))
        if comp and 8 * table[2] > size
    )
    start, size, _ = slot
    mc[start + size] = 0x01
    problems = mcode.codec_violations(bytes(mc))
    assert any("padding" in p for p in problems), problems


def test_a_missing_header_record_is_reported():
    mc = bytearray(_neu("dma_tiles/relu_1x64x56x56.axmodel.gz"))
    _, start, _, _ = _first_compressed(bytes(mc))
    assert mc[start + 1] == 0xA7
    mc[start + 1] = 0xA1
    problems = mcode.codec_violations(bytes(mc))
    assert any("a7 00 00 header" in p for p in problems), problems


def test_the_fixed_1x1_scale_patch_passes_the_check():
    """The device retest's MCode: #1790's 1x1 holdout scale-patched with the
    fixed `patch_scales`. It passes the decompressing check and decodes to the
    native build's segment lengths. What still differs from native, in decoded
    records, is the zero points and the float32 1/x_s that `patch_scales` does
    not cover, plus one reordered group of records."""
    import patch_scales

    ref = _neu("conv_learn_256to512/c1x1_reference.axmodel.gz")
    native = _neu("conv_learn_256to512/c1x1_holdout_native.axmodel.gz")
    # `quant_axmodel.json` scales of the reference and holdout builds.
    old = {"x": 0.007058821618556976, "w": 0.0010011724662035704}
    old["y"] = 0.015794403851032257
    new = {"x": 0.007058814167976379, "w": 0.001063680392690003}
    new["y"] = 0.016209039837121964
    slots = patch_scales.find_scale_slots(ref, old, new)
    assert [s[0] for s in slots if s[3] != s[4]] == [3167, 3175, 3183, 3191]
    out = bytearray(ref)
    for at, _, _, old_b, new_b in slots:
        out[at : at + len(old_b)] = new_b
    out = bytes(out)
    assert mcode.check(out) == []
    decoded = codec.decode_segments(out)
    assert [len(d) for d in decoded] == [len(d) for d in codec.decode_segments(native)]
    differing = [
        (s, r // 8)
        for s, (a, b) in enumerate(zip(decoded, codec.decode_segments(native)))
        for r in range(0, len(a), 8)
        if a[r : r + 8] != b[r : r + 8]
    ]
    assert differing == [
        (0, 58),  # x zero point 128 -> 127
        (0, 187),  # the reordered group
        (0, 188),
        (0, 190),
        (1, 61),  # x zero point
        (2, 134),  # x zero point (0x1b10)
        *[(2, r) for r in range(140, 148)],  # float32 1/x_s, low mantissa
        (2, 479),  # y zero point 125 -> 132 (0x1a90)
    ]
