"""Patched MCode checked with the LZ77 codec (docs/axera-mcode-segments-fix.md).

`short_unit_codec` decodes every native fixture segment into whole 8-byte
records. The only committed blobs that do not decode cleanly came from our own
in-place patchers. Both are ResNet18 1x1-downsample holdouts: `patch_scales`
matched a bfloat16 scale against LZ77 token bytes and overwrote them.
`patch_scales` now skips such matches, and `emitter.emit_mcode` refuses to
write a learned field that sits on a token byte.
"""

import gzip
import os
import sys

import numpy as np
import onnx
import pytest

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import emitter  # noqa: E402
import mcode  # noqa: E402
import short_unit_codec  # noqa: E402

_FIXTURES = os.path.join(_AXERA_DIR, "fixtures")


def _neu(rel):
    with gzip.open(os.path.join(_FIXTURES, rel), "rb") as f:
        model = onnx.load_model_from_string(f.read())
    return next(
        bytes(i.raw_data) for i in model.graph.initializer if i.name.endswith("_neu")
    )


def _decoded_lengths(mc):
    return [len(d) for d in short_unit_codec.decode_segments(mc)]


def _non_record_words(mc):
    return [
        (i, w)
        for i, d in enumerate(short_unit_codec.decode_segments(mc))
        for w in range(0, len(d), 8)
        if any(d[w : w + 8]) and not (d[w] >= 0xA0 and d[w + 1] == 0)
    ]


_REF = "conv_learn_256to512/c1x1_reference.axmodel.gz"
_CORRUPT = [
    "conv_learn_256to512/c1x1_holdout_emitted.axmodel.gz",
    "conv_256to512_tiled_fix/c1x1_emitted_scaled.axmodel.gz",
]


@pytest.mark.parametrize("rel", _CORRUPT)
def test_the_1x1_holdout_emissions_re_decode_their_segments(rel):
    """Four one-byte scale writes hit back-reference tokens. That grows
    three segments' decoded output (the native holdout of the same shape
    decodes to exactly the reference's lengths) and leaves non-record words in
    segment 1. `mcode.check` used to pass it; since
    docs/axera-mcode-check-decompress.md it decompresses every segment and
    reports the non-record words."""
    ref, out = _neu(_REF), _neu(rel)
    native = _neu("conv_learn_256to512/c1x1_holdout_native.axmodel.gz")
    assert _decoded_lengths(ref) == [1536, 1536, 4352, 2176, 5760]
    assert _decoded_lengths(native) == _decoded_lengths(ref)
    assert _decoded_lengths(out) == [1544, 1544, 4368, 2176, 5760]
    tokens = short_unit_codec.token_bytes(ref)
    assert {724, 1340, 2916, 3150} <= tokens
    assert [i for i in range(len(ref)) if ref[i] != out[i] and i in tokens] == [
        724,
        1340,
        2916,
        3150,
    ]
    assert [i for i, _ in _non_record_words(out)] == [1, 1]
    assert _non_record_words(ref) == []
    # The decompressing check now catches it (it used to return []).
    problems = mcode.check(out)
    assert problems
    assert all("not records" in p for p in problems), problems
    assert mcode.check(ref) == []
    assert mcode.check(native) == []


def test_the_3x3_holdout_emission_only_touched_literals():
    ref = _neu("conv_learn_256to512/c3x3_reference.axmodel.gz")
    out = _neu("conv_learn_256to512/c3x3_holdout_emitted.axmodel.gz")
    tokens = short_unit_codec.token_bytes(ref)
    diff = [i for i in range(len(ref)) if ref[i] != out[i]]
    assert diff and not set(diff) & tokens
    assert _decoded_lengths(out) == _decoded_lengths(ref)


def test_emit_mcode_refuses_a_field_on_a_token_byte():
    ref = np.frombuffer(_neu(_REF), dtype=np.uint8)
    fields = {
        "scale_offsets": [3166, 3174, 3182, 3190],
        "zero_offsets": [724, 1340, 2916, 3150],
    }
    with pytest.raises(ValueError, match="LZ77 token bytes"):
        emitter.emit_mcode(ref, fields, 0.01, 139)
    # The scale copies are literal payload, so writing only them is allowed
    # and keeps every segment's decoded length.
    out = emitter.emit_mcode(ref, {"scale_offsets": fields["scale_offsets"]}, 0.01, 0)
    assert _decoded_lengths(out.tobytes()) == _decoded_lengths(ref.tobytes())


def test_patch_scales_skips_matches_on_token_bytes():
    """The four corrupting edits are bfloat16 `83 3a` -> `8b 3a` (a scale of
    ~0.0010 moving to ~0.0011). In this reference, each `83 3a` is a
    back-reference token plus its offset byte, not a stored scale.
    `find_scale_slots` used to patch all four and now finds none."""
    import struct

    import patch_scales

    ref = _neu(_REF)
    old = struct.unpack("<f", b"\x00\x00\x83\x3a")[0]
    new = struct.unpack("<f", b"\x00\x00\x8b\x3a")[0]
    assert patch_scales.find_scale_slots(ref, {"x": old}, {"x": new}) == []
    assert [ref.find(b"\x83\x3a", o) for o in (0, 725, 1341, 2917)] == [
        724,
        1340,
        2916,
        3150,
    ]


def test_token_bytes_covers_each_segment_opening_token():
    with gzip.open(os.path.join(_FIXTURES, "adam_update_fp32.mcode.gz"), "rb") as f:
        mc = f.read()
    tokens = short_unit_codec.token_bytes(mc)
    _, segs = mcode.segments(mc)
    for pos, _, table in segs:
        if 5 in table:
            assert pos in tokens  # the literal-run header
            assert pos + 1 not in tokens  # the a7 it introduces
