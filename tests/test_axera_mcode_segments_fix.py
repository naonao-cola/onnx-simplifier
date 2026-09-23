"""`mcode.segments` places every segment where its LZ77 stream really starts.

Segments are 8-byte aligned; the FlatBuffers tail vector after them is only
4-byte aligned. When the vector sits at 4 mod 8, four zero bytes of padding
precede it, and `segments` used to tile backward from the vector through that
padding -- every segment start landed 4 bytes late, splitting the stream's
opening `t a7 00 00` record. See docs/axera-mcode-segments-fix.md.
"""

import gzip
import os
import sys

import pytest

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import mcode  # noqa: E402
import short_unit_codec  # noqa: E402

_FIXTURES = os.path.join(_AXERA_DIR, "fixtures")


def _blob(name):
    with gzip.open(os.path.join(_FIXTURES, name), "rb") as f:
        return f.read()


def _fixture_mcodes():
    for root, _, files in os.walk(_FIXTURES):
        for f in sorted(files):
            if f.endswith(".mcode.gz"):
                yield os.path.relpath(os.path.join(root, f), _FIXTURES)


def _stream_opens_at(mc, start, table):
    if 5 in table:  # compressed: literal token, then the a7 header record
        return mc[start] < 0x80 and mc[start + 1 : start + 4] == b"\xa7\x00\x00"
    return mc[start : start + 3] == b"\xa7\x00\x00"


# (fixture, tail padding, header end = first segment start)
_EXAMPLES = [
    ("adam_update_fp32.mcode.gz", 4, 288),
    ("reshape_gather_bwd.mcode.gz", 4, 280),
    ("attn_qkv_softmax.mcode.gz", 0, 280),
    ("resnet18_int8.mcode.gz", 0, 320),
]


@pytest.mark.parametrize("name,pad,header", _EXAMPLES)
def test_padded_and_unpadded_blobs_have_the_right_header(name, pad, header):
    mc = _blob(name)
    assert mcode.tail_padding(mc) == pad
    got_header, segs = mcode.segments(mc)
    assert got_header == header
    assert segs[0][0] == header
    last_pos, last_len, _ = segs[-1]
    assert last_pos + last_len + pad == mcode.tail_vector(mc)


def test_both_padding_groups_are_represented():
    pads = {mcode.tail_padding(_blob(n)) for n in _fixture_mcodes()}
    assert pads == {0, 4}


@pytest.mark.parametrize("name", list(_fixture_mcodes()))
def test_every_segment_starts_on_its_stream_header(name):
    mc = _blob(name)
    _, segs = mcode.segments(mc)
    for i, (pos, _length, table) in enumerate(segs):
        assert pos % 8 == 0, (i, pos)
        assert _stream_opens_at(mc, pos, table), (i, pos, mc[pos : pos + 8].hex())
        assert short_unit_codec.stream_start(mc, pos, table) == pos


def test_non_zero_padding_is_rejected():
    mc = bytearray(_blob("adam_update_fp32.mcode.gz"))
    vec = mcode.tail_vector(bytes(mc))
    mc[vec - 1] = 1
    with pytest.raises(AssertionError):
        mcode.segments(bytes(mc))
    assert mcode.check(bytes(mc)) != []
