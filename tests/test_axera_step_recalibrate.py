"""No-device checks for step_recalibrate: segment re-layout must reproduce
native Pulsar2 builds byte for byte, and recalibration must refuse anything
that is not a same-shape value edit."""

import os
import struct
import sys

import pytest

_AXERA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA not in sys.path:
    sys.path.insert(0, _AXERA)

import mcode  # noqa: E402
import short_unit_codec as codec  # noqa: E402
import step_recalibrate as sr  # noqa: E402

FIX = os.path.join(_AXERA, "fixtures")
STEP = os.path.join(FIX, "step_recalib")
RELU = os.path.join(FIX, "elementwise_scale_emit")

# (smaller build, larger build): the same graph where one segment's padded
# slot differs. toyf has no gap before the tail; the Relu has 4 bytes.
PAIRS = [
    (
        os.path.join(STEP, "toyf_A1.axmodel.gz"),
        os.path.join(STEP, "toyf_Bx2.axmodel.gz"),
    ),
    (
        os.path.join(RELU, "relu_16x64x56x56_x0_y0.axmodel.gz"),
        os.path.join(RELU, "relu_16x64x56x56_x128_y128.axmodel.gz"),
    ),
]


def _mc(path):
    return sr.get_mcode(sr.load(path))


def _splice_native(src, dst):
    for i, (s, n, _, comp) in enumerate(codec.segment_streams(dst)):
        if comp:
            src = sr.relayout(src, i, dst[s : s + n])
    return src


@pytest.mark.parametrize("small,large", PAIRS)
def test_relayout_reproduces_native_growth_and_shrink(small, large):
    a, b = _mc(small), _mc(large)
    assert len(a) != len(b)
    assert _splice_native(a, b) == b
    assert _splice_native(b, a) == a


def test_header_refs_skip_string_bytes_that_look_like_offsets():
    a = _mc(PAIRS[0][0])
    s0 = codec.segment_streams(a)[0][0]
    refs = dict(sr.header_refs(a, s0))
    # Words that native growth changes (read off toyf_A1 -> toyf_Bx2).
    for o in (32, 36, 48, 52, 56, s0 - 8):
        assert refs.get(o) == "u"
    assert refs.get(60) == "s"
    # String bytes whose value, read as a uoffset, lands past segment 2.
    for o in (364, 1316, 1800, 2588):
        assert o not in refs


def test_relayout_same_size_only_rewrites_key5():
    a = _mc(PAIRS[0][0])
    s, n, _, _ = codec.segment_streams(a)[1]
    assert sr.relayout(a, 1, a[s : s + n]) == a


def test_recalibrate_a1_to_bx2_matches_native_bx2():
    a, b = sr.load(PAIRS[0][0]), sr.load(PAIRS[0][1])
    new, report = sr.recalibrate(a, b)
    assert report["records"] == {2: 570}
    assert report["params_bytes"] == 92
    assert report["mcode_bytes"] == [31032, 31160]
    mc = sr.get_mcode(new)
    assert codec.decode_segments(mc) == codec.decode_segments(sr.get_mcode(b))
    assert mcode.check(mc) == []
    name = sr.mcode_name(new)
    init = next(i for i in new.graph.initializer if i.name == name)
    assert list(init.dims) == [len(mc)]
    vi = next(v for v in new.graph.value_info if v.name == name)
    assert vi.type.tensor_type.shape.dim[0].dim_value == len(mc)
    # Apart from the re-encoded stream, the whole model is the native build.
    native = sr.load(PAIRS[0][1])
    for i in native.graph.initializer:
        if i.name == name:
            i.raw_data = mc
    assert new.SerializeToString() == native.SerializeToString()


def test_recalibrate_refuses_a_record_count_change():
    # Relu zero point 0 -> 128: Pulsar2 emits extra records for a nonzero
    # zero point, so segment 2 decodes to a different length.
    a, b = sr.load(PAIRS[1][0]), sr.load(PAIRS[1][1])
    with pytest.raises(sr.RelayoutError, match="record count"):
        sr.recalibrate(a, b)


def test_recalibrate_refuses_npu_params_length_change():
    a, b = sr.load(PAIRS[0][0]), sr.load(PAIRS[0][1])
    for i in b.graph.initializer:
        if i.name == sr.PARAMS:
            i.raw_data = bytes(i.raw_data) + b"\0"
    with pytest.raises(sr.RelayoutError, match="npu_params"):
        sr.recalibrate(a, b)


def test_relayout_rejects_a_bad_length_prefix():
    a = bytearray(_mc(PAIRS[0][0]))
    s0 = codec.segment_streams(bytes(a))[0][0]
    struct.pack_into("<I", a, s0 - 4, 1)
    b = _mc(PAIRS[0][1])
    s, n, _, _ = codec.segment_streams(b)[2]
    with pytest.raises(sr.RelayoutError, match="length prefix"):
        sr.relayout(bytes(a), 2, b[s : s + n])
