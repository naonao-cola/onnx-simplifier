"""No-device checks for the short_unit_codec device harness: every variant it
runs on hardware must first round-trip exactly in software."""

import os
import struct
import sys

import pytest

_AXERA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA not in sys.path:
    sys.path.insert(0, _AXERA)

import short_unit_codec as codec  # noqa: E402
import short_unit_codec_device_check as dc  # noqa: E402


@pytest.mark.parametrize("path", dc.ROUNDTRIPS)
def test_reencode_all_decodes_back_exactly(path):
    mc = dc.get_mcode(dc.load(path))
    new, changed = dc.reencode_all(mc)
    assert codec.decode_segments(new) == codec.decode_segments(mc)
    assert len(new) == len(mc)
    # These fixtures were picked because the encoder does not reproduce the
    # native stream for at least one segment, so the device run tests it.
    assert changed


@pytest.mark.parametrize("name,src,dst", dc.TRANSPLANTS)
def test_transplant_reproduces_the_target_programs(name, src, dst):
    a, b = dc.get_mcode(dc.load(src)), dc.get_mcode(dc.load(dst))
    new, moved = dc.transplant(a, b)
    assert moved
    assert codec.decode_segments(new) == codec.decode_segments(b)
    assert len(new) == len(a)


def test_scale_edit_changes_only_the_targeted_record():
    mc = dc.get_mcode(dc.load(dc.RELU))
    new, old, value = dc.scale_edit(mc, 2, 0x0F70, 1, 2.0)
    assert value == pytest.approx(2 * old)
    before, after = codec.decode_segments(mc), codec.decode_segments(new)
    assert [i for i in range(len(before)) if before[i] != after[i]] == [2]
    changed = dc.diff_records(before[2], after[2])
    assert len(changed) == 1
    rec = after[2][changed[0] * 8 : changed[0] * 8 + 8]
    assert rec[2] | rec[3] << 8 == 0x0F70
    assert struct.unpack("<f", rec[4:])[0] == pytest.approx(value)


def test_lane_effect_counts_one_lane():
    import numpy as np

    c = np.arange(1, 17, dtype=np.float32)
    p = c.copy()
    p[2::4] *= 2
    res = dc.lane_effect(c.tobytes(), p.tobytes())
    assert [res[k]["changed"] for k in range(4)] == [0, 0, 4, 0]
    assert res[2]["ratio_min"] == res[2]["ratio_max"] == 2.0
