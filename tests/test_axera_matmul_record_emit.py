"""No-device checks for matmul_record_emit: recalibrating a live-operand
MatMul template from scales alone must reproduce a native Pulsar2 build at
the target calibration, record for record and npu_params byte for byte, and
must refuse what it cannot explain."""

import os
import struct
import sys

import pytest

_AXERA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA not in sys.path:
    sys.path.insert(0, _AXERA)

import matmul_record_emit as mre  # noqa: E402
import short_unit_codec as codec  # noqa: E402
import step_recalibrate as sr  # noqa: E402

FIX = os.path.join(_AXERA, "fixtures")
HERE = os.path.join(FIX, "matmul_record_emit")
GATHER = os.path.join(FIX, "gather_aggregate_real")


def _build(model, quant):
    return mre.load_model(model), mre.load_scales(quant)


SMALL = {
    name: _build(
        os.path.join(HERE, f"matmul_2x4x8x8_{name}.axmodel.gz"),
        os.path.join(HERE, f"matmul_2x4x8x8_{name}.quant.json.gz"),
    )
    for name in ("acal_narrow", "acal_wide")
}
# The step's conv5 im2col MatMul, [1,1,512,4608] x [16,1,4608,49], in its
# real Gather -> Mul -> Reshape -> MatMul chain, at two calibrations.
STEP_CHAIN = {
    "a": _build(
        os.path.join(GATHER, "a_reference_narrow.axmodel.gz"),
        os.path.join(HERE, "gather_a_reference.quant.json.gz"),
    ),
    "c": _build(
        os.path.join(GATHER, "c_wide_reference.axmodel.gz"),
        os.path.join(HERE, "gather_c_wide_reference.quant.json.gz"),
    ),
}
# The step's classifier: MatMul [16,512] x [512,1000] + live bias [1000].
GEMM = _build(
    os.path.join(HERE, "matmul_add_16x512x1000.axmodel.gz"),
    os.path.join(HERE, "matmul_add_16x512x1000.quant.json.gz"),
)


def _exact(report):
    assert report["structure"]
    assert report["record_diffs"] == []
    assert report["params_diff_bytes"] == 0


@pytest.mark.parametrize(
    "src,dst", [("acal_narrow", "acal_wide"), ("acal_wide", "acal_narrow")]
)
def test_small_matmul_recalibrates_to_native_build(src, dst):
    (model, old), (want, new) = SMALL[src], SMALL[dst]
    assert old != new
    out, report = mre.recalibrate(model, old, new)
    assert report["segments_changed"] == [2]
    _exact(mre.compare(out, want))


@pytest.mark.parametrize("src,dst", [("a", "c"), ("c", "a")])
def test_step_shape_chain_recalibrates_to_native_build(src, dst):
    (model, old), (want, new) = STEP_CHAIN[src], STEP_CHAIN[dst]
    assert old["y"] != new["y"]
    out, report = mre.recalibrate(model, old, new)
    assert report["records"] == 890
    assert report["params_words"] == 98
    got = mre.compare(out, want)
    _exact(got)
    # Only segment 0's tail rotation differs, as between any two rebuilds.
    assert got["noise_diffs"] == 4


def test_gemm_template_every_lane_explained():
    model, scales = GEMM
    found = mre.locate(model, scales)
    assert all(roles for *_, roles in found["records"])
    kinds = {
        min((r for r in roles), key=lambda r: mre.PRECEDENCE.index(r[0]))
        for *_, roles in found["records"]
    }
    assert {("inv32", "a"), ("inv32", "b"), ("inv32", "c"), ("s", "z")} <= kinds
    assert {("zp", "z"), ("zp", "c")} <= kinds
    lanes = {tuple(roles[:1]) for _, _, roles in found["params"]}
    assert lanes == {(("zpf", "t"),), (("mult", "a", "b", "t"),)}
    assert len(found["params"]) == 2 * 520


def test_gemm_template_identity_and_round_trip():
    model, scales = GEMM
    same, _ = mre.recalibrate(model, scales, scales)
    assert sr.get_mcode(same) == sr.get_mcode(model)
    assert mre.params_of(same) == mre.params_of(model)

    moved = {t: (s * 1.37, z) for t, (s, z) in scales.items()}
    moved["z"] = (moved["z"][0], 120.0)
    out, _ = mre.recalibrate(model, scales, moved)
    lanes = mre.locate(out, moved)
    assert all(roles for *_, roles in lanes["records"])
    zp = [v for _, _, reg, v, _ in lanes["records"] if reg == 0x1A90]
    assert zp == [120]
    back, _ = mre.recalibrate(out, moved, scales)
    assert codec.decode_segments(sr.get_mcode(back)) == codec.decode_segments(
        sr.get_mcode(model)
    )
    assert mre.params_of(back) == mre.params_of(model)


def test_refuses_zero_point_crossing_and_missing_tensor():
    model, scales = GEMM
    to_zero = dict(scales, z=(scales["z"][0], 0.0))
    with pytest.raises(mre.CalibrationError, match="zero and nonzero"):
        mre.recalibrate(model, scales, to_zero)
    with pytest.raises(mre.CalibrationError, match="miss"):
        mre.recalibrate(model, scales, {t: v for t, v in scales.items() if t != "c"})


def test_refuses_unexplained_and_ambiguous_values():
    scales = {"p": (0.5, 0.0), "q": (0.5, 0.0)}
    with pytest.raises(mre.CalibrationError, match="no scale formula"):
        mre._new_value("x", 0x12345678, [], scales)
    # 1/s_p and 1/s_q agree at the template and part at the new scales.
    old = struct.unpack("<I", struct.pack("<f", 2.0))[0]
    roles = [("inv32", "p"), ("inv32", "q")]
    assert mre._new_value("x", old, roles, scales) == old
    with pytest.raises(mre.CalibrationError, match="ambiguous"):
        mre._new_value("x", old, roles, {"p": (0.5, 0.0), "q": (0.25, 0.0)})


def test_step_shape_tables_cover_the_step():
    assert sum(n for n, _ in mre.STEP_SHAPES.values()) == 42  # 41 MatMul + Gemm
    assert sum(mre.STEP_CONV_MATMULS.values()) == 20
