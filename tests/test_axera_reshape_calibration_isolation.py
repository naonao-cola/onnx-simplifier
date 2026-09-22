"""Regression coverage for the Reshape calibration-isolation characterization.

No Docker/device is required: it checks the decode logic against committed
fixtures, reproducing the numbers in
``docs/axera-reshape-calibration-isolation.md``.
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

from reshape_calibration_isolation import (  # noqa: E402
    FIELD_OFFSETS,
    fields_match_formula,
    fixed_range_samples,
)

_FIXTURES = os.path.join(_AXERA_DIR, "fixtures", "reshape_calib_isolation")


def _unzip(tmp_path, name):
    path = tmp_path / f"{name}.axmodel"
    with gzip.open(os.path.join(_FIXTURES, f"{name}.axmodel.gz"), "rb") as source:
        path.write_bytes(source.read())
    return str(path)


def _mcode(path):
    model = onnx.load(path, load_external_data=False)
    return bytes(
        next(i.raw_data for i in model.graph.initializer if i.name.endswith("_neu"))
    )


def _decode(tmp_path, c, name):
    import struct

    s = _seg2(tmp_path, c, name)
    return {
        "row_bytes_36c": struct.unpack_from("<H", s, FIELD_OFFSETS["row_bytes_36c"])[0],
        "c_minus_1_a": s[FIELD_OFFSETS["c_minus_1_a"]],
        "c_minus_1_b": s[FIELD_OFFSETS["c_minus_1_b"]],
        "half_c_sq_minus_1": struct.unpack_from(
            "<H", s, FIELD_OFFSETS["half_c_sq_minus_1"]
        )[0],
    }


def _seg2(tmp_path, c, name):
    import mcode as mcodelib

    path = _unzip(tmp_path, name)
    mc = _mcode(path)
    segs = mcodelib.segments(mc)[1]
    start, length = segs[2][0], segs[2][1]
    return mc[start : start + length]


@pytest.mark.parametrize(
    "name,c",
    [("fixed_C40", 40), ("fixed_C56", 56), ("fixed_C60", 60), ("fixed_C64", 64)],
)
def test_matched_shapes_satisfy_the_decoded_formulas(tmp_path, name, c):
    fields = _decode(tmp_path, c, name)
    assert fields_match_formula(c, fields)


def test_unmatched_shapes_do_not_satisfy_the_formulas(tmp_path):
    fields = _decode(tmp_path, 44, "fixed_C44")
    assert not fields_match_formula(44, fields)


def test_matched_pair_differs_by_far_fewer_bytes_than_an_unmatched_pair(tmp_path):
    a = _seg2(tmp_path, 40, "fixed_C40")
    b = _seg2(tmp_path, 64, "fixed_C64")
    c = _seg2(tmp_path, 44, "fixed_C44")
    assert len(a) == len(b) == len(c)
    matched_diff = sum(1 for i in range(len(a)) if a[i] != b[i])
    unmatched_diff = sum(1 for i in range(len(a)) if a[i] != c[i])
    # C=40 vs C=64 (both satisfy the formula): 29 bytes.
    # C=40 vs C=44 (44 does not satisfy it): 353 bytes.
    assert matched_diff < 50
    assert unmatched_diff > 300


def test_calibration_range_alone_changes_a_handful_of_bytes(tmp_path):
    narrow = _mcode(_unzip(tmp_path, "sens_narrow"))
    wide = _mcode(_unzip(tmp_path, "sens_wide"))
    assert len(narrow) == len(wide)
    diffs = [
        i for i in range(len(narrow)) if narrow[i] != wide[i] and not (301 <= i < 326)
    ]
    # Same shape, same layout, only the calibration range (-0.9..0.9 vs -9..9)
    # differs: a real but small effect, nowhere near the hundreds of bytes a
    # shape change causes.
    assert 0 < len(diffs) < 50


def test_fixed_range_samples_bakes_in_the_exact_bounds():
    samples = fixed_range_samples([4, 4], n=2, lo=-0.9, hi=0.9)
    assert len(samples) == 2
    for arr in samples:
        assert arr.min() == pytest.approx(-0.9, abs=1e-6) or arr.flat[
            0
        ] == pytest.approx(-0.9, abs=1e-6)
        assert arr.flat[1] == pytest.approx(0.9, abs=1e-6)
