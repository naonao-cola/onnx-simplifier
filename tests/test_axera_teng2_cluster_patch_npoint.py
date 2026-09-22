"""Regression coverage for the widened-field teng2 cluster extrapolation tool.
See scripts/axera/teng2_cluster_patch_npoint.py's docstring and
docs/axera-teng2-cluster-patch-refine.md for what this fixes and does not."""

import gzip
import os
import sys

import onnx

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

from teng2_cluster_patch_npoint import extrapolate  # noqa: E402

_SQRT_BLOCKS = os.path.join(_AXERA_DIR, "fixtures", "teng2_sqrt_blocks")
_REFINE = os.path.join(_AXERA_DIR, "fixtures", "teng2_cluster_refine")


def _unzip(tmp_path, fixture_dir, name):
    path = tmp_path / f"{name}.axmodel"
    with gzip.open(os.path.join(fixture_dir, f"{name}.axmodel.gz"), "rb") as source:
        path.write_bytes(source.read())
    return str(path)


def _mcode(path):
    model = onnx.load(path, load_external_data=False)
    return bytes(
        next(i.raw_data for i in model.graph.initializer if i.name.endswith("_neu"))
    )


def _diff(pred_mcode, real_mcode):
    return [
        i
        for i in range(len(pred_mcode))
        if pred_mcode[i] != real_mcode[i] and not (301 <= i < 326)
    ]


def test_extrapolate_closes_the_153_154_155_carry_residual(tmp_path):
    # teng2_cluster_patch.extrapolate leaves 2 residual bytes here (a real,
    # documented carry limitation). The widened field window closes it to zero
    # using the SAME two references -- no third build needed for this cluster.
    ref_a = _unzip(tmp_path, _SQRT_BLOCKS, "sqrt_c153")
    ref_b = _unzip(tmp_path, _SQRT_BLOCKS, "sqrt_c154")
    real_c155 = _unzip(tmp_path, _SQRT_BLOCKS, "sqrt_c155")
    out = str(tmp_path / "pred_155.axmodel")

    extrapolate(ref_a, ref_b, 155, out)

    assert _diff(_mcode(out), _mcode(real_c155)) == []


def test_extrapolate_still_exact_for_the_157_158_159_cluster_no_regression(tmp_path):
    ref_a = _unzip(tmp_path, _SQRT_BLOCKS, "sqrt_c157")
    ref_b = _unzip(tmp_path, _SQRT_BLOCKS, "sqrt_c158")
    real_c159 = _unzip(tmp_path, _SQRT_BLOCKS, "sqrt_c159")
    out = str(tmp_path / "pred_159.axmodel")

    extrapolate(ref_a, ref_b, 159, out)

    assert _diff(_mcode(out), _mcode(real_c159)) == []


def test_extrapolate_reduces_but_does_not_close_the_165_166_167_residual(tmp_path):
    # A genuinely new build (not previously committed): teng2_cluster_patch's
    # unwidened tool leaves 31 residual bytes here. Widening fixes 2 of them (the
    # carry-type ones) but the remaining 29 are a real byte-INSERTION between
    # C=166 and C=167 (a length-changing re-encoding, not a carry) -- see the
    # module docstring. This is not fixable by any fixed-width delta model.
    ref_a = _unzip(tmp_path, _REFINE, "sqrt_c165")
    ref_b = _unzip(tmp_path, _REFINE, "sqrt_c166")
    real_c167 = _unzip(tmp_path, _REFINE, "sqrt_c167")
    out = str(tmp_path / "pred_167.axmodel")

    extrapolate(ref_a, ref_b, 167, out)

    mismatches = _diff(_mcode(out), _mcode(real_c167))
    # A real, known residual from a length-changing re-encoding this tool cannot
    # represent. If a future fix drives this to zero, tighten this test.
    assert 0 < len(mismatches) <= 29
