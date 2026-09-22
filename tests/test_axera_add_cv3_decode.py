"""Regression coverage for the Add `cv3` segment characterization.

Confirms the two decisive findings from `docs/axera-add-cv3-decode.md`
against the committed fixtures: `cv3`'s content is calibration-invariant, and
its length is not a function of `dma_tile_predict.py`'s tile-`entries` model
(nor of `N*C`, `H*W`, or their product alone).
"""

import os
import sys

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

from add_cv3_decode import MEASURED, load_cv3  # noqa: E402


def test_every_measured_fixture_loads_and_matches_recorded_length():
    for shape, cv3_len, _entries, fixture in MEASURED:
        assert len(load_cv3(fixture)) == cv3_len, (shape, fixture)


def test_cv3_is_calibration_invariant_at_a_fixed_shape():
    # Same [1,16,16,16] shape, three distinct calibration ranges (symmetric,
    # and two different asymmetric x/z splits) -- byte-identical cv3.
    fixtures = [
        "add_tiles/add_asym_1x16x16x16.axmodel.gz",
        "add_cv3_decode/sym_1x16x16x16.axmodel.gz",
        "add_cv3_decode/asym2_1x16x16x16.axmodel.gz",
    ]
    blobs = [load_cv3(f) for f in fixtures]
    assert len(set(blobs)) == 1


def test_cv3_length_is_not_a_function_of_nc_alone():
    # Two shapes share N*C=16 but differ in cv3_len depending on H*W.
    small_row = next(
        length for shape, length, _, _ in MEASURED if shape == (1, 16, 8, 8)
    )
    large_row = next(
        length for shape, length, _, _ in MEASURED if shape == (1, 16, 64, 64)
    )
    assert small_row != large_row


def test_cv3_length_is_not_a_function_of_tile_entries_alone():
    # Two shapes share the same predicted dma_tile_predict entries count (4)
    # but differ in cv3_len depending on the shape itself.
    entries4 = {shape: length for shape, length, entries, _ in MEASURED if entries == 4}
    assert len(set(entries4.values())) > 1


def test_cv3_length_boundary_is_bracketed_within_the_untiled_regime():
    # Both shapes are untiled (dma_tile_predict entries=1); the 256/32 split
    # is a second, finer threshold that model has no concept of.
    below = {
        shape: length
        for shape, length, entries, _ in MEASURED
        if entries == 1 and shape[0] * shape[1] * shape[2] * shape[3] * 4 <= 16384
    }
    above = {
        shape: length
        for shape, length, entries, _ in MEASURED
        if entries == 1 and shape[0] * shape[1] * shape[2] * shape[3] * 4 >= 65536
    }
    assert below and above
    assert set(below.values()) == {256}
    assert set(above.values()) == {32}
