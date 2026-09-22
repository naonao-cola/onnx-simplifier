"""Regression coverage for the stem Gather + leading-Reshape index-layout
finding (docs/axera-stem-gather-reshape.md). No Docker or device required."""

import os
import sys

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

from stem_gather_reshape_check import best_shift_match, table_length_diff  # noqa: E402


def test_table_length_diff_is_80_bytes():
    assert table_length_diff() == 80


def test_index_region_does_not_match_at_any_small_shift():
    # A "small block inserted at the front" layout (e.g. 20 new metadata
    # words matching the 80-byte length diff) would give a near-1.0 match at
    # some shift. The real result stays near the chance level for a mostly
    # small/clamped-index array (many zeros), confirming the layout was
    # genuinely rewritten, not just shifted.
    shift, match = best_shift_match()
    assert match < 0.05
