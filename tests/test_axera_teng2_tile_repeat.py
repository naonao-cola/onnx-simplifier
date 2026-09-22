"""Regression coverage for the `teng2`-repeats-per-tile hypothesis check.

Confirms, against the `dma_tiles` fixtures committed for
`dma_tile_predict.py` (PR #1754), that segment 2 (`teng2`, the compute
program) is NOT `n` copies of one per-tile block, and characterizes segment 4
(`sdma4`) instead. See `docs/axera-teng2-tiled-repeat.md`.
"""

import os
import sys

import pytest

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

from teng2_tile_repeat_check import (  # noqa: E402
    load_mcode_params_shape,
    n_entries,
    sdma4_summary,
    teng2_periodicity,
)

_FIXTURES = os.path.join(_AXERA_DIR, "fixtures", "dma_tiles")

# name -> (tile entry count n, expected sdma4 empty?)
_TILED_CASES = {
    "relu_1x64x56x56.axmodel.gz": (4, False),
    "relu_1x64x56x56_rb1.axmodel.gz": (4, False),
    "relu_1x192x56x56.axmodel.gz": (8, False),
    "relu_4x64x56x56.axmodel.gz": (8, False),
    "relu_1x384x56x56.axmodel.gz": (16, False),
    "relu_16x128x28x28.axmodel.gz": (16, True),
    "relu_16x64x56x56.axmodel.gz": (32, True),
}


@pytest.mark.parametrize("gz_name,expected", _TILED_CASES.items())
def test_teng2_is_not_periodic_at_size_over_n(gz_name, expected):
    """Segment 2's self-similarity at period=size/n stays near the noise floor --
    nowhere close to a real per-tile repeat (which would be well above 0.5, the way
    segment 4 -- see `test_sdma4_is_task_structured_not_tile_structured` -- is)."""
    n, _ = expected
    mcode, params, _ = load_mcode_params_shape(os.path.join(_FIXTURES, gz_name))
    assert n_entries(params) == n
    result = teng2_periodicity(mcode, params)
    assert result["n"] == n
    sim = result["byte_selfsim_at_size_over_n"]
    assert sim is not None
    assert sim < 0.15
    # Record-level check: when the record count is divisible by n, the n blocks (split by
    # record count, not byte offset) still don't share a form -- only the trivial self-match.
    blocks = result["record_blocks_matching_first"]
    if blocks is not None:
        assert blocks == 1


@pytest.mark.parametrize("gz_name,expected", _TILED_CASES.items())
def test_sdma4_is_task_structured_not_tile_structured(gz_name, expected):
    """Segment 4 (`sdma4`) is either a real, `0xa3`-terminated task queue or a bare 32-byte
    placeholder -- and which one depends on total tensor bytes, not on `n` (16x128x28x28 and
    1x384x56x56 share n=16 but disagree on this)."""
    _, empty = expected
    mcode, _, _ = load_mcode_params_shape(os.path.join(_FIXTURES, gz_name))
    summary = sdma4_summary(mcode)
    assert summary["empty"] == empty
    if not empty:
        assert summary["verb_counts"].get(0xA3, 0) > 0


def test_untiled_relu_has_no_tile_period():
    mcode, params, _ = load_mcode_params_shape(
        os.path.join(_FIXTURES, "relu_1x16x32x32.axmodel.gz")
    )
    assert n_entries(params) == 1
    result = teng2_periodicity(mcode, params)
    assert result["byte_selfsim_at_size_over_n"] is None
