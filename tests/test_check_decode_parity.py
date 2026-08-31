"""Unit tests for scripts/apple/check_decode_parity.py's comparison logic.

`compare_token_sequences` is plain list-of-ints logic with no coremltools/
torch/transformers dependency, so it's tested directly here rather than only
via the coremltools-gated scripts under scripts/apple -- see that module's
docstring for why the check is agreement-rate-based, not exact-match.
"""

import os
import sys

import pytest

_APPLE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "apple"
)
if _APPLE_DIR not in sys.path:
    sys.path.insert(0, _APPLE_DIR)

from check_decode_parity import compare_token_sequences  # noqa: E402


def test_identical_sequences_agree_fully():
    result = compare_token_sequences([1, 2, 3, 4], [1, 2, 3, 4])
    assert result == {"compared": 4, "agreement_rate": 1.0, "first_divergence": None}


def test_diverges_at_first_mismatch():
    result = compare_token_sequences([1, 2, 3, 4], [1, 2, 9, 4])
    assert result["compared"] == 4
    assert result["agreement_rate"] == 0.75
    assert result["first_divergence"] == 2


def test_diverges_immediately():
    result = compare_token_sequences([1, 2, 3], [9, 2, 3])
    assert result["first_divergence"] == 0
    assert result["agreement_rate"] == pytest.approx(2 / 3)


def test_completely_disjoint():
    result = compare_token_sequences([1, 2, 3], [4, 5, 6])
    assert result == {"compared": 3, "agreement_rate": 0.0, "first_divergence": 0}


def test_compares_only_the_overlapping_length():
    # One side generated fewer tokens (e.g. hit EOS earlier) -- only the
    # shared prefix is compared, not padded/truncated to match.
    result = compare_token_sequences([1, 2, 3, 4, 5], [1, 2, 3])
    assert result["compared"] == 3
    assert result["agreement_rate"] == 1.0
    assert result["first_divergence"] is None


def test_empty_sequences():
    result = compare_token_sequences([], [])
    assert result == {"compared": 0, "agreement_rate": 1.0, "first_divergence": None}
