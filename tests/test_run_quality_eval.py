"""Unit tests for scripts/apple/run_quality_eval.py's pure completed-only
accounting logic (`is_completed`, `aggregate_completed_only`) -- the
lm_eval/transformers-dependent parts (tokenizer loading, `simple_evaluate`)
aren't exercised here, see that module's docstring and
bench/TODO_quality_retention_eval.md for how those are validated (an actual
CI run, plus the manual token-count-vs-cap check that motivated this design).
"""

import os
import sys

import pytest

_APPLE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "apple"
)
if _APPLE_DIR not in sys.path:
    sys.path.insert(0, _APPLE_DIR)

from run_quality_eval import aggregate_completed_only, is_completed  # noqa: E402


def test_is_completed_true_when_under_cap():
    assert is_completed(50, 256) is True


def test_is_completed_false_when_at_cap():
    assert is_completed(256, 256) is False


def test_is_completed_false_when_over_cap():
    assert is_completed(300, 256) is False


def test_is_completed_none_when_cap_unknown():
    assert is_completed(50, None) is None


def test_aggregate_completed_only_scalar_mean():
    assert aggregate_completed_only([1, 0, 1, 1]) == 0.75


def test_aggregate_completed_only_flattens_list_valued_metrics():
    # ifeval's inst_level_*_acc is a list of per-instruction booleans per
    # sample -- lm_eval pools every instruction across every sample before
    # averaging, this should match that for the completed-only subset.
    assert aggregate_completed_only([[True, False], [True]]) == pytest.approx(2 / 3)


def test_aggregate_completed_only_mixed_scalar_and_list():
    assert aggregate_completed_only([1, [1, 0]]) == pytest.approx(2 / 3)


def test_aggregate_completed_only_empty_returns_none():
    assert aggregate_completed_only([]) is None
