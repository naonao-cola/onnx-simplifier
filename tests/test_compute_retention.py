"""Unit tests for scripts/apple/compute_retention.py's retention computation.

`compute_retention` is plain dict logic with no lm-evaluation-harness/torch/
coremltools dependency, so it's tested directly here -- see that module's
docstring for what retention means and why a zero float score reports `None`
rather than `inf`/`nan`.
"""

import os
import sys

import pytest

_APPLE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "apple"
)
if _APPLE_DIR not in sys.path:
    sys.path.insert(0, _APPLE_DIR)

from compute_retention import compute_retention  # noqa: E402


def test_full_retention_when_scores_match():
    result = compute_retention(
        {"ifeval": {"prompt_level_strict_acc,none": 0.8}},
        {"ifeval": {"prompt_level_strict_acc,none": 0.8}},
    )
    assert result == {
        "ifeval": {
            "prompt_level_strict_acc,none": {
                "float": 0.8,
                "quantized": 0.8,
                "retention": 1.0,
            }
        }
    }


def test_partial_retention():
    result = compute_retention(
        {"ifeval": {"acc": 0.8}},
        {"ifeval": {"acc": 0.6}},
    )
    assert result["ifeval"]["acc"]["retention"] == pytest.approx(0.75)


def test_zero_float_score_reports_none_not_inf():
    result = compute_retention(
        {"hendrycks_math500": {"exact_match": 0.0}},
        {"hendrycks_math500": {"exact_match": 0.0}},
    )
    assert result["hendrycks_math500"]["exact_match"]["retention"] is None


def test_skips_tasks_missing_on_one_side():
    result = compute_retention(
        {"ifeval": {"acc": 0.8}, "mmlu_pro_biology": {"acc": 0.5}},
        {"ifeval": {"acc": 0.7}},
    )
    assert list(result.keys()) == ["ifeval"]


def test_skips_metrics_missing_on_one_side():
    result = compute_retention(
        {"ifeval": {"acc": 0.8, "extra_metric": 1.0}},
        {"ifeval": {"acc": 0.7}},
    )
    assert list(result["ifeval"].keys()) == ["acc"]


def test_skips_non_numeric_metrics():
    result = compute_retention(
        {"ifeval": {"acc": 0.8, "alias": "ifeval"}},
        {"ifeval": {"acc": 0.7, "alias": "ifeval"}},
    )
    assert list(result["ifeval"].keys()) == ["acc"]


def test_no_overlap_returns_empty():
    result = compute_retention(
        {"ifeval": {"acc": 0.8}}, {"mmlu_pro_biology": {"acc": 0.5}}
    )
    assert result == {}
