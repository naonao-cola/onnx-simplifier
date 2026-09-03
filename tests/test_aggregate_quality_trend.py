"""Unit tests for scripts/apple/aggregate_quality_trend.py's pure grouping
logic (`build_trend`) -- file loading/CLI plumbing (`_load_records`, `main`)
isn't exercised here, see that module's docstring for how it's meant to be
run (against compute_retention.py --output files from several past
quality-eval-macos runs).
"""

import os
import sys

_APPLE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "apple"
)
if _APPLE_DIR not in sys.path:
    sys.path.insert(0, _APPLE_DIR)

from aggregate_quality_trend import build_trend  # noqa: E402


def test_groups_by_model_benchmark_metric():
    records = [
        {
            "model_id": "m1",
            "benchmark": "ifeval",
            "metric": "acc",
            "retention": 0.8,
            "source": "run1.json",
        },
        {
            "model_id": "m1",
            "benchmark": "ifeval",
            "metric": "acc",
            "retention": 0.9,
            "source": "run2.json",
        },
        {
            "model_id": "m1",
            "benchmark": "hendrycks_math500",
            "metric": "exact_match",
            "retention": 0.5,
            "source": "run1.json",
        },
    ]
    trend = build_trend(records)
    assert set(trend.keys()) == {
        ("m1", "ifeval", "acc"),
        ("m1", "hendrycks_math500", "exact_match"),
    }
    assert [r["source"] for r in trend[("m1", "ifeval", "acc")]] == [
        "run1.json",
        "run2.json",
    ]


def test_preserves_input_order_within_group():
    records = [
        {
            "model_id": "m",
            "benchmark": "b",
            "metric": "x",
            "retention": 0.1,
            "source": "a",
        },
        {
            "model_id": "m",
            "benchmark": "b",
            "metric": "x",
            "retention": 0.2,
            "source": "b",
        },
        {
            "model_id": "m",
            "benchmark": "b",
            "metric": "x",
            "retention": 0.3,
            "source": "c",
        },
    ]
    trend = build_trend(records)
    assert [r["retention"] for r in trend[("m", "b", "x")]] == [0.1, 0.2, 0.3]


def test_empty_records_returns_empty_dict():
    assert build_trend([]) == {}


def test_different_model_ids_kept_separate():
    records = [
        {
            "model_id": "m1",
            "benchmark": "ifeval",
            "metric": "acc",
            "retention": 0.8,
            "source": "a",
        },
        {
            "model_id": "m2",
            "benchmark": "ifeval",
            "metric": "acc",
            "retention": 0.7,
            "source": "b",
        },
    ]
    trend = build_trend(records)
    assert set(trend.keys()) == {("m1", "ifeval", "acc"), ("m2", "ifeval", "acc")}
