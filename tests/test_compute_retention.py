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

from compute_retention import build_records, compute_retention  # noqa: E402


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
                "float_basis": "acc",
                "quantized_basis": "acc",
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


def test_build_records_flattens_one_entry_per_task_metric():
    retention = compute_retention(
        {"ifeval": {"acc": 0.8}, "hendrycks_math500": {"exact_match": 0.5}},
        {"ifeval": {"acc": 0.7}, "hendrycks_math500": {"exact_match": 0.4}},
    )
    records = build_records("smollm2-135m", 10, retention)
    assert len(records) == 2
    assert {r["benchmark"] for r in records} == {"ifeval", "hendrycks_math500"}
    ifeval_record = next(r for r in records if r["benchmark"] == "ifeval")
    assert ifeval_record == {
        "model_id": "smollm2-135m",
        "benchmark": "ifeval",
        "metric": "acc",
        "subset_n": 10,
        "float_acc": 0.8,
        "quantized_acc": 0.7,
        "retention": pytest.approx(0.875),
        "float_basis": "acc",
        "quantized_basis": "acc",
    }


def test_build_records_on_empty_retention_returns_empty_list():
    assert build_records("model", 10, {}) == []


def test_prefers_acc_completed_over_acc_when_both_present():
    # run_quality_eval.py's --max-gen-toks-aware output shape: acc_completed is
    # DeviceMark's own retention definition (see compute_retention.py's
    # _resolve_score), so it should win over the plain no-answer-counts-as-wrong
    # acc even though both are present.
    result = compute_retention(
        {
            "ifeval": {
                "acc": {
                    "acc": 0.5,
                    "completed_n": 4,
                    "total_n": 10,
                    "acc_completed": 0.8,
                }
            }
        },
        {
            "ifeval": {
                "acc": {
                    "acc": 0.4,
                    "completed_n": 6,
                    "total_n": 10,
                    "acc_completed": 0.6,
                }
            }
        },
    )
    entry = result["ifeval"]["acc"]
    assert entry["float"] == 0.8
    assert entry["quantized"] == 0.6
    assert entry["retention"] == pytest.approx(0.75)
    assert entry["float_basis"] == "acc_completed"
    assert entry["quantized_basis"] == "acc_completed"


def test_falls_back_to_acc_when_acc_completed_is_none():
    # Every sampled item hit the generation cap -- acc_completed has nothing
    # to average, so it's None; falls back to plain acc rather than dropping
    # the metric.
    result = compute_retention(
        {
            "ifeval": {
                "acc": {
                    "acc": 0.5,
                    "completed_n": 0,
                    "total_n": 10,
                    "acc_completed": None,
                }
            }
        },
        {
            "ifeval": {
                "acc": {
                    "acc": 0.3,
                    "completed_n": 0,
                    "total_n": 10,
                    "acc_completed": None,
                }
            }
        },
    )
    entry = result["ifeval"]["acc"]
    assert entry["float"] == 0.5
    assert entry["quantized"] == 0.3
    assert entry["float_basis"] == "acc"
    assert entry["quantized_basis"] == "acc"


def test_falls_back_to_acc_when_acc_completed_key_absent():
    # No explicit --max-gen-toks was passed to run_quality_eval.py, so
    # completed-only accounting was never attempted -- only "acc" is present.
    result = compute_retention(
        {"ifeval": {"acc": {"acc": 0.5}}},
        {"ifeval": {"acc": {"acc": 0.4}}},
    )
    entry = result["ifeval"]["acc"]
    assert entry["float"] == 0.5
    assert entry["float_basis"] == "acc"


def test_basis_tracked_independently_per_side():
    # float side has a known --max-gen-toks (acc_completed available),
    # quantized side doesn't -- each side should resolve independently.
    result = compute_retention(
        {"ifeval": {"acc": {"acc": 0.5, "acc_completed": 0.8}}},
        {"ifeval": {"acc": {"acc": 0.4}}},
    )
    entry = result["ifeval"]["acc"]
    assert entry["float"] == 0.8
    assert entry["float_basis"] == "acc_completed"
    assert entry["quantized"] == 0.4
    assert entry["quantized_basis"] == "acc"
