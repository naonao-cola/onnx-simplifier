#!/usr/bin/env python3
"""Compute quantized-vs-float retention from two `run_quality_eval.py` result files.

`retention = quantized_score / float_score`, per (task, metric) -- how much of
the float model's benchmark score the quantized (Core ML) model keeps, the
same shape of number DeviceMark's (https://devicemark.github.io/) leaderboard
reports alongside decode speed and memory. See
`bench/TODO_quality_retention_eval.md` for the full plan this is one piece of.

Usage:
    python run_quality_eval.py --model hf --model-args pretrained=... \\
        --tasks ifeval --limit 20 --output float_ifeval.json
    python run_quality_eval.py --model coreml --model-args pretrained=... \\
        --tasks ifeval --limit 20 --output coreml_ifeval.json
    python compute_retention.py float_ifeval.json coreml_ifeval.json
"""

from __future__ import annotations

import argparse
import json
import sys


def _resolve_score(value) -> tuple[float, str] | tuple[None, None]:
    """Extract a plain accuracy number and its basis from one `run_quality_eval.py`
    `--output` metric entry.

    `run_quality_eval.py` now reports each metric as `{"acc": ..., possibly
    "acc_completed": ...}` (see that module's docstring) rather than a bare
    number -- `acc_completed`, when present and defined, is DeviceMark's own
    retention definition (accuracy over only the samples that produced an
    answer before exhausting their generation budget, see
    `bench/TODO_quality_retention_eval.md`'s "What DeviceMark measures"), so
    it's preferred here. Falls back to plain `acc` when `acc_completed` is
    absent (no explicit `--max-gen-toks` was passed to that run) or `None`
    (every sampled item hit the cap -- nothing to average). A bare number is
    also accepted directly (returned as-is, basis `"acc"`) for callers
    (including tests) building this dict by hand rather than via
    `run_quality_eval.py --output`. Anything else (a non-numeric field like an
    internal alias) resolves to `(None, None)`.
    """
    if isinstance(value, (int, float)):
        return value, "acc"
    if isinstance(value, dict):
        completed = value.get("acc_completed")
        if isinstance(completed, (int, float)):
            return completed, "acc_completed"
        acc = value.get("acc")
        if isinstance(acc, (int, float)):
            return acc, "acc"
    return None, None


def compute_retention(float_tasks: dict, quantized_tasks: dict) -> dict:
    """Per-(task, metric) retention for every metric present on both sides.

    `float_tasks`/`quantized_tasks` are `run_quality_eval.py` output's "tasks"
    field: `{task: {metric: value}}`, `value` resolved via `_resolve_score`
    (DeviceMark's completed-only accuracy when available, plain accuracy
    otherwise). A metric present on only one side (a task run with different
    `--tasks` on each side, or a non-numeric field like an internal alias) is
    silently skipped rather than raising -- the two result files come from
    independent runs and aren't guaranteed to line up exactly. A float score
    of exactly 0 makes the ratio undefined (nothing to retain, or the metric
    legitimately can't score 0 by construction and something else broke) --
    reported as `None`, not `inf` or `nan`.
    """
    retention = {}
    for task, float_metrics in float_tasks.items():
        quantized_metrics = quantized_tasks.get(task)
        if not quantized_metrics:
            continue

        task_retention = {}
        for metric, float_value in float_metrics.items():
            if metric not in quantized_metrics:
                continue
            quantized_value = quantized_metrics[metric]
            float_score, float_basis = _resolve_score(float_value)
            quantized_score, quantized_basis = _resolve_score(quantized_value)
            if float_score is None or quantized_score is None:
                continue
            task_retention[metric] = {
                "float": float_score,
                "quantized": quantized_score,
                "retention": (quantized_score / float_score) if float_score else None,
                "float_basis": float_basis,
                "quantized_basis": quantized_basis,
            }
        if task_retention:
            retention[task] = task_retention
    return retention


def build_records(
    model_id: str | None, subset_n: int | None, retention: dict
) -> list[dict]:
    """Flatten `compute_retention`'s `{task: {metric: {...}}}` into one flat
    record per (task, metric): `{model_id, benchmark, metric, subset_n,
    float_acc, quantized_acc, retention, float_basis, quantized_basis}`
    (`*_basis` is `"acc_completed"` or `"acc"` -- see `_resolve_score`).

    The nested shape is convenient to print but awkward to accumulate across
    runs/models into a trend table without re-parsing it; this is the
    machine-readable shape `bench/TODO_quality_retention_eval.md`'s
    "Reporting" section describes, written to `--output` alongside the
    existing nested `retention` payload (not instead of it, so nothing
    reading the old shape breaks).
    """
    records = []
    for task, metrics in retention.items():
        for metric, values in metrics.items():
            records.append(
                {
                    "model_id": model_id,
                    "benchmark": task,
                    "metric": metric,
                    "subset_n": subset_n,
                    "float_acc": values["float"],
                    "quantized_acc": values["quantized"],
                    "retention": values["retention"],
                    "float_basis": values["float_basis"],
                    "quantized_basis": values["quantized_basis"],
                }
            )
    return records


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "float_result", help="JSON file from run_quality_eval.py --model hf"
    )
    ap.add_argument(
        "quantized_result", help="JSON file from run_quality_eval.py --model coreml"
    )
    ap.add_argument(
        "--output", help="Write the JSON summary here (default: stdout only)"
    )
    ap.add_argument(
        "--model-id",
        default=None,
        help="Model id/name to embed in --output's flat 'records' list (default: "
        "the float result file's own 'model_args' field, e.g. "
        "'pretrained=HuggingFaceTB/SmolLM2-135M-Instruct,dtype=float32'). Only "
        "affects --output's JSON, not the printed summary.",
    )
    args = ap.parse_args()

    with open(args.float_result) as f:
        float_result = json.load(f)
    with open(args.quantized_result) as f:
        quantized_result = json.load(f)

    retention = compute_retention(float_result["tasks"], quantized_result["tasks"])
    if not retention:
        print(
            "No task/metric overlap between the two result files -- nothing to "
            "compute retention for.",
            file=sys.stderr,
        )
        return 1

    for task, metrics in retention.items():
        print(f"\n{task}:")
        for metric, values in metrics.items():
            ret = values["retention"]
            ret_str = f"{ret:.1%}" if ret is not None else "N/A (float score was 0)"
            basis_note = ""
            if "acc_completed" in (values["float_basis"], values["quantized_basis"]):
                basis_note = (
                    f" [basis: float={values['float_basis']}, "
                    f"quantized={values['quantized_basis']}]"
                )
            print(
                f"  {metric}: float={values['float']:.4f} "
                f"quantized={values['quantized']:.4f} retention={ret_str}{basis_note}"
            )

    if args.output:
        model_id = args.model_id or float_result.get("model_args")
        subset_n = float_result.get("limit")
        with open(args.output, "w") as f:
            json.dump(
                {
                    "retention": retention,
                    "records": build_records(model_id, subset_n, retention),
                },
                f,
                indent=2,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
