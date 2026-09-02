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


def compute_retention(float_tasks: dict, quantized_tasks: dict) -> dict:
    """Per-(task, metric) retention for every metric present on both sides.

    `float_tasks`/`quantized_tasks` are `run_quality_eval.py` output's "tasks"
    field: `{task: {metric: value}}`. A metric present on only one side (a
    task run with different `--tasks` on each side, or a non-numeric field
    like an internal alias) is silently skipped rather than raising -- the
    two result files come from independent runs and aren't guaranteed to
    line up exactly. A float score of exactly 0 makes the ratio undefined
    (nothing to retain, or the metric legitimately can't score 0 by
    construction and something else broke) -- reported as `None`, not `inf`
    or `nan`.
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
            if not isinstance(float_value, (int, float)) or not isinstance(
                quantized_value, (int, float)
            ):
                continue
            task_retention[metric] = {
                "float": float_value,
                "quantized": quantized_value,
                "retention": (quantized_value / float_value) if float_value else None,
            }
        if task_retention:
            retention[task] = task_retention
    return retention


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
            print(
                f"  {metric}: float={values['float']:.4f} "
                f"quantized={values['quantized']:.4f} retention={ret_str}"
            )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(retention, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
