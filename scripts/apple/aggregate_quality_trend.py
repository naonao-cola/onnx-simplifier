#!/usr/bin/env python3
"""Aggregate `compute_retention.py --output` files from multiple runs into a trend.

`compute_retention.py --output` writes a flat `"records"` list (one object per
(task, metric): `{model_id, benchmark, metric, subset_n, float_acc,
quantized_acc, retention, float_basis, quantized_basis}`, see that module's
`build_records()`) alongside its nested summary -- `quality-eval-macos`
uploads these as the `quality-retention-results` CI artifact on every run,
see `bench/TODO_quality_retention_eval.md`'s "Reporting" section. This script
is the "next piece" that doc flagged: read several of those files (e.g.
downloaded from a handful of past `quality-eval-macos` runs via `gh run
download` or the Actions UI -- fetching that history automatically isn't
wired into CI itself, see the module docstring's last paragraph) and group
them by `(model_id, benchmark, metric)` so a run's retention/accuracy numbers
can be read as a trend instead of one isolated data point per run.

Not wired into CI: `quality-eval-macos` runs a single fixed model
(`HuggingFaceTB/SmolLM2-135M-Instruct`) with no run-history persistence
today, so there's nothing to aggregate automatically yet -- this script is
meant to be run by hand (or from a separate follow-up job) against
artifacts downloaded from several past runs. Records carry no timestamp;
the order files are passed in on the command line is treated as
chronological (oldest first) -- pass them in that order.

Usage:
    python aggregate_quality_trend.py retention_ifeval_run1.json \\
        retention_ifeval_run2.json retention_math500_run1.json \\
        retention_math500_run2.json --output trend.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_trend(records: list[dict]) -> dict[tuple, list[dict]]:
    """Group flat records (`compute_retention.py`'s `build_records()` shape,
    each expected to carry a `"source"` field identifying which run/file it
    came from -- see `_load_records`) by `(model_id, benchmark, metric)`,
    preserving input order. Callers are responsible for passing `records` in
    chronological order (oldest first) across however many input files they
    concatenate -- there's no timestamp field to sort by independently.
    """
    trend: dict[tuple, list[dict]] = {}
    for record in records:
        key = (record.get("model_id"), record["benchmark"], record["metric"])
        trend.setdefault(key, []).append(record)
    return trend


def _load_records(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    records = data.get("records")
    if records is None:
        raise ValueError(
            f"{path}: no 'records' field -- pass a compute_retention.py --output "
            "file (needs the flat 'records' list build_records() writes, not just "
            "the nested 'retention' summary)"
        )
    source = Path(path).name
    return [{**record, "source": source} for record in records]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "files",
        nargs="+",
        help="compute_retention.py --output JSON files, oldest run first",
    )
    ap.add_argument(
        "--output",
        help="Write the grouped trend as JSON here (default: printed summary only)",
    )
    args = ap.parse_args()

    all_records = []
    for path in args.files:
        all_records.extend(_load_records(path))

    trend = build_trend(all_records)
    if not trend:
        print("No records found across the given files.", file=sys.stderr)
        return 1

    for (model_id, benchmark, metric), entries in trend.items():
        print(f"\n{model_id or '?'} / {benchmark} / {metric}:")
        for entry in entries:
            ret = entry["retention"]
            ret_str = f"{ret:.1%}" if ret is not None else "N/A"
            print(
                f"  {entry['source']}: retention={ret_str} "
                f"(float={entry['float_acc']:.4f} [{entry['float_basis']}], "
                f"quantized={entry['quantized_acc']:.4f} [{entry['quantized_basis']}], "
                f"n={entry['subset_n']})"
            )
        defined = [e["retention"] for e in entries if e["retention"] is not None]
        if len(defined) >= 2:
            print(
                f"  change ({entries[0]['source']} -> {entries[-1]['source']}, "
                f"first/last defined retention): {defined[-1] - defined[0]:+.1%}"
            )

    if args.output:
        json_trend = [
            {
                "model_id": model_id,
                "benchmark": benchmark,
                "metric": metric,
                "runs": entries,
            }
            for (model_id, benchmark, metric), entries in trend.items()
        ]
        with open(args.output, "w") as f:
            json.dump(json_trend, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
