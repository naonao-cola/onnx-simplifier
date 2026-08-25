#!/usr/bin/env python3
"""Summarize per-model ONNXSIM_PROFILE traces from the model-regression set.

Each trace is a Chrome trace written by onnxsim's C++ profiler (see
``onnxsim/profiler.cpp``): one ``ph:"X"`` / ``cat:"fixed_point"`` event per
fixed-point-function call (``Simplify``, ``Pipeline``, ``OptAndShape``,
``FoldConstant``, ``Optimize``, ``InferShapes``, ``OrtSession``, ...), each
carrying its duration and ``args.peak_rss_mb``. This script aggregates those
per model (mirroring the console table ``PrintSummary`` prints, but across the
whole sampled set at once) and, when given the regression CSVs alongside the
traces, reports how much of each model's *total* wall time the profiler's own
spans actually cover -- see ``bench/RESULTS_profiling_survey.md`` for why that
gap matters: for a large model, most of the time can be Python<->C++
marshalling that happens outside the profiled ``Simplify()`` call entirely.

Usage:
    # profile-summary.md from profiles/*.json
    summarize_profiles.py profiles/*.json

    # + the coverage-gap column, cross-referencing the regression CSVs'
    # wall-clock `seconds` column for the same models
    summarize_profiles.py profiles/*.json --csv "shard-*.csv" slow.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

# The span names PrintSummary (and this script) know about, in the order
# they nest -- purely for a stable, readable column/row order; any other span
# name found in a trace is still aggregated and shown, just after these.
KNOWN_SPANS = (
    "Simplify",
    "Pipeline",
    "OptAndShape",
    "FoldConstant",
    "Fingerprint",
    "Optimize",
    "InferShapes",
    "Rewrite",
    "OrtSession",
    "OrtSessionInit",
    "OrtSessionRun",
)

# Spans that just wrap other profiled spans and so trivially contain ~all of
# their nested children's time -- excluded from "dominant span" detection,
# which is meant to surface the actual leaf-level cost (e.g. Optimize vs.
# FoldConstant vs. InferShapes), not restate "most time is in the fixed-point
# loop" via whichever wrapper happens to be widest.
WRAPPER_SPANS = frozenset({"Simplify", "Pipeline", "OptAndShape", "Fingerprint"})


def _model_name(path):
    # worker.py writes <model_id with "/" -> "__">.json.
    return os.path.splitext(os.path.basename(path))[0].replace("__", "/", 1)


def load_trace(path):
    """Aggregate one profile JSON's fixed-point spans by name.

    Returns ``(spans, root)`` where ``spans`` maps span name -> aggregate dict
    (``calls``, ``wall_ms``, ``cpu_ms``, ``max_wall_ms``, ``peak_mib``,
    ``min_depth``), and ``root`` is the outermost span's aggregate (the one
    with ``min_depth == 0``, normally ``Simplify``), or ``None`` if the trace
    has no fixed-point events (e.g. a crash before any span completed).
    """
    with open(path) as f:
        trace = json.load(f)
    if not isinstance(trace, dict):
        raise ValueError("not a Chrome trace object (expected a JSON object)")
    spans = {}
    for ev in trace.get("traceEvents", []):
        if ev.get("ph") != "X" or ev.get("cat") != "fixed_point":
            continue
        name = ev.get("name", "?")
        args = ev.get("args", {}) or {}
        a = spans.setdefault(
            name,
            {"calls": 0, "wall_ms": 0.0, "cpu_ms": 0.0, "max_wall_ms": 0.0,
             "peak_mib": 0.0, "min_depth": 1 << 30},
        )
        wall_ms = ev.get("dur", 0) / 1000.0
        a["calls"] += 1
        a["wall_ms"] += wall_ms
        a["cpu_ms"] += args.get("cpu_ms", 0.0) or 0.0
        a["max_wall_ms"] = max(a["max_wall_ms"], wall_ms)
        a["peak_mib"] = max(a["peak_mib"], args.get("peak_rss_mb", 0.0) or 0.0)
        a["min_depth"] = min(a["min_depth"], args.get("depth", 0))

    root = None
    for a in spans.values():
        if a["min_depth"] == 0 and (root is None or a["wall_ms"] > root["wall_ms"]):
            root = a
    return spans, root


def load_seconds_by_model(csv_globs):
    """model -> wall-clock ``seconds`` from the regression CSVs, for the
    profiled-span coverage-gap column. Best-effort: a missing/unparseable
    column just leaves that model without a coverage figure."""
    seconds = {}
    for pat in csv_globs:
        for p in sorted(glob.glob(pat)):
            with open(p, newline="") as f:
                for row in csv.DictReader(f):
                    try:
                        seconds[row["model"]] = float(row["seconds"])
                    except (KeyError, TypeError, ValueError):
                        continue
    return seconds


def fmt_span_table(spans, indent="  "):
    order = [n for n in KNOWN_SPANS if n in spans] + sorted(
        n for n in spans if n not in KNOWN_SPANS
    )
    lines = [f"{indent}| span | calls | wall (ms) | cpu (ms) | max wall (ms) | peak (MiB) |",
              f"{indent}| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name in order:
        a = spans[name]
        lines.append(
            f"{indent}| {name} | {a['calls']} | {a['wall_ms']:.1f} | {a['cpu_ms']:.1f} "
            f"| {a['max_wall_ms']:.1f} | {a['peak_mib']:.1f} |"
        )
    return lines


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("json_globs", nargs="*", default=["*.json"],
                     help="glob pattern(s) for ONNXSIM_PROFILE trace files")
    ap.add_argument(
        "--csv", nargs="*", default=[], metavar="CSV_GLOB",
        help="regression CSV glob(s) (needs a `model` and `seconds` column) "
        "to report each model's profiled-span coverage gap against its real "
        "wall-clock time",
    )
    ap.add_argument("--output", default="profile-summary.md")
    ap.add_argument("--top", type=int, default=15,
                     help="how many of the slowest models to show a full "
                     "per-span breakdown for")
    args = ap.parse_args(argv[1:])

    paths = sorted({p for pat in args.json_globs for p in glob.glob(pat)})
    models = []
    unparseable = []
    for p in paths:
        try:
            spans, root = load_trace(p)
        except (OSError, ValueError) as e:
            unparseable.append((p, str(e)))
            continue
        if root is None:
            unparseable.append((p, "no fixed-point spans in trace"))
            continue
        models.append({
            "model": _model_name(p),
            "path": p,
            "spans": spans,
            "root": root,
        })
    models.sort(key=lambda m: -m["root"]["wall_ms"])

    seconds_by_model = load_seconds_by_model(args.csv) if args.csv else {}

    lines = ["# Model-regression profiling summary\n"]
    lines.append(
        f"**{len(models)} model trace(s) parsed**"
        + (f", {len(unparseable)} unparseable/empty" if unparseable else "")
        + ".\n"
    )

    # --- Overview: one row per model, cheapest way to spot an outlier ------- #
    lines.append("## Overview (slowest first)\n")
    header = ["model", "root wall (s)", "peak RSS (MiB)", "dominant span", "dominant %"]
    if seconds_by_model:
        header.insert(2, "actual wall (s)")
        header.insert(3, "coverage gap")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" if h == "model" else "---:" for h in header) + " |")
    for m in models:
        root = m["root"]
        # The "dominant span" is meant to name the actual leaf-level cost
        # (Optimize, FoldConstant, InferShapes, OrtSession, ...), so exclude
        # wrapper spans that just contain ~all of some child's time and would
        # otherwise trivially "win" without saying anything new.
        leaves = {n: a for n, a in m["spans"].items() if n not in WRAPPER_SPANS}
        candidates = leaves or {n: a for n, a in m["spans"].items() if a is not root}
        dominant_name, dominant = (
            max(candidates.items(), key=lambda kv: kv[1]["wall_ms"])
            if candidates else (None, None)
        )
        dominant_pct = f"{100 * dominant['wall_ms'] / root['wall_ms']:.0f}%" if dominant and root["wall_ms"] else "n/a"
        root_wall_s = f"{root['wall_ms'] / 1000:.2f}"
        peak = f"{root['peak_mib']:.0f}"
        row = [m["model"], root_wall_s, peak, dominant_name or "n/a", dominant_pct]
        if seconds_by_model:
            actual = seconds_by_model.get(m["model"])
            actual_s = f"{actual:.2f}" if actual else "n/a"
            gap = f"{actual * 1000 / root['wall_ms']:.1f}x" if actual and root["wall_ms"] else "n/a"
            row = [m["model"], root_wall_s, actual_s, gap, peak, dominant_name or "n/a", dominant_pct]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    if seconds_by_model:
        lines.append(
            "_\"coverage gap\" = actual wall-clock (from the regression CSV) divided by "
            "the profiler's own root-span time -- close to 1x means the profiler "
            "accounts for essentially all of the model's time; much larger means most "
            "time is spent outside the profiled `Simplify()` call (Python<->C++ "
            "marshalling, `onnx.checker.check_model`, ...) -- see "
            "`bench/RESULTS_profiling_survey.md`._\n"
        )

    # --- Aggregate: which span dominates across the whole sampled set ------- #
    # Same reasoning as the per-model "dominant span" above: exclude wrapper
    # spans, or e.g. OptAndShape (which contains ~all of Optimize's time)
    # would double-count against it and the percentages below wouldn't mean
    # what they say.
    totals = {}
    for m in models:
        for name, a in m["spans"].items():
            if name in WRAPPER_SPANS:
                continue
            totals[name] = totals.get(name, 0.0) + a["wall_ms"]
    grand_total = sum(m["root"]["wall_ms"] for m in models) or 1.0
    if totals:
        lines.append("## Where time goes, aggregated across all sampled models\n")
        lines.append("| span | total wall (s) | % of summed root wall time |")
        lines.append("| --- | ---: | ---: |")
        for name, total_ms in sorted(totals.items(), key=lambda kv: -kv[1]):
            lines.append(f"| {name} | {total_ms / 1000:.2f} | {100 * total_ms / grand_total:.1f}% |")
        lines.append("")
        lines.append(
            "_Percentages are of the summed **profiled** root-span time across all "
            "models, not of their real wall-clock time -- see the coverage-gap column "
            "above for how much of the latter this table actually accounts for._\n"
        )

    # --- Per-model detail for the slowest N ---------------------------------- #
    top = models[: args.top]
    if top:
        lines.append(f"## Per-span breakdown, {len(top)} slowest model(s)\n")
        for m in top:
            lines.append(f"### {m['model']} ({m['root']['wall_ms'] / 1000:.2f}s)\n")
            lines.extend(fmt_span_table(m["spans"], indent=""))
            lines.append("")

    if unparseable:
        lines.append("## Unparseable / empty traces\n")
        lines.append("| file | reason |")
        lines.append("| --- | --- |")
        for p, reason in unparseable:
            lines.append(f"| {p} | {reason} |")
        lines.append("")

    text = "\n".join(lines)

    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a") as f:
            f.write(text)

    with open(args.output, "w") as f:
        f.write(text)

    print(text)
    return 0  # informational only -- never gates the job


if __name__ == "__main__":
    sys.exit(main(sys.argv))
