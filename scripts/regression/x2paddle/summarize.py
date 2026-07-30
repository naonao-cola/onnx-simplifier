#!/usr/bin/env python3
"""Combine per-shard X2Paddle regression CSVs into a Markdown summary.

Writes a merged ``x2paddle-regression-report.csv`` and a Markdown summary to
``$GITHUB_STEP_SUMMARY`` (or stdout when run locally), plus a standalone
``x2paddle-regression-summary.md``. Exits non-zero if any model has a gating
verdict (``onnxsim_fail`` or ``regression``), so the summary job reflects the
overall result even though each shard already gates itself.

Verdicts
--------
pass                  onnxsim ok and X2Paddle converted the simplified graph.
regression   (gates)  X2Paddle converted the original but not the simplified
                      graph -- onnxsim broke a working downstream conversion.
onnxsim_fail (gates)  onnxsim crashed / timed out / failed its own check.
baseline_unsupported  X2Paddle can't convert the original either (its own
                      limitation on that model); not onnxsim's fault.
improved              original failed, simplified converted (onnxsim unblocked
                      X2Paddle).
"""

from __future__ import annotations

import csv
import glob
import os
import statistics
import sys

GATING = ("onnxsim_fail", "regression")


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _median(xs):
    return statistics.median(xs) if xs else None


def main(argv):
    csv_paths = []
    for pat in argv[1:] or ["*.csv"]:
        csv_paths.extend(sorted(glob.glob(pat)))
    rows = []
    for p in csv_paths:
        if os.path.basename(p) in ("x2paddle-regression-report.csv",):
            continue
        with open(p, newline="") as f:
            rows.extend(csv.DictReader(f))
    rows.sort(key=lambda r: r["model"])

    fields = [
        "model",
        "verdict",
        "status",
        "orig_nodes",
        "simp_nodes",
        "reduction_pct",
        "onnxsim_status",
        "onnxsim_valid",
        "onnxsim_seconds",
        "skipped_optimizers",
        "baseline_conv_status",
        "baseline_ops",
        "baseline_conv_seconds",
        "baseline_error",
        "simp_conv_status",
        "simp_ops",
        "simp_conv_seconds",
        "simp_error",
        "error",
    ]
    with open("x2paddle-regression-report.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    def with_verdict(v):
        return [r for r in rows if r.get("verdict") == v]

    passed = with_verdict("pass")
    regressions = with_verdict("regression")
    onnxsim_fails = with_verdict("onnxsim_fail")
    unsupported = with_verdict("baseline_unsupported")
    improved = with_verdict("improved")
    errored = [r for r in rows if r.get("verdict") in ("error", None, "")]
    skipped_any = [r for r in rows if r.get("skipped_optimizers")]
    gating = [r for r in rows if r.get("verdict") in GATING] + errored

    lines = []
    lines.append("# onnxsim → X2Paddle downstream regression\n")
    lines.append(
        "onnxsim is X2Paddle's built-in ONNX optimize step, so this checks that "
        "onnxsim doesn't break X2Paddle's ONNX→PaddlePaddle conversion. Each model "
        "is converted twice — from the original graph and after `onnxsim.simplify` "
        "— each conversion isolated in its own process.\n"
    )
    lines.append(
        f"**{len(passed)}/{len(rows)} models pass.** "
        f"{len(regressions)} regression(s), {len(onnxsim_fails)} onnxsim failure(s), "
        f"{len(unsupported)} X2Paddle-unsupported (non-gating), "
        f"{len(improved)} improved, {len(errored)} harness error(s).\n"
    )

    if regressions:
        lines.append(
            "## ❌ Regressions — onnxsim broke a working X2Paddle conversion\n"
        )
        lines.append("| model | orig nodes → simp | X2Paddle (base → simp) | error |")
        lines.append("| --- | --- | --- | --- |")
        for r in regressions:
            err = (r.get("simp_error") or "").replace("|", "\\|")[:160]
            lines.append(
                f"| {r['model']} | {r.get('orig_nodes')}→{r.get('simp_nodes')} | "
                f"{r.get('baseline_conv_status')}→{r.get('simp_conv_status')} | {err} |"
            )
        lines.append("")

    if onnxsim_fails:
        lines.append("## ❌ onnxsim failures (crash / timeout / unvalidated)\n")
        lines.append("| model | onnxsim status | error |")
        lines.append("| --- | --- | --- |")
        for r in onnxsim_fails:
            err = (r.get("error") or "").replace("|", "\\|")[:160]
            lines.append(f"| {r['model']} | {r.get('onnxsim_status')} | {err} |")
        lines.append("")

    if errored:
        lines.append("## ❌ Harness errors\n")
        lines.append("| model | error |")
        lines.append("| --- | --- |")
        for r in errored:
            err = (r.get("error") or "").replace("|", "\\|")[:160]
            lines.append(f"| {r['model']} | {err} |")
        lines.append("")

    if skipped_any:
        lines.append("## ⚠️ Optimizer passes skipped (would otherwise abort)\n")
        lines.append("| model | skipped pass(es) |")
        lines.append("| --- | --- |")
        for r in skipped_any:
            lines.append(f"| {r['model']} | {r['skipped_optimizers']} |")
        lines.append("")

    if unsupported:
        lines.append("## X2Paddle-unsupported originals (non-gating)\n")
        lines.append(
            "X2Paddle can't convert these originals either, so onnxsim is "
            "not implicated. Recorded for coverage.\n"
        )
        lines.append("| model | onnxsim | X2Paddle error (original) |")
        lines.append("| --- | --- | --- |")
        for r in unsupported:
            err = (r.get("baseline_error") or "").replace("|", "\\|")[:160]
            lines.append(
                f"| {r['model']} | {r.get('onnxsim_status')} {r.get('orig_nodes')}→{r.get('simp_nodes')} | {err} |"
            )
        lines.append("")

    if improved:
        lines.append("## onnxsim unblocked X2Paddle (non-gating)\n")
        lines.append("| model | X2Paddle (base → simp) |")
        lines.append("| --- | --- |")
        for r in improved:
            lines.append(
                f"| {r['model']} | {r.get('baseline_conv_status')}→{r.get('simp_conv_status')} |"
            )
        lines.append("")

    # onnxsim node reduction over the models it simplified.
    reds = [
        _num(r.get("reduction_pct"))
        for r in rows
        if _num(r.get("reduction_pct")) is not None
    ]
    if reds:
        lines.append("## onnxsim optimization on this set\n")
        lines.append(
            f"- median node reduction: **{_median(reds):.1f}%** "
            f"over {len(reds)} simplified models."
        )
        secs = [
            _num(r.get("onnxsim_seconds"))
            for r in rows
            if _num(r.get("onnxsim_seconds"))
        ]
        if secs:
            lines.append(f"- median onnxsim time: {_median(secs):.1f}s.")
        lines.append("")

    lines.append("## All models\n")
    lines.append(
        "| model | verdict | onnxsim (orig→simp) | X2Paddle base | X2Paddle simp |"
    )
    lines.append("| --- | --- | --- | --- | --- |")
    mark = {
        "pass": "✅",
        "improved": "✅",
        "baseline_unsupported": "➖",
        "regression": "❌",
        "onnxsim_fail": "❌",
    }
    for r in rows:
        m = mark.get(r.get("verdict"), "❓")
        lines.append(
            f"| {r['model']} | {m} {r.get('verdict')} | "
            f"{r.get('orig_nodes')}→{r.get('simp_nodes')} ({r.get('onnxsim_status')}) | "
            f"{r.get('baseline_conv_status')} ({r.get('baseline_ops')}) | "
            f"{r.get('simp_conv_status')} ({r.get('simp_ops')}) |"
        )
    lines.append("")

    text = "\n".join(lines)

    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a") as f:
            f.write(text)
    with open("x2paddle-regression-summary.md", "w") as f:
        f.write(text)

    job_output = os.environ.get("GITHUB_OUTPUT")
    if job_output:
        with open(job_output, "a") as f:
            f.write(f"status={'fail' if gating else 'pass'}\n")
            f.write(f"gating_failures={len(gating)}\n")

    print(text)
    return 1 if gating else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
