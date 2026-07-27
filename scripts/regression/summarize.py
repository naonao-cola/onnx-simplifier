#!/usr/bin/env python3
"""Combine per-shard regression CSVs into a Markdown summary.

Writes a merged ``regression-report.csv`` and a Markdown table to the path in
``$GITHUB_STEP_SUMMARY`` (or stdout when run locally). Exits non-zero if any
model has a non-ok status, so the summary job reflects the overall result even
though each shard already gates itself.
"""

from __future__ import annotations

import csv
import glob
import os
import sys


def main(argv):
    csv_paths = []
    for pat in argv[1:] or ["*.csv"]:
        csv_paths.extend(sorted(glob.glob(pat)))
    rows = []
    for p in csv_paths:
        if os.path.basename(p) == "regression-report.csv":
            continue
        # Rows from the non-blocking "slow" job are reported but never gate the
        # overall result (those models are known to exceed the standard cap).
        nonblocking = "slow" in os.path.basename(p).lower()
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                r["_nonblocking"] = nonblocking
                rows.append(r)

    rows.sort(key=lambda r: r["model"])
    fields = [
        "model",
        "status",
        "orig_nodes",
        "simp_nodes",
        "baseline_simp_nodes",
        "reduction_pct",
        "seconds",
        "skipped_optimizers",
        "valid",
        "error",
    ]
    with open("regression-report.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r["status"] == "ok"]
    # Only blocking rows count toward pass/fail; slow-job rows are informational.
    bad = [r for r in rows if r["status"] != "ok" and not r["_nonblocking"]]
    slow_notok = [r for r in rows if r["status"] != "ok" and r["_nonblocking"]]
    skipped_any = [r for r in rows if r.get("skipped_optimizers")]

    lines = []
    lines.append("# onnxsim large-model regression\n")
    lines.append(f"**{len(ok)}/{len(rows)} models passed** "
                 f"({len(bad)} blocking failures, {len(slow_notok)} known-slow not-ok, "
                 f"{len(skipped_any)} needed a pass skipped)\n")

    if bad:
        lines.append("## ❌ Failures (blocking)\n")
        lines.append("| model | status | error |")
        lines.append("| --- | --- | --- |")
        for r in bad:
            err = (r.get("error") or "").replace("|", "\\|")[:160]
            lines.append(f"| {r['model']} | {r['status']} | {err} |")
        lines.append("")

    if slow_notok:
        lines.append("## ⏱️ Known-slow models not completing (non-blocking)\n")
        lines.append("| model | status | error |")
        lines.append("| --- | --- | --- |")
        for r in slow_notok:
            err = (r.get("error") or "").replace("|", "\\|")[:160]
            lines.append(f"| {r['model']} | {r['status']} | {err} |")
        lines.append("")

    if skipped_any:
        lines.append("## ⚠️ Optimizer passes skipped (would otherwise abort)\n")
        lines.append("| model | skipped pass(es) |")
        lines.append("| --- | --- |")
        for r in skipped_any:
            lines.append(f"| {r['model']} | {r['skipped_optimizers']} |")
        lines.append("")

    lines.append("## All models\n")
    lines.append("| model | status | nodes (orig→simp) | baseline simp | Δ | time (s) |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for r in rows:
        on, sn = r.get("orig_nodes", ""), r.get("simp_nodes", "")
        base = r.get("baseline_simp_nodes") or ""
        delta = ""
        try:
            if sn != "" and base != "":
                d = int(sn) - int(base)
                delta = f"{d:+d}" if d else "0"
        except ValueError:
            pass
        mark = "✅" if r["status"] == "ok" else "❌"
        lines.append(
            f"| {r['model']} | {mark} {r['status']} | {on}→{sn} | {base} | {delta} | {r.get('seconds', '')} |"
        )
    lines.append("")

    text = "\n".join(lines)
    out = os.environ.get("GITHUB_STEP_SUMMARY")
    if out:
        with open(out, "a") as f:
            f.write(text)
    print(text)

    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
