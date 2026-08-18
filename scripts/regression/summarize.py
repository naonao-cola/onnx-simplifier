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
import statistics
import sys


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


def _p90(xs):
    if len(xs) < 2:
        return xs[0] if xs else None
    return statistics.quantiles(xs, n=10, method="inclusive")[8]


def comparison_section(rows):
    """Build the onnxsim-vs-onnxslim comparison (robustness / optimization /
    speed). Returns a list of Markdown lines, or [] if no onnxslim data is
    present (older CSVs)."""
    if not any(r.get("slim_status") for r in rows):
        return []

    lines = ["## onnxsim vs onnxslim (comparison, non-gating)\n"]

    # --- Robustness: per-tool status breakdown ------------------------------- #
    order = ["ok", "unvalidated", "timeout", "crash", "error"]
    def counts(key):
        c = {k: 0 for k in order}
        other = 0
        for r in rows:
            s = (r.get(key) or "").strip()
            if s in c:
                c[s] += 1
            elif s:
                other += 1
        return c, other
    sim_c, sim_o = counts("status")
    slim_c, slim_o = counts("slim_status")
    lines.append("### Robustness\n")
    lines.append("| status | onnxsim | onnxslim |")
    lines.append("| --- | --- | --- |")
    for k in order:
        lines.append(f"| {k} | {sim_c[k]} | {slim_c[k]} |")
    if sim_o or slim_o:
        lines.append(f"| other | {sim_o} | {slim_o} |")
    lines.append("")

    # Models where the two tools disagree on whether the graph even survived
    # (one ran, i.e. ok/unvalidated; the other crashed/timed out/errored).
    ran = {"ok", "unvalidated"}
    disagree = []
    for r in rows:
        s, sl = (r.get("status") or ""), (r.get("slim_status") or "")
        if (s in ran) != (sl in ran) and sl:
            disagree.append(r)
    if disagree:
        lines.append("Models where only one tool produced a usable graph:\n")
        lines.append("| model | onnxsim | onnxslim | detail |")
        lines.append("| --- | --- | --- | --- |")
        for r in sorted(disagree, key=lambda r: r["model"]):
            err = (r.get("error") if (r.get("status") or "") not in ran
                   else r.get("slim_error")) or ""
            err = err.replace("|", "\\|")[:120]
            lines.append(f"| {r['model']} | {r.get('status')} | {r.get('slim_status')} | {err} |")
        lines.append("")

    # --- Optimization strength ---------------------------------------------- #
    both = []
    for r in rows:
        on, sn, ssn = _int(r.get("orig_nodes")), _int(r.get("simp_nodes")), _int(r.get("slim_simp_nodes"))
        if on and sn is not None and ssn is not None:
            both.append((r, on, sn, ssn))
    lines.append("### Optimization strength\n")
    if both:
        sim_reds = [100 * (on - sn) / on for _, on, sn, _ in both]
        slim_reds = [100 * (on - ssn) / on for _, on, _, ssn in both]
        slim_fewer = sum(1 for _, _, sn, ssn in both if ssn < sn)
        sim_fewer = sum(1 for _, _, sn, ssn in both if sn < ssn)
        equal = sum(1 for _, _, sn, ssn in both if sn == ssn)
        lines.append(f"Over the {len(both)} models both tools simplified:\n")
        lines.append("| metric | onnxsim | onnxslim |")
        lines.append("| --- | --- | --- |")
        lines.append(f"| median node reduction | {_median(sim_reds):.1f}% | {_median(slim_reds):.1f}% |")
        lines.append("")
        lines.append(f"- onnxslim produced **fewer** nodes on {slim_fewer}, "
                     f"onnxsim fewer on {sim_fewer}, equal on {equal}.")
        lines.append("")
        # Biggest node-count gaps (either direction).
        div = sorted(both, key=lambda t: -abs(t[2] - t[3]))[:15]
        div = [t for t in div if t[2] != t[3]]
        if div:
            lines.append("Largest node-count divergences (orig → onnxsim / onnxslim):\n")
            lines.append("| model | orig | onnxsim | onnxslim | fewer |")
            lines.append("| --- | --- | --- | --- | --- |")
            for r, on, sn, ssn in div:
                fewer = "onnxslim" if ssn < sn else "onnxsim"
                lines.append(f"| {r['model']} | {on} | {sn} | {ssn} | {fewer} |")
            lines.append("")
    else:
        lines.append("_No model was simplified by both tools._\n")

    # --- Speed / memory ------------------------------------------------------ #
    lines.append("### Speed & peak memory\n")
    # Only compare models both tools actually completed, so a timeout/crash on
    # one side is not miscounted as the other side being "faster".
    ran = {"ok", "unvalidated"}
    timed = []
    for r in rows:
        if (r.get("status") or "") in ran and (r.get("slim_status") or "") in ran:
            a, b = _num(r.get("seconds")), _num(r.get("slim_seconds"))
            if a is not None and b is not None:
                timed.append((r, a, b))
    if timed:
        slim_faster = sum(1 for _, a, b in timed if b < a)
        sim_faster = sum(1 for _, a, b in timed if a < b)
        sim_secs = [a for _, a, _ in timed]
        slim_secs = [b for _, _, b in timed]
        lines.append(f"Over the {len(timed)} models both tools completed:\n")
        lines.append("| metric | onnxsim | onnxslim |")
        lines.append("| --- | --- | --- |")
        lines.append(f"| fastest (s) | {min(sim_secs):.1f} | {min(slim_secs):.1f} |")
        lines.append(f"| median time (s) | {_median(sim_secs):.1f} | {_median(slim_secs):.1f} |")
        lines.append(f"| mean time (s) | {statistics.mean(sim_secs):.1f} | {statistics.mean(slim_secs):.1f} |")
        lines.append(f"| p90 time (s) | {_p90(sim_secs):.1f} | {_p90(slim_secs):.1f} |")
        lines.append(f"| slowest (s) | {max(sim_secs):.1f} | {max(slim_secs):.1f} |")
        sim_rss = [_num(r.get("peak_rss_mb")) for r, _, _ in timed if _num(r.get("peak_rss_mb"))]
        slim_rss = [_num(r.get("slim_peak_rss_mb")) for r, _, _ in timed if _num(r.get("slim_peak_rss_mb"))]
        if sim_rss and slim_rss:
            lines.append(f"| median peak RSS (MiB) | {_median(sim_rss):.0f} | {_median(slim_rss):.0f} |")
        lines.append("")
        lines.append(f"- onnxslim was faster on {slim_faster}/{len(timed)} models, "
                     f"onnxsim was faster on {sim_faster}/{len(timed)}, "
                     f"tied on {len(timed) - slim_faster - sim_faster}.")
        lines.append("")

        # Biggest wall-clock gaps (either direction), same spirit as the
        # node-count divergence table above.
        div = sorted(timed, key=lambda t: -abs(t[1] - t[2]))[:15]
        div = [t for t in div if t[1] != t[2]]
        if div:
            lines.append("Largest speed divergences (onnxsim vs onnxslim):\n")
            lines.append("| model | onnxsim (s) | onnxslim (s) | faster |")
            lines.append("| --- | ---: | ---: | --- |")
            for r, a, b in div:
                faster = "onnxslim" if b < a else "onnxsim"
                lines.append(f"| {r['model']} | {a:.1f} | {b:.1f} | {faster} |")
            lines.append("")
    else:
        lines.append("_No model was completed by both tools._\n")

    return lines


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
        "peak_rss_mb",
        "error",
        "slim_status",
        "slim_simp_nodes",
        "slim_reduction_pct",
        "slim_seconds",
        "slim_valid",
        "slim_peak_rss_mb",
        "slim_error",
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

    lines.extend(comparison_section(rows))

    lines.append("## All models (onnxsim)\n")
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

    # The run summary page.
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a") as f:
            f.write(text)

    # A standalone copy the issue-reporting step reads as the issue body.
    with open("regression-summary.md", "w") as f:
        f.write(text)

    # Machine-readable outcome for the workflow (drives the failure issue).
    job_output = os.environ.get("GITHUB_OUTPUT")
    if job_output:
        with open(job_output, "a") as f:
            f.write(f"status={'fail' if bad else 'pass'}\n")
            f.write(f"blocking_failures={len(bad)}\n")

    print(text)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
