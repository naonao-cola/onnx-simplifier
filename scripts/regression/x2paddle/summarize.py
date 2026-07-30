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


def comparison_section(rows):
    """onnxsim vs onnxslim, both as downstream inputs to X2Paddle.

    Two axes, both comparison-only (onnxslim never gates this harness):
      - X2Paddle-convertibility: over the models X2Paddle converted from the
        original graph, how often each simplifier's output still converts. This
        is the axis that matters here — a simplifier that reduces more nodes but
        breaks X2Paddle is worse for this downstream, not better.
      - Node reduction: median reduction each simplifier achieved, and where the
        two diverge most.
    Returns [] when no onnxslim data is present (older CSVs).
    """
    if not any(r.get("onnxslim_status") for r in rows):
        return []

    lines = ["## onnxsim vs onnxslim (comparison, non-gating)\n"]

    # --- Robustness: does the simplified graph still convert? ---------------- #
    order = ["ok", "unvalidated", "crash", "timeout", "error"]

    def counts(key):
        c = {k: 0 for k in order}
        for r in rows:
            s = (r.get(key) or "").strip()
            if s in c:
                c[s] += 1
        return c

    sim = counts("simp_conv_status")
    slim = counts("slim_conv_status")
    lines.append("### X2Paddle conversion of the simplified graph\n")
    lines.append("| X2Paddle status | after onnxsim | after onnxslim |")
    lines.append("| --- | --- | --- |")
    for k in order:
        if sim[k] or slim[k]:
            lines.append(f"| {k} | {sim[k]} | {slim[k]} |")
    lines.append("")

    # Where the two simplifiers diverge on whether X2Paddle can convert the
    # result, among models X2Paddle converted from the original.
    ran = {"ok"}
    divergent = []
    for r in rows:
        if (r.get("baseline_conv_status") or "") not in ran:
            continue
        s = (r.get("simp_conv_status") or "") in ran
        sl = (r.get("slim_conv_status") or "") in ran
        if s != sl:
            divergent.append(r)
    if divergent:
        lines.append(
            "Models where only one simplifier's graph still converts (baseline "
            "converted):\n"
        )
        lines.append("| model | after onnxsim | after onnxslim |")
        lines.append("| --- | --- | --- |")
        for r in sorted(divergent, key=lambda r: r["model"]):
            lines.append(
                f"| {r['model']} | {r.get('simp_conv_status')} | "
                f"{r.get('slim_conv_status')} |"
            )
        lines.append("")

    # --- Optimization strength ---------------------------------------------- #
    both = []
    for r in rows:
        on = _int(r.get("orig_nodes"))
        sn = _int(r.get("simp_nodes"))
        sln = _int(r.get("slim_nodes"))
        if on and sn is not None and sln is not None:
            both.append((r, on, sn, sln))
    lines.append("### Node reduction\n")
    if both:
        sim_reds = [100 * (on - sn) / on for _, on, sn, _ in both]
        slim_reds = [100 * (on - sln) / on for _, on, _, sln in both]
        slim_fewer = sum(1 for _, _, sn, sln in both if sln < sn)
        sim_fewer = sum(1 for _, _, sn, sln in both if sn < sln)
        equal = sum(1 for _, _, sn, sln in both if sn == sln)
        lines.append(f"Over the {len(both)} models both simplifiers processed:\n")
        lines.append("| metric | onnxsim | onnxslim |")
        lines.append("| --- | --- | --- |")
        lines.append(
            f"| median node reduction | {_median(sim_reds):.1f}% | "
            f"{_median(slim_reds):.1f}% |"
        )
        lines.append("")
        lines.append(
            f"- onnxslim produced **fewer** nodes on {slim_fewer}, onnxsim fewer "
            f"on {sim_fewer}, equal on {equal}."
        )
        lines.append("")
        div = [
            t for t in sorted(both, key=lambda t: -abs(t[2] - t[3])) if t[2] != t[3]
        ][:15]
        if div:
            lines.append(
                "Largest node-count divergences (orig → onnxsim / onnxslim):\n"
            )
            lines.append("| model | orig | onnxsim | onnxslim | fewer |")
            lines.append("| --- | --- | --- | --- | --- |")
            for r, on, sn, sln in div:
                fewer = "onnxslim" if sln < sn else "onnxsim"
                lines.append(f"| {r['model']} | {on} | {sn} | {sln} | {fewer} |")
            lines.append("")
    else:
        lines.append("_No model was processed by both simplifiers._\n")

    return lines


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
        "onnxslim_status",
        "slim_nodes",
        "slim_reduction_pct",
        "onnxslim_seconds",
        "slim_conv_status",
        "slim_conv_ops",
        "slim_conv_seconds",
        "slim_conv_error",
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

    lines.extend(comparison_section(rows))

    lines.append("## All models\n")
    lines.append(
        "| model | verdict | onnxsim (orig→simp) | X2Paddle base | X2Paddle simp | "
        "onnxslim (orig→slim) | X2Paddle slim |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
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
            f"{r.get('simp_conv_status')} ({r.get('simp_ops')}) | "
            f"{r.get('orig_nodes')}→{r.get('slim_nodes')} ({r.get('onnxslim_status')}) | "
            f"{r.get('slim_conv_status')} ({r.get('slim_conv_ops')}) |"
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
