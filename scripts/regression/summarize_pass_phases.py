#!/usr/bin/env python3
"""Summarize per-model ONNXSIM_PROFILE_PASS_PHASES tables from the
model-regression set into a "where to optimize next" bottleneck hint.

Every regression/known-slow shard captures one ``<model>.pass_phases.txt``
per model unconditionally (see ``run_regression.py``'s
``--profile-pass-phases-dir``, default on) -- onnxsim's raw stderr dump of
its per-optimizer-pass timing tables (see ``onnxsim.cpp``'s
``profile_pass_phases`` block for the exact format this parses). Today those
files are only concatenated verbatim into a ``profile-pass-phases`` artifact
and the job log -- useful for a deep dive on one model, but nothing
aggregates them into "which pass is actually worth optimizing next", so
answering that has meant manually fetching CI logs and eyeballing ~90 tables
by hand. This script does that aggregation once, mirroring
``summarize_profiles.py``'s structure (and its ``$GITHUB_STEP_SUMMARY``
write) for the finer-grained pass-phase data.

Usage:
    # pass-phase-summary.md from pass_phases/*.pass_phases.txt
    summarize_pass_phases.py "pass_phases/*.pass_phases.txt"
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

# Matches a "total runPass() time, all pass kinds" table row: a bare
# identifier-like pass name, then total(ms) and calls, right-aligned in
# fixed-width columns (see onnxsim.cpp's std::setw calls) -- whitespace-split
# is safe since pass names never contain spaces.
_ROW_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s+([\d.]+)\s+(\d+)\s*$")

_ALL_KINDS_HEADER = "total runPass() time, all pass kinds"
_GRAND_TOTAL_RE = re.compile(
    r"GRAND TOTAL \(sum of all pass runPass\(\) calls\):\s*([\d.]+)ms"
)

# The CSETensorHash/CSETensorCompare breakdown's fixed row labels (see
# onnxsim.cpp) -> the key to aggregate them under.
_CSE_ROW_RE = re.compile(
    r"^(raw_data hash|typed-field hash|raw_data compare|typed-field compare)"
    r"\s+([\d.]+)\s+(\d+)\s*$"
)
_CSE_CACHE_RE = re.compile(
    r"raw_data hash cache: (\d+)/(\d+) calls .*?\(([\d.]+)ms\) vs (\d+) "
    r"misses \(([\d.]+)ms\)"
)


def _model_name(path):
    # worker.py writes <model_id with "/" -> "__">.pass_phases.txt -- strip
    # the whole ".pass_phases.txt" suffix, not just the last extension
    # (os.path.splitext would leave a stray ".pass_phases" behind).
    base = os.path.basename(path)
    if base.endswith(".pass_phases.txt"):
        base = base[: -len(".pass_phases.txt")]
    return base.replace("__", "/", 1)


def parse_pass_phases(text):
    """Parse one model's pass-phase dump.

    Returns a dict: ``passes`` (pass name -> {"total_ms", "calls"}),
    ``grand_total_ms``, ``cse`` (row label -> {"ms", "calls"}), ``cse_cache``
    (hit/miss counts+ms, or ``None`` if the model had no raw_data hash
    calls). Any section not found in ``text`` is left empty/``None`` rather
    than raising -- a truncated or unexpected dump just contributes less,
    since this is a best-effort hint, not something that should ever fail
    the job.
    """
    lines = text.splitlines()
    passes = {}
    grand_total_ms = None
    in_all_kinds_table = False
    for line in lines:
        if _ALL_KINDS_HEADER in line:
            in_all_kinds_table = True
            continue
        if in_all_kinds_table:
            m = _GRAND_TOTAL_RE.search(line)
            if m:
                grand_total_ms = float(m.group(1))
                in_all_kinds_table = False
                continue
            row = _ROW_RE.match(line)
            if row:
                name, total_ms, calls = row.groups()
                passes[name] = {"total_ms": float(total_ms), "calls": int(calls)}
            # A non-matching, non-blank line inside the table (the header
            # row itself, or a "---" divider) just falls through -- neither
            # a data row nor the terminator, so keep scanning.

    cse = {}
    cse_cache = None
    for line in lines:
        row = _CSE_ROW_RE.match(line.strip())
        if row:
            label, ms, calls = row.groups()
            cse[label] = {"ms": float(ms), "calls": int(calls)}
            continue
        m = _CSE_CACHE_RE.search(line)
        if m:
            hits, total, hit_ms, misses, miss_ms = m.groups()
            cse_cache = {
                "hits": int(hits), "total": int(total), "hit_ms": float(hit_ms),
                "misses": int(misses), "miss_ms": float(miss_ms),
            }

    return {
        "passes": passes,
        "grand_total_ms": grand_total_ms,
        "cse": cse,
        "cse_cache": cse_cache,
    }


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("txt_globs", nargs="*", default=["*.pass_phases.txt"],
                     help="glob pattern(s) for *.pass_phases.txt files")
    ap.add_argument("--output", default="pass-phase-summary.md")
    ap.add_argument("--top-passes", type=int, default=12,
                     help="how many passes to show in the aggregate ranking")
    ap.add_argument("--top-models", type=int, default=10,
                     help="how many of the slowest models to list")
    args = ap.parse_args(argv[1:])

    paths = sorted({p for pat in args.txt_globs for p in glob.glob(pat)})
    models = []
    unparseable = []
    for p in paths:
        try:
            with open(p, encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError as e:
            unparseable.append((p, str(e)))
            continue
        parsed = parse_pass_phases(text)
        if not parsed["passes"]:
            unparseable.append((p, "no pass-phase table found"))
            continue
        parsed["model"] = _model_name(p)
        models.append(parsed)
    models.sort(key=lambda m: -(m["grand_total_ms"] or 0))

    lines = ["# Model-regression pass-phase bottleneck hint\n"]
    lines.append(
        f"**{len(models)} model(s) parsed**"
        + (f", {len(unparseable)} unparseable/empty" if unparseable else "")
        + " from `ONNXSIM_PROFILE_PASS_PHASES` (captured unconditionally on "
        "every run -- see `scripts/regression/README.md#per-pass-profiling"
        "-onnxsim_profile_pass_phases`).\n"
    )

    # --- Aggregate: which pass dominates across the whole sampled set ------ #
    totals = {}
    dominant_counts = {}
    for m in models:
        for name, a in m["passes"].items():
            totals[name] = totals.get(name, 0.0) + a["total_ms"]
        if m["passes"]:
            top_name = max(m["passes"].items(), key=lambda kv: kv[1]["total_ms"])[0]
            dominant_counts[top_name] = dominant_counts.get(top_name, 0) + 1
    grand_total = sum(m["grand_total_ms"] or 0.0 for m in models) or 1.0
    if totals:
        lines.append(
            f"## Top {min(args.top_passes, len(totals))} passes by aggregate time\n"
        )
        lines.append(
            "| pass | total (s) | % of summed grand total | dominant in N model(s) |"
        )
        lines.append("| --- | ---: | ---: | ---: |")
        ranked = sorted(totals.items(), key=lambda kv: -kv[1])[: args.top_passes]
        for name, total_ms in ranked:
            lines.append(
                f"| `{name}` | {total_ms / 1000:.2f} | "
                f"{100 * total_ms / grand_total:.1f}% | "
                f"{dominant_counts.get(name, 0)} |"
            )
        lines.append("")
        lines.append(
            "_\"dominant in N model(s)\" = the pass with the single highest "
            "`total(ms)` for that model -- a rough proxy for where a fix would "
            "help the most models, not just move the biggest aggregate number._\n"
        )

    # --- Slowest models, so the aggregate ranking's absolute impact is clear #
    if models:
        top = models[: args.top_models]
        lines.append(f"## {len(top)} slowest model(s) (by pass-suite grand total)\n")
        lines.append("| model | grand total (s) | dominant pass | dominant % |")
        lines.append("| --- | ---: | --- | ---: |")
        for m in top:
            gt = m["grand_total_ms"] or 0.0
            if m["passes"]:
                dom_name, dom = max(m["passes"].items(), key=lambda kv: kv[1]["total_ms"])
                dom_pct = f"{100 * dom['total_ms'] / gt:.0f}%" if gt else "n/a"
            else:
                dom_name, dom_pct = "n/a", "n/a"
            lines.append(f"| {m['model']} | {gt / 1000:.2f} | `{dom_name}` | {dom_pct} |")
        lines.append("")

    # --- CSETensorHash/CSETensorCompare, aggregated -------------------------- #
    # eliminate_duplicate_initializer / eliminate_common_subexpression's own
    # hash-map lookup cost has twice now been the actual bottleneck once the
    # top-ranked pass above was fixed (onnxsim#717, #720) -- surface it
    # directly instead of requiring a re-dive into the raw per-model dumps
    # every time.
    cse_totals = {}
    cache_agg = {"hits": 0, "total": 0, "hit_ms": 0.0, "misses": 0, "miss_ms": 0.0}
    have_cache = False
    for m in models:
        for label, a in m["cse"].items():
            t = cse_totals.setdefault(label, {"ms": 0.0, "calls": 0})
            t["ms"] += a["ms"]
            t["calls"] += a["calls"]
        if m["cse_cache"]:
            have_cache = True
            for k in ("hits", "total", "misses"):
                cache_agg[k] += m["cse_cache"][k]
            for k in ("hit_ms", "miss_ms"):
                cache_agg[k] += m["cse_cache"][k]
    if cse_totals:
        lines.append("## CSETensorHash / CSETensorCompare, aggregated\n")
        lines.append("| | total (ms) | calls |")
        lines.append("| --- | ---: | ---: |")
        for label in ("raw_data hash", "typed-field hash", "raw_data compare",
                      "typed-field compare"):
            if label in cse_totals:
                t = cse_totals[label]
                lines.append(f"| {label} | {t['ms']:.2f} | {t['calls']} |")
        lines.append("")
        if have_cache and cache_agg["total"]:
            hit_rate = 100 * cache_agg["hits"] / cache_agg["total"]
            lines.append(
                f"raw_data hash cache: {cache_agg['hits']}/{cache_agg['total']} "
                f"calls ({hit_rate:.1f}%) were hits ({cache_agg['hit_ms']:.2f}ms) "
                f"vs {cache_agg['misses']} misses ({cache_agg['miss_ms']:.2f}ms).\n"
            )

    if unparseable:
        lines.append("## Unparseable / empty files\n")
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
