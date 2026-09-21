#!/usr/bin/env python3
"""Attribute a compiled AX650 step's MCode, ``npu_params`` and cycles to ops.

Input: a compiled ``.axmodel`` plus the two files ``pulsar2 build
--compiler.npu_perf --debug.dump_frontend_graph`` writes next to it
(``compiler/debug/subgraph_npu_0/b1/op_profile.csv`` and ``trace.json``).
Nothing here predicts or emits MCode; it measures.

What is exact and what is an estimate
-------------------------------------
* **Cycles by op type** -- exact: the sum of ``op_profile.csv`` ``cycles`` per
  ``type``.
* **MCode layout** -- exact: the five stream segments of the FlatBuffers tail
  (``mcode.segments``). Measured on four builds, they are the instruction
  queues of the five engines the trace names, in this order: two convolution
  engines (``conv0``/``conv1``), ``teng2``, ``cv3``, ``sdma4``. The DMA
  segment's count of ``0xa3`` verbs equals the number of ``sdma4`` tasks in
  the trace in every build checked, and ``check_segment_engines`` re-tests it.
* **``npu_params`` bytes by op type** -- exact up to the loads the trace shows:
  each ``ld:param`` task reads ``params:[start:]`` and its length is the
  destination OCM range; the consuming op comes from ``dep_by``, else from the
  first other task that reads the tensor the load writes.
* **MCode bytes by op type** -- an *estimate*. Bytes are known per engine
  segment, not per task, so a segment's bytes are split over its engine's tasks
  by op type, either by task count or by cycles. The two weightings bracket
  the answer; the split within one engine is not measured. An attempt to align
  the DMA segment's 792 per-task groups with trace tasks found no feature
  that correlated with group length, so no per-task attribution is claimed.

Usage::

    step_attribution.py MODEL.axmodel [--profile op_profile.csv] [--trace trace.json]
                                      [--json]

With no ``--profile``/``--trace``, the files are looked up in the standard
location under ``MODEL``'s directory.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import re
import sys

import onnx

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import mcode  # noqa: E402

#: Segment order (stream order) -> engine track name in ``trace.json``.
#: ``conv`` stands for the ``conv0``/``conv1`` pair, whose segments are
#: near-equal in every build; which segment is which is not established.
SEGMENT_ENGINES = ("conv", "conv", "teng2", "cv3", "sdma4")

_ENGINE_TRACKS = {
    "conv": ("conv0", "conv1"),
    "teng2": ("teng2",),
    "cv3": ("cv3",),
    "sdma4": ("sdma4",),
}

#: Bucket for trace tasks that cannot be tied to an ``op_profile.csv`` row.
UNATTRIBUTED = "(unattributed ld/st)"

# What this repository's emitters can do per op type. Values are
# (capability, scope) where capability is one of:
#   "retarget"  -- rewrites parameters of a compiler-built reference of the
#                  same shape; the MCode is kept from the reference
#   "metadata"  -- relabels shapes of a compiler-built reference; MCode kept
#   "none"      -- no emitter
COVERAGE = {
    "AxQuantizedConv": (
        "retarget",
        "emitter.py: weight table of a learned shape (k reference builds); "
        "tiny_emit.py: site-A scales. MCode from reference.",
    ),
    "AxMatMul": (
        "retarget",
        "tiny_emit.py: A-input scale (same-form only); weight table via "
        "emitter.py for a learned shape.",
    ),
    "AxMul": ("retarget", "tiny_emit.py: input/output scales, zero point x."),
    "AxAdd": ("retarget", "tiny_emit.py: three output-scale fields."),
    "AxSub": ("retarget", "tiny_emit.py: Sub scale patcher."),
    "AxQuantizedNeg": ("retarget", "tiny_emit.py: Neg output scale."),
    "AxSlice": (
        "retarget",
        "memory_emit.py: static Slice float32 [1,8] axis 1, steps 1-4 only; "
        "nothing at DMA-load ('ld:') scale.",
    ),
    "AxReshape": (
        "metadata",
        "reshape_emit.py: only Reshapes Pulsar2 fuses into a Relu neighbour.",
    ),
    "AxTranspose": ("none", "docs/axera-transpose-mcode.md: not predictable."),
}


def load_mcode(axmodel_path: str) -> tuple[bytes, int]:
    """(MCode blob, ``npu_params`` length) of a compiled model."""
    model = onnx.load(axmodel_path, load_external_data=False)
    inits = {i.name: i for i in model.graph.initializer}
    neu = [k for k in inits if k.endswith("_neu")]
    if len(neu) != 1:
        raise ValueError(f"expected one *_neu initializer, found {len(neu)}")
    params = len(inits["npu_params"].raw_data) if "npu_params" in inits else 0
    return bytes(inits[neu[0]].raw_data), params


def layout(blob: bytes) -> list[dict]:
    """Per-segment offset, bytes, engine label, and count of ``0xa3`` verbs."""
    header, segs = mcode.segments(blob)
    if len(segs) != len(SEGMENT_ENGINES):
        raise ValueError(f"expected {len(SEGMENT_ENGINES)} segments, got {len(segs)}")
    out = []
    for (pos, length, table), engine in zip(segs, SEGMENT_ENGINES):
        recs = mcode.decode(blob, start=pos, end=pos + length, **mcode.FULL_RULE)
        out.append(
            {
                "offset": pos,
                "bytes": length,
                "words": table.get(2, 0),
                "engine": engine,
                "a3_verbs": sum(
                    1 for r in recs if r["kind"] == "V" and r["verb"] == 0xA3
                ),
            }
        )
    return out


def load_profile(path: str) -> dict[str, dict]:
    with open(path, newline="") as f:
        return {
            r["op"]: {"type": r["type"], "cycles": float(r["cycles"])}
            for r in csv.DictReader(f)
        }


def load_trace(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data["traceEvents"] if isinstance(data, dict) else data


_TILE = re.compile(r"(?<=_gop)_\d+(_\d+)?$|(?<=_sp)_\d+(_\d+)?$")
_TRAILING = re.compile(r"_\d+$")
_BRACKETS = re.compile(r"\[[^\]]*\]")


def resolve_op(name: str, profile: dict[str, dict]) -> str | None:
    """The ``op_profile.csv`` op a trace task belongs to, or None.

    Trace tasks are named ``<op>_s<k>_gop_<i>_<j>`` (or ``_sp_``); the profile
    keeps one row per ``<op>_s<k>_gop``/``_sp``. DMA tasks carry an ``ld:`` or
    ``st:`` prefix, and bare ops end in ``_<i>_<j>``.
    """
    stem = name
    for prefix in ("ld:", "st:"):
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
    cands = [name, _TILE.sub("", name), stem, _TILE.sub("", stem)]
    cands.append(_TILE.sub("", _BRACKETS.sub("", stem)))
    trimmed = stem
    for _ in range(3):
        trimmed = _TRAILING.sub("", trimmed)
        cands.append(trimmed)
        cands.append("ld:" + trimmed)
    return next((c for c in cands if c in profile), None)


_EMBEDDED = re.compile(r"Ax[A-Za-z]+")
_NAME_HINTS = (
    ("pre_transpose", "AxTranspose"),
    ("__sub_", "AxSub"),
    ("__add_", "AxAdd"),
    ("__mul_", "AxMul"),
    ("__div_", "AxDiv"),
)


def embedded_type(name: str) -> str | None:
    """Op type named inside a store task (``st:op_..:AxClip_out[..]16_s3_gop``)."""
    match = _EMBEDDED.search(name)
    if match:
        return match.group(0)
    lowered = name.lower()
    return next((t for hint, t in _NAME_HINTS if hint in lowered), None)


def _tensor_names(text: str) -> list[str]:
    """Tensor names in a trace ``inputs``/``outputs`` block (``name - space:[..]``)."""
    return [line.split(" - ", 1)[0] for line in text.splitlines() if " - " in line]


def build_readers(events: list[dict]) -> dict[str, list[dict]]:
    """``tensor name -> tasks that list it as an input``."""
    readers: dict[str, list[dict]] = collections.defaultdict(list)
    for ev in events:
        for tensor in _tensor_names(ev.get("args", {}).get("inputs", "")):
            readers[tensor].append(ev)
    return readers


def event_type(
    event: dict,
    profile: dict[str, dict],
    readers: dict[str, list[dict]] | None = None,
) -> str:
    """Op type of a trace task.

    In order: its own profile row; for ``ld:``/``st:`` transfers, the op type
    the task name embeds; for loads (``ld:param``/``ld:input``), the type of the
    first other task that reads the tensor the load writes. Those transfers are
    data movement *for* that op type and are counted with it. Runtime-value
    loads (``__rtv_*``) and anything else stay ``UNATTRIBUTED``.
    """
    name = event["name"]
    op = resolve_op(name, profile)
    if op:
        return profile[op]["type"]
    if not name.startswith(("ld:", "st:")):
        return UNATTRIBUTED
    embedded = embedded_type(name)
    if embedded:
        return embedded
    if readers is not None:
        for tensor in _tensor_names(event.get("args", {}).get("outputs", "")):
            for reader in readers.get(tensor, []):
                if reader is event:
                    continue
                found = event_type(reader, profile)
                if found != UNATTRIBUTED:
                    return found
    return UNATTRIBUTED


def cycles_by_type(profile: dict[str, dict]) -> dict[str, dict]:
    out: dict[str, dict] = collections.defaultdict(lambda: {"ops": 0, "cycles": 0.0})
    for row in profile.values():
        out[row["type"]]["ops"] += 1
        out[row["type"]]["cycles"] += row["cycles"]
    return dict(out)


def task_matrix(events: list[dict], profile: dict[str, dict]) -> dict:
    """``{engine: {op_type: {"tasks": n, "dur": summed trace duration}}}``."""
    track_to_engine = {t: e for e, ts in _ENGINE_TRACKS.items() for t in ts}
    readers = build_readers(events)
    out: dict = collections.defaultdict(
        lambda: collections.defaultdict(lambda: {"tasks": 0, "dur": 0.0})
    )
    for ev in events:
        engine = track_to_engine.get(ev["tid"])
        if engine is None:
            continue
        cell = out[engine][event_type(ev, profile, readers)]
        cell["tasks"] += 1
        cell["dur"] += float(ev.get("dur", 0.0))
    return {e: dict(v) for e, v in out.items()}


def check_segment_engines(segs: list[dict], events: list[dict]) -> list[str]:
    """Problems with the segment<->engine reading (empty = consistent)."""
    problems = []
    sdma = sum(1 for e in events if e["tid"] == "sdma4")
    dma_seg = [s for s in segs if s["engine"] == "sdma4"][0]
    if dma_seg["a3_verbs"] != sdma:
        problems.append(
            f"DMA segment has {dma_seg['a3_verbs']} 0xa3 verbs but the trace has {sdma} sdma4 tasks"
        )
    return problems


def mcode_estimate(
    segs: list[dict], matrix: dict, weight: str = "tasks"
) -> dict[str, float]:
    """Split each engine segment's bytes over its op types (an estimate)."""
    key = "tasks" if weight == "tasks" else "dur"
    by_engine = collections.defaultdict(float)
    for s in segs:
        by_engine[s["engine"]] += s["bytes"]
    out: dict[str, float] = collections.defaultdict(float)
    for engine, total in by_engine.items():
        cells = matrix.get(engine, {})
        denom = sum(c[key] for c in cells.values())
        if not denom:
            out["(no tasks in engine)"] += total
            continue
        for op_type, cell in cells.items():
            out[op_type] += total * cell[key] / denom
    return dict(out)


_OCM = re.compile(r"ocm_base:\[(\d+):(\d+)\]")
_PARAM = re.compile(r"params:\[(\d*):\]")


def params_by_type(
    events: list[dict], profile: dict[str, dict], params_len: int
) -> dict:
    """``npu_params`` bytes read by ``ld:param`` tasks, by consuming op type.

    A load's length is the size of its destination OCM range. Its consumer is
    the first task in ``dep_by`` that resolves to a profile op, else the first
    other task whose ``inputs`` name the tensor the load writes; loads with no
    resolvable consumer are counted under ``UNATTRIBUTED``. Overlapping loads
    of the same range are counted once.
    """
    readers = build_readers(events)
    spans: dict[tuple[int, int], str] = {}
    for ev in events:
        args = ev.get("args", {})
        if args.get("cname") != "ld:param":
            continue
        start = _PARAM.search(args.get("inputs", ""))
        dest = _OCM.search(args.get("outputs", ""))
        if not start or not dest:
            continue
        begin = int(start.group(1) or 0)
        length = int(dest.group(2)) - int(dest.group(1))
        consumer = UNATTRIBUTED
        for dep in args.get("dep_by", []):
            op = resolve_op(dep, profile)
            if op:
                consumer = profile[op]["type"]
                break
        if consumer == UNATTRIBUTED:
            consumer = event_type(ev, profile, readers)
        spans.setdefault((begin, begin + length), consumer)
    out: dict[str, int] = collections.defaultdict(int)
    covered = 0
    last_end = -1
    for (a, b), consumer in sorted(spans.items()):
        a = max(a, last_end)
        if b > a:
            out[consumer] += b - a
            covered += b - a
            last_end = b
    return {"by_type": dict(out), "covered": covered, "params_len": params_len}


def coverage_report(shares: dict[str, float]) -> dict:
    """Fraction of ``shares`` (op type -> bytes or cycles) per emitter class."""
    total = sum(shares.values()) or 1.0
    out = {"retarget": 0.0, "metadata": 0.0, "none": 0.0, "unattributed": 0.0}
    for op_type, value in shares.items():
        cap = COVERAGE.get(op_type, ("none", ""))[0]
        if op_type in (UNATTRIBUTED, "(no tasks in engine)"):
            cap = "unattributed"
        out[cap] += value / total
    return out


def _pct(x: float, total: float) -> str:
    return f"{100 * x / total:5.1f}%" if total else "  n/a"


def report(axmodel: str, profile_path: str, trace_path: str) -> dict:
    blob, params_len = load_mcode(axmodel)
    segs = layout(blob)
    profile = load_profile(profile_path)
    events = load_trace(trace_path)
    matrix = task_matrix(events, profile)
    return {
        "mcode_bytes": len(blob),
        "params_bytes": params_len,
        "segments": segs,
        "problems": check_segment_engines(segs, events),
        "cycles": cycles_by_type(profile),
        "tasks": matrix,
        "mcode_by_tasks": mcode_estimate(segs, matrix, "tasks"),
        "mcode_by_dur": mcode_estimate(segs, matrix, "dur"),
        "params": params_by_type(events, profile, params_len),
    }


SCOPE_CAVEAT = (
    "NOTE: 'retarget'/'metadata' mean an emitter exists for the op type, and\n"
    "only for the specific shapes it measured. Every emitter patches a\n"
    "compiler-built reference of the same graph; none synthesizes MCode. The\n"
    "percentages below are an upper bound, not a claim that this step can be\n"
    "emitted -- see docs/axera-step-attribution.md for the shape comparison."
)


def print_report(r: dict) -> None:
    print(f"MCode {r['mcode_bytes']} B, npu_params {r['params_bytes']} B")
    print("\nsegments (stream order):")
    stream = sum(s["bytes"] for s in r["segments"])
    for s in r["segments"]:
        print(
            f"  @{s['offset']:7d} {s['bytes']:7d} B {_pct(s['bytes'], stream)}  "
            f"engine={s['engine']:6s} a3 verbs={s['a3_verbs']}"
        )
    print("check:", r["problems"] or "segment/engine reading consistent")
    total_cyc = sum(v["cycles"] for v in r["cycles"].values())
    total_mc = sum(r["mcode_by_tasks"].values())
    pb = r["params"]["by_type"]
    print(
        f"\n{'op type':26s} {'ops':>5s} {'cycles%':>8s} {'MCode% (tasks)':>15s} {'MCode% (dur)':>13s} {'params B':>10s}"
    )
    types = set(r["cycles"]) | set(r["mcode_by_tasks"]) | set(pb)
    for t in sorted(types, key=lambda t: -r["cycles"].get(t, {"cycles": 0})["cycles"]):
        c = r["cycles"].get(t, {"ops": 0, "cycles": 0.0})
        print(
            f"{t:26s} {c['ops']:5d} {_pct(c['cycles'], total_cyc):>8s} "
            f"{_pct(r['mcode_by_tasks'].get(t, 0), total_mc):>15s} "
            f"{_pct(r['mcode_by_dur'].get(t, 0), total_mc):>13s} {pb.get(t, 0):10d}"
        )
    print(
        f"\nparams covered by ld:param tasks: {r['params']['covered']} of {r['params']['params_len']} B"
    )
    print("\n" + SCOPE_CAVEAT)
    for label, shares in (
        ("MCode (task-count estimate)", r["mcode_by_tasks"]),
        ("MCode (duration estimate)", r["mcode_by_dur"]),
        ("cycles", {t: v["cycles"] for t, v in r["cycles"].items()}),
        ("params", r["params"]["by_type"]),
    ):
        cov = coverage_report(shares)
        print(
            f"emitter coverage of {label}: retarget {100 * cov['retarget']:.1f}%, "
            f"metadata {100 * cov['metadata']:.1f}%, none {100 * cov['none']:.1f}%, "
            f"unattributed {100 * cov['unattributed']:.1f}%"
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("axmodel")
    ap.add_argument("--profile")
    ap.add_argument("--trace")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    debug = os.path.join(
        os.path.dirname(os.path.abspath(args.axmodel)),
        "compiler",
        "debug",
        "subgraph_npu_0",
        "b1",
    )
    profile = args.profile or os.path.join(debug, "op_profile.csv")
    trace = args.trace or os.path.join(debug, "trace.json")
    result = report(args.axmodel, profile, trace)
    if args.json:
        print(json.dumps(result, indent=1, default=float))
    else:
        print_report(result)
    return 1 if result["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
