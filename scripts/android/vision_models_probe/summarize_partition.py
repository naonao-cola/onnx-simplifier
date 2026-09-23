#!/usr/bin/env python3
"""Summarize a partition_report.sh run: which ops QNN refused, their tensor ranks, and timings.

usage: summarize_partition.py <model.onnx> <logcat> <htp-fallback.out> <htp.out>

QNN reports each refused op in logcat as "Failed to validate op <name> with error 0x...". This
maps names back to the ONNX graph (ORT may append suffixes like `_2` to partition copies) and
prints op type and the maximum rank among its inputs/outputs, because the HTP's rank-5 limit
turned out to be the dominant rejection reason for the transformer models probed here.
"""
from __future__ import annotations

import collections
import re
import sys

import onnx


def ranks(model_path: str) -> tuple[dict, dict]:
    m = onnx.load(model_path, load_external_data=False)
    try:
        m = onnx.shape_inference.infer_shapes(m)
    except Exception:  # noqa: BLE001 -- best effort; large models may not infer cleanly
        pass
    g = m.graph
    vi = {v.name: len(v.type.tensor_type.shape.dim) for v in [*g.value_info, *g.input, *g.output]}
    for t in g.initializer:
        vi[t.name] = len(t.dims)
    return {n.name: n for n in g.node}, vi


def main() -> None:
    model, logcat, fb, strict = sys.argv[1:5]
    nodes, vi = ranks(model)
    text = open(logcat, errors="replace").read()
    failed = collections.OrderedDict()
    for name, err in re.findall(r"Failed to validate op (\S+) with error (0x[0-9a-f]+)", text):
        failed.setdefault(re.sub(r"_\d+$", "", name) if name not in nodes else name, err)
    by_type = collections.Counter()
    by_rank = collections.Counter()
    unknown = []
    for name, err in failed.items():
        n = nodes.get(name)
        if n is None:
            unknown.append(name)
            continue
        r = max([vi.get(t, -1) for t in [*n.input, *n.output] if t] or [-1])
        by_type[(n.op_type, err)] += 1
        by_rank[r] += 1
    print(f"model: {model}  nodes: {len(nodes)}  ops QNN refused: {len(failed)}")
    for (t, e), c in by_type.most_common():
        print(f"  {t:22s} {e}  x{c}")
    print(f"  refused ops by max tensor rank: {dict(sorted(by_rank.items()))}")
    if unknown:
        print(f"  (names not found in graph, likely ORT-fused: {unknown[:8]})")
    graphs = len(set(re.findall(r"QnnGraph_create started for graph (\S+)", text)))
    print(f"  QNN graphs created with CPU fallback allowed: {graphs}")
    for label, path in (("htp-fallback", fb), ("htp strict", strict)):
        out = open(path, errors="replace").read()
        med = re.search(r"median_ms ([0-9.]+)", out)
        verdict = "PASS" if "PASS" in out else "FAIL"
        err = "" if verdict == "PASS" else " | " + out.strip().splitlines()[-1][:160] if out.strip() else ""
        print(f"  {label:13s} {verdict}  median {med.group(1) if med else '-'} ms{err}")


if __name__ == "__main__":
    main()
