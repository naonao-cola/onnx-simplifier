"""Summarize a partition run: which ops QNN refused on the HTP, why, and the tensor ranks involved.

QNN reports each refused op in logcat as "Failed to validate op <name> with error 0x...". The
names map back to the ONNX graph (ORT may append `_<n>` to copies). Rank is reported because the
HTP's rank-5 limit was the only rejection reason in the transformer models probed so far
(../../vision_models_plan.md).
"""
from __future__ import annotations

import collections
import re
from pathlib import Path

import onnx


def summarize(model: Path, logcat: Path, fallback_out: str, strict_out: str) -> dict:
    m = onnx.load(str(model), load_external_data=False)
    try:
        m = onnx.shape_inference.infer_shapes(m)
    except Exception:  # noqa: BLE001 -- best effort
        pass
    g = m.graph
    rank = {v.name: len(v.type.tensor_type.shape.dim) for v in [*g.value_info, *g.input, *g.output]}
    for t in g.initializer:
        rank[t.name] = len(t.dims)
    nodes = {n.name: n for n in g.node}
    text = logcat.read_text(errors="replace")
    refused = {}
    for name, err in re.findall(r"Failed to validate op (\S+) with error (0x[0-9a-f]+)", text):
        key = name if name in nodes else re.sub(r"_\d+$", "", name)
        refused.setdefault(key, err)
    rows = []
    for name, err in refused.items():
        n = nodes.get(name)
        r = max([rank.get(t, -1) for t in [*n.input, *n.output] if t] or [-1]) if n else -1
        rows.append({"name": name, "op": n.op_type if n else "?", "error": err, "max_rank": r})

    def med(out):
        x = re.search(r"median_ms ([0-9.]+)", out)
        return float(x.group(1)) if x else None

    return {
        "model_nodes": len(g.node),
        "refused": rows,
        "refused_by_op": dict(collections.Counter((r["op"], r["error"]) for r in rows).most_common()),
        "refused_by_rank": dict(collections.Counter(r["max_rank"] for r in rows)),
        "qnn_graphs_with_fallback": len(set(re.findall(r"QnnGraph_create started for graph (\S+)", text))),
        "fallback": {"pass": "PASS" in fallback_out, "median_ms": med(fallback_out)},
        "strict": {"pass": "PASS" in strict_out, "median_ms": med(strict_out),
                   "error": None if "PASS" in strict_out else (strict_out.strip().splitlines() or [""])[-1][:300]},
    }


def render(rep: dict) -> str:
    s = [f"  {rep['model_nodes']} nodes; QNN refused {len(rep['refused'])}; "
         f"{rep['qnn_graphs_with_fallback']} QNN graph(s) with CPU fallback allowed"]
    for (op, err), c in rep["refused_by_op"].items():
        s.append(f"    refused {op:20s} {err} x{c}")
    if rep["refused"]:
        s.append(f"    refused by max tensor rank: {rep['refused_by_rank']}")
    for k in ("fallback", "strict"):
        v = rep[k]
        s.append(f"  {k:8s} {'PASS' if v['pass'] else 'FAIL'} median {v['median_ms']} ms"
                 + (f"  ({v['error']})" if k == "strict" and not v["pass"] else ""))
    return "\n".join(s)
