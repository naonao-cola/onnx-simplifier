#!/usr/bin/env python3
"""Attribute the backbone's HTP time to ops, from QNN's own profiling CSVs.

Inputs: backbone.onnx, prof_detailed.csv (per-node HTP cycles), prof_basic.csv (steady-state
accelerator time without per-node instrumentation), and the ceiling summary (summary.burst.json).

Detailed profiling inflates total accelerator cycles ~1.8x vs basic (it appears to serialize ops;
48 of 220 QNN nodes report 0 cycles, so it is not a flat per-node tax). So per-node cycles are used
as *relative shares*, scaled to basic-mode steady-state accelerator time. Per-conv TMAC/s below is
therefore inferred (proportional scaling), not a direct per-op timer.

usage: analyze_profile.py <backbone.onnx> <prof_detailed.csv> <prof_basic.csv> <summary.json> [conv_profile.json]
"""
import collections
import csv
import json
import re
import statistics
import sys

import onnx
from onnx import shape_inference

WARM = 2


def executes(path, key):
    ex, cur = [], None
    for r in csv.DictReader(open(path)):
        if r["Message"] == "BACKEND" and r["Event Identifier"] == "RPC (execute) time":
            cur = {"nodes": []}
            ex.append(cur)
            continue
        if cur is None:
            continue
        ev = r["Event Identifier"]
        if r["Message"] == "NODE":
            cur["nodes"].append((ev.replace(" (cycles)", ""), int(r["Time"])))
        elif r["Message"] == "BACKEND" and r["Unit of Measurement"] == "US":
            cur[ev] = int(r["Time"])
        elif r["Message"] == "EXECUTE":
            cur[ev] = int(r["Time"])
    return ex[WARM:]


def main():
    model, det, basic, summ = sys.argv[1:5]
    conv_prof = sys.argv[5] if len(sys.argv) > 5 else None
    m = shape_inference.infer_shapes(onnx.load(model))
    g = m.graph
    vi = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
          for v in list(g.value_info) + list(g.input) + list(g.output)}
    init = {i.name: list(i.dims) for i in g.initializer}
    prod = {o: n for n in g.node for o in n.output}

    def wshape(name):
        if name in init:
            return init[name]
        n = prod.get(name)
        return wshape(n.input[0]) if n is not None else None

    by_name = {n.name: n for n in g.node}
    names_sorted = sorted(by_name, key=len, reverse=True)

    def onnx_node(qnn_name):
        base = qnn_name.split(":OpId_")[0]
        if base in by_name:
            return by_name[base]
        b2 = re.sub(r"_token_\d+$", "", base)
        if b2 in by_name:
            return by_name[b2]
        for nm in names_sorted:  # longest ONNX name that prefixes the QNN name
            if base.startswith(nm + "_") or base == nm:
                return by_name[nm]
        return None

    det_ex = executes(det, "d")
    bas_ex = executes(basic, "b")
    acc_ms = statistics.median(e["Accelerator (execute) time"] for e in bas_ex) / 1e3
    qnn_ms = statistics.median(e["QNN (execute) time"] for e in bas_ex) / 1e3
    nodes = [n for n, _ in det_ex[0]["nodes"]]
    cyc = {n: statistics.mean(e["nodes"][i][1] for e in det_ex) for i, n in enumerate(nodes)}
    tot = sum(cyc.values())
    ms = {n: c / tot * acc_ms for n, c in cyc.items()}

    ceil = json.load(open(summ))
    ceiling = max(r["tmacs"] for r in ceil["rows"] if r["variant"] == "u8")

    cls = collections.defaultdict(float)
    convs = []
    unmapped = []
    for n in nodes:
        on = onnx_node(n)
        if n.startswith("Input "):
            cls["graph input"] += ms[n]
            continue
        if on is None:
            unmapped.append((n, ms[n]))
            cls["unmapped"] += ms[n]
            continue
        op = on.op_type
        if op == "Conv":
            w = wshape(on.input[1])
            o = vi[on.output[0]]
            macs = o[2] * o[3] * w[0] * w[1] * w[2] * w[3]
            convs.append({"qnn": n, "onnx": on.name, "w": w, "out": o, "macs": macs, "ms": ms[n]})
            cls["Conv"] += ms[n]
        else:
            cls[op] += ms[n]

    # aggregate convs by (kernel, cin, cout, out spatial) like maskrcnn_e2e/profile_data/conv_profile.json
    agg = collections.defaultdict(lambda: {"n": 0, "macs": 0, "ms": 0.0})
    for c in convs:
        k = (c["w"][2], c["w"][1], c["w"][0], c["out"][2], c["out"][3])
        a = agg[k]
        a["n"] += 1
        a["macs"] += c["macs"]
        a["ms"] += c["ms"]

    conv_ms = sum(c["ms"] for c in convs)
    conv_macs = sum(c["macs"] for c in convs)
    print(f"steady state: QNN execute {qnn_ms:.1f} ms, accelerator {acc_ms:.1f} ms, "
          f"host-side (execute - accelerator) {qnn_ms - acc_ms:.1f} ms")
    print(f"practical ceiling (best u8 dense layer): {ceiling:.2f} TMAC/s")
    print(f"backbone: {conv_macs / 1e9:.1f} GMAC; accelerator-only {conv_macs / acc_ms / 1e9:.2f} TMAC/s "
          f"= {conv_macs / acc_ms / 1e9 / ceiling * 100:.0f}% of ceiling")
    print(f"ideal (all MACs at ceiling): {conv_macs / ceiling / 1e9:.1f} ms\n")
    print("| op class | ms (scaled) | % of accelerator |")
    print("|---|---:|---:|")
    for k, v in sorted(cls.items(), key=lambda kv: -kv[1]):
        print(f"| {k} | {v:.2f} | {v / acc_ms * 100:.1f}% |")
    print(f"\nconvs: {conv_ms:.1f} ms scaled for {conv_macs / 1e9:.1f} GMAC -> "
          f"{conv_macs / conv_ms / 1e9:.2f} TMAC/s inside conv nodes = {conv_macs / conv_ms / 1e9 / ceiling * 100:.0f}% of ceiling\n")
    print("| kxk | cin->cout | out HxW | count | GMAC | ms | TMAC/s | % ceiling | ms lost vs ceiling |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|")
    lost_total = 0
    for k, a in sorted(agg.items(), key=lambda kv: -(kv[1]["ms"] - kv[1]["macs"] / ceiling / 1e9)):
        t = a["macs"] / a["ms"] / 1e9 if a["ms"] > 0 else float("inf")
        lost = a["ms"] - a["macs"] / ceiling / 1e9
        lost_total += lost
        print(f"| {k[0]}x{k[0]} | {k[1]}->{k[2]} | {k[3]}x{k[4]} | {a['n']} | {a['macs'] / 1e9:.2f} | "
              f"{a['ms']:.2f} | {t:.2f} | {t / ceiling * 100:.0f}% | {lost:.2f} |")
    print(f"\nconv time lost vs ceiling: {lost_total:.1f} ms")
    if unmapped:
        print("\nunmapped QNN nodes (top):", sorted(unmapped, key=lambda x: -x[1])[:10])
    json.dump({"acc_ms": acc_ms, "qnn_ms": qnn_ms, "ceiling": ceiling, "classes": cls,
               "convs": convs}, open(det.replace(".csv", "_analysis.json"), "w"), indent=1, default=str)
    if conv_prof:
        print("\n(old HVX-era conv_profile.json keys for comparison:)",
              len(json.load(open(conv_prof))), "entries")


if __name__ == "__main__":
    main()
