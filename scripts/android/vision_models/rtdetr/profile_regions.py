#!/usr/bin/env python3
"""Share of one HTP execute by model region and by op type, from a QNN detailed-profiling CSV.

  profile_regions.py <model.onnx> <prof.csv>
Regions come from the exported node names (HF module paths): backbone, encoder (AIFI + CCFM),
query selection (enc_output/score/bbox heads + TopK + gathers), and per decoder layer: self-attn,
MSDA sampling (offset/attention-weight MatMuls, grid), MSDA gather (value_proj, GridSample,
weighted sum, output_proj), FFN/norms/heads. Detailed profiling serializes ops, so shares are
relative -- scale them by the plain run's median.
"""

import collections
import csv
import re
import sys

import onnx

m = onnx.load(sys.argv[1], load_external_data=False)
op = {n.name: n.op_type for n in m.graph.node}
rows = [r for r in csv.DictReader(open(sys.argv[2])) if r["Message"] == "NODE"]
names = [re.sub(r":OpId_\d+ \(cycles\)$", "", r["Event Identifier"]) for r in rows]
last = len(names) - 1 - names[::-1].index(names[0]) if names else 0


def region(name):
    n = name
    if "/backbone/" in n or "encoder_input_proj" in n:
        return "backbone"
    if "/model/encoder/" in n:
        return "hybrid encoder"
    mm = re.search(r"/decoder/layers\.(\d+)/(\w+)", n)
    if mm:
        L, part = mm.groups()
        if part == "encoder_attn":
            if re.search(
                r"value_proj|GridSample|output_proj|ReduceSum|Mul_\d*$|Add_\d*$",
                n.split("encoder_attn")[1],
            ):
                sub = "msda gather"
            else:
                sub = "msda sampling"
            return f"dec{L} {sub}"
        if part == "self_attn":
            return f"dec{L} self-attn"
        return f"dec{L} ffn/norm"
    if "/decoder/" in n:
        return "decoder heads/pos"
    return "query selection/other"


by_reg, by_op = collections.Counter(), collections.Counter()
for name, r in zip(names[last:], rows[last:]):
    base = re.sub(r"_token_\d+$", "", name)
    t = (
        op.get(base)
        or op.get(name)
        or "QNN:" + re.sub(r"[_/].*", "", name.split("/")[-1])
    )
    by_reg[region(base)] += int(r["Time"])
    by_op[t] += int(r["Time"])
tot = sum(by_reg.values())
print(f"{len(names) - last} QNN nodes, {tot / 1e6:.1f} Mcycles")
for t, c in sorted(by_reg.items(), key=lambda kv: -kv[1]):
    print(f"  {t:24s} {100 * c / tot:5.1f}%")
print("by op type:")
for t, c in by_op.most_common(12):
    print(f"  {t:24s} {100 * c / tot:5.1f}%")
