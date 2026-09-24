#!/usr/bin/env python3
"""Share of one HTP execute by model region and by op type, from a QNN detailed-profiling CSV.

  profile_regions.py <model.onnx> <prof.csv>
Regions come from the exported node names (rfdetr module paths): the DINOv2 backbone (patch embed,
attention, MLP, norms), the projector, two-stage query selection, and the decoder (self-attention,
deformable cross-attention, FFN) plus heads. Detailed profiling serializes ops, so shares are
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
last = (
    len(names) - 1 - names[::-1].index(names[0]) if names else 0
)  # start of the last execute


def region(n):
    # the DINOv2 window partition / merge: Reshape/Transpose directly under encoder/encoder/ or
    # embeddings/ (not inside a layer.N block)
    if re.search(
        r"/encoder/encoder/(embeddings/)?(Reshape|Transpose|Concat|Slice|Tile)", n
    ):
        return "backbone window partition / merge"
    if "/backbone/0/encoder" in n or "/backbone.0/encoder" in n:
        for k, v in [
            ("attention", "backbone attention"),
            ("/mlp/", "backbone MLP"),
            ("embeddings", "backbone embed"),
        ]:
            if k in n:
                return v
        return "backbone other (norms, scales, windowing)"
    if "/backbone/" in n or "/backbone.0/" in n:
        return "projector"
    if "/decoder/" in n:
        for k, v in [
            ("cross_attn", "decoder cross-attn (MSDA)"),
            ("self_attn", "decoder self-attn"),
        ]:
            if k in n:
                return v
        return "decoder FFN / norms / ref points"
    if "/transformer/" in n:
        return "query selection (enc heads, TopK, gathers)"
    return "heads / other"


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
tot = sum(by_reg.values()) or 1
print("region share:")
for k, v in by_reg.most_common():
    print(f"  {100 * v / tot:5.1f}%  {k}")
print("op share:")
for k, v in by_op.most_common(14):
    print(f"  {100 * v / tot:5.1f}%  {k}")
