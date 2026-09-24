#!/usr/bin/env python3
"""Locate Mask R-CNN's whole RPN post-processing span in the real graph and capture its real
inputs/outputs from ONNX Runtime, for the fused DSP call in this directory.

The span (all in rest.onnx, per FPN level P2..P6, then merged):
    TopK(objectness, k) -> proposal decode (../proposal_decode) -> width/height "min-size" filter
    (w = x2-x1+1, h = y2-y1+1 re-quantized on their own uint8 grid, keep w>=0 && h>=0, NonZero,
    Gather) -> NonMaxSuppression(iou 0.7) -> first <=1000 kept -> Gather scores/boxes
    -> Concat the 5 levels -> TopK(min(1000, N)) -> Gather boxes -> Q/DQ  ==> the proposal list.
It ends at the proposal list because the next ops (FPN level assignment: sqrt/log/floor, per-level
NonZero/Gather/ScatterElements around RoiAlign) have no DSP implementation yet.

Every structural fact and constant is read out of the graph with assertions, not assumed.

Model constants and per-level tensor names come from ../proposal_decode/capture_proposal_decode.py
(run it once, on any image, into PD_DIR): its levels.txt / levels.json / lN_anchors.bin are
image-independent. This script adds one ORT run of the full model per image:

    python capture_rpn_fused.py --model maskrcnn_sim.onnx --rest rest.onnx --pd PD_DIR \
        --images a.jpg b.jpg ... --out DATA

DATA/model.txt       per level: filter scale/zero point, NMS iou/max, per-level cap; then post cap
DATA/rpn_region.onnx the span cut out of rest.onnx (10 inputs -> proposal list), for the ORT
                     baselines
DATA/<image>/        l{0..4}_{scores,deltas,nchw_q}.bin inputs, per-level intermediates
                     (TopK vals/idx, decoded boxes, NMS selection), final proposals.bin
"""

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "maskrcnn_e2e"))
from eval_common import canvas  # noqa: E402

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--model", required=True, help="full simplified model (maskrcnn_sim.onnx)")
p.add_argument("--rest", required=True, help="rest.onnx from prepare.py")
p.add_argument("--pd", type=Path, required=True, help="capture_proposal_decode.py output dir")
p.add_argument("--images", nargs="+", required=True)
p.add_argument("--out", type=Path, required=True)
args = p.parse_args()
args.out.mkdir(parents=True, exist_ok=True)

rest = onnx.load(args.rest)
g = rest.graph
cv = {t.name: numpy_helper.to_array(t) for t in g.initializer}
for n in g.node:
    if n.op_type == "Constant":
        cv[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)
prod = {o: n for n in g.node for o in n.output}
cons = collections.defaultdict(list)
for n in g.node:
    for i in n.input:
        cons[i].append(n)
rest_inputs = [i.name for i in g.input]


def only(xs, what):
    assert len(xs) == 1, (what, [x.name for x in xs])
    return xs[0]


def fwd_until(tensor, op):
    """nodes of type `op` first reached forward from `tensor` (not crossing another `op`/TopK)"""
    seen, q, hits = set(), [tensor], []
    while q:
        t = q.pop()
        for c in cons[t]:
            if c.name in seen:
                continue
            seen.add(c.name)
            if c.op_type == op:
                hits.append(c)
                continue
            if c.op_type in ("TopK", "NonMaxSuppression"):
                continue
            q.extend(c.output)
    return hits, seen


def back_until(tensors):
    """nodes reached backward from `tensors`, not crossing TopK/NMS (those are included, not expanded)"""
    seen, out, st = set(), [], list(tensors)
    while st:
        n = prod.get(st.pop())
        if n is None or n.name in seen or n.op_type == "Constant":
            continue
        seen.add(n.name)
        out.append(n)
        if n.op_type not in ("TopK", "NonMaxSuppression"):
            st.extend(i for i in n.input if i)
    return out


all_nms = [n for n in g.node if n.op_type == "NonMaxSuppression"]
pd_levels = json.load(open(args.pd / "levels.json"))  # P2 (most anchors) first
levels = []
for L in pd_levels:
    topk = only([n for n in g.node if n.name == L["topk"]], "topk")
    assert topk.input[0] in rest_inputs
    attrs = {a.name: onnx.helper.get_attribute_value(a) for a in topk.attribute}
    assert attrs.get("largest", 1) == 1 and attrs.get("sorted", 1) == 1, attrs
    # the per-level NMS: its boxes trace back to this level's decoded boxes and its scores to this
    # level's TopK, with no other NMS anywhere upstream (the 80 per-class NMS also reach these
    # boxes, but only through the per-level NMS and the post-NMS TopK)
    def upstream(n, i):
        return back_until([n.input[i]])

    nms = only([n for n in all_nms
                if L["box_out"] in {o for x in upstream(n, 0) for o in x.output}
                and topk.name in {x.name for x in upstream(n, 1)}
                and not any(x.op_type == "NonMaxSuppression" for i in (0, 1) for x in upstream(n, i))], "nms")
    pd_region = {x.name for x in back_until([L["box_out"]])}  # decode itself, and TopK
    region = [x for x in back_until(nms.input[:2]) if x.name not in pd_region and x.op_type != "NonMaxSuppression"]
    iou = float(cv[nms.input[3]].reshape(-1)[0])
    max_out = int(cv[nms.input[2]].reshape(-1)[0])
    assert len(nms.input) == 4 or not nms.input[4], "score_threshold present"
    nattr = {a.name: onnx.helper.get_attribute_value(a) for a in nms.attribute}
    assert nattr.get("center_point_box", 0) == 0
    # the min-size filter: Less(x, 0.0) pairs -> Not -> And -> Cast -> NonZero, on a Q/DQ'd w/h
    less =[n for n in region if n.op_type == "Less"]
    assert len(less) == 2 and all(float(cv[x.input[1]]) == 0.0 for x in less), [x.name for x in less]
    assert sum(n.op_type == "NonZero" for n in region) == 1
    # w/h grid: the DequantizeLinear feeding each Less (all Q/DQs in the filter must be uint8)
    lq = {(float(cv[prod[x.input[0]].input[1]]), int(cv[prod[x.input[0]].input[2]]),
           str(cv[prod[x.input[0]].input[2]].dtype)) for x in less if prod[x.input[0]].op_type == "DequantizeLinear"}
    assert len(lq) == 1, lq
    fs, fz, fdt = lq.pop()
    qs = {str(cv[n.input[2]].dtype) for n in region if n.op_type in ("QuantizeLinear", "DequantizeLinear")}
    assert fdt == "uint8" and qs == {"uint8"}, (fdt, qs)
    # after NMS: Gather(sel, 2) -> Squeeze -> Slice [0:cap]
    sl = only(fwd_until(nms.output[0], "Slice")[0], "post-nms slice")
    cap = int(cv[sl.input[2]].reshape(-1)[0])
    assert int(cv[sl.input[1]].reshape(-1)[0]) == 0
    levels.append(dict(topk=topk.name, scores=topk.input[0], k=L["k"], topk_vals=topk.output[0],
                       topk_idx=topk.output[1], box_out=L["box_out"], nms=nms.name, nms_sel=nms.output[0],
                       iou=iou, max_out=max_out, filt_s=fs, filt_z=fz, cap=cap,
                       deltas=L["deltas_in"], nchw_q=L["nchw_q"]))

# the merge: Concat(level scores) -> TopK(k = ReduceMin([cap, N])) -> Gather(Concat boxes) -> Q/DQ
level_topks = {L["topk"] for L in levels}


def post_topk(L):  # the TopK whose *values* input traces back to this level's NMS with no
    # TopK in between other than the per-level ones (the final detection cap sits after this one)
    return only([t for t in fwd_until(L["nms_sel"], "TopK")[0]
                 if L["nms"] in {x.name for x in back_until([t.input[0]])}
                 and not any(x.op_type == "TopK" and x.name not in level_topks
                             for x in back_until([t.input[0]]))], "post-nms topk")


post = post_topk(levels[0])
assert all(post_topk(L).name == post.name for L in levels)
pk = prod[post.input[1]]
while pk.op_type != "ReduceMin":
    pk = prod[pk.input[0]]
cc = prod[pk.input[0]]
while cc.op_type != "Concat":
    cc = prod[cc.input[0]]
post_cap = int(only([cv[i] for i in cc.input if i in cv], "post cap").reshape(-1)[0])
gb = only([c for c in cons[post.output[1]] if c.op_type == "Gather"], "final gather")
q = only(cons[gb.output[0]], "final q")
dq = only(cons[q.output[0]], "final dq")
assert (q.op_type, dq.op_type) == ("QuantizeLinear", "DequantizeLinear")
assert abs(float(cv[q.input[1]]) - pd_levels[0]["box_q"][0]) < 1e-6 and int(cv[q.input[2]]) == 0  # box grid
final = dq.output[0]
print("final proposal tensor:", final, "post cap:", post_cap)
for i, L in enumerate(levels):
    print(f"level{i}: k={L['k']} nms iou={L['iou']} max_out={L['max_out']} cap={L['cap']} "
          f"filter grid=({L['filt_s']}, {L['filt_z']} uint8)")

with open(args.out / "model.txt", "w") as f:  # one line per level, then the post cap
    for L in levels:
        f.write(f"{L['k']} {L['iou']!r} {L['max_out']} {L['cap']} {L['filt_s']!r} {L['filt_z']}\n")
    f.write(f"{post_cap}\n")
json.dump(dict(levels=levels, post_topk=post.name, post_idx=post.output[1], final=final),
          open(args.out / "span.json", "w"), indent=1)

# the span as its own model, for the ORT baselines: exactly the 10 region inputs -> proposal list
need, nodes, st = set(), set(), [final]
while st:
    t = st.pop()
    if t in rest_inputs:
        need.add(t)
        continue
    n = prod.get(t)
    if n is None or n.name in nodes:
        continue
    nodes.add(n.name)
    st.extend(i for i in n.input if i)
order_in = [x for L in levels for x in (L["scores"], L["deltas"])]
assert need == set(order_in), (need, order_in)
onnx.utils.extract_model(args.rest, str(args.out / "rpn_region.onnx"), order_in, [final])
with open(args.out / "rpn_region_io.txt", "w") as f:  # for ort_rpn_bench.c: name file ndim dims...
    for i, (L, pl) in enumerate(zip(levels, pd_levels)):
        f.write(f"{L['scores']} l{i}_scores.bin 2 1 {pl['A']}\n")
        f.write(f"{L['deltas']} l{i}_deltas.bin 3 1 {pl['A']} 4\n")
    f.write(f"OUT {final} proposals.bin\n")
print("rpn_region.onnx:", sum(1 for n in onnx.load(args.out / "rpn_region.onnx").graph.node
                                if n.op_type != "Constant"), "non-Constant nodes; inputs", order_in)

m = onnx.load(args.model)
have = {o.name for o in m.graph.output}
names = [final, post.output[1]]
for L in levels:
    names += [L["scores"], L["deltas"], L["nchw_q"], L["topk_vals"], L["topk_idx"], L["box_out"], L["nms_sel"]]
for nm in names:
    if nm not in have:
        m.graph.output.append(onnx.helper.make_empty_tensor_value_info(nm))
s = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
for img_path in args.images:
    d = args.out / Path(img_path).stem
    d.mkdir(exist_ok=True)
    img = canvas(Path(img_path), 800, 1088)
    outs = dict(zip(names, s.run(names, {s.get_inputs()[0].name: img})))
    for i, L in enumerate(levels):
        outs[L["scores"]].astype(np.float32).reshape(-1).tofile(d / f"l{i}_scores.bin")
        outs[L["deltas"]].astype(np.float32).reshape(-1).tofile(d / f"l{i}_deltas.bin")
        outs[L["nchw_q"]].tofile(d / f"l{i}_nchw_q.bin")
        outs[L["topk_vals"]].astype(np.float32).reshape(-1).tofile(d / f"l{i}_topk_vals.bin")
        outs[L["topk_idx"]].astype(np.int64).reshape(-1).tofile(d / f"l{i}_topk_idx.bin")
        outs[L["box_out"]].astype(np.float32).reshape(-1).tofile(d / f"l{i}_boxes.bin")
        sel = outs[L["nms_sel"]].astype(np.int64)
        sel[:, 2].astype(np.int32).tofile(d / f"l{i}_nms_sel.bin")
    outs[post.output[1]].astype(np.int64).reshape(-1).tofile(d / "post_idx.bin")
    prop = outs[final].astype(np.float32)
    prop.reshape(-1).tofile(d / "proposals.bin")
    kept = [int(np.fromfile(d / f"l{i}_nms_sel.bin", np.int32).size) for i in range(len(levels))]
    print(f"{d.name}: nms kept per level {kept} -> {sum(kept)}; proposals {prop.shape}")
