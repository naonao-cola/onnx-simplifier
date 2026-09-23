#!/usr/bin/env python3
"""Locate the RPN proposal-decode region of every FPN level in the real Mask R-CNN graph and
capture its real inputs/outputs from one ONNX Runtime run of the full model.

Per level, the region this kernel replaces is (all in rest.onnx, after the per-level TopK):
    Squeeze(TopK indices) -> Gather anchors / Gather deltas -> Q/DQ deltas -> Reshape -> Q/DQ ->
    Slice dx,dy,dw,dh -> width/height/center (+1 convention) -> Clip(dw,dh <= log(1000/16)) -> Exp
    -> box corners (x2/y2 get -1) -> Concat -> Clip x to [0,xmax], y to [0,ymax] -> Q/DQ onto the
    uint8 box grid.
The output tensor is the grid-quantized [k,4] boxes that feed the min-size filter and NMS.

It also captures the backbone's pre-transpose conv output for the deltas (NCHW, uint8), so the
kernel can gather straight from it and the backbone's full-map Transpose/Reshape can be priced.

    python capture_proposal_decode.py --model maskrcnn_sim.onnx --rest rest.onnx \
        --image img_000000000139.jpg --out DIR
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
p.add_argument(
    "--model", required=True, help="full simplified model (maskrcnn_sim.onnx)"
)
p.add_argument("--rest", required=True, help="rest.onnx from prepare.py")
p.add_argument("--image", required=True)
p.add_argument("--out", type=Path, required=True)
args = p.parse_args()
args.out.mkdir(parents=True, exist_ok=True)

full = onnx.load(args.model)
fg = full.graph
consts = {t.name: numpy_helper.to_array(t) for t in fg.initializer}
for n in fg.node:
    if n.op_type == "Constant":
        consts[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)
prod = {o: n for n in fg.node for o in n.output}
cons = collections.defaultdict(list)
for n in fg.node:
    for i in n.input:
        cons[i].append(n)
rest_inputs = [i.name for i in onnx.load(args.rest).graph.input]


def only(xs, what):
    assert len(xs) == 1, (what, [x.name for x in xs])
    return xs[0]


def cval(name):
    return consts[name]


levels = []
for n in fg.node:
    if n.op_type != "TopK" or n.input[0] not in rest_inputs:
        continue
    k = int(cval(n.input[1]).reshape(-1)[0])
    sq = only([c for c in cons[n.output[1]] if c.op_type == "Squeeze"], "squeeze")
    idx = sq.output[0]
    gathers = [c for c in cons[idx] if c.op_type == "Gather"]
    ga = only([g for g in gathers if g.input[0] in consts], "anchor gather")
    gd = only([g for g in gathers if g.input[0] in rest_inputs], "delta gather")
    anchors = cval(ga.input[0]).reshape(-1, 4).astype(np.float32)
    # delta Q/DQ chain: Gather -> Q -> DQ -> Reshape -> Q -> DQ
    q1 = only(cons[gd.output[0]], "q1")
    dq1 = only(cons[q1.output[0]], "dq1")
    rs = only(cons[dq1.output[0]], "reshape")
    q2 = only(cons[rs.output[0]], "q2")
    dq2 = only(cons[q2.output[0]], "dq2")
    assert (q1.op_type, dq1.op_type, rs.op_type, q2.op_type, dq2.op_type) == (
        "QuantizeLinear",
        "DequantizeLinear",
        "Reshape",
        "QuantizeLinear",
        "DequantizeLinear",
    )
    dqp = [
        (float(cval(q.input[1])), int(cval(q.input[2])), str(cval(q.input[2]).dtype))
        for q in (q1, q2)
    ]
    # box clip + grid Q/DQ: find the Clip nodes bounding coordinates, then Unsqueeze->Concat->Reshape->Q->DQ
    seen, q, clips = set(), [dq2], []
    while q:
        x = q.pop()
        if x.name in seen:
            continue
        seen.add(x.name)
        if (
            x.op_type == "Clip"
            and x.input[1] in consts
            and float(cval(x.input[1])) == 0.0
        ):
            clips.append(x)
            continue
        for o in x.output:
            q.extend(
                c for c in cons[o] if c.op_type not in ("NonMaxSuppression", "TopK")
            )
    assert len(clips) == 2, [c.name for c in clips]
    clip_bounds = {}
    for c in clips:
        # Slice over the [x1,y1,x2,y2] axis: start 0 -> x's, start 1 -> y's
        sl = prod[c.input[0]]
        start = int(cval(sl.input[1]).reshape(-1)[0])
        clip_bounds["x" if start == 0 else "y"] = float(cval(c.input[2]))
    un = only(cons[clips[0].output[0]], "unsq")
    cc = only(cons[un.output[0]], "concat")
    rs2 = only(cons[cc.output[0]], "reshape2")
    bq = only(cons[rs2.output[0]], "boxq")
    bdq = only(
        [c for c in cons[bq.output[0]] if c.op_type == "DequantizeLinear"], "boxdq"
    )
    box_q = (
        float(cval(bq.input[1])),
        int(cval(bq.input[2])),
        str(cval(bq.input[2]).dtype),
    )
    # the -1/+1 and 0.5 and the dw/dh clip constant are read from the graph, not assumed
    exp_clip = sorted(
        {
            float(cval(x.input[2]))
            for x in fg.node
            if x.name in seen
            and x.op_type == "Clip"
            and x.input[1] in consts
            and float(cval(x.input[1])) < -1e30
        }
    )
    assert len(exp_clip) == 1, exp_clip
    # backbone side: rest delta input <- DQ <- Q <- Reshape <- DQ <- Q <- Transpose <- DQ(<nchw>)
    x = gd.input[0]
    chain = []
    while x in prod and prod[x].op_type != "Transpose":
        chain.append(prod[x].op_type)
        x = prod[x].input[0]
    tr = prod[x]
    perm = [int(v) for v in onnx.helper.get_attribute_value(tr.attribute[0])]
    tr_dq = prod[tr.input[0]]  # DQ of the 5-D [1,A,4,H,W] tensor
    # walk up through Q/DQ to the Reshape of the 4-D conv output; its input is a DQ whose
    # quantized input is the NCHW uint8 tensor the kernel can gather from directly
    y = tr_dq.input[0]
    while prod[y].op_type != "Reshape":
        y = prod[y].input[0]
    y = prod[y].input[0]
    nchw_q = prod[y].input[0] if prod[y].op_type == "DequantizeLinear" else y
    bb_dq = (
        float(cval(tr_dq.input[1])),
        int(cval(tr_dq.input[2])),
        str(cval(tr_dq.input[2]).dtype),
    )
    levels.append(
        dict(
            topk=n.name,
            k=k,
            idx=idx,
            anchor_const=ga.input[0],
            deltas_in=gd.input[0],
            delta_qdq=dqp,
            clip_x=clip_bounds["x"],
            clip_y=clip_bounds["y"],
            exp_clip=exp_clip[0],
            box_q=box_q,
            box_pre=rs2.input[0],
            box_out=bdq.output[0],
            transpose_perm=perm,
            backbone_chain=chain,
            nchw_q=nchw_q,
            nchw_dq=bb_dq,
            A=int(anchors.shape[0]),
        )
    )

levels.sort(key=lambda L: -L["A"])  # P2 (most anchors) first
anchor_arr = {}
for L in levels:
    anchor_arr[L["topk"]] = cval(L["anchor_const"]).reshape(-1, 4).astype(np.float32)

names = []
for L in levels:
    for key in ("idx", "deltas_in", "box_pre", "box_out", "nchw_q"):
        if L[key] not in names:
            names.append(L[key])
m = onnx.load(args.model)
have = {o.name for o in m.graph.output}
for nm in names:
    if nm not in have:
        m.graph.output.append(onnx.helper.make_empty_tensor_value_info(nm))
img = canvas(Path(args.image), 800, 1088)
s = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
outs = dict(zip(names, s.run(names, {s.get_inputs()[0].name: img})))

for i, L in enumerate(levels):
    a = anchor_arr[L["topk"]]
    idx = outs[L["idx"]].astype(np.int64).reshape(-1)
    d = outs[L["deltas_in"]].reshape(-1, 4).astype(np.float32)
    nq = outs[L["nchw_q"]]
    L["nchw_shape"] = list(nq.shape)
    L["nchw_dtype"] = str(nq.dtype)
    a.tofile(args.out / f"l{i}_anchors.bin")
    idx.astype(np.int32).tofile(args.out / f"l{i}_idx.bin")
    idx.tofile(args.out / f"l{i}_idx64.bin")
    d.tofile(args.out / f"l{i}_deltas.bin")
    nq.tofile(args.out / f"l{i}_nchw_q.bin")
    outs[L["box_pre"]].astype(np.float32).tofile(args.out / f"l{i}_box_pre.bin")
    outs[L["box_out"]].astype(np.float32).tofile(args.out / f"l{i}_ref.bin")
    print(
        f"level{i}: A={L['A']} k={L['k']} nchw={nq.shape} {nq.dtype} delta_qdq={L['delta_qdq']} "
        f"clip=({L['clip_x']},{L['clip_y']}) exp_clip={L['exp_clip']} box_q={L['box_q']} "
        f"perm={L['transpose_perm']} backbone_chain={L['backbone_chain']} out={outs[L['box_out']].shape}"
    )
json.dump(levels, open(args.out / "levels.json", "w"), indent=1)
# one line per level for the C harnesses (host check, qemu, phone client):
# A k H W s1 z1 s2 z2 exp_clip clip_x clip_y box_s box_z bb_s bb_z
with open(args.out / "levels.txt", "w") as f:
    for L in levels:
        (s1, z1, _), (s2, z2, _) = L["delta_qdq"]
        f.write(
            " ".join(
                str(v)
                for v in (
                    L["A"],
                    L["k"],
                    L["nchw_shape"][2],
                    L["nchw_shape"][3],
                    repr(s1),
                    z1,
                    repr(s2),
                    z2,
                    repr(L["exp_clip"]),
                    repr(L["clip_x"]),
                    repr(L["clip_y"]),
                    repr(L["box_q"][0]),
                    L["box_q"][1],
                    repr(L["nchw_dq"][0]),
                    L["nchw_dq"][1],
                )
            )
            + "\n"
        )
