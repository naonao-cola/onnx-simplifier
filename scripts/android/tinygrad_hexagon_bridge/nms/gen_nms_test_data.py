#!/usr/bin/env python3
"""Turn dump_real_nms_io.py's captured real NonMaxSuppression calls into flat binaries for
nms_host_check.c, nms_qemu.c and nms_client.c. The 85 calls split into two groups, matching the
model's structure: `level` (the 5 per-FPN-level RPN NMS calls, iou=0.7) and `class` (the 80
per-class box-head NMS calls, iou=0.5). Per group: `{g}_calls.txt` (one `n iou max_out n_selected`
line per call, in graph order), `{g}_boxes.bin` (all calls' [n,4] fp32 boxes concatenated),
`{g}_scores.bin` (fp32 scores concatenated) and `{g}_ref.bin` (ORT's selected box indices, int32,
concatenated in ORT's output order) and `{g}_thr.bin` (each call's iou_threshold as raw fp32, for the
freestanding qemu harness, which has no float parser)."""
import json
import sys
from pathlib import Path

import numpy as np

src = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
out = Path(sys.argv[2] if len(sys.argv) > 2 else ".")
out.mkdir(parents=True, exist_ok=True)
d = np.load(src / "nms_real.npz")
meta = json.load(open(src / "nms_meta.json"))
groups = {"level": [], "class": []}
for n in meta["nodes"]:
    assert n["center_point_box"] == 0
    ins = n["inputs"] + [""] * (5 - len(n["inputs"]))
    boxes, scores = d[ins[0]], d[ins[1]]
    max_out = int(d[ins[2]].reshape(-1)[0]) if ins[2] in d else 0
    iou = float(d[ins[3]].reshape(-1)[0]) if ins[3] in d else 0.0
    assert ins[4] not in d, "score_threshold not expected in this model"
    assert boxes.shape[0] == 1 and scores.shape[:2] == (1, 1)
    sel = d[n["output"]]
    assert (sel[:, 0] == 0).all() and (sel[:, 1] == 0).all()
    groups["level" if iou == np.float32(0.7) else "class"].append(
        (boxes[0].astype(np.float32), scores[0, 0].astype(np.float32), max_out, iou, sel[:, 2].astype(np.int32)))
for g, calls in groups.items():
    (out / f"{g}_calls.txt").write_text("".join(f"{len(s)} {iou!r} {mo} {len(r)}\n" for b, s, mo, iou, r in calls))
    np.concatenate([b.reshape(-1) for b, *_ in calls]).tofile(out / f"{g}_boxes.bin")
    np.concatenate([s for _, s, *_ in calls]).tofile(out / f"{g}_scores.bin")
    np.concatenate([r for *_, r in calls]).astype(np.int32).tofile(out / f"{g}_ref.bin")
    np.array([iou for *_, iou, _ in calls], dtype=np.float32).tofile(out / f"{g}_thr.bin")
    ns = [len(c[1]) for c in calls]
    print(f"{g}: {len(calls)} calls, boxes total={sum(ns)} max={max(ns)} empty={ns.count(0)} "
          f"selected total={sum(len(c[4]) for c in calls)}")
