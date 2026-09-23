#!/usr/bin/env python3
"""Run one real image through the full MaskRCNN-12-qdq model in ONNX Runtime and capture every
NonMaxSuppression node's real inputs (boxes, scores, max_output_boxes_per_class, iou_threshold,
score_threshold) and output (selected_indices) into nms_real.npz / nms_meta.json.

    python dump_real_nms_io.py --model MaskRCNN-12-qdq.onnx --image 000000000139.jpg --out DIR
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "maskrcnn_e2e"))
from eval_common import canvas  # noqa: E402

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--model", required=True)
p.add_argument("--image", required=True)
p.add_argument("--out", type=Path, default=Path("."))
args = p.parse_args()
args.out.mkdir(parents=True, exist_ok=True)


def key(name):
    return "t_" + name.replace("/", "_").replace(":", "_")


m = onnx.load(args.model)
init = {i.name: onnx.numpy_helper.to_array(i) for i in m.graph.initializer}
nms = [n for n in m.graph.node if n.op_type == "NonMaxSuppression"]
names = []
for n in nms:
    for i in [*n.input, n.output[0]]:
        if i and i not in names and i not in init:
            names.append(i)
existing = {o.name for o in m.graph.output}
for nm in names:
    if nm not in existing:
        m.graph.output.append(onnx.helper.make_empty_tensor_value_info(nm))
img = canvas(Path(args.image), 800, 1088)
s = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
outs = s.run(names, {s.get_inputs()[0].name: img})
vals = dict(zip(names, outs))
vals.update({k: v for k, v in init.items() if any(k in n.input for n in nms)})
np.savez(args.out / "nms_real.npz", **{key(k): v for k, v in vals.items()})
meta = {"nodes": [{"name": n.name, "inputs": [key(i) for i in n.input], "output": key(n.output[0]),
                   "center_point_box": next((a.i for a in n.attribute if a.name == "center_point_box"), 0)}
                  for n in nms]}
json.dump(meta, open(args.out / "nms_meta.json", "w"), indent=1)
bykey = {key(k): v for k, v in vals.items()}
for n in meta["nodes"]:
    ins = [bykey.get(i) for i in n["inputs"]] + [None] * (5 - len(n["inputs"]))
    b, sc, mo, iou, st = ins
    print(n["name"], "boxes", b.shape, "scores", sc.shape, "max_out", mo, "iou", iou, "score_thr", st,
          "selected", bykey[n["output"]].shape)
