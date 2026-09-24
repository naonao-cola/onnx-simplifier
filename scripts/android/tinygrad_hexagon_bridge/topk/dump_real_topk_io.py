#!/usr/bin/env python3
"""Run one real image through the full MaskRCNN-12-qdq model in ONNX Runtime and capture every
TopK node's real input scores, real k, and real (values, indices) outputs, plus attributes, into
topk_real.npz / topk_meta.json -- the ground truth gen_topk_test_data.py consumes.

    python dump_real_topk_io.py --model MaskRCNN-12-qdq.onnx --image 000000000139.jpg --out DIR
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

m = onnx.load(args.model)
tks = [n for n in m.graph.node if n.op_type == "TopK"]
names = []
for n in tks:
    for t in [*n.input, *n.output]:
        if t not in names:
            names.append(t)
existing = {o.name for o in m.graph.output}
for nm in names:
    if nm not in existing:
        m.graph.output.append(onnx.helper.make_empty_tensor_value_info(nm))
img = canvas(Path(args.image), 800, 1088)
s = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
outs = dict(zip(names, s.run(names, {s.get_inputs()[0].name: img})))
save, meta = {}, {"nodes": []}
for i, n in enumerate(tks):
    a = {x.name: onnx.helper.get_attribute_value(x) for x in n.attribute}
    x, k = outs[n.input[0]], outs[n.input[1]]
    vals, idx = outs[n.output[0]], outs[n.output[1]]
    save[f"n{i}_x"], save[f"n{i}_k"], save[f"n{i}_vals"], save[f"n{i}_idx"] = x, k, vals, idx
    meta["nodes"].append({"name": n.name, "attrs": a, "x_shape": list(x.shape), "x_dtype": str(x.dtype),
                          "k": int(k.reshape(-1)[0]), "idx_dtype": str(idx.dtype)})
    print(n.name, a, "x", x.shape, x.dtype, "k", int(k.reshape(-1)[0]), "-> vals", vals.shape, "idx", idx.shape, idx.dtype)
np.savez(args.out / "topk_real.npz", **save)
json.dump(meta, open(args.out / "topk_meta.json", "w"), indent=1)
