#!/usr/bin/env python3
"""Host ORT run of a (quantized) RF-DETR ONNX over the 20 eval images vs the fp32 library model.
usage: host_eval.py <variant> <model.onnx>   (graph optimizations off: ORT's fused int8 kernels
saturate on CPUs without VNNI, so this measures the quantization, not the host CPU)"""

import sys

import common as C
import export
import onnxruntime as ort

variant, path = sys.argv[1], sys.argv[2]
paths = C.image_paths("eval")
ref = export.refs(variant, paths)
res = int(ref[0]["res"])
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
name = s.get_inputs()[0].name
u8 = s.get_inputs()[0].type == "tensor(uint8)"
pairs = []
for p, r in zip(paths, ref):
    rgb = C.load_rgb_u8(p, res)
    lg, bx = s.run(["logits", "boxes"], {name: rgb[None] if u8 else C.to_pixels(rgb)})
    pairs.append(((r["logits"], r["boxes"]), (lg, bx)))
t = export.tally(pairs, res)
print(
    f"{path.split('/')[-1]} (host): matched {t['matched']}/{t['ref']} (det {t['det']})"
)
