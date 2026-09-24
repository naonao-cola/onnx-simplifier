#!/usr/bin/env python3
"""Build one single-node NonMaxSuppression ONNX model per real call (same opset/attribute, and the
real max_output_boxes_per_class / iou_threshold as initializers, like rest.onnx) plus raw inputs,
and time each in ONNX Runtime CPU on the host -- the op's current execution path, since the whole
rest.onnx remainder runs on ORT CPU today. The models/inputs are also what ort_nms_bench.c times on
the phone's CPU. Writes `{g}_callN.onnx`, `{g}_callN_boxes.bin`, `{g}_callN_scores.bin` and
`ort_calls.txt` (`group index n n_selected` per call).

    python make_nms_single_node_models.py NMS_DUMP_DIR OUT_DIR
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
d = np.load(src / "nms_real.npz")
meta = json.load(open(src / "nms_meta.json"))
lines, tot, cnt = [], {}, {}
for n in meta["nodes"]:
    ins = n["inputs"]
    boxes, scores = d[ins[0]], d[ins[1]]
    mo, iou = d[ins[2]], d[ins[3]]
    g = "level" if iou.reshape(-1)[0] == np.float32(0.7) else "class"
    i = cnt[g] = cnt.get(g, -1) + 1
    node = helper.make_node("NonMaxSuppression", ["boxes", "scores", "max_out", "iou"], ["sel"], center_point_box=0)
    graph = helper.make_graph([node], "nms", [
        helper.make_tensor_value_info("boxes", TensorProto.FLOAT, list(boxes.shape)),
        helper.make_tensor_value_info("scores", TensorProto.FLOAT, list(scores.shape))],
        [helper.make_tensor_value_info("sel", TensorProto.INT64, None)],
        [numpy_helper.from_array(mo.astype(np.int64), "max_out"), numpy_helper.from_array(iou.astype(np.float32), "iou")])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    m.ir_version = 7
    onnx.save(m, out / f"{g}_call{i}.onnx")
    boxes.astype(np.float32).tofile(out / f"{g}_call{i}_boxes.bin")
    scores.astype(np.float32).tofile(out / f"{g}_call{i}_scores.bin")
    feeds = {"boxes": boxes, "scores": scores}
    for th in (1, 0):
        so = ort.SessionOptions(); so.intra_op_num_threads = th
        s = ort.InferenceSession(str(out / f"{g}_call{i}.onnx"), so, providers=["CPUExecutionProvider"])
        assert np.array_equal(s.run(None, feeds)[0], d[n["output"]])
        ts = []
        for _ in range(15):
            t = time.perf_counter(); s.run(None, feeds); ts.append(time.perf_counter() - t)
        tot[(g, th)] = tot.get((g, th), 0.0) + sorted(ts)[7] * 1e3
    lines.append(f"{g} {i} {boxes.shape[1]} {len(d[n['output']])}")
(out / "ort_calls.txt").write_text("\n".join(lines) + "\n")
for g in ("level", "class"):
    print(f"host ORT {g}: {cnt[g] + 1} calls, sum of medians 1thr={tot[(g, 1)]:.3f}ms default={tot[(g, 0)]:.3f}ms")
