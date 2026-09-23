#!/usr/bin/env python3
"""From dump_real_topk_io.py's topk_real.npz/topk_meta.json, write the flat files the host check,
the qemu check and the phone client read (calls.txt: `n k axis` per real TopK node, callN_x.bin
fp32 scores, callN_vals.bin/callN_idx.bin ORT's real outputs), plus one single-node TopK ONNX model
per call (same opset/attributes as the real node) for the ORT baselines, and time those on host ORT.

    python gen_topk_test_data.py SRC_DIR OUT_DIR
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
d = np.load(src / "topk_real.npz")
meta = json.load(open(src / "topk_meta.json"))
lines, tot = [], {1: 0.0, 0: 0.0}
for i, n in enumerate(meta["nodes"]):
    x = d[f"n{i}_x"].astype(np.float32)
    k, vals, idx = n["k"], d[f"n{i}_vals"], d[f"n{i}_idx"]
    x.reshape(-1).tofile(out / f"call{i}_x.bin")
    vals.astype(np.float32).reshape(-1).tofile(out / f"call{i}_vals.bin")
    idx.astype(np.int64).reshape(-1).tofile(out / f"call{i}_idx.bin")
    lines.append(f"{x.size} {k} {n['attrs']['axis']}")
    node = helper.make_node("TopK", ["X", "K"], ["V", "I"], **n["attrs"])
    g = helper.make_graph([node], "topk",
                          [helper.make_tensor_value_info("X", TensorProto.FLOAT, list(x.shape)),
                           helper.make_tensor_value_info("K", TensorProto.INT64, [1])],
                          [helper.make_tensor_value_info("V", TensorProto.FLOAT, None),
                           helper.make_tensor_value_info("I", TensorProto.INT64, None)])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 12)])
    m.ir_version = 7
    onnx.save(m, out / f"call{i}.onnx")
    feeds = {"X": x, "K": np.array([k], np.int64)}
    res = {}
    for th in (1, 0):
        so = ort.SessionOptions()
        so.intra_op_num_threads = th
        s = ort.InferenceSession(str(out / f"call{i}.onnx"), so, providers=["CPUExecutionProvider"])
        v, ix = s.run(None, feeds)
        assert np.array_equal(v, vals) and np.array_equal(ix, idx), f"call{i}: single-node ORT != full-graph ORT"
        ts = []
        for _ in range(21):
            t = time.perf_counter()
            s.run(None, feeds)
            ts.append(time.perf_counter() - t)
        res[th] = sorted(ts)[10] * 1e3
        tot[th] += res[th]
    print(f"call{i} {n['name']} n={x.size} k={k} axis={n['attrs']['axis']} "
          f"host_ort_1thr={res[1]:.4f}ms host_ort_default={res[0]:.4f}ms")
(out / "calls.txt").write_text("\n".join(lines) + "\n")
print(f"TOTAL host_ort_1thr={tot[1]:.3f}ms host_ort_default={tot[0]:.3f}ms")
