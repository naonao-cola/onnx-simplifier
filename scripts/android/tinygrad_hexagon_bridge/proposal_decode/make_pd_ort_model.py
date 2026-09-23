#!/usr/bin/env python3
"""Cut the exact proposal-decode region (all 5 FPN levels) out of the real rest.onnx as one ONNX
model -- inputs: each level's squeezed TopK indices + per-anchor deltas; outputs: each level's
grid-quantized boxes -- so ONNX Runtime's cost for precisely the nodes the kernel replaces can be
timed (host here, phone via ort_pd_bench.c) in a single Run, the way it executes inside rest.onnx.

    python make_pd_ort_model.py --rest rest.onnx --data DATA_DIR   # writes DATA_DIR/pd_region.onnx
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx.utils import Extractor

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--rest", required=True)
p.add_argument("--data", type=Path, required=True)
args = p.parse_args()
levels = json.load(open(args.data / "levels.json"))
ins = [x for L in levels for x in (L["idx"], L["deltas_in"])]
outs = [L["box_out"] for L in levels]
m = Extractor(onnx.load(args.rest)).extract_model(ins, outs)
ops = [n.op_type for n in m.graph.node]
print(
    f"region: {len(ops)} nodes ({len([o for o in ops if o != 'Constant'])} non-Constant):",
    {o: ops.count(o) for o in sorted(set(ops))},
)
onnx.save(m, args.data / "pd_region.onnx")

feeds = {}
for i, L in enumerate(levels):
    feeds[L["idx"]] = np.fromfile(args.data / f"l{i}_idx64.bin", np.int64)
    feeds[L["deltas_in"]] = np.fromfile(
        args.data / f"l{i}_deltas.bin", np.float32
    ).reshape(1, L["A"], 4)
for th in (1, 0):
    so = ort.SessionOptions()
    so.intra_op_num_threads = th
    s = ort.InferenceSession(
        str(args.data / "pd_region.onnx"), so, providers=["CPUExecutionProvider"]
    )
    res = s.run(outs, feeds)
    for i, r in enumerate(res):
        ref = np.fromfile(args.data / f"l{i}_ref.bin", np.float32).reshape(r.shape)
        assert np.array_equal(r, ref), i
    ts = []
    for _ in range(21):
        t = time.perf_counter()
        s.run(outs, feeds)
        ts.append(time.perf_counter() - t)
    print(
        f"host ORT CPU, intra_op_threads={th or 'default'}: median {sorted(ts)[10] * 1e3:.3f} ms (outputs match the full-model run)"
    )
# io file for the phone bench: "name dtype file ndims dims..." per input, then "OUT name ref_file n"
with open(args.data / "pd_region_io.txt", "w") as f:
    for i, L in enumerate(levels):
        f.write(f"{L['idx']} i64 l{i}_idx64.bin 1 {L['k']}\n")
        f.write(f"{L['deltas_in']} f32 l{i}_deltas.bin 3 1 {L['A']} 4\n")
    for i, L in enumerate(levels):
        f.write(f"OUT {L['box_out']} l{i}_ref.bin {L['k'] * 4}\n")
