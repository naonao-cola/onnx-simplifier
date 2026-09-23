#!/usr/bin/env python3
"""Probe which Gather shapes QNN's HTP backend finalizes (fp16 via the QNN EP): writes
<out>/g_<rows>_<n>_<k>.onnx + inputs; k gathers of n indices each from a (rows, 64) table, summed.
usage: probe_gather.py <out> rows n k [c]"""
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import parser

out, rows, n, k = Path(sys.argv[1]), *map(int, sys.argv[2:5])
c = int(sys.argv[5]) if len(sys.argv) > 5 else 64
out.mkdir(parents=True, exist_ok=True)
ins = ", ".join(f"int32[{n}] i{j}" for j in range(k))
body = "\n".join(f"g{j} = Gather<axis=0>(t, i{j})" for j in range(k))
acc = "g0"
for j in range(1, k):
    body += f"\na{j} = Add({acc}, g{j})"
    acc = f"a{j}"
body += f"\ny = Identity({acc})"
m = parser.parse_model(f"""<ir_version: 8, opset_import: ["": 17]>
g (float[{rows},{c}] t, {ins}) => (float[{n},{c}] y) {{ {body} }}""")
tag = f"g_{rows}_{n}_{k}_{c}"
onnx.save(m, out / f"{tag}.onnx")
d = out / f"{tag}.in"
d.mkdir(exist_ok=True)
rng = np.random.default_rng(0)
with open(d / "manifest.txt", "w") as f:
    t = rng.standard_normal((rows, c)).astype(np.float32)
    t.tofile(d / "t.bin")
    f.write(f"t f32 {(d / 't.bin').resolve()} {rows},{c}\n")
    acc = 0
    for j in range(k):
        i = rng.integers(0, rows, n).astype(np.int32)
        i.tofile(d / f"i{j}.bin")
        f.write(f"i{j} i32 {(d / f'i{j}.bin').resolve()} {n}\n")
        acc = acc + t[i]
    np.asarray(acc, np.float32).tofile(d / "ref_y.bin")
print(tag)
