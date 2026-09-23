#!/usr/bin/env python3
"""Find where an HTP (fp16) run of a piece drifts from fp32: expose every float intermediate.

  bisect_precision.py make <work> <piece> [substr...]
                                            -> <work>/<piece>.dbg.onnx (every float tensor, or those
                                               whose name contains a substr, as extra outputs) +
                                               <piece>.in/dbgref_<tensor>.bin from ORT CPU
Extra outputs change what the HTP compiler fuses, so exposing everything can hide the problem
(it did for the encoder: all-outputs gave cos 0.99999, the plain graph 0.916); expose a few.
  bisect_precision.py report <work> <piece>  -> per-tensor cosine of the phone run, in graph order
Run the phone step between the two with ./run_phone.sh <work> <piece> .dbg.
"""
import re
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

cmd, w, piece = sys.argv[1:4]
pick = sys.argv[4:]
w = Path(w)
if cmd == "make":
    m = onnx.shape_inference.infer_shapes(onnx.load(str(w / f"{piece}.sim.onnx")))
    vi = {v.name: v for v in m.graph.value_info}
    have = {o.name for o in m.graph.output}
    for n in m.graph.node:
        for o in n.output:
            v = vi.get(o)
            if (v is not None and o not in have and v.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
                    and (not pick or any(p in o for p in pick))):
                m.graph.output.append(v)
                have.add(o)
    onnx.save(m, str(w / f"{piece}.dbg.onnx"))
    feeds = {}
    for line in (w / f"{piece}.in" / "manifest.txt").read_text().splitlines():
        name, _, path, dims = line.split()
        feeds[name] = np.fromfile(path, np.float32).reshape([int(d) for d in dims.split(",")])
    s = ort.InferenceSession(str(w / f"{piece}.dbg.onnx"), providers=["CPUExecutionProvider"])
    for o, v in zip(s.get_outputs(), s.run(None, feeds)):
        v.astype(np.float32).tofile(w / f"{piece}.in" / f"dbgref_{o.name.replace('/', '_')}.bin")
    print(f"{len(m.graph.output)} outputs")
else:
    import subprocess

    for f in w.glob(f"{piece}.dbg.htp.o*.bin"):
        f.unlink()
    txt = (w / f"{piece}.dbg.htp.out").read_text()
    r = __import__("os").environ.get("R", "/data/local/tmp/bevformer_tiny")
    for i in re.findall(r"^out (\d+) ", txt, re.M):
        subprocess.run(["adb", "-s", "239dbd8f", "pull", "-q", f"{r}/out_htp_o{i}.bin",
                        str(w / f"{piece}.dbg.htp.o{i}.bin")], check=True)
    prod = {o: n for n in onnx.load(str(w / f"{piece}.dbg.onnx"), load_external_data=False).graph.node for o in n.output}
    for i, name in re.findall(r"^out (\d+) (\S+) f32", txt, re.M):
        ref = np.fromfile(w / f"{piece}.in" / f"dbgref_{name.replace('/', '_')}.bin", np.float32).astype(np.float64)
        got = np.fromfile(w / f"{piece}.dbg.htp.o{i}.bin", np.float32).astype(np.float64)
        if ref.size != got.size:
            print(f"{name}: size mismatch")
            continue
        cos = ref @ got / (np.linalg.norm(ref) * np.linalg.norm(got) + 1e-30)
        n = prod.get(name)
        print(f"{cos:9.5f}  max|ref| {np.abs(ref).max():9.3g}  max err {np.abs(ref - got).max():9.3g}  "
              f"{n.op_type if n else '':18s} {name}")
