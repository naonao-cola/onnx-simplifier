#!/usr/bin/env python3
"""sensitivity.py <variant> [--method M] [--keep K ...]: which encoder nodes int8 hurts.

1. `onnxsim.pick_calibration` (whole-graph QDQ) scores the calibration methods on held-out
   calibration images with an embedding-cosine metric (the mask decoder reads the embedding).
2. Calibrate once (`--method`), then for every quantizable node build "only this node int8"
   (onnxsim.full_qdq with the cached ranges, all other nodes float) and record the embedding
   cosine vs fp32 on 2 eval images: the per-node cost of int8.
3. For each K in --keep: everything int8 except the K worst nodes (kept float = fp16 on the HTP)
   -> enc.int8keep<K>.onnx (+ quant json, uint8 I/O), evaluated by `sam.py host/phone`.
Results: <work>/<variant>/sensitivity.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import sam


def emb_cos(a, b):
    return float(np.mean([sam.cos(x, y) for x, y in zip(a, b)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant")
    ap.add_argument("--method", default="percentile")
    ap.add_argument("--keep", type=int, nargs="*", default=[4, 8, 16])
    ap.add_argument("--skip-sweep", action="store_true", help="reuse sensitivity.json's order")
    ap.add_argument("--pick", action="store_true", help="also run onnxsim.pick_calibration")
    a = ap.parse_args()
    import onnx

    from onnxsim.calibration import calibrate
    from onnxsim.full_qdq import quantize_full_qdq, quantized_io

    d = sam.wdir(a.variant)
    r = d / "ref"
    m = onnx.load(d / "enc.sim.onnx")
    calib = [{"pixels": sam.nchw(np.load(r / f"calib_{i}_img.npy"))} for i in sam.CALIB_IDS]
    ev_ids = sam.EVAL_IDS[:2]
    ev = [{"pixels": sam.nchw(np.load(r / f"eval_{i}_img.npy"))} for i in ev_ids]
    ref = [np.load(r / f"eval_{i}_emb.npy") for i in ev_ids]
    out = {}
    if a.pick:
        from onnxsim.calibration_pick import pick_calibration

        t = time.time()
        res = pick_calibration(m, calib[:12], calib[12:],
                               metric=lambda f, q: float(np.mean([  # one {name: array}/batch
                                   sam.cos(x["image_embeddings"], y["image_embeddings"])
                                   for x, y in zip(f, q)])),
                               full_graph=True, verbose=True)
        out["pick"] = {"best": res.method, "scores": res.scores}
        print("pick_calibration", out["pick"], f"{time.time() - t:.0f} s")
    floats = [o for n in m.graph.node for o in n.output]
    t = time.time()
    meth, _, pct = a.method.partition(":")  # "percentile:99.99" as pick_calibration names it
    kw = {"percentile": float(pct)} if pct else {}
    ranges = calibrate(m, calib, method=meth, extra_tensor_names=floats, **kw)
    ranges["pixels"] = (0.0, 255.0)
    print(f"calibrated {len(ranges)} tensors ({a.method}) in {time.time() - t:.0f} s")
    names = [n.name for n in m.graph.node if n.op_type in ("Conv", "Gemm", "MatMul", "Gelu",
                                                            "Add", "Mul", "LayerNormalization",
                                                            "Resize", "Sigmoid", "Relu")]
    sens = {}
    if a.skip_sweep:
        prev = json.loads((d / "sensitivity.json").read_text())
        out, sens, names = prev, prev["sensitivity"], []
    for k, nm in enumerate(names):
        q = quantize_full_qdq(m, None, ranges=ranges, method=meth,
                              exclude_nodes=[x for x in names if x != nm] + [
                                  n.name for n in m.graph.node if n.name not in names])
        s = sam.ort_sess_bytes(q.SerializeToString())
        got = [s.run(None, x)[0] for x in ev]
        sens[nm] = emb_cos(ref, got)
        if k % 20 == 0:
            print(f"  {k}/{len(names)} {nm} {sens[nm]:.5f}", flush=True)
    worst = sorted(sens, key=sens.get)
    out.update({"method": a.method, "sensitivity": sens, "worst": worst[:32]})
    print("worst:", [(w, round(sens[w], 4)) for w in worst[:12]])
    for K in a.keep:
        q = quantize_full_qdq(m, None, ranges=ranges, method=meth, exclude_nodes=worst[:K])
        q, io = quantized_io(q, inputs=["pixels"], outputs=["image_embeddings"],
                             nhwc_inputs=["pixels"])
        tag = f"int8keep{K}" + ("" if a.method == "percentile" else "_" + a.method.replace(":", ""))
        onnx.save(q, d / f"enc.{tag}.onnx")
        (d / f"quant_{tag}.json").write_text(json.dumps({"enc": io, "dec": None}, indent=1,
                                                        default=str))
        print("wrote", tag)
    (d / "sensitivity.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    sys.exit(main())
