#!/usr/bin/env python3
"""int8 (QDQ) MapTR-tiny backbone for the HTP with onnxsim's whole-graph quantizer (the recipe of
../bevformer_tiny/quantize.py: same R50 + FPN at 480x800, different weights).

  quantize.py calib    --data <nuscenes-mini> --work <work> [--frames 3]
      the camera images of the first keyframes of CALIB_SCENES (disjoint from the eval scene-0103)
      -> <work>/calib/<i>.npz
  quantize.py backbone --work <work> [--method mse]
      calibrate backbone1 (one camera per batch) and apply the ranges to backbone6 (same tensor
      names): <work>/backbone6.q8.onnx, uint8 NHWC image in (the host quantizes the normalized image
      where it normalizes it anyway), float feats out; qparams in <work>/backbone6.q8.json.
  quantize.py host     --work <work>
      <work>/msda_frames/<i>/img_u8.u8 from img.f32 with backbone6.q8.json's input qparams.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CALIB_SCENES = ["scene-0061", "scene-0553", "scene-0757", "scene-1077"]  # 1077: night


def calib(a):
    sys.path.insert(0, str(HERE.parent / "bevformer_tiny"))
    from nuscenes import NuScenesMini

    ns = NuScenesMini(a.data)
    out = Path(a.work) / "calib"
    out.mkdir(parents=True, exist_ok=True)
    k = 0
    for scene in CALIB_SCENES:
        for tok in ns.scene_samples(scene)[: a.frames]:
            np.savez(out / f"{k}.npz", img=ns.frame(tok)["img"].numpy())
            k += 1
    print(f"{k} calibration frames ({6 * k} camera images)")


def backbone(a):
    import onnx

    sys.path.insert(0, str(HERE.parents[3]))  # the repo's onnxsim
    import onnxsim.full_qdq as F
    from onnxsim.calibration import calibrate

    work = Path(a.work)
    imgs = [np.load(p)["img"] for p in sorted((work / "calib").glob("*.npz"))]
    one = onnx.load(str(work / "backbone1.sim.onnx"))
    data = [{"img": x[c:c + 1]} for x in imgs for c in range(6)]
    names = [o for n in one.graph.node for o in n.output] + ["img"]
    ranges = calibrate(one, data, method=a.method, extra_tensor_names=names)
    del one, data
    q = F.quantize_full_qdq(onnx.load(str(work / "backbone6.sim.onnx")), ranges=ranges)
    q, info = F.quantized_io(q, inputs=["img"], outputs=[], nhwc_inputs=["img"])
    onnx.save(q, str(work / "backbone6.q8.onnx"))
    (work / "backbone6.q8.json").write_text(json.dumps(info, indent=1))
    print(f"backbone6.q8 ({a.method}): {len(q.graph.node)} nodes, io {info}")


def host(a):
    work = Path(a.work)
    qp = json.loads((work / "backbone6.q8.json").read_text())["img"]
    for d in sorted((work / "msda_frames").iterdir()):
        x = np.fromfile(d / "img.f32", np.float32).reshape(6, 3, 480, 800).transpose(0, 2, 3, 1)
        u = np.clip(np.round(x / qp["scale"]) + qp["zero_point"], 0, 255).astype(np.uint8)
        u.tofile(d / "img_u8.u8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["calib", "backbone", "host"])
    ap.add_argument("--data")
    ap.add_argument("--work", required=True)
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--method", default="mse")
    a = ap.parse_args()
    {"calib": calib, "backbone": backbone, "host": host}[a.cmd](a)


if __name__ == "__main__":
    main()
