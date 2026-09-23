#!/usr/bin/env python3
"""Preprocess images exactly as scripts/android/maskrcnn_e2e/ does (eval_common.canvas: resize to
fit 800x1088, BGR, mean subtraction, zero pad) into raw fp32 [3,800,1088] files, and compute the
all-ONNX-Runtime reference detections (the whole maskrcnn_sim.onnx on host ORT CPU).

  python prepare_inputs.py maskrcnn_sim.onnx OUT_DIR image.jpg...
writes OUT_DIR/<stem>.bin and OUT_DIR/ref/<stem>_{0..3}.npy (boxes, labels, scores, masks)
"""

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "maskrcnn_e2e"))
from eval_common import canvas  # noqa: E402


def main():
    model, out, images = sys.argv[1], Path(sys.argv[2]), sys.argv[3:]
    (out / "ref").mkdir(parents=True, exist_ok=True)
    so = ort.SessionOptions()
    so.log_severity_level = 3
    s = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])
    for im in images:
        x = canvas(Path(im), 800, 1088)
        stem = Path(im).stem
        x.astype(np.float32).tofile(out / f"{stem}.bin")
        res = s.run(None, {"image": x})
        for k, r in enumerate(res):
            np.save(out / "ref" / f"{stem}_{k}.npy", r)
        print(stem, "detections >0.5:", int((res[2] > 0.5).sum()), flush=True)


if __name__ == "__main__":
    main()
