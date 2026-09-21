"""Top-1 of ONNX models on a random val subset via ONNX Runtime CPU (no TensorRT).

Separates "the quantized ONNX is itself inaccurate" (ModelOpt scales / calibration) from
"TensorRT mishandles it": if a model collapses here too, TensorRT is not the cause.
Run under any interpreter with onnxruntime.

    python ort_cpu_check.py DATA_DIR MODEL.onnx [MODEL.onnx ...] [--n 150] [--mean .5 .5 .5 --std .5 .5 .5]
"""

import argparse
import time

import numpy as np
import onnxruntime as ort


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("models", nargs="+")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--mean", type=float, nargs=3, default=[0.485, 0.456, 0.406])
    ap.add_argument("--std", type=float, nargs=3, default=[0.229, 0.224, 0.225])
    a = ap.parse_args()
    x = np.load(f"{a.data}/val_x.npy", mmap_mode="r")
    y = np.load(f"{a.data}/val_y.npy")
    idx = np.random.default_rng(1).choice(len(y), a.n, replace=False)
    mean, std = np.array(a.mean, np.float32), np.array(a.std, np.float32)
    for path in a.models:
        s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        name = s.get_inputs()[0].name
        t0, ok = time.time(), 0
        for i in idx:
            inp = ((np.asarray(x[i]).astype(np.float32) / 255 - mean) / std).transpose(2, 0, 1)[None]
            ok += int(s.run(None, {name: inp})[0].argmax() == y[i])
        print(f"{path.split('/')[-1]}: top-1 {ok / a.n * 100:.1f}% on {a.n} val images ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
