"""Top-1 accuracy of every ``modelopt_pipeline.py`` variant, run through real TensorRT.

Run under the TensorRT interpreter. Builds each ``*.onnx`` matching ``--glob`` (fp32 and
fp16 for plain models, ``--int8 --fp16`` for ModelOpt ``*.int8*`` models), streams the
``imagenette_data.py`` val set through the engine and reports 1000-way top-1 against the
labels plus top-1 agreement with the ``--ref`` model's fp32 engine.

    python eval_accuracy.py DIR DATA_DIR [--glob 'b8.*.onnx'] [--ref b8.raw.onnx]
"""

import argparse
import ctypes
import json
import sys
from pathlib import Path

import numpy as np
import tensorrt as trt

import trt_harness as h



def predict(blob, x_u8, mean, std):
    """Argmax class ids for uint8 NHWC images, through a fixed-batch engine."""
    cuda = h.Cudart()
    engine = trt.Runtime(h.LOGGER).deserialize_cuda_engine(blob)
    ctx = engine.create_execution_context()
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    inp = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
    out = next(n for n in names if n != inp)
    batch = engine.get_tensor_shape(inp)[0]
    dev = {n: cuda.malloc(int(np.prod(engine.get_tensor_shape(n))) *
                          np.dtype(trt.nptype(engine.get_tensor_dtype(n))).itemsize)
           for n in (inp, out)}
    for n in dev:
        ctx.set_tensor_address(n, dev[n].value)
    ydtype = trt.nptype(engine.get_tensor_dtype(out))
    yhost = np.empty(engine.get_tensor_shape(out), ydtype)
    stream = ctypes.c_void_p()
    cuda.check(cuda.lib.cudaStreamCreate(ctypes.byref(stream)))
    preds = []
    for i in range(0, len(x_u8), batch):
        chunk = x_u8[i:i + batch]
        n = len(chunk)
        if n < batch:  # pad the tail; padded rows are discarded
            chunk = np.concatenate([chunk, np.zeros((batch - n, *chunk.shape[1:]), np.uint8)])
        x = ((chunk.astype(np.float32) / 255.0 - mean) / std).transpose(0, 3, 1, 2)
        cuda.memcpy_htod(dev[inp], np.ascontiguousarray(x, dtype=trt.nptype(engine.get_tensor_dtype(inp))))
        ctx.execute_async_v3(stream.value)
        cuda.sync()
        cuda.memcpy_dtoh(yhost, dev[out])
        preds.append(yhost[:n].astype(np.float32).argmax(1))
    for p in dev.values():
        cuda.free(p)
    return np.concatenate(preds)


def configs(path):
    if ".int8" in path.name:
        return [("int8fp16", dict(fp16=True, int8=True))]
    return [("fp32", {}), ("fp16", dict(fp16=True))]


def get_engine(path, prec, kw):
    """Engine cache shared with bench_trtexec.py: ``DIR/engines/<stem>.<prec>.engine``."""
    cache = path.parent / "engines" / f"{path.stem}.{prec}.engine"
    if cache.exists():
        return cache.read_bytes(), {}
    blob, info = h.build_engine(path, **kw)
    if blob is not None:
        cache.parent.mkdir(exist_ok=True)
        cache.write_bytes(blob)
    return blob, info


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("data")
    ap.add_argument("--glob", default="b8.*.onnx")
    ap.add_argument("--ref", default="b8.raw.onnx", help="model whose fp32 engine is the agreement reference")
    ap.add_argument("--mean", type=float, nargs=3, default=[0.485, 0.456, 0.406])
    ap.add_argument("--std", type=float, nargs=3, default=[0.229, 0.224, 0.225])
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    x = np.load(Path(args.data) / "val_x.npy")
    y = np.load(Path(args.data) / "val_y.npy")

    mean = np.array(args.mean, np.float32)
    std = np.array(args.std, np.float32)
    blob, info = get_engine(Path(args.dir) / args.ref, "fp32", {})
    assert blob, info
    ref = predict(blob, x, mean, std)
    print(f"reference ({args.ref} fp32): top-1 {np.mean(ref == y) * 100:.2f}%", flush=True)

    rows = []
    for path in sorted(Path(args.dir).glob(args.glob)):
        for prec, kw in configs(path):
            blob, info = get_engine(path, prec, kw)
            if blob is None:
                rows.append({"model": path.stem, "precision": prec, "error": info.get("error")})
                print(rows[-1], flush=True)
                continue
            pred = predict(blob, x, mean, std)
            rows.append({"model": path.stem, "precision": prec,
                         "top1": round(float(np.mean(pred == y)) * 100, 2),
                         "agree_fp32": round(float(np.mean(pred == ref)) * 100, 2)})
            print(rows[-1], flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
    print(f"\n{'model':<44}{'precision':<11}{'top-1 %':>9}{'agree w/ fp32 %':>17}")
    for r in rows:
        if "error" in r:
            print(f"{r['model']:<44}{r['precision']:<11}  ERROR {r['error']}")
        else:
            print(f"{r['model']:<44}{r['precision']:<11}{r['top1']:>9.2f}{r['agree_fp32']:>17.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
