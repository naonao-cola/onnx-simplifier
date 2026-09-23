#!/usr/bin/env python3
"""Run a pipe_<stage>.txt on the host with CPU stand-ins for every non-CPU step, to check that the
stitching is exact before anything runs on the phone:
  htp steps      -> the same model on ORT CPU
  rpn            -> rpn_region.onnx (the 604 rest.onnx nodes the DSP kernel replaces) on ORT CPU;
                    for SRC 1 the raw uint8 deltas go through adapt_deltas.onnx first
  roialign       -> ORT's own RoiAlign on the NCHW view of the map, output stored as NHWC rows
  quant_in / dq  -> numpy, same arithmetic as QuantizeLinear / DequantizeLinear

  python host_emulate.py OUT_DIR STAGE image.bin[,image.bin...] RESULT_DIR
Each image's four outputs are written as RESULT_DIR/<stage>/<image stem>_<k>.npy.
"""

import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

_sess = {}


def sess(path):
    if path not in _sess:
        so = ort.SessionOptions()
        so.log_severity_level = 3
        _sess[path] = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    return _sess[path]


def run_model(path, store, ins, outs, override=None):
    s = sess(path)
    feeds = {}
    for k, i in enumerate(s.get_inputs()):  # bound by name; an ortpad head's input 0 is the padded tensor
        x = (override or {}).get(k)
        feeds[i.name] = x if x is not None else store[i.name]
    res = s.run(None, feeds)
    got = dict(zip([o.name for o in s.get_outputs()], res))
    for o in outs:
        store[o] = got[o]


_ra_models = {}


def roialign_model(oh, ow, sr, scale):
    key = (oh, ow, sr, scale)
    if key not in _ra_models:
        n = helper.make_node("RoiAlign", ["X", "rois", "bi"], ["Y"], output_height=oh, output_width=ow,
                             sampling_ratio=sr, spatial_scale=scale, mode="avg")
        g = helper.make_graph([n], "ra", [helper.make_tensor_value_info("X", TensorProto.FLOAT, None),
                                          helper.make_tensor_value_info("rois", TensorProto.FLOAT, None),
                                          helper.make_tensor_value_info("bi", TensorProto.INT64, None)],
                              [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)])
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 12)])
        m.ir_version = 7
        so = ort.SessionOptions()
        so.log_severity_level = 3
        _ra_models[key] = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
    return _ra_models[key]


def run_pipe(out, stage, image):
    store = {"image": image}
    for line in (out / f"pipe_{stage}.txt").read_text().splitlines():
        f = line.split()
        op = f[0]
        L = lambda s: [] if s == "-" else s.split(",")  # noqa: E731
        if op == "ort":
            run_model(out / f[2], store, L(f[5]), L(f[6]))
        elif op == "ortpad":
            padin, buckets, ins, outs = f[4], f[5], L(f[6]), L(f[7])
            x = store[padin]
            n = x.shape[0]
            b, model = min(((int(k), v) for k, v in (e.split(":") for e in buckets.split(","))
                            if int(k) >= n), key=lambda t: t[0])
            s = sess(out / model)
            shp = [d for d in s.get_inputs()[0].shape]
            xp = np.zeros([b] + list(x.shape[1:]), x.dtype)
            xp[:n] = x
            run_model(out / model, store, ins, outs, {0: xp.reshape(shp)})
            for o in outs:
                store[o] = store[o][:n]
        elif op == "quant_in":
            s, z = float(f[3]), int(f[4])
            x = store[f[1]]
            q = np.clip(np.rint(x / np.float32(s)) + z, 0, 255).astype(np.uint8)
            store[f[2]] = q.transpose(1, 2, 0)[None].copy()
        elif op == "dq":
            s, z = np.float32(float(f[3])), int(f[4])
            store[f[2]] = ((store[f[1]].astype(np.int32) - z).astype(np.float32) * s)
        elif op == "rpn":
            src, scores, deltas, dst = int(f[1]), L(f[2]), L(f[3]), f[4]
            if src == 1:
                s = sess(out / "adapt_deltas.onnx")
                res = s.run(None, {i.name: store[deltas[k]] for k, i in enumerate(s.get_inputs())})
                dmap = {o.name: r for o, r in zip(s.get_outputs(), res)}
            else:
                dmap = {n: store[n] for n in deltas}
            r = sess(out / "rpn_region.onnx")
            feeds = {}
            for i in r.get_inputs():
                feeds[i.name] = store[i.name] if i.name in store and i.name not in dmap else dmap[i.name]
            (store[dst],) = r.run(None, feeds)
        elif op == "roialign":
            x, rois, dst = store[f[1]], store[f[2]], f[3]
            oh, ow, sr, sc = int(f[4]), int(f[5]), int(f[6]), float(f[7])
            R = rois.shape[0]
            if R == 0:
                store[dst] = np.zeros((0, x.shape[3], oh, ow), np.float32)
                continue
            (y,) = roialign_model(oh, ow, sr, sc).run(None, {
                "X": np.ascontiguousarray(x.transpose(0, 3, 1, 2)), "rois": rois,
                "bi": np.zeros(R, np.int64)})
            store[dst] = np.ascontiguousarray(y.transpose(0, 2, 3, 1)).reshape(y.shape)
        else:
            raise ValueError(line)
    return store


def main():
    out, stage, images, res = Path(sys.argv[1]), sys.argv[2], sys.argv[3].split(","), Path(sys.argv[4])
    (res / stage).mkdir(parents=True, exist_ok=True)
    final = ["6568", "6570", "6572", "6887"]
    for im in images:
        x = np.fromfile(im, np.float32).reshape(3, 800, 1088)
        st = run_pipe(out, stage, x)
        for k, n in enumerate(final):
            np.save(res / stage / f"{Path(im).stem}_{k}.npy", st[n])
        print(stage, Path(im).stem, [st[n].shape for n in final], flush=True)


if __name__ == "__main__":
    main()
