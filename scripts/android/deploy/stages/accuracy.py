"""Accuracy of the phone's outputs against fp32 ORT on the host (the `accuracy` stage).

Three rows per metric, so a regression can be attributed:
  host int8   the deployed graph (rewrite stage) + post on host ORT CPU   -> quantization error
  phone       what pipe_run wrote on the phone (bench stage outputs)     -> + HTP numerics
both against the fp32 reference: the simplified (pre-quantization) model + the same post graph.

spec `accuracy:`:
  kind: detection_match   match detections by class and box IoU > iou (default 0.5), both sides
                          filtered at score >= score (default 0.25); reports matched/ref,
                          detections, mean IoU of matches, mean |score delta| -- the metric
                          ../../maskrcnn_e2e/README.md and ../../e2e_pipeline/ use
  kind: tensor            per-output max/mean abs error and cosine similarity
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import post as postlib


def _load_phone(out_dir: Path, stem: str, k: int) -> np.ndarray:
    shp = (out_dir / f"{stem}_{k}.shape").read_text().split()
    ty, dims = int(shp[0]), [int(x) for x in shp[1:]]
    dt = {1: np.float32, 2: np.uint8, 7: np.int64, 6: np.int32}[ty]
    return np.fromfile(out_dir / f"{stem}_{k}.bin", dt).reshape(dims)


def _iou(a, b):
    x1, y1 = np.maximum(a[0], b[0]), np.maximum(a[1], b[1])
    x2, y2 = np.minimum(a[2], b[2]), np.minimum(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def det_match(ref, got, iou_t=0.5, score_t=0.25):
    rb, rs, rc = ref
    gb, gs, gc = got
    ri = np.where(rs >= score_t)[0]
    gi = list(np.where(gs >= score_t)[0])
    matched, ious, ds = 0, [], []
    for i in ri[np.argsort(-rs[ri])]:
        best, bj = iou_t, None
        for j in gi:
            if gc[j] == rc[i]:
                v = _iou(rb[i], gb[j])
                if v > best:
                    best, bj = v, j
        if bj is not None:
            gi.remove(bj)
            matched += 1
            ious.append(best)
            ds.append(abs(float(rs[i]) - float(gs[bj])))
    return {"ref": int(len(ri)), "det": int((gs >= score_t).sum()), "matched": matched,
            "iou_sum": float(sum(ious)), "dscore_sum": float(sum(ds))}


def run(ctx, d: Path) -> None:
    import onnxruntime as ort

    acc = ctx.spec.get("accuracy", {}) or {}
    kind = acc.get("kind", "tensor")
    meta = json.loads((ctx.work / "pipe" / "pipe_meta.json").read_text())
    pdir = ctx.work / "pipe"
    phone = ctx.work / "bench" / "outputs"
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4

    def sess(p):
        return ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])

    # fp32 reference: simplified model + the same post-processing, built for its fp32 head
    fp32 = sess(ctx.work / "simplify" / "model.onnx")
    ref_post = None
    if ctx.spec.get("postprocess"):
        (d / "ref").mkdir(exist_ok=True)
        postlib.build(ctx.spec["postprocess"], ctx.work / "simplify" / "model.onnx", d / "ref")
        ref_post = sess(d / "ref" / "post.onnx")
    int8 = sess(pdir / meta["model"])
    int8_post = sess(pdir / meta["post"]) if meta["post"] else None
    (in_name,) = list(ctx.spec["inputs"])
    u = meta["uint8_input"]

    def run_chain(net, pst, x_net):
        ys = net.run(None, {net.get_inputs()[0].name: x_net})
        if pst is None:
            return ys
        feed = {pst.get_inputs()[0].name: ys[[o.name for o in net.get_outputs()].index(pst.get_inputs()[0].name)]}
        return pst.run(None, feed)

    rows = {"host_int8": [], "phone": []}
    for stem in meta["inputs"]:
        raw = np.fromfile(pdir / "inputs" / f"{stem}.bin", np.uint8 if meta["host_quantize"] else np.float32)
        shape = ctx.spec["inputs"][in_name]["shape"]
        if meta["host_quantize"]:
            q = raw.reshape(u["shape"])
            x = ((q.astype(np.float32) - u["zero_point"]) * u["scale"])
            x = (x.transpose(0, 3, 1, 2) if u["layout"] == "nhwc" else x).reshape(shape)
        else:
            x = raw.reshape(shape)
        ref = run_chain(fp32, ref_post, x)
        if u:
            q = np.clip(np.rint(x[0] / u["scale"]) + u["zero_point"], 0, 255).astype(np.uint8)
            x8 = (q.transpose(1, 2, 0) if u["layout"] == "nhwc" else q)[None]
        else:
            x8 = x
        host = run_chain(int8, int8_post, x8)
        got = [_load_phone(phone, stem, k) for k in range(len(meta["outputs"]))]
        for label, out in (("host_int8", host), ("phone", got)):
            if kind == "detection_match":
                r = det_match(ref[:3], out[:3], acc.get("iou", 0.5), acc.get("score", 0.25))
            else:
                r = {}
                for k, (a, b) in enumerate(zip(ref, out)):
                    a = a.astype(np.float64).ravel()
                    b = b.astype(np.float64).ravel()
                    r[f"out{k}"] = {"max_abs": float(np.abs(a - b).max()), "mean_abs": float(np.abs(a - b).mean()),
                                    "cos": float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))}
            r["input"] = stem
            rows[label].append(r)
    summary = {}
    for label, rs in rows.items():
        if kind == "detection_match":
            tot = {k: sum(r[k] for r in rs) for k in ("ref", "det", "matched", "iou_sum", "dscore_sum")}
            m = max(tot["matched"], 1)
            summary[label] = {"matched": tot["matched"], "ref": tot["ref"], "det": tot["det"],
                              "match_rate": tot["matched"] / max(tot["ref"], 1),
                              "mean_iou": tot["iou_sum"] / m, "mean_dscore": tot["dscore_sum"] / m}
        else:
            summary[label] = {"min_cos": min(v["cos"] for r in rs for k, v in r.items() if k != "input")}
    (d / "accuracy.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    for label, s in summary.items():
        print(f"  {label:9s} vs fp32: " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                                   for k, v in s.items()))
