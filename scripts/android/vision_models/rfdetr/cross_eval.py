#!/usr/bin/env python3
"""Accuracy of operating points on common yardsticks, from phone_eval.py's pulled outputs.

  cross_eval.py <tag> <reference variant>   phone detections vs another variant's fp32
                                            (e.g. nano@320.u8 vs nano)
  cross_eval.py <tag> gt                    phone detections vs COCO val2017 ground truth:
                                            recall and precision at score >= 0.3, IoU >= 0.5,
                                            same category (not mAP)
  cross_eval.py fp32:<variant> gt           the same for a variant's fp32 reference

<tag> is a phone_eval stage (the model file's stem, e.g. nano.u8). RF-DETR's labels are COCO
category ids. Boxes are normalized to the (non-aspect-preserving) square input, so they scale back
to the original image by its width and height. Ground truth: ~/.cache/coco/instances_val2017.json
(from http://images.cocodataset.org/annotations/annotations_trainval2017.zip), crowd boxes skipped.
"""

import json
import sys
from pathlib import Path

import common as C
import export
import numpy as np
from PIL import Image


def phone_outputs(tag, i):
    pulled = C.WORK / f"phone_{tag}" / "pulled"
    outs = {}
    for line in (pulled / f"log{i}.txt").read_text().splitlines():
        if line.startswith("out "):
            _, k, name, dt, shp = line.split()
            a = np.fromfile(pulled / f"out{i}_o{k}.bin", np.float32)
            outs[name] = a.reshape([int(s) for s in shp.strip(",").split(",")])
    return outs["logits"], outs["boxes"]


def gt_eval(get, paths, ids):
    ann = json.load(open(Path.home() / ".cache/coco/instances_val2017.json"))[
        "annotations"
    ]
    tot = {"gt": 0, "det": 0, "tp": 0}
    for i, (p, img_id) in enumerate(zip(paths, ids)):
        w, h = Image.open(p).size
        gb = [
            (a["bbox"], a["category_id"])
            for a in ann
            if a["image_id"] == img_id and not a["iscrowd"]
        ]
        g_xyxy = np.array([[x, y, x + bw, y + bh] for (x, y, bw, bh), _ in gb]).reshape(
            -1, 4
        )
        g_cls = [c for _, c in gb]
        b, s, c = C.decode(*get(i), 1)
        keep = s >= 0.3
        b = b[keep] * np.array([w, h, w, h])
        c = c[keep]
        free = list(range(len(gb)))
        for bi in np.argsort(-s[keep]):
            best, bj = 0.5, None
            for j in free:
                if g_cls[j] == c[bi]:
                    x1, y1 = np.maximum(b[bi, :2], g_xyxy[j, :2])
                    x2, y2 = np.minimum(b[bi, 2:], g_xyxy[j, 2:])
                    inter = max(0, x2 - x1) * max(0, y2 - y1)
                    a1 = (b[bi, 2] - b[bi, 0]) * (b[bi, 3] - b[bi, 1])
                    a2 = (g_xyxy[j, 2] - g_xyxy[j, 0]) * (g_xyxy[j, 3] - g_xyxy[j, 1])
                    iou = inter / max(a1 + a2 - inter, 1e-9)
                    if iou > best:
                        best, bj = iou, j
            if bj is not None:
                free.remove(bj)
                tot["tp"] += 1
        tot["gt"] += len(gb)
        tot["det"] += int(keep.sum())
    return tot


def main():
    tag, yard = sys.argv[1], sys.argv[2]
    paths = C.image_paths("eval")
    if tag.startswith("fp32:"):
        ref = export.refs(tag[5:], paths)

        def get(i):
            return ref[i]["logits"], ref[i]["boxes"]
    else:

        def get(i):
            return phone_outputs(tag, i)

    if yard == "gt":
        t = gt_eval(get, paths, C.coco_ids("eval"))
        print(
            f"{tag} vs COCO GT: recall {t['tp']}/{t['gt']} ({100 * t['tp'] / t['gt']:.1f}%), "
            f"precision {t['tp']}/{t['det']} ({100 * t['tp'] / max(t['det'], 1):.1f}%)"
        )
        return
    ref = export.refs(yard, paths)
    size = int(ref[0]["res"])
    tot = {"ref": 0, "det": 0, "matched": 0}
    for i, r in enumerate(ref):
        m = C.match((r["logits"], r["boxes"]), get(i), size)
        for k in tot:
            tot[k] += m[k]
    print(
        f"{tag} vs {yard} fp32: matched {tot['matched']}/{tot['ref']} (det {tot['det']})"
    )


if __name__ == "__main__":
    main()
