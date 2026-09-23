"""Host decode + NMS for Fast-BEV M0 (anchor head) and Fast-BEV++ (CenterHead).

Follows upstream's test-time post-processing exactly (numpy here, the NMS loops in csrc/postproc.c
through ctypes; the phone runner links the same C file):

M0: Anchor3DHead.get_bboxes_single + box3d_multiclass_scale_nms (Fast-BEV test_cfg)
  8 anchors per cell (AlignedAnchor3DRangeGenerator: 4 sizes x 2 rotations, centers at
  -49.5 + i on the 100 x 100 map, z = -1.8), sigmoid scores, top 1000 by max class score,
  DeltaXYZWLHRBBoxCoder decode, per class: score > 0.05, BEV (w, l) scaled by nms_rescale_factor,
  rotated NMS (thr 0.2) or circle NMS (barrier: squared distance <= 1, at most 83), at most 500
  overall, then the direction-classifier yaw fix (dir_offset pi/4).
PP: CenterHead.get_bboxes (BEVDet): sigmoid heatmap, top 500 over (class, y, x), center +
  reg offset, x/y * 8 * 0.1 - 51.2, exp(dim), atan2(sin, cos) yaw, score > 0.1 and centers inside
  +-61.2 m, dims scaled per class by nms_rescale_factor for a rotated NMS (thr 0.2, 1000 in, 500
  out).
Both return (boxes (N, 9) [x, y, z, w, l, h, yaw, vx, vy], scores (N,), class names list).
"""
from __future__ import annotations

import ctypes
import math
import subprocess
from pathlib import Path

import numpy as np

M0_CLASSES = ["car", "truck", "trailer", "bus", "construction_vehicle", "bicycle", "motorcycle",
              "pedestrian", "traffic_cone", "barrier"]
PP_CLASSES = ["car", "truck", "construction_vehicle", "bus", "trailer", "barrier", "motorcycle", "bicycle",
              "pedestrian", "traffic_cone"]
M0_SIZES = [[0.8660, 2.5981, 1.0], [0.5774, 1.7321, 1.0], [1.0, 1.0, 1.0], [0.4, 0.4, 1.0]]
M0_NMS_TYPE = ["rotate"] * 9 + ["circle"]
M0_NMS_THR = [0.2] * 7 + [0.5, 0.5, 0.2]
M0_NMS_RADIUS = [4, 12, 10, 10, 12, 0.85, 0.85, 0.175, 0.175, 1]
M0_NMS_RESCALE = [1.0, 0.7, 0.55, 0.4, 0.7, 1.0, 1.0, 4.5, 9.0, 1.0]
PP_NMS_RESCALE = [1.0, 0.7, 0.7, 0.4, 0.55, 1.1, 1.0, 1.0, 1.5, 3.5]

_lib = None


def lib():
    global _lib
    if _lib is None:
        src = Path(__file__).with_name("csrc") / "postproc.c"
        so = Path.home() / ".cache" / "fastbev" / "libpostproc.so"
        if not so.exists() or so.stat().st_mtime < src.stat().st_mtime:
            so.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["cc", "-O2", "-shared", "-fPIC", str(src), "-o", str(so), "-lm"], check=True)
        _lib = ctypes.CDLL(str(so))
        for f in (_lib.nms_rotated, _lib.nms_circle):
            f.restype = ctypes.c_int
            f.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_void_p]
    return _lib


def _nms(kind, arr, thr, max_keep):
    arr = np.ascontiguousarray(arr, np.float32)
    keep = np.zeros(max(len(arr), 1), np.int32)
    f = lib().nms_rotated if kind == "rotate" else lib().nms_circle
    n = f(arr.ctypes.data, len(arr), float(thr), int(max_keep), keep.ctypes.data)
    return keep[:n]


def _desc(scores):
    """torch.sort(descending=True) order (stable here; ties are measure-zero on real scores)."""
    return np.argsort(-scores, kind="stable")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def m0_anchors(h=100, w=100):
    ys = np.linspace(-50, 50, h + 1, dtype=np.float32)[:h] + np.float32(0.5)
    xs = np.linspace(-50, 50, w + 1, dtype=np.float32)[:w] + np.float32(0.5)
    a = np.zeros((h, w, 4, 2, 9), np.float32)
    a[..., 0] = xs[None, :, None, None]
    a[..., 1] = ys[:, None, None, None]
    a[..., 2] = -1.8
    a[..., 3:6] = np.array(M0_SIZES, np.float32)[None, None, :, None, :]
    a[..., 6] = np.array([0, 1.57], np.float32)[None, None, None, :]
    return a.reshape(-1, 9)


def m0_decode(cls, reg, dirc, score_thr=0.05, nms_pre=1000, max_num=500):
    """cls (100, 100, 80), reg (100, 100, 72), dir (100, 100, 16) NHWC head maps of one frame."""
    scores = sigmoid(cls.reshape(-1, 10).astype(np.float32))
    reg = reg.reshape(-1, 9).astype(np.float32)
    dir_score = np.argmax(dirc.reshape(-1, 2), axis=1)
    anchors = m0_anchors()
    top = _desc(scores.max(1))[:nms_pre]
    a, d, scores, dir_score = anchors[top], reg[top], scores[top], dir_score[top]
    za = a[:, 2] + a[:, 5] / 2
    diag = np.sqrt(a[:, 4] ** 2 + a[:, 3] ** 2)
    boxes = np.stack([d[:, 0] * diag + a[:, 0], d[:, 1] * diag + a[:, 1], d[:, 2] * a[:, 5] + za,
                      np.exp(d[:, 3]) * a[:, 3], np.exp(d[:, 4]) * a[:, 4], np.exp(d[:, 5]) * a[:, 5],
                      d[:, 6] + a[:, 6], d[:, 7] + a[:, 7], d[:, 8] + a[:, 8]], 1)
    boxes[:, 2] -= boxes[:, 5] / 2
    ob, os_, ol, od = [], [], [], []
    for i in range(10):
        m = np.nonzero(scores[:, i] > score_thr)[0]
        if not len(m):
            continue
        s, b = scores[m, i], boxes[m]
        order = _desc(s)
        s, b, dd = s[order], b[order], dir_score[m][order]
        if M0_NMS_TYPE[i] == "rotate":
            bev = b[:, [0, 1, 3, 4, 6]].copy()
            bev[:, 2:4] *= M0_NMS_RESCALE[i]
            keep = _nms("rotate", bev, M0_NMS_THR[i], len(bev))
        else:
            keep = _nms("circle", b[:, :2], M0_NMS_RADIUS[i], 83)
        ob.append(b[keep])
        os_.append(s[keep])
        ol += [i] * len(keep)
        od.append(dd[keep])
    if not ob:
        return np.zeros((0, 9), np.float32), np.zeros(0, np.float32), []
    b, s, lab, dd = np.concatenate(ob), np.concatenate(os_), np.array(ol), np.concatenate(od)
    if len(b) > max_num:
        o = _desc(s)[:max_num]
        b, s, lab, dd = b[o], s[o], lab[o], dd[o]
    r = b[:, 6] - math.pi / 4
    r = r - np.floor(r / math.pi) * math.pi  # limit_period(offset 0, period pi)
    b[:, 6] = r + math.pi / 4 + math.pi * dd
    return b, s, [M0_CLASSES[i] for i in lab]


def pp_decode(head, K=500, score_thr=0.1):
    """head (128, 128, 20) NHWC: heatmap 10 | reg 2 | height 1 | dim 3 | rot 2 | vel 2."""
    head = head.astype(np.float32)
    H, W, _ = head.shape
    heat = sigmoid(head[..., :10]).transpose(2, 0, 1).reshape(-1)  # (cls, y, x)
    top = _desc(heat)[:K]
    s = heat[top]
    cls = top // (H * W)
    pix = top % (H * W)
    ys, xs = pix // W, pix % W
    g = head.reshape(-1, 20)[pix]
    x = (xs + g[:, 10]) * 8 * 0.1 - 51.2
    y = (ys + g[:, 11]) * 8 * 0.1 - 51.2
    boxes = np.stack([x, y, g[:, 12], *np.exp(g[:, 13:16]).T, np.arctan2(g[:, 16], g[:, 17]), g[:, 18], g[:, 19]], 1)
    m = (s > score_thr) & (np.abs(boxes[:, 0]) <= 61.2) & (np.abs(boxes[:, 1]) <= 61.2) & (np.abs(boxes[:, 2]) <= 10)
    boxes, s, cls = boxes[m], s[m], cls[m]
    bev = boxes[:, [0, 1, 3, 4, 6]].copy()
    bev[:, 2:4] *= np.array(PP_NMS_RESCALE, np.float32)[cls][:, None]
    order = _desc(s)[:1000]
    keep = order[_nms("rotate", bev[order], 0.2, 500)]
    b = boxes[keep]
    b[:, 2] -= b[:, 5] * 0.5
    return b, s[keep], [PP_CLASSES[i] for i in cls[keep]]
