"""Real nuScenes-mini frames for BEVFormer-tiny, without nuscenes-devkit or mmdet3d.

nuScenes-mini (v1.0-mini.tgz) downloads without an account; fetch_data.sh keeps only the metadata
JSON and the 6 camera keyframe directories. This reproduces what BEVFormer's data pipeline feeds
the model for one keyframe:
  * lidar2img per camera: tools/data_converter/nuscenes_converter.py obtain_sensor2top() +
    NuScenesDataset.get_data_info(), then RandomScaleImageMultiViewImage(0.5)
  * image: NormalizeMultiviewImage (RGB, mean/std) -> 0.5x bilinear resize -> pad to /32 (480x800)
  * can_bus: [ego translation, ego rotation quat, 9 CAN-bus signals, yaw rad, yaw deg]; with the
    temporal test-time deltas of BEVFormer.forward_test (translation/yaw deg relative to the
    previous frame, zero on a scene's first frame). The 9 CAN-bus signals (accel, rotation rate,
    velocity) come from the separate CAN-bus expansion, not in v1.0-mini: left at 0 here.
  * GT: annotation centers in the lidar frame + the 10 detection class names, for a sanity match.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import IMG_MEAN, IMG_STD

CAMS = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
CLASSES = ["car", "truck", "construction_vehicle", "bus", "trailer", "barrier", "motorcycle", "bicycle",
           "pedestrian", "traffic_cone"]
CATEGORY = {
    "vehicle.car": "car", "vehicle.truck": "truck", "vehicle.construction": "construction_vehicle",
    "vehicle.bus.bendy": "bus", "vehicle.bus.rigid": "bus", "vehicle.trailer": "trailer",
    "movable_object.barrier": "barrier", "vehicle.motorcycle": "motorcycle", "vehicle.bicycle": "bicycle",
    "human.pedestrian.adult": "pedestrian", "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian", "human.pedestrian.police_officer": "pedestrian",
    "movable_object.trafficcone": "traffic_cone",
}


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quat_yaw(q):
    v = quat_to_mat(q) @ np.array([1.0, 0.0, 0.0])
    return math.atan2(v[1], v[0])


class NuScenesMini:
    def __init__(self, root):
        self.root = Path(root)
        t = {}
        for name in ("scene", "sample", "sample_data", "calibrated_sensor", "ego_pose", "sensor",
                     "sample_annotation", "instance", "category"):
            t[name] = json.loads((self.root / "v1.0-mini" / f"{name}.json").read_text())
        self.tok = {k: {r["token"]: r for r in v} for k, v in t.items()}
        self.scenes = {s["name"]: s for s in t["scene"]}
        # sample -> {channel: keyframe sample_data}
        self.sd = {}
        for r in t["sample_data"]:
            if not r["is_key_frame"]:
                continue
            ch = self.tok["sensor"][self.tok["calibrated_sensor"][r["calibrated_sensor_token"]]["sensor_token"]]["channel"]
            self.sd.setdefault(r["sample_token"], {})[ch] = r
        self.anns = {}
        for a in t["sample_annotation"]:
            self.anns.setdefault(a["sample_token"], []).append(a)

    def scene_samples(self, scene):
        s = self.scenes[scene]
        out, tok = [], s["first_sample_token"]
        while tok:
            out.append(tok)
            tok = self.tok["sample"][tok]["next"]
        return out

    def _pose(self, sd):
        cs = self.tok["calibrated_sensor"][sd["calibrated_sensor_token"]]
        ep = self.tok["ego_pose"][sd["ego_pose_token"]]
        return (quat_to_mat(cs["rotation"]), np.array(cs["translation"]), quat_to_mat(ep["rotation"]),
                np.array(ep["translation"]), cs, ep)

    def frame(self, sample_token, scale=0.5, pad_hw=(480, 800)):
        sds = self.sd[sample_token]
        l2e_r, l2e_t, e2g_r, e2g_t, _, ep = self._pose(sds["LIDAR_TOP"])
        inv = np.linalg.inv(e2g_r).T @ np.linalg.inv(l2e_r).T
        imgs, l2i = [], []
        mean, std = torch.tensor(IMG_MEAN).view(3, 1, 1), torch.tensor(IMG_STD).view(3, 1, 1)
        from PIL import Image

        for cam in CAMS:
            sd = sds[cam]
            l2e_r_s, l2e_t_s, e2g_r_s, e2g_t_s, cs, _ = self._pose(sd)
            R = (l2e_r_s.T @ e2g_r_s.T) @ inv
            T = (l2e_t_s @ e2g_r_s.T + e2g_t_s) @ inv
            T -= e2g_t @ inv + l2e_t @ np.linalg.inv(l2e_r).T
            s2l_r, s2l_t = R.T, T
            lidar2cam_r = np.linalg.inv(s2l_r)
            lidar2cam_t = s2l_t @ lidar2cam_r.T
            rt = np.eye(4)
            rt[:3, :3] = lidar2cam_r.T
            rt[3, :3] = -lidar2cam_t
            viewpad = np.eye(4)
            viewpad[:3, :3] = np.array(cs["camera_intrinsic"])
            m = viewpad @ rt.T
            m = np.diag([scale, scale, 1, 1]) @ m
            l2i.append(m)
            img = torch.from_numpy(np.asarray(Image.open(self.root / sd["filename"]).convert("RGB"), dtype=np.float32))
            img = (img.permute(2, 0, 1) - mean) / std
            h, w = img.shape[1:]
            img = F.interpolate(img[None], size=(int(h * scale + 0.5), int(w * scale + 0.5)), mode="bilinear",
                                align_corners=False)[0]
            img = F.pad(img, (0, pad_hw[1] - img.shape[2], 0, pad_hw[0] - img.shape[1]))
            imgs.append(img)
        rot = ep["rotation"]
        yaw = quat_yaw(rot) / math.pi * 180
        if yaw < 0:
            yaw += 360
        can_bus = np.zeros(18)
        can_bus[:3] = ep["translation"]
        can_bus[3:7] = rot
        can_bus[-2] = yaw / 180 * math.pi
        can_bus[-1] = yaw
        gt = []
        for a in self.anns.get(sample_token, []):
            cat = self.tok["category"][self.tok["instance"][a["instance_token"]]["category_token"]]["name"]
            if cat not in CATEGORY or a["num_lidar_pts"] + a["num_radar_pts"] == 0:
                continue
            p = np.linalg.inv(e2g_r) @ (np.array(a["translation"]) - e2g_t)
            p = np.linalg.inv(l2e_r) @ (p - l2e_t)
            gt.append((CATEGORY[cat], p))
        return {"img": torch.stack(imgs), "lidar2img": torch.tensor(np.stack(l2i), dtype=torch.float32),
                "can_bus_abs": can_bus, "gt": gt, "scene_token": self.tok["sample"][sample_token]["scene_token"]}


def temporal_can_bus(cur_abs, prev_abs):
    """BEVFormer.forward_test: can_bus[:3] and can_bus[-1] relative to the previous frame, or 0."""
    cb = cur_abs.copy()
    if prev_abs is None:
        cb[:3] = 0
        cb[-1] = 0
    else:
        cb[:3] -= prev_abs[:3]
        cb[-1] -= prev_abs[-1]
    return torch.tensor(cb, dtype=torch.float32)


def match(boxes, scores, labels, gt, thr=0.3, dist=2.0):
    """Greedy BEV center-distance match (nuScenes' 2 m threshold): -> (tp, n_pred, n_gt)."""
    keep = scores >= thr
    b, lab = boxes[keep], labels[keep]
    used = set()
    tp = 0
    for i in torch.argsort(scores[keep], descending=True).tolist():
        best, bj = dist, None
        for j, (name, p) in enumerate(gt):
            if j in used or name != CLASSES[int(lab[i])]:
                continue
            d = math.hypot(float(b[i, 0]) - p[0], float(b[i, 1]) - p[1])
            if d < best:
                best, bj = d, j
        if bj is not None:
            used.add(bj)
            tp += 1
    return tp, int(keep.sum()), len(gt)
