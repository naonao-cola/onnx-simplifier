"""nuScenes-mini keyframes as Sparse4D v3's test pipeline feeds them, without nuscenes-devkit / mmdet3d.

Reuses ../bevformer_tiny's data (fetch_data.sh: the metadata JSON + the 6 camera keyframe dirs,
~/.cache/onnxsim-bevformer/nuscenes-mini) and restates its pose math here (bevformer_tiny's
nuscenes.py imports that model's module, so it isn't imported). Per frame:
  * lidar2img: tools/nuscenes_converter.py obtain_sensor2top() + NuScenes3DDetTrackDataset.get_data_info(),
    then ResizeCropFlipImage's test-mode aug (resize max(256/900, 704/1600) = 0.44 -> 704x396,
    crop (0, 140, 704, 396)): rows 0/1 of lidar2img scaled by 0.44, minus crop x/y times row 2
  * image: PIL resize (BICUBIC, Pillow's default for RGB) + crop, then NormalizeMultiviewImage
    (RGB mean/std) -> (6, 3, 256, 704) float, or the uint8 RGB for the phone (normalization folded)
  * timestamp (sample timestamp / 1e6), T_global = lidar2global, T_global_inv
  * GT: annotation centers in the lidar frame + detection class names (same as bevformer_tiny)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

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
MEAN = np.array([123.675, 116.28, 103.53], np.float32)
STD = np.array([58.395, 57.12, 57.375], np.float32)
RESIZE, CROP = 0.44, (0, 140, 704, 396)


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class NuScenesMini:
    def __init__(self, root):
        self.root = Path(root)
        t = {}
        for name in ("scene", "sample", "sample_data", "calibrated_sensor", "ego_pose", "sensor",
                     "sample_annotation", "instance", "category"):
            t[name] = json.loads((self.root / "v1.0-mini" / f"{name}.json").read_text())
        self.tok = {k: {r["token"]: r for r in v} for k, v in t.items()}
        self.scenes = {s["name"]: s for s in t["scene"]}
        self.sd = {}
        for r in t["sample_data"]:
            if r["is_key_frame"]:
                ch = self.tok["sensor"][self.tok["calibrated_sensor"][r["calibrated_sensor_token"]]["sensor_token"]]["channel"]
                self.sd.setdefault(r["sample_token"], {})[ch] = r
        self.anns = {}
        for a in t["sample_annotation"]:
            self.anns.setdefault(a["sample_token"], []).append(a)

    def scene_samples(self, scene):
        out, tok = [], self.scenes[scene]["first_sample_token"]
        while tok:
            out.append(tok)
            tok = self.tok["sample"][tok]["next"]
        return out

    def _pose(self, sd):
        cs = self.tok["calibrated_sensor"][sd["calibrated_sensor_token"]]
        ep = self.tok["ego_pose"][sd["ego_pose_token"]]
        return (quat_to_mat(cs["rotation"]), np.array(cs["translation"]), quat_to_mat(ep["rotation"]),
                np.array(ep["translation"]), cs)

    def frame(self, sample_token):
        from PIL import Image

        sds = self.sd[sample_token]
        l2e_r, l2e_t, e2g_r, e2g_t, _ = self._pose(sds["LIDAR_TOP"])
        inv = np.linalg.inv(e2g_r).T @ np.linalg.inv(l2e_r).T
        aug = np.eye(4)
        aug[:2, :2] *= RESIZE
        aug[:2, 2] -= np.array(CROP[:2])
        imgs, l2i = [], []
        for cam in CAMS:
            sd = sds[cam]
            l2e_r_s, l2e_t_s, e2g_r_s, e2g_t_s, cs = self._pose(sd)
            R = (l2e_r_s.T @ e2g_r_s.T) @ inv
            T = (l2e_t_s @ e2g_r_s.T + e2g_t_s) @ inv
            T -= e2g_t @ inv + l2e_t @ np.linalg.inv(l2e_r).T
            lidar2cam_r = np.linalg.inv(R.T)
            lidar2cam_t = T @ lidar2cam_r.T
            rt = np.eye(4)
            rt[:3, :3] = lidar2cam_r.T
            rt[3, :3] = -lidar2cam_t
            viewpad = np.eye(4)
            viewpad[:3, :3] = np.array(cs["camera_intrinsic"])
            l2i.append(aug @ viewpad @ rt.T)
            im = Image.open(self.root / sd["filename"]).convert("RGB")
            w, h = im.size
            im = im.resize((int(w * RESIZE), int(h * RESIZE)), Image.BICUBIC).crop(CROP)
            imgs.append(np.asarray(im, dtype=np.uint8))
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3], lidar2ego[:3, 3] = l2e_r, l2e_t
        ego2global = np.eye(4)
        ego2global[:3, :3], ego2global[:3, 3] = e2g_r, e2g_t
        T_global = ego2global @ lidar2ego
        gt = []
        for a in self.anns.get(sample_token, []):
            cat = self.tok["category"][self.tok["instance"][a["instance_token"]]["category_token"]]["name"]
            if cat not in CATEGORY or a["num_lidar_pts"] + a["num_radar_pts"] == 0:
                continue
            p = np.linalg.inv(e2g_r) @ (np.array(a["translation"]) - e2g_t)
            p = np.linalg.inv(l2e_r) @ (p - l2e_t)
            gt.append((CATEGORY[cat], p))
        rgb = np.stack(imgs)  # (6, 256, 704, 3) uint8
        img = torch.from_numpy((rgb.astype(np.float32) - MEAN) / STD).permute(0, 3, 1, 2).contiguous()
        metas = {
            "projection_mat": torch.tensor(np.stack(l2i), dtype=torch.float32),
            "image_wh": torch.tensor([[704.0, 256.0]] * 6),
            "timestamp": self.tok["sample"][sample_token]["timestamp"] / 1e6,
            "T_global": T_global,
            "T_global_inv": np.linalg.inv(T_global),
        }
        return {"img": img, "rgb": rgb, "metas": metas, "gt": gt}


def match(boxes, scores, labels, gt, thr=0.3, dist=2.0):
    """Greedy BEV center-distance match (nuScenes' 2 m threshold), same as bevformer_tiny: -> (tp, n_pred, n_gt)."""
    keep = scores >= thr
    b, lab = boxes[keep], labels[keep]
    used, tp = set(), 0
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
