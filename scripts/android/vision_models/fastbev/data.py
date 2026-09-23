"""nuScenes-mini frames for Fast-BEV (M0) and Fast-BEV++, without nuscenes-devkit or mmdet3d.

Uses the same camera-only slice of nuScenes-mini as ../bevformer_tiny (fetch_data.sh there: the
v1.0-mini metadata JSON plus the six cameras' keyframe images). The JSON/pose helpers are copied
from ../bevformer_tiny/nuscenes.py (that file imports BEVFormer's model module, so it is not
imported here).

Fast-BEV M0 (Sense-GVT/Fast-BEV, configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.py)
  * cameras in mmdet3d's converter order F, FR, FL, B, BL, BR; lidar2img from
    obtain_sensor2top() + NuScenesDataset.get_data_info() + RandomAugImageMultiViewImage (test:
    resize 704/1600 = 0.44 with PIL's default BICUBIC, crop rows 70..326 of the 396-row image),
    NormalizeMultiviewImage (RGB mean/std)
  * 4 time steps: the current keyframe and prev "sweeps" 1, 3, 5 of the interval-3 info list
    (tools/data_converter/nuscenes_seq_converter.py), i.e. camera frames 6, 12, 18 back along the
    CAM_FRONT chain, other cameras by nearest timestamp. On the eval scene-0103 these are
    exactly the previous 1, 2, 3 keyframes (checked: every pick must be a keyframe file, the only
    camera images in the slice), with the upstream clamp to the oldest one available early in a
    scene. In some other scenes a pick is a non-keyframe sweep; `keyframe_fallback` (calibration
    only) then takes the nearest keyframe. The previous frames' projections
    are the current cameras' sensor2lidar moved by lidar_adj->lidar_cur (ego poses of the current
    lidar sample and of the adjacent CAM_FRONT frame), as get_data_info does.
  * upstream quirk, kept by default (`adj_cam_swap=True`): the adjacent frames' image list follows
    the seq converter's camera order (..., B, BR, BL) while their projections are indexed with the
    current order (..., B, BL, BR), so previous frames' back-left image is projected with the
    back-right calibration and vice versa. The checkpoint was trained that way.
  * a scene's first keyframe has no previous frames; upstream test then falls back to *future*
    frames ('next'), which a live stream cannot have. Here the current frame is repeated (with
    identity motion) instead -- a documented deviation, only on frame 0 of a scene.

Fast-BEV++ R50 (ymlab/advanced-fastbev, configs/fastbev/paper/fastbev-r50-cbgs.py, BEVDet-based)
  * cameras FL, F, FR, BL, B, BR; PrepareImageInputs test: resize 0.44, crop rows 140..396 (the
    bottom 256 rows), mmlabNormalize: PIL RGB array through imnormalize(to_rgb=True), which swaps it
    to BGR before subtracting the RGB mean -- a BEVDet quirk the checkpoint was trained with.
  * sensor2keyego = inv(ego2global of camera 0 (FL)) @ ego2global(cam) @ sensor2ego(cam)
  * single frame.

GT: annotation centers (with >= 1 lidar/radar point), class names, in the frame each model outputs:
the lidar frame for Fast-BEV, ego-at-lidar-time for Fast-BEV++ (BEVDet's evaluation treats its
ego-frame boxes that way).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

M0_CAMS = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
# nuscenes_seq_converter.py's order for the adjacent frames' image list
M0_ADJ_CAMS = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_RIGHT", "CAM_BACK_LEFT"]
PP_CAMS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
IMG_MEAN = np.array([123.675, 116.28, 103.53], np.float32)
IMG_STD = np.array([58.395, 57.12, 57.375], np.float32)
SRC_HW = (900, 1600)
IN_HW = (256, 704)
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


def rt(r, t):
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = t
    return m


def resize_crop(path, top):
    """PIL resize to 704x396 (default filter: BICUBIC for RGB) then crop rows top..top+256 -> uint8 HWC RGB."""
    from PIL import Image

    img = Image.open(path).convert("RGB")
    s = IN_HW[1] / SRC_HW[1]
    img = img.resize((int(SRC_HW[1] * s), int(SRC_HW[0] * s)), Image.BICUBIC)
    img = img.crop((0, top, IN_HW[1], top + IN_HW[0]))
    return np.asarray(img, dtype=np.uint8)


def normalize(rgb_u8, bgr=False):
    """-> float32 CHW. bgr=True reproduces mmlabNormalize (channels swapped, RGB mean/std)."""
    x = rgb_u8.astype(np.float32)
    if bgr:
        x = x[..., ::-1]
    return ((x - IMG_MEAN) / IMG_STD).transpose(2, 0, 1).copy()


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
                self.sd.setdefault(r["sample_token"], {})[self.channel(r)] = r
        self.anns = {}
        for a in t["sample_annotation"]:
            self.anns.setdefault(a["sample_token"], []).append(a)

    def channel(self, sd):
        cs = self.tok["calibrated_sensor"][sd["calibrated_sensor_token"]]
        return self.tok["sensor"][cs["sensor_token"]]["channel"]

    def scene_samples(self, scene):
        out, tok = [], self.scenes[scene]["first_sample_token"]
        while tok:
            out.append(tok)
            tok = self.tok["sample"][tok]["next"]
        return out

    def pose(self, sd):
        """-> sensor2ego (4x4), ego2global (4x4), calibrated_sensor record"""
        cs = self.tok["calibrated_sensor"][sd["calibrated_sensor_token"]]
        ep = self.tok["ego_pose"][sd["ego_pose_token"]]
        return (rt(quat_to_mat(cs["rotation"]), cs["translation"]),
                rt(quat_to_mat(ep["rotation"]), ep["translation"]), cs)

    def gt(self, sample_token, world2frame):
        out = []
        for a in self.anns.get(sample_token, []):
            cat = self.tok["category"][self.tok["instance"][a["instance_token"]]["category_token"]]["name"]
            if cat not in CATEGORY or a["num_lidar_pts"] + a["num_radar_pts"] == 0:
                continue
            p = world2frame @ np.array([*a["translation"], 1.0])
            out.append((CATEGORY[cat], p[:3]))
        return out

    # ---------------------------------------------------------------- Fast-BEV M0
    def _sensor2lidar(self, cam_sd, l2e, e2g):
        """mmdet3d obtain_sensor2top(): camera -> lidar (at the lidar sample's time)."""
        s2e, e2g_s, _ = self.pose(cam_sd)
        return np.linalg.inv(l2e) @ np.linalg.inv(e2g) @ e2g_s @ s2e

    def _adjacent(self, sample_token, k, keyframe_fallback=False):
        """Prev 'sweep' k (1, 3, 5 -> camera frames 6, 12, 18 back) as the seq converter + the
        test-time clamp pick it: -> ({cam: sample_data}, CAM_FRONT sample_data) or None.
        The camera-only slice has keyframes only; in scenes where a pick is a sweep,
        keyframe_fallback=True takes that camera's keyframe nearest in time instead (calibration
        only -- the eval scene-0103's picks are all keyframes)."""
        sds = self.sd[sample_token]
        chains = {}
        for cam in M0_CAMS:
            lst, s = [], sds[cam]
            while s["prev"] and len(lst) < 60:
                s = self.tok["sample_data"][s["prev"]]
                lst.append(s)
            chains[cam] = lst
        counts = list(range(2, min(60, len(chains["CAM_FRONT"])), 3))
        if not counts:
            return None
        count = counts[min(k, len(counts) - 1)]
        front = chains["CAM_FRONT"][count]
        picks = {}
        for cam in M0_CAMS:
            ts = np.array([x["timestamp"] for x in chains[cam]], np.int64)
            picks[cam] = chains[cam][int(np.argmin(np.abs(ts - front["timestamp"])))]
        for cam, s in list(picks.items()):
            if s["is_key_frame"]:
                continue
            assert keyframe_fallback, f"adjacent frame {cam} {s['filename']} is a sweep, not in the keyframe slice"
            keys = [x for x in chains[cam] if x["is_key_frame"]] or [sds[cam]]
            picks[cam] = min(keys, key=lambda x: abs(x["timestamp"] - s["timestamp"]))
        if not front["is_key_frame"]:
            front = picks["CAM_FRONT"]
        return picks, front

    def m0_frame(self, sample_token, adj_cam_swap=True, n_times=4, adj_ids=(1, 3, 5), keyframe_fallback=False):
        """-> dict(img_u8 (T, 6, 256, 704, 3) RGB, img (T*6, 3, 256, 704) normalized,
                   lidar2img (T, 6, 4, 4) incl. the image resize/crop, gt [(name, xyz lidar)])"""
        sds = self.sd[sample_token]
        l2e, e2g, _ = self.pose(sds["LIDAR_TOP"])
        s = IN_HW[1] / SRC_HW[1]
        top = (int(SRC_HW[0] * s) - IN_HW[0]) // 2
        post = np.eye(4)
        post[0, 0] = post[1, 1] = s
        post[1, 2] = -top
        s2l, intr = [], []
        for cam in M0_CAMS:
            s2l.append(self._sensor2lidar(sds[cam], l2e, e2g))
            intr.append(np.array(self.pose(sds[cam])[2]["camera_intrinsic"]))

        def proj(s2l_m, K):
            vp = np.eye(4)
            vp[:3, :3] = K
            return post @ vp @ np.linalg.inv(s2l_m)

        imgs = [[resize_crop(self.root / sds[c]["filename"], top) for c in M0_CAMS]]
        l2i = [[proj(s2l[i], intr[i]) for i in range(6)]]
        for t in range(1, n_times):
            adj = self._adjacent(sample_token, adj_ids[t - 1], keyframe_fallback)
            if adj is None:  # scene start: repeat the current frame (see module doc)
                imgs.append(imgs[0])
                l2i.append(l2i[0])
                continue
            picks, front = adj
            _, e2g_adj, _ = self.pose(front)
            lidaradj2lidarcur = np.linalg.inv(l2e) @ np.linalg.inv(e2g) @ e2g_adj @ l2e
            order = M0_ADJ_CAMS if adj_cam_swap else M0_CAMS
            imgs.append([resize_crop(self.root / picks[c]["filename"], top) for c in order])
            l2i.append([proj(lidaradj2lidarcur @ s2l[i], intr[i]) for i in range(6)])
        img_u8 = np.stack([np.stack(x) for x in imgs])
        img = np.stack([normalize(im) for row in imgs for im in row])
        gt = self.gt(sample_token, np.linalg.inv(e2g @ l2e))
        return {"img_u8": img_u8, "img": img, "lidar2img": np.array(l2i, np.float64), "gt": gt,
                "token": sample_token}

    # ---------------------------------------------------------------- Fast-BEV++
    def pp_frame(self, sample_token):
        """-> dict(img_u8 (6, 256, 704, 3) RGB, img (6, 3, 256, 704) normalized as BEVDet does,
                   sensor2keyego (6, 4, 4), intrin (6, 3, 3), post (3, 3) resize/crop,
                   gt [(name, xyz ego@lidar time)])"""
        sds = self.sd[sample_token]
        s = IN_HW[1] / SRC_HW[1]
        new_h = int(SRC_HW[0] * s)
        top = new_h - IN_HW[0]
        imgs, s2k, intr = [], [], []
        key_e2g = self.pose(sds[PP_CAMS[0]])[1]
        for cam in PP_CAMS:
            s2e, e2g, cs = self.pose(sds[cam])
            imgs.append(resize_crop(self.root / sds[cam]["filename"], top))
            s2k.append(np.linalg.inv(key_e2g) @ e2g @ s2e)
            intr.append(np.array(cs["camera_intrinsic"]))
        post = np.eye(3)
        post[0, 0] = post[1, 1] = s
        post[1, 2] = -top
        _, e2g_l, _ = self.pose(sds["LIDAR_TOP"])
        img_u8 = np.stack(imgs)
        return {"img_u8": img_u8, "img": np.stack([normalize(i, bgr=True) for i in imgs]),
                "sensor2keyego": np.array(s2k), "intrin": np.array(intr), "post": post,
                "gt": self.gt(sample_token, np.linalg.inv(e2g_l)), "token": sample_token}


def match(boxes_xy, scores, names, gt, thr=0.3, dist=2.0):
    """Greedy BEV center-distance match, same criteria as ../bevformer_tiny (score >= 0.3, same
    class, within 2 m) -> (tp, n_pred, n_gt)."""
    keep = np.nonzero(scores >= thr)[0]
    used, tp = set(), 0
    for i in keep[np.argsort(-scores[keep], kind="stable")]:
        best, bj = dist, None
        for j, (name, p) in enumerate(gt):
            if j in used or name != names[i]:
                continue
            d = math.hypot(float(boxes_xy[i, 0]) - p[0], float(boxes_xy[i, 1]) - p[1])
            if d < best:
                best, bj = d, j
        if bj is not None:
            used.add(bj)
            tp += 1
    return tp, len(keep), len(gt)
