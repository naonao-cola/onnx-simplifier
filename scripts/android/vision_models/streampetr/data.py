"""nuScenes-mini frames the way StreamPETR's test pipeline feeds them, without nuscenes-devkit/mmdet3d.

Reuses ../bevformer_tiny/nuscenes.py's metadata tables, GT extraction and 2 m center-distance match
(loaded by path: that file imports its own ``model``). Per keyframe this reproduces:
  * NuScenesDataset.get_data_info(): ego_pose = ego2global @ lidar2ego (lidar -> global), its
    inverse, timestamp in seconds, per camera intrinsics (4x4 viewpad) and lidar2cam extrinsics
  * LoadMultiViewImageFromFiles (mmcv BGR) -> ResizeCropFlipRotImage(training=False) at 256x704:
    resize = max(256/900, 704/1600) = 0.44 -> PIL resize to 704x396 (PIL's default filter) ->
    crop rows 140..396; intrinsics <- ida_mat @ intrinsics; lidar2img = intrinsics @ extrinsics
  * NormalizeMultiviewImage(to_rgb=True) mean/std; PadMultiViewImage(32) is a no-op at 256x704
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MEAN = np.array([123.675, 116.28, 103.53], np.float32)
STD = np.array([58.395, 57.12, 57.375], np.float32)
H, W = 256, 704
RESIZE = max(H / 900, W / 1600)
RESIZE_DIMS = (int(1600 * RESIZE), int(900 * RESIZE))  # (704, 396)
CROP = (0, RESIZE_DIMS[1] - H, W, RESIZE_DIMS[1])  # (0, 140, 704, 396)


def _load_bevformer_nuscenes():
    """../bevformer_tiny/nuscenes.py without putting its directory (and its ``model``) on sys.path."""
    path = Path(__file__).resolve().parents[1] / "bevformer_tiny" / "nuscenes.py"
    saved = sys.modules.get("model")
    stub = types.ModuleType("model")
    stub.IMG_MEAN, stub.IMG_STD = [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]  # only its frame() uses them
    sys.modules["model"] = stub
    try:
        spec = importlib.util.spec_from_file_location("bevformer_nuscenes", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        if saved is None:
            sys.modules.pop("model", None)
        else:
            sys.modules["model"] = saved
    return mod


NS = _load_bevformer_nuscenes()
CAMS, CLASSES, match = NS.CAMS, NS.CLASSES, NS.match


def _rt(r, t):
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = t
    return m


class NuScenesStream(NS.NuScenesMini):
    def stream_frame(self, sample_token):
        sds = self.sd[sample_token]
        l2e_r, l2e_t, e2g_r, e2g_t, _, _ = self._pose(sds["LIDAR_TOP"])
        ego_pose = _rt(e2g_r, e2g_t) @ _rt(l2e_r, l2e_t)  # lidar -> global
        ida = np.eye(3)
        ida[:2, :2] *= RESIZE
        ida[:2, 2] -= CROP[:2]
        imgs_u8, intr, extr = [], [], []
        for cam in CAMS:
            sd = sds[cam]
            s_l2e_r, s_l2e_t, s_e2g_r, s_e2g_t, cs, _ = self._pose(sd)
            # obtain_sensor2top: camera -> lidar (the converter's sensor2lidar_rotation/translation)
            cam2global = _rt(s_e2g_r, s_e2g_t) @ _rt(s_l2e_r, s_l2e_t)
            cam2lidar = np.linalg.inv(ego_pose) @ cam2global
            extr.append(np.linalg.inv(cam2lidar))
            k = np.eye(4)
            k[:3, :3] = ida @ np.array(cs["camera_intrinsic"])
            intr.append(k)
            img = Image.open(self.root / sd["filename"]).convert("RGB")
            img = img.resize(RESIZE_DIMS).crop(CROP)
            imgs_u8.append(np.asarray(img, np.uint8))
        intr, extr = np.stack(intr), np.stack(extr)
        info = self.frame(sample_token)  # GT (lidar frame) + scene token; its image is unused
        return {
            "img_u8": np.stack(imgs_u8),  # (6, 256, 704, 3) RGB, what a camera pipeline hands over
            "intrinsics": intr, "extrinsics": extr, "lidar2img": intr @ extr,
            "ego_pose": ego_pose, "ego_pose_inv": np.linalg.inv(ego_pose),
            "timestamp": self.tok["sample"][sample_token]["timestamp"] / 1e6,
            "gt": info["gt"], "scene_token": info["scene_token"],
        }


def normalize(img_u8):
    """(6, H, W, 3) uint8 RGB -> (6, 3, H, W) float32 normalized, NormalizeMultiviewImage."""
    x = (img_u8.astype(np.float32) - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))
