"""Shared bits: the COCO id lists (the deploy pipeline's yolo11n calibration/eval ids, the same images
RT-DETR used), RF-DETR's preprocessing and decode, detection matching (same metric as
../../deploy/stages/accuracy.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "deploy"))

from stages.accuracy import det_match  # noqa: E402
from stages.images import fetch_images  # noqa: E402

IMAGES = Path.home() / ".cache/onnxsim-deploy/_images"
WORK = Path.home() / ".cache/onnxsim-rfdetr/work"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def coco_ids(section: str) -> list[int]:
    spec = yaml.safe_load(open(HERE.parent.parent / "deploy/models/yolo11n.yaml"))
    return spec[section]["coco_val2017"]


def image_paths(section: str) -> list[Path]:
    sec = {"coco_val2017": coco_ids(section)}
    fetch_images(sec, IMAGES)
    return [IMAGES / f"coco_{i:012d}.jpg" for i in sec["coco_val2017"]]


def load_rgb_u8(path: Path, size: int) -> np.ndarray:
    """(size, size, 3) uint8 RGB. RF-DETR's predict() resizes the float image bilinearly without
    antialias (torchvision F.resize(antialias=False)) to a square; do the same, then round to uint8 so
    the phone (uint8 input) and the fp32 reference see identical pixels."""
    import torch
    import torch.nn.functional as F

    im = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    t = torch.from_numpy(im).permute(2, 0, 1)[None]
    t = F.interpolate(
        t, size=(size, size), mode="bilinear", align_corners=False, antialias=False
    )
    return t[0].permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8).numpy()


def to_pixels(rgb_u8: np.ndarray) -> np.ndarray:
    """(1, 3, H, W) float32, ImageNet-normalized (RF-DETR's means/stds)."""
    x = (rgb_u8.astype(np.float32) / 255.0 - MEAN) / STD
    return x.transpose(2, 0, 1)[None].copy()


def decode(logits: np.ndarray, boxes: np.ndarray, size: int, k: int = 300):
    """RF-DETR PostProcess: sigmoid, top-k over queries x classes, cxcywh -> xyxy (in `size` px)."""
    lg = logits.reshape(-1, logits.shape[-1])
    s = 1 / (1 + np.exp(-lg.astype(np.float64)))
    flat = s.ravel()
    idx = np.argsort(-flat, kind="stable")[:k]
    q, c = idx // lg.shape[1], idx % lg.shape[1]
    b = boxes.reshape(-1, 4)[q].astype(np.float64)
    xyxy = np.stack(
        [
            b[:, 0] - b[:, 2] / 2,
            b[:, 1] - b[:, 3] / 2,
            b[:, 0] + b[:, 2] / 2,
            b[:, 1] + b[:, 3] / 2,
        ],
        1,
    )
    return xyxy * size, flat[idx], c


def match(ref, got, size, iou=0.5, score=0.3):
    """ref/got: (logits, boxes). Detections at score >= `score` matched by class + IoU."""
    return det_match(decode(*ref, size), decode(*got, size), iou, score)
