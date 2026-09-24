"""Shared bits: image preprocessing (HF RTDetrImageProcessor: bilinear resize to 640x640, /255, no
normalization), the COCO id lists (the deploy pipeline's yolo11n calibration/eval ids), detection
matching (same metric as ../../deploy/stages/accuracy.py)."""

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


def coco_ids(section: str) -> list[int]:
    spec = yaml.safe_load(open(HERE.parent.parent / "deploy/models/yolo11n.yaml"))
    return spec[section]["coco_val2017"]


def image_paths(section: str) -> list[Path]:
    sec = {"coco_val2017": coco_ids(section)}
    fetch_images(sec, IMAGES)
    return [IMAGES / f"coco_{i:012d}.jpg" for i in sec["coco_val2017"]]


def load_rgb_u8(path: Path, size: int = 640) -> np.ndarray:
    """(640, 640, 3) uint8 RGB, PIL bilinear resize like the HF processor (resample=2)."""
    im = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def to_pixels(rgb_u8: np.ndarray) -> np.ndarray:
    """(1, 3, H, W) float32 in [0, 1]."""
    return (rgb_u8.astype(np.float32) / 255.0).transpose(2, 0, 1)[None].copy()


def decode(logits: np.ndarray, boxes: np.ndarray, k: int = 300):
    """HF post_process_object_detection (use_focal_loss): sigmoid, top-k over queries x classes.
    Returns (boxes xyxy in 640 px, scores, labels)."""
    lg = logits.reshape(-1, logits.shape[-1])
    s = 1 / (1 + np.exp(-lg.astype(np.float64)))
    flat = s.ravel()
    idx = np.argsort(-flat, kind="stable")[:k]
    q, c = idx // lg.shape[1], idx % lg.shape[1]
    b = boxes.reshape(-1, 4)[q].astype(np.float64)
    xyxy = (
        np.stack(
            [
                b[:, 0] - b[:, 2] / 2,
                b[:, 1] - b[:, 3] / 2,
                b[:, 0] + b[:, 2] / 2,
                b[:, 1] + b[:, 3] / 2,
            ],
            1,
        )
        * 640
    )
    return xyxy, flat[idx], c


def match(ref, got, iou=0.5, score=0.3):
    """ref/got: (logits, boxes). Detections at score >= `score` matched by class + IoU."""
    return det_match(decode(*ref), decode(*got), iou, score)
