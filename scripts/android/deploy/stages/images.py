"""Real images for calibration and evaluation, and the per-model preprocessing.

Spec sections `calibration:` / `eval:` name images by COCO val2017 id (downloaded once into the
shared work/_images cache) and/or local paths:

    calibration: {coco_val2017: [139, 285, ...]}
    eval:        {coco_val2017: [139, 632], files: [/path/to/cats.jpg]}

`preprocess:` describes how an image becomes the model input (float32, CHW, batch dim added by
the caller):
    kind: letterbox        # YOLO-style: keep aspect, pad with `pad` (default 114), RGB, x/255
    kind: resize_normalize # plain resize to `size`, RGB, (x/255 - mean) / std
    size: [H, W]
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

COCO = "http://images.cocodataset.org/val2017/{:012d}.jpg"


def _paths(section: dict, cache: Path) -> list[Path]:
    out = [cache / f"coco_{i:012d}.jpg" for i in section.get("coco_val2017", [])]
    out += [Path(p).expanduser() for p in section.get("files", [])]
    return out


def fetch_images(section: dict, cache: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    cache.mkdir(parents=True, exist_ok=True)

    def get(i):  # atomic: an interrupted download never leaves a truncated .jpg behind
        p = cache / f"coco_{i:012d}.jpg"
        if p.exists():
            return
        for attempt in range(4):  # the COCO image server times out now and then
            try:
                with urllib.request.urlopen(COCO.format(i), timeout=120) as r:
                    data = r.read()
                break
            except OSError:
                if attempt == 3:
                    raise
        tmp = p.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.rename(p)

    with ThreadPoolExecutor(8) as ex:  # network-bound; images.cocodataset.org is slow per request
        list(ex.map(get, section.get("coco_val2017", [])))
    missing = [p for p in _paths(section, cache) if not p.exists()]
    if missing:
        raise SystemExit(f"missing images: {missing[:3]}")


def list_images(section: dict, cache: Path) -> list[Path]:
    return _paths(section, cache)


def load_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


def preprocess(path: Path, pre: dict) -> tuple[np.ndarray, dict]:
    """-> (float32 CHW tensor, meta for mapping outputs back to the original image)."""
    from PIL import Image

    img = load_rgb(path)
    h0, w0 = img.shape[:2]
    H, W = pre["size"]
    kind = pre.get("kind", "letterbox")
    if kind == "letterbox":
        r = min(H / h0, W / w0)
        nh, nw = round(h0 * r), round(w0 * r)
        top, left = (H - nh) // 2, (W - nw) // 2
        canvas = np.full((H, W, 3), pre.get("pad", 114), np.uint8)
        canvas[top:top + nh, left:left + nw] = np.asarray(
            Image.fromarray(img).resize((nw, nh), Image.BILINEAR))
        x = canvas.astype(np.float32) / 255.0
        meta = {"scale": r, "pad": [left, top], "orig": [h0, w0]}
    elif kind == "resize_normalize":
        x = np.asarray(Image.fromarray(img).resize((W, H), Image.BICUBIC)).astype(np.float32) / 255.0
        x = (x - np.array(pre.get("mean", [0, 0, 0]), np.float32)) / np.array(pre.get("std", [1, 1, 1]), np.float32)
        meta = {"scale": [H / h0, W / w0], "orig": [h0, w0]}
    else:
        raise SystemExit(f"unknown preprocess kind {kind}")
    return np.ascontiguousarray(x.transpose(2, 0, 1)), meta
