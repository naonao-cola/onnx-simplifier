"""Preprocess Imagenette (a 10-class subset of real ImageNet images) for accuracy evals.

Imagenette's val split is 3,925 actual ImageNet images of 10 ImageNet-1k classes, so a
pretrained ImageNet classifier can be scored by its ordinary 1000-way top-1 on them (a
usable accuracy proxy when the gated ImageNet val set is unavailable; the classes are
easy ones, so absolute numbers run high -- compare variants, not against published
ImageNet top-1).

    curl -LO https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz && tar xzf it.tgz
    python imagenette_data.py imagenette2-320 OUT_DIR [imagenet|vit]

Writes ``val_x.npy`` (uint8 N,224,224,3 -- resize 256, center-crop 224), ``val_y.npy``
(ImageNet-1k class ids), and ``calib_x.npy`` (128 *train* images, float32 NCHW,
ImageNet-normalized) for ModelOpt calibration, kept disjoint from val.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Imagenette wnid -> ImageNet-1k class index (wnids sorted, as in the standard labels).
WNID_TO_IDX = {
    "n01440764": 0,    # tench
    "n02102040": 217,  # English springer
    "n02979186": 482,  # cassette player
    "n03000684": 491,  # chain saw
    "n03028079": 497,  # church
    "n03394916": 566,  # French horn
    "n03417042": 569,  # garbage truck
    "n03425413": 571,  # gas pump
    "n03445777": 574,  # golf ball
    "n03888257": 701,  # parachute
}
PRESETS = {
    # torchvision/zoo CNNs: resize short side to 256, center-crop 224, ImageNet mean/std
    "imagenet": dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), crop=True),
    # HF google/vit-base-patch16-224: direct resize to 224x224, mean=std=0.5
    "vit": dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop=False),
}


def load(path, crop=True):
    im = Image.open(path).convert("RGB")
    if not crop:
        return np.asarray(im.resize((224, 224), Image.BILINEAR), dtype=np.uint8)
    w, h = im.size
    s = 256 / min(w, h)
    im = im.resize((max(256, round(w * s)), max(256, round(h * s))), Image.BILINEAR)
    w, h = im.size
    l, t = (w - 224) // 2, (h - 224) // 2
    return np.asarray(im.crop((l, t, l + 224, t + 224)), dtype=np.uint8)


def normalize(x_uint8_nhwc, mean, std):
    x = (x_uint8_nhwc.astype(np.float32) / 255.0 - np.array(mean, np.float32)) / np.array(std, np.float32)
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


def collect(root):
    items = []
    for wnid, idx in WNID_TO_IDX.items():
        for p in sorted((Path(root) / wnid).glob("*.JPEG")):
            items.append((p, idx))
    return items


def main(root, out_dir, preset="imagenet"):
    cfg = PRESETS[preset]
    root, out = Path(root), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    val = collect(root / "val")
    np.save(out / "val_x.npy", np.stack([load(p, cfg["crop"]) for p, _ in val]))
    np.save(out / "val_y.npy", np.array([y for _, y in val], np.int64))
    train = collect(root / "train")
    pick = np.random.default_rng(0).choice(len(train), 128, replace=False)
    np.save(out / "calib_x.npy", normalize(np.stack([load(train[i][0], cfg["crop"]) for i in pick]), cfg["mean"], cfg["std"]))
    print(f"val: {len(val)} images, calib: 128 train images")


if __name__ == "__main__":
    main(*sys.argv[1:4])
