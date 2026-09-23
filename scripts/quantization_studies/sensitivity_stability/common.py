"""Shared setup for the sensitivity-stability study: pinned models, data, metric."""

from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np

WORK = Path(os.environ.get("SENSSTAB_WORK", Path.home() / ".cache" / "sensstab"))
IMAGES = os.environ.get("SENSSTAB_IMAGES", str(Path.home() / ".cache" / "coco128"))

# name -> (family, source, identifier). Same family == identical architecture, so node names
# (and therefore block groups and mixed-precision policies) line up between checkpoints.
MODELS = {
    "tv_r50_v1": ("resnet50_tv", "torchvision", "IMAGENET1K_V1"),
    "tv_r50_v2": ("resnet50_tv", "torchvision", "IMAGENET1K_V2"),
    "timm_r50_a1": ("resnet50_timm", "timm", "resnet50.a1_in1k"),
    "timm_r50_a2": ("resnet50_timm", "timm", "resnet50.a2_in1k"),
    "timm_r50_a3": ("resnet50_timm", "timm", "resnet50.a3_in1k"),
    "timm_r50_ssl": ("resnet50_timm", "timm", "resnet50.fb_ssl_yfcc100m_ft_in1k"),
    "vit_s_augreg": ("vit_s16", "timm", "vit_small_patch16_224.augreg_in21k_ft_in1k"),
    "deit_s": ("vit_s16", "timm", "deit_small_patch16_224.fb_in1k"),
}

# bottleneck / transformer-block granularity (the default block key would stop at the stage)
BLOCK_REGEX = {
    "resnet50_tv": r"/(layer\d/layer\d\.\d+)/",
    "resnet50_timm": r"/(layer\d/layer\d\.\d+)/",
    "vit_s16": r"/(blocks/blocks\.\d+)/",
}


def torch_model(name):
    import torch  # noqa: F401

    fam, src, ident = MODELS[name]
    if src == "torchvision":
        import torchvision

        return torchvision.models.resnet50(weights=ident).eval()
    import timm

    return timm.create_model(ident, pretrained=True).eval()


def norm_params(name):
    fam, src, ident = MODELS[name]
    if src == "timm":
        import timm

        cfg = timm.data.resolve_data_config({}, model=torch_model(name))
        return tuple(cfg["mean"]), tuple(cfg["std"])
    return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def onnx_path(name) -> Path:
    return WORK / "models" / f"{name}.onnx"


def image_files():
    files = sorted(glob.glob(os.path.join(IMAGES, "**", "*.jpg"), recursive=True))
    assert len(files) >= 128, f"need >= 128 images under {IMAGES}, found {len(files)}"
    return files[:128]


def split(which: str):
    """(calibration files, eval files): 'A' = first/last 64, 'B' = even/odd (the noise floor
    reruns with B, a different disjoint split of the same pool)."""
    f = image_files()
    if which == "A":
        return f[:64], f[64:]
    if which == "B":
        return f[0::2], f[1::2]
    raise ValueError(which)


def load_batches(files, mean, std, input_name="input"):
    from PIL import Image

    out = []
    m = np.array(mean, np.float32).reshape(1, 3, 1, 1)
    s = np.array(std, np.float32).reshape(1, 3, 1, 1)
    for p in files:
        im = Image.open(p).convert("RGB")
        w, h = im.size
        k = 256 / min(w, h)
        im = im.resize((round(w * k), round(h * k)), Image.BICUBIC)
        w, h = im.size
        left, top = (w - 224) // 2, (h - 224) // 2
        x = np.asarray(im.crop((left, top, left + 224, top + 224)), np.float32) / 255.0
        x = (x.transpose(2, 0, 1)[None] - m) / s
        out.append({input_name: x.astype(np.float32)})
    return out


def mean_logit_sqnr(float_outputs, quant_outputs) -> float:
    """Mean over images of the logits' SQNR in dB (continuous, so small per-block effects rank)."""
    vals = []
    for fo, qo in zip(float_outputs, quant_outputs):
        for k in fo:
            a = fo[k].astype(np.float64).ravel()
            b = qo[k].astype(np.float64).ravel()
            if not np.all(np.isfinite(b)):
                return float("-inf")
            err = np.sum((a - b) ** 2)
            vals.append(
                200.0 if err == 0 else min(200.0, 10 * np.log10(np.sum(a * a) / err))
            )
    return float(np.mean(vals))


def top1_agreement(float_outputs, quant_outputs) -> float:
    agree = [
        int(np.argmax(fo[k]) == np.argmax(qo[k]))
        for fo, qo in zip(float_outputs, quant_outputs)
        for k in fo
    ]
    return float(np.mean(agree))
