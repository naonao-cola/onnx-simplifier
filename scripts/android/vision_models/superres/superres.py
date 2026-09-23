#!/usr/bin/env python3
"""Single-image neural super-resolution (x4) on the Xiaomi 12S HTP -- the open, camera-usable part of a
"DLSS on a phone" demo (DLSS itself is closed; see README.md).

  superres.py fetch                      weights (sha256-pinned) -> $SR_WORK/w
  superres.py build  <model> <H> <W>     float core + fp16 (uint8 NHWC I/O) + int8 (full_qdq, uint8 I/O)
  superres.py host   <model>             PSNR vs ground truth on nuScenes/COCO crops: fp32, int8 (host ORT), bicubic
  superres.py phone  <model> <H> <W>     fp16 + int8 on the phone (strict all-HTP) -> ms + PSNR vs host fp32
  superres.py report                     README tables from $SR_WORK/results/*.json

Models (all x4, RGB in [0, 1]):
  quicksrnet{small,medium,large}, xlsr   Qualcomm AI Hub float ONNX (BSD-3-Clause)
  realesr-general-x4v3, realesr-animevideov3   Real-ESRGAN SRVGGNetCompact (BSD-3-Clause)
Each phone run goes through ~/.cache/android-phone/phone-run (the shared phone lock) via
../sam/phone.sh, with R=/data/local/tmp/codex-android-superres.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("SR_WORK", Path.home() / ".cache" / "superres"))
W = WORK / "w"
SCALE = 4
AIHUB = "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/{m}/releases/v0.62.2/{m}-onnx-float.zip"
ESRGAN = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/{m}.pth"
# sha256 of the downloaded archive / checkpoint
WEIGHTS = {
    "quicksrnetsmall": ("aihub", "ac9abc7193d0019f"),
    "quicksrnetmedium": ("aihub", "c335adddccc02cfb"),
    "quicksrnetlarge": ("aihub", "6510c68a67013ca9"),
    "xlsr": ("aihub", "ef5cdbe2450ed928"),
    "realesr-general-x4v3": ("esrgan", "8dc7edb9ac80ccdc", 32),
    "realesr-animevideov3": ("esrgan", "b8a8376811077954", 16),
}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def fetch() -> None:
    W.mkdir(parents=True, exist_ok=True)
    for m, spec in WEIGHTS.items():
        kind, sha = spec[0], spec[1]
        dst = W / (f"{m}.zip" if kind == "aihub" else f"{m}.pth")
        if not dst.exists():
            urllib.request.urlretrieve(
                (AIHUB if kind == "aihub" else ESRGAN).format(m=m), dst
            )
        got = _sha(dst)
        if got != sha:
            sys.exit(f"{dst}: sha256 {got}... != pinned {sha}...")
        if kind == "aihub" and not (W / m).exists():
            zipfile.ZipFile(dst).extractall(W / m)
        print(m, "ok")


# ---------------------------------------------------------------- float core (NCHW, [0, 1] in/out)
def _srvgg_onnx(m: str, num_conv: int) -> onnx.ModelProto:
    """Real-ESRGAN's SRVGGNetCompact as plain ONNX (Conv/PRelu/Add/DepthToSpace). The nearest-x4
    residual `out += interpolate(x, 4, nearest)` becomes a fixed 1x1 conv of the input image added
    before the PixelShuffle (CRD): pre-shuffle channel c*16+k is input channel c -- exact, no Resize."""
    import torch

    sd = torch.load(W / f"{m}.pth", map_location="cpu")
    sd = sd.get("params", sd.get("params_ema", sd))
    convs = [
        k[: -len(".weight")] for k in sd if k.endswith(".weight") and sd[k].dim() == 4
    ]
    prelus = [
        k[: -len(".weight")] for k in sd if k.endswith(".weight") and sd[k].dim() == 1
    ]
    assert len(convs) == num_conv + 2 and len(prelus) == num_conv + 1, (
        len(convs),
        len(prelus),
    )
    nodes, inits = [], []
    x = "image"
    for i, c in enumerate(convs):
        w = sd[c + ".weight"].numpy().astype(np.float32)
        b = sd[c + ".bias"].numpy().astype(np.float32)
        inits += [
            numpy_helper.from_array(w, f"w{i}"),
            numpy_helper.from_array(b, f"b{i}"),
        ]
        nodes.append(
            helper.make_node(
                "Conv", [x, f"w{i}", f"b{i}"], [f"c{i}"], pads=[1, 1, 1, 1]
            )
        )
        x = f"c{i}"
        if i < len(prelus):
            s = sd[prelus[i] + ".weight"].numpy().astype(np.float32).reshape(-1, 1, 1)
            inits.append(numpy_helper.from_array(s, f"s{i}"))
            nodes.append(helper.make_node("PRelu", [x, f"s{i}"], [f"a{i}"]))
            x = f"a{i}"
    rep = np.zeros((3 * SCALE * SCALE, 3, 1, 1), np.float32)
    for ch in range(3):
        rep[ch * SCALE * SCALE : (ch + 1) * SCALE * SCALE, ch] = 1.0
    inits.append(numpy_helper.from_array(rep, "w_rep"))
    nodes += [
        helper.make_node("Conv", ["image", "w_rep"], ["base"]),
        helper.make_node("Add", [x, "base"], ["pre"]),
        helper.make_node(
            "DepthToSpace", ["pre"], ["upscaled_image"], blocksize=SCALE, mode="CRD"
        ),
    ]
    g = helper.make_graph(
        nodes,
        m,
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, "H", "W"])],
        [
            helper.make_tensor_value_info(
                "upscaled_image", TensorProto.FLOAT, [1, 3, "H4", "W4"]
            )
        ],
        inits,
    )
    return helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])


def core(m: str, h: int, w: int) -> onnx.ModelProto:
    spec = WEIGHTS[m]
    if spec[0] == "aihub":
        model = onnx.load(glob.glob(str(W / m / "*" / f"{m}.onnx"))[0])
    else:
        model = _srvgg_onnx(m, spec[2])
    g = model.graph
    del g.value_info[:]
    for vi, (hh, ww) in ((g.input[0], (h, w)), (g.output[0], (h * SCALE, w * SCALE))):
        d = vi.type.tensor_type.shape.dim
        for k, v in enumerate((1, 3, hh, ww)):
            d[k].Clear()
            d[k].dim_value = v
    return onnx.shape_inference.infer_shapes(model)


def _nhwc_float(core_m: onnx.ModelProto) -> onnx.ModelProto:
    """NHWC float [0,1] in/out around the NCHW core; the output is clamped to [0, 1]."""
    g = core_m.graph
    h, w = (d.dim_value for d in g.input[0].type.tensor_type.shape.dim[2:])
    xin, yout = g.input[0].name, g.output[0].name
    for n in g.node:
        n.input[:] = ["_x_nchw" if i == xin else i for i in n.input]
        n.output[:] = ["_y_nchw" if o == yout else o for o in n.output]
    pre = helper.make_node("Transpose", ["lr"], ["_x_nchw"], perm=[0, 3, 1, 2])
    post = [
        helper.make_node("Clip", ["_y_nchw", "_lo", "_hi"], ["_y_clip"]),
        helper.make_node("Transpose", ["_y_clip"], ["sr"], perm=[0, 2, 3, 1]),
    ]
    g.initializer.extend(
        [
            numpy_helper.from_array(np.array(0, np.float32), "_lo"),
            numpy_helper.from_array(np.array(1, np.float32), "_hi"),
        ]
    )
    nodes = [pre] + list(g.node) + post
    del g.node[:]
    g.node.extend(nodes)
    del g.input[:], g.output[:]
    g.input.append(helper.make_tensor_value_info("lr", TensorProto.FLOAT, [1, h, w, 3]))
    g.output.append(
        helper.make_tensor_value_info(
            "sr", TensorProto.FLOAT, [1, h * SCALE, w * SCALE, 3]
        )
    )
    return core_m


def _fp16_u8io(fm: onnx.ModelProto) -> onnx.ModelProto:
    """uint8 NHWC in/out (scale 1/255, zp 0) around the float NHWC model; QNN runs the rest in fp16."""
    m = onnx.ModelProto()
    m.CopyFrom(fm)
    g = m.graph
    for n in g.node:
        n.input[:] = ["_lr_f" if i == "lr" else i for i in n.input]
        n.output[:] = ["_sr_f" if o == "sr" else o for o in n.output]
    g.initializer.extend(
        [
            numpy_helper.from_array(np.array(1 / 255, np.float32), "_s"),
            numpy_helper.from_array(np.array(0, np.uint8), "_z"),
        ]
    )
    nodes = (
        [helper.make_node("DequantizeLinear", ["lr", "_s", "_z"], ["_lr_f"])]
        + list(g.node)
        + [helper.make_node("QuantizeLinear", ["_sr_f", "_s", "_z"], ["sr"])]
    )
    del g.node[:]
    g.node.extend(nodes)
    g.input[0].type.tensor_type.elem_type = TensorProto.UINT8
    g.output[0].type.tensor_type.elem_type = TensorProto.UINT8
    return m


# ---------------------------------------------------------------- data
def _imgs(kind: str) -> list[Path]:
    if kind == "coco":
        return sorted(Path.home().glob(".cache/coco128/coco128/images/train2017/*.jpg"))
    return sorted(
        Path.home().glob(".cache/onnxsim-bevformer/nuscenes-mini/samples/CAM_*/*.jpg")
    )


def _lr_hr(p: Path, hr_h: int, hr_w: int, off: int = 0):
    """Center crop hr_h x hr_w (HR) and its bicubic /4 (LR), both float [0,1] HWC."""
    from PIL import Image

    im = Image.open(p).convert("RGB")
    if im.width < hr_w or im.height < hr_h:
        im = im.resize((max(hr_w, im.width), max(hr_h, im.height)), Image.BICUBIC)
    x0 = (im.width - hr_w) // 2 + off
    y0 = (im.height - hr_h) // 2
    hr = im.crop((x0, y0, x0 + hr_w, y0 + hr_h))
    lr = hr.resize((hr_w // SCALE, hr_h // SCALE), Image.BICUBIC)
    return np.asarray(lr, np.float32) / 255, np.asarray(hr, np.float32) / 255


def _bicubic_up(lr: np.ndarray) -> np.ndarray:
    from PIL import Image

    im = Image.fromarray(np.round(lr * 255).astype(np.uint8))
    return (
        np.asarray(
            im.resize((im.width * SCALE, im.height * SCALE), Image.BICUBIC), np.float32
        )
        / 255
    )


def _psnr(a: np.ndarray, b: np.ndarray, border: int = SCALE) -> float:
    """PSNR on the Y channel (BT.601), `border` px shaved -- the usual SR convention."""

    def y(x):
        return (
            16 / 255
            + (65.481 * x[..., 0] + 128.553 * x[..., 1] + 24.966 * x[..., 2]) / 255
        )

    d = (y(np.clip(a, 0, 1)) - y(np.clip(b, 0, 1)))[border:-border, border:-border]
    return float(10 * np.log10(1 / max(np.mean(d * d), 1e-12)))


def _run(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    )  # no fused int8 kernels
    s = ort.InferenceSession(
        model.SerializeToString(), so, providers=["CPUExecutionProvider"]
    )
    return s.run(None, {s.get_inputs()[0].name: x})[0]


# ---------------------------------------------------------------- build
def _out(m: str, h: int, w: int) -> Path:
    d = WORK / "models" / f"{m}_{h}x{w}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def build(m: str, h: int, w: int) -> None:
    from onnxsim.full_qdq import quantize_full_qdq, quantized_io

    d = _out(m, h, w)
    fm = _nhwc_float(core(m, h, w))
    onnx.save(fm, d / "fp32.onnx")
    onnx.save(_fp16_u8io(fm), d / "fp16.onnx")
    calib = []  # 16 LR crops of the phone's input size from COCO (disjoint from the nuScenes eval)
    for p in _imgs("coco")[:16]:
        lr, _ = _lr_hr(p, h * SCALE, w * SCALE)
        calib.append({"lr": lr[None]})
    q = quantize_full_qdq(
        fm,
        calib,
        method=os.environ.get("SR_CALIB", "mse"),
        ranges={"lr": (0.0, 1.0), "sr": (0.0, 1.0)},
    )
    q, info = quantized_io(q)
    onnx.save(q, d / "int8.onnx")
    print(m, h, w, "->", d, {k: v for k, v in info.items()})


# ---------------------------------------------------------------- host accuracy vs ground truth
def host(m: str) -> None:
    """GT PSNR at a 224x400 LR (896x1600 HR) crop of 12 nuScenes images and 12 COCO 128x160 crops."""
    from onnxsim.full_qdq import quantize_full_qdq

    res = {}
    for kind, (lh, lw), n in (("nuscenes", (224, 400), 12), ("coco", (128, 160), 12)):
        fm = _nhwc_float(core(m, lh, lw))
        calib = [
            {"lr": _lr_hr(p, lh * 4, lw * 4)[0][None]} for p in _imgs("coco")[-16:]
        ]
        q = quantize_full_qdq(
            fm,
            calib,
            method=os.environ.get("SR_CALIB", "mse"),
            ranges={"lr": (0.0, 1.0), "sr": (0.0, 1.0)},
        )
        imgs = _imgs(kind)[:: max(1, len(_imgs(kind)) // n)][:n]
        acc = {"fp32": [], "int8": [], "bicubic": []}
        for p in imgs:
            lr, hr = _lr_hr(p, lh * SCALE, lw * SCALE)
            acc["fp32"].append(_psnr(_run(fm, lr[None])[0], hr))
            acc["int8"].append(_psnr(_run(q, lr[None])[0], hr))
            acc["bicubic"].append(_psnr(_bicubic_up(lr), hr))
        res[kind] = {k: round(float(np.mean(v)), 3) for k, v in acc.items()}
        print(m, kind, res[kind])
    rp = WORK / "results" / f"{m}.json"
    rp.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(rp.read_text()) if rp.exists() else {}
    old["host"] = res
    rp.write_text(json.dumps(old, indent=1))


# ---------------------------------------------------------------- phone
def phone(m: str, h: int, w: int, iters: int = 20) -> None:
    import re

    d = _out(m, h, w)
    fm = onnx.load(d / "fp32.onnx")
    loc = d / "phone"
    loc.mkdir(exist_ok=True)
    # 3 nuScenes frames at exactly the phone input size (1600x900 -> 480x270 etc.; 16:9)
    from PIL import Image

    lrs = []
    for k, p in enumerate(_imgs("nuscenes")[::40][:3]):
        lr = np.asarray(
            Image.open(p).convert("RGB").resize((w, h), Image.BICUBIC), np.uint8
        )
        (loc / f"lr{k}.bin").write_bytes(lr.tobytes())
        (loc / f"m{k}.txt").write_text(f"lr u8 lr{k}.bin 1,{h},{w},3\n")
        lrs.append(lr)
    rp = WORK / "results" / f"{m}.json"
    rp.parent.mkdir(parents=True, exist_ok=True)
    res = json.loads(rp.read_text()) if rp.exists() else {}
    for prec in ("fp16", "int8"):
        env = dict(
            os.environ,
            R="/data/local/tmp/codex-android-superres",
            PHONE_LOCK_OWNER="codex/android-superres",
        )
        cmd = [
            str(Path.home() / ".cache/android-phone/phone-run"),
            str(HERE.parent / "sam/phone.sh"),
            str(d / f"{prec}.onnx"),
            "htp",
            str(iters),
            str(loc),
            f"{m}_{h}x{w}_{prec}",
            "m0.txt",
            "m1.txt",
            "m2.txt",
        ]
        log = subprocess.run(cmd, env=env, capture_output=True, text=True).stdout
        (loc / f"{prec}.log").write_text(log)
        med = re.search(r"median_ms ([\d.]+)", log)
        ent = {
            "median_ms": float(med.group(1)) if med else None,
            "strict_htp": "PASS" in log,
        }
        ps = []
        for k, lr in enumerate(lrs):
            f = loc / f"out{k}_o0.bin"
            if not f.exists():
                break
            out = (
                np.frombuffer(f.read_bytes(), np.uint8).reshape(h * SCALE, w * SCALE, 3)
                / 255
            )
            ref = _run(fm, (lr.astype(np.float32) / 255)[None])[0]
            ps.append(_psnr(out, ref))
            f.unlink()
        ent["psnr_vs_fp32"] = round(float(np.mean(ps)), 2) if ps else None
        res.setdefault("phone", {})[f"{h}x{w}_{prec}"] = ent
        print(m, h, w, prec, ent)
    rp.write_text(json.dumps(res, indent=1))


def report() -> None:
    rows = [
        "| model | params | GT PSNR nuScenes fp32 / int8 (bicubic) | GT PSNR COCO fp32 / int8 (bicubic) |",
        "|---|---|---|---|",
    ]
    prow = [
        "| model | LR -> HR | fp16 HTP | int8 HTP | fp16 / int8 PSNR vs host fp32 |",
        "|---|---|---|---|---|",
    ]
    for rp in sorted((WORK / "results").glob("*.json")):
        m = rp.stem
        r = json.loads(rp.read_text())
        npar = sum(int(np.prod(i.dims)) for i in core(m, 8, 8).graph.initializer)
        if "host" in r:
            a, b = r["host"]["nuscenes"], r["host"]["coco"]
            rows.append(
                f"| {m} | {npar / 1e3:.0f}k | {a['fp32']:.2f} / {a['int8']:.2f} ({a['bicubic']:.2f}) | "
                f"{b['fp32']:.2f} / {b['int8']:.2f} ({b['bicubic']:.2f}) |"
            )
        for res in sorted({k.rsplit("_", 1)[0] for k in r.get("phone", {})}):
            f, q = r["phone"].get(f"{res}_fp16", {}), r["phone"].get(f"{res}_int8", {})
            h, w = (int(v) for v in res.split("x"))
            ms = lambda e: f"{e['median_ms']:.1f} ms" if e.get("median_ms") else "fails"  # noqa: E731
            prow.append(
                f"| {m} | {w}x{h} -> {w * 4}x{h * 4} | {ms(f)} | {ms(q)} | "
                f"{f.get('psnr_vs_fp32')} / {q.get('psnr_vs_fp32')} dB |"
            )
    print("\n".join(rows) + "\n\n" + "\n".join(prow))


if __name__ == "__main__":
    a = sys.argv[1:]
    {
        "fetch": lambda: fetch(),
        "build": lambda: build(a[1], int(a[2]), int(a[3])),
        "host": lambda: host(a[1]),
        "phone": lambda: phone(a[1], int(a[2]), int(a[3])),
        "report": lambda: report(),
    }[a[0]]()
