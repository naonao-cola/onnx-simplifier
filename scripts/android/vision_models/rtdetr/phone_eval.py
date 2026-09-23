#!/usr/bin/env python3
"""Run an RT-DETR model on the phone's HTP over the eval images and match its detections against
the fp32 torch model.

usage: phone_eval.py <model.onnx> [--input f32|u8] [--set eval|calibration] [--iters N]

Inputs are written per image (f32 NCHW pixels or uint8 NHWC RGB), pushed once, and run through
../../htp_exploration/qnn_shell/qnn_run_multi in strict all-HTP mode with an EP-context cache (the
first image compiles, the rest load it). Every phone step is wrapped in the shared host lock
(~/.cache/android-phone/phone-run). The fp32 torch reference is cached under <work>/ref/.
Prints matched/ref detections (score >= 0.3, IoU >= 0.5, same class) and the median latency.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import common as C
import numpy as np

HERE = Path(__file__).resolve().parent
Q = HERE.parent.parent / "htp_exploration/qnn_shell"
SERIAL = os.environ.get("DEVICE_SERIAL", "239dbd8f")
R = os.environ.get("R", "/data/local/tmp/codex-android-rtdetr")
LOCK = [str(Path.home() / ".cache/android-phone/phone-run")]
ENV = dict(os.environ, PHONE_LOCK_OWNER="codex/android-rtdetr")
NDK_CXX = "/usr/lib/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++"


def sh(*cmd, check=True):
    return subprocess.run(
        list(cmd), check=check, env=ENV, capture_output=True, text=True
    ).stdout


def adb(*a):
    return ["adb", "-s", SERIAL, *a]


def refs(work: Path, paths):
    import model as M
    import torch

    d = work / "ref"
    d.mkdir(parents=True, exist_ok=True)
    todo = [p for p in paths if not (d / f"{p.stem}.npz").exists()]
    if todo:
        torch.set_grad_enabled(False)
        m = M.load(patched=False)
        for p in todo:
            o = m(pixel_values=torch.from_numpy(C.to_pixels(C.load_rgb_u8(p))))
            np.savez(
                d / f"{p.stem}.npz", logits=o.logits.numpy(), boxes=o.pred_boxes.numpy()
            )
    return [np.load(d / f"{p.stem}.npz") for p in paths]


def harness():
    b = HERE / ".qnn_run_multi"
    src = Q / "qnn_run_multi.cpp"
    if not b.exists() or b.stat().st_mtime < src.stat().st_mtime:
        sh(
            NDK_CXX,
            "-O2",
            "-std=c++17",
            "-static-libstdc++",
            "-I",
            str(Q / "headers"),
            "-o",
            str(b),
            str(src),
            "-L",
            str(Q / "libs"),
            "-lonnxruntime",
        )
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--input", default="f32", choices=["f32", "u8"])
    ap.add_argument("--set", default="eval")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--work", default=str(Path.home() / ".cache/onnxsim-rtdetr/work"))
    ap.add_argument("--extra", default="", help="QNN_EXTRA for the harness (k=v,k=v)")
    ap.add_argument("--n", type=int, default=0, help="only the first N images")
    ap.add_argument(
        "--profile",
        action="store_true",
        help="QNN detailed profile of image 0 -> <stage>/pulled/prof.csv",
    )
    a = ap.parse_args()
    work, model = Path(a.work), Path(a.model)
    paths = C.image_paths(a.set)
    if a.n:
        paths = paths[: a.n]
    if a.profile:
        a.extra = ",".join(
            x
            for x in [
                a.extra,
                "profiling_level=detailed",
                f"profiling_file_path={R}/{Path(a.model).stem}/res/prof.csv",
            ]
            if x
        )
    ref = refs(work, paths)
    tag = model.stem
    stage = work / f"phone_{tag}"
    stage.mkdir(exist_ok=True)
    import onnx

    in_name = onnx.load(str(model), load_external_data=False).graph.input[0].name
    for i, p in enumerate(paths):
        rgb = C.load_rgb_u8(p)
        if a.input == "u8":
            x, dt, dims = rgb[None], "u8", "1,640,640,3"
        else:
            x, dt, dims = C.to_pixels(rgb), "f32", "1,3,640,640"
        x.tofile(stage / f"in{i}.bin")
        (stage / f"m{i}.txt").write_text(f"{in_name} {dt} in{i}.bin {dims}\n")
    b = harness()
    rd = f"{R}/{tag}"
    (stage / "phone.sh").write_text(
        f"cd {rd} && rm -rf ctx.onnx res && mkdir res\n"
        f"export QNN_PERF=burst LD_LIBRARY_PATH={R} "
        f"ADSP_LIBRARY_PATH='{R};/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp'\n"
        + (f"export QNN_EXTRA='{a.extra}'\n" if a.extra else "")
        + f"i=0; while [ $i -lt {len(paths)} ]; do\n"
        f"  n=1; [ $i = 0 ] && n={a.iters}\n"
        f"  {R}/qnn_run_multi {model.name} m$i.txt htp $n res/out$i ctx.onnx > res/log$i.txt 2>&1\n"
        f"  i=$((i+1)); done\n"
    )
    script = stage / "run.sh"
    pushes = [f"adb -s {SERIAL} shell mkdir -p {rd}"]
    for f in [b, *sorted((Q / "libs").glob("*"))]:
        name = "qnn_run_multi" if f.name == ".qnn_run_multi" else f.name
        pushes.append(
            f"[ \"$(adb -s {SERIAL} shell stat -c %s {R}/{name} 2>/dev/null | tr -d '\\r')\" = {f.stat().st_size} ] "
            f"|| adb -s {SERIAL} push -q {f} {R}/{name}"
        )
    ins = " ".join(
        str(stage / f)
        for f in ["phone.sh"]
        + [f"in{i}.bin" for i in range(len(paths))]
        + [f"m{i}.txt" for i in range(len(paths))]
    )
    pushes.append(f"adb -s {SERIAL} push -q {model} {ins} {rd}/")
    pushes.append(f"adb -s {SERIAL} shell sh {rd}/phone.sh")
    pushes.append(
        f"rm -rf {stage}/pulled && adb -s {SERIAL} pull -q {rd}/res {stage}/pulled"
    )
    script.write_text("set -e\n" + "\n".join(pushes) + "\n")
    subprocess.run(LOCK + ["bash", str(script)], check=True, env=ENV)
    pulled = stage / "pulled"
    log0 = (pulled / "log0.txt").read_text()
    med = re.search(r"median_ms ([\d.]+)", log0)
    tot = {"ref": 0, "det": 0, "matched": 0}
    fails = 0
    for i, r in enumerate(ref):
        log = (pulled / f"log{i}.txt").read_text()
        if "PASS" not in log:
            fails += 1
            continue
        outs = {}
        for line in log.splitlines():
            if line.startswith("out "):
                _, k, name, dt, shp = line.split()
                arr = np.fromfile(
                    pulled / f"out{i}_o{k}.bin",
                    {"f32": np.float32, "u8": np.uint8, "u16": np.uint16}[dt],
                )
                outs[name] = arr.reshape(
                    [int(s) for s in shp.strip(",").split(",")]
                ).astype(np.float32)
        m = C.match((r["logits"], r["boxes"]), (outs["logits"], outs["boxes"]))
        for k in tot:
            tot[k] += m[k]
    print(
        f"{model.name}: median {med.group(1) if med else '?'} ms, matched {tot['matched']}/{tot['ref']} "
        f"(det {tot['det']}) over {len(paths) - fails} images"
        + (f", {fails} FAILED" if fails else "")
    )
    if fails:
        print(log0[-2000:])


if __name__ == "__main__":
    sys.exit(main())
