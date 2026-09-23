#!/usr/bin/env python3
"""Fast-BEV M0 / Fast-BEV++ end to end on the phone (runtime/fastbev_run.cpp), frame after frame on
the frames validate.py saved (scene-0103 frames 0-5), vs GT and vs the fp32 torch detections.

usage: e2e_phone.py m0|pp --work <dir> [--iters 10] [--gather dsp|htp] [--env K=V,...]
Stages a directory (int8 models, io.txt, per-frame uint8 images; M0: per-slot projection matrices +
slot ages; PP: per-frame LUTs), pushes it with runtime/build.sh's binaries under the phone lock,
runs, pulls each frame's detections and scores them (../bevformer_tiny's criteria: score >= 0.3,
same class, center within 2 m). Session/EP-context caches stay on the phone between runs.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import data as D
import decode as DEC
import geometry as G
import numpy as np
import torch

R = "/data/local/tmp/codex-android-fastbev"
LOCK = Path.home() / ".cache/android-phone/phone-run"
SERIAL = "239dbd8f"


def stage(a, work, frames, out):
    out.mkdir(parents=True, exist_ok=True)
    io = json.loads((work / f"{a.fam}_io.json").read_text())
    lines = []
    for stem, q in io.items():
        for k, v in q.items():
            if k != "img":
                lines.append(f"{k} {v['scale']!r} {v['zero_point']}")
    (out / "io.txt").write_text("\n".join(dict.fromkeys(lines)) + "\n")
    models = ["m0_enc.q8", "m0_bev.q8", "m0_viewbev.q8.c4"] if a.fam == "m0" else ["pp_enc.q8", "pp_viewbev.q8"]
    for m in models:
        (out / f"{m}.onnx").write_bytes((work / f"{m}.onnx").read_bytes())
    for i, f in enumerate(frames):
        if a.fam == "m0":
            f["img_u8"][0].tofile(out / f"f{i}_img.bin")
            np.stack([G.m0_projection(f["lidar2img"][t]).numpy() for t in range(4)]).astype(np.float32).tofile(
                out / f"f{i}_proj.bin")
            ages = [0] + [min(k, i) for k in (1, 2, 3)]  # frame i of the scene has i previous keyframes
            (out / f"f{i}_ages.txt").write_text(" ".join(map(str, ages)) + " 1\n")
        else:
            f["img_u8"].tofile(out / f"f{i}_img.bin")
            f["idx"].tofile(out / f"f{i}_idx.bin")
            f["didx"].tofile(out / f"f{i}_didx.bin")


def adb(*args, check=True):
    return subprocess.run(["adb", "-s", SERIAL, *args], check=check, capture_output=True, text=True).stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fam", choices=["m0", "pp"])
    ap.add_argument("--work", required=True)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--gather", default="dsp", choices=["dsp", "htp"])
    ap.add_argument("--env", default="")
    a = ap.parse_args()
    work = Path(a.work)
    frames = [torch.load(p, weights_only=False)
              for p in sorted((work / f"{a.fam}_frames").glob("*.pt"), key=lambda p: int(p.stem))]
    out = work / f"phone_{a.fam}"
    stage(a, work, frames, out)
    build = Path.home() / ".cache/fastbev/runtime_build"
    qlibs = Path(__file__).resolve().parents[2] / "htp_exploration/qnn_shell/libs"
    env = " ".join(["GATHER=" + a.gather, "LD_LIBRARY_PATH=" + R,
                    f"ADSP_LIBRARY_PATH='{R};/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp'"]
                   + [kv for kv in a.env.split(",") if kv])
    script = f"""set -e
adb -s {SERIAL} shell mkdir -p {R}/{a.fam}
for f in {build}/fastbev_run {build}/fbgather_rpc.so {qlibs}/*; do
  d=$(basename $f); sz=$(stat -c %s $f); dsz=$(adb -s {SERIAL} shell "stat -c %s {R}/$d 2>/dev/null" | tr -d '\\r' || true)
  [ "$sz" = "$dsz" ] || adb -s {SERIAL} push -q $f {R}/$d
done
adb -s {SERIAL} push -q {out}/. {R}/{a.fam}/
adb -s {SERIAL} shell "chmod 755 {R}/fastbev_run && cd {R} && {env} ./fastbev_run {a.fam} {R}/{a.fam} {len(frames)} {a.iters}"
for i in $(seq 0 {len(frames) - 1}); do adb -s {SERIAL} pull -q {R}/{a.fam}/f${{i}}_dets.bin {out}/; done
"""
    r = subprocess.run([str(LOCK), "bash", "-c", script], capture_output=True, text=True,
                       env={**__import__("os").environ, "PHONE_LOCK_OWNER": "codex/android-fastbev"})
    log = r.stdout + r.stderr
    print("\n".join(line for line in log.splitlines() if re.match(r"(stage|total|sessions|FAIL)", line)))
    if r.returncode:
        print(log[-2000:])
        raise SystemExit(1)
    names = DEC.M0_CLASSES if a.fam == "m0" else DEC.PP_CLASSES
    tot, tot_ref = np.zeros(3, int), np.zeros(3, int)
    for i, f in enumerate(frames):
        d = np.fromfile(out / f"f{i}_dets.bin", np.float32).reshape(-1, 11)
        m = D.match(d[:, :2], d[:, 9], [names[int(x)] for x in d[:, 10]], f["gt"])
        if a.fam == "m0":
            rb, rs, rn = DEC.m0_decode(*f["head"])
        else:
            rb, rs, rn = DEC.pp_decode(f["head"])
        mr = D.match(rb[:, :2], rs, rn, f["gt"])
        tot += m
        tot_ref += mr
        print(f"frame {i}: phone {m[0]} of {m[2]} GT ({m[1]} dets)   fp32 torch {mr[0]} ({mr[1]} dets)")
    print(f"{a.fam} phone int8 ({a.gather} gather): {tot[0]} matched of {tot[2]} GT ({tot[1]} dets >= 0.3); "
          f"fp32 torch {tot_ref[0]}")


if __name__ == "__main__":
    main()
