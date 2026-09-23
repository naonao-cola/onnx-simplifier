#!/usr/bin/env python3
"""Run the split RT-DETR (split.py pieces + the HVX MSDA kernel) on the phone with dec_run over the
dumped eval images, and match its detections against fp32 HF.

usage: phone_split.py <pre model: pre.sim.onnx | pre.front8.onnx> [--n N] [--warmup W] [--reps R]
env:   BUILD (dir with dec_run, msda_rpc.so; default ~/.cache/onnxsim-rtdetr/msda_build), MSDA_FLAGS

Needs split.py export/quant/dump first. Every phone step runs under the shared host lock.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402

SERIAL = os.environ.get("DEVICE_SERIAL", "239dbd8f")
R = "/data/local/tmp/codex-android-rtdetr"
QLIBS = HERE.parents[2] / "htp_exploration/qnn_shell/libs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pre")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--work", default=str(Path.home() / ".cache/onnxsim-rtdetr/work"))
    a = ap.parse_args()
    work = Path(a.work)
    sd = work / "split"
    build = Path(
        os.environ.get("BUILD", Path.home() / ".cache/onnxsim-rtdetr/msda_build")
    )
    u8 = "front8" in a.pre or "u8" in a.pre
    rd = f"{R}/split"
    imgs = [f"img{i}" for i in range(a.n)]
    lines = ["set -e", f"adb -s {SERIAL} shell mkdir -p {rd}"]
    for f in sorted(QLIBS.glob("*")):
        lines.append(
            f"[ \"$(adb -s {SERIAL} shell stat -c %s {R}/{f.name} 2>/dev/null | tr -d '\\r')\" = {f.stat().st_size} ] "
            f"|| adb -s {SERIAL} push -q {f} {R}/{f.name}"
        )
    lines.append(
        f"adb -s {SERIAL} push -q {build}/dec_run {build}/msda_rpc.so {sd}/{a.pre} {sd}/mid0.sim.onnx "
        f"{sd}/mid1.sim.onnx {sd}/post.sim.onnx {rd}/"
    )
    for im in imgs:
        lines.append(
            f"adb -s {SERIAL} shell mkdir -p {rd}/{im} && adb -s {SERIAL} push -q {sd}/{im}/{'image.u8' if u8 else 'pixels.f32'} {rd}/{im}/"
        )
    flags = os.environ.get("MSDA_FLAGS", "4")
    lines.append(
        f'adb -s {SERIAL} shell "cd {rd} && chmod 755 dec_run && MSDA_FLAGS={flags} QNN_PERF=burst '
        f"LD_LIBRARY_PATH={R}:/vendor/lib64 ADSP_LIBRARY_PATH='{rd};{R};/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp' "
        f'./dec_run . {a.pre} {a.warmup} {a.reps} {" ".join(imgs)} > log.txt 2>&1" || true'
    )
    lines.append(f"adb -s {SERIAL} pull -q {rd}/log.txt {sd}/log.txt")
    for im in imgs:
        lines.append(
            f"adb -s {SERIAL} pull -q {rd}/{im}/logits.f32 {rd}/{im}/boxes.f32 {sd}/{im}/ || true"
        )
    script = sd / "run_split.sh"
    script.write_text("\n".join(lines) + "\n")
    env = dict(os.environ, PHONE_LOCK_OWNER="codex/android-rtdetr")
    subprocess.run(
        [str(Path.home() / ".cache/android-phone/phone-run"), "bash", str(script)],
        check=True,
        env=env,
    )
    log = (sd / "log.txt").read_text()
    if "PASS" not in log:
        print(log[-3000:])
        return 1
    tot = {"ref": 0, "det": 0, "matched": 0}
    totals = [float(x) for x in re.findall(r"total_ms median ([\d.]+)", log)]
    for im in imgs:
        ref = np.load(sd / im / "ref.npz")
        lg = np.fromfile(sd / im / "logits.f32", np.float32).reshape(1, 300, 80)
        bx = np.fromfile(sd / im / "boxes.f32", np.float32).reshape(1, 300, 4)
        r = C.match((ref["logits"], ref["boxes"]), (lg, bx))
        for k in tot:
            tot[k] += r[k]
    print(
        "\n".join(
            line
            for line in log.splitlines()
            if line.startswith("sessions") or line.startswith("  ")
        ).split("\n  step pre")[0]
    )
    print("image 0 breakdown:")
    print("\n".join(line for line in log.split(" total_ms")[1].splitlines()[1:10]))
    print(
        f"{a.pre}: total median over images {np.median(totals):.2f} ms, matched {tot['matched']}/{tot['ref']} (det {tot['det']})"
    )


if __name__ == "__main__":
    sys.exit(main())
