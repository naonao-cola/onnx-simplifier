"""Sparse4D v3 on the phone's HTP, frame after frame, with the phone's own outputs carried.

  PHONE_LOCK_OWNER=codex/android-sparse4d ~/.cache/android-phone/phone-run \\
    python phone_chain.py --work <work> [--suffix .sim] [--iters 10]

For each of the saved frames (validate.py --work): the host runs the instance bank on the
*phone's* previous outputs (cache the top-600 by decayed confidence, project their anchors by ego
motion, time interval). It feeds frame_first<suffix>.onnx (frame 0) or frame_temp<suffix>.onnx
through qnn_run_multi (strict all-HTP, EP-context cache), pulls cls/box/quality/feat, decodes and
GT-matches. Prints per-frame median ms, the output cosines vs the fp32 torch chain (same frame,
torch's own state), and the GT totals at score >= 0.3 / 0.2.
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from data import match
from model import InstanceBank, decode

HERE = Path(__file__).resolve().parent
Q = HERE.parents[1] / "htp_exploration" / "qnn_shell"
SERIAL = os.environ.get("DEVICE_SERIAL", "239dbd8f")
R = os.environ.get("R", "/data/local/tmp/codex-android-sparse4d")
TMP = Path(os.environ.get("SPARSE4D_TMP", Path.home() / ".cache" / "sparse4d" / "tmp"))
NDK_CXX = os.environ.get(
    "NDK_CXX", "/usr/lib/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++")


def adb(*a, check=True, capture=False):
    r = subprocess.run(["adb", "-s", SERIAL, *a], check=check, capture_output=capture, text=True)
    return r.stdout if capture else None


def push_if_changed(src, dst):
    sz = os.path.getsize(src)
    got = adb("shell", f"stat -c %s {dst} 2>/dev/null || true", capture=True).strip()
    if got != str(sz):
        adb("push", "-q", str(src), dst)


def setup(work, models):
    binp = HERE / ".qnn_run_multi"
    if not binp.exists() or binp.stat().st_mtime < (Q / "qnn_run_multi.cpp").stat().st_mtime:
        subprocess.run([NDK_CXX, "-O2", "-std=c++17", "-static-libstdc++", "-I", str(Q / "headers"), "-o", str(binp),
                        str(Q / "qnn_run_multi.cpp"), "-L", str(Q / "libs"), "-lonnxruntime"], check=True)
    adb("shell", f"mkdir -p {R}")
    push_if_changed(binp, f"{R}/qnn_run_multi")
    for f in sorted((Q / "libs").iterdir()):
        push_if_changed(f, f"{R}/{f.name}")
    for m in models:
        push_if_changed(work / m, f"{R}/{m}")


def run(model, inputs, iters, tag):
    """inputs: [(name, dtype, array)] -> (outputs [cls, box, quality, feat], median ms)."""
    lines = []
    for name, dt, a in inputs:
        fn = f"{tag}_{name}.bin"
        p = TMP / (f"{os.getpid()}_{fn}")
        np.ascontiguousarray(a).tofile(p)
        adb("push", "-q", str(p), f"{R}/{fn}")
        p.unlink()
        lines.append(f"{name} {dt} {fn} {','.join(map(str, a.shape))}")
    man = TMP / (f"{os.getpid()}_manifest.txt")
    man.write_text("\n".join(lines) + "\n")
    adb("push", "-q", str(man), f"{R}/{tag}_manifest.txt")
    man.unlink()
    ctx = model.replace(".onnx", ".ctx.onnx")
    out = adb("shell", f"cd {R} && ORT_SPIN=0 QNN_PERF=burst LD_LIBRARY_PATH={R} "
              f"ADSP_LIBRARY_PATH='{R};/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp' "
              f"./qnn_run_multi {model} {tag}_manifest.txt htp {iters} {tag}_out {ctx} 2>&1", capture=True)
    med = re.search(r"median_ms\s*[=:]?\s*([\d.]+)", out)
    if not med:
        sys.exit(f"phone run failed:\n{out[-3000:]}")
    shapes = {int(m.group(1)): (m.group(3), [int(x) for x in m.group(4).split(",") if x])
              for m in re.finditer(r"out (\d+) (\S+) (\S+) ([\d,]+)", out)}
    res = []
    for i in range(4):
        dt, shape = shapes[i]
        local = TMP / (f"{os.getpid()}_o{i}.bin")
        adb("pull", "-q", f"{R}/{tag}_out_o{i}.bin", str(local))
        res.append(np.fromfile(local, dtype={"f32": np.float32, "u8": np.uint8, "u16": np.uint16}.get(dt, np.float32))
                   .reshape(shape))
        local.unlink()
    return res, float(med.group(1))


def cos(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--suffix", default=".sim")
    ap.add_argument("--iters", type=int, default=10)
    a = ap.parse_args()
    work = Path(a.work)
    first, temp = f"frame_first{a.suffix}.onnx", f"frame_temp{a.suffix}.onnx"
    TMP.mkdir(parents=True, exist_ok=True)
    setup(work, [first, temp])
    frames = sorted((work / "frames").glob("*.pkl"), key=lambda p: int(p.stem))
    bank = InstanceBank()
    tot = {0.3: [0, 0, 0], 0.2: [0, 0, 0]}
    times = []
    for k, p in enumerate(frames):
        fr = pickle.load(open(p, "rb"))
        metas = fr["metas"]
        temp_feat, temp_anchor, dt = bank.get(metas)
        ins = [("rgb", "u8", fr["rgb"]), ("proj", "f32", metas["projection_mat"].numpy())]
        if temp_feat is not None:
            ins += [("dt", "f32", np.array([float(dt)], np.float32)),
                    ("temp_feat", "f32", temp_feat.numpy()), ("temp_anchor", "f32", temp_anchor.numpy())]
        (cls, box, q, feat), ms = run(temp if temp_feat is not None else first, ins, a.iters, f"f{k}")
        times.append(ms)
        cls_t, box_t, q_t, feat_t = (torch.from_numpy(x.astype(np.float32)) for x in (cls, box, q, feat))
        bank.cache(feat_t, box_t, cls_t, metas)
        boxes, scores, labels = decode(cls_t, box_t, q_t)
        res = {t: match(boxes, scores, labels, fr["gt"], thr=t) for t in tot}
        for t, r in res.items():
            tot[t] = [x + y for x, y in zip(tot[t], r)]
        ref = fr["ref"]
        print(f"frame {k}: {ms:.1f} ms  GT >=0.3 {res[0.3][0]}/{res[0.3][2]}  >=0.2 {res[0.2][0]}  "
              f"cos vs fp32 torch chain: cls {cos(cls, ref[0]):.5f} box {cos(box, ref[1]):.5f}", flush=True)
    print(f"median over frames: {float(np.median(times)):.1f} ms/frame ({1000 / float(np.median(times)):.1f} FPS)")
    for t, (tp, npred, ngt) in tot.items():
        print(f"score >= {t}: GT matched {tp}/{ngt}, predictions {npred}")


if __name__ == "__main__":
    main()
