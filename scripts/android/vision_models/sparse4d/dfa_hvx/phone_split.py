"""Sparse4D v3 on the phone, split around the HVX DFA, frame after frame (the phone's own outputs
carried through the host instance bank).

  PHONE_LOCK_OWNER=codex/android-sparse4d ~/.cache/android-phone/phone-run \\
    python phone_split.py --work <work> --build <build dir> [--reps 8]

Per frame: host instance bank -> frame inputs -> s4d_run (bb, pre0, 6 x [dfa, mid/post]) ->
cls / box / quality / feat back -> bank.cache, decode, GT match. Prints s4d_run's per-step medians
for each frame and the GT totals.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import phone_chain as pc  # noqa: E402
from data import match  # noqa: E402
from export import normalized_proj  # noqa: E402
from model import InstanceBank, decode  # noqa: E402
from split import load_frames  # noqa: E402

R = os.environ.get("R", "/data/local/tmp/codex-android-sparse4d")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--build", required=True)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--flags", default="4")
    ap.add_argument("--models", default="", help="variant of the pre0 / mid pieces: w16 (io16.py), w2 (split.py --w-layout v2)")
    ap.add_argument("--scene", action="store_true", help="s4d_scene: the whole scene in one process, bank on the phone")
    ap.add_argument("--pipeline", type=int, default=0,
                    help="with --scene: 1 = next frame's bb + pre0 on a free thread, 2 = during DFA calls 0 / 1")
    a = ap.parse_args()
    work, build = Path(a.work), Path(a.build)
    sp = work / "split"
    pc.TMP.mkdir(parents=True, exist_ok=True)
    pc.adb("shell", f"mkdir -p {R}/split")
    for f in sorted((pc.Q / "libs").iterdir()):
        pc.push_if_changed(f, f"{R}/{f.name}")
    for f in ("s4d_run", "s4d_scene", "dfa_rpc.so"):
        pc.adb("push", "-q", str(build / f), f"{R}/{f}")
    wp = ["pre0"] + [f"mid{k}{t}" for k in range(5) for t in "FT"]
    files = ["bb.q8.onnx"] + [f"{n}.{a.models or 'sim'}.onnx" for n in wp] + ["postF.sim.onnx", "postT.sim.onnx"]
    for fn in files:
        pc.push_if_changed(sp / fn, f"{R}/split/{fn}")
    for n in ("instance_feature.f32", "anchor.f32"):
        pc.push_if_changed(sp / n, f"{R}/split/{n}")
    if a.scene:
        return scene(a, work)
    bank = InstanceBank()
    tot = {0.3: [0, 0, 0], 0.2: [0, 0, 0]}
    times = []
    for k, fr in enumerate(load_frames(work)):
        metas = fr["metas"]
        temp_feat, temp_anchor, dt = bank.get(metas)
        ins = {"rgb.u8": fr["rgb"], "proj.f32": metas["projection_mat"].numpy(),
               "proj_n.f32": normalized_proj(metas["projection_mat"]).numpy()}
        if temp_feat is not None:
            ins.update({"dt.f32": np.array(float(dt), np.float32), "temp_feat.f32": temp_feat.numpy(),
                        "temp_anchor.f32": temp_anchor.numpy()})
        d = f"{R}/f{k}"
        pc.adb("shell", f"rm -rf {d} && mkdir -p {d}")
        for n, v in ins.items():
            p = pc.TMP / n
            np.ascontiguousarray(v, dtype=v.dtype).tofile(p)
            pc.adb("push", "-q", str(p), f"{d}/{n}")
            p.unlink()
        out = pc.adb("shell", f"cd {R} && ORT_SPIN=0 QNN_PERF=burst DFA_FLAGS={a.flags} S4D_MODELS={a.models} "
                     f"LD_LIBRARY_PATH={R} "
                     f"ADSP_LIBRARY_PATH='{R};/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp' "
                     f"./s4d_run split {d} {a.warmup} {a.reps} 2>&1", capture=True)
        if "PASS" not in out:
            raise SystemExit(f"frame {k}: s4d_run failed\n{out[-3000:]}")
        res = {}
        for n, shape in (("cls", (900, 10)), ("box", (900, 11)), ("quality", (900, 2)), ("feat", (900, 256))):
            p = pc.TMP / f"{n}.f32"
            subprocess.run(["adb", "-s", pc.SERIAL, "pull", "-q", f"{d}/{n}.f32", str(p)], check=True)
            res[n] = torch.from_numpy(np.fromfile(p, np.float32).reshape(shape))
            p.unlink()
        bank.cache(res["feat"], res["box"], res["cls"], metas)
        boxes, scores, labels = decode(res["cls"], res["box"], res["quality"])
        r = {t: match(boxes, scores, labels, fr["gt"], thr=t) for t in tot}
        for t in tot:
            tot[t] = [x + y for x, y in zip(tot[t], r[t])]
        ms = float(re.search(r"total_ms median ([\d.]+)", out).group(1))
        times.append(ms)
        print(f"frame {k}: {ms:.1f} ms  GT >=0.3 {r[0.3][0]}/{r[0.3][2]}  >=0.2 {r[0.2][0]}", flush=True)
        for line in out.splitlines():
            if line.startswith(("sessions", "  step", "  htp")):
                print("   ", line.strip())
    med = float(np.median(times))
    print(f"median over frames: {med:.1f} ms/frame ({1000 / med:.1f} FPS)")
    for t, (tp, npred, ngt) in tot.items():
        print(f"score >= {t}: GT matched {tp}/{ngt}, predictions {npred}")


def scene(a, work):
    """Push the scene's frame inputs once, run s4d_scene (instance bank on the phone), then decode and
    GT-match its per-frame outputs on the host."""
    frames = load_frames(work)
    for k, fr in enumerate(frames):
        metas = fr["metas"]
        ins = {"rgb.u8": fr["rgb"], "proj.f32": metas["projection_mat"].numpy(),
               "proj_n.f32": normalized_proj(metas["projection_mat"]).numpy(),
               "meta.f64": np.concatenate([[metas["timestamp"]], np.asarray(metas["T_global"]).ravel(),
                                           np.asarray(metas["T_global_inv"]).ravel()]).astype(np.float64)}
        d = f"{R}/scene/f{k}"
        pc.adb("shell", f"mkdir -p {d}")
        for n, v in ins.items():
            p = pc.TMP / n
            np.ascontiguousarray(v).tofile(p)
            pc.push_if_changed(p, f"{d}/{n}")
            p.unlink()
    out = pc.adb("shell", f"cd {R} && ORT_SPIN=0 QNN_PERF=burst DFA_FLAGS={a.flags} S4D_MODELS={a.models} "
                 f"PIPELINE={a.pipeline} LD_LIBRARY_PATH={R} "
                 f"ADSP_LIBRARY_PATH='{R};/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp' "
                 f"./s4d_scene split {R}/scene {len(frames)} {a.warmup + a.reps} 2>&1", capture=True)
    if "PASS" not in out:
        raise SystemExit(f"s4d_scene failed\n{out[-3000:]}")
    print("\n".join(line for line in out.splitlines() if line.startswith(("sessions", "frame_ms", "dfa", "frame "))))
    tot = {0.3: [0, 0, 0], 0.2: [0, 0, 0]}
    for k, fr in enumerate(frames):
        res = {}
        for n, shape in (("cls", (900, 10)), ("box", (900, 11)), ("quality", (900, 2))):
            p = pc.TMP / f"{n}.f32"
            subprocess.run(["adb", "-s", pc.SERIAL, "pull", "-q", f"{R}/scene/f{k}/{n}.f32", str(p)], check=True)
            res[n] = torch.from_numpy(np.fromfile(p, np.float32).reshape(shape))
            p.unlink()
        boxes, scores, labels = decode(res["cls"], res["box"], res["quality"])
        r = {t: match(boxes, scores, labels, fr["gt"], thr=t) for t in tot}
        for t in tot:
            tot[t] = [x + y for x, y in zip(tot[t], r[t])]
        print(f"frame {k}: GT >=0.3 {r[0.3][0]}/{r[0.3][2]}  >=0.2 {r[0.2][0]}", flush=True)
    for t, (tp, npred, ngt) in tot.items():
        print(f"score >= {t}: GT matched {tp}/{ngt}, predictions {npred}")


if __name__ == "__main__":
    main()
