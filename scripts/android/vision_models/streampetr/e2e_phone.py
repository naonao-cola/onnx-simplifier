"""StreamPETR end to end with both pieces on the phone's HTP, frame after frame, vs fp32 torch and GT.

  python e2e_phone.py --ckpt <pth> --work <work> [--img img.sim] [--head head.sim] [--scene scene-0103]

Per frame of validate.py's dumps (<work>/frames/<scene>/<i>.npz): the image piece runs on the HTP,
its output tokens feed HostState.pre (the memory queue carried from the *phone's* previous frames),
the head piece runs on the HTP, HostState.post decodes and propagates. Every phone call goes through
qnn_run_multi with a QNN context cache (<piece>.ctx.onnx) so a frame doesn't recompile the graph.

If <work>/<piece>.json exists (quantize.py's quantized-I/O qparams), those inputs are quantized on
the host (and the image transposed to NHWC when the entry says so) and those outputs dequantized, as
an app would. Wrap the whole run in ~/.cache/android-phone/phone-run.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import data as D
import model as M
import numpy as np
import torch

DEV = os.environ.get("DEVICE_SERIAL", "239dbd8f")
R = os.environ.get("R", "/data/local/tmp/codex-android-streampetr")
ADB = ["adb", "-s", DEV]
Q = Path(__file__).resolve().parents[2] / "htp_exploration" / "qnn_shell"
NP = {"f32": np.float32, "u8": np.uint8, "u16": np.uint16}


def sh(cmd):
    return subprocess.run([*ADB, "shell", cmd], capture_output=True, text=True).stdout


def setup(work: Path, pieces):
    sh(f"mkdir -p {R}")
    for f in [Q / "libs" / x for x in os.listdir(Q / "libs")] + [Q.parents[1] / "vision_models_probe" / ".qnn_run_multi"]:
        name = "qnn_run_multi" if f.name == ".qnn_run_multi" else f.name
        if sh(f"stat -c %s {R}/{name} 2>/dev/null").strip() != str(f.stat().st_size):
            subprocess.run([*ADB, "push", "-q", str(f), f"{R}/{name}"], check=True)
    for p in pieces:
        subprocess.run([*ADB, "push", "-q", str(work / f"{p}.onnx"), f"{R}/{p}.onnx"], check=True)
        sh(f"rm -f {R}/{p}.ctx.onnx")


def phone(work: Path, piece: str, inputs: dict, iters=1):
    tmp = work / "e2e_tmp"
    tmp.mkdir(exist_ok=True)
    qp = work / f"{piece}.json"
    qio = json.loads(qp.read_text()) if qp.exists() else {}
    lines = []
    for k, v in inputs.items():
        v, dt = np.ascontiguousarray(v, np.float32), "f32"
        if k in qio:
            q = qio[k]
            if q.get("layout") == "nhwc":
                v = np.ascontiguousarray(v.transpose(0, 2, 3, 1))
            v = np.clip(np.round(v / q["scale"]) + q["zero_point"], 0, np.iinfo(q["dtype"]).max).astype(q["dtype"])
            dt = {"uint8": "u8", "uint16": "u16"}[q["dtype"]]
        v.tofile(tmp / f"{k}.bin")
        subprocess.run([*ADB, "push", "-q", str(tmp / f"{k}.bin"), f"{R}/{k}.bin"], check=True)
        lines.append(f"{k} {dt} {k}.bin {','.join(map(str, v.shape))}")
    (tmp / "m.txt").write_text("\n".join(lines) + "\n")
    subprocess.run([*ADB, "push", "-q", str(tmp / "m.txt"), f"{R}/e2e_{piece}.txt"], check=True)
    out = sh(f"cd {R} && ORT_SPIN=0 LD_LIBRARY_PATH={R} QNN_PERF=burst "
             f"ADSP_LIBRARY_PATH='{R};/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp' "
             f"./qnn_run_multi {piece}.onnx e2e_{piece}.txt htp {iters} e2e_{piece} {piece}.ctx.onnx 2>&1")
    if "PASS" not in out:
        raise RuntimeError(f"{piece}: {out[-600:]}")
    res = []
    for i, name, dt, shape in re.findall(r"^out (\d+) (\S+) (\S+) (\S+)", out, re.M):
        subprocess.run([*ADB, "pull", "-q", f"{R}/e2e_{piece}_o{i}.bin", str(tmp / f"o{i}.bin")], check=True)
        y = np.fromfile(tmp / f"o{i}.bin", NP[dt]).reshape([int(d) for d in re.findall(r"\d+", shape)])
        if name in qio:
            y = (y.astype(np.float32) - qio[name]["zero_point"]) * np.float32(qio[name]["scale"])
        res.append(y.astype(np.float32))
    med = re.search(r"median_ms ([0-9.]+)", out)
    return res, float(med.group(1)) if med else float("nan")


def cos(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--img", default="img.sim")
    ap.add_argument("--head", default="head.sim")
    ap.add_argument("--scene", default="scene-0103")
    ap.add_argument("--thr", type=float, nargs="+", default=[0.3, 0.2])
    ap.add_argument("--iters", type=int, default=1, help="runs per phone call (>= 3 prints medians)")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    work = Path(a.work)
    _, head = M.load_official(a.ckpt)
    core = M.HeadCore(head).eval()
    setup(work, [a.img, a.head])
    host, ref_host = M.HostState(head), M.HostState(head)
    frames = sorted((work / "frames" / a.scene).glob("*.npz"), key=lambda p: int(p.stem))
    tot = {t: [0, 0, 0] for t in a.thr}
    ref_tot = {t: [0, 0, 0] for t in a.thr}
    ms_all = []
    for i, fp in enumerate(frames):
        z = np.load(fp)
        t = M.to_torch({k: z[k] for k in ("lidar2img", "intrinsics", "ego_pose", "ego_pose_inv")} | {"timestamp": float(z["timestamp"])})
        prev = bool(z["prev"])
        gt = list(zip(z["gt_names"].tolist(), z["gt_xyz"]))
        img = z["img_u8"].transpose(0, 3, 1, 2).astype(np.float32) if "raw" in a.img else D.normalize(z["img_u8"]).numpy()
        (feat,), ms_img = phone(work, a.img, {"img": img}, a.iters)
        feat_t = torch.from_numpy(feat.reshape(-1, M.EMBED))
        ins = {"feat": feat_t, **host.rig_inputs(t), **host.pre(t, prev)}
        (cls, reg, dec), ms_head = phone(work, a.head, {k: v.numpy() for k, v in ins.items()}, a.iters)
        cls, box = host.post(t, torch.from_numpy(cls), torch.from_numpy(reg), torch.from_numpy(dec))
        # fp32 torch chain on the torch features (validate.py's dump) for the same frame
        r_ins = {"feat": torch.from_numpy(z["feat"]), **ref_host.rig_inputs(t), **ref_host.pre(t, prev)}
        r_cls, r_reg, r_dec = core(*[r_ins[k] for k in export_order()])
        r_cls, r_box = ref_host.post(t, r_cls, r_reg, r_dec)
        line = (f"frame {i}: img {ms_img:.1f} ms head {ms_head:.1f} ms | cos feat {cos(feat.reshape(-1, M.EMBED), z['feat']):.5f} "
                f"cls {cos(cls, r_cls):.5f} box {cos(box, r_box):.5f}")
        for thr in a.thr:
            b, s, lab = M.decode(cls, box)
            r = D.match(b, s, lab, gt, thr=thr)
            tot[thr] = [x + y for x, y in zip(tot[thr], r)]
            rb, rs, rl = M.decode(r_cls, r_box)
            rr = D.match(rb, rs, rl, gt, thr=thr)
            ref_tot[thr] = [x + y for x, y in zip(ref_tot[thr], rr)]
            line += f" | @{thr} phone {r[0]}/{r[2]} torch {rr[0]}/{rr[2]}"
        ms_all += [ms_img, ms_head]
        print(line, flush=True)
    ms = [x for x in ms_all if x == x]
    if ms:
        print(f"median over frames: img {np.median([x for x in ms_all[0::2] if x == x]):.2f} ms, "
              f"head {np.median([x for x in ms_all[1::2] if x == x]):.2f} ms")
    for thr in a.thr:
        print(f"total @{thr}: phone {tot[thr][0]}/{tot[thr][2]} (pred {tot[thr][1]}), "
              f"fp32 torch {ref_tot[thr][0]}/{ref_tot[thr][2]} (pred {ref_tot[thr][1]})")


def export_order():
    return ["feat", "pe", "sa_gamma", "sa_beta", "mem_emb", "mem_pe3d", "mem_time", "mem_motion"]


if __name__ == "__main__":
    main()
