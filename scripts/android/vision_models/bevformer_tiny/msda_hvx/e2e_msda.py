#!/usr/bin/env python3
"""BEVFormer-tiny on the phone with the encoder split around the HVX deformable-sampling kernel.

Per keyframe of a scene: backbone (int8, HTP) -> encoder (enc_run: 7 fp16 HTP pieces + 6 msda
calls on the CDSP, one process) -> decoder (fp16, HTP), each piece fed the previous one's phone
output, the phone's own BEV carried to the next frame as prev_bev (rotated on the host), exactly
like ../e2e_phone.py. fp32 torch runs alongside; reports per-frame cos and GT matches for both, and
enc_run's latency breakdown.

usage: e2e_msda.py --ckpt <pth> --data <nuscenes-mini> --work <work> [--scene scene-0103] [--frames 6]
       [--backbone backbone6.q8] [--decoder decoder.sim] [--reps 10]
Everything goes to a phone directory of its own (R, default /data/local/tmp/codex-android-bevformer-msda-hvx):
this script pushes the ORT/QNN libs (../../../htp_exploration/qnn_shell/libs), build.sh's enc_run,
qnn_run_multi and msda_rpc.so, the split pieces (split.py export) and the backbone/decoder pieces
(export.py / quantize.py). Every adb call goes through the host's phone lock when PHONE_RUN is set
(e.g. PHONE_RUN=~/.cache/android-phone/phone-run PHONE_LOCK_OWNER=<branch>).
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
import e2e_phone as E  # noqa: E402
import model as M  # noqa: E402
import split as S  # noqa: E402
from nuscenes import NuScenesMini, match, temporal_can_bus  # noqa: E402

PIECES = ["pre", "mid0", "post0", "mid1", "post1", "mid2", "post2"]
QNN_LIBS = HERE.parents[2] / "htp_exploration" / "qnn_shell" / "libs"
E.R = os.environ.get("R", "/data/local/tmp/codex-android-bevformer-msda-hvx")
if os.environ.get("PHONE_RUN"):
    E.ADB = [os.environ["PHONE_RUN"], *E.ADB]


def push_runtime(work: Path, build: Path, pieces):
    R = E.R
    subprocess.run([*E.ADB, "shell", f"mkdir -p {R}/msda_split"], check=True)
    for f in ("enc_run", "qnn_run_multi", "msda_rpc.so"):
        subprocess.run([*E.ADB, "push", "-q", str(build / f), f"{R}/{f}"], check=True)
    subprocess.run(
        [*E.ADB, "push", "-q", *[str(f) for f in sorted(QNN_LIBS.iterdir())], f"{R}/"],
        check=True,
    )
    for p in pieces:
        subprocess.run(
            [*E.ADB, "push", "-q", str(work / f"{p}.onnx"), f"{R}/{p}.onnx"], check=True
        )
    subprocess.run([*E.ADB, "shell", f"chmod 755 {R}/qnn_run_multi"], check=True)
    for p in PIECES:
        subprocess.run(
            [
                *E.ADB,
                "push",
                "-q",
                str(work / "msda_split" / f"{p}.sim.onnx"),
                f"{R}/msda_split/{p}.onnx",
            ],
            check=True,
        )
    subprocess.run([*E.ADB, "shell", f"chmod 755 {R}/enc_run"], check=True)


def run_encoder(work: Path, inputs: dict, reps: int):
    """inputs -> (bev (2500, 256), enc_run's report)."""
    R = E.R
    fd = work / "e2e_msda_frame"
    fd.mkdir(exist_ok=True)
    for k, v in inputs.items():
        v.tofile(fd / (f"{k}.u8" if v.dtype == np.uint8 else f"{k}.f32"))
    subprocess.run(
        [*E.ADB, "shell", f"rm -rf {R}/msda_frame && mkdir -p {R}/msda_frame"],
        check=True,
    )
    subprocess.run(
        [*E.ADB, "push", "-q", *[str(p) for p in fd.iterdir()], f"{R}/msda_frame/"],
        check=True,
    )
    out = subprocess.run(
        [
            *E.ADB,
            "shell",
            f"cd {R} && LD_LIBRARY_PATH={R} QNN_PERF=burst "
            f"ADSP_LIBRARY_PATH='{R};/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp' "
            f"./enc_run msda_split 2 {reps} msda_frame 2>&1",
        ],
        capture_output=True,
        text=True,
    ).stdout
    if "PASS" not in out:
        raise RuntimeError(out[-1500:])
    subprocess.run(
        [*E.ADB, "pull", "-q", f"{R}/msda_frame/bev.f32", str(fd / "bev.f32")],
        check=True,
    )
    return np.fromfile(fd / "bev.f32", np.float32).reshape(M.NQ, M.EMBED), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument(
        "--build",
        default=str(HERE / "build"),
        help="build.sh OUT (enc_run, msda_rpc.so)",
    )
    ap.add_argument("--scene", default="scene-0103")
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--backbone", default="backbone6.q8")
    ap.add_argument("--decoder", default="decoder.sim")
    ap.add_argument("--reps", type=int, default=10)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    work = Path(a.work)
    push_runtime(work, Path(a.build), [a.backbone, a.decoder])
    bb, enc, dec = M.load_official(a.ckpt)
    ns = NuScenesMini(a.data)
    prev = {"cpu": None, "htp": None}
    prev_abs = None
    tot = {"cpu": [0, 0, 0], "htp": [0, 0, 0]}
    enc_ms = []
    for i, tok in enumerate(ns.scene_samples(a.scene)[: a.frames]):
        f = ns.frame(tok)
        can_bus = temporal_can_bus(f["can_bus_abs"], prev_abs)
        ref_cam, bev_mask = M.reference_points_cam(f["lidar2img"])
        first = prev["cpu"] is None
        has_prev = torch.zeros(1) if first else torch.ones(1)
        shift = torch.zeros(2) if first else M.can_bus_shift(can_bus)
        feats = bb(f["img"])
        p_in = (
            torch.zeros(M.NQ, M.EMBED)
            if first
            else M.rotate_prev_bev(prev["cpu"], can_bus)
        )
        bev = enc(feats, p_in, has_prev, shift, can_bus, ref_cam, bev_mask)
        cls, bbox = dec(bev)
        (feats_h,), t_b = E.phone(work, a.backbone, {"img": f["img"].numpy()})
        p_h = (
            np.zeros((M.NQ, M.EMBED), np.float32)
            if first
            else M.rotate_prev_bev(torch.from_numpy(prev["htp"]), can_bus).numpy()
        )
        enc_in = (
            torch.from_numpy(feats_h),
            torch.from_numpy(p_h),
            has_prev,
            shift,
            can_bus,
            ref_cam,
            bev_mask,
        )
        tsa_ref, ref_cam_h, vis = S.host_inputs(enc_in)
        f32 = lambda t: np.ascontiguousarray(np.asarray(t, np.float32))  # noqa: E731
        bev_h, report = run_encoder(
            work,
            {
                "feats": f32(feats_h),
                "prev_bev": f32(p_h),
                "has_prev": f32(has_prev),
                "can_bus": f32(can_bus),
                "tsa_ref": f32(tsa_ref),
                "ref_cam": f32(ref_cam_h),
                "vis": np.ascontiguousarray(vis.numpy().astype(np.uint8)),
            },
            a.reps,
        )
        t_e = float(re.search(r"encoder_ms median ([0-9.]+)", report).group(1))
        enc_cos = E.cos(
            enc(*enc_in), bev_h
        )  # the encoder alone: fp32 torch on the phone's own inputs
        enc_ms.append(t_e)
        if i == 0:
            print(report[report.index("encoder_ms") - 12 :].strip())
        (cls_h, bbox_h), t_d = E.phone(work, a.decoder, {"bev_embed": bev_h})
        r = {}
        for k, (c, b) in {
            "cpu": (cls, bbox),
            "htp": (torch.from_numpy(cls_h), torch.from_numpy(bbox_h)),
        }.items():
            boxes, scores, labels = M.decode(c, b)
            r[k] = match(boxes, scores, labels, f["gt"])
            for j in range(3):
                tot[k][j] += r[k][j]
        print(
            f"frame {i}: ms backbone {t_b:.0f} encoder {t_e:.1f} decoder {t_d:.0f} | cos vs fp32: feats "
            f"{E.cos(feats, feats_h):.5f} bev {E.cos(bev, bev_h):.5f} (encoder alone {enc_cos:.6f}) cls {E.cos(cls, cls_h):.5f} "
            f"bbox {E.cos(bbox, bbox_h):.5f} | matched/pred/GT fp32 {r['cpu']} phone {r['htp']}",
            flush=True,
        )
        prev["cpu"], prev["htp"], prev_abs = bev, bev_h, f["can_bus_abs"]
    for k, (tp, npred, ngt) in tot.items():
        print(
            f"{k}: {tp} matched of {ngt} GT ({npred} detections >= 0.3) over {a.frames} frames"
        )
    print(f"encoder (enc_run) median over frames: {np.median(enc_ms):.1f} ms")


if __name__ == "__main__":
    main()
