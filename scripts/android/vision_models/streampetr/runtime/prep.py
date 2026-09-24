"""Everything petr_run needs for a scene, written once on the host:

  python runtime/prep.py --ckpt <pth> --work <work> --out <dir> [--scene scene-0103]

<dir>/consts.bin   reference_points (300, 3) + pseudo_reference_points (128, 3) + coords_d (64), f32
<dir>/f<i>/img.u8  (6, 256, 704, 3) raw RGB bytes, the int8 image piece's input as is
<dir>/f<i>/meta.bin ego_pose (16 f32), ego_pose_inv (16 f32), timestamp (f64), prev (i32),
                   lidar2img (6 x 16 f32), intrinsics (6 x 16 f32)
<dir>/f<i>/ref_pe_in.bin ref_cone.bin  model.position_embedding's (4224, 192) / (4224, 8), for check_run.py
lidar2img changes every frame on nuScenes (camera / lidar ego-motion compensation), so petr_run
computes the position embedding's geometry per frame and head_pe.sim.onnx its MLPs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import model as M  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scene", default="scene-0103")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    _, head = M.load_official(a.ckpt)
    np.concatenate([head.reference_points.weight.numpy().ravel(),
                    head.pseudo_reference_points.weight.numpy().ravel(),
                    head.coords_d.numpy().ravel()]).astype(np.float32).tofile(out / "consts.bin")
    frames = sorted((Path(a.work) / "frames" / a.scene).glob("*.npz"), key=lambda p: int(p.stem))
    for i, fp in enumerate(frames):
        z = np.load(fp)
        t = M.to_torch({k: z[k] for k in ("lidar2img", "intrinsics", "ego_pose", "ego_pose_inv")} | {"timestamp": float(z["timestamp"])})
        pe_in, cone = M.position_embedding(head, t["lidar2img"], t["intrinsics"])
        d = out / f"f{i}"
        d.mkdir(exist_ok=True)
        np.ascontiguousarray(z["img_u8"]).tofile(d / "img.u8")
        with open(d / "meta.bin", "wb") as f:
            f.write(z["ego_pose"].astype(np.float32).tobytes() + z["ego_pose_inv"].astype(np.float32).tobytes())
            f.write(np.float64(z["timestamp"]).tobytes() + np.int32(bool(z["prev"])).tobytes())
            f.write(z["lidar2img"].astype(np.float32).tobytes() + z["intrinsics"].astype(np.float32).tobytes())
        pe_in.numpy().astype(np.float32).tofile(d / "ref_pe_in.bin")
        cone.numpy().astype(np.float32).tofile(d / "ref_cone.bin")
    print(f"{out}: {len(frames)} frames")


if __name__ == "__main__":
    main()
