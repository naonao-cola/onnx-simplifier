"""Check the demo app's MCC mode (../../maskrcnn_demo_app, native/mcc_engine.cpp) against this directory's
Python pipeline, on the app's own intermediate tensors.

  adb shell am start -n org.onnxsim.maskrcnndemo/.MccActivity --es mode images --es image quest2.jpg \
      --es tap 0.506,0.491 --ez recon true --es opts dump=1
  adb exec-out run-as org.onnxsim.maskrcnndemo tar c files/mcc_dump | tar x -C <dir>
  python app_check.py --dump <dir>/files/mcc_dump --ckpt <co3dv2_all_categories.pth>

From the app's working image, SAM mask and MoGe-2 points (phone outputs, taken as given):
  1. model.prep + xyz_windows on the host vs the app's C++ port (encoder inputs),
  2. the host fp32 encoder on those inputs vs the phone's K/V,
  3. the dense host fp32 grid at the app's granularity vs the phone's coarse-to-fine result
     (recall / precision of the occupied points, chamfer, color), as mcc.py recon scores it.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import model as M  # noqa: E402
import queries as Qs  # noqa: E402
from mcc import cos  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--ckpt", default=str(Path.home() / ".cache/onnxsim-mcc/co3dv2_all_categories.pth"))
    ap.add_argument("--thr", type=float, default=0.3)
    a = ap.parse_args()
    d = Path(a.dump)

    def rd(name, dtype, shape):
        return np.fromfile(d / name, dtype=dtype).reshape(shape)

    W, H, nt, _ = rd("dims.i32", np.int32, (4,))
    rgb = rd("rgb.u8", np.uint8, (H, W, 3))
    mask = rd("mask.u8", np.uint8, (H, W)).astype(bool)
    pts = rd("moge_points.f32", np.float32, (H, W, 3)) * np.array([1, -1, -1], np.float32)
    app_img = rd("img.f32", np.float32, (1, 3, 224, 224))
    app_win = rd("xyz_win.f32", np.float32, (196, 64, 3))
    app_val = rd("valid.f32", np.float32, (196, 64))
    app_k, app_v = (rd(f"{n}.f32", np.float32, (8, 16, 197, 32)) for n in ("k", "v"))
    app_p = rd("p.f32", np.float32, (nt,) * 3)
    app_rgb = rd("rgb.f32", np.float32, (nt,) * 3 + (3,))
    print(f"working image {W}x{H}, mask {mask.sum()} px, grid {nt}^3")

    torch.set_grad_enabled(False)
    img, xyz112 = M.prep(torch.from_numpy(rgb.astype(np.float32) / 255), torch.from_numpy(pts), torch.from_numpy(mask))
    win, val = M.xyz_windows(xyz112)
    both = (val.numpy() > 0.5) & (app_val > 0.5)
    print(
        f"prep: img max abs {np.abs(img.numpy() - app_img).max():.3g}; valid {int(val.sum())} host vs "
        f"{int(app_val.sum())} app, {int((val.numpy() != app_val).sum())} differ; "
        f"xyz max abs (both valid) {np.abs(win.numpy() - app_win)[both].max():.3g}"
    )
    m = M.load_mcc(a.ckpt)
    k, v = M.Encoder(m).eval()(img, win, val)
    print(f"encoder: phone K cos {cos(app_k, k.numpy()):.6f}, V cos {cos(app_v, v.numpy()):.6f} (vs host fp32 on host prep)")
    dec = M.QueryDecoder(m).eval()
    g = M.grid(6.0 / nt)
    occ, col = zip(*(dec(g[:, s : s + 4096], k, v) for s in range(0, g.shape[1], 4096)))
    p = torch.sigmoid(torch.cat(occ, 1)[0]).numpy().reshape((nt,) * 3)
    ref_rgb = torch.cat(col, 1)[0].numpy().reshape((nt,) * 3 + (3,))
    occ_ref, occ_app = p > a.thr, app_p > a.thr
    inter = occ_ref & occ_app
    print(
        f"recon vs dense host fp32 (p > {a.thr}): {int(occ_app.sum())} app vs {int(occ_ref.sum())} host points, "
        f"recall {inter.sum() / max(occ_ref.sum(), 1):.4f} precision {inter.sum() / max(occ_app.sum(), 1):.4f} "
        f"chamfer {Qs.chamfer(Qs.coords(occ_app, nt), Qs.coords(occ_ref, nt)):.4f} "
        f"color L1 {np.abs(app_rgb[inter] - ref_rgb[inter]).mean() * 255:.2f}/255"
    )


if __name__ == "__main__":
    main()
