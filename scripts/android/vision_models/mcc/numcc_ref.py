"""NU-MCC (Lionar et al., NeurIPS 2023; sail-sg/numcc, Apache-2.0 code) on the host, on the same
input as MCC, to decide whether it is the better phone target.

  python numcc_ref.py --ckpt udf-ep99.pth --work <dir>     (NUMCC_REPO: clone of sail-sg/numcc)

Runs upstream's demo inference (demo_iphone.run_viz_udf) step by step on CPU:
  encoder + anchor decoder (once) -> grid queries inside the anchors' box (+0.3) -> UDF ->
  keep UDF < 0.23 -> 10 x move along -grad(UDF) (autograd through the decoder) + repulsion ->
  final color pass.
and reports queries / forward-equivalents / host time per stage, the per-query decoder cost next
to MCC's, and NU-MCC's points vs MCC's dense occupied set (chamfer). pytorch3d (imported at module
level upstream, used only for training-time FPS / rotation augmentation) is stubbed.
"""

import argparse
import os
import sys
import time
import types

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model as M  # noqa: E402


def stub_pytorch3d():
    """Upstream imports pytorch3d at module level (datasets, losses, training augmentation);
    none of it runs at inference. Stub every imported name."""
    names = {
        "pytorch3d.implicitron.dataset.dataset_base": ["FrameData"],
        "pytorch3d.implicitron.dataset.dataset_map_provider": ["DatasetMap"],
        "pytorch3d.implicitron.tools.config": ["expand_args_fields"],
        "pytorch3d.io": ["IO", "load_obj"],
        "pytorch3d.io.obj_io": ["load_obj"],
        "pytorch3d.loss": ["chamfer_distance"],
        "pytorch3d.ops": ["sample_farthest_points", "sample_points_from_meshes"],
        "pytorch3d.ops.knn": ["knn_gather", "knn_points"],
        "pytorch3d.renderer.cameras": ["CamerasBase"],
        "pytorch3d.structures": ["Pointclouds"],
        "pytorch3d.structures.pointclouds": ["Pointclouds"],
        "pytorch3d.transforms": ["RotateAxisAngle"],
        "pytorch3d.vis.plotly_vis": ["plot_scene"],
    }
    for full, attrs in names.items():
        parts = full.split(".")
        for i in range(1, len(parts) + 1):
            mod = sys.modules.setdefault(
                ".".join(parts[:i]), types.ModuleType(".".join(parts[:i]))
            )
            mod.__path__ = []  # a package, so submodules import
        for a_ in attrs:
            setattr(sys.modules[full], a_, None)
    six = types.ModuleType("torch._six")  # upstream util/misc.py (torch < 2.0 API)
    six.inf = float("inf")
    sys.modules.setdefault("torch._six", six)


def load_numcc(ckpt, repo):
    stub_pytorch3d()
    sys.path.insert(0, repo)
    if not hasattr(np, "float"):
        np.float = float
    import parser_and_builder  # noqa: E402
    from src.model.nu_mcc import NUMCC  # noqa: E402

    args = parser_and_builder.get_args_parser().parse_args([])
    args.device = "cpu"
    args.drop_path = 0.0
    args.distributed = False
    m = NUMCC(args=args)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not missing, missing[:5]
    return m.eval(), args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--gran", type=float, default=0.1)
    ap.add_argument("--chunk", type=int, default=4096)
    a = ap.parse_args()
    repo = os.environ.get(
        "NUMCC_REPO", os.path.expanduser("~/.cache/onnxsim-mcc/numcc")
    )
    mcc_repo = os.environ.get(
        "MCC_REPO", os.path.expanduser("~/.cache/onnxsim-mcc/MCC")
    )
    model, args = load_numcc(a.ckpt, repo)
    from src.fns import (  # noqa: E402
        move_points,
        preprocess_img,
        shrink_points_beyond_threshold,
    )

    img, xyz = M.load_demo(mcc_repo, "quest2")  # same input as the MCC reference
    img = img * torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1) + torch.tensor(
        [0.485, 0.456, 0.406]
    ).reshape(1, 3, 1, 1)  # NU-MCC normalizes inside
    seen = xyz.clone()[None]
    valid = torch.isfinite(seen.sum(-1))
    seen[~valid] = -100.0
    seen = shrink_points_beyond_threshold(seen, args.shrink_threshold)

    t = time.time()
    with torch.no_grad():
        latent, up = model.encoder(preprocess_img(img), seen, valid)
        fea = model.decoderl1(latent)
    t_enc = time.time() - t
    centers = fea["anchors_xyz"]
    lo, hi = centers.min(1)[0][0] - 0.3, centers.max(1)[0][0] + 0.3
    q = M.grid(a.gran)
    keep = ((q[0] > lo) & (q[0] < hi)).all(-1)
    q = q[:, keep]

    def dec(x):
        return model.fc_out(model.decoderl2(x, seen, valid, fea, up))

    # stage 1: UDF on the boxed grid
    t = time.time()
    udfs = []
    with torch.no_grad():
        for s in range(0, q.shape[1], a.chunk):
            udfs.append(F.relu(dec(q[:, s : s + a.chunk])[..., 0]))
    udf = torch.cat(udfs, 1).clamp(max=0.5)
    t_q = time.time() - t
    pts = q[:, (udf[0] < args.udf_threshold)]
    # stage 2: move the candidates (autograd) + repulsion, as upstream
    for p_ in model.parameters():
        p_.requires_grad = False
    t = time.time()
    moved = []
    for s in range(0, pts.shape[1], a.chunk):
        moved.append(
            move_points(
                model,
                pts[:, s : s + a.chunk].clone(),
                seen,
                valid,
                fea,
                up,
                args,
                n_iter=args.udf_n_iter,
            )
        )
    moved = torch.cat(moved, 1).detach()
    t_mv = time.time() - t
    t = time.time()
    with torch.no_grad():
        col = torch.cat(
            [
                dec(moved[:, s : s + a.chunk])[..., 1:]
                for s in range(0, moved.shape[1], a.chunk)
            ],
            1,
        )
    t_col = time.time() - t
    rgb = col.reshape(-1, 3, 256).argmax(-1).float() / 255

    # per-query decoder cost: NU-MCC decoderl2+fc_out vs MCC QueryDecoder, same 4096 queries, same host
    x = q[:, : a.chunk]
    with torch.no_grad():
        t = time.time()
        dec(x)
        t_nu = (time.time() - t) / x.shape[1]
        mcc = M.load_mcc(
            os.path.expanduser("~/.cache/onnxsim-mcc/co3dv2_all_categories.pth"),
            mcc_repo,
        )
        kv = np.load(os.path.join(a.work, "kv_quest2.npz"))
        qd = M.QueryDecoder(mcc)
        t = time.time()
        qd(x, torch.from_numpy(kv["k"]), torch.from_numpy(kv["v"]))
        t_mcc = (time.time() - t) / x.shape[1]

    ref = np.load(os.path.join(a.work, f"ref_quest2_{a.gran}.npz"))
    occ = ref["xyz"][1 / (1 + np.exp(-ref["occ"])) > 0.3]
    import queries as Qs

    ch = Qs.chamfer(moved[0].numpy(), occ)
    n_q, n_c = q.shape[1], pts.shape[1]
    fwd = n_q + n_c * (args.udf_n_iter * 3 + 1)  # a backward ~ 2 forwards
    print(f"NU-MCC on host CPU (quest2, granularity {a.gran}):")
    print(f"  encoder + anchor decoder: {t_enc:.2f} s")
    print(
        f"  box-filtered grid: {n_q} of {int(keep.numel())} queries, UDF pass {t_q:.1f} s"
    )
    print(
        f"  candidates (UDF < {args.udf_threshold}): {n_c}; {args.udf_n_iter} move iterations {t_mv:.1f} s; color {t_col:.1f} s"
    )
    print(
        f"  forward-equivalents ~{fwd} (vs MCC dense {int(keep.numel())}, coarse-to-fine ~36.7k)"
    )
    print(
        f"  per-query decoder cost, host: NU-MCC {t_nu * 1e6:.0f} us vs MCC {t_mcc * 1e6:.0f} us"
    )
    print(
        f"  output points {moved.shape[1]}; chamfer vs MCC occupied set (p > 0.3, {len(occ)} pts): {ch:.4f}"
    )
    np.savez(
        os.path.join(a.work, f"numcc_quest2_{a.gran}.npz"),
        xyz=moved[0].numpy(),
        rgb=rgb.numpy(),
    )


if __name__ == "__main__":
    main()
