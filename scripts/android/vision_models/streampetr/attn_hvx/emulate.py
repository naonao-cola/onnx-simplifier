"""Host emulation of the HVX cross-attention kernel's integer math, on StreamPETR's real chain.

  python attn_hvx/emulate.py --ckpt <pth> --work <work> calib     # per-layer Q/K/V uint8 ranges
  python attn_hvx/emulate.py --ckpt <pth> --work <work> eval      # scene-0103 x 6, chained, vs fp32

The kernel contract (``attn_int``): per layer the HTP emits Q (428, 256) (already scaled by 1/sqrt(32)),
K (4224, 256), V (4224, 256) as uint8 with one (scale, zero point) per tensor. Per head h and query row:
  S   = (q - zq) . (k - zk)             exact int32 (the kernel computes q.k with ub x ub vrmpy + sums)
  p   = round(255 * exp((S - max S) * sq * sk))        uint8, 0 beyond ~6.2 nats below the max
  O   = round(sum p * v / sum p)        uint8 with V's (scale, zero point): a convex combination of V
so the kernel needs no float at all except in building the exp table. ``--exp poly`` switches the
ideal exp for the kernel's fixed-point one (``exp_u8``, bit-exact with attn_kernel.h)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data as D  # noqa: E402
import model as M  # noqa: E402
from attn_hvx.attn_contract import div_round, exp_params, exp_u8  # noqa: E402
from attn_hvx.attn_contract import write_case as _write_u8

CALIB = ["scene-0061", "scene-0553", "scene-0757", "scene-1077"]
IN = ["feat", "pe", "sa_gamma", "sa_beta", "mem_emb", "mem_pe3d", "mem_time", "mem_motion"]


def qkv(mha, q, k, v):
    wq, wk, wv = mha.in_proj_weight.chunk(3)
    bq, bk, bv = mha.in_proj_bias.chunk(3)
    return (torch.nn.functional.linear(q, wq, bq) * (32 ** -0.5), torch.nn.functional.linear(k, wk, bk),
            torch.nn.functional.linear(v, wv, bv))


def qparams(lo, hi):
    lo, hi = min(lo, 0.0), max(hi, 0.0)
    s = (hi - lo) / 255 or 1.0
    return s, int(np.clip(round(-lo / s), 0, 255))


def quant(x, s, z):
    return np.clip(np.round(x / s) + z, 0, 255).astype(np.int64)


def attn_int(qf, kf, vf, pq, pk, pv, exp="ideal"):
    """(Lq, 256) float Q (pre-scaled), K, V -> (Lq, 256) float output after the uint8 kernel."""
    (sq, zq), (sk, zk), (sv, zv) = pq, pk, pv
    qu, ku, vu = quant(qf, sq, zq), quant(kf, sk, zk), quant(vf, sv, zv)
    out = np.empty(qu.shape, np.int64)
    for h in range(8):
        sl = slice(32 * h, 32 * h + 32)
        s = (qu[:, sl] - zq) @ (ku[:, sl] - zk).T  # exact int
        d = s.max(1, keepdims=True) - s
        if exp == "ideal":
            p = np.round(255 * np.exp(-d * (sq * sk))).astype(np.int64)
            acc, sp = p @ vu[:, sl], p.sum(1, keepdims=True)
            out[:, sl] = (acc + sp // 2) // sp
        else:
            p = exp_u8(d, *exp_params(sq * sk))
            out[:, sl] = div_round(p @ vu[:, sl], p.sum(1, keepdims=True))
    return ((out - zv) * sv).astype(np.float32)


def write_case(d, qf, kf, vf, pq, pk, pv):
    """A real-layer case directory (attn_contract.write_case) from float Q (pre-scaled), K, V."""
    (sq, zq), (sk, zk), (sv, zv) = pq, pk, pv
    _write_u8(d, quant(qf, sq, zq), quant(kf, sk, zk), quant(vf, sv, zv), zq, zk, *exp_params(sq * sk))
    print(f"{d}: LQ {len(qf)} LK {len(kf)} zq {zq} params {exp_params(sq * sk)}")


def run_frame(core, host, z, prev, mode, qp=None, rec=None, exp="ideal", cap=None):
    t = M.to_torch({k: z[k] for k in ("lidar2img", "intrinsics", "ego_pose", "ego_pose_inv")} | {"timestamp": float(z["timestamp"])})
    ins = {"feat": torch.from_numpy(z["feat"]), **host.rig_inputs(t), **host.pre(t, prev)}
    for li, layer in enumerate(core.h.decoder_layers):
        mha = layer.attentions[1].attn

        def fwd(q, k, v, mha=mha, li=li):
            Q, K, V = qkv(mha, q, k, v)
            if cap is not None:
                cap[li] = (Q.numpy(), K.numpy(), V.numpy())
            if rec is not None:
                for n, x in zip("qkv", (Q, K, V)):
                    lo, hi = rec.setdefault(f"{li}{n}", [1e9, -1e9])
                    rec[f"{li}{n}"] = [min(lo, float(x.min())), max(hi, float(x.max()))]
            if mode == "fp32":
                o = torch.softmax(Q.view(-1, 8, 32).transpose(0, 1) @ K.view(-1, 8, 32).permute(1, 2, 0), -1) @ V.view(-1, 8, 32).transpose(0, 1)
                o = o.transpose(0, 1).reshape(-1, 256)
            else:
                o = torch.from_numpy(attn_int(Q.numpy(), K.numpy(), V.numpy(), *[qp[f"{li}{n}"] for n in "qkv"], exp=exp))
            return mha.out_proj(o)
        mha.forward = fwd
    cls, reg, dec = core(*[ins[k] for k in IN])
    return host.post(t, cls, reg, dec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["calib", "eval", "case"])
    ap.add_argument("--out", default="cases", help="case: output directory")
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 5])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--exp", default="ideal", choices=["ideal", "poly"])
    ap.add_argument("--scene", default="scene-0103")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    work = Path(a.work)
    _, head = M.load_official(a.ckpt)
    core = M.HeadCore(head).eval()
    frames = lambda sc: sorted((work / "frames" / sc).glob("*.npz"), key=lambda p: int(p.stem))  # noqa: E731
    if a.cmd == "calib":
        rec = {}
        for sc in CALIB:
            host = M.HostState(head)
            for fp in frames(sc):
                z = np.load(fp)
                run_frame(core, host, z, bool(z["prev"]), "fp32", rec=rec)
        qp = {k: qparams(*v) for k, v in rec.items()}
        (work / "attn_qparams.json").write_text(json.dumps(qp, indent=1))
        print(json.dumps({k: [round(v[0], 5), v[1]] for k, v in qp.items()}))
        return
    qp = {k: tuple(v) for k, v in json.loads((work / "attn_qparams.json").read_text()).items()}
    if a.cmd == "case":
        z = np.load(frames(a.scene)[0])
        cap = {}
        run_frame(core, M.HostState(head), z, bool(z["prev"]), "fp32", cap=cap)
        for li in a.layers:
            write_case(Path(a.out) / f"{a.scene}_l{li}", *cap[li], *[qp[f"{li}{n}"] for n in "qkv"])
        return
    h_ref, h_q = M.HostState(head), M.HostState(head)
    tot = {0.3: [0, 0], 0.2: [0, 0]}
    for i, fp in enumerate(frames(a.scene)):
        z = np.load(fp)
        prev = bool(z["prev"])
        r_cls, r_box = run_frame(core, h_ref, z, prev, "fp32")
        cls, box = run_frame(core, h_q, z, prev, "int", qp, exp=a.exp)
        gt = list(zip(z["gt_names"].tolist(), z["gt_xyz"]))
        line = f"frame {i}: cos cls {np.corrcoef(cls.ravel(), r_cls.ravel())[0, 1]:.6f} box {np.corrcoef(box.ravel(), r_box.ravel())[0, 1]:.6f}"
        for thr in tot:
            m = D.match(*M.decode(cls, box), gt, thr=thr)[0]
            rm = D.match(*M.decode(r_cls, r_box), gt, thr=thr)[0]
            tot[thr][0] += m
            tot[thr][1] += rm
            line += f" | @{thr} int {m} fp32 {rm}"
        print(line, flush=True)
    for thr, (m, rm) in tot.items():
        print(f"total @{thr}: int-attention {m}/190, fp32 {rm}/190")


if __name__ == "__main__":
    main()
