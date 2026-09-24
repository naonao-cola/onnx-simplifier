"""Run nfru_kernels.cl on a host OpenCL device (pyopencl) against the torch golden dumps (nfru.py golden).

  nfru_cl_check.py [--windows N]

Per stage (open loop: every stage gets the golden inputs) -- the block matcher level by level (pyramid,
warped search image, vectors, hint mask, median, joint bilateral), the dynamic mask, warp_mv, warp_flow,
preprocess, postprocess -- then closed loop (golden frame inputs only; the network is the int8 ONNX model
on host ORT fed the kernels' uint8 input), with PSNR vs the golden fp32 output and vs the ground truth.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WORK = Path(os.environ.get("NFRU_WORK", Path.home() / ".cache/arm-nfru"))
GOLD = WORK / "golden"
T = 0.5
PSC, PZP = (
    0.35356706380844116,
    172.0,
)  # the network's uint8 output quantization (Arm's QAT, see nfru.py)


def psnr(a, b) -> float:
    d = np.clip(a, 0, 1).astype(np.float64) - np.clip(b, 0, 1).astype(np.float64)
    return float(10 * np.log10(1 / max(float((d * d).mean()), 1e-12)))


def f32(a):
    return np.ascontiguousarray(a, np.float32)


class NfruCL:
    def __init__(self):
        import pyopencl as cl

        self.cl = cl
        dev = None
        for p in cl.get_platforms():
            try:
                ds = p.get_devices()
            except cl.Error:
                continue
            if ds:
                dev = ds[0]
                break
        self.ctx = cl.Context([dev])
        self.q = cl.CommandQueue(self.ctx)
        self.prg = cl.Program(self.ctx, (HERE / "nfru_kernels.cl").read_text()).build(
            options=["-cl-fp32-correctly-rounded-divide-sqrt"]
        )
        self.kern = {}
        self.null = None

    def buf(self, a):
        mf = self.cl.mem_flags
        a = np.ascontiguousarray(a)
        return self.cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a)

    def empty(self, nbytes):
        return self.cl.Buffer(
            self.ctx, self.cl.mem_flags.READ_WRITE, max(int(nbytes), 4)
        )

    def get(self, b, shape, dtype):
        o = np.empty(shape, dtype)
        self.cl.enqueue_copy(self.q, o, b)
        return o

    def run(self, name, gsz, *args):
        if name not in self.kern:
            self.kern[name] = self.cl.Kernel(self.prg, name)
        k = self.kern[name]
        cl_args = []
        for a in args:
            if isinstance(a, (int, np.integer)) and not isinstance(a, bool):
                cl_args.append(np.int32(a))
            elif isinstance(a, float):
                cl_args.append(np.float32(a))
            else:
                cl_args.append(a)
        k(self.q, gsz, None, *cl_args)

    # --- the block matcher -------------------------------------------------------------------------------
    def pyramid(self, rgb):
        """uint8 luma pyramids, coarsest first: [(buf, Hpad, Wpad, Htrue, Wtrue)] (levels 0..3 of the gym's
        reversed pyramid; 34x60 ... 270x480)."""
        _, _, H, W = rgb.shape
        y = self.empty(H * W)
        self.run("luma8", (H * W,), self.buf(f32(rgb)), H * W, y)
        lv = [(y, H, W, H, W)]
        for i in range(1, 6):
            s, h, w, _, _ = lv[-1]
            hd, wd = h // 2, w // 2
            ho, wo = hd + hd % 2, wd + wd % 2
            o = self.empty(ho * wo)
            self.run(
                "pyr_down",
                (wo, ho),
                s,
                h,
                w,
                int((i - 1) in (1, 2, 3, 4)),
                int(i == 5),
                o,
                ho,
                wo,
                hd,
                wd,
            )
            lv.append((o, ho, wo, hd, wd))
        return lv[::-1][:4]  # coarsest first, through the target level (index 3)

    def hint(self, sy, depth, H, W):
        _, _, h, w = sy.shape
        o = self.empty(2 * H * W * 4)
        self.run(
            "hint_mv", (W, H), self.buf(f32(sy)), self.buf(f32(depth)), h, w, o, H, W
        )
        return o

    def level(self, srch, tmpl, H, W, Ht, Wt, prev=None, hint_mv=None):
        """One level; prev = (buf, Hi, Wi) the previous level's cropped vectors. Returns a dict of bufs."""
        r = {}
        if prev is not None:
            vp = self.empty(2 * H * W * 4)
            self.run("bm_upsample", (W, H), prev[0], prev[1], prev[2], vp, H, W)
            sw = self.empty(H * W)
            self.run("bm_warp", (W, H), srch, H, W, vp, H, W, sw)
        else:
            vp, sw = self.null, srch
        r["search"] = sw
        hint = self.null
        if hint_mv is not None:
            hint = self.empty(H * W)
            self.run("bm_warp", (W, H), srch, H, W, hint_mv, H, W, hint)
            r["hintimg"] = hint
        vec, won = self.empty(2 * H * W * 4), self.empty(H * W)
        self.run("bm_match", (W, H), sw, tmpl, hint, H, W, vp, vec, won)
        med = self.empty(2 * H * W * 4)
        self.run("bm_median", (W, H), vec, H, W, med)
        out = self.empty(2 * Ht * Wt * 4)
        self.run(
            "bm_jbf",
            (Wt, Ht),
            med,
            tmpl,
            H,
            W,
            won if hint_mv is not None else self.null,
            hint_mv,
            H,
            W,
            out,
            Ht,
            Wt,
        )
        r.update(premed=vec, won=won, med=med, jbf=out)
        return r


def check_stages(g: NfruCL, z, report):
    # pyramid (templates = m1)
    pm, pp = g.pyramid(z["rgb_m1"]), g.pyramid(z["rgb_p1"])
    for lvl, (b, h, w, _, _) in enumerate(pm):
        report(
            f"pyramid template{lvl}",
            g.get(b, (h, w), np.uint8),
            z[f"bm_template{lvl}"][0, 0],
        )
    report(
        "pyramid search0", g.get(pp[0][0], pp[0][1:3], np.uint8), z["bm_search0"][0, 0]
    )
    Hh, Wh = pm[3][1], pm[3][2]
    hint = g.hint(z["sy_m1_f30_p1"], z["depth_m1"], Hh, Wh)
    for lvl in range(4):
        (sb, H, W, Ht, Wt), tb = pp[lvl], pm[lvl][0]
        prev = None
        if lvl:
            ph, pw = pm[lvl - 1][3], pm[lvl - 1][4]
            pv = f32(z[f"bm_jbf{lvl - 1}"][0][:, :ph, :pw])
            prev = (g.buf(pv), ph, pw)
        r = g.level(sb, tb, H, W, Ht, Wt, prev, hint if lvl == 3 else None)
        report(
            f"bm{lvl} search(warped)",
            g.get(r["search"], (H, W), np.uint8),
            z[f"bm_search{lvl}"][0, 0],
        )
        if lvl == 3:
            report(
                "bm3 hint image",
                g.get(r["hintimg"], (H, W), np.uint8),
                z["bm_hintimg3"][0, 0],
            )
            won = g.get(r["won"], (H, W), np.uint8)
            report("bm3 hint mask", won, z["bm_hintmask3"][0, :, :, 0].astype(np.uint8))
        report(
            f"bm{lvl} vector+subpixel+prev",
            g.get(r["premed"], (2, H, W), np.float32),
            z[f"bm_premed{lvl}"][0].astype(np.float32),
        )
        # open loop: median / JBF from the golden inputs
        med_in = g.buf(f32(z[f"bm_premed{lvl}"][0]))
        med = g.empty(2 * H * W * 4)
        g.run("bm_median", (W, H), med_in, H, W, med)
        report(
            f"bm{lvl} median",
            g.get(med, (2, H, W), np.float32),
            z[f"bm_med{lvl}"][0].astype(np.float32),
        )
        jb = g.empty(2 * H * W * 4)
        g.run(
            "bm_jbf",
            (W, H),
            g.buf(f32(z[f"bm_med{lvl}"][0])),
            tb,
            H,
            W,
            g.null,
            g.null,
            H,
            W,
            jb,
            H,
            W,
        )
        report(
            f"bm{lvl} joint bilateral",
            g.get(jb, (2, H, W), np.float32),
            z[f"bm_jbf{lvl}"][0].astype(np.float32),
        )
    # flow normalization
    fr = z["flow_raw"][0]
    fh, fw = fr.shape[1:]
    fl = g.empty(2 * fh * fw * 4)
    g.run("norm_mv", (fh * fw,), g.buf(f32(fr)), fh * fw, 1.0, float(fh), float(fw), fl)
    report("flow normalize", g.get(fl, (2, fh, fw), np.float32), z["flow"][0])
    # the dynamic mask (previous frame m1 vs m3)
    _, _, H, W = z["depth_m1"].shape
    mvm1 = g.empty(2 * H * W * 4)
    g.run(
        "norm_mv",
        (H * W,),
        g.buf(f32(z["mv_m1_f30_m3"][0])),
        H * W,
        1.0,
        float(H),
        float(W),
        mvm1,
    )
    mvp1 = g.empty(2 * H * W * 4)
    g.run(
        "norm_mv",
        (H * W,),
        g.buf(f32(z["mv_p1_f30_m1"][0])),
        H * W,
        1.0,
        float(H),
        float(W),
        mvp1,
    )
    dyn = g.empty(H * W * 4)
    g.run(
        "dyn_mask",
        (W, H),
        g.buf(f32(z["depth_m1"])),
        mvm1,
        g.buf(f32(z["motion_mat_m3"][0])),
        H,
        W,
        dyn,
    )
    report("dynamic mask", g.get(dyn, (H, W), np.float32), z["dyn"][0, 0])
    # warp_mv (golden dynamic mask)
    packed, ht, hm = g.empty(H * W * 4), g.empty(H * W * 4), g.empty(H * W * 4)
    for b in (packed, ht, hm):
        g.run("zero_i32", (H * W,), b)
    mm = z["motion_mat"][0]
    g.run(
        "warp_mv",
        (W, H),
        g.buf(f32(z["depth_p1"])),
        g.buf(f32(z["depth_m1"])),
        mvp1,
        g.buf(f32(z["dyn"])),
        g.buf(f32(mm[1])),
        H,
        W,
        T,
        packed,
        ht,
        hm,
    )
    mvt = g.empty(2 * H * W * 4)
    g.run("fill_mv", (W, H), packed, H, W, mvt)
    report("warp_mv mv_t", g.get(mvt, (2, H, W), np.float32), z["mv_t"][0])
    report(
        "warp_mv holes_t",
        g.get(ht, (H, W), np.int32).astype(np.float32),
        z["holes_t"][0, 0],
    )
    report(
        "warp_mv holes_tm1",
        g.get(hm, (H, W), np.int32).astype(np.float32),
        z["holes_tm1"][0, 0],
    )
    # warp_flow (golden flow)
    pf = g.empty(fh * fw * 4)
    g.run("zero_i32", (fh * fw,), pf)
    g.run(
        "warp_flow",
        (fw, fh),
        g.buf(f32(z["depth_m1"])),
        H,
        W,
        g.buf(f32(z["flow"])),
        fh,
        fw,
        1.0 - T,
        pf,
    )
    flt = g.empty(2 * fh * fw * 4)
    g.run("fill_mv", (fw, fh), pf, fh, fw, flt)
    report("warp_flow flow_t", g.get(flt, (2, fh, fw), np.float32), z["flow_t"][0])
    # preprocess (golden inputs)
    _, _, cH, cW = z["rgb_m1"].shape
    netf, netu = g.empty(16 * fh * fw * 4), g.empty(16 * fh * fw)
    dp = z["depth_params"].astype(np.float32)
    g.run(
        "preprocess",
        (fw, fh),
        g.buf(f32(z["flow_t"])),
        g.buf(f32(z["mv_t"])),
        H,
        W,
        g.buf(f32(z["rgb_m1"])),
        g.buf(f32(z["rgb_p1"])),
        cH,
        cW,
        g.buf(f32(z["depth_m1"])),
        g.buf(f32(z["depth_p1"])),
        g.buf(z["holes_t"].astype(np.int32)),
        g.buf(z["holes_tm1"].astype(np.int32)),
        g.buf(f32(mm[1])),
        g.buf(f32(mm[0])),
        np.array(dp, np.float32).view(cl_float4()),
        T,
        np.uint32(int(z["seed"]) & 0xFFFFFFFF),
        fh,
        fw,
        netf,
        netu,
    )
    report("preprocess net_in", g.get(netf, (16, fh, fw), np.float32), z["net_in"][0])
    nu = g.get(netu, (fh, fw, 16), np.uint8)
    ref_u = np.clip(np.rint(z["net_in"][0].transpose(1, 2, 0) * 255), 0, 255).astype(
        np.uint8
    )
    report("preprocess net_in uint8", nu, ref_u)
    # postprocess: golden logits quantized to the network's uint8 output; the reference is the torch
    # postprocess on the same (dequantized) logits
    pu = np.clip(np.rint(z["params"][0].transpose(1, 2, 0) / PSC + PZP), 0, 255).astype(
        np.uint8
    )
    ref = torch_post(z, (pu.astype(np.float32) - np.float32(PZP)) * np.float32(PSC))
    out, rgba = g.empty(3 * cH * cW * 4), g.empty(cH * cW * 4)
    g.run(
        "postprocess",
        (cW, cH),
        g.buf(f32(z["flow_t"])),
        fh,
        fw,
        g.buf(f32(z["mv_t"])),
        H,
        W,
        g.buf(pu),
        fh,
        fw,
        PSC,
        PZP,
        g.buf(f32(z["rgb_m1"])),
        g.buf(f32(z["rgb_p1"])),
        cH,
        cW,
        T,
        out,
        rgba,
    )
    report("postprocess", g.get(out, (3, cH, cW), np.float32), ref)


def cl_float4():
    import pyopencl.cltypes as ct

    return ct.float4


def torch_post(z, params_hwc):
    import importlib

    import nfru_gym
    import torch

    nfru_gym._install()
    pp = importlib.import_module(
        "ng_model_gym.usecases.nfru.model.torch_processing.postprocess"
    )
    p = torch.from_numpy(np.ascontiguousarray(params_hwc.transpose(2, 0, 1)))[None]
    t = lambda k: torch.from_numpy(f32(z[k]))  # noqa: E731
    return pp.postprocess_torch(t("flow_t"), t("mv_t"), t("rgb_m1"), t("rgb_p1"), p, T)[
        0
    ].numpy()


def closed_loop(g: NfruCL, z, sess):
    """Golden frame inputs only (tonemapped colours, depths, motion, matrices) -> the generated frame."""
    pm, pp = g.pyramid(z["rgb_m1"]), g.pyramid(z["rgb_p1"])
    Hh, Wh = pm[3][1], pm[3][2]
    hint = g.hint(z["sy_m1_f30_p1"], z["depth_m1"], Hh, Wh)
    prev = None
    for lvl in range(4):
        (sb, H, W, Ht, Wt), tb = pp[lvl], pm[lvl][0]
        r = g.level(sb, tb, H, W, Ht, Wt, prev, hint if lvl == 3 else None)
        prev = (r["jbf"], Ht, Wt)
    fh, fw = prev[1], prev[2]
    flow = g.empty(2 * fh * fw * 4)
    g.run("norm_mv", (fh * fw,), prev[0], fh * fw, -4.0, float(fh), float(fw), flow)
    _, _, H, W = z["depth_m1"].shape
    dm1, dp1 = g.buf(f32(z["depth_m1"])), g.buf(f32(z["depth_p1"]))
    mvm1, mvp1 = g.empty(2 * H * W * 4), g.empty(2 * H * W * 4)
    g.run(
        "norm_mv",
        (H * W,),
        g.buf(f32(z["mv_m1_f30_m3"][0])),
        H * W,
        1.0,
        float(H),
        float(W),
        mvm1,
    )
    g.run(
        "norm_mv",
        (H * W,),
        g.buf(f32(z["mv_p1_f30_m1"][0])),
        H * W,
        1.0,
        float(H),
        float(W),
        mvp1,
    )
    dyn = g.empty(H * W * 4)
    g.run("dyn_mask", (W, H), dm1, mvm1, g.buf(f32(z["motion_mat_m3"][0])), H, W, dyn)
    packed, ht, hm, pf = (
        g.empty(H * W * 4),
        g.empty(H * W * 4),
        g.empty(H * W * 4),
        g.empty(fh * fw * 4),
    )
    for b, n in ((packed, H * W), (ht, H * W), (hm, H * W), (pf, fh * fw)):
        g.run("zero_i32", (n,), b)
    mm = z["motion_mat"][0]
    m0, m1 = g.buf(f32(mm[0])), g.buf(f32(mm[1]))
    g.run("warp_mv", (W, H), dp1, dm1, mvp1, dyn, m1, H, W, T, packed, ht, hm)
    mvt = g.empty(2 * H * W * 4)
    g.run("fill_mv", (W, H), packed, H, W, mvt)
    g.run("warp_flow", (fw, fh), dm1, H, W, flow, fh, fw, 1.0 - T, pf)
    flt = g.empty(2 * fh * fw * 4)
    g.run("fill_mv", (fw, fh), pf, fh, fw, flt)
    _, _, cH, cW = z["rgb_m1"].shape
    rm1, rp1 = g.buf(f32(z["rgb_m1"])), g.buf(f32(z["rgb_p1"]))
    netu = g.empty(16 * fh * fw)
    dp = np.array(z["depth_params"], np.float32).view(cl_float4())
    g.run(
        "preprocess",
        (fw, fh),
        flt,
        mvt,
        H,
        W,
        rm1,
        rp1,
        cH,
        cW,
        dm1,
        dp1,
        ht,
        hm,
        m1,
        m0,
        dp,
        T,
        np.uint32(int(z["seed"]) & 0xFFFFFFFF),
        fh,
        fw,
        g.null,
        netu,
    )
    nu = g.get(netu, (1, fh, fw, 16), np.uint8)
    pu = np.ascontiguousarray(sess.run(None, {sess.get_inputs()[0].name: nu})[0])
    out, rgba = g.empty(3 * cH * cW * 4), g.empty(cH * cW * 4)
    g.run(
        "postprocess",
        (cW, cH),
        flt,
        fh,
        fw,
        mvt,
        H,
        W,
        g.buf(pu),
        fh,
        fw,
        PSC,
        PZP,
        rm1,
        rp1,
        cH,
        cW,
        T,
        out,
        rgba,
    )
    return g.get(out, (3, cH, cW), np.float32), g.get(flt, (2, fh, fw), np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument(
        "--stages", type=int, default=1, help="windows to run the per-stage checks on"
    )
    a = ap.parse_args()
    import onnxruntime as ort

    g = NfruCL()
    print("device:", g.ctx.devices[0].name)

    def report(name, got, ref):
        got, ref = np.asarray(got), np.asarray(ref)
        assert got.shape == ref.shape, (name, got.shape, ref.shape)
        d = np.abs(got.astype(np.float64) - ref.astype(np.float64))
        print(
            f"  {name:32s} max|d| {d.max():.3g}  mismatches {int((d > 0).sum())}/{d.size}"
            f"  (>1e-3: {int((d > 1e-3).sum())})"
        )

    gold = sorted(GOLD.glob("w*.npz"))[: a.windows]
    for p in gold[: a.stages]:
        print(p.name, "per stage (open loop)")
        check_stages(g, np.load(p), report)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess = ort.InferenceSession(
        str(WORK / "onnx/net_int8_qat.onnx"), so, providers=["CPUExecutionProvider"]
    )
    print("closed loop (kernels + int8 net on host ORT):")
    for p in gold:
        z = np.load(p)
        out, flt = closed_loop(g, z, sess)
        print(
            f"  {p.name}: psnr vs fp32 torch {psnr(out, z['out'][0]):.2f} dB, vs GT {psnr(out, z['gt'][0]):.2f}"
            f" (torch fp32 vs GT {psnr(z['out'][0], z['gt'][0]):.2f}); flow_t max|d|"
            f" {np.abs(flt - z['flow_t'][0]).max():.3g}"
        )


if __name__ == "__main__":
    main()
