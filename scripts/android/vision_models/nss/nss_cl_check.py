"""Run nss_kernels.cl on a host OpenCL device (pyopencl) against the torch golden dumps.

Per stage (open loop: every stage gets the golden inputs) and closed loop (the kernels' outputs become
the next frame's state; the CNN is the same int8-QAT ONNX model on host ORT, fed the kernels' uint8
input), reporting max abs error, uint8 mismatches and PSNR vs the golden output / the ground truth.
"""

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _psnr(a, b) -> float:
    d = np.clip(a, 0, 1).astype(np.float64) - np.clip(b, 0, 1).astype(np.float64)
    return float(10 * np.log10(1 / max(float((d * d).mean()), 1e-12)))


class NssCL:
    def __init__(self, ctx=None):
        import pyopencl as cl

        self.cl = cl
        if ctx is None:
            dev = None
            for p in cl.get_platforms():
                try:
                    ds = p.get_devices()
                except cl.Error:
                    continue
                if ds:
                    dev = ds[0]
                    break
            ctx = cl.Context([dev])
        self.ctx = ctx
        self.q = cl.CommandQueue(ctx)
        self.prg = cl.Program(ctx, (HERE / "nss_kernels.cl").read_text()).build(
            options=["-cl-fp32-correctly-rounded-divide-sqrt"]
        )

    def buf(self, a, write=False):
        mf = self.cl.mem_flags
        a = np.ascontiguousarray(a)
        return self.cl.Buffer(
            self.ctx,
            (mf.READ_WRITE if write else mf.READ_ONLY) | mf.COPY_HOST_PTR,
            hostbuf=a,
        )

    def out(self, shape, dtype):
        a = np.empty(shape, dtype)
        return a, self.cl.Buffer(self.ctx, self.cl.mem_flags.READ_WRITE, a.nbytes)

    def get(self, a, b):
        self.cl.enqueue_copy(self.q, a, b)
        return a

    def depth_scatter(self, z):
        _, _, H, W = z["depth"].shape
        Hd, Wd = H // 2, W // 2
        rec, rb = self.out((Hd, Wd), np.int32)
        self.prg.depth_scatter_init(self.q, (Hd * Wd,), None, rb)
        self.prg.depth_scatter(
            self.q,
            (Wd, Hd),
            None,
            self.buf(z["motion"]),
            self.buf(z["depth"]),
            np.int32(H),
            np.int32(W),
            rb,
            np.int32(Hd),
            np.int32(Wd),
        )
        return self.get(rec, rb)

    def preprocess(self, z, recon, feedback_tm1, derivative_tm1, history):
        _, _, H, W = z["colour"].shape
        Hp, Wp = feedback_tm1.shape[-2:]
        Hh, Wh = history.shape[-2:]
        Hd, Wd = recon.shape[-2:]
        cin, cb = self.out((12, Hp, Wp), np.float32)
        cu8, cub = self.out((Hp, Wp, 12), np.uint8)
        der, db = self.out((4, H, W), np.float32)
        dis, disb = self.out((H, W), np.float32)
        code, codeb = self.out((H, W), np.uint8)
        j = z["jitter"].ravel()
        rs = z["render_size"].ravel()
        dp = z["depth_params"].ravel().astype(np.float32)
        self.prg.preprocess(
            self.q,
            (Wp, Hp),
            None,
            self.buf(z["colour"]),
            self.buf(history),
            self.buf(z["motion"]),
            self.buf(z["depth"]),
            self.buf(feedback_tm1),
            self.buf(derivative_tm1),
            self.buf(recon.astype(np.int32)),
            np.int32(H),
            np.int32(W),
            np.int32(Hp),
            np.int32(Wp),
            np.int32(Hh),
            np.int32(Wh),
            np.int32(Hd),
            np.int32(Wd),
            np.float32(j[0]),
            np.float32(j[1]),
            np.float32(z["exposure"].ravel()[0]),
            np.float32(rs[0]),
            np.float32(rs[1]),
            np.array(dp, dtype=np.float32).view(self.cl.cltypes.float4)[0],
            cb,
            cub,
            db,
            disb,
            codeb,
        )
        return (
            self.get(cin, cb),
            self.get(cu8, cub),
            self.get(der, db),
            self.get(dis, disb),
            self.get(code, codeb),
        )

    def postprocess(self, z, history, code, kpn_u8, temporal_u8):
        _, _, H, W = z["colour"].shape
        Ho, Wo = history.shape[-2:]
        _, Hk, Wk, Kc = kpn_u8.shape
        _, Ht, Wt, _ = temporal_u8.shape
        lut = z["offset_lut"][0]  # (6, tiles, taps)
        taps = lut.shape[-1]
        mh, mw = (int(v) for v in z["idx_modulo"].ravel()[:2])
        lin, lb = self.out((3, Ho, Wo), np.float32)
        rgba, rgb = self.out((Ho, Wo, 4), np.uint8)
        self.prg.postprocess(
            self.q,
            (Wo, Ho),
            None,
            self.buf(z["colour"]),
            self.buf(history),
            self.buf(z["motion"]),
            self.buf(code),
            self.buf(kpn_u8),
            self.buf(temporal_u8),
            self.buf(lut.reshape(6, -1)),
            np.int32(H),
            np.int32(W),
            np.int32(Ho),
            np.int32(Wo),
            np.int32(Hk),
            np.int32(Wk),
            np.int32(Kc),
            np.int32(Ht),
            np.int32(Wt),
            np.int32(mh),
            np.int32(mw),
            np.int32(taps),
            np.float32(z["exposure"].ravel()[0]),
            np.float32(z["reset"].ravel()[0]),
            lb,
            rgb,
        )
        return self.get(lin, lb), self.get(rgba, rgb)


def _tm(lin, e):  # the golden's reinhard tonemap of the linear output
    x = np.maximum(lin * e, 0)
    return np.clip(x * (1.0 / (1.0 + x)), 0, 1)


def run(gold: Path, n: int) -> None:
    import onnxruntime as ort

    k = NssCL()
    print("device:", k.ctx.devices[0].name)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    cnn = ort.InferenceSession(
        str(gold.parent / "onnx" / "cnn_int8_qat.onnx"),
        so,
        providers=["CPUExecutionProvider"],
    )
    state = None
    for t in range(n):
        z = dict(np.load(gold / f"f{t:03d}.npz"))
        e = float(z["exposure"].ravel()[0])
        # --- open loop, per stage
        rec = k.depth_scatter(z)
        ds_mis = int((rec != z["recon_depth"][0, 0]).sum())
        cin, cu8, der, dis, code = k.preprocess(
            z,
            z["recon_depth"][0, 0],
            z["feedback_tm1"][0],
            z["derivative_tm1"][0],
            z["history"][0],
        )
        g_code = np.rint(z["nearest_offset"][0, 0] * 255).astype(np.uint8)
        lin, _ = k.postprocess(
            z, z["history"][0], g_code, z["kpn_u8"], z["temporal_u8"]
        )
        print(
            f"f{t:03d} open  | depth_scatter mism {ds_mis} | pre: cnn_in max {np.abs(cin - z['cnn_in'][0]).max():.2e}"
            f" u8 mism {int((cu8 != z['cnn_in_u8'][0]).sum())}/{cu8.size}"
            f" deriv max {np.abs(der - z['derivative'][0]).max():.2e}"
            f" disocc max {np.abs(dis - z['disocclusion'][0, 0]).max():.2e}"
            f" code mism {int((code != g_code).sum())}"
            f" | post: lin max {np.abs(lin - z['output_linear'][0]).max():.2e}"
            f" psnr vs golden {_psnr(_tm(lin, e), z['output'][0]):.1f} dB",
            flush=True,
        )
        # --- closed loop: our own state, host ORT CNN on our uint8 input
        if state is None:
            state = (z["history"][0], z["feedback_tm1"][0], z["derivative_tm1"][0])
        hist, fb, dtm1 = state
        rec_c = k.depth_scatter(z)
        _, cu8c, derc, _, codec = k.preprocess(z, rec_c, fb, dtm1, hist)
        kpn_u8, tmp_u8 = cnn.run(None, {"x": cu8c[None]})
        linc, _ = k.postprocess(z, hist, codec, kpn_u8, tmp_u8)
        state = (
            linc,
            (tmp_u8[0].astype(np.float32) / 255).transpose(2, 0, 1).copy(),
            derc,
        )
        print(
            f"f{t:03d} closed| psnr vs GT {_psnr(_tm(linc, e), z['ground_truth'][0]):.2f} dB"
            f" (golden {_psnr(z['output'][0], z['ground_truth'][0]):.2f})"
            f" vs golden output {_psnr(_tm(linc, e), z['output'][0]):.1f} dB",
            flush=True,
        )
