"""Run nss_kernels.cl on a host OpenCL device (pyopencl) against the torch golden dumps.

Per stage (open loop: every stage gets the golden inputs) and closed loop (the kernels' outputs become
the next frame's state; the CNN is the same int8-QAT ONNX model on host ORT, fed the kernels' uint8
input), reporting max abs error, uint8 mismatches and PSNR vs the golden output / the ground truth.
"""

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def rgba(a):  # (C, H, W) planar -> (H, W, 4) float32 interleaved (float4 buffers)
    c, h, w = a.shape
    o = np.zeros((h, w, 4), np.float32)
    o[..., :c] = a.transpose(1, 2, 0)
    return o


def yx2(a):  # (2, H, W) -> (H, W, 2)
    return np.ascontiguousarray(a.transpose(1, 2, 0), np.float32)


def fb_u8(
    a,
):  # golden float feedback_tm1 (4, H, W) = k / 255 -> the HTP's uint8 NHWC temporal output
    return np.ascontiguousarray(np.rint(a.transpose(1, 2, 0) * 255).astype(np.uint8))


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

    def img(
        self, a4, write=False
    ):  # (H, W, 4) float32 -> RGBA32F image2d (None -> uninitialized)
        cl = self.cl
        fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT)
        mf = cl.mem_flags
        if write:
            return cl.Image(self.ctx, mf.READ_WRITE, fmt, shape=(a4[1], a4[0]))
        a4 = np.ascontiguousarray(a4, np.float32)
        return cl.Image(
            self.ctx,
            mf.READ_ONLY | mf.COPY_HOST_PTR,
            fmt,
            shape=(a4.shape[1], a4.shape[0]),
            hostbuf=a4,
        )

    def img_u8(self, a):  # (H, W, 4) uint8 -> RGBA8 unsigned-int image2d
        cl = self.cl
        fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNSIGNED_INT8)
        a = np.ascontiguousarray(a, np.uint8)
        return cl.Image(
            self.ctx,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            fmt,
            shape=(a.shape[1], a.shape[0]),
            hostbuf=a,
        )

    def get_img(self, a, b):
        self.cl.enqueue_copy(
            self.q, a, b, origin=(0, 0), region=(a.shape[1], a.shape[0])
        )
        return a

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
            self.buf(yx2(z["motion"][0])),
            self.buf(z["depth"]),
            np.int32(H),
            np.int32(W),
            rb,
            np.int32(Hd),
            np.int32(Wd),
        )
        return self.get(rec, rb)

    def preprocess(self, z, recon, feedback_u8, derivative_tm1, history):
        _, _, H, W = z["colour"].shape
        Hp, Wp = feedback_u8.shape[:2]
        Hh, Wh = history.shape[:2]
        Hd, Wd = recon.shape[-2:]
        cin, cb = self.out((12, Hp, Wp), np.float32)
        cu8, cub = self.out((Hp, Wp, 12), np.uint8)
        der = np.empty((H, W, 4), np.float32)
        db = self.img((H, W), write=True)
        dis, disb = self.out((H, W), np.float32)
        code, codeb = self.out((H, W), np.uint8)
        j = z["jitter"].ravel()
        rs = z["render_size"].ravel()
        dp = z["depth_params"].ravel().astype(np.float32)
        self.prg.preprocess(
            self.q,
            (Wp, Hp),
            (32, 8),
            self.img(rgba(z["colour"][0])),
            self.img(history),
            self.buf(yx2(z["motion"][0])),
            self.buf(z["depth"]),
            self.img_u8(feedback_u8),
            self.img(derivative_tm1),
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
            self.get_img(der, db),
            self.get(dis, disb),
            self.get(code, codeb),
        )

    def postprocess(self, z, history, code, kpn_u8, temporal_u8):
        _, _, H, W = z["colour"].shape
        Ho, Wo = history.shape[:2]
        _, Hk, Wk, Kc = kpn_u8.shape
        _, Ht, Wt, _ = temporal_u8.shape
        lut = z["offset_lut"][0]  # (6, tiles, taps)
        taps = lut.shape[-1]
        mh, mw = (int(v) for v in z["idx_modulo"].ravel()[:2])
        lin = np.empty((Ho, Wo, 4), np.float32)
        lb = self.img((Ho, Wo), write=True)
        rgba_, rgb = self.out((Ho, Wo, 4), np.uint8)
        self.prg.postprocess(
            self.q,
            (Wo, Ho),
            (32, 8),
            self.img(rgba(z["colour"][0])),
            self.img(history),
            self.buf(yx2(z["motion"][0])),
            self.buf(code),
            self.buf(kpn_u8),
            self.img_u8(temporal_u8[0]),
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
        self.cl.enqueue_copy(self.q, lin, lb, origin=(0, 0), region=(Wo, Ho))
        return lin[..., :3].transpose(2, 0, 1), self.get(rgba_, rgb)


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
            fb_u8(z["feedback_tm1"][0]),
            rgba(z["derivative_tm1"][0]),
            rgba(z["history"][0]),
        )
        der = der.transpose(2, 0, 1)
        g_code = np.rint(z["nearest_offset"][0, 0] * 255).astype(np.uint8)
        lin, _ = k.postprocess(
            z, rgba(z["history"][0]), g_code, z["kpn_u8"], z["temporal_u8"]
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
            state = (
                rgba(z["history"][0]),
                fb_u8(z["feedback_tm1"][0]),
                rgba(z["derivative_tm1"][0]),
            )
        hist, fb, dtm1 = state
        rec_c = k.depth_scatter(z)
        _, cu8c, derc, _, codec = k.preprocess(z, rec_c, fb, dtm1, hist)
        kpn_u8, tmp_u8 = cnn.run(None, {"x": cu8c[None]})
        linc, _ = k.postprocess(z, hist, codec, kpn_u8, tmp_u8)
        state = (rgba(linc), np.ascontiguousarray(tmp_u8[0]), derc)
        print(
            f"f{t:03d} closed| psnr vs GT {_psnr(_tm(linc, e), z['ground_truth'][0]):.2f} dB"
            f" (golden {_psnr(z['output'][0], z['ground_truth'][0]):.2f})"
            f" vs golden output {_psnr(_tm(linc, e), z['output'][0]):.1f} dB",
            flush=True,
        )
