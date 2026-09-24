"""Two NSS stages written in tinygrad, rendered by tinygrad's OpenCL backend, compared with the
hand-written twins in tg_compare.cl -- on the host OpenCL device and (via cl_bench) on the Adreno.

  tg_nss.py host    tinygrad (DEV=CL) vs hw kernels on the host GPU: max abs diff; dumps the generated
                    kernels + their launch dims + the inputs/outputs to WORK/tg/
  tg_nss.py phone   runs both on the phone with cl_bench: median kernel ms, outputs diffed vs the host

Needs TINYGRAD_PATH (the onnxsim/tinygrad fork; its OpenCL backend is upstream's).
Stages (what tinygrad can express without data-dependent gathers, which it lowers to one-hot compares):
  accum  postprocess tail at 1080p: rectify, Karis blend, inverse tonemap, display tonemap (elementwise)
  deriv  preprocess YCoCg derivative / instability at 540p: a +-1 replicate-pad stencil + a state machine
"""

import ctypes
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("NSS_WORK", Path.home() / ".cache/arm-nss")) / "tg"
EPS = 1e-7


def _inputs(seed=0):
    r = np.random.default_rng(seed)
    H, W, h, w = 1080, 1920, 540, 960
    f = lambda *s, lo=0.0, hi=1.0: r.uniform(lo, hi, s).astype(np.float32)  # noqa: E731
    m1 = f(3, H, W, hi=4.0)
    acc = {
        "m1": m1,
        "m2": (m1 * m1 + f(3, H, W, hi=0.5)).astype(np.float32),
        "wh": f(3, H, W, lo=-0.2, hi=5.0),
        "cc": f(3, H, W, hi=5.0),
        "theta": f(H, W),
        "alpha": f(H, W, lo=0.05, hi=0.4),
        "gamma": f(H, W, hi=2.0),
        "onscreen": (f(H, W) > 0.05).astype(np.float32),
        "cv": (f(H, W) > 0.02).astype(np.float32),
    }
    der = {
        "col": f(3, h, w, hi=3.0),
        "dt": np.concatenate([f(3, h, w, lo=-0.5, hi=1.5), f(1, h, w, hi=0.5)]),
        "dis": f(h, w, hi=0.03),
    }
    return acc, der, 7.389056, 1.0


def _lerp(a, b, w):
    from tinygrad import Tensor

    if not isinstance(w, Tensor):  # scalar weight: torch picks the branch once
        return a + w * (b - a) if abs(w) < 0.5 else b - (b - a) * (1.0 - w)
    a = a if isinstance(a, Tensor) else Tensor(a)
    b = b if isinstance(b, Tensor) else Tensor(b)
    return (w.abs() < 0.5).where(a + w * (b - a), b - (b - a) * (1.0 - w))


def _sat(x):
    return x.clip(0.0, 1.0)


def tg_accum(t, e, reset):
    m1, m2, wh, cc = t["m1"], t["m2"], t["wh"], t["cc"]
    var = (m2 - m1 * m1).abs().maximum(EPS)
    sigma = var.sqrt() * t["gamma"]
    hcl = wh.minimum(m1 + sigma).maximum(m1 - sigma)
    rect = _lerp(_lerp(m1, hcl, reset), wh, t["theta"] * t["onscreen"] * reset)

    def karis(x):
        n = x.maximum(0.0)
        return _sat(n * (1.0 / (1.0 + n.max(axis=0))))

    a = t["alpha"] * t["cv"] * reset
    acc = _lerp(karis(rect), karis(cc), a).clip(0.0, 1.0 - EPS)
    cl = acc.maximum(0.0).minimum(65504.0 * (1.0 / (1.0 + 65504.0)))
    lin = cl * (1.0 / (1.0 - cl.max(axis=0))) * (1.0 / e)
    x = (lin * e).maximum(0.0)
    return lin, _sat(x * (1.0 / (1.0 + x)))


def tg_deriv(t, e):
    col, dt, dis = t["col"], t["dt"], t["dis"]
    c = (col * e).maximum(0.0).sqrt()
    co = c[0] - c[2]
    tmp = c[2] + co * 0.5
    cg = c[1] - tmp
    y = (tmp + cg * 0.5).stack(co, cg)  # (3, h, w)
    yp = y.pad(((0, 0), (1, 1), (1, 1)), mode="replicate")
    h, w = col.shape[1:]
    sh = lambda dy, dx: yp[:, 1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]  # noqa: E731

    def delta(a, b):
        d = a - b
        return (
            d[0] * d[0] + (d[1] * 1.25) * (d[1] * 1.25) + (d[2] * 1.25) * (d[2] * 1.25)
        ).sqrt()

    d_c = delta(y, dt[:3])
    d_n, d_s, d_e, d_w = (
        delta(y, sh(0, -1)),
        delta(y, sh(0, 1)),
        delta(y, sh(1, 0)),
        delta(y, sh(-1, 0)),
    )
    s_sum = d_n + d_s + d_e + d_w
    s_max = d_n.maximum(d_s).maximum(d_e.maximum(d_w))
    prev = dt[3]
    sup = _lerp(d_c, (s_sum - s_max).maximum(0.0) * 0.3333333432674408, 0.3) * 0.75
    recall = _sat((sup - 0.065) * 2.816901445388794)
    exc = _sat(((sup - prev).maximum(0.0) - 0.025) * 7.407407283782959)
    mean_gate = _sat((sup - 0.07) * 6.25)
    raw_entry = recall.sqrt() * exc.sqrt() * mean_gate
    heat = _sat((prev - 0.11) * 10.0)
    s_floor, s_ceil = _lerp(0.177, 0.157, heat), _lerp(0.33, 0.305, heat)
    s_gate = _sat((_lerp(prev, sup, 0.3) - s_floor) * (1.0 / (s_ceil - s_floor)))
    s_gate = s_gate * s_gate
    hot = _sat((prev - 0.18) * 10.0)
    hot = hot * hot
    raw_inst = raw_entry.maximum(prev * s_gate.maximum(hot * 0.12) * 0.8)
    fall = _lerp(0.24, 0.05, s_gate.maximum(heat * heat * 0.5))
    rise = _lerp(0.08, 0.22, (recall * mean_gate).sqrt())
    filt = _lerp(prev, raw_inst, (raw_inst > prev).where(rise, fall))
    vis = _lerp(prev, filt, (filt > prev).where(0.75, 0.8))
    disb = (dis > 0.01).where(1.0, 0.0)
    uninit = ((dt[0].abs() + dt[1].abs() + dt[2].abs() + prev.abs()) < 0.0001).where(
        1.0, 0.0
    )
    vis = vis * (1.0 - disb)
    st = y.cat(filt.unsqueeze(0))
    rs = y.cat((filt * 0).unsqueeze(0))
    state = _lerp(_lerp(st, rs, disb), rs, uninit)
    return state, _lerp(vis, 0.0, uninit)


def _capture(fn):
    """Realize fn()'s outputs on DEV=CL, recording every launched kernel (source, launch dims, buffers)."""
    from tinygrad.runtime import ops_cl

    rec, srcs, sizes = [], {}, {}
    init0, call0 = ops_cl.CLProgram.__init__, ops_cl.CLProgram.__call__

    def init(self, device, obj):
        init0(self, device, obj)
        self._tg_name, self._tg_src = (
            obj.name,
            obj.lib.decode(),
        )  # names repeat across different kernels

    def call(self, *bufs, global_size=(1, 1, 1), local_size=None, vals=(), **kw):
        gs = (
            tuple(int(g * lz) for g, lz in zip(global_size, local_size))
            if local_size
            else tuple(global_size)
        )
        for b in bufs:
            sz = ctypes.c_size_t()
            ops_cl.cl.clGetMemObjectInfo(
                b, ops_cl.cl.CL_MEM_SIZE, ctypes.sizeof(sz), ctypes.byref(sz), None
            )
            sizes[_addr(b)] = sz.value
        rec.append(
            (
                self._tg_name,
                gs,
                tuple(local_size) if local_size else None,
                [_addr(b) for b in bufs],
                vals,
            )
        )
        srcs[len(rec) - 1] = self._tg_src
        return call0(
            self, *bufs, global_size=global_size, local_size=local_size, vals=vals, **kw
        )

    ops_cl.CLProgram.__init__, ops_cl.CLProgram.__call__ = init, call
    try:
        outs = fn()
    finally:
        ops_cl.CLProgram.__init__, ops_cl.CLProgram.__call__ = init0, call0
    return outs, rec, srcs, sizes


def _addr(p):  # a ctypes cl_mem (struct pointer) -> its address
    return ctypes.cast(p, ctypes.c_void_p).value


def _handle(t):
    return _addr(t.uop.base.buffer._buf)


def host() -> None:
    sys.path.insert(0, os.environ["TINYGRAD_PATH"])
    import nss_cl_check
    from tinygrad import Tensor

    WORK.mkdir(parents=True, exist_ok=True)
    acc_np, der_np, e, reset = _inputs()
    ck = nss_cl_check.NssCL()
    prg = ck.cl.Program(ck.ctx, (HERE / "tg_compare.cl").read_text()).build()
    report = []
    for stage, arrs, fn, outs_spec in (
        (
            "accum",
            acc_np,
            lambda t: tg_accum(t, e, reset),
            (("lin", (3, 1080, 1920)), ("disp", (3, 1080, 1920))),
        ),
        (
            "deriv",
            der_np,
            lambda t: tg_deriv(t, e),
            (("state", (4, 540, 960)), ("vis", (540, 960))),
        ),
    ):
        ts = {k: Tensor(v, device="CL").realize() for k, v in arrs.items()}
        (o1, o2), rec, srcs, sizes = _capture(
            lambda: tuple(x.realize() for x in fn(ts))
        )
        names = {_handle(v): f"in_{k}" for k, v in ts.items()} | {
            _handle(o1): "out_0",
            _handle(o2): "out_1",
        }
        tg_out = (o1.numpy(), o2.numpy())
        d = WORK / stage
        d.mkdir(exist_ok=True)
        for k, v in arrs.items():
            np.ascontiguousarray(v, np.float32).tofile(d / f"in_{k}.bin")
        lines, seen = [], set()
        for i, (name, gs, ls, bufs, vals) in enumerate(rec):
            (d / f"k{i}.cl").write_text(srcs[i])
            for b in bufs:
                nm = names.get(b, f"tmp_{b}")
                if nm in seen:
                    continue
                seen.add(nm)
                src = f" {stage}/{nm}.bin" if nm.startswith("in_") else ""
                lines.append(f"BUF {nm} {sizes[b]}{src}")
            args = [names.get(b, f"tmp_{b}") for b in bufs] + [f"i:{v}" for v in vals]
            loc = ",".join(map(str, ls)) if ls else "-"
            lines.append(
                f"K {stage}/k{i}.cl {name} {','.join(map(str, gs))} {loc} {' '.join(args)}"
            )
        lines += [f"OUT out_0 {stage}/ptg_0.bin", f"OUT out_1 {stage}/ptg_1.bin"]
        (d / "tg.plan").write_text("\n".join(lines) + "\n")
        # hand-written twin on the host device
        mf, q = ck.cl.mem_flags, ck.q
        if stage == "accum":
            n = 1080 * 1920
            ob = [ck.cl.Buffer(ck.ctx, mf.READ_WRITE, 3 * n * 4) for _ in range(2)]
            prg.hw_accum(
                q,
                (n,),
                (256,),
                *[
                    ck.buf(arrs[k])
                    for k in (
                        "m1",
                        "m2",
                        "wh",
                        "cc",
                        "theta",
                        "alpha",
                        "gamma",
                        "onscreen",
                        "cv",
                    )
                ],
                np.float32(e),
                np.float32(reset),
                np.int32(n),
                *ob,
            )
            hw = [np.empty((3, 1080, 1920), np.float32) for _ in range(2)]
        else:
            n = 540 * 960
            ob = [
                ck.cl.Buffer(ck.ctx, mf.READ_WRITE, 4 * n * 4),
                ck.cl.Buffer(ck.ctx, mf.READ_WRITE, n * 4),
            ]
            prg.hw_deriv(
                q,
                (n,),
                (256,),
                ck.buf(arrs["col"]),
                ck.buf(arrs["dt"]),
                ck.buf(arrs["dis"]),
                np.float32(e),
                np.int32(540),
                np.int32(960),
                *ob,
            )
            hw = [np.empty((4, 540, 960), np.float32), np.empty((540, 960), np.float32)]
        for a, b in zip(hw, ob):
            ck.cl.enqueue_copy(q, a, b)
        if stage == "accum":
            n = 1080 * 1920
            ins = ["m1", "m2", "wh", "cc", "theta", "alpha", "gamma", "onscreen", "cv"]
            hl = [f"BUF in_{k} {arrs[k].nbytes} {stage}/in_{k}.bin" for k in ins]
            hl += [f"BUF o0 {3 * n * 4}", f"BUF o1 {3 * n * 4}"]
            hl.append(
                f"K tg_compare.cl hw_accum {n},1,1 256,1,1 "
                + " ".join(f"in_{k}" for k in ins)
                + f" f:{e} f:{reset} i:{n} o0 o1"
            )
        else:
            n = 540 * 960
            hl = [
                f"BUF in_{k} {arrs[k].nbytes} {stage}/in_{k}.bin"
                for k in ("col", "dt", "dis")
            ]
            hl += [f"BUF o0 {4 * n * 4}", f"BUF o1 {n * 4}"]
            hl.append(
                f"K tg_compare.cl hw_deriv {n},1,1 256,1,1 in_col in_dt in_dis f:{e} i:540 i:960 o0 o1"
            )
        hl += [f"OUT o0 {stage}/phw_0.bin", f"OUT o1 {stage}/phw_1.bin"]
        (d / "hw.plan").write_text("\n".join(hl) + "\n")
        for (on, _), a, b in zip(outs_spec, tg_out, hw):
            b.tofile(d / f"hw_{on}.bin")
            np.ascontiguousarray(a, np.float32).tofile(d / f"tg_{on}.bin")
        diffs = [float(np.abs(a - b).max()) for a, b in zip(tg_out, hw)]
        report.append(
            f"{stage}: {len(rec)} tinygrad kernel(s) {[r[0] for r in rec]}, max |tg - hw| {diffs}"
        )
    print("\n".join(report))


def phone() -> None:
    serial = os.environ.get("ANDROID_SERIAL", "239dbd8f")
    remote = (
        os.environ.get("NSS_REMOTE", "/data/local/tmp/codex-android-nss-gpu") + "/tg"
    )
    adb = " ".join(["adb", "-s", serial])
    lock = [str(Path.home() / ".cache/android-phone/phone-run")]
    env = {**os.environ, "PHONE_LOCK_OWNER": "codex/android-nss-gpu"}
    subprocess.run(
        [str(HERE / "build_gpu.sh")],
        check=True,
        env={**env, "OUT": str(HERE / "build")},
    )
    cmds = [
        f"{adb} shell mkdir -p {remote}",
        f"{adb} push -q {HERE / 'build' / 'cl_bench'} {HERE / 'tg_compare.cl'} {remote}/",
    ]
    cmds += [f"{adb} push -q {WORK / st} {remote}/" for st in ("accum", "deriv")]
    runs = [
        f"./cl_bench --plan {st}/{p}.plan 50"
        for st in ("accum", "deriv")
        for p in ("tg", "hw")
    ]
    cmds.append(f'{adb} shell "cd {remote} && ' + " && ".join(runs) + '"')
    cmds += [
        f"{adb} pull -q {remote}/{st}/{pre}_{k}.bin {WORK / st}/"
        for st in ("accum", "deriv")
        for pre in ("ptg", "phw")
        for k in (0, 1)
    ]
    r = subprocess.run(
        lock + ["bash", "-c", " && ".join(cmds)],
        env=env,
        capture_output=True,
        text=True,
    )
    print(r.stdout)
    if r.returncode:
        sys.exit(r.stderr[-3000:])
    for stage, outs in (("accum", ("lin", "disp")), ("deriv", ("state", "vis"))):
        d = WORK / stage
        for k, on in enumerate(outs):
            tg = np.fromfile(d / f"ptg_{k}.bin", np.float32)
            hw = np.fromfile(d / f"phw_{k}.bin", np.float32)
            ref = np.fromfile(d / f"hw_{on}.bin", np.float32)
            print(
                f"{stage}.{on}: phone tg vs phone hw max {np.abs(tg - hw).max():.2e};"
                f" phone hw vs host hw max {np.abs(hw - ref).max():.2e}"
            )


if __name__ == "__main__":
    {"host": host, "phone": phone}[sys.argv[1]]()
