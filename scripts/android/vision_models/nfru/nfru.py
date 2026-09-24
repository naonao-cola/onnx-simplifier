"""Arm Neural Frame Rate Upscaling (NFRU v1) -- frame generation, the DLSS-3-style counterpart of NSS --
with the network on the HTP and the pre/post-processing on the Adreno GPU (see README.md).

  nfru.py fetch                  Arm's weights + license (HF Arm/neural-frame-rate-upscaling) and the test
                                 sequence (Arm/neural-graphics-dataset nfru/test, 2.9 GB) -> ~/.cache/arm-nfru
  nfru.py golden [--windows N]   the gym's torch pipeline (QAT weights, torch backend) window by window;
                                 dumps every GPU-kernel input, intermediate and output -> golden/
  nfru.py build                  the network -> int8 QDQ ONNX with uint8 NHWC I/O (onnx/net_int8_qat.onnx)
  nfru.py host-net               open loop: golden postprocess with the int8 network's logits (host ORT)
  nfru.py phone [--windows N]    nfru_run on the phone: OpenCL kernels on the Adreno + the int8 network on
                                 the HTP, closed loop from the rendered frames; PSNR vs torch and GT
(nfru_cl_check.py runs the same kernels on a host OpenCL device, stage by stage against golden/.)
"""

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WORK = Path(os.environ.get("NFRU_WORK", Path.home() / ".cache/arm-nfru"))
GOLD = WORK / "golden"
HF = "https://huggingface.co/{repo}/resolve/main/{path}"
FILES = {  # local name: (repo, path, sha256)
    "nfru_v1_fp32.pt": ("Arm/neural-frame-rate-upscaling", "nfru_v1_fp32.pt", None),
    "nfru_v1_int8.pt": ("Arm/neural-frame-rate-upscaling", "nfru_v1_int8.pt", None),
    "nfru_v1_int8_metadata.json": (
        "Arm/neural-frame-rate-upscaling",
        "nfru_v1_int8_metadata.json",
        None,
    ),
    "LICENSE_Arm_AI_Model_Community.pdf": (
        "Arm/neural-frame-rate-upscaling",
        "Arm_AI_Model_Community_License_v1_0_PRE-1154.pdf",
        None,
    ),
    "test.safetensors": (
        "datasets/Arm/neural-graphics-dataset",
        "nfru/test/0000.safetensors",
        None,
    ),
}
SHA_FILE = HERE / "sha256.json"
# the test windows: 60 fps capture, a 30 fps game -- interpolate t = n between m1 = n - 1 and p1 = n + 1
REF_FPS, CAPTURE_FPS, MIN_OFF, MAX_OFF = 60, 30, 3, 1
INPUTS = [
    "rgb_linear_m1",
    "rgb_linear_p1",
    "depth_m1",
    "depth_p1",
    "mv_p1_f30_m1",
    "sy_m1_f30_p1",
    "mv_m1_f30_m3",
    "sy_m1_f30_m3",
    "exposure_p1",
    "DepthParams_p1",
    "NearPlane_p1",
    "FarPlane_p1",
    "FovY_p1",
    "infinite_zFar_p1",
    "ViewProj_m3",
    "ViewProj_m1",
    "ViewProj_p1",
    "rgb_linear_t",
]


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def fetch() -> None:
    import json

    WORK.mkdir(parents=True, exist_ok=True)
    pinned = json.loads(SHA_FILE.read_text()) if SHA_FILE.exists() else {}
    got = {}
    for name, (repo, path, _) in FILES.items():
        dst = WORK / name
        if not dst.exists():
            print("fetch", name, flush=True)
            urllib.request.urlretrieve(HF.format(repo=repo, path=path), dst)
        got[name] = _sha(dst)
        if name in pinned and pinned[name] != got[name]:
            sys.exit(
                f"sha256 mismatch for {name}: {got[name]} != pinned {pinned[name]}"
            )
    if not pinned:
        SHA_FILE.write_text(json.dumps(got, indent=1) + "\n")
    print("ok:", ", ".join(got))


def windows(n: int):
    """The test windows as the gym's NFRU test loader builds them (process_data, no augmentation)."""
    import importlib

    import nfru_gym
    import safetensors
    import torch

    nfru_gym._install()
    naming = importlib.import_module("ng_model_gym.usecases.nfru.data.naming")
    proc = importlib.import_module("ng_model_gym.usecases.nfru.data.processing")
    step = REF_FPS // CAPTURE_FPS
    with safetensors.safe_open(
        WORK / "test.safetensors", framework="numpy", device="cpu"
    ) as f:
        length = int(f.metadata()["Length"])
        centres = list(range(MIN_OFF, length - MAX_OFF, step))[:n]
        for c in centres:
            start = c - MIN_OFF
            fr = {}
            for key in INPUTS:
                dv = naming.DataVariable(key)
                sk = dv.generate_non_concrete_variable(timeline_fps=REF_FPS)
                if dv.is_mv:
                    off = dv.ivec_from
                else:
                    ks = key.split("_")
                    off = naming.convert_str_offset_to_int(ks[-1])
                    sk = "_".join(ks[:-1])
                fr[key] = torch.from_numpy(
                    f.get_slice(sk)[start + MIN_OFF + off].copy()
                ).unsqueeze(0)
            x, y = proc.process_data(
                fr,
                augment=False,
                shape_aug=False,
                shape_aug_num_shapes=0,
                shape_aug_max_size=0,
                shape_aug_max_displacement=0,
                shape_aug_probability=0.0,
                brightness_shape_aug_probability=0.0,
            )
            yield (
                c,
                {
                    k: v.unsqueeze(0)
                    if v.dim() in (0, 2, 3) and k != "MotionMat"
                    else v
                    for k, v in x.items()
                },
                y,
            )


def _record_blockmatch():
    """Wrap the block matcher's modules for one _resolve_flow call; returns a getter of the per-level
    search (warped) / template / mv-hint images (uint8), vectors before the median, after the median,
    after the joint bilateral filter, and the hint mask -- keys bm_<what><level>."""
    import importlib

    bmm = importlib.import_module(
        "ng_model_gym.usecases.nfru.model.optical_flow.blockmatch_v321"
    )
    rec, lvl = {}, [-1]
    esw, et, cv, jbf, med = (
        bmm.ExtractSearchWindows.forward,
        bmm.ExtractTemplates.forward,
        bmm.CalculateVector.forward,
        bmm.JointBilateralFilter.forward,
        bmm.median_filter2d,
    )

    def esw_f(self, inputs, search_range):
        lvl[0] += 1
        rec[f"bm_search{lvl[0]}"] = inputs
        return esw(self, inputs, search_range)

    def et_f(self, inputs, dtype):
        import torch

        if (
            dtype == torch.uint8
        ):  # per level: [hint image (target level only), template image]
            rec.setdefault(f"_u8{lvl[0]}", []).append(inputs)
        return et(self, inputs, dtype)

    def cv_f(self, inputs):
        r = cv(self, inputs)
        rec[f"bm_vec{lvl[0]}"], rec[f"bm_hintmask{lvl[0]}"] = r[0], r[2]
        return r

    def med_f(x, *a, **k):
        rec[f"bm_premed{lvl[0]}"] = x
        y = med(x, *a, **k)
        rec[f"bm_med{lvl[0]}"] = y
        return y

    def jbf_f(self, inputs):
        y = jbf(self, inputs)
        rec[f"bm_jbf{lvl[0]}"] = y
        return y

    bmm.ExtractSearchWindows.forward, bmm.ExtractTemplates.forward = esw_f, et_f
    (
        bmm.CalculateVector.forward,
        bmm.JointBilateralFilter.forward,
        bmm.median_filter2d,
    ) = cv_f, jbf_f, med_f

    def done():
        bmm.ExtractSearchWindows.forward, bmm.ExtractTemplates.forward = esw, et
        (
            bmm.CalculateVector.forward,
            bmm.JointBilateralFilter.forward,
            bmm.median_filter2d,
        ) = cv, jbf, med
        for k in [k for k in rec if k.startswith("_u8")]:
            u8 = rec.pop(k)
            rec[f"bm_template{k[3:]}"] = u8[-1]
            if len(u8) == 2:
                rec[f"bm_hintimg{k[3:]}"] = u8[0]
        return rec

    return done


def golden(n: int) -> None:
    import importlib

    import nfru_gym
    import torch

    torch.set_grad_enabled(False)
    GOLD.mkdir(parents=True, exist_ok=True)
    core = nfru_gym.build_core("qat")
    gu = importlib.import_module("ng_model_gym.core.model.graphics_utils")
    for i, (c, x, y) in enumerate(windows(n)):
        x = {k: v.float() for k, v in x.items()}
        if x["MotionMat"].dim() == 3:
            x["MotionMat"] = x["MotionMat"].unsqueeze(0)
        for k in ("ViewProj_m3", "ViewProj_m1", "ViewProj_p1"):
            if x[k].dim() == 3:
                x[k] = x[k].unsqueeze(1)
        rgb_m1 = core.color_pipeline(x["rgb_linear_m1"], x, "m1")
        rgb_p1 = core.color_pipeline(x["rgb_linear_p1"], x, "p1")
        gt = core.color_pipeline(y.unsqueeze(0) if y.dim() == 3 else y, x, "t")
        depth_m1, depth_p1 = x["depth_m1"], x["depth_p1"]
        bm = _record_blockmatch()
        flow_raw = core._resolve_flow(x, rgb_m1, rgb_p1, depth_m1)
        bm_rec = bm()
        mm = x["MotionMat"]
        mv_p1 = gu.normalize_mvs(x["mv_p1_f30_m1"])
        mv_m1 = gu.normalize_mvs(x["mv_m1_f30_m3"])
        flow = gu.normalize_mvs(flow_raw)
        mm3 = x["ViewProj_m3"][:, 0] @ torch.linalg.inv(x["ViewProj_m1"][:, 0])
        dyn = core.previous_dynamic_mask(depth_m1, mv_m1, mm3)
        ts = 0.5
        mv_t, next_mask, holes_t, holes_tm1 = core.warp_mv(
            depth_m1,
            depth_p1,
            mv_p1,
            dyn,
            mm[:, 1],
            mm[:, 0],
            ts,
            1,
            list(depth_m1.shape[2:]),
        )
        flow_t = core.warp_flow(depth_m1, flow, 1.0 - ts, 1, list(flow.shape[2:]))
        seed = 12345 + c
        net_in = core.preprocess(
            flow_t_f30_xx=flow_t,
            mv_t_f30_m1=mv_t,
            rgb_m1=rgb_m1,
            rgb_p1=rgb_p1,
            depth_m1=depth_m1,
            depth_p1=depth_p1,
            depth_p1_warp_t=holes_t,
            depth_p1_warp_p1=holes_tm1,
            motion_mat_m1p1=mm[:, 1],
            motion_mat_p1m1=mm[:, 0],
            depth_params=x["DepthParams_p1"].reshape(1, 4, 1, 1),
            timestep=ts,
            random_seed=seed,
        )
        params = core.auto_encoder(net_in)
        out = core.postprocess(
            flow_t_f30_xx=flow_t,
            mv_t_f30_m1=mv_t,
            rgb_m1=rgb_m1,
            rgb_p1=rgb_p1,
            learnt_params=params,
            timestep=ts,
        )
        d = dict(
            rgb_m1=rgb_m1,
            rgb_p1=rgb_p1,
            gt=gt,
            depth_m1=depth_m1,
            depth_p1=depth_p1,
            sy_m1_f30_p1=x["sy_m1_f30_p1"],
            mv_p1_f30_m1=x["mv_p1_f30_m1"],
            mv_m1_f30_m3=x["mv_m1_f30_m3"],
            motion_mat=mm,
            motion_mat_m3=mm3,
            depth_params=x["DepthParams_p1"],
            flow_raw=flow_raw,
            flow=flow,
            dyn=dyn,
            mv_t=mv_t,
            holes_t=holes_t,
            holes_tm1=holes_tm1,
            flow_t=flow_t,
            net_in=net_in,
            params=params,
            out=out,
            seed=torch.tensor(seed),
            **bm_rec,
        )
        np.savez(
            GOLD / f"w{i:03d}.npz",
            **{k: v.detach().cpu().numpy() for k, v in d.items()},
        )
        mse = lambda a, b: float(((a.clamp(0, 1) - b.clamp(0, 1)) ** 2).mean())  # noqa: E731
        psnr = lambda a, b: 10 * np.log10(1 / max(mse(a, b), 1e-12))  # noqa: E731
        print(
            f"golden w{i:03d} (frame {c}): rgb {tuple(rgb_m1.shape)} depth {tuple(depth_m1.shape)} flow"
            f" {tuple(flow.shape)} net_in {tuple(net_in.shape)} | psnr vs GT {psnr(out, gt):.2f} dB"
            f" (m1 {psnr(rgb_m1, gt):.2f}, blend {psnr((rgb_m1 + rgb_p1) / 2, gt):.2f})",
            flush=True,
        )


def _out_range():
    """Arm's QAT output quantization (int8 scale s, zero point z) as a uint8 [lo, hi] range."""
    import json

    q = json.loads((WORK / "nfru_v1_int8_metadata.json").read_text())["outputs"][
        "getitem"
    ]["SINT"]
    s, z = float(q["scale"]), int(q["zero_point"]) + 128
    return (-z * s, (255 - z) * s), s, z


def build() -> None:
    """The network as ONNX (QAT weights, 16 x 270 x 480 in, 4 logits out) -> int8 QDQ (onnxsim full_qdq,
    input pinned to [0, 1], output to Arm's QAT output quantization) with uint8 NHWC I/O."""
    import nfru_gym
    import onnx
    import onnxruntime as ort
    import torch
    from onnx import helper

    from onnxsim.full_qdq import quantize_full_qdq, quantized_io

    torch.set_grad_enabled(False)
    gold = sorted(GOLD.glob("w*.npz"))
    if not gold:
        sys.exit("run `nfru.py golden` first")
    z0 = np.load(gold[0])
    _, c, h, w = z0["net_in"].shape
    d = WORK / "onnx"
    d.mkdir(exist_ok=True)
    ae = nfru_gym.build_core("qat").auto_encoder
    fp = d / "net_qat_f32.onnx"
    torch.onnx.export(
        ae,
        (torch.zeros(1, c, h, w),),
        str(fp),
        input_names=["x"],
        output_names=["params"],
        opset_version=17,
        dynamo=False,
    )
    xt = torch.from_numpy(z0["net_in"])
    s = ort.InferenceSession(str(fp), providers=["CPUExecutionProvider"])
    print(
        "onnx vs torch max abs",
        float(np.abs(s.run(None, {"x": xt.numpy()})[0] - ae(xt).numpy()).max()),
    )
    calib = [{"x": np.load(p)["net_in"]} for p in gold]
    (lo, hi), sc, zp = _out_range()
    q = quantize_full_qdq(
        onnx.load(fp), calib, method="mse", ranges={"x": (0.0, 1.0), "params": (lo, hi)}
    )
    m, info = quantized_io(q, nhwc_inputs=["x"])
    g = m.graph
    for o in list(g.output):  # uint8 NCHW -> NHWC
        prod = {oo: n for n in g.node for oo in n.output}[o.name]
        prod.output[:] = [
            o.name + "_nchw" if oo == o.name else oo for oo in prod.output
        ]
        g.node.append(
            helper.make_node(
                "Transpose", [o.name + "_nchw"], [o.name], perm=[0, 2, 3, 1]
            )
        )
        dims = [dd.dim_value for dd in o.type.tensor_type.shape.dim]
        o.type.tensor_type.shape.Clear()
        for v in (dims[0], dims[2], dims[3], dims[1]):
            o.type.tensor_type.shape.dim.add().dim_value = v
    print("I/O quantization:", info)
    assert (
        abs(info["params"]["scale"] - sc) < 1e-6 and info["params"]["zero_point"] == zp
    ), info
    onnx.save(m, d / "net_int8_qat.onnx")


def host_net() -> None:
    """Open loop on the host: every window's golden postprocess re-run with the int8 network's logits
    (host ORT, BASIC optimizations -- no fused int8 kernels); PSNR vs the fp32 torch output and vs GT."""
    import importlib

    import nfru_gym
    import onnxruntime as ort
    import torch

    nfru_gym._install()
    pp = importlib.import_module(
        "ng_model_gym.usecases.nfru.model.torch_processing.postprocess"
    )
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    s = ort.InferenceSession(
        str(WORK / "onnx" / "net_int8_qat.onnx"), so, providers=["CPUExecutionProvider"]
    )
    _, sc, zp = _out_range()

    def psnr(a, b):
        d = np.clip(a, 0, 1) - np.clip(b, 0, 1)
        return float(10 * np.log10(1 / max(float((d * d).mean()), 1e-12)))

    for p in sorted(GOLD.glob("w*.npz")):
        z = np.load(p)
        xu8 = np.clip(np.rint(z["net_in"][0].transpose(1, 2, 0) * 255), 0, 255).astype(
            np.uint8
        )[None]
        (pu8,) = s.run(None, {"x": xu8})
        params = (
            torch.from_numpy((pu8.astype(np.float32) - zp) * sc)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        t = {k: torch.from_numpy(z[k]) for k in ("flow_t", "mv_t", "rgb_m1", "rgb_p1")}
        out = pp.postprocess_torch(
            warped_flow=t["flow_t"],
            warped_mv=t["mv_t"],
            rgb_m1=t["rgb_m1"],
            rgb_p1=t["rgb_p1"],
            learnt_params=params,
            timestep=0.5,
        ).numpy()
        print(
            f"{p.stem}: int8 net psnr vs GT {psnr(out, z['gt']):.2f} dB (fp32 torch {psnr(z['out'], z['gt']):.2f});"
            f" vs fp32 output {psnr(out, z['out']):.1f} dB; logits max abs {np.abs(params.numpy() - z['params']).max():.2f}"
        )


SERIAL = os.environ.get("ANDROID_SERIAL", "239dbd8f")
REMOTE = os.environ.get("NFRU_REMOTE", "/data/local/tmp/codex-android-nfru-gpu")
LOCK = [str(Path.home() / ".cache/android-phone/phone-run")]
QNN = HERE.parent.parent / "htp_exploration" / "qnn_shell"


def _frames_for_phone(n: int, out: Path) -> None:
    """golden/ + the test sequence -> rendered frames fNNN.bin (float32 linear rgb, depth, motion in pixels,
    the sy hint; frame k is m1 of window k and p1 of window k - 1) + per-window wNNN.txt (motion matrices,
    depth params, seed)."""
    import nfru_cl_check as ck

    out.mkdir(parents=True, exist_ok=True)
    frames = {}  # k -> dict

    def put(k, key, v):
        d = frames.setdefault(k, {})
        if key in d:
            assert np.array_equal(d[key], v), (k, key)
        d[key] = np.ascontiguousarray(v, np.float32)

    for w in range(n):
        z = np.load(GOLD / f"w{w:03d}.npz")
        c = int(z["seed"]) - ck.SEED0
        put(w, "lin", ck.linear_rgb(c - 1)[0])
        put(w + 1, "lin", ck.linear_rgb(c + 1)[0])
        put(w, "depth", z["depth_m1"][0, 0])
        put(w + 1, "depth", z["depth_p1"][0, 0])
        put(w, "mv", z["mv_m1_f30_m3"][0])
        put(w + 1, "mv", z["mv_p1_f30_m1"][0])
        put(w, "sy", z["sy_m1_f30_p1"][0])
        mm = z["motion_mat"][0]
        v = [
            *mm[0].ravel(),
            *mm[1].ravel(),
            *z["motion_mat_m3"][0].ravel(),
            *z["depth_params"].ravel(),
        ]
        (out / f"w{w:03d}.txt").write_text(
            " ".join(repr(float(x)) for x in v) + f" {int(z['seed'])}\n"
        )
    for k, d in frames.items():
        sy = d.get("sy", np.zeros_like(d["mv"]))
        with open(out / f"f{k:03d}.bin", "wb") as f:
            for a in (d["lin"], d["depth"], d["mv"], sy):
                f.write(a.tobytes())


def phone(n: int, iters: int) -> None:
    """Build nfru_run, push it with the kernels, the int8 network, the QNN/ORT libs and n windows' frames,
    run it on the phone (under the phone lock) and compare the generated frames with torch and GT."""
    import subprocess

    import nfru_cl_check as ck

    build = HERE / "build"
    subprocess.run(
        [str(HERE / "build_gpu.sh")], check=True, env={**os.environ, "OUT": str(build)}
    )
    stage = WORK / "phone_gpu"
    _frames_for_phone(n, stage)
    adb = " ".join(["adb", "-s", SERIAL])
    env = {**os.environ, "PHONE_LOCK_OWNER": "codex/android-nfru-gpu"}
    files = [
        build / "nfru_run",
        HERE / "nfru_kernels.cl",
        WORK / "onnx" / "net_int8_qat.onnx",
    ]
    files += sorted((QNN / "libs").glob("*.so"))
    files += [stage / f"f{k:03d}.bin" for k in range(n + 1)] + [
        stage / f"w{w:03d}.txt" for w in range(n)
    ]
    push = " && ".join(f"{adb} push -q {f} {REMOTE}/" for f in files)
    run = (
        f"cd {REMOTE} && {os.environ.get('NFRU_ENV', '')} LD_LIBRARY_PATH={REMOTE} ADSP_LIBRARY_PATH='{REMOTE};"
        f"/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp' ./nfru_run . net_int8_qat.onnx"
        f" net_ctx.onnx {n} {iters}"
    )
    pull = " && ".join(
        f"{adb} pull -q {REMOTE}/out{w:03d}.bin {stage}/" for w in range(n)
    )
    cmd = f'{adb} shell mkdir -p {REMOTE} && {push} && {adb} shell "{run}" && {pull}'
    r = subprocess.run(
        LOCK + ["bash", "-c", cmd], env=env, capture_output=True, text=True
    )
    print(r.stdout)
    if r.returncode:
        sys.exit(r.stderr[-3000:])
    for w in range(n):
        z = np.load(GOLD / f"w{w:03d}.npz")
        o = np.fromfile(stage / f"out{w:03d}.bin", np.uint8).reshape(1080, 1920, 4)[
            ..., :3
        ]
        o = o.transpose(2, 0, 1).astype(np.float32) / 255
        print(
            f"w{w:03d} phone psnr vs fp32 torch {ck.psnr(o, z['out'][0]):.2f} dB (RGBA8), vs GT"
            f" {ck.psnr(o, z['gt'][0]):.2f} (torch fp32 {ck.psnr(z['out'][0], z['gt'][0]):.2f})"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "golden", "build", "host-net", "phone"])
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--iters", type=int, default=5)
    a = ap.parse_args()
    {
        "fetch": fetch,
        "golden": lambda: golden(a.windows),
        "build": build,
        "host-net": host_net,
        "phone": lambda: phone(a.windows, a.iters),
    }[a.cmd]()


if __name__ == "__main__":
    main()
