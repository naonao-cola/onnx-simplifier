#!/usr/bin/env python3
"""Arm Neural Super Sampling (NSS v1) -- an open, mobile-targeted temporal super sampler (the DLSS 2
counterpart) -- on the Xiaomi 12S: its parameter-prediction CNN on the HTP, validated end to end on
Arm's own test sequence. See README.md.

  nss.py fetch                     weights + test sequence (sha256-pinned) and the model-gym source
  nss.py host   [--frames N]       the full torch pipeline (gym preprocess -> CNN -> postprocess) on the
                                   test sequence, recurrent: PSNR vs ground truth for NSS fp32, bicubic,
                                   and NSS with the int8 CNN (host ORT, same QDQ graph); saves the CNN
                                   inputs/outputs of the fp32 run
  nss.py build                     the CNN as ONNX (fp32 check vs torch), fp16 and int8 variants with
                                   uint8 NHWC I/O (int8: onnxsim full_qdq calibrated on the saved inputs)
  nss.py phone                     both CNN variants on the phone (strict all-HTP, per-frame inputs of the
                                   fp32 run): ms, and the full pipeline with the phone's CNN outputs
  nss.py report                    README tables from $NSS_WORK/results.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("NSS_WORK", Path.home() / ".cache/arm-nss"))
HF = "https://huggingface.co/{repo}/resolve/{rev}/{path}"
# (repo, revision, path, sha256 prefix)
FILES = {
    "nss_v1_0_1_high_fp32.pt": (
        "Arm/neural-super-sampling",
        "main",
        "nss_v1_0_1_high_fp32.pt",
        None,
    ),
    "nss_v1_0_1_high_int8.pt": (
        "Arm/neural-super-sampling",
        "main",
        "nss_v1_0_1_high_int8.pt",
        None,
    ),
    "LICENSE_Arm_AI_Model_Community.pdf": (
        "Arm/neural-super-sampling",
        "main",
        "Arm_AI_Model_Community_License_v1_0_PRE-1154.pdf",
        None,
    ),
    "test_sample.safetensors": (
        "datasets/Arm/neural-graphics-dataset",
        "main",
        "nss/test/test_full_resolution_sample.safetensors",
        None,
    ),
}
SHA = (
    json.loads((HERE / "sha256.json").read_text())
    if (HERE / "sha256.json").exists()
    else {}
)
RESULTS = WORK / "results.json"


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def fetch() -> None:
    import nss_gym

    WORK.mkdir(parents=True, exist_ok=True)
    for name, (repo, rev, path, _) in FILES.items():
        dst = WORK / name
        if not dst.exists():
            print("download", name)
            urllib.request.urlretrieve(HF.format(repo=repo, rev=rev, path=path), dst)
        got = _sha(dst)
        if name in SHA and SHA[name] != got:
            sys.exit(f"{name}: sha256 {got} != pinned {SHA[name]}")
        print(name, got[:16])
    if not (nss_gym.GYM / "src").exists():
        subprocess.run(
            [
                "git",
                "clone",
                "-q",
                "https://github.com/arm/neural-graphics-model-gym.git",
                str(nss_gym.GYM),
            ],
            check=True,
        )
    subprocess.run(
        ["git", "-C", str(nss_gym.GYM), "checkout", "-q", nss_gym.GYM_COMMIT],
        check=True,
    )


# ---------------------------------------------------------------- data + the recurrent pipeline
def _frames(n: int):
    """The test sequence frame by frame, as the gym's test loader prepares it (T = 1 windows)."""
    import nss_gym
    import safetensors
    import torch
    import torch.nn.functional as F

    keys = [
        "colour_linear",
        "ground_truth_linear",
        "motion",
        "depth",
        "depth_params",
        "exposure",
        "jitter",
        "render_size",
        "zNear",
        "zFar",
        "motion_lr",
    ]
    with safetensors.safe_open(
        WORK / "test_sample.safetensors", framework="pt", device="cpu"
    ) as f:
        length = int(f.metadata()["Length"])
        for t in range(min(n, length)):
            fr = {k: f.get_slice(k)[t : t + 1].clone() for k in f.keys() if k in keys}
            if "motion_lr" not in fr:
                fr["motion_lr"] = (
                    F.interpolate(fr["motion"], scale_factor=0.5, mode="nearest") * 0.5
                )
            if "exposure" not in fr:
                fr["exposure"] = torch.zeros((1, 1))
            x, y = nss_gym.process_frame(fr)
            x = {k: v.unsqueeze(0) for k, v in x.items()}  # batch
            x["seq"] = torch.ones(
                (1, 1, 1, 1, 1)
            )  # one sequence: history carries over every frame
            yield x, y.unsqueeze(0)


def _step(model, x, cnn=None):
    """One NSSV1Model.forward frame (core_forward + buffer update) with a replaceable CNN.
    Returns (tonemapped output, tonemapped GT, CNN input, (kpn, temporal))."""
    inputs = model.set_buffers({k: v[:, 0] for k, v in x.items()})
    inp, deriv, disocc, ndo = model.preprocess(inputs)
    kpn, temporal = model.autoencoder(inp) if cnn is None else cnn(inp)
    out = model.postprocess(kpn, inputs, temporal, ndo, deriv, disocc)
    out.pop("motion", None)
    out["reset_event"] = inputs["reset_event"]
    model.update_buffers(inputs, out)
    return out["output"], out["ground_truth"], inp, (kpn, temporal)


def _model(weights: str = "fp32"):
    import nss_gym
    import torch

    m = nss_gym.build("high")
    if weights == "fp32":
        sd = torch.load(WORK / "nss_v1_0_1_high_fp32.pt", map_location="cpu")[
            "model_state_dict"
        ]
    else:  # Arm's QAT checkpoint: the same layers as graph constants, in graph order
        q = torch.load(WORK / "nss_v1_0_1_high_int8.pt", map_location="cpu")[
            "model_state_dict"
        ]
        order = [
            "conv2d_0",
            "conv2d_1",
            "conv2d_2",
            "conv2d_3",
            "conv2d_4",
            "conv2d_5",
            "conv2d_6",
            "conv2d_7",
            "conv2d_8",
            "kpn_params",
            "conv2d_9",
            "conv2d_10",
            "conv2d_11",
            "temporal_params_out_conv",
        ]
        sd = {}
        for i, name in enumerate(order):
            sd[f"autoencoder.{name}.conv2d.weight"] = q[
                f"autoencoder._param_constant{2 * i}"
            ]
            sd[f"autoencoder.{name}.conv2d.bias"] = q[
                f"autoencoder._param_constant{2 * i + 1}"
            ]
    m.load_state_dict(sd, strict=True)
    return m.eval()


def _psnr(a, b) -> float:
    d = (a.clamp(0, 1) - b.clamp(0, 1)).double()
    return float(10 * np.log10(1 / max(float((d * d).mean()), 1e-12)))


def _bicubic(x, hr_hw):
    """Bicubic upscale of the (tonemapped, jittered) low-res input frame -- the no-NSS baseline."""
    import torch.nn.functional as F
    from ng_model_gym.core.data.data_utils import tonemap_forward

    c = tonemap_forward(x["colour_linear"][:, 0] * x["exposure"][:, 0], mode="reinhard")
    return F.interpolate(c, size=hr_hw, mode="bicubic", align_corners=False).clamp(0, 1)


class OrtCnn:
    """The CNN as a uint8-NHWC-I/O ONNX model (fp16 or int8 variant) run by host ORT."""

    def __init__(self, path: Path):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        )  # no fused int8 kernels
        self.s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])

    def __call__(self, inp):
        q = _to_u8(inp)
        kpn, tmp = self.s.run(None, {"x": q})
        return _from_u8(kpn), _from_u8(tmp)


def _to_u8(inp) -> np.ndarray:  # (1, 12, H, W) float in [0, 1] -> (1, H, W, 12) uint8
    return np.clip(np.round(inp.permute(0, 2, 3, 1).numpy() * 255), 0, 255).astype(
        np.uint8
    )


def _from_u8(a: np.ndarray):
    import torch

    return torch.from_numpy(a.astype(np.float32) / 255).permute(0, 3, 1, 2).contiguous()


def host(n: int) -> None:
    import torch

    torch.set_grad_enabled(False)
    cap = WORK / "cnn_io"
    cap.mkdir(exist_ok=True)
    res = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    runs = {"nss_fp32": (_model("fp32"), None)}
    for v in ("int8", "int8_qat", "fp16"):
        p = WORK / "onnx" / f"cnn_{v}.onnx"
        if p.exists():
            runs[f"nss_{v}_host"] = (_model("fp32"), OrtCnn(p))
    per = {k: [] for k in list(runs) + ["bicubic"]}
    for t, (x, _) in enumerate(_frames(n)):
        for k, (m, cnn) in runs.items():
            out, gt, inp, (kpn, tmp) = _step(m, x, cnn)
            per[k].append(_psnr(out, gt))
            if k == "nss_fp32":
                np.save(cap / f"in_{t:03d}.npy", _to_u8(inp))
                np.save(cap / f"kpn_{t:03d}.npy", kpn.numpy())
                np.save(cap / f"tmp_{t:03d}.npy", tmp.numpy())
                hr = gt.shape[-2:]
        per["bicubic"].append(_psnr(_bicubic(x, hr), gt))
        print(t, {k: round(v[-1], 2) for k, v in per.items()}, flush=True)
    res["host"] = {
        "frames": len(per["bicubic"]),
        "cnn_in": list(inp.shape),
        "hr": list(hr),
        **{
            k: {
                "psnr_mean": round(float(np.mean(v)), 3),
                "psnr_per_frame": [round(p, 2) for p in v],
            }
            for k, v in per.items()
        },
    }
    RESULTS.write_text(json.dumps(res, indent=1))


# ---------------------------------------------------------------- the CNN as ONNX
def _nhwc_u8(fm: onnx.ModelProto, quant: bool) -> onnx.ModelProto:
    """uint8 NHWC in ('x', scale 1/255) and out ('kpn', 'temporal', sigmoid outputs, scale 1/255) around
    the NCHW float CNN; fp16 variant: DQ/Q at the boundary, QNN runs the rest in fp16."""
    g = fm.graph
    ren = {
        g.input[0].name: "_x_nchw",
        g.output[0].name: "_kpn_nchw",
        g.output[1].name: "_tmp_nchw",
    }
    for n in g.node:
        n.input[:] = [ren.get(i, i) for i in n.input]
        n.output[:] = [ren.get(o, o) for o in n.output]
    shapes = [
        [d.dim_value for d in v.type.tensor_type.shape.dim]
        for v in (g.input[0], g.output[0], g.output[1])
    ]
    g.initializer.extend(
        [
            numpy_helper.from_array(np.array(1 / 255, np.float32), "_s"),
            numpy_helper.from_array(np.array(0, np.uint8), "_z"),
        ]
    )
    pre = [
        helper.make_node("DequantizeLinear", ["x", "_s", "_z"], ["_x_f"]),
        helper.make_node("Transpose", ["_x_f"], ["_x_nchw"], perm=[0, 3, 1, 2]),
    ]
    post = []
    for src, dst in (("_kpn_nchw", "kpn"), ("_tmp_nchw", "temporal")):
        post += [
            helper.make_node("Transpose", [src], [src + "_t"], perm=[0, 2, 3, 1]),
            helper.make_node("QuantizeLinear", [src + "_t", "_s", "_z"], [dst]),
        ]
    nodes = pre + list(g.node) + post
    del g.node[:], g.input[:], g.output[:], g.value_info[:]
    g.node.extend(nodes)
    n_, c, h, w = shapes[0]
    g.input.append(helper.make_tensor_value_info("x", TensorProto.UINT8, [n_, h, w, c]))
    for (nn, cc, hh, ww), name in zip(shapes[1:], ("kpn", "temporal")):
        g.output.append(
            helper.make_tensor_value_info(name, TensorProto.UINT8, [nn, hh, ww, cc])
        )
    return fm


def build() -> None:
    import onnxruntime as ort
    import torch

    from onnxsim.full_qdq import quantize_full_qdq

    torch.set_grad_enabled(False)
    cap = WORK / "cnn_io"
    ins = sorted(cap.glob("in_*.npy"))
    if not ins:
        sys.exit("run `nss.py host` first (it saves the CNN inputs)")
    x0 = np.load(ins[0])  # (1, H, W, 12) uint8
    _, h, w, c = x0.shape
    d = WORK / "onnx"
    d.mkdir(exist_ok=True)
    res = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    for weights in ("fp32", "qat"):
        ae = _model(weights).autoencoder
        fp = d / f"cnn_{weights}_f32.onnx"
        torch.onnx.export(
            ae,
            (torch.zeros(1, c, h, w),),
            str(fp),
            input_names=["x"],
            output_names=["kpn", "temporal"],
            opset_version=17,
            dynamo=False,
        )
        fm = onnx.load(fp)
        xt = (
            torch.from_numpy(x0.astype(np.float32) / 255)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        ref = ae(xt)
        s = ort.InferenceSession(str(fp), providers=["CPUExecutionProvider"])
        got = s.run(None, {"x": xt.numpy()})
        diff = max(float(np.abs(g - r.numpy()).max()) for g, r in zip(got, ref))
        print(weights, "onnx vs torch max abs", diff)
        res.setdefault("build", {})[f"onnx_vs_torch_{weights}"] = diff
        if weights == "fp32":
            fp16 = _nhwc_u8(onnx.load(fp), quant=False)
            onnx.save(fp16, d / "cnn_fp16.onnx")
        calib = [
            {"x": (np.load(p).astype(np.float32) / 255).transpose(0, 3, 1, 2)}
            for p in ins[::4][:16]
        ]
        q = quantize_full_qdq(
            fm,
            calib,
            method="mse",
            ranges={"x": (0.0, 1.0), "kpn": (0.0, 1.0), "temporal": (0.0, 1.0)},
        )
        # boundary Q/DQ become the uint8 NCHW I/O, then NHWC like the fp16 variant
        q8 = _qdq_to_nhwc_u8(q)
        onnx.save(
            q8, d / ("cnn_int8.onnx" if weights == "fp32" else "cnn_int8_qat.onnx")
        )
    RESULTS.write_text(json.dumps(res, indent=1))


def _qdq_to_nhwc_u8(q: onnx.ModelProto) -> onnx.ModelProto:
    """A full_qdq CNN (float NCHW I/O, pinned [0, 1] I/O ranges) -> uint8 NHWC I/O (scale 1/255)."""
    from onnxsim.full_qdq import quantized_io

    m, info = quantized_io(q, nhwc_inputs=["x"])
    g = m.graph
    for o in list(g.output):  # outputs come back uint8 NCHW: add the NHWC transpose
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
    for k, v in info.items():
        assert abs(v["scale"] - 1 / 255) < 1e-6 and v["zero_point"] == 0, (k, v)
    return m


# ---------------------------------------------------------------- phone
def phone(n: int, iters: int = 20) -> None:
    import torch

    torch.set_grad_enabled(False)
    cap = WORK / "cnn_io"
    loc = WORK / "phone"
    loc.mkdir(exist_ok=True)
    ins = sorted(cap.glob("in_*.npy"))[:n]
    mans = []
    for t, p in enumerate(ins):
        a = np.load(p)
        (loc / f"in{t}.bin").write_bytes(a.tobytes())
        (loc / f"m{t}.txt").write_text(
            f"x u8 in{t}.bin {','.join(map(str, a.shape))}\n"
        )
        mans.append(f"m{t}.txt")
    res = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    for v in ("fp16", "int8", "int8_qat"):
        mp = WORK / "onnx" / f"cnn_{v}.onnx"
        if not mp.exists():
            continue
        env = dict(
            os.environ,
            R="/data/local/tmp/codex-android-arm-nss",
            PHONE_LOCK_OWNER="codex/android-arm-nss",
        )
        cmd = [
            str(Path.home() / ".cache/android-phone/phone-run"),
            str(HERE.parent / "sam/phone.sh"),
            str(mp),
            "htp",
            str(iters),
            str(loc),
            f"nss_cnn_{v}",
            *mans,
        ]
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        log = r.stdout + r.stderr
        (loc / f"{v}.log").write_text(log)
        med = re.search(r"median_ms ([\d.]+)", log)
        ent = {
            "median_ms": float(med.group(1)) if med else None,
            "strict_htp": "PASS" in log,
        }
        # the full pipeline with the phone's CNN outputs, frame by frame (recurrent, closed loop)
        outs = {}
        for t in range(len(ins)):
            f0, f1 = loc / f"out{t}_o0.bin", loc / f"out{t}_o1.bin"
            if f0.exists() and f1.exists():
                outs[t] = (f0.read_bytes(), f1.read_bytes())
                f0.unlink()
                f1.unlink()
        if len(outs) == len(ins):
            ent.update(_replay(outs, len(ins)))
        res.setdefault("phone", {})[v] = ent
        print(v, {k: vv for k, vv in ent.items() if k != "psnr_per_frame"}, flush=True)
    RESULTS.write_text(json.dumps(res, indent=1))


def _replay(outs: dict, n: int) -> dict:
    """Open loop: frame by frame through the fp32 pipeline; at each frame the postprocess also runs
    with the phone's CNN outputs (computed on this very frame's CNN input) in place of the fp32 CNN's.
    PSNR of that output vs ground truth and vs the fp32 output. (No accumulation over frames -- the
    history stays the fp32 pipeline's; the closed-loop drift is `host`'s nss_int8_host run.)"""
    ref = _model("fp32")
    vs_gt, vs_ref = [], []
    for t, (x, _) in enumerate(_frames(n)):
        inputs = ref.set_buffers({k: v[:, 0] for k, v in x.items()})
        inp, deriv, disocc, ndo = ref.preprocess(inputs)
        k_ref, t_ref = ref.autoencoder(inp)
        kb, tb = outs[t]
        k_ph = _from_u8(
            np.frombuffer(kb, np.uint8).reshape(
                1, k_ref.shape[2], k_ref.shape[3], k_ref.shape[1]
            )
        )
        t_ph = _from_u8(
            np.frombuffer(tb, np.uint8).reshape(
                1, t_ref.shape[2], t_ref.shape[3], t_ref.shape[1]
            )
        )
        o_ph = ref.postprocess(k_ph, inputs, t_ph, ndo, deriv, disocc)
        out = ref.postprocess(k_ref, inputs, t_ref, ndo, deriv, disocc)
        vs_gt.append(_psnr(o_ph["output"], out["ground_truth"]))
        vs_ref.append(_psnr(o_ph["output"], out["output"]))
        out.pop("motion", None)
        out["reset_event"] = inputs["reset_event"]
        ref.update_buffers(inputs, out)
    return {
        "psnr_vs_gt": round(float(np.mean(vs_gt)), 3),
        "psnr_vs_fp32": round(float(np.mean(vs_ref)), 2),
        "psnr_per_frame": [round(p, 2) for p in vs_gt],
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "host", "build", "phone", "report"])
    ap.add_argument("--frames", type=int, default=32)
    a = ap.parse_args()
    sys.path.insert(0, str(HERE))
    {
        "fetch": fetch,
        "host": lambda: host(a.frames),
        "build": build,
        "phone": lambda: phone(a.frames),
        "report": lambda: print(json.dumps(json.loads(RESULTS.read_text()), indent=1)),
    }[a.cmd]()
