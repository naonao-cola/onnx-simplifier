"""NSS v1 ("high") with the CNN on the HTP and the pre/post-processing on the Adreno GPU (OpenCL).

Arm deploys NSS's pre/post-processing as GPU shaders; those shader sources carry a proprietary notice,
so the OpenCL kernels here (`nss_kernels.cl`) are a fresh translation of the Apache-2.0 torch reference
in arm/neural-graphics-model-gym (`usecases/nss/model/torch_{pre,post}process`, pinned in nss_gym.py),
restricted to the "high" quality path (full-res preprocess, 2x2 depth scatter, Catmull-Rom history,
dense 6x6 KPN, YCoCg luma derivative, sharp theta).

  nss_gpu.py golden [--frames N]   torch pipeline (int8-QAT CNN via host ORT, uint8 NHWC I/O) frame by
                                   frame; dumps every GPU-kernel input/state/intermediate -> golden/
  nss_gpu.py host-cl [--frames N]  run nss_kernels.cl on the host OpenCL device against golden/, per stage
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WORK = Path(os.environ.get("NSS_WORK", Path.home() / ".cache/arm-nss"))
GOLD = WORK / "golden"


def golden(n: int) -> None:
    import nss
    import torch

    torch.set_grad_enabled(False)
    GOLD.mkdir(parents=True, exist_ok=True)
    m = nss._model("qat")
    cnn = nss.OrtCnn(WORK / "onnx" / "cnn_int8_qat.onnx")
    pre_mod = sys.modules["ng_model_gym.usecases.nss.model.torch_preprocess.depth"]
    for t, (x, _y) in enumerate(nss._frames(n)):
        inputs = m.set_buffers({k: v[:, 0] for k, v in x.items()})
        in_shape, _, hr_shape, _, depth_shape = m._calculate_dispatch_dims(inputs)
        recon = pre_mod.depth_scatter(
            inputs[m.motion_key],
            inputs["depth"],
            inputs["render_size"],
            output_shape=depth_shape[-2:],
            quarter_res_input=False,
        )
        inp, deriv, disocc, ndo = m.preprocess(inputs)
        q = nss._to_u8(inp)
        kpn_u8, tmp_u8 = cnn.s.run(None, {"x": q})
        kpn, temporal = nss._from_u8(kpn_u8), nss._from_u8(tmp_u8)
        lut, idx_mod = m._generate_offset_lut(inputs["jitter"], in_shape, hr_shape)
        out = m.postprocess(kpn, inputs, temporal, ndo, deriv, disocc)
        d = {
            "colour": inputs["colour_linear"],
            "history": inputs["history"],
            "motion": inputs[m.motion_key],
            "depth": inputs["depth"],
            "feedback_tm1": inputs["temporal_params_tm1"],
            "derivative_tm1": inputs["derivative_tm1"],
            "jitter": inputs["jitter"],
            "exposure": inputs["exposure"],
            "render_size": inputs["render_size"],
            "depth_params": inputs["depth_params"],
            "reset": 1.0 - (inputs["reset_event"] == 0.0).float(),
            "recon_depth": recon,
            "cnn_in": inp,
            "derivative": deriv,
            "disocclusion": disocc,
            "nearest_offset": ndo,
            "offset_lut": lut,
            "idx_modulo": idx_mod,
            "output_linear": out["output_linear"],
            "output": out["output"],
            "ground_truth": out["ground_truth"],
        }
        np.savez(
            GOLD / f"f{t:03d}.npz",
            cnn_in_u8=q,
            kpn_u8=kpn_u8,
            temporal_u8=tmp_u8,
            **{k: v.detach().cpu().numpy() for k, v in d.items()},
        )
        out.pop("motion", None)
        out["reset_event"] = inputs["reset_event"]
        m.update_buffers(inputs, out)
        print(
            f"golden f{t:03d}: in {tuple(in_shape)} hr {tuple(hr_shape)} depth {tuple(depth_shape)}"
            f" psnr {nss._psnr(out['output'], out['ground_truth']):.2f} dB",
            flush=True,
        )


SERIAL = os.environ.get("ANDROID_SERIAL", "239dbd8f")
REMOTE = os.environ.get("NSS_REMOTE", "/data/local/tmp/codex-android-nss-gpu")
LOCK = [str(Path.home() / ".cache/android-phone/phone-run")]
QNN = HERE.parent.parent / "htp_exploration" / "qnn_shell"


def _frames_for_phone(n: int, out: Path) -> None:
    """golden/ inputs -> frameNNN.bin (colour float4 RGBA, motion float2, depth) + frameNNN.txt (scalars, LUT)."""
    out.mkdir(parents=True, exist_ok=True)
    for t in range(n):
        z = np.load(GOLD / f"f{t:03d}.npz")
        import nss_cl_check as ck

        with open(out / f"frame{t:03d}.bin", "wb") as f:
            f.write(ck.rgba(z["colour"][0]).tobytes())
            f.write(ck.yx2(z["motion"][0]).tobytes())
            f.write(np.ascontiguousarray(z["depth"][0, 0], np.float32).tobytes())
        lut = z["offset_lut"][0]
        mh, mw = (int(v) for v in z["idx_modulo"].ravel()[:2])
        s = [
            *z["jitter"].ravel()[:2],
            z["exposure"].ravel()[0],
            *z["render_size"].ravel()[:2],
        ]
        s += [*z["depth_params"].ravel()[:4], z["reset"].ravel()[0]]
        head = " ".join(repr(float(v)) for v in s) + f" {mh} {mw} {lut.shape[-1]}\n"
        (out / f"frame{t:03d}.txt").write_text(
            head + " ".join(repr(float(v)) for v in lut.reshape(-1)) + "\n"
        )


def phone(n: int, iters: int) -> None:
    import subprocess

    build = HERE / "build"
    subprocess.run(
        [str(HERE / "build_gpu.sh")], check=True, env={**os.environ, "OUT": str(build)}
    )
    stage = WORK / "phone_gpu"
    _frames_for_phone(n, stage)
    adb = ["adb", "-s", SERIAL]
    env = {**os.environ, "PHONE_LOCK_OWNER": "codex/android-nss-gpu"}
    files = [
        build / "nss_run",
        HERE / "nss_kernels.cl",
        WORK / "onnx" / "cnn_int8_qat.onnx",
    ]
    files += sorted((QNN / "libs").glob("*.so")) + sorted(stage.glob("frame*"))
    push = " && ".join(f"{' '.join(adb)} push -q {f} {REMOTE}/" for f in files)
    flags = os.environ.get("NSS_CLFLAGS", "")
    run = (
        f"cd {REMOTE} && NSS_CLFLAGS='{flags}' {os.environ.get('NSS_ENV', '')} LD_LIBRARY_PATH={REMOTE} ADSP_LIBRARY_PATH='{REMOTE};/vendor/dsp/cdsp;"
        f"/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp' ./nss_run . cnn_int8_qat.onnx cnn_ctx.onnx {n} {iters}"
    )
    pull = " && ".join(
        f"{' '.join(adb)} pull -q {REMOTE}/out{t:03d}.bin {stage}/" for t in range(n)
    )
    cmd = f'{" ".join(adb)} shell mkdir -p {REMOTE} && {push} && {" ".join(adb)} shell "{run}" && {pull}'
    r = subprocess.run(
        LOCK + ["bash", "-c", cmd], env=env, capture_output=True, text=True
    )
    print(r.stdout)
    if r.returncode:
        sys.exit(r.stderr[-3000:])
    import nss_cl_check

    for t in range(n):
        z = np.load(GOLD / f"f{t:03d}.npz")
        o = np.fromfile(stage / f"out{t:03d}.bin", np.uint8).reshape(1080, 1920, 4)[
            ..., :3
        ]
        o = o.transpose(2, 0, 1).astype(np.float32) / 255
        print(
            f"f{t:03d} phone psnr vs GT {nss_cl_check._psnr(o, z['ground_truth'][0]):.2f} dB"
            f" (golden {nss_cl_check._psnr(z['output'][0], z['ground_truth'][0]):.2f});"
            f" vs golden output {nss_cl_check._psnr(o, z['output'][0]):.1f} dB (RGBA8)"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["golden", "host-cl", "phone"])
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--iters", type=int, default=5)
    a = ap.parse_args()
    if a.cmd == "golden":
        golden(a.frames)
    elif a.cmd == "phone":
        phone(a.frames, a.iters)
    else:
        import nss_cl_check

        nss_cl_check.run(GOLD, a.frames)


if __name__ == "__main__":
    main()
