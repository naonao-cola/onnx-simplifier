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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["golden", "host-cl"])
    ap.add_argument("--frames", type=int, default=8)
    a = ap.parse_args()
    if a.cmd == "golden":
        golden(a.frames)
    else:
        import nss_cl_check

        nss_cl_check.run(GOLD, a.frames)


if __name__ == "__main__":
    main()
