# Neural super-resolution on the Xiaomi 12S HTP (an open "DLSS on a phone")

NVIDIA DLSS (and XeSS, FSR4, MetalFX) are closed: no public weights, vendor GPU only, and the
temporal versions need game-engine motion vectors and depth. This directory builds the open,
camera-usable part: single-image x4 neural super-resolution, strict all-HTP on the Snapdragon 8+
Gen 1 (Hexagon V69), with uint8 NHWC input/output so a camera frame goes in and a displayable frame
comes out with no host float conversion.

Status: the SR models, their phone numbers and accuracy (#1906), and the demo app's
"Super-res" mode (`../../maskrcnn_demo_app`, README "Super-resolution mode": XLSR int8, 54 FPS
end to end on test images, the camera's 30 FPS with a bicubic / original / low-res comparison).

## Models

| model | source | license | params | pinned sha256 (prefix) |
|---|---|---|---|---|
| QuickSRNet small / medium / large | Qualcomm AI Hub float ONNX v0.62.2 (`qualcomm/QuickSRNet*` on Hugging Face) | BSD-3-Clause | 33k / 61k / 436k | `ac9abc71` / `c335addd` / `6510c68a` |
| XLSR | Qualcomm AI Hub float ONNX v0.62.2 (`qualcomm/XLSR`) | BSD-3-Clause | 28k | `ef5cdbe2` |
| Real-ESRGAN `realesr-animevideov3` | xinntao/Real-ESRGAN v0.2.5.0 (SRVGGNetCompact, 16 convs) | BSD-3-Clause | 622k | `b8a83768` |
| Real-ESRGAN `realesr-general-x4v3` | xinntao/Real-ESRGAN v0.2.5.0 (SRVGGNetCompact, 32 convs) | BSD-3-Clause | 1213k | `8dc7edb9` |

All are x4, RGB in [0, 1], and made only of Conv / Clip / PRelu / Add / DepthToSpace. Real-ESRGAN's
`out + interpolate(x, 4, nearest)` residual is rewritten exactly (max abs 3e-6 vs the torch module)
as a fixed 1x1 conv of the input image added before the PixelShuffle (DepthToSpace CRD), so there is
no Resize in any graph.

Each model is wrapped as `lr` uint8 `[1, H, W, 3]` -> `sr` uint8 `[1, 4H, 4W, 3]` (scale 1/255, zero
point 0; the output is clamped to [0, 1]):
- **fp16:** `DequantizeLinear` in, the float graph (QNN runs it in fp16), `QuantizeLinear` out.
- **int8:** `onnxsim.full_qdq.quantize_full_qdq` (int8 per-channel weights, uint8 activations, MSE
  calibration on 16 COCO crops, I/O ranges pinned to [0, 1]) + `quantized_io`, so the graph's own
  boundary Q/DQ become the uint8 I/O.

## Phone results (strict all-HTP, ORT 1.27 + QNN EP, burst, medians of 18 runs, under the phone lock)

| model | LR -> HR | fp16 HTP | int8 HTP | fp16 / int8 PSNR vs host fp32 |
|---|---|---|---|---|
| **quicksrnetsmall** | 480x270 -> 1920x1080 | 11.4 ms | **4.2 ms** | 62.3 / 42.9 dB |
| quicksrnetmedium | 480x270 -> 1920x1080 | 16.0 ms | 4.9 ms | 61.9 / 46.4 dB |
| quicksrnetlarge | 480x270 -> 1920x1080 | 38.5 ms | 9.6 ms | 63.2 / 50.5 dB |
| **xlsr** | 480x270 -> 1920x1080 | 15.0 ms | **5.9 ms** | 62.2 / 46.0 dB |
| realesr-animevideov3 | 480x270 -> 1920x1080 | 46.2 ms | 12.8 ms | 63.5 / 46.5 dB |
| realesr-general-x4v3 | 480x270 -> 1920x1080 | 74.4 ms | 21.0 ms | 63.2 / 40.3 dB |
| quicksrnetsmall | 640x360 -> 2560x1440 | 31.9 ms | 11.6 ms | 62.3 / 42.9 dB |
| quicksrnetmedium | 640x360 -> 2560x1440 | 33.2 ms | 11.0 ms | 61.9 / 46.7 dB |
| xlsr | 640x360 -> 2560x1440 | 32.2 ms | 12.7 ms | 61.9 / 45.8 dB |

"PSNR vs host fp32" is the phone's uint8 output against the fp32 model on the host for the same
input (3 nuScenes frames downscaled to the phone input size): fp16 on the HTP is effectively exact
(> 61 dB), int8 costs 40-50 dB there. 1080p output at 4-6 ms leaves most of a 30/60 FPS frame for the
rest of a pipeline; the whole frame fits one graph, so tiling isn't needed at these sizes.

## Accuracy vs ground truth (host)

x4 PSNR on the Y channel (4 px border shaved), HR = center crops, LR = PIL bicubic /4 of them:
12 nuScenes camera images (896x1600 HR) and 12 COCO images (512x640 HR). int8 = the same full_qdq
recipe on the host (ORT basic optimization level, no fused int8 kernels).

| model | nuScenes fp32 / int8 (bicubic) | COCO fp32 / int8 (bicubic) |
|---|---|---|
| quicksrnetsmall | 37.13 / 34.25 (34.93) | 28.81 / 28.11 (27.62) |
| quicksrnetmedium | 37.27 / 35.28 (34.93) | 28.95 / 28.46 (27.62) |
| quicksrnetlarge | 37.48 / 35.85 (34.93) | 29.13 / 28.70 (27.62) |
| **xlsr** | 37.09 / **36.73** (34.93) | 28.83 / **28.72** (27.62) |
| realesr-animevideov3 | 33.33 / 33.09 (34.93) | 27.06 / 27.05 (27.62) |
| realesr-general-x4v3 | 28.44 / 27.92 (34.93) | 26.26 / 25.99 (27.62) |

- The PSNR-trained models beat bicubic by 1.2-2.5 dB in fp32. XLSR keeps nearly all of it in int8
  (-0.1 to -0.4 dB), which with 5.9 ms at 1080p makes it the default for the demo; QuickSRNet-small
  is the fastest (4.2 ms) but its int8 loses more (-2.9 dB on nuScenes).
- Real-ESRGAN scores *below* bicubic by PSNR: it is GAN-trained on synthetically degraded input for
  perceptual sharpness (hallucinated texture), which PSNR penalizes. It's kept for the visual demo,
  not as a fidelity model.

## Reproduce

```
S="systemd-run --user --wait --collect --pipe -p MemoryMax=12G -p MemorySwapMax=0"
$S python superres.py fetch                       # -> ~/.cache/superres/w (sha256-checked)
$S python superres.py build xlsr 270 480          # fp32/fp16/int8 ONNX -> ~/.cache/superres/models/xlsr_270x480
$S python superres.py host xlsr                   # GT PSNR table row
python superres.py phone xlsr 270 480             # takes the phone lock itself, per model run
python superres.py report                         # the tables above
```

`phone` uses `../sam/phone.sh` (ORT + QNN EP runner, `../../htp_exploration/qnn_shell/qnn_run_multi.cpp`)
with its own phone dir `/data/local/tmp/codex-android-superres`.

## Next steps

1. Demo app "super resolution" mode: done (`../../maskrcnn_demo_app`).
2. Arm's open mobile temporal super sampling, `Arm/neural-super-sampling` (NSS v1, Arm AI Model
   Community License): export with the `arm/neural-graphics-model-gym` definition, HTP/HVX split,
   phone ms and PSNR vs fp32 on its `scenario/` data; then a "game upscaling" replay demo.
3. Arm's `Arm/neural-frame-rate-upscaling` (NFRU v1) frame generation, replacing RIFE if it runs.
