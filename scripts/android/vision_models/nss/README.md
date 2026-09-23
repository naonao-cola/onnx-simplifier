# Arm Neural Super Sampling (NSS v1) on the Xiaomi 12S HTP

DLSS, XeSS, FSR4 and MetalFX are closed. Arm's **NSS** is the open, mobile-targeted counterpart of a
DLSS-2-style temporal super sampler: a small *parameter-prediction* CNN (a 12-channel, 148K-parameter
UNet) predicts per-pixel kernel (KPN) weights and temporal blend parameters, and fixed shaders do the
rest -- reprojecting the previous output with motion vectors, rejecting disoccluded history, and
filtering the jittered low-res frame into the high-res one. Arm ships the weights (fp32, and a QAT int8
checkpoint), the model definition/training code, and a rendered test sequence with ground truth.

This directory runs the **CNN on the HTP** and validates the full NSS pipeline end to end on Arm's
test sequence (Bistro, 960x540 -> 1920x1080, x2), with the CNN swapped for the ONNX/phone variants.

| piece | source | license |
|---|---|---|
| NSS v1 "high" weights, fp32 + QAT int8 | `Arm/neural-super-sampling` (Hugging Face) | Arm AI Model Community License v1.0 |
| test sequence (100 frames: colour/depth/motion/jitter + GT) | `Arm/neural-graphics-dataset`, `nss/test/test_full_resolution_sample.safetensors` | same |
| model + torch pre/post-processing | `arm/neural-graphics-model-gym` @ `fc5fdaf6` | Apache-2.0 |

Nothing of Arm's is committed here: `nss.py fetch` downloads the files (sha256 pinned in `sha256.json`)
and clones the gym at the pinned commit. The Arm AI Model Community License allows use, modification
and redistribution (with the license text, Arm's notices and its Clause 4 acceptable-use terms passed
on) and forbids reverse engineering the models; this port only loads the released weights into Arm's
published Apache-2.0 architecture and exports that to ONNX. `nss_gym.py` imports the gym's NSS model and
torch pre/post-processing without its heavy framework dependencies (executorch, pydantic, slangtorch):
empty parent packages plus four small stand-ins, `processing_backend="torch"`.

## Results

### The CNN on the HTP

The CNN at the "high" preset's input size (12 x 544 x 960: 960x540 padded to a multiple of 8) as
ONNX (`torch.onnx`, max abs 1.0e-5 vs torch for the fp32 weights, 3.6e-6 for the QAT weights), wrapped
to uint8 NHWC in (the preprocess tensor, [0, 1] -- the same 1/255 grid Arm's int8 metadata gives it)
and uint8 NHWC out (both heads end in a sigmoid): `kpn` 36 x 136 x 240, `temporal` 4 x 544 x 960.
Strict all-HTP (ORT 1.27 + QNN EP, burst), medians of 18 runs under the phone lock:

| CNN variant | HTP | full pipeline, 16 frames: PSNR vs GT | vs the fp32 pipeline |
|---|---:|---:|---:|
| fp32 torch (host, reference) | - | 25.63 dB | - |
| fp16 (uint8 I/O) | 10.09 ms | 25.63 dB | 58.5 dB |
| int8, onnxsim `full_qdq` on the fp32 weights | 2.69 ms | 25.55 dB | 45.4 dB |
| **int8, onnxsim `full_qdq` on Arm's QAT weights** | **2.70 ms** | **25.71 dB** | 46.0 dB |

"Full pipeline" = open loop: every frame goes through the fp32 torch pipeline, and its postprocess is
re-run with the phone's CNN outputs for that frame's input; PSNR of the tonemapped 1920x1080 output
vs the ground truth (and vs the fp32 output). int8 PTQ calibration: MSE on 8 of the saved CNN inputs,
input/output ranges pinned to [0, 1].

### Closed loop over 32 frames (host ORT, the same ONNX graphs)

The CNN variant also drives the recurrence (its outputs feed the history of the next frame):

| | PSNR vs GT, frames 0-31 |
|---|---:|
| bicubic upscale of the low-res input (no NSS) | 21.11 dB |
| NSS, fp32 CNN | 26.05 dB |
| NSS, fp16 CNN (uint8 I/O) | 26.08 dB |
| NSS, int8 CNN from the fp32 weights | 25.06 dB |
| **NSS, int8 CNN from Arm's QAT weights** | **26.15 dB** |

NSS is +4.9 dB over bicubic on this sequence (the gap grows as history accumulates: +2.2 dB on
frame 0, +5.7 dB by frame 31). PTQ of the fp32 weights costs 1 dB once the error feeds back through
the history; Arm's QAT weights quantized by onnxsim lose nothing (slightly above fp32). So the HTP
runs NSS's network at 2.7 ms per 540p frame -- ~8% of a 33 ms frame, and cheaper than the 10.1 ms fp16.

### What is not on the phone yet: the pre/post-processing

Most of NSS is fixed-function work around the CNN: depth dilation/scatter, disocclusion masks, luma
derivatives (pre), then history reprojection with motion vectors (Catmull-Rom), KPN filtering of the
jittered low-res colour and the temporal blend (post). Arm deploys those as **GPU shaders** (the GLSL
fragment/compute shaders in the HF repo's `scenario/`, run by the ML SDK for Vulkan). The gym's
`processing_backend="torch"` port used here is a training/validation reference: 0.35-1.2 s (pre) and
3-4.5 s (post) per frame on the host, and it doesn't trace to ONNX (Python `round()` on tensors). So
a real-time NSS on this phone needs those shaders ported -- the natural split on a Snapdragon is
Arm's GLSL on the Adreno GPU via Vulkan (plain fragment/compute shaders; only the CNN used the ML
extension) with the CNN on the HTP, or a native/HVX port of the same math. That, and the "game
upscaling" replay demo and NFRU (frame generation, the same shape: optical-flow and warp shaders
around a small CNN) that depend on it, are the follow-ups.

## Reproduce

```
S="systemd-run --user --wait --collect --pipe -p MemoryMax=12G -p MemorySwapMax=0"
$S python nss.py fetch            # ~2.9 GB into ~/.cache/arm-nss (test sequence 2.8 GB)
$S python nss.py host --frames 32 # fp32 pipeline; saves the CNN inputs -> cnn_io/
$S python nss.py build            # ONNX: fp16 + int8 (fp32 weights) + int8 (Arm QAT weights)
$S python nss.py host --frames 32 # again: adds the ONNX CNN variants (closed loop, host ORT)
python nss.py phone --frames 16   # takes the phone lock itself per variant
```
