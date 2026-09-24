# Arm Neural Super Sampling (NSS v1) on the Xiaomi 12S: CNN on the HTP, pre/post on the Adreno GPU

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

### The whole NSS on the phone: pre/post-processing on the Adreno GPU, CNN on the HTP

Most of NSS is fixed-function work around the CNN: depth scatter, disocclusion, luma derivatives
(pre), then history reprojection with motion vectors (Catmull-Rom), KPN filtering of the jittered
low-res colour and the temporal blend (post). Arm deploys those as GPU shaders; the shader sources in
the HF repo carry a proprietary notice, so `nss_kernels.cl` is a fresh OpenCL translation of the gym's
Apache-2.0 torch reference (`torch_preprocess` / `torch_postprocess`, the "high" path only).
`nss_run.cpp` runs a whole frame in one process: depth scatter + preprocess on the GPU write the CNN's
uint8 NHWC input into a mapped buffer, ORT + QNN EP runs the int8-QAT CNN on the HTP straight into
mapped uint8 output buffers, postprocess reads them (the temporal map through an image view of the
same memory) and writes next frame's history. `libOpenCL.so` is dlopen'ed from the vendor partition.

**Correctness.** `nss_gpu.py golden` dumps every input, state and intermediate of the torch pipeline
(int8-QAT CNN on host ORT); `nss_gpu.py host-cl` runs the kernels on a host OpenCL device against it:
depth scatter exact, CNN input max abs 3e-5 (0-3 of 6.3M uint8 values off by one), postprocess 120 dB
vs the golden output, and in closed loop (own state, host ORT CNN on the kernels' uint8 input) the
PSNR vs GT equals the torch pipeline's to 0.01 dB on every frame. On the phone (8 frames, closed loop
with the HTP CNN): PSNR vs GT equals the torch pipeline's (+-0.01 dB) every frame, 52-54 dB vs its
output (the RGBA8 ceiling).

**Speed**, Xiaomi 12S (Adreno 730 + HTP), 1920x1080 output, median over 8 frames x 4 passes, GPU times
from OpenCL profiling events:

| step | preprocess | HTP CNN | postprocess | frame |
|---|---:|---:|---:|---:|
| first version (planar fp32 buffers) | 56 ms | 6 ms | 170 ms | 238 ms |
| vector layouts, uint8 temporal read directly, LUT in constant memory | 40 | 6 | 140 | 190 |
| no dynamically indexed private arrays (unrolled) | 15 | 4 | 116 | 140 |
| `fma1()`: no OpenCL `fma()` (software-emulated on Adreno) | 16 | 4 | 77 | 101 |
| history as an RGBA32F image (texture path) | 13 | 4 | 28 | 49 |
| lookup tables in `__local`, temporal map as an image view | 12 | 4 | 14 | 33 |
| **colour + derivative state as images** | **5.9** | **3.3** | **13.0** | **25** |

25 ms/frame is 40 FPS at 1080p output (the per-frame upload of colour/motion/depth from the CPU,
~4 ms, is excluded: in a game those already live on the GPU). NSS is a recurrence -- frame t+1's
preprocess needs frame t's output and CNN feedback -- so the GPU and the HTP cannot overlap across
frames (only the depth scatter could). What mattered on the Adreno, in order: (1) dynamically indexed
private arrays (spilled to memory, and one kernel's per-thread constant tables filled 32 KB of local
memory); (2) OpenCL `fma()` -- the reference's single-rounding multiply-adds -- is software-emulated
(no native fused fp32 FMA): ~50 ms of the postprocess; an error-free Dekker product + TwoSum instead;
(3) 2D gathers through images rather than buffers; (4) divergent `__constant` lookups serialize --
stage tables in `__local`. `-cl-fast-relaxed-math` and `cl_qcom_perf_hint` changed nothing measurable.
The Adreno driver rejects `-cl-fp32-correctly-rounded-divide-sqrt`; its default divide/sqrt accuracy
is what the phone numbers above use.

### tinygrad-generated OpenCL vs hand-written

`tg_nss.py` writes two stages in tinygrad, renders them with tinygrad's OpenCL backend on the host
(`DEV=CL`, fork `onnxsim/tinygrad` @ `62d98031`, upstream's OpenCL renderer), captures every launched
kernel (source, launch dims, buffers) and replays them on the phone with `cl_bench --plan`, next to
hand-written twins with the same math (`tg_compare.cl`), on the same random inputs:

| stage (Adreno 730) | tinygrad | hand-written | outputs |
|---|---:|---:|---|
| postprocess accumulate tail, 1080p (elementwise + per-pixel channel max) | 15.3 ms (7 kernels) | 5.1 ms (1) | identical (max 1.2e-7) |
| preprocess YCoCg derivative, 540p (+-1 stencil + instability state machine) | 8.7 ms (2 kernels) | 1.0 ms (1) | identical |

tinygrad's math is exact, but its schedule is the cost: the per-pixel max over 3 channels splits the
accumulate graph into reduce kernels that materialize 1080p intermediates, and the derivative graph
becomes one kernel per output, each recomputing the stencil. The rest of NSS -- the data-dependent
gathers (motion-reprojected bilinear/Catmull-Rom, the KPN taps, the nearest-depth search) -- is not
expressible efficiently: tinygrad lowers a data-dependent gather to one-hot compares (O(pixels^2)), so
those kernels stay hand-written.

### Not done yet

NFRU (Arm's neural frame-rate upscaling: the same shape -- a small CNN plus warp/blend shaders) and the
demo app's "game upscaling" replay mode build on this runner. The per-frame filter offset LUT is
computed on the device from the jitter (`nss_lut.h`, bit-identical to the gym's `_compute_lut` over 512
jitters, `lut_check.cpp`), so a frame needs only colour, motion, depth and a few scalars.

## Reproduce

```
S="systemd-run --user --wait --collect --pipe -p MemoryMax=12G -p MemorySwapMax=0"
$S python nss.py fetch            # ~2.9 GB into ~/.cache/arm-nss (test sequence 2.8 GB)
$S python nss.py host --frames 32 # fp32 pipeline; saves the CNN inputs -> cnn_io/
$S python nss.py build            # ONNX: fp16 + int8 (fp32 weights) + int8 (Arm QAT weights)
$S python nss.py host --frames 32 # again: adds the ONNX CNN variants (closed loop, host ORT)
python nss.py phone --frames 16   # takes the phone lock itself per variant

# GPU pre/post-processing (OpenCL) + HTP CNN
$S python nss_gpu.py golden --frames 8   # torch golden dumps -> ~/.cache/arm-nss/golden/
$S env PYTHONPATH=<pyopencl> python nss_gpu.py host-cl --frames 8   # kernels on a host OpenCL device
CL_HEADERS=/usr/include python nss_gpu.py phone --frames 8 --iters 4   # builds nss_run, runs under the lock
$S env DEV=CL TINYGRAD_PATH=<tinygrad> PYTHONPATH=<pyopencl> python tg_nss.py host   # tinygrad vs hand
python tg_nss.py phone
```
