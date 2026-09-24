# Arm Neural Frame Rate Upscaling (NFRU v1) on the Xiaomi 12S: network on the HTP, the rest on the Adreno GPU

NFRU is Arm's open frame-generation model, the DLSS-3-style counterpart of [NSS](../nss): from two
rendered frames m1 and p1 (colour, depth, motion vectors, camera matrices) it generates the frame half
way between them. A small autoencoder (16 x 270 x 480 in) predicts 4 per-pixel logits; fixed shaders do
the rest -- block-matching optical flow between m1 and p1, a dynamic-object mask, depth-aware forward
splats of the rendered motion and of the flow to t = 0.5, the 16 network inputs, and a softmax blend of
four warped colour candidates at 1080p.

| piece | source | license |
|---|---|---|
| NFRU v1 weights, fp32 + QAT int8 | `Arm/neural-frame-rate-upscaling` (Hugging Face) | Arm AI Model Community License v1.0 |
| test sequence (56 frames at 60 fps: rgb, depth, motion, matrices) | `Arm/neural-graphics-dataset`, `nfru/test/0000.safetensors` | same |
| model + torch pre/post-processing | `arm/neural-graphics-model-gym` @ `fc5fdaf6` (as ../nss) | Apache-2.0 |

Nothing of Arm's is committed: `nfru.py fetch` downloads it (sha256 pinned in `sha256.json`). The
OpenCL kernels (`nfru_kernels.cl`) are a fresh translation of the gym's Apache-2.0 torch backend
(`usecases/nfru/model/{optical_flow/blockmatch_v321.py, torch_processing/*.py}`); Arm's GLSL/Slang
shaders (proprietary notice) are not used.

## Pipeline

```
nfru.py fetch        weights + license + test sequence -> ~/.cache/arm-nfru
nfru.py golden       the gym's torch pipeline window by window (test windows: m1 = n-1, p1 = n+1 of the
                     60 fps capture, i.e. a 30 fps game doubled), every kernel input/intermediate -> golden/
nfru.py build        the autoencoder -> int8 QDQ (onnxsim full_qdq on Arm's QAT weights, output pinned to
                     Arm's QAT output quantization), uint8 NHWC I/O
nfru_cl_check.py     the kernels on a host OpenCL device, per stage and closed loop (int8 net on host ORT)
nfru.py phone        nfru_run on the phone: kernels on the Adreno 730, the network on the HTP (ORT + QNN EP)
```

`nfru_run` processes every rendered frame once (colour pipeline, luma pyramid, motion normalization); it
is p1 of one window and m1 of the next. Everything stays in GPU buffers/images; only the network's
uint8 input/output are host-mapped buffers shared with ORT.

## Results (8 test windows)

Host OpenCL (RTX 5050), open loop per stage against the torch golden: colour, pyramid, warps, block
vectors + sub-pixel fit, median, hint mask, warp_flow, the uint8 network input are **exact**; the joint
bilateral differs on <0.02% of vectors (a sum order), the dynamic mask on 2-3 of 518K pixels (einsum
order), warp_mv on <10 values. Closed loop from the rendered frames only:

| window | torch fp32 vs GT | host kernels + int8 net: vs GT / vs torch | **phone** (RGBA8 out): vs GT / vs torch |
|---|---:|---:|---:|
| w000 | 23.38 dB | 23.39 / 53.1 dB | **23.38** / 46.0 dB |
| w001 | 23.29 | 23.30 / 51.6 | **23.30** / 44.5 |
| w002 | 23.06 | 23.07 / 54.5 | **23.06** / 45.9 |
| w003 | 26.71 | 26.73 / 55.2 | **26.72** / 45.2 |
| w004 | 26.48 | 26.51 / 54.3 | **26.49** / 45.9 |
| w005 | 26.69 | 26.72 / 54.6 | **26.70** / 46.4 |
| w006 | 26.54 | 26.59 / 52.4 | **26.60** / 43.3 |
| w007 | 26.83 | 26.85 / 54.9 | **26.86** / 45.0 |

(For reference: repeating m1 gives 15.4-16.4 dB, averaging m1 and p1 17.1-18.3 dB.)

Phone time per generated frame (OpenCL profiling events, the HTP's `Run`; medians over 16 windows,
`cl_qcom_perf_hint` high; the per-frame upload from the CPU is excluded -- in a game those buffers already
live on the GPU):

| stage | first version | now |
|---|---:|---:|
| colour + luma (1080p) + pyramid | 2.3 ms | 2.6 ms |
| block matching (4 levels, 270p finest) | 30.7 | 13.7 |
| dynamic mask + motion splats (540p/270p) | 3.1 | 3.2 |
| preprocess (270p, 16 ch) | 10.7 | 2.2 |
| network, int8 on the HTP | 2.6 | 2.5 |
| postprocess (1080p blend) | 203 | 3.5 |
| **per generated frame (wall)** | **256** | **31.6** |

What mattered, as in NSS: the colour as an RGBA32F **image** (postprocess 203 -> 3.5 ms: 16 texel reads
instead of 48 scattered planar buffer loads per pixel), uchar16/uchar4 vector I/O for the network, block
matching from **local-memory tiles** with 16 x 16 work-groups, and fp16 rounding by `convert_half_rte`
instead of a private half store + load (block matching 18 -> 13.7 ms) -- with fp16 *subnormals rounded by
hand*: the Adreno flushes them, which cost 0.5-1 dB against torch. A box-sum SAD through local memory was
slower (barriers). Reproducing the reference bit-for-bit needed two torch-CPU quirks: `F.interpolate(x0.5,
bilinear)` rounds differently for its 68x120 -> 34x60 level (the four weighted taps summed) than for the
larger ones (row-separable), and division must be correctly rounded (k / 255 comes from a table).

### NSS + NFRU

NSS upscales each rendered 540p frame to 1080p in ~27 ms (../nss, with the on-device jitter LUT); NFRU
generates one frame between each pair in 31.6 ms. Run back to back on the same GPU/HTP, two displayed
frames cost 27 + 31.6 = 58.6 ms plus the game's own rendering R of one 540p frame. With R excluded that
is **34 displayed FPS at 1080p**, vs 37 FPS from NSS alone (each displayed frame rendered); NFRU raises the
displayed rate whenever R > 4.6 ms (2 / (R + 58.6 ms) > 1 / (R + 27 ms)) -- e.g. R = 16 ms: 27 FPS with
NFRU vs 23 without. Block matching (13.7 ms, mostly the 270p level) is what to shrink next.
