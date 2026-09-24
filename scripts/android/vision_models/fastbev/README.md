# Fast-BEV and Fast-BEV++ on the Xiaomi 12S (SM8475, Hexagon V69)

Camera-only 3D detection with a *lookup-table* view transform (no attention, no depth
distribution splatting): every voxel takes one camera pixel's feature row, through indices that
only depend on calibration and ego motion. On the phone that is a Gather with host-computed
indices between two ordinary conv nets.

| model | upstream | config | checkpoint |
|---|---|---|---|
| Fast-BEV M0 | [Sense-GVT/Fast-BEV](https://github.com/Sense-GVT/Fast-BEV) | `fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4` (R18, 256x704, 200x200x4 voxels, 4 frames) | `epoch_20.pth` (NDS 0.4114 on nuScenes val in its eval log) |
| Fast-BEV++ R50 | [ymlab/advanced-fastbev](https://github.com/ymlab/advanced-fastbev) ([arXiv 2512.08237](https://arxiv.org/abs/2512.08237)) | `fastbev-r50-cbgs` (R50, 256x704, 128x128x7 voxels, depth-weighted gather, 1 frame) | `epoch_20_ema.pth` |

Both rebuilt in plain PyTorch (no mmcv/mmdet/mmdet3d); `fetch_data.sh` downloads both checkpoints
(sha256-pinned, Google Drive via gdown) and reuses `../bevformer_tiny`'s nuScenes-mini slice.

Fast-BEV++'s public code is the BEVDet-based `advanced-fastbev`; its Index-Gather-Reshape view
transform is reproduced as upstream implements it (`FastrayTransformer`, first-camera priority,
depth probability gathered per voxel, sum over height).

**Result: both run end to end on the phone in int8 at 31 FPS (M0) / 40 FPS (Fast-BEV++), 6x-8x
BEVFormer-tiny's 5.1 FPS, with more GT matches (123 / 124 vs 107 of 190).**

## Files

| file | what |
|---|---|
| `fetch_data.sh` | checkpoints + nuScenes-mini (camera keyframes only) |
| `data.py` | frames exactly as each upstream's test pipeline builds them (resize/crop, normalization, lidar2img / sensor2keyego, M0's previous-keyframe selection) + GT |
| `model.py` | encoder / view (gather) / BEV pieces for both models, `load_m0()` / `load_pp()` name maps, the conv1 input-normalization fold |
| `geometry.py` | host LUTs (`m0_lut`, `pp_lut`) + verbatim copies of upstream's `backproject_inplace` / `FastrayTransformer` loops |
| `decode.py`, `csrc/postproc.c` | upstream test-time decode + (rotated / circle) NMS; the C file is shared with the phone runner |
| `validate.py` | view transform vs the upstream loops, detections vs GT, saves frames for export |
| `export.py` | one piece -> ONNX -> ORT CPU check -> onnxsim -> check, phone inputs |
| `quantize.py` | calibration set (4 scenes disjoint from the eval scene) + int8 whole-graph QDQ with `onnxsim.full_qdq` |
| `eval_q8.py`, `q8_inputs.py`, `compare_q8.py` | host int8 end to end; phone inputs + byte comparison for the int8 pieces |
| `run_phone.sh`, `profile_phone.sh` | one piece on the HTP (partition report, strict all-HTP, outputs vs reference) / QNN per-op profile, under the shared phone lock |
| `probe_gather.py`, `compose_m0.py` | HTP Gather size probes; M0's gathers composed in front of the int8 BEV net as one HTP graph |
| `dsp/` | `fbgather`: M0's view transform as one HVX FastRPC call (kernel header, IDL, skel, qemu + phone checks, `build_dsp.sh`); `dsp_inputs.py` writes its test data |
| `runtime/fastbev_run.cpp`, `runtime/build.sh`, `e2e_phone.py` | the native phone pipeline and its host driver/scorer |

## Rebuild check (fp32, host, scene-0103 frames 0-5)

```
C=~/.cache/fastbev; S="systemd-run --user --wait --collect --pipe -p MemoryMax=8G -p MemorySwapMax=0"
./fetch_data.sh
$S python3 validate.py m0 --ckpt $C/ckpt/m0_epoch_20.pth --data ~/.cache/onnxsim-bevformer/nuscenes-mini --work $C/work
$S python3 validate.py pp --ckpt $C/ckpt/fbpp_r50_cbgs_epoch_20_ema.pth --data ~/.cache/onnxsim-bevformer/nuscenes-mini --work $C/work
```

* LUT + gather vs upstream's loop: max abs diff **0** on every frame, both models.
* GT matches (score >= 0.3, same class, center within 2 m -- `../bevformer_tiny`'s criteria):

| model (fp32 torch) | GT matched / 190 |
|---|---|
| BEVFormer-tiny (`../bevformer_tiny`) | 106 |
| Fast-BEV M0 | 121 |
| Fast-BEV++ R50 | 130 |

Upstream behaviour kept on purpose (see `data.py` / `geometry.py` docstrings):
* M0's previous frames are nuScenes' previous 1/2/3 keyframes (the seq converter's "sweeps" 1/3/5 at
  interval 3 are exactly those), clamped to the oldest available early in a scene. On a scene's
  first keyframe upstream test uses *future* frames; a live stream cannot, so the current frame is
  repeated there.
* M0's adjacent frames pair the back-left image with the back-right calibration and vice versa
  (the seq converter's camera order differs from the current frame's); the checkpoint was trained
  that way. `--no-adj-swap` projects them correctly: 124 instead of 121 matches, i.e. noise.
* Fast-BEV++ normalizes BGR with the RGB mean/std (BEVDet's `mmlabNormalize`), truncates (not
  rounds) pixel coordinates, and zeroes camera 0's pixel (0, 0).

## On the phone

Everything below: Xiaomi 12S (SM8475, Hexagon V69), ORT + QNN EP on the HTP (`../../htp_exploration/qnn_shell` libs), medians, measured
under `~/.cache/android-phone/phone-run` (the phone is shared; no timing overlapped another job).

### Pieces, fp16 -> int8 (strict all-HTP, `run_phone.sh`, float or uint8 graph I/O as exported)

| piece | fp16 | int8 | notes |
|---|---|---|---|
| M0 encoder (R18 + FPN + fuse, 6 cams 256x704) | 45.2 ms | **6.96 ms** | uint8 NHWC camera pixels in (scale 1, zero point 0), uint8 NHWC features out |
| M0 BEV net (fuse 1024->256, M2BevNeck, anchor head) | 60.6 ms | **10.9 ms** | incl. the 41 MB uint8 volume input |
| M0 gather (4 x 160000 rows of 64) on the HTP | does not finalize | 171 ms / 40.7 ms (4 chunks) | fp16: `QNN_COMMON_ERROR_MEM_ALLOC` -- see below |
| Fast-BEV++ encoder (R50 + FPN + depth net) | 55.0 ms | **10.5 ms** | 39.5 ms before the conv1 fold |
| Fast-BEV++ gather + BEV net | does not finalize (BEV net alone: 25.6 ms) | **18.9 ms** | uint8 Gathers inside the graph |

* int8 is `onnxsim.full_qdq` (uint8 activations, per-channel int8 weights, Relu folded, data-movement
  ops keep their input's qparams) calibrated on 12 frames of scene-0061/-0553/-0757/-1077 (the last
  at night), minmax. Ranges are pinned where pieces meet so bytes pass through unchanged: the image
  input is [0, 255] (the camera's uint8 RGB *is* the input), and M0's BEV input / Fast-BEV++'s
  feature table use the encoder output's range (a "no camera" voxel is the zero point). Box
  regression outputs are uint16 (a single uint8 scale over Fast-BEV++'s concatenated head -- sub-cell
  offsets next to logits -- would zero them, the YOLO11 final-Concat problem), so its six head maps
  are separate outputs.
* **Folding the input normalization into conv1** (`model.fold_input_norm`): Fast-BEV++'s in-graph
  BGR swap (`Slice` step -1) and `/ std` on the full-resolution images were 71% + 12% of its encoder
  on the HTP (QNN per-op profile, `profile_phone.sh`). With 1/std and the channel swap folded into
  conv1's weights and only `x - mean` left in the graph (exact, zero padding included), the encoder
  went 39.5 -> 10.5 ms; M0's 12.1 -> 6.96 ms.
* **The HTP Gather**: `probe_gather.py` finds fp16 Gathers finalize only for a few-MB table and
  output (a 67585 x 64 table fails even for 10k indices; 4225 x 64 fails for 114688 indices). In
  uint8 both of this repo's gathers finalize, but M0's 41 MB volume gather costs 30 ms or more.
  Fast-BEV++'s small one (114688 rows of a 4225-row table, times one depth probability each) stays
  in its BEV graph.

### M0's gather on the DSP (`dsp/`, `fbgather`)

One FastRPC call: 4 uint8 feature tables (the ring slots, each (67584 + 1) x 64 with the zero-point
row last) + 4 int32 LUTs -> the (1, 200, 200, 1024) uint8 volume, channel z*256 + t*64 + c, straight
into an rpcmem buffer the BEV session reads. HVX body: per voxel row two unaligned 128-byte row loads,
`vror` + `vmux` -> two aligned 128-byte stores. Byte-exact vs numpy under `qemu-hexagon-static`
(HVX and plain-C bodies, 40,960,000 bytes) and on the CDSP:

| threads | 1 | 2 | 4 | 6 |
|---|---|---|---|---|
| DSP time | 18.1 ms | 12.4 ms | **8.1 ms** | 9.2 ms |
| round trip | 18.6 ms | 12.9 ms | 8.5 ms | 9.7 ms |

### End to end (`runtime/fastbev_run.cpp`, `e2e_phone.py`)

One native process, frames 0-5 of scene-0103 in order, 10 passes after a warm-up pass. Per-frame
detections are scored with `../bevformer_tiny`'s criteria.

```
./runtime/build.sh
$S python3 e2e_phone.py m0 --work $C/work     # [--gather htp] for the all-HTP M0 variant
$S python3 e2e_phone.py pp --work $C/work
```

| M0 stage | where | ms |
|---|---|---|
| encoder | HTP | 7.0 |
| LUTs for the 4 time steps (10.6 ms, overlapped with the encoder) | CPU, 4 threads | 3.6 wait |
| volume gather (DSP 7.3) | DSP | 7.6 |
| BEV net | HTP | 11.1 |
| decode + NMS | CPU | 2.4 |
| **total** | | **31.9 (31.3 FPS)** |

| Fast-BEV++ stage | where | ms |
|---|---|---|
| encoder | HTP | 13.2 |
| gather + BEV net | HTP | 10.4 |
| decode + NMS | CPU | 1.6 |
| **total** | | **25.2 (39.7 FPS)** |

| scene-0103, 6 frames | per frame | GT matched / 190 |
|---|---|---|
| BEVFormer-tiny, int8 backbone + fp16 encoder/decoder (`../bevformer_tiny`) | 197 ms (5.1 FPS) | 107 (fp32 106) |
| **Fast-BEV M0**, int8, phone | **31.9 ms (31.3 FPS)** | **123** (fp32 121, host int8 123) |
| **Fast-BEV++ R50**, int8, phone | **25.2 ms (39.7 FPS)** | **124** (fp32 130, host int8 128) |

Temporal fusion (M0): the encoder writes each frame's features into slot `i % 4` of an rpcmem ring;
the previous 3 frames' tables are *re-projected* into the current frame every frame (their LUTs
change with ego motion) -- exactly upstream's 4-frame fusion, no BEV warping approximation; only the
current frame's images go through the encoder. The current slot's LUT depends only on the rig and is
reused while its projections don't change (on nuScenes they change every frame: each camera's
lidar2img includes the ego motion between its and the lidar's timestamp).

CPU-side details: `top_k_u8` picks the pre-NMS candidates on the uint8 logits (sigmoid is
monotonic) with one histogram pass, ordered like a stable sort; the rotated IoU exits early when the
boxes' circumscribed circles don't meet (exact). Together with overlapping the LUTs these took M0
from 45.8 to 31.9 ms and Fast-BEV++ from 30.4 to 25.2 ms.

### Tried, no gain

* More CPU threads for the LUTs (6-7) or fewer DSP gather threads (3): within noise (31.2-32.3 ms).
  A strip-wise, branch-free LUT loop (`-O3`, vectorizable) takes the overlapped LUT time from 10.5
  to 9.8 ms but the frame stays at 31.7 ms.
* QNN EP's HTP shared-memory allocator (to drop the QNN graph-I/O copies of the 41 MB volume and
  the feature tables): with this plugin QNN EP (onnxruntime-qnn 2.6.0) `CreateSharedAllocator` on
  the NPU EP device rejects both memory types ("Invalid memory type for OrtEpDevice"), and a
  session with `enable_htp_shared_memory_allocator=1` fails to create the EP ("Unknown exception
  occurred while creating QNN EP"). Not pursued further.

### Follow-ups: pipelining, requantization (`codex/android-fastbev-opt`)

Same phone, scene-0103 frames 0-5, 10 passes after a warm-up, all under the phone lock with the phone
otherwise idle. **Note on the CPU**: after a phone reboot the prime core ran capped at 2.25 GHz and
M0's LUTs took 27 ms (10.6 ms in the table above, measured before the reboot) -- device state this
work may not change; the sequential and pipelined rows below were measured back to back in the
same state.

`fastbev_run` with `PIPELINE=1` (`e2e_phone.py --pipeline` runs both modes and checks the pipelined
detections are **byte-identical** to the sequential ones -- they are, for both models): three
threads, (encoder on the HTP + LUTs on the CPU) -> (M0: DSP gather + HTP BEV net / PP: HTP view+BEV)
-> CPU decode, bounded queues, M0's feature ring grown to 6 slots and the LUT/output buffers to 3 sets
so no in-flight frame aliases another's. The 4 slot LUTs are one threaded pass (`m0_lut_n_mt`).

| model | sequential | **pipelined** | pipelined latency | GT / 190 | bound by |
|---|---|---|---|---|---|
| Fast-BEV M0 | 47.9 ms (20.9 FPS) | **25.6 ms (39.1 FPS)** | 60.7 ms | 123 | CPU LUTs (27 ms, overlapped) |
| Fast-BEV++ R50 (view+BEV requantized) | 22.6 ms (44.3 FPS) | **21.0 ms (47.6 FPS)** | 61.1 ms | 124 | HTP (encoder 13.3 + view+BEV 8.1 ms) |

* **Fast-BEV++ view+BEV requantized with today's `onnxsim.full_qdq`** (same calibration, same
  script): 10.47 -> **8.07 ms** on the HTP, GT unchanged at 124. Re-quantizing the encoders and M0's
  BEV net the same way changed nothing (13.25 / 7.0 / 10.8 ms).
* **Fast-BEV++ view as one batched MatMul** (`FASTBEV_PP_VIEW=matmul`: (YX, 1, 7) @ (YX, 7, 64)
  instead of the broadcast Mul + ReduceSum that is ~40% of the view+BEV graph): QNN runs the 16384
  tiny batches far slower, **27.4 ms** (125 GT). Not the default.
* View+BEV per-op profile now (36 Mcycles): Mul 33%, Resize 20% (the neck's upsamples), Gather 17%,
  Conv 14%, ReduceSum 7%.
* **M0: folding the BEV net's first 1x1 conv into the DSP gather -- not done, arithmetic says no.**
  That conv is 1024 -> 256 channels over 200 x 200 cells = 10.5 GMAC per frame. At the HVX `vrmpy`
  peak (32 lanes x 4 int8 MACs per cycle per thread, ~1 GHz, 4 threads: ~0.5 TMAC/s, before any
  memory stalls) that is >= 21 ms, twice the HTP's *whole* BEV net (10.8 ms). The alternative of
  projecting each feature table through the conv's 16 (height, time) weight slices before the gather
  is 17.7 GMAC per frame plus a 277 M-element intermediate. The volume stays; the pipeline already
  hides the gather (7.5 ms DSP) behind the HTP.

### Remaining levers

* M0: the BEV net's first 1x1 conv (1024 -> 256 channels on 200 x 200) could run on the DSP as part
  of the gather (gather + matmul, never writing the 41 MB volume); the HTP BEV net would then start
  from a 10 MB tensor.
* M0's LUTs on the CPU now bound its pipeline: computing them on the DSP next to the gather (the
  projection is 3 affine maps + 2 divides per voxel and camera; exact rounding against torch has to
  be kept) or restoring the CPU's pre-reboot clocks would bring M0 to its HTP bound (~18 ms).
* Fast-BEV++: the neck's Resize (20% of view+BEV) as an exact depthwise ConvTranspose/DepthToSpace
  (the trick that took EfficientViT-SAM's bicubic neck from 1503 to 42 ms), and the view's
  Mul + ReduceSum on the DSP (a gather + weighted sum like `fbgather`, taking ~5 ms off the HTP).

