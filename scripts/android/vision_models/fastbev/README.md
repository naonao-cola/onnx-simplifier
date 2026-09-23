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

## Files

| file | what |
|---|---|
| `fetch_data.sh` | checkpoints + nuScenes-mini (camera keyframes only) |
| `data.py` | frames exactly as each upstream's test pipeline builds them (resize/crop, normalization, lidar2img / sensor2keyego, M0's previous-keyframe selection) + GT |
| `model.py` | encoder / view (gather) / BEV pieces for both models, `load_m0()` / `load_pp()` name maps |
| `geometry.py` | host LUTs (`m0_lut`, `pp_lut`) + verbatim copies of upstream's `backproject_inplace` / `FastrayTransformer` loops |
| `decode.py`, `csrc/postproc.c` | upstream test-time decode + (rotated / circle) NMS; the C file is shared with the phone runner |
| `validate.py` | view transform vs the upstream loops, detections vs GT, saves frames for export |

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
