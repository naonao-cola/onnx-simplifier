# StreamPETR on the Xiaomi 12S (SM8475, Hexagon V69 HTP)

StreamPETR (exiawsh/StreamPETR) is a sparse-query, camera-only 3D detector: object queries attend
over all 6 cameras' image tokens (dense attention, 3D position embeddings) and the top-128 queries of
each frame are carried to the next as memory -- no BEV grid, no deformable sampling. It is the
"different design" among the BEV models here (BEVFormer: BEV grid + deformable attention; Fast-BEV:
LUT view transform).

Model: the smallest public checkpoint, **R50, 428 queries (300 + 128 propagated), 256x704**,
NuImages-pretrained, 60 epochs (NDS 54.6 / mAP 44.9 on nuScenes val upstream):
`stream_petr_r50_flash_704_bs2_seq_428q_nui_60e.pth`, sha256
`72900a1c9bfbbd65dd52165b46e2de4931d5a4a7a04e58bf68a9add80cdd2c5f`
(https://github.com/exiawsh/storage/releases/download/v1.0/stream_petr_r50_flash_704_bs2_seq_428q_nui_60e.pth).

## Files

| file | what |
|---|---|
| `data.py` | nuScenes-mini frames as StreamPETR's test pipeline builds them (lidar-frame rig, 0.44 resize + crop to 256x704, ego pose, timestamps); reuses `../bevformer_tiny/nuscenes.py`'s tables, GT and 2 m match |
| `model.py` | plain-PyTorch rebuild: ResNet-50 + CPFPN, `load_official()` name map; `UpstreamHead` (literal transcription of the head + memory queue) and the deployment split `HeadCore` (HTP) + `HostState` (CPU) |
| `validate.py` | runs both head paths on the same features over a scene, diffs them, matches GT; dumps per-frame inputs/outputs for export and the phone |
| `export.py` | `img` / `head` pieces -> ONNX (opset 17) -> ORT CPU check -> onnxsim; phone inputs + torch references |
| `e2e_phone.py` | both pieces on the HTP frame after frame (phone outputs feed the host memory queue), vs the fp32 torch chain and GT |

## Deployment split

* **image piece (HTP)**: 6 cameras, ResNet-50 C4/C5 + CPFPN level 0 -> (6, 256, 16, 44) = 4224 tokens.
* **head piece (HTP)**: `HeadCore` -- memory_embed, spatial-alignment modulation, featurized PE, the
  6-layer decoder (428 queries: self-attention over 428 + 384 memory keys, cross-attention over 4224
  image tokens), post-norm and the last level's cls/reg branches only (the earlier levels' outputs
  are never used at test time). Query-side constants (the 300 learned queries' embeddings after the
  identity ego-motion modulation and time embedding) are folded into buffers.
* **host (CPU)**: what is rig-static (the 3D position embedding through `position_encoder`, the
  spatial-alignment gamma/beta -- recomputed only when the camera rig changes), trigonometric
  (sine / nerf encodings of memory positions, timestamps and ego poses -- fp16 would wreck
  `sin(32 * translation)`) or bookkeeping (the 512-entry memory queue: ego-motion transforms,
  top-128 propagation, box decode). All on <= 640-row arrays.

## Results

GT match: scene-0103 x 6 keyframes (the same frames and criteria as `../bevformer_tiny/`: score
>= 0.3, same class, BEV center within 2 m), a scene's first frame resets the memory.

| | GT matched / 190 (score >= 0.3) | @ 0.2 | predictions >= 0.3 |
|---|---|---|---|
| fp32 torch, upstream-literal head | 105 | | 165 |
| fp32 torch, deployment split | 105 | 149 | 165 |
| **fp16 HTP, both pieces, chained on the phone** | **106** | **151** | 162 |

Phone latency (strict all-HTP, QNN EP via ORT, `partition_report.sh`, 0 refused ops, median of 10):

| piece | fp16, f32 graph I/O |
|---|---|
| image (6 x R50 + CPFPN, 256x704) | 55.6 ms |
| head (6-layer decoder, 428 queries x 4224 tokens) | 29.9 ms |
| frame (sum) | 85.5 ms (11.7 FPS) |

fp16 on the phone vs the fp32 chain: frame 0 (empty memory) cls cos 0.99999, boxes 0.99998. From
frame 1 on, per-row cosines drop to ~0.99 / 0.82 because the 128 propagated query rows are ordered
by each chain's own top-k, so rows stop lining up -- the detections still match (106 vs 105).

Split vs upstream-literal, max |diff| over the 6 chained frames: cls logits 8e-4, boxes 4e-3 m,
propagated memory 4e-4 (float noise accumulated through the memory queue; identical detections).
StreamPETR's focal-loss scores sit lower than BEVFormer's: at the shared 0.3 threshold it matches
105 (BEVFormer-tiny fp32: 106), at 0.2 it matches 149.

## Reproduce

```
C=~/.cache/onnxsim-bevformer   # nuScenes-mini from ../bevformer_tiny/fetch_data.sh
python validate.py --ckpt stream_petr_r50_flash_704_bs2_seq_428q_nui_60e.pth --data $C/nuscenes-mini --work work --extra-thr 0.2
python export.py img --ckpt ... --work work; python export.py head --ckpt ... --work work
# phone (wrap in ~/.cache/android-phone/phone-run when the phone is shared)
R=/data/local/tmp/streampetr ../../vision_models_probe/partition_report.sh work/img.sim.onnx work/img.in/manifest.txt 10
R=/data/local/tmp/streampetr python e2e_phone.py --ckpt ... --work work
```
