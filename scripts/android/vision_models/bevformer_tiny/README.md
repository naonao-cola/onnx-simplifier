# BEVFormer-tiny on the Hexagon HTP

The real BEVFormer-tiny (official `bevformer_tiny_epoch_24.pth`), rebuilt in plain PyTorch
(no mmcv / mmdet / mmdet3d / mmdeploy), validated against upstream semantics on real nuScenes-mini
frames, then exported and run piece by piece on the phone's HTP through QNN.

## Files

| file | what |
|---|---|
| `fetch_data.sh` | sha256-pinned checkpoint + the camera-only prefix of nuScenes-mini (no account needed) |
| `model.py` | backbone+FPN, 3-layer encoder (TSA + SCA), 6-layer decoder + head, NMS-free decode, host-side geometry; `load_official()` name map; rank-5 MSDA + a verbatim copy of mmcv's `multi_scale_deformable_attn_pytorch` |
| `nuscenes.py` | lidar2img / image preprocessing / can_bus exactly as BEVFormer's converter + test pipeline, GT for a sanity match |
| `validate.py` | rank-5 path vs the upstream-literal path (6-D MSDA, SCA `nonzero()` rebatch), temporal, detections vs GT |
| `export.py` | one piece -> ONNX (TorchScript exporter, opset 17) -> ORT CPU check -> onnxsim -> check; phone inputs + fp32 reference outputs |
| `run_phone.sh`, `compare_out.py` | partition report + strict all-HTP run of a piece (`../../vision_models_probe/partition_report.sh`), outputs vs fp32 |
| `e2e_phone.py` | whole model on the HTP frame after frame (phone outputs chained, HTP prev_bev carried) vs fp32 torch and GT |
| `bisect_precision.py`, `bisect_run.sh` | expose chosen intermediates as outputs, run on the HTP, per-tensor cosine vs ORT CPU |

Reproduce (each heavy step under `systemd-run --user --wait --collect --pipe -p MemoryMax=16G -p MemorySwapMax=0`):

```sh
./fetch_data.sh                                   # ~/.cache/onnxsim-bevformer/{*.pth,nuscenes-mini}
C=~/.cache/onnxsim-bevformer
python3 validate.py --ckpt $C/bevformer_tiny_epoch_24.pth --data $C/nuscenes-mini --work $C/work
for p in backbone1 backbone6 enc1 enc3 decoder; do
  python3 export.py $p --ckpt $C/bevformer_tiny_epoch_24.pth --work $C/work
  ./run_phone.sh $C/work $p
done
```

## Validation (CPU fp32, scene-0103 frames 0-2, peak 1.3 GB)

* `msda_rank5` vs mmcv's reference: max abs diff 0.
* checkpoint: all 643 tensors mapped; unused = `code_weights` (loss weights) + `cls_branches.0-4`
  (aux classifiers of the 5 intermediate decoder layers; inference decodes only the last layer).
* rank-5 encoder/decoder vs the upstream-literal path: max abs diff 0 on every frame (the SCA
  all-queries + visibility-mask formulation is exact).
* detections (score >= 0.3, same class within 2 m of a GT center): frame 0 8/23 GT, frame 1 11/29,
  frame 2 16/30 -- temporal context (prev_bev) helps as expected. The 9 CAN-bus signals are 0
  (CAN-bus expansion not in v1.0-mini), so this is a sanity check, not an mAP number.

## Export (each piece its own systemd-run, MemoryMax=16G)

| piece | inputs | ONNX nodes (sim) | ORT CPU vs torch | peak RSS |
|---|---|---|---|---|
| backbone1 | img 1x3x480x800 | 121 | 1.4e-5 max abs | 1.5 GB |
| backbone6 | img 6x3x480x800 | 121 | 2.2e-5 | 2.6 GB |
| enc1 | feats 6x256x15x25, prev_bev, has_prev, shift, can_bus, ref_cam 6x2500x4x2, bev_mask | 107 | 7.7e-6 | 1.8 GB |
| enc3 | same | 245 | 1.1e-5 | 2.0 GB |
| decoder | bev_embed 2500x256 | 482 | 2.1e-4 | 1.3 GB |

Nothing came close to the 16 GB cap; no external data was needed (largest file: backbone, 98 MB).

## Phone: Snapdragon HTP (V69) via ORT 1.26 + QNN EP 2.6.0 / QNN 2.50, fp32 graph run as fp16

Frame 1 of scene-0103 (has_prev = 1), median of 10, burst perf mode:

| piece | QNN refused ops | strict all-HTP | median ms | cos vs fp32 torch |
|---|---|---|---|---|
| backbone1 | 0 | PASS | 20.4 | 1.00000 |
| backbone6 | 0 | PASS | 129-184 (shared phone, varies) | 1.00000 |
| enc1 | 0 | PASS | 105 | 0.99999 |
| enc3 | 0 | PASS | 241 | 0.99999 |
| decoder (+head) | 0 | PASS | 25 | 1.00000 (cls and bbox) |

Every piece of the real model runs entirely on the HTP (rank-5 MSDA; nothing refused).

### End to end on real frames (`e2e_phone.py`, scene-0103 frames 0-5)

The three pieces chained on the phone, each fed the previous piece's HTP output, with the HTP's
own BEV carried to the next frame as prev_bev (rotated/shifted on the host):

| | fp32 torch (CPU) | HTP fp16 |
|---|---|---|
| per-frame cos vs fp32 (feats / bev / cls / bbox) | - | 1.00000 / 0.99999 / >=0.99999 / 0.99999, no drift over 6 frames |
| GT matched (same class, <2 m, score >= 0.3), 6 frames | 106 / 190 (273 dets) | 105 / 190 (274 dets) |
| latency per frame | - | backbone6 126 + enc3 241 + decoder 25 = ~392 ms (2.5 FPS) |

fp16 on the HTP is accuracy-neutral for BEVFormer-tiny. The encoder (60% of the time, 6
GridSamples on 2500 queries) is the next target; int8 PTQ of the backbone the other.

### fp16 precision bisect (the "cos 0.986" of the plan's synthetic probe; 0.916 on the real model)

Two independent causes, both fixed in `model.py` with exact (fp32-identical) rewrites:

1. **ref_cam overflow.** Upstream divides projected points by `max(depth, 1e-5)`, so pillar
   points behind a camera land at |xy| up to 6.7e6, beyond fp16's 65504. Clamping to [-5, 6] on
   the host is exact (validate.py: clamped vs unclamped max abs diff 0) because such points still
   sample zero padding.
2. **QNN miscompiles stack -> reshape -> Gemm.** TSA projects its 2-frame value queue as
   `value_proj(stack([prev, cur]))` = reshape (5000, 256) -> Gemm. On the HTP that Gemm output
   is scrambled (cos 0.12 vs ORT CPU), but only when the intermediates are *not* graph outputs:
   exposing all of them (the first bisect attempt) hides the bug (cos 0.99999). Found by
   exposing a few tensors at a time (`bisect_run.sh`): the Concat is correct, the Gemm right
   after the reshape is not. Projecting each frame separately and stacking afterwards fixes it
   (enc1 cos 0.916 -> 0.99999) and is 36% faster (165 -> 105 ms).
