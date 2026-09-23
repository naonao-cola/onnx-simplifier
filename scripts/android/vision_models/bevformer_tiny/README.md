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

## Validation (CPU fp32, scene-0103 frames 0-2, peak 1.3 GB)

* `msda_rank5` vs mmcv's reference: max abs diff 0.
* checkpoint: all 643 tensors mapped; unused = `code_weights` (loss weights) + `cls_branches.0-4`
  (aux classifiers of the 5 intermediate decoder layers; inference decodes only the last layer).
* rank-5 encoder/decoder vs the upstream-literal path: max abs diff 0 on every frame (the SCA
  all-queries + visibility-mask formulation is exact).
* detections (score >= 0.3, same class within 2 m of a GT center): frame 0 8/23 GT, frame 1 11/29,
  frame 2 16/30 -- temporal context (prev_bev) helps as expected. The 9 CAN-bus signals are 0
  (CAN-bus expansion not in v1.0-mini), so this is a sanity check, not an mAP number.
