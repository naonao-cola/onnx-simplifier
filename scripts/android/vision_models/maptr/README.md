# MapTR-tiny (online vectorized HD map) on the Xiaomi 12S

MapTR predicts the local HD map around the ego car (lane dividers, pedestrian crossings, road
boundaries) as polylines of 20 points, from the 6 nuScenes cameras. Target: the Xiaomi 12S
(Snapdragon 8+ Gen 1, SM8475, Hexagon V69), no root. The HTP (NPU, via ORT + QNN EP) runs the
dense parts; the deformable attention runs on the DSP's HVX through the generic MSDA kernel
(`../../msda_hvx/`, the one BEVFormer and RT-DETR use).

## Model and data

- **Checkpoint:** MapTRv2's nuScenes checkpoints are not released (upstream README: "WIP"). This is
  the official **MapTR-tiny, R50, BEVFormer encoder, 24 ep** (`maptr_tiny_r50_24e_bevformer.pth`,
  48.7 mAP on nuScenes val; Google Drive id `1y-UBwGBSb2xiV40AuQEBhB-xJyV7VusX`, sha256
  `d4d802df91896a632eb893f19a5c4a3114318fec647b6011755da25fe166f3c8`). It is the published variant
  whose encoder and decoder attention map onto the MSDA kernel. MapTRv2 changes the decoder
  (decoupled self-attention, one-to-many matching, auxiliary depth/segmentation losses) and ships
  an LSS ("bevpool") encoder in its configs; neither is covered here.
- **Rebuild:** `model.py` is plain PyTorch (no mmcv/mmdet3d), `load_official()` maps every
  checkpoint key. It follows `../bevformer_tiny/model.py` with MapTR's shapes: BEV 200 x 100 over
  x +-15 m, y +-30 m (20000 queries), 1 encoder layer, 50 instances x 20 points = 1000 decoder
  queries, 3 classes. MapTR's test mode keeps no temporal state (`video_test_mode=False`:
  prev_bev None, can_bus position/yaw zeroed).
- **Data:** nuScenes-mini camera frames through `../bevformer_tiny/nuscenes.py` (the same
  pipeline: 0.5x resize, pad to 480 x 800). **No map GT:** the map expansion needs a nuScenes
  account, so accuracy is phone vs the fp32 model: polylines with score >= 0.4, matched by class
  at symmetric Chamfer distance < 1.0 m (MapTR's middle AP threshold). Eval: scene-0103, 6
  keyframes. Calibration (int8): scene-0061, -0553, -0757, -1077 (night), first 3 keyframes.

## Results (phone, all runs under the host phone lock; medians)

`validate.py`: the rebuild vs the upstream-literal path (mmcv MSDA, SCA nonzero rebatch): max abs
0 on all 6 frames (encoder and decoder); the ref_cam clamp to [-1, 2] is exact.

| configuration | backbone | encoder | decoder | frame | polylines vs fp32 |
|---|---|---|---|---|---|
| fp16 pieces, all-HTP (`export.py` + `run_phone.sh`) | 127 ms | 2790 ms, bev cos 0.815 | 64.7 ms | ~2980 ms | -- |
| fp16, encoder split around the HVX kernel (`msda_hvx/`) | 127 ms | 239 ms | 67 ms | 435 ms | 59/59 |

The encoder on the HTP is what breaks: 20000 BEV queries x 6 cameras x 8 heads x 8 points of
broadcast grid math and GridSample (and fp16 overflow in it). Split around the kernel it is 5
steps (fp16 run, frame 1): `pre` 95 ms, TSA kernel 42 ms (DSP 40), `mid` 55 ms, SCA kernel 32 ms
(DSP 30), `post` 18 ms; bev cos 0.999998 vs fp32. Only 18% of the (camera, query) pairs are
visible, which the kernel skips.

**TSA folding.** Without temporal state TSA's 2-frame queue is [q, q]: both frames sample the
same value map at the same reference points, so it is one value map with the frames' 4 + 4 points
and every weight halved -- half the value traffic of the literal 2-map call (exact; `split.py
check`).

## Files

| file | what |
|---|---|
| `model.py` | the plain-torch MapTR-tiny, `load_official()`, `decode()`, host geometry |
| `validate.py` | rebuild vs upstream-literal on real frames; saves `<work>/frames/<i>.pt` |
| `export.py`, `run_phone.sh`, `compare_out.py` | one piece -> ONNX -> onnxsim; run it on the HTP (partition report + strict) and compare |
| `quantize.py` | calibration images; int8 backbone (`onnxsim.full_qdq`, uint8 NHWC input) |
| `msda_hvx/split.py` | the pieces around the kernel (`check`, `export`, `host`) |
| `msda_hvx/map_run.cpp`, `build.sh`, `phone.sh` | the whole frame on the phone in one process; build; push/run/pull |
| `msda_hvx/compare.py` | phone outputs vs fp32: cosines and polyline matches |

## Reproduce

Each heavy host step under `systemd-run --user --wait --collect --pipe -p MemoryMax=16G -p MemorySwapMax=0`;
phone steps under `PHONE_LOCK_OWNER=<you> ~/.cache/android-phone/phone-run`.

```
C=~/.cache/onnxsim-maptr; D=<nuscenes-mini>   # ../bevformer_tiny/fetch_data.sh
python validate.py --ckpt $C/maptr_tiny_r50_24e_bevformer.pth --data $D --work $C/work
python export.py backbone6 --ckpt ... --work $C/work     # also backbone1, decoder, encoder
./run_phone.sh $C/work backbone6
cd msda_hvx
python split.py check  --ckpt ... --work $C/work
python split.py export --dec --ckpt ... --work $C/work
python split.py host --work $C/work
HEXAGON_SDK_ROOT=... HEXAGON_TOOLCHAIN=... OUT=<build> ./build.sh
./phone.sh <build> <piece dir> $C/work/msda_frames 2 8 0 1 2 3 4 5   # DEC=split for the decoder split
python compare.py $C/work/msda_frames 0 1 2 3 4 5
```
