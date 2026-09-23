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

| configuration | backbone | encoder | decoder | frame | FPS | polylines vs fp32 |
|---|---|---|---|---|---|---|
| fp16 pieces, all-HTP (`export.py` + `run_phone.sh`) | 127 ms | 2790 ms, bev cos 0.815 | 64.7 ms | ~2980 ms | 0.3 | -- |
| fp16, encoder split around the HVX kernel | 127 ms | 239 ms | 67 ms | 435 ms | 2.3 | 59/59 |
| int8 backbone, split encoder, decoder split too | 21.7 ms | 232 ms | 84 ms | 338 ms | 3.0 | 58/59 |
| **int8 backbone, split encoder with CPU-built TSA inputs, HTP decoder** | **22.3 ms** | **139 ms** | **64.4 ms** | **225 ms** | **4.4** | **58/59** |

(`map_run`, scene-0103 x 6 frames, 8 reps after 2 warm-up, medians; every frame within 223-227 ms.)

**Encoder.** On the HTP it is what breaks: 20000 BEV queries x 6 cameras x 8 heads x 8 points of
broadcast grid math and GridSample (and fp16 overflow in it). Split around the kernel, the
current default is (frame 1): CPU TSA inputs 4.5 ms, `prev` (SCA value maps) 1.6 ms, TSA kernel
36.9 ms (DSP 35.4), `midc` 47.8 ms, SCA kernel 31.0 ms (DSP 29.6), `post` 17.2 ms. Only 18% of the
(camera, query) pairs are visible, which the kernel skips. In fp16 the split encoder's bev matches
fp32 at cos 0.999998 (59/59 polylines).

**int8 backbone** (`quantize.py`, `onnxsim.full_qdq`, mse calibration on 72 images of 4 other
scenes, uint8 NHWC image input): 127 -> 22 ms. It costs bev cos 0.998 vs fp32 (the encoder adds
nothing: the CPU-TSA run and the HTP-TSA run agree to 4 decimals) and one polyline of 59 (a
low-score divider on frame 4 drops under 0.4). The large worst-case point errors (3-7 m) are on
queries decode drops; every matched polyline is within 1 m.

**CPU-built TSA inputs.** TSA's value, offsets and weights depend on the frame only through the
256-vector `c = can_bus_mlp(can_bus)` (q0 = bev_embedding + c and the Linears are linear), so
`map_run` builds them from constants (`split.py tsa_consts`: tsa_v = V0 + Wv c, off = A + Bo c,
w = 0.5 softmax4(Aw + Bw c)) in 4.5 ms on 4 CPU threads, straight into the kernel's rpcmem
buffers. That replaces the `pre` HTP piece (88.8 ms, almost all of it writing 57 MB of fp32
outputs); `midc` recomputes q0 in-graph instead of reading 20 MB.

**Decoder split: slower.** 6 x (`dpre` 3.5 ms + kernel 2.9 ms (DSP 1.9) + `dpost` 1.3 ms) is
46 ms, but `dvals` (every layer's value_proj of the BEV) writes 6 x 20 MB of fp32: 37.6 ms, so
84 ms against 64 ms for the whole decoder on the HTP. The kernel's uint8 value path would cut
dvals' output 4x (not tried).

**Next levers** (not done): `midc` (48 ms) and `post` (17 ms) are mostly fp32 graph I/O (q1,
sca_off/sca_w, sca_out: ~75 MB) -- uint8/fp16 boundaries or folding post into the decoder piece;
the decoder split with uint8 value maps; overlapping frames across HTP / DSP / CPU (the
encoder's two kernel calls are 68 ms of DSP time the HTP is idle through).

**Alongside a detector.** MapTR-tiny's backbone is the same R50 + FPN at 480 x 800 as
BEVFormer-tiny's, but with different weights, and the BEV grids differ (200 x 100 over 30 x 60 m
vs 50 x 50 over 102.4 m), so nothing is shared without a jointly trained model. Run one after
the other on this phone: ~225 + ~100 ms (BEVFormer-tiny, #1859) = ~3 FPS for map + boxes; both
use the HTP and the DSP, so overlapping them buys time only where one waits on the other.

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
python ../export.py backbone1 --ckpt ... --work $C/work
python ../quantize.py calib --data $D --work $C/work && python ../quantize.py backbone --work $C/work
python ../quantize.py host --work $C/work                  # img_u8.u8 per frame
# default piece dir: backbone.onnx -> backbone6.q8.onnx, prev/midc/post.onnx + tsa_*.f32 from
# msda_split/, decoder.onnx -> decoder.sim.onnx (map_run takes the CPU-TSA path when prev.onnx exists)
HEXAGON_SDK_ROOT=... HEXAGON_TOOLCHAIN=... OUT=<build> ./build.sh
./phone.sh <build> <piece dir> $C/work/msda_frames 2 8 0 1 2 3 4 5   # DEC=split for the decoder split
python compare.py $C/work/msda_frames 0 1 2 3 4 5
```
