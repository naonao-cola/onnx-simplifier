# Sparse4D v3 on the Snapdragon 8+ Gen 1 (Xiaomi 12S, Hexagon V69)

[Sparse4D v3](https://github.com/HorizonRobotics/Sparse4D) (`sparse4dv3_temporal_r50_1x8_bs6_256x704`,
the only public v3 checkpoint: ResNet-50, 6 cameras at 256x704, 900 anchors of which 600 are carried
between frames, 6 decoder layers) rebuilt in plain PyTorch and run on the phone.

Same data and GT criteria as `../bevformer_tiny` and Fast-BEV: nuScenes-mini scene-0103, its first
6 keyframes chained with the temporal state; a detection matches a GT object if its score is
>= 0.3, the class is the same and the BEV centers are within 2 m (greedy, 190 GT objects).

| file | what |
|---|---|
| `model.py` | the model without mmcv/mmdet3d (`load_official()` loads the checkpoint strictly), the host-side temporal instance bank, the decoder, and DFA twice: upstream's rank-6 code verbatim and a rank-4 rewrite |
| `data.py` | nuScenes-mini keyframes as the upstream test pipeline makes them (reuses bevformer_tiny's data) |
| `validate.py` | fp32 over the scene: GT match, and the rank-4 DFA checked against upstream's at every call |
| `export.py` | the whole frame as one HTP graph (frame_first / frame_temp), checked against torch on ORT CPU on every frame, simplified with onnxsim |
| `quantize.py` | int8 ResNet-50 + FPN (onnxsim.full_qdq), decoder left float |
| `phone_chain.py` | the 6-frame chain on the phone, strict all-HTP, the phone's own outputs carried through the host instance bank |
| `bisect_phone.py` | chosen intermediates, phone vs ORT CPU (how the three bugs below were found) |
| `dfa_hvx/dfa_core.h` | one decoder layer's DFA as 24 calls (6 cameras x 4 levels) of the unmodified `../../msda_hvx` kernel, summed |
| `dfa_hvx/dfa_rpc.idl`, `dfa_impl.c`, `build.sh` | the FastRPC skel (one RPC per layer, QuRT threads over anchor blocks) and the `s4d_run` chain runner |
| `dfa_hvx/dfa_case.py`, `dfa_host_check.c`, `dfa_sim.c` | real DFA calls as cases; the scalar body on the host and the HVX body on hexagon-sim vs torch |
| `dfa_hvx/split.py` | the 14 HTP pieces around the DFA (int8 backbone with uint8 channels-last outputs), and the same split chain in torch |
| `dfa_hvx/s4d_run.cpp`, `phone_split.py` | the split frame on the phone (rpcmem buffers shared by ORT and the DSP), chained with the host instance bank |

## Reproduce

```sh
C=~/.cache/sparse4d; S="systemd-run --user --wait --collect --pipe -p MemoryMax=12G -p MemorySwapMax=0 --working-directory=$PWD"
curl -L -o $C/sparse4dv3_r50.pth https://github.com/HorizonRobotics/Sparse4D/releases/download/v3.0/sparse4dv3_r50.pth
echo "5beed4d4933ca6448d72586b0f8812863574289ff3c4192de71dc9f46a42f0ed  $C/sparse4dv3_r50.pth" | sha256sum -c
../bevformer_tiny/fetch_data.sh        # nuScenes-mini cameras -> ~/.cache/onnxsim-bevformer/nuscenes-mini
$S python3 validate.py --ckpt $C/sparse4dv3_r50.pth --data ~/.cache/onnxsim-bevformer/nuscenes-mini --work $C/work
$S python3 export.py frame --ckpt $C/sparse4dv3_r50.pth --work $C/work
```

## fp32 (host)

| scene-0103, 6 frames chained | GT matched / 190 | predictions | precision |
|---|---|---|---|
| score >= 0.3 (the bevformer_tiny / Fast-BEV criterion) | 71 | 80 | 89% |
| score >= 0.2 | 116 | 184 | 63% |

Per frame (>= 0.3): 6/23, 8/29, 10/30, 12/35, 16/34, 19/39. Frames get better as the temporal
instances accumulate. Sparse4D's final score is the class probability times the sigmoid of its
centerness estimate. So its scores sit lower than BEVFormer's, and at 0.3 it is precise but
conservative. The comparisons below give both thresholds.

The rank-4 DFA (`dfa_rank4`, what the HTP graphs use) against upstream's rank-6 code, on the same
inputs at all 36 calls: max abs 3.8e-6.

## All-HTP baseline (fp16, strict: nothing on the CPU)

| scene-0103, 6 frames chained on the phone | frame ms | FPS | GT >= 0.3 | GT >= 0.2 |
|---|---|---|---|---|
| fp32 torch (host) | | | 71 / 190 | 116 / 190 |
| **fp16, whole frame on the HTP** | **3307** | 0.3 | **71 / 190** | 114 / 190 |

Frame 0's outputs match fp32 at cls cos 0.99994, box 0.99999. Later frames can't be compared row by
row: the top-600 instance cache reorders on tiny differences. They are compared by GT matches,
which equal fp32's on every frame at >= 0.3 (6, 8, 10, 12, 16, 19).

Three things had to be fixed before the HTP graph was right. `bisect_phone.py` found each one.
1. **Keypoints behind a camera.** Upstream divides by max(depth, 1e-5), so these project to ~4e9,
   past fp16's range. The graph normalizes the projection on the host (`proj_n`), floors the depth
   at 1 cm and clamps the location to [-1.5, 2.5]. Exact on all 6 frames (<= 9e-4). A 10 cm floor
   is not exact.
2. **A uint8 graph input into a plain Cast.** QNN reads a uint8 input as a quantized tensor. The
   FPN levels came out at cos 0.03 - 0.15 against ORT CPU. An exact DequantizeLinear(1, 0) instead
   gives 0.999999.
3. **The weighted sum miscomputed.** A broadcast Mul over a middle axis, (6, 8, 32, N*13) x
   (6, 8, 1, N*13), came out at cos 0.23 on the HTP with both inputs right. Broadcasting along the
   last axis, (6, N*13, 8, 32) x (6, N*13, 8, 1), is correct.

The frame costs 3.3 s because the DFA runs as dense GridSample + broadcast + reduce on the HTP:
6 layers x 4 levels x 6 cameras x 900 anchors x 13 points. That is what the HVX sampler below
replaces.

## DFA on the HVX: 3307 -> 266 ms/frame

The DFA maps onto the generic MSDA kernel (`../../msda_hvx`, unmodified) as one call per (camera,
level):
- NV = 1, L = 1, 8 groups x 32 channels;
- the 13 keypoints padded to 16 with zero weights;
- mode `MSDA_REF_PIX` with all-zero offsets, and the projected points as the per-query references;
- a per-camera visibility mask, so anchors a camera can't see cost nothing (~21% of (camera, anchor)
  pairs are visible);
- uint8 value maps, with one scale / zero point per FPN level.

`dfa_impl.c` runs a layer's 24 calls in one RPC and sums them over cameras on the DSP.

Checks against torch on real layer inputs (`dfa_case.py`):
- host scalar body: max abs <= 7e-6;
- HVX body on hexagon-sim: cos 0.99992, the msda kernel's Q15 weight rounding;
- the uint8 value maps end to end (torch split chain): 72/190 GT at >= 0.3, against fp32's 71.

| scene-0103, 6 frames chained on the phone | frame ms | FPS | GT >= 0.3 | GT >= 0.2 |
|---|---|---|---|---|
| fp32 torch (host) | | | 71 / 190 | 116 / 190 |
| fp16, whole frame on the HTP (baseline) | 3307 | 0.3 | 71 / 190 | 114 / 190 |
| **int8 backbone + fp16 decoder on the HTP, DFA on the HVX** | **266** | **3.8** | **71 / 190** | 103 / 190 |

Per frame (median of 6 runs, temporal frame): 8 HTP pieces take 146 ms and the 6 DFA calls take
117 ms. Of the DFA time, 109 ms is in the DSP and 8.4 ms is FastRPC.

| step | ms |
|---|---|
| bb (int8 ResNet-50 + FPN, uint8 outputs) | 16.2 |
| pre0 (layer 0's keypoints + weights) | 12.4 |
| dfa0..5 (each) | 18.9 (17.5 in the DSP) |
| mid0..4 (each: output proj, FFN, refine, graph attention, next DFA inputs) | 23.1 |
| post | 2.0 |

At >= 0.2 the phone finds 103 against fp32's 116 (-11%). At the 0.3 criterion used for the other
BEV models it matches fp32 exactly.

**Next levers** (not done):
- The mid pieces are fp16, with fp32 graph I/O at every boundary: pts (6, 900, 16, 2) and
  w (24, 900, 8, 16) out, agg in. That is the EP-context fp32-boundary cost found before. uint8 or
  fp16 I/O for those tensors, and int8 Linears in the mids, are the obvious next steps.
- On the DSP side, the 24 calls per layer each rebuild their tap lists. A fused per-anchor loop
  over (camera, level) would share the coordinate math.
