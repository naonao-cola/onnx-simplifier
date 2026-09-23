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
| `e2e_phone.py` | both pieces on the HTP frame after frame (phone outputs feed the host memory queue), vs the fp32 torch chain and GT; quantized I/O through `<piece>.json` |
| `quantize.py` | `onnxsim.full_qdq` whole-graph QDQ (uint8 or uint16 activations, int8 per-channel weights) + `quantized_io`, calibrated on 4 scenes x 6 frames disjoint from scene-0103 |
| `sensitivity.py` | host (ORT CPU, no QDQ fusion) per-op-type quantization sensitivity of the head, teacher-forced on scene-0103 |
| `tf_phone.py` | the same teacher-forced head check on the phone's HTP |
| `profile_ops.py` | per-op-type share of an HTP execute from a QNN detailed-profiling CSV (copy of `../bevformer_tiny/`'s) |

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

GT match: scene-0103 x 6 keyframes -- the same frames and criteria as `../bevformer_tiny/` and
Fast-BEV (score >= 0.3, same class, BEV center within 2 m); a scene's first frame resets the memory.
All phone numbers are chained on the phone (each frame's phone outputs feed the host memory queue),
strict all-HTP (QNN EP via ORT, 0 refused ops), medians under the shared phone lock.

| StreamPETR R50 428q, 256x704 | image piece | head piece | HTP / frame | GT @0.3 | GT @0.2 |
|---|---|---|---|---|---|
| fp32 torch (upstream-literal head = deployment split) | | | | 105 | 149 |
| fp16 / fp16 | 55.6 ms | 29.9 ms | 85.5 ms | 106 | 151 |
| **int8 raw-pixel image / fp16 head (default)** | **9.1 ms** | **29.7 ms** | **38.8 ms (25.8 FPS)** | **112** | **150** |
| int8 (normalized uint8 input) / fp16 | 8.7 ms | 29.8 ms | 38.5 ms | 111 | 148 |
| fp16 / W8A16 head | 54.8 ms | 23.6 ms | 78.4 ms | 92 | 138 |
| int8 / int8 head | 8.7 ms | 17.7 ms | 26.4 ms | 50 | 102 |

For context, on the same frames and phone: BEVFormer-tiny ~100 ms/frame, 107 GT
(`../bevformer_tiny/`); Fast-BEV M0 31.9 ms / 123, Fast-BEV++ 25.2 ms / 124 (#1873). StreamPETR's
focal-loss scores sit lower, so the shared 0.3 threshold undercounts it: at 0.2 it matches 150/190.

Host (CPU, Python/torch, per frame): the memory queue + encodings (`HostState.pre`) ~11 ms, post +
decode ~1 ms, not overlapped here (the image piece of frame t+1 needs no memory and could run
during it). Rig-static encodings (position embedding through `position_encoder`, spatial-alignment
gamma/beta): 112 ms once per camera rig.

### What was done, what worked

* **Rebuild + split** (`model.py`): the upstream head's float64 timestamps, nerf / sine encodings and
  4x4 ego-motion transforms stay on the host (fp16 would wreck `sin(32 * translation)`); the HTP head
  is 325 standard ops at rank <= 3. The last decoder level's branches only (earlier levels are
  never read at test time).
* **int8 image piece** (`quantize.py img_raw`, `onnxsim.full_qdq` + `quantized_io`): 55.6 -> 9.1 ms,
  accuracy-neutral (feature cos 0.992-0.995). NormalizeMultiviewImage is folded into the stem conv
  exactly (`export.py`'s `ImgRaw`: conv(W / s, pad0(x)) plus one constant map that carries the
  per-channel bias and the zero-padding border terms), so the uint8 NHWC input calibrates to scale
  1.0 / zero point 0: the camera's RGB bytes are the input, no host normalize+quantize (16.4 ms in
  numpy otherwise). Folded vs original features: max |diff| 8e-4, cos 1.000000000.
* **The head stays fp16.** Profile (QNN detailed): MatMul 78%, the cross-attention `A @ V`
  ((8, 428, 4224) x (8, 4224, 32), softmax fused in) alone 10.1% per layer = 60% of the head.
  Tried, all exact, all slower on the HTP:
  * scores transposed + softmax over the key axis + `V^T A^T` (428-wide output): 143 ms
  * `out_proj` folded into each head's values (`sum_h A_h (V_h W_o,h^T)`, 256-wide output): 107 ms
  * rig-static tensors baked in + uint8 `feat` input (no float image tokens over the boundary): 30.5
    ms -- the head is compute-bound, not I/O-bound
  * **quantization**: W8A16 (uint16 activations) is 23.6 ms but loses 13 GT; on the host (ORT CPU,
    `sensitivity.py`) the same QDQ graph is near exact (teacher-forced worst cos cls 0.99998 / reg
    0.9988), on the HTP it is not (reg 0.83, `tf_phone.py`). Bisected on the phone: the culprit is
    **uint16 x uint16 MatMul** (the attention's activation-activation products) -- keeping MatMul in
    fp16 restores cos 0.9977 but the fp16 islands in a uint16 graph make it 147 ms. uint8
    activations break the head even with MatMul in fp16 (reg cos 0.71).
* **Next levers**: done below -- the cross-attention on the HVX (measured slower) and the host side in
  C++ with the image piece pipelined (40 ms/frame end to end).

## Follow-up: the cross-attention on the HVX (measured: slower, stays on the HTP)

`attn_hvx/`: the head's cross-attention `softmax(Q K^T) V` (8 heads x 32 dims, 428 queries x 4224
image tokens) as one FastRPC call per layer on the cDSP's HVX, taking Q / K / V as the HTP would emit
them (uint8, one scale / zero point per tensor) and returning uint8 in V's own qparams:

* **integer contract** (`attn_contract.py`, torch-free): `s = (q - zq).(k - zk)` exact, `p =
  round(255 e^-(max s - s) sq sk)` by a Q11 `2^-t` with a Q15 cubic (max rel err 3.6e-4), `out =
  round(sum p v / sum p)` -- no float anywhere on the DSP. `attn_kernel.h` has a scalar and an HVX body
  (`vrmpy` ub x ub for both `Q K^T` over packed K and `P V` over packed V, 4 query rows per vector load,
  the exp in halfword lanes); both are **bit-exact** with the contract: host (`attn_host_check.c`),
  hexagon-sim (`attn_sim.c`), and the phone (`attn_client`) on real layer-0 / layer-5 Q / K / V of
  scene-0103 (`emulate.py case`), all 109568 output bytes equal. `tests/test_streampetr_attn_hvx.py`
  runs the synthetic-case checks in CI.
* **phone, per layer (4 threads, median)**: 13.4-15.2 ms DSP (K / V packing 3.6-5.6 ms + attention
  ~9.6 ms), wall +1 ms FastRPC -- vs about 3.5 ms for the same step inside the fp16 HTP head (the
  `A V` matmul is 10.1% of the 29.7 ms head per layer, `Q K^T` + softmax a bit more).
* **why it can't win**: hexagon-sim's per-section profile (`-DATTN_PROF`): QK 4.2k / exp 5.0k / AV
  3.6k pcycles per (row, head), about one packet per 4 cycles per hardware thread -- on V69 a thread
  issues every ~4th cycle, so 4 threads give ~1 packet/cycle in total (the phone's 9.6 ms matches
  428 x 8 x ~3.2k packets / 1.2 GHz). The kernel needs 2 x 1056 `vrmpy` per (row, head) plus the exp;
  even two `vrmpy` per packet and a free exp leave >= 5.7 ms per layer, above the HTP's ~3.5 ms. Dense
  attention belongs on the HTP's matrix unit; the HVX pays off for gathers (the deformable attention
  of BEVFormer / RT-DETR / Sparse4D), not for `Q K^T`.
* The HTP and HVX do run concurrently (#1879), so moving one or two layers' attention off the HTP could
  still shave a few ms of *throughput* in a pipelined chain, at ~13 ms extra latency per layer and a
  split head; not pursued.

## Follow-up: the whole frame in one phone process (C++ host, pipelined)

`runtime/`: `petr_run.cpp` runs the int8 image piece and the head on the HTP (ORT QNN EP) and does
everything `HostState` did in Python in C++ -- the 512-entry memory queue (ego-motion transforms,
top-128 propagation), the sine / nerf encodings, box decoding -- plus the position embedding's geometry.

* **The position embedding is per frame on nuScenes**, not rig-static: `lidar2img` differs frame to
  frame (camera / lidar ego-motion compensation; `pe` changes by up to 1.5), so `e2e_phone.py`'s
  `HostState.rig_inputs` recomputed it every frame in Python (112 ms, not counted in the table above).
  `export.py head_pe` moves its MLPs (`position_encoder`, `spatial_alignment`) into the head and takes
  `pe_in` / `cone` (the C++ geometry: 6 inverse 4x4 + 4224 x 64 points) plus the image piece's uint8
  NHWC `feat` directly (an exact `DequantizeLinear` on the graph input: a uint8 input into a plain
  `Cast` is miscomputed by the HTP, and a DQ behind a uint8 `Reshape` falls back to the CPU).
* **Checked against model.py** (`runtime/check_run.py`, replaying `HostState` on the phone's own head
  outputs): memory rows exact (`mem_emb` 0, `mem_pe3d` 2.6e-5, `mem_motion` 3.9e-3 -- sines of large
  arguments), `pe_in` 3.9e-3, boxes 3.7e-4. The top-128 is ranked by `sigmoid(logit)` in fp32 with ties
  to the lower index, as `model.py` / torch's CPU `topk` do: ranking by the raw logit selects the same
  rows in another order, which is equivalent but sums the fp16 attention in another order -- that
  alone flipped one borderline detection (111 vs 112).
* **pipe**: a second thread runs frame t+1's image piece and geometry (neither needs the memory queue)
  while the main thread does frame t's queue update, head and post. Output files are byte-identical to
  `seq`.

Phone, scene-0103 x 6 frames x 5 passes, strict all-HTP, under the phone lock (two runs each):

| StreamPETR R50 428q, int8 image / fp16 head | per frame | FPS | latency | GT @0.3 | GT @0.2 |
|---|---|---|---|---|---|
| #1884: e2e_phone.py (Python host; HTP time only, host ~11 ms + per-frame `pe` 112 ms not counted) | 38.8 ms HTP | (25.8) | -- | 112 | 150 |
| **petr_run seq** (one process, C++ host, everything counted) | 52-55 ms | 18.2-19.3 | 50-56 ms | **112** | 149 |
| **petr_run pipe** | **40.0-40.1 ms** | **24.9-25.0** | 59 ms | **112** | 149 |

Per frame (medians): image 9.5 ms, geometry 6.6 ms, queue update + encodings 2.7-8.0 ms, head 31.0 ms,
post 0.5 ms. Pipelined, the frame time is the two HTP pieces back to back (9.5 + 31; the head reads
~36 ms there because the next frame's image piece queues on the same HTP) -- all CPU work is hidden,
so from here on only a faster head helps, and the cross-attention section above shows the HVX isn't it.

## Reproduce

```
C=~/.cache/onnxsim-bevformer   # nuScenes-mini from ../bevformer_tiny/fetch_data.sh
python validate.py --ckpt stream_petr_r50_flash_704_bs2_seq_428q_nui_60e.pth --data $C/nuscenes-mini --work work --extra-thr 0.2
python export.py img --ckpt ... --work work; python export.py head --ckpt ... --work work
# phone (wrap in ~/.cache/android-phone/phone-run when the phone is shared)
R=/data/local/tmp/streampetr ../../vision_models_probe/partition_report.sh work/img.sim.onnx work/img.in/manifest.txt 10
R=/data/local/tmp/streampetr python e2e_phone.py --ckpt ... --work work
# int8 image piece (calibration dumps first), then the default chain
python validate.py --ckpt ... --data $C/nuscenes-mini --work work --no-upstream --scene scene-0061 scene-0553 scene-0757 scene-1077
python export.py img_raw --ckpt ... --work work && python quantize.py img_raw --work work
python e2e_phone.py --ckpt ... --work work --img img_raw.sim.q8 --head head.sim --iters 12
# one phone process (runtime/): head with per-frame position embedding, scene data, C++ runner
python export.py head_pe --ckpt ... --work work && python runtime/prep.py --ckpt ... --work work --out scene
OUT=build runtime/build.sh   # then push petr_run, the two .onnx and scene/ with qnn_shell's libs
petr_run scene img_raw.sim.q8.onnx head_pe.sim.onnx pipe 1 5 out 6   # DUMP=1 + runtime/check_run.py to verify
# HVX cross-attention (attn_hvx/): real-layer cases, then the phone bench
python attn_hvx/emulate.py calib|case --ckpt ... --work work; OUT=build CASES="cases/*" FLAGS=4,6 attn_hvx/build.sh
# head experiments
python quantize.py head --act uint16 [--exclude-ops MatMul --tag=_mm]; python sensitivity.py --work work
python tf_phone.py --work work --head head.sim head.sim.q16
```
