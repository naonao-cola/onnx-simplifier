# RT-DETR-r18vd on the Xiaomi 12S HTP

Plan item 3 of `../../vision_models_plan.md`: RT-DETR-r18vd (Hugging Face `PekingU/rtdetr_r18vd`,
Apache-2.0) at 640x640, whole network on the Snapdragon 8+ Gen 1 HTP (V69) through ORT + QNN EP.

| file | what |
|---|---|
| `model.py` | loads the HF model and patches the decoder's multi-scale deformable attention (MSDA) into an exact rank <= 4 form; finite invalid-anchor marker |
| `export.py` | `validate` (patched vs HF on the eval images), `full` / `full_u8` ONNX export + ORT check + onnxsim |
| `quantize.py` | int8 QDQ with `onnxsim.full_qdq`, region-wise mixed precision, uint8 NHWC image input |
| `phone_eval.py` | strict all-HTP runs over the eval images on the phone, detections matched vs the fp32 torch model |
| `profile_regions.py` | share of one HTP execute by model region / op type from a QNN detailed profile |
| `common.py` | preprocessing (HF processor: bilinear resize to 640x640, /255), COCO ids, detection matching |

Images are the deploy pipeline's yolo11n COCO val2017 lists (`../../deploy/models/yolo11n.yaml`):
64 calibration ids, 20 eval ids, cached in `~/.cache/onnxsim-deploy/_images`. Accuracy is detections
(score >= 0.3, top-300 over queries x classes like HF's post-processing) matched by class and
IoU >= 0.5 against the fp32 torch model on the same image.

## Reproduce

Each heavy step under `systemd-run --user --wait --collect --pipe -p MemoryMax=12G -p MemorySwapMax=0`;
every phone step goes through the shared host lock (`~/.cache/android-phone/phone-run`, which
`phone_eval.py` uses itself; `partition_report.sh` needs it as a prefix).

```
python3 export.py validate                  # patched vs HF: max abs 9e-5, 195/195 detections
python3 export.py full                      # -> ~/.cache/onnxsim-rtdetr/work/full.sim.onnx (630 nodes, max rank 4)
python3 phone_eval.py $W/full.sim.onnx      # fp16, strict all-HTP
python3 quantize.py front8                  # -> $W/full.front8.onnx
python3 phone_eval.py $W/full.front8.onnx --input u8
python3 phone_eval.py $W/full.sim.onnx --n 1 --iters 3 --profile && \
  python3 profile_regions.py $W/full.sim.onnx $W/phone_full.sim/pulled/prof.csv
```

## The rank <= 4 MSDA rewrite

HF's MSDA builds (batch, queries, heads, levels, points, 2) sampling locations. That is rank 6, and
QNN refused all 39 ops that touch it (plan item 3: 4 QNN graphs, 90.9 ms with CPU fallback, strict
all-HTP fails). `model.py` computes the same thing one level at a time, with heads as the batch
axis, and no tensor above rank 4:

- the `sampling_offsets` / `attention_weights` Linears become per-head MatMuls, (1,Q,C) x (H,C,P*2),
  using the same weights re-sliced;
- the grid is `(2 ref_xy - 1) + offsets * ref_wh / P`, (H,Q,P,2), which is exactly `2 * loc - 1`;
- values are `(H, D, h, w)` per level, then `GridSample -> * attn -> ReduceSum(P)`, summed over
  levels.

Checked against HF on the 20 eval images: max abs diff 9e-5, 195/195 detections identical. The
exported graph has 630 nodes, max rank 4, and **QNN refuses no ops: strict all-HTP works**.

The anchors' "invalid" marker (float32 max, which becomes inf in fp16) is replaced by a finite 1e4.
Both saturate sigmoid to exactly 1, so the model output is unchanged.

## Results (strict all-HTP, median under the phone lock, 20 eval images)

| model | HTP ms | FPS (HTP only) | matched vs fp32 |
|---|---|---|---|
| fp16 (`full.sim.onnx`, f32 I/O) | 30.4 | 33 | 194/195 |
| int8 backbone + hybrid encoder, fp16 query selection + decoder (`front8`) | **19.3** | **52** | 183/195 (93.8%) |
| **backbone uint8, hybrid encoder uint16 activations (W8A16), rest fp16 (`bb8enc16`)** | **20.9** | **48** | **190/195 (97.4%)** |
| backbone + encoder uint16 activations (`front16`) | 23.9 | 42 | 188/195 |
| front8 + decoder Linears int8 (`front8lin`) | 25.2 | | 182/195 |
| everything int8 but LayerNorm/Softmax/GridSample (`all8`) | 20.8 | | 0/195 |
| all8 + int8 GridSample with uint16 sampling coordinates (`mix8`) | 25.6 | | 0/195 |

The int8 rows use a uint8 NHWC image input (the camera's RGB bytes, scale 1/255). For context,
YOLO11n is ~2.6 ms on the same HTP (`../../deploy/`, 88.8-90.4% vs fp32).

**Default: `bb8enc16`.** It is 10 ms faster than fp16 and loses 4 detections where front8 loses 11,
for 1.6 ms more than front8.

- **Quantize only the CNN part** (backbone, hybrid encoder). The decoder in int8 is either slower
  (its Linears: quantize/dequantize around fp16 neighbours) or broken (box refinement, reference
  points and the `Log`s of inverse-sigmoid in uint8).
- **The AIFI MLP's exact GELU** (`Div, Erf, Add, Mul, Mul`) stays fp16 as a whole. QNN EP fuses the
  float pattern into Gelu, but it has no Erf op of its own, float or quantized. An int8 Erf node
  unit, or a float Erf between quantized neighbours, lands on the CPU.
- **The uint8 NHWC input.** `full_qdq.quantized_io`'s `uint8 -> Transpose -> DQ` gets a lone uint8
  Transpose refused here (0xc26). `quantize.py` makes it a data-movement QDQ unit instead
  (`DQ -> Transpose -> Q -> DQ`, the same scale/zero point, exact).

### Where front8's 11 lost detections come from

`host_eval.py` measures a QDQ model on ORT CPU with graph optimizations off: the quantization
error alone, with no integer kernels. It tracks the phone closely (front8: 185 on the host, 183 on
the phone), so the bisect ran there.

| policy (host ORT) | matched |
|---|---|
| front8 (backbone + encoder uint8) | 185/195 |
| backbone uint8 only (`bb8`) | 189/195 |
| encoder uint8 only (`enc8`) | 187/195 |
| front8 with one group kept fp16 (`front8x:<regex>`): stem, stage 0/1/2/3, input projections, lateral/downsample convs, FPN blocks, PAN blocks | 183-186/195 |
| front8 with AIFI's LayerNorm/Softmax fp16 (`front8s`, phone) / the whole AIFI fp16 (`front8a`, phone) | 180 / 182 |
| backbone uint8, encoder uint16 (`bb8enc16`) | 188/195 |
| backbone + encoder uint16 (`front16`) | 189/195 |

The loss isn't in one layer:
- Both halves contribute: the backbone about 6 detections, the encoder about 8.
- Keeping any single group in fp16 recovers at most one detection, and AIFI isn't the cause.
- MSE calibration and 64 calibration images don't change it either (183 and 182 on the phone).

What recovers it is resolution in the encoder's activations: 16-bit activations there
(`bb8enc16`). Doing the same in the backbone costs 3 ms more for no gain (`front16`).

### Where the fp16 time goes (QNN detailed profile, one execute)

| region | share |
|---|---|
| backbone | 19.4% |
| hybrid encoder (AIFI + CCFM) | 21.2% |
| query selection (enc_output LayerNorm, heads, TopK) + QNN layout transposes | 23.1% |
| decoder MSDA gather (value_proj, GridSample, weighted sum, output_proj), 3 layers | 22.1% |
| decoder MSDA sampling (offset/weight MatMuls, grid), 3 layers | 9.6% |
| decoder self-attention, FFN, norms, heads | 4.8% |

By op type, GridSample is 12.7% and the Mul/ReduceSum after it most of the 15.2% + 3.0%.

## The decoder's MSDA on the HVX (`msda_hvx/`)

The HVX kernel is `../../msda_hvx/` (a generic mmcv-contract kernel). Its mode `MSDA_REF_BOX` is
RT-DETR's own location formula, so the kernel takes the raw sampling offsets and the grid math
leaves the HTP too. `msda_hvx/split.py` cuts the model into 4 HTP pieces around 3 kernel calls:

    pre -> [msda0] -> mid0 -> [msda1] -> mid1 -> [msda2] -> post

- **pre**: backbone, encoder, query selection, the 3 layers' value maps
  (`value_proj_i(memory)`, 8400x256 each), and layer 0 up to its cross-attention.
- **mid_i**: the rest of layer i, then layer i+1 up to its cross-attention.
- **post**: the rest of layer 2, then the heads.

`split.py check` runs the chain with the kernel's torch contract (`msda_ref.msda_reference`)
against HF: max abs 4.7e-5, 195/195 detections. `dec_run.cpp` chains the pieces on the phone
in-process: rpcmem buffers, with ORT writing piece outputs straight into the buffers the DSP maps.
`phone_split.py` pushes, runs it under the phone lock, and matches the detections.

The value maps can go to the kernel as fp32 or as uint8 (`split.py quant --u8-values`: a per-tensor
QuantizeLinear, min/max-calibrated at the end of `pre`'s fp16 value path; the scale and zero point
are stored in the model's metadata, where `dec_run` reads them). The kernel's uint8 value input
(`vdtype = MSDA_U8`) came from #1859's 89bad72e.

| split | value maps | kernel flags | pre | 3 msda calls (in-DSP) | mid0 + mid1 + post | total | matched |
|---|---|---|---|---|---|---|---|
| fp16 pre | fp32 | 4 | 23.1 | 3.82 (3.08) | 3.30 | 30.8 ms | 194/195 |
| front8 pre | fp32 | 4 | 11.6 | 3.84 (3.22) | 3.11 | 19.1 ms | 182/195 |
| bb8enc16 pre | fp32 | 4 | 13.1 | 3.57 (2.94) | 3.07 | 20.5 ms | 190/195 |
| **front8 pre** | **uint8** | 260 | 10.1 | 3.03 (2.41) | 3.04 | **16.6 ms** | 182/195 |
| **bb8enc16 pre** | **uint8** | 260 | 11.8 | 3.03 (2.36) | 3.06 | **18.3 ms** | **189/195** |
| front8 pre, value_proj int8 too (`front8v`) | fp32 | 4 | 11.2 | | | 18.9 ms | 173/195 |

Flags 260 means 4 threads and 16-query jobs; with 4 threads and 32-query jobs the uint8 rows are
0.3 ms slower. With uint8 values the kernel takes 0.8 ms per layer in the DSP and 1.0 ms with
FastRPC (fp32: 1.0 and 1.2 ms).

Where the split's time went: the kernel replaces ~31% of the fp16 profile (the MSDA gather and
sampling), but `pre` must hand the 3 value maps (3 x 8400 x 256) over as graph outputs:

| pre variant (front8), strict all-HTP | ms |
|---|---|
| no value maps (lower bound) | 6.67 |
| value maps fp32 outputs | 11.57 |
| value maps fp16 outputs | 11.05 |
| value maps uint8 outputs (fp16 value_proj, per-tensor Q) | 10.24 |
| value_proj int8, value maps uint8 outputs | 8.85 (but int8 value_proj: 173/195) |

- **Most of it is the value_proj compute.** The value_proj matmuls themselves (3 x 8400x256x256,
  fp16) run inside the all-HTP model too.
- **uint8 value maps are what makes the split win.** They cut the output bytes to a quarter and
  make the kernel ~20% faster.
- **They cost one detection:** 189 vs 190 for bb8enc16, and front8 is unchanged at 182. Simulated
  on the host with everything else fp32, per-tensor uint8 value maps give 191/195 and per-channel
  193/195, but the HTP emits per-tensor only.

### Summary: RT-DETR-r18vd on the Xiaomi 12S

| configuration | frame ms | FPS | matched vs fp32 |
|---|---|---|---|
| fp16, all-HTP | 30.4 | 33 | 194/195 |
| bb8enc16, all-HTP | 20.9 | 48 | 190/195 |
| **bb8enc16 + decoder MSDA on the HVX (uint8 value maps)** | **18.3** | **55** | **189/195** |
| front8, all-HTP | 19.3 | 52 | 183/195 |
| front8 + decoder MSDA on the HVX (uint8 value maps) | 16.6 | 60 | 182/195 |

Reproduce (after `export.py`):

```
python3 msda_hvx/split.py check && python3 msda_hvx/split.py export && python3 msda_hvx/split.py dump
python3 msda_hvx/split.py quant --policy bb8enc16 --u8-values   # -> pre.bb8enc16.v8.onnx
HEXAGON_SDK_ROOT=... HEXAGON_TOOLCHAIN=... msda_hvx/build.sh          # skel + dec_run
MSDA_FLAGS=260 python3 msda_hvx/phone_split.py pre.bb8enc16.v8.onnx
```

## What a deploy spec / the demo app would need

- **Deploy pipeline** (`../../deploy/`): the all-HTP `bb8enc16` model fits today, with two
  additions.
  - A preprocess kind `resize` (a plain 640x640 bilinear stretch, no letterbox, /255).
  - A postprocess kind `detr` (sigmoid, top-k over queries x classes, cxcywh -> xyxy).
  - Its quantize stage would need the region policy (a node-exclusion list plus per-tensor
    uint16 overrides), i.e. `onnxsim.full_qdq` instead of `quantize_static`.
- **The split**: a pipeline stage for "HTP pieces + DSP kernel calls". The Mask R-CNN demo app's
  native chain already has that shape (HTP sessions and FastRPC kernels over rpcmem).
