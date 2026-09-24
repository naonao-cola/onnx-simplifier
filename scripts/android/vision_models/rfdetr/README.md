# RF-DETR on the Xiaomi 12S (Snapdragon 8+ Gen 1, Hexagon V69)

[RF-DETR](https://github.com/roboflow/rf-detr) (Roboflow, 2025; Apache-2.0 for Nano/Small/Medium): a
DINOv2 ViT-S backbone with windowed + global attention (patch 16), an LW-DETR projector to a single
P4 feature map, and a 2-4 layer deformable-attention decoder (300 queries, 16 cross-attention heads
of dim 16, 2 points), with two-stage query selection. Loaded through the `rfdetr` package (1.10.1,
checkpoints md5-checked by the package; `rf-detr-nano.pth` md5 `fb6504cce7fbdc783f7a46991f07639f`).

| variant | resolution | tokens | decoder layers | params |
|---|---|---|---|---|
| Nano | 384 | 24x24 | 2 | 30.5 M |
| Small | 512 | 32x32 | 3 | 32.1 M |
| Medium | 576 | 36x36 | 4 | 33.7 M |

## Files

| file | what |
|---|---|
| `common.py` | eval images (the deploy pipeline's yolo11n COCO ids, the same RT-DETR used), RF-DETR preprocessing / decode, detection matching |
| `model.py` | the library model in export mode + a rank <= 4 `MSDeformAttn.forward` |
| `graph.py` | onnxsim, the rank-5 window-transpose fold, the uint8 NHWC input with the normalization folded into the patch-embed conv |
| `export.py` | `refs` / `validate` / `full` |
| `phone_eval.py` | strict all-HTP run over the 20 eval images under the phone lock, detections vs the fp32 library model |
| `quantize.py` | `onnxsim.full_qdq` policies (see the int8 table) |
| `host_eval.py` | host ORT check of a quantized model (graph optimizations off) |
| `profile_regions.py` | QNN detailed profile -> share by region / op type |
| `cross_eval.py` | phone outputs vs another variant's fp32, or vs COCO val2017 ground truth |
| `smooth.py` | SmoothQuant with its scales folded into the LayerNorms |
| `sens.py` | per-block activation sensitivity and a mixed-precision budget search (onnxsim PR #1935) |

## Making it strict all-HTP (exact rewrites)

- **Deformable attention at rank <= 4.** The library's export path builds (B, Q, heads, points, 2)
  sampling locations (rank 5). `model.py` computes the same thing with batch 1 and heads as the
  GridSample batch, like `../rtdetr/model.py`. On the 20 eval images: max abs diff 0 vs the library,
  142/142 detections.
- **DINOv2 window partition / merge at rank 4.** Reshape(a,b,c,d,e) -> Transpose(0,2,1,3,4) ->
  Reshape keeps the last two axes together, so `graph.py` merges them: Reshape(a,b,c,d*e) ->
  Transpose(0,2,1,3). 5 such pairs (1 partition, 4 merges for the 4 output features); max rank 4.
- **GELU** exports at opset 20 as `Gelu` (at opset 17 it is `Erf`, which QNN puts on the CPU).
- **uint8 NHWC input.** `DequantizeLinear(1, 0)` (a uint8 `Cast` as the first node breaks the HTP) ->
  Transpose -> patch-embed Conv with /255 and the ImageNet mean/std folded into its weights. The
  patch embed has stride == kernel and no padding, so the fold is exact: the patch-embed output
  differs by 5.5e-6 from the float-input graph.

Preprocessing: RF-DETR's `predict()` resizes the float image (bilinear, no antialias) to a square;
the phone input is that image rounded to uint8, and the fp32 reference sees the same pixels.

## Results (strict all-HTP, median of 10 runs of image 0 under the phone lock, 20 eval images)

- **vs fp32**: the phone's detections (score >= 0.3) matched to the same variant's fp32 library
  model (same class, IoU >= 0.5).
- **COCO GT**: recall / precision against COCO val2017 ground truth on the same 20 images (134
  objects), score >= 0.3, IoU >= 0.5, same category; not mAP (`cross_eval.py <tag> gt`).

| model | precision | ms | FPS | vs fp32 | COCO GT recall / precision |
|---|---|---|---|---|---|
| Medium @576 | fp16 | 134.8 | 7 | 145/147 | 69.4% / 63.3% (= fp32) |
| **Medium @448** | **fp16** | **64.3** | **16** | 151/154 | **70.9% / 62.1%** (fp32: 71.6% / 62.3%) |
| Medium @384 | fp16 | 51.2 | 20 | 141/144 | 69.4% / 65.5% (fp32: 69.4% / 64.6%) |
| Medium @320 | fp16 | 29.6 | 34 | 127/127 | 61.9% / 64.8% (fp32: 61.9% / 65.4%) |
| Small @512 | fp16 | 91.2 | 11 | 141/142 | 70.1% / 65.3% (fp32: 70.1% / 66.2%) |
| Nano @384 | fp16 | 46.7 | 21 | 142/142 | 67.9% / 64.1% (= fp32) |
| **Nano @384** | **fp16, folded** (`FOLD=1`, below) | **41.8** | **24** | **142/142** | **67.9% / 64.5%** |
| Nano @384 | W8A16, GELU 16-bit (`bb16g`) | 49.4 | 20 | 137/142 | 67.9% / 64.1% |
| Nano @384 | int8, GELU 16-bit (`bb8g16`) | 37.3 | 27 | 116/142 | 61.9% / 62.4% |
| Nano @384 | folded + `bb8g16` | 33.4 | 30 | 114/142 | |
| **Nano @320** | **fp16** | **26.4** | **38** | 125/126 | **63.4% / 67.5%** (= fp32) |
| Nano @256 | fp16 | 20.1 | 50 | 124/125 | 56.0% / 59.5% (fp32: 56.0% / 60.0%) |

The lower-resolution rows are RF-DETR's weight-sharing operating points: the same weights at a
lower resolution (positional encodings interpolated by the library). Nano, Small and Medium share
the same DINOv2 ViT-S (patch 16) and differ in their trained weights, default resolution and
decoder depth (2 / 3 / 4 layers), so **latency follows the token count**: Medium @448 (784 tokens)
beats Small @512 (1024 tokens) on both speed and recall, and Medium @384 costs Nano @384 plus one
decoder layer (+4.5 ms). On 20 images, recall differences of 1-2 objects (e.g. Medium vs Nano at
384) are within noise. For context, on the same
phone: RT-DETR-r18 at 640 runs in 18.3 ms (`../rtdetr/`, int8 backbone + HVX MSDA), YOLO26n in
~2.6 ms on the HTP (`../../deploy/`).

**fp16 is the recommendation, and resolution is the speed knob.** Nano @320 fp16 (26.4 ms) is both
faster and more accurate than any int8 Nano @384; Medium @448 (64.3 ms) is the accuracy point.

### Where the time goes (Nano @384 fp16, `profile_regions.py`, shares of a detailed-profile run)

| region | share |
|---|---|
| backbone attention | 29.5% |
| backbone MLP | 18.4% |
| backbone norms, LayerScale, residuals, windowing | 13.2% |
| heads / unnamed | 19.5% |
| decoder deformable cross-attention (MSDA) | 6.6% |
| decoder FFN / norms, self-attention, query selection, projector | ~11% |

By op: GELU 18.4%, MatMul + Gemm 28%, Add + Mul 24%, LayerNorm 7%, Softmax 0.9%. The DINOv2 ViT-S
is the cost. The window partition / merge transposes are 0.8%: no layout rewrite is worth it.

Inside the backbone the Add + Mul share is mostly small elementwise ops next to Linears: the q/k/v
bias Adds after the fused qkv MatMul -> Split (10.8%), the LayerScale Muls (5.4%) and the attention
scale Muls on q and k (2.5%). `graph.py`'s `fold_backbone_elementwise` (opt-in, `FOLD=1`; exact up to
float rounding, max abs 1e-4 on the outputs) folds them into the neighbouring weights and makes
qkv one 2-D Gemm with its bias: **46.7 -> 41.8 ms (-10%), 142/142**.

GELU alone as a 16-bit quantized op (`g16`, fp16 elsewhere; SAM's trick) is **91.2 ms**: the 24
fp16 <-> uint16 converts around 12 GELUs cost far more than the lookup table saves.

### int8 does not fit the DINOv2 backbone (all tried, `quantize.py`)

| policy | ms | vs fp32 | notes |
|---|---|---|---|
| `bb8`: backbone int8, LN/Softmax/GELU fp16 | 77.6 | 110/142 | slower than fp16: every fp16 island costs a convert |
| `bb16`: W8A16, GELU fp16 | 99.3 | 135/142 | the fp16 GELU islands on 16-bit tensors are the worst case |
| `bb16g`: W8A16, GELU 16-bit | 49.4 | 137/142 | no fp16 islands, but no faster than fp16 |
| `bb8g16`: int8, GELU 16-bit | 37.3 | 116/142 | fast, but loses 26 detections |
| `bb8g16`, MSE / percentile calibration | 37.2 / 37.3 | 113 / 108 | calibration does not help |
| `bb8x16`: int8, LN/Softmax/GELU 16-bit | 38.2 | 115/142 | |
| `bb8r16`: + residual stream 16-bit | 41.2 | 117/142 | the outliers are not only in the residual stream |
| folded + `bb8g16` | 33.4 | 114/142 | the fold makes int8 faster too |
| folded + SmoothQuant alpha 0.8 + `bb8g16` | 35.1 | 115/142 | alpha 0.5: 107/142 on the host |

**SmoothQuant** (`smooth.py`, `onnxsim.apply_smoothquant`, calibrated on 16 COCO calibration
images): its `1/s` is folded into the preceding LayerNorm's gamma/beta for the 24 LayerNorm-fed
Linears (backbone qkv and fc1; free at run time), kept as a Mul before the 12 fc2 (after GELU), and
undone everywhere else. It moves int8 from 114 to 115/142 on the phone: the loss is not per-channel
outliers at the Linear inputs.

**Where the loss is** (`sens.py`, onnxsim's `activation_sensitivity` from PR #1935, on the
SmoothQuant 0.8 model): each group quantized alone (uint8, everything else float), detections of
the float model kept on 24 held-out COCO images (calibration ids 32-55; float: 196):

| group | kept | | group | kept |
|---|---|---|---|---|
| decoder (+ query selection, heads) | 107 | | layer 09 | 181 |
| backbone rest (embed, projector) | 176 | | layer 06 | 182 |
| layer 00 | 177 | | layer 07 | 185 |
| layer 03 | 179 | | layers 02, 08 | 186 |
| layer 10 | 179 | | layers 11, 01, 04, 05 | 187, 188, 190, 191 |

All uint8: 108. The backbone's loss is spread over every block (5-19 detections each), not one
outlier block. `search_activation_precision_for_budget` (ladder uint8 -> uint16 -> float, budget
190/196) meets the budget only with **every group float except layer 01 at uint16** (191/196):
there is no useful mixed-precision point, and that one uint16 block island fails to compile on the
HTP (QNN `graphAddNode` 6007 on a Quantize at a float -> uint16 Gemm boundary; not pursued, since
it could only match fp16).

The first `bb8` attempt failed to compile (QNN `graphAddNode` 6007 on a Quantize): onnxsim renames
the Linear layers to `Gemm_NN`, so a name-based backbone region left them float between int8
neighbours. `quantize.py` now finds the backbone structurally (everything upstream of the named
`/backbone/` nodes' outputs).

This matches the SAM ViT encoders (`../sam/`): post-training int8 of DINOv2-style activations
loses too much; the RT-DETR CNN backbone quantizes fine. Quantization-aware fine-tuning would be
the way to a fast int8 RF-DETR, not calibration, SmoothQuant or mixed precision.

Not tried: token merging (ToMe) -- it needs a patched DINOv2 attention and trades accuracy the same
way a lower resolution does, which the NAS operating points already give for free.

### The decoder's deformable attention stays on the HTP

RT-DETR moved its decoder MSDA to the shared HVX kernel (`../../msda_hvx/`, 30.4 -> 18.3 ms with an
int8 backbone). RF-DETR is different: a single P4 level (24 x 24 at Nano @384), 16 heads of dim 16
and 2 points. The MSDA is 6.6% of the fp16 run (~3 ms), and head dim 16 is outside the kernel's
HVX fast path (head_dim 32), so a split would cost more in HTP <-> DSP handoffs than it can save.

## Reproduce

Heavy steps under `systemd-run --user --wait --collect --pipe -p MemoryMax=12G -p MemorySwapMax=0`,
in a venv with `rfdetr==1.10.1`, CPU torch, onnx and onnxruntime:

```
export ONNXSIM_REPO=/path/to/onnx-simplifier   # a checkout with the built onnxsim extension
python3 export.py validate nano                # rank<=4 MSDA vs the library
python3 export.py full nano                    # -> ~/.cache/onnxsim-rfdetr/work/nano.{sim,u8}.onnx
python3 phone_eval.py nano ~/.cache/onnxsim-rfdetr/work/nano.u8.onnx --input u8
python3 export.py full nano@320 && python3 phone_eval.py nano@320 $W/nano@320.u8.onnx
FOLD=1 python3 graph.py $W/nano.raw.onnx $W/nano_fold.sim.onnx $W/nano_fold.u8.onnx   # (repo onnxsim env)
python3 phone_eval.py nano $W/nano_fold.u8.onnx
python3 cross_eval.py nano@320.u8 gt           # needs ~/.cache/coco/instances_val2017.json
PYTHONPATH=<repo> python3 quantize.py nano bb8g16 && python3 host_eval.py nano $W/nano.bb8g16.onnx
python3 phone_eval.py nano $W/nano.u8.onnx --n 1 --iters 3 --profile && \
  python3 profile_regions.py $W/nano.u8.onnx $W/phone_nano.u8/pulled/prof.csv
```

## Demo app

The demo app (`../../maskrcnn_demo_app`, "RF-DETR" button) runs Nano @320 fp16 as one strict-HTP
graph in its YOLO activity (`post=detr`): 25-26 FPS on test images (27 ms inference), 30 FPS from
the camera (camera-capped; 26 ms HTP + 4 ms YUV conversion). See that README's RF-DETR section.
