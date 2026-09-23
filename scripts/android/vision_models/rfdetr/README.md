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
| Small @512 | fp16 | 91.2 | 11 | 141/142 | 70.1% / 65.3% (fp32: 70.1% / 66.2%) |
| **Nano @384** | **fp16** | **46.7** | **21** | **142/142** | **67.9% / 64.1%** (= fp32) |
| Nano @384 | W8A16, GELU 16-bit (`bb16g`) | 49.4 | 20 | 137/142 | 67.9% / 64.1% |
| Nano @384 | int8, GELU 16-bit (`bb8g16`) | 37.3 | 27 | 116/142 | 61.9% / 62.4% |
| **Nano @320** | **fp16** | **26.4** | **38** | 125/126 | **63.4% / 67.5%** (= fp32) |
| Nano @256 | fp16 | 20.1 | 50 | 124/125 | 56.0% / 59.5% (fp32: 56.0% / 60.0%) |

The @320 / @256 rows are RF-DETR's weight-sharing operating points: the same Nano weights at a
lower resolution (positional encodings interpolated by the library). For context, on the same
phone: RT-DETR-r18 at 640 runs in 18.3 ms (`../rtdetr/`, int8 backbone + HVX MSDA), YOLO26n in
~2.6 ms on the HTP (`../../deploy/`).

**fp16 is the recommendation, and resolution is the speed knob.** Nano @320 fp16 (26.4 ms) is both
faster and more accurate than any int8 Nano @384.

### Where the time goes (Nano @384 fp16, `profile_regions.py`, shares of a detailed-profile run)

| region | share |
|---|---|
| backbone attention | 29.5% |
| backbone MLP | 18.4% |
| backbone norms, LayerScale, residuals, windowing | 13.2% |
| heads / unnamed | 19.5% |
| decoder deformable cross-attention (MSDA) | 6.6% |
| decoder FFN / norms, self-attention, query selection, projector | ~11% |

By op: GELU 18.4%, MatMul + Gemm 28%, Add + Mul 24%, LayerNorm 7%. The DINOv2 ViT-S is the cost.

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

The first `bb8` attempt failed to compile (QNN `graphAddNode` 6007 on a Quantize): onnxsim renames
the Linear layers to `Gemm_NN`, so a name-based backbone region left them float between int8
neighbours. `quantize.py` now finds the backbone structurally (everything upstream of the named
`/backbone/` nodes' outputs).

This matches the SAM ViT encoders (`../sam/`): post-training int8 of DINOv2-style activations
loses too much; the RT-DETR CNN backbone quantizes fine. Quantization-aware fine-tuning would be
the way to a fast int8 RF-DETR, not calibration.

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
python3 cross_eval.py nano@320.u8 gt           # needs ~/.cache/coco/instances_val2017.json
PYTHONPATH=<repo> python3 quantize.py nano bb8g16 && python3 host_eval.py nano $W/nano.bb8g16.onnx
python3 phone_eval.py nano $W/nano.u8.onnx --n 1 --iters 3 --profile && \
  python3 profile_regions.py $W/nano.u8.onnx $W/phone_nano.u8/pulled/prof.csv
```
