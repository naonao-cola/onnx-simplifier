# RF-DETR on the Xiaomi 12S (Snapdragon 8+ Gen 1, Hexagon V69)

[RF-DETR](https://github.com/roboflow/rf-detr) (Roboflow, 2025; Apache-2.0 for Nano/Small/Medium): a
DINOv2 ViT-S backbone with windowed + global attention (patch 16), an LW-DETR projector to a single
P4 feature map, and a 2-4 layer deformable-attention decoder (300 queries, 16 cross-attention heads
of dim 16, 2 points), with two-stage query selection. Loaded through the `rfdetr` package (1.10.1,
checkpoints md5-checked by the package; `rf-detr-nano.pth` md5 `fb6504cce7fbdc783f7a46991f07639f`).

| variant | resolution | tokens | decoder layers | params |
|---|---|---|---|---|
| Nano | 384 | 24x24 | 2 | 30.5 M |
| Small | 512 | 32x32 | 3 | |
| Medium | 576 | 36x36 | 4 | |

## Files

| file | what |
|---|---|
| `common.py` | eval images (the deploy pipeline's yolo11n COCO ids, the same RT-DETR used), RF-DETR preprocessing / decode, detection matching |
| `model.py` | the library model in export mode + a rank <= 4 `MSDeformAttn.forward` |
| `graph.py` | onnxsim, the rank-5 window-transpose fold, the uint8 NHWC input with the normalization folded into the patch-embed conv |
| `export.py` | `refs` / `validate` / `full` |
| `phone_eval.py` | strict all-HTP run over the 20 eval images under the phone lock, detections vs the fp32 library model |

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

Matched = detections of the fp32 library model (score >= 0.3) found by the phone with the same class
at IoU >= 0.5.

| model | precision | ms | matched vs fp32 |
|---|---|---|---|
| Nano, 384 | fp16 | 47.1 | 142/142 |

## Reproduce

Heavy steps under `systemd-run --user --wait --collect --pipe -p MemoryMax=12G -p MemorySwapMax=0`,
in a venv with `rfdetr==1.10.1`, CPU torch, onnx and onnxruntime:

```
export ONNXSIM_REPO=/path/to/onnx-simplifier   # a checkout with the built onnxsim extension
python3 export.py validate nano                # rank<=4 MSDA vs the library
python3 export.py full nano                    # -> ~/.cache/onnxsim-rfdetr/work/nano.{sim,u8}.onnx
python3 phone_eval.py nano ~/.cache/onnxsim-rfdetr/work/nano.u8.onnx --input u8
```
