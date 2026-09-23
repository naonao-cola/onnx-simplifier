# Is quantization sensitivity fixed by the architecture, or does it move with training?

Host-only study with `onnxsim.analyze_activation_sensitivity` (activation quantization, uint8,
whole-graph QDQ via `onnxsim.full_qdq`, ORT at the basic optimization level). The same
architecture is compared across checkpoints trained with different recipes or data. Each
checkpoint-to-checkpoint agreement is compared against a **noise floor**: the same checkpoint
re-run on a different calibration/eval split.

## Setup

| family | checkpoints (pinned) | what differs |
|---|---|---|
| ResNet-50, torchvision | `IMAGENET1K_V1` (`resnet50-0676ba61.pth`), `IMAGENET1K_V2` (`resnet50-11ad3fa6.pth`) | recipe (V2: long schedule, heavy augmentation, EMA) |
| ResNet-50, timm | `resnet50.a1_in1k`, `a2_in1k`, `a3_in1k` | "ResNet strikes back" recipes A1/A2/A3 |
| ViT-S/16 | `deit_small_patch16_224.fb_in1k`, `vit_small_patch16_224.augreg_in21k_ft_in1k` | recipe and data (DeiT on IN-1k vs AugReg pretrained on IN-21k) |

- **Data:** coco128 images, center-cropped to 224. Split A = first 64 calibration / last 64 eval; split B (noise floor) = even / odd.
- **Metric:** mean over eval images of the logits' SQNR vs fp32 (dB). Per-group harm is compared on a linear scale: in `only` mode, the noise power one quantized group adds (10^(-SQNR/10)); in `all_but` mode, the fraction of the all-uint8 noise removed by keeping that group in float.
- **Groups:** ResNet bottlenecks (`layerN.M`, 16) + stem + head; ViT blocks (12) + patch-embed + head.

## Reproduce

```
python export.py                       # static fp32 ONNX, opset 17, into $SENSSTAB_WORK (~/.cache/sensstab)
python run_sensitivity.py tv_r50_v1 A  # block only / all_but + op-type only -> results/<model>_<split>.json
python analyze.py                      # the tables below
python transfer.py deit_s vit_s_augreg 25   # policy transfer (budget in dB)
python outliers.py vit_s_augreg deit_s # residual-stream outlier channels
```

Each heavy step ran under `systemd-run --user -p MemoryMax=16G`; a sweep takes ~4–6 min per checkpoint on the host.

## Results


### block_only: Spearman of block sensitivity, top-k overlap, top-3 share of total

| pair | kind | groups | Spearman | top-3 | top-5 | top-3 share (a / b) |
|---|---|---|---|---|---|---|
| tv_r50_v1/A vs tv_r50_v1/B | noise floor (other data split) | 18 | +0.97 | 2/3 | 5/5 | 0.66 / 0.68 |
| tv_r50_v1/A vs tv_r50_v2/A | same arch, new recipe | 18 | +0.70 | 3/3 | 4/5 | 0.66 / 0.48 |
| timm_r50_a1/A vs timm_r50_a1/B | noise floor (other data split) | 18 | +0.97 | 3/3 | 4/5 | 0.72 / 0.74 |
| timm_r50_a1/A vs timm_r50_a2/A | same arch, recipe A1 vs A2 | 18 | +0.92 | 3/3 | 4/5 | 0.72 / 0.79 |
| timm_r50_a1/A vs timm_r50_a3/A | same arch, recipe A1 vs A3 | 18 | +0.88 | 2/3 | 3/5 | 0.72 / 0.71 |
| timm_r50_a2/A vs timm_r50_a3/A | same arch, recipe A2 vs A3 | 18 | +0.79 | 2/3 | 4/5 | 0.79 / 0.71 |
| tv_r50_v1/A vs timm_r50_a1/A | same arch, torchvision vs timm | 18 | +0.74 | 2/3 | 4/5 | 0.66 / 0.72 |
| tv_r50_v2/A vs timm_r50_a1/A | same arch, torchvision V2 vs timm A1 | 18 | +0.66 | 2/3 | 4/5 | 0.48 / 0.72 |
| deit_s/A vs deit_s/B | noise floor (other data split) | 14 | +0.99 | 3/3 | 5/5 | 0.40 / 0.39 |
| deit_s/A vs vit_s_augreg/A | same arch, DeiT vs AugReg (IN-21k) | 14 | +0.90 | 2/3 | 4/5 | 0.40 / 0.52 |

### block_all_but: Spearman of block sensitivity, top-k overlap, top-3 share of total

| pair | kind | groups | Spearman | top-3 | top-5 | top-3 share (a / b) |
|---|---|---|---|---|---|---|
| tv_r50_v1/A vs tv_r50_v1/B | noise floor (other data split) | 18 | +0.67 | 3/3 | 4/5 | 0.70 / 0.60 |
| tv_r50_v1/A vs tv_r50_v2/A | same arch, new recipe | 18 | +0.54 | 2/3 | 3/5 | 0.70 / 0.67 |
| timm_r50_a1/A vs timm_r50_a1/B | noise floor (other data split) | 18 | +0.52 | 1/3 | 3/5 | 0.70 / 0.68 |
| timm_r50_a1/A vs timm_r50_a2/A | same arch, recipe A1 vs A2 | 18 | +0.36 | 1/3 | 3/5 | 0.70 / 0.81 |
| timm_r50_a1/A vs timm_r50_a3/A | same arch, recipe A1 vs A3 | 18 | +0.83 | 1/3 | 4/5 | 0.70 / 0.77 |
| timm_r50_a2/A vs timm_r50_a3/A | same arch, recipe A2 vs A3 | 18 | +0.29 | 2/3 | 3/5 | 0.81 / 0.77 |
| tv_r50_v1/A vs timm_r50_a1/A | same arch, torchvision vs timm | 18 | +0.20 | 2/3 | 2/5 | 0.70 / 0.70 |
| tv_r50_v2/A vs timm_r50_a1/A | same arch, torchvision V2 vs timm A1 | 18 | +0.43 | 2/3 | 2/5 | 0.67 / 0.70 |
| deit_s/A vs deit_s/B | noise floor (other data split) | 14 | +0.77 | 3/3 | 4/5 | 0.41 / 0.39 |
| deit_s/A vs vit_s_augreg/A | same arch, DeiT vs AugReg (IN-21k) | 14 | +0.65 | 1/3 | 4/5 | 0.41 / 0.59 |

### Op-type ranking (only mode), worst first

| checkpoint | all-uint8 SQNR | op types, worst first (nodes, share of summed noise) |
|---|---|---|
| deit_s | 1.7 dB | Add (85 nodes, 46%), LayerNormalization (25 nodes, 42%), Erf (12 nodes, 5%), MatMul (72 nodes, 3%) |
| timm_r50_a1 | 27.1 dB | Relu (49 nodes, 48%), Conv (53 nodes, 37%), Add (16 nodes, 13%), GlobalAveragePool (1 nodes, 1%) |
| timm_r50_a2 | 28.0 dB | Relu (49 nodes, 36%), Conv (53 nodes, 35%), Add (16 nodes, 28%), Gemm (1 nodes, 1%) |
| timm_r50_a3 | 33.2 dB | Relu (49 nodes, 43%), Conv (53 nodes, 41%), Add (16 nodes, 14%), Gemm (1 nodes, 1%) |
| tv_r50_v1 | 22.0 dB | Relu (49 nodes, 50%), Conv (53 nodes, 40%), Add (16 nodes, 9%), Gemm (1 nodes, 1%) |
| tv_r50_v2 | 16.1 dB | Relu (49 nodes, 38%), Conv (53 nodes, 35%), Add (16 nodes, 24%), Gemm (1 nodes, 2%) |
| vit_s_augreg | -0.0 dB | Add (85 nodes, 48%), LayerNormalization (25 nodes, 41%), Erf (12 nodes, 4%), MatMul (72 nodes, 3%) |

### Worst 3 blocks (only mode)

| checkpoint | worst blocks (share of summed per-block noise) |
|---|---|
| deit_s_A | blocks/blocks.8 (14%), blocks/blocks.11 (14%), blocks/blocks.7 (12%) |
| deit_s_B | blocks/blocks.8 (14%), blocks/blocks.11 (13%), blocks/blocks.7 (12%) |
| timm_r50_a1_A | layer1/layer1.0 (41%), stem (18%), layer1/layer1.2 (13%) |
| timm_r50_a1_B | layer1/layer1.0 (44%), stem (17%), layer1/layer1.2 (13%) |
| timm_r50_a2_A | layer1/layer1.2 (42%), layer1/layer1.0 (23%), stem (13%) |
| timm_r50_a3_A | layer1/layer1.0 (48%), stem (17%), layer1/layer1.1 (6%) |
| tv_r50_v1_A | layer1/layer1.0 (34%), stem (26%), head (7%) |
| tv_r50_v1_B | layer1/layer1.0 (32%), stem (31%), layer1/layer1.2 (5%) |
| tv_r50_v2_A | layer1/layer1.0 (26%), stem (13%), head (9%) |
| vit_s_augreg_A | blocks/blocks.7 (21%), blocks/blocks.8 (16%), blocks/blocks.9 (15%) |
