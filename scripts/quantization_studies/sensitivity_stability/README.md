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

## Answer

**Partly structural, partly checkpoint-specific; the ranking is mostly shared, the magnitude is not.**

- **Where it hurts is largely fixed by the architecture.** In `only` mode, block rankings agree
  across training recipes with Spearman 0.66–0.92, and 2–3 of the top-3 blocks are shared in every
  pair. The worst region is the same in all five ResNet-50s (the first bottleneck `layer1.0` plus
  the stem, or `layer1.2` for A2) and in both ViT-S (blocks 7–9, then 11). On the ViTs, residual
  `Add` and `LayerNormalization` carry 88–89% of the per-op-type noise in both checkpoints.
- **But checkpoints differ more than data noise.** The noise floor, the same checkpoint on another
  calibration/eval split, is Spearman 0.97–0.99. Every checkpoint-to-checkpoint pair (0.66–0.92)
  falls below it, so the ranking does move with training, mainly below the top few blocks.
- **How much it hurts is checkpoint-specific.** All-uint8 SQNR spans 16.1 dB (torchvision V2) to
  33.2 dB (timm A3) across five ResNet-50s with identical graphs. At the same 28 dB budget, V1
  needs 6 groups promoted and V2 needs 17. The concentration of the noise also changes: the top-3
  blocks hold 39–79% of it.
- **Outlier channels persist, but move.** Both ViT-S have a massive-activation channel in the
  middle residual stream. It is channel 202 (up to 73× the median, blocks 5–11) in AugReg and
  channel 256 (about 27×, blocks 4–7) in DeiT. The phenomenon and its depth are structural; the
  index and the size come from training.
- **Policies transfer one way.** A policy searched on a harder checkpoint meets the budget on an
  easier one (V2→V1, A1→A2/A3), but over-provisions 2–3× the nodes. The reverse misses: V1→V2
  gives 19.8 dB against a 28 dB budget. Between the two ViTs it misses narrowly in both directions
  (14.3 and 14.9 dB against 15), even though 10 of 11 or 13 groups are shared.
- **`all_but` rankings are noisy.** Their noise floor is only Spearman 0.52–0.77, so with a small
  eval set, rank by `only` mode and use `all_but` just to confirm the top few.

**Practical guidance**
- Seed the policy from structure: stem and first stage, the middle ViT blocks, and
  Add/LayerNorm/Softmax in float or 16-bit.
- Re-run `search_activation_precision_for_budget` for every new checkpoint or fine-tune. It is
  cheap: one calibration pass, block groups, about 10–40 min on the host here.
- If one policy must serve several checkpoints, use the union of their searched policies and
  verify each checkpoint against the budget.

## Caveats
- Classification only, on 64 calibration and 64 eval images (coco128, center crops, used here as a
  distribution sample, not for accuracy). The metric is logit SQNR; top-1 agreement is reported for
  the transfer runs.
- Activations are uint8 per-tensor; weights are int8 per-channel (`onnxsim.full_qdq`). No SmoothQuant
  or other outlier migration, which would change the ViT picture.
- The detector pair (YOLO11n vs YOLO26n) was not run.

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

### Policy transfer (`transfer.py`, groups promoted = float/uint16)

| A -> B | budget | A: groups promoted (float/uint16), nodes | A policy on B | B own: groups, nodes, score | B groups within A |
|---|---|---|---|---|---|
| tv_r50_v1 -> tv_r50_v2 | 35 dB | 17 (17/0), 112 | 28.5 dB, top-1 0.97, MISSES | 18 (18/0), 120, 200.0 dB | 17/18 |
| tv_r50_v2 -> tv_r50_v1 | 35 dB | 18 (18/0), 120 | 200.0 dB, top-1 1.00, meets | 17 (17/0), 112, 35.4 dB | 17/17 |
| timm_r50_a1 -> timm_r50_a2 | 35 dB | 8 (7/1), 48 | 35.5 dB, top-1 0.97, meets | 4 (4/0), 24, 35.7 dB | 4/4 |
| timm_r50_a1 -> timm_r50_a3 | 35 dB | 8 (7/1), 48 | 38.5 dB, top-1 0.95, meets | 2 (1/1), 15, 35.4 dB | 2/2 |
| deit_s -> vit_s_augreg | 25 dB | 14 (13/1), 321 | 25.0 dB, top-1 0.95, meets | 14 (13/1), 321, 30.1 dB | 14/14 |
| vit_s_augreg -> deit_s | 25 dB | 14 (13/1), 321 | 26.0 dB, top-1 0.95, meets | 14 (13/1), 321, 27.9 dB | 14/14 |
| tv_r50_v1 -> tv_r50_v2 | 28 dB | 6 (6/0), 34 | 19.8 dB, top-1 0.95, MISSES | 17 (16/1), 113, 29.1 dB | 6/17 |
| tv_r50_v2 -> tv_r50_v1 | 28 dB | 17 (16/1), 113 | 38.5 dB, top-1 1.00, meets | 6 (6/0), 34, 28.1 dB | 6/6 |
| deit_s -> vit_s_augreg | 15 dB | 13 (13/0), 314 | 14.3 dB, top-1 0.81, MISSES | 11 (11/0), 243, 21.0 dB | 10/11 |
| vit_s_augreg -> deit_s | 15 dB | 11 (11/0), 243 | 14.9 dB, top-1 0.89, MISSES | 13 (13/0), 314, 16.0 dB | 10/13 |

### ViT residual-stream outlier channels (`outliers.py`, max|x| per channel over 32 images vs the median channel)

| checkpoint | block: channel (× median) |
|---|---|
| vit_s_augreg | L5 ch202 29×, L6 ch202 73×, L7 ch202 71×, L8 ch202 60×, L9 ch202 47×, L10 ch202 26×, L11 ch202 24× |
| deit_s | L4 ch256 22×, L5 ch256 27×, L6 ch256 26×, L7 ch256 23× |
