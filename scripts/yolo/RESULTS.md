# YOLO × onnxsim — captured results

Result of running `simplify_yolo.py --all` (plus the YOLO26-only
`depth` / `sem` / `p2` / `p6` heads) — exporting the newest Ultralytics YOLO
generations to ONNX via `YOLO.export(format="onnx")` and simplifying each
with onnxsim, reproducing Ultralytics' own `simplify=True` step
(`onnxsim.simplify(check_n=3)`).

## Environment

| package | version |
|---|---|
| ultralytics | 8.4.112 |
| onnxsim | 0.7.0 |
| onnx | 1.22.0 |
| onnxruntime | 1.28.0 |
| torch | 2.13.0 |
| numpy | 2.4.6 |
| Python | 3.11.15 (Linux, CPU) |

Export opset 17, static batch size 1, nano (`n`) scale. Input 640² for every
head except classification (224²). Models built from Ultralytics YAML configs
(random weights) — the graph structure is identical to the pretrained `.pt`
models.

## Results — newest generations × every task head

| Spec | Task | Input | Nodes before → after | Reduction | `check_ok` |
|---|---|---|---|---|:---:|
| yolo26n | detect | 640² | 550 → 384 | 30% | ✅ |
| yolo26n-seg | segment | 640² | 625 → 434 | 30% | ✅ |
| yolo26n-pose | pose | 640² | 613 → 419 | 31% | ✅ |
| yolo26n-obb | obb | 640² | 595 → 421 | 29% | ✅ |
| yolo26n-cls | classify | 224² | 185 → 144 | 22% | ✅ |
| yolo12n | detect | 640² | 744 → 496 | 33% | ✅ |
| yolo12n-seg | segment | 640² | 791 → 531 | 32% | ✅ |
| yolo12n-pose | pose | 640² | 799 → 530 | 33% | ✅ |
| yolo12n-obb | obb | 640² | 780 → 532 | 31% | ✅ |
| yolo12n-cls | classify | 224² | 446 → 282 | 36% | ✅ |
| yolo11n | detect | 640² | 429 → 318 | 25% | ✅ |
| yolo11n-seg | segment | 640² | 476 → 353 | 25% | ✅ |
| yolo11n-pose | pose | 640² | 484 → 352 | 27% | ✅ |
| yolo11n-obb | obb | 640² | 465 → 354 | 23% | ✅ |
| yolo11n-cls | classify | 224² | 185 → 144 | 22% | ✅ |

**15 / 15 specs simplify and pass onnxsim's numerical check.**

## Results — YOLO26-only heads

The latest generation adds heads the prior ones don't ship. These also
simplify cleanly:

| Spec | Task | Input | Nodes before → after | Reduction | `check_ok` |
|---|---|---|---|---|:---:|
| yolo26n-depth | depth | 640² | 429 → 325 | 24% | ✅ |
| yolo26n-sem | semantic | 640² | 286 → 221 | 22% | ✅ |
| yolo26n-p2 | detect | 640² | 671 → 476 | 29% | ✅ |
| yolo26n-p6 | detect | 640² | 713 → 510 | 28% | ✅ |

**19 / 19 specs across all tested heads pass out of the box.**

## Summary

- **Every tested spec simplifies and passes `check_n=3`.** No numerical-check
  failures on any current YOLO generation or head — unlike the deep RF-DETR
  transformer variants, YOLO's CNN backbones don't accumulate enough
  folding perturbation to break onnxsim's strict `allclose` tolerance.
- **Structural reduction is 22–36%.** onnxsim folds the anchor-grid /
  shape-arithmetic constants in the detection head and fuses Conv+BatchNorm
  in the backbone. YOLO12's area-attention ("A2") backbone exports the most
  redundant nodes and sees the largest reduction (~31–36%).
- The simplified graphs load and run in onnxruntime with unchanged I/O
  signatures (verified by `tests/test_yolo.py`).
