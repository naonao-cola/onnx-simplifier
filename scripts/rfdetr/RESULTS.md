# RF-DETR × onnxsim — captured results

Result of running `simplify_rfdetr.py --all` (plus the two `[plus]`
segmentation variants) — exporting every RF-DETR model variant to ONNX via
the public `RFDETR*.export()` API and simplifying it with onnxsim,
reproducing RF-DETR's own `onnxsim.simplify(check_n=3, ...)` call.

## Environment

| package | version |
|---|---|
| rfdetr | 1.8.3 |
| onnxsim | 0.7.0 |
| onnx | 1.22.0 |
| onnxruntime | 1.28.0 |
| torch | 2.13.0+cpu |
| numpy | 2.4.6 |
| Python | 3.11.15 (Linux, CPU) |

Export opset 17, static batch size 1, default per-variant resolution.

## Results

| Variant | Head | Input | Nodes before → after | Reduction | `check_ok` |
|---|---|---|---|---|:---:|
| RFDETRNano | detect | 384² | 1478 → 770 | 48% | ✅ |
| RFDETRSmall | detect | 512² | 1634 → 844 | 48% | ✅ |
| RFDETRMedium | detect | 576² | 1792 → 918 | 49% | ✅ |
| RFDETRBase | detect | 560² | 1580 → 809 | 49% | ✅ |
| RFDETRLarge | detect | 704² | 1792 → 918 | 49% | ✅ |
| RFDETRSegPreview | segment | 432² | 1879 → 983 | 48% | ✅ |
| RFDETRSegNano | segment | 312² | 1836 → 962 | 48% | ✅ |
| RFDETRSegSmall | segment | 384² | 1879 → 983 | 48% | ✅ |
| RFDETRSegMedium | segment | 432² | 2050 → 1069 | 48% | ✅ |
| RFDETRSegLarge | segment | 504² | 2050 → 1069 | 48% | ✅ |
| RFDETRKeypointPreview | keypoint | 576² | 3743 → 1392 | 63% | ✅ |
| RFDETRSegXLarge †  | segment | 624² | 2221 → 1155 | 48% | ⚠️ |
| RFDETRSeg2XLarge † | segment | 768² | 2221 → 1155 | 48% | ⚠️ |

† requires `pip install rfdetr[plus]`.

**11 / 13 variants simplify and pass onnxsim's numerical check out of the
box.** Every variant simplifies structurally (~48–63% fewer nodes).

## The two ⚠️ variants are functionally equivalent

`RFDETRSegXLarge` / `RFDETRSeg2XLarge` simplify correctly but report
`check_ok=False`. This is a strict-tolerance artifact, not a wrong graph.
See **[`FAILURE_ANALYSIS.md`](FAILURE_ANALYSIS.md)** for the full
onnxruntime-based investigation; the short version:

onnxsim's `check_n` compares original vs. simplified outputs with
`np.allclose(rtol=1e-4, atol=1e-5)` (`onnxsim/model_checking.py`). onnxsim's
**constant folding** (`Mul` 150→140, `Add` 244→236, mostly in the deformable
cross-attention decoder — there are **no** BatchNorm nodes to fuse; the ViT
backbone uses LayerNorm) bakes precomputed constants in at slightly
different fp rounding. That sub-`1e-5` perturbation enters the DINOv2 ViT
backbone encoder and **accumulates through 12+ residual transformer layers**
into the mask head. onnxruntime comparison:

```
dets    maxdiff ~1e-4    (values in [0,1])
labels  maxdiff ~1e-3    (raw logits)
masks   maxdiff ~1e-2..1.6e-1  (raw mask logits, 300×156×156)
no NaNs in either graph
```

The first tensor to break tolerance is a LayerNorm output at maxdiff
`1.5e-5` — LayerNorm emits near-zero values, so `allclose` there collapses
to the `atol=1e-5` floor. **Predictions are unaffected**: 100% class-argmax
agreement, identical top-10 detections, sub-pixel boxes, and ~0.0001% of
mask pixels flip at the 0.5 threshold. The failure is a scale effect of the
strict check on a deep transformer, not a wrong simplification. Consumers
who want the simplified XL models can pass `check_n=0` (skip the check) or
verify with a task-level tolerance.

## Notes

- **onnxruntime is required for the check.** Without it, onnxsim falls back
  to onnx's Python reference evaluator, whose `GatherElements` uses
  `numpy.choose` (capped at 64 choices) and cannot run RF-DETR's decoder —
  the check then raises `ValueError: Need at least 0 and at most 64 array
  objects` instead of simplifying. RF-DETR's `[onnx]` extra already requires
  onnxruntime, so a correct RF-DETR install never hits this.
- **RF-DETR pins `onnxsim<0.6.0`** in its `[onnx]` extra, but these exports
  all simplify with onnxsim 0.7.0 — the pin's upper bound could be relaxed.
