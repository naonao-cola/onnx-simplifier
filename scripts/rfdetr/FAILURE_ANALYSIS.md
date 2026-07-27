# Why `RFDETRSegXLarge` / `RFDETRSeg2XLarge` report `check_ok=False`

These two `[plus]` segmentation variants are the only RF-DETR exports that
fail onnxsim's `check_n` verification. This is an onnxruntime-backed
inspection of *why*. Reproduce with:

```bash
python scripts/rfdetr/inspect_failures.py --dir rfdetr_onnx --stem rfdetr-seg-xlarge
python scripts/rfdetr/inspect_failures.py --dir rfdetr_onnx --stem rfdetr-seg-2xlarge
```

**Bottom line: the simplified graph is functionally correct.** The failure is
onnxsim's strict `np.allclose(rtol=1e-4, atol=1e-5)` check flagging
floating-point noise that accumulates across a very deep transformer, not a
wrong transformation. Both variants behave identically (same architecture,
different input resolution: 624² vs 768²).

## 1. The drift is intrinsic, not an artifact of random input

onnxruntime comparison of original vs. simplified outputs under three input
regimes (2 seeds each):

| input regime | `dets` maxdiff | `labels` maxdiff | `masks` maxdiff | passes check? |
|---|---|---|---|:---:|
| uniform `[0,1)` | ~2–5e-4 | ~6e-4 | ~6e-3–2e-2 | no |
| ImageNet-normalized (what RF-DETR feeds) | ~3–5e-4 | ~1–3e-3 | ~1–5e-2 | no |
| zeros | ~4e-4 | ~1e-3 | ~3e-2 | no |

The check fails under **all** regimes — including the ImageNet-normalized
input RF-DETR actually passes to `onnxsim.simplify`. So RF-DETR's own export
would see `check_ok=False` here too; it isn't caused by the harness's random
input.

## 2. It does not change predictions

Post-processing the raw outputs (ImageNet input, 3 seeds) shows the
detections are effectively identical:

| metric | result |
|---|---|
| per-query argmax class agreement | **100%** |
| top-10 query set match | **10 / 10** |
| class score maxdiff (post-sigmoid) | ~2e-5 |
| box maxdiff (normalized coords) | ~1–5e-4 (sub-pixel) |
| mask-pixel flips at 0.5 threshold | **~0.0001%** (a few boundary px of 7.3M) |

The ~0.15 raw-`masks` maxdiff is on pre-sigmoid logits; after
sigmoid+threshold it moves essentially no pixels.

## 3. Where the drift originates

Exposing intermediate tensors with onnxruntime and comparing every tensor
common to both graphs (in topological order):

- **First divergence is deep in the DINOv2 ViT backbone encoder**
  (`backbone.0/encoder/.../layer.8/norm2`, then `layer.10`, `layer.11`), at
  `LayerNormalization` / `Mul` / `Add` outputs. The op *types* are identical
  in both graphs — onnxsim did not swap these ops.
- The very first tensor over tolerance is a LayerNorm output with maxdiff
  **1.49e-5**. LayerNorm emits near-zero values, so at those elements the
  `allclose` bound collapses to the `atol=1e-5` floor and a ~1.5e-5
  difference violates it.
- From there the difference **accumulates through 12+ residual transformer
  layers** and the segmentation mask head (upsampling convs over 156²),
  reaching ~1e-1 on raw mask logits. 193 of 554 compared tensors drift.

### What onnxsim actually changed

There are **zero `BatchNormalization` nodes** — the ViT backbone uses
LayerNorm, so this is *not* BN-into-Conv fusion. The structural diff shows
onnxsim's **constant folding** removed:

- `Mul`: 150 → 140 (10 folded)
- `Add`: 244 → 236 (8 folded)

located almost entirely in the **deformable cross-attention decoder**
(`transformer/decoder/layers.*/cross_attn`) plus one in the backbone
encoder. Folding these constant subexpressions bakes precomputed constants
into the graph at slightly different floating-point rounding than the
original runtime ops. Those sub-1e-5 constant perturbations are the seed
that the deep residual stack amplifies.

## Why only XL / 2XL

The smaller detection and segmentation variants pass the same check. XL/2XL
have the deepest/widest DINOv2 backbone, so the same class of fp
perturbation accumulates further — and the large mask head amplifies it —
until it crosses onnxsim's strict floor. It is a scale effect on the check's
tolerance, not a new failure mode.

## Recommendation / fix

The simplification is safe to use for these variants. onnxsim exposes the
check tolerance so the verification can succeed without disabling it:

```python
model_opt, ok = onnxsim.simplify(
    model, check_n=3, check_rtol=1e-2, check_atol=1e-3
)  # ok == True for SegXLarge/Seg2XLarge
```

or on the command line:

```bash
onnxsim in.onnx out.onnx 3 --check-rtol 1e-2 --check-atol 1e-3
```

The looser tolerance is appropriate here because the difference is bounded
floating-point noise from correct op reordering, not a wrong graph (the
default `rtol=1e-4, atol=1e-5` stays in force for every other model).
Alternatively, run `onnxsim.simplify(..., check_n=0)` and validate at the
task level (class/box/mask agreement, as in `inspect_failures.py`).
