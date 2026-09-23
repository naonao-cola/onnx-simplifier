# Quantization sensitivity: which layers break, and how to find them

Practical notes from quantizing ~20 vision and language models for the
Hexagon HTP (Xiaomi 12S, SM8475 / V69) with onnxsim's own quantizers
(`scripts/android/`). They answer three questions: *what kind of layer
is usually sensitive*, *is that fixed by the architecture or does it
move with the checkpoint*, and *which onnxsim tool finds it*.

## 1. Sensitivity has a structural part and a checkpoint part

**Structural -- predictable from the architecture** (the same op types
broke across unrelated trainings):

| pattern | why | seen in |
|---|---|---|
| Softmax, LayerNorm/RMSNorm internals, attention scores | wide or heavy-tailed range by construction | every transformer below |
| geometry / coordinate math (grid construction, projected points) | Sub/Add of large coordinates, then used as sample locations | BEVFormer SCA grid (worst op types in `bevformer_tiny/sensitivity.py`), Sparse4D keypoints behind a camera |
| one tensor mixing different units | one scale can't cover both | YOLO11 final Concat of box pixels (0-640) and class probs (0-1): keep it float |
| score / classification heads | clipping the rare high scores kills detections while barely moving L2 | YOLO11n/YOLO26n (entropy and tight percentiles lose detections) |
| first conv on raw pixels, prediction heads | input/output range, few channels | most detectors |
| transformer residual-stream outlier channels | "massive activations" appear in almost every trained transformer | SmolLM2-135M layer 12 channel ~2e4 (overflowed fp16 RMSNorm), DINOv2 in RF-DETR, SAM ViTs |

**Checkpoint-specific -- moves with the training method and data:**

- *which* channel and layer carry the outliers (the phenomenon persists,
  the position does not);
- which of several similar blocks is the worst;
- per-channel weight ranges, especially depthwise convs;
- activation distributions after fine-tuning (calibration ranges shift and
  the ranking can reorder);
- how concentrated the error is: EdgeSAM's int8 error is spread ~1% per
  node across the whole network (keeping 96/194 nodes in fp16 still only
  reached embedding cos 0.93), RT-DETR-r18's int8 loss splits across the
  backbone and encoder, while BEVFormer's concentrates in the grid ops.
  YOLO11n and YOLO26n (similar architectures, different training) needed
  different calibration methods (MSE vs percentile with the score path at
  exact range).

A controlled measurement of how much block rankings move between
checkpoints of the same architecture (ResNet-50 V1/V2 and timm recipes,
ViT-S AugReg vs DeiT, with a calibration-noise floor and policy transfer)
is in `scripts/quantization_studies/sensitivity_stability/` once that
study lands.

**The target backend adds its own failures** -- found only by comparing
device against host per tensor, never by a host sweep:

| failure | backend behaviour | fix |
|---|---|---|
| RMSNorm `mean(x^2)` in fp16 | HTP overflows at 65504 (host ORT accumulates wider); dividing by the row max instead flushed normal channels to zero | scale each row so its max is 128 (exact normalization) |
| 16-bit x 16-bit MatMul | wrong on the HTP, near-exact on host (StreamPETR head) | keep that head fp16 |
| uint8 graph input -> `Cast` | HTP miscomputes (backbone cos 0.03-0.15) | `DequantizeLinear(scale=1, zp=0)` instead |
| broadcast Mul along a middle axis | HTP miscomputes (cos 0.23) | broadcast along the last axis |
| attention mask `-3.4e38` | becomes -inf in fp16 and breaks quantization | bound it to `-1e4` (masks identically) |
| fp16 keypoints behind a camera | overflow in the projection | exact 1 cm depth floor + clamp |

## 2. Practical guidance

1. **Treat a mixed-precision policy as a property of architecture x
   checkpoint x calibration data x backend.** Re-run sensitivity after
   every re-train or fine-tune; don't copy a policy from another
   checkpoint without re-measuring.
2. **Seed with structural rules, then let data decide the rest:** keep
   Softmax/LayerNorm in fp16, keep graph outputs and Sigmoid/Softmax
   outputs (and score heads) at exact range, don't share one scale
   between tensors of different units -- then search for the
   checkpoint-specific blocks.
3. **Score with the task metric, not output L2.** A detector can lose
   detections while relative L2 barely moves (clipped class scores).
4. **Cross-fit calibration choices.** Holding out 16 of 64 calibration
   images ranked percentile-99.99 first for YOLO11n -- wrong on 128
   images and on the phone; 4-fold cross-fitting picked MSE, which the
   phone confirmed (596/659 vs 575/659).
5. **Beware guards that match too much.** SiLU is `x * Sigmoid(x)`; a
   "never clip Sigmoid inputs" guard first protected almost every YOLO
   conv output (569 -> 589 matched boxes once it skipped that pattern).
6. **Confirm on the device.** Section 1's backend failures are invisible
   on the host.
7. **Test the quantized graph, not the CPU's int8 kernels.** ORT's fused
   DQ->Gemm/MatMul/Conv->Q int8 kernels saturate on x86 CPUs without VNNI
   (CI runners): compare with graph optimizations at BASIC or disabled.
8. **Post-training quantization has a floor.** When error is spread
   evenly (EdgeSAM, EfficientViT-SAM int8, RF-DETR's DINOv2 at int8
   activations), no policy recovers it cheaply -- that needs outlier
   migration (SmoothQuant), or QAT/distillation.

## 3. Tools in onnxsim

| question | tool |
|---|---|
| which tensors *can* be quantized | `onnxsim_cpp2py_export.list_quantizable_activations` (used by `calibrate`) |
| apply a mixed-precision *policy* (which op types / nodes get int8, uint16, or stay float) | `onnxsim.full_qdq.quantize_full_qdq(op_types=..., exclude_nodes=..., activation_dtype=...)` |
| which calibration *method* (minmax / mse / percentile / entropy), per model or per tensor | `onnxsim.pick_calibration` (task metric, cross-fitted), `calibrate(method="auto")` |
| **which activation groups are sensitive**, and the cheapest policy meeting a budget | `onnxsim.analyze_activation_sensitivity`, `onnxsim.search_activation_precision_for_budget` (node / op-type / block groups; "only" and "all_but" sweeps; uint8 -> uint16 -> float ladder; one calibration pass) -- added by PR #1935 |
| **weight-only** int4 vs int8 per layer (LLM-style) | `onnxsim.apply_mixed_precision_quantization`, `search_mixed_precision_for_budget` -- see [mixed-precision.md](mixed-precision.md) |
| whole-model accuracy drop / a global scheme | `onnxsim.accuracy.measure_accuracy_drop`, `recommend_quantization` -- see [accuracy-drop.md](accuracy-drop.md) |
| migrate activation outliers into weights before int8 | `onnxsim.apply_smoothquant` |

On-device bisect (expose intermediates, run on the target, per-tensor
cosine vs host) lives next to each model because the runner is
device-specific: `scripts/android/vision_models/bevformer_tiny/bisect_precision.py`,
`scripts/android/vision_models/sparse4d/bisect_phone.py`,
`scripts/android/llm_tinygrad/bisect_llm.py`.

## 4. Case studies (numbers from `scripts/android/`)

| model | what was sensitive | kind | outcome |
|---|---|---|---|
| YOLO11n | final Concat (mixed units); class-score clipping | structural | Concat kept float; MSE calibration: 596/659 vs fp32 (ORT MinMax 585) |
| YOLO26n | score path | structural + checkpoint | percentile + score path at exact range: 482/565 on the phone |
| BEVFormer-tiny encoder | grid-construction Sub/Add/Reshape feeding GridSample | structural | int8 GridSample only viable with a tighter (exact) ref clamp; after moving sampling to HVX, int8 encoder is accurate but slower on the HTP |
| RT-DETR-r18 | spread over backbone (~6) and encoder (~8 detections) | checkpoint | encoder at 16-bit activations: 190/195 vs 183 at int8 |
| StreamPETR head | HTP 16x16-bit MatMul (backend) | backend | head stays fp16 |
| SmolLM2-135M | residual outlier channel -> fp16 RMSNorm overflow (backend + outliers) | structural + backend | exact row rescale: 9/10 prompts identical over 32 tokens |
| EdgeSAM / EfficientViT-SAM | error spread evenly | checkpoint | PTQ floor; stays fp16 |
| RF-DETR (DINOv2) | int8 activations lose 25+ detections | outliers | fp16 default; SmoothQuant + activation-sensitivity search in progress |

Per-model details: each model's README under `scripts/android/`.
