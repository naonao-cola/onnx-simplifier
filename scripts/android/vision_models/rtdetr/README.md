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
| **int8 backbone + hybrid encoder, fp16 query selection + decoder (`front8`, uint8 NHWC input)** | **19.3** | **52** | 183/195 (93.8%) |
| front8, MSE calibration | 19.3 | | 183/195 |
| front8, 64 calibration images | 18.9 | | 182/195 |
| front8 with the AIFI layer's LayerNorm/Softmax fp16 (`front8s`) | 20.3 | | 180/195 |
| front8 with the whole AIFI layer fp16 (`front8a`) | 19.4 | | 182/195 |
| front8 + decoder Linears int8 (`front8lin`) | 25.2 | | 182/195 |
| everything int8 but LayerNorm/Softmax/GridSample (`all8`) | 20.8 | | 0/195 |
| all8 + int8 GridSample with uint16 sampling coordinates (`mix8`) | 25.6 | | 0/195 |

For context, YOLO11n is ~2.6 ms on the same HTP (`../../deploy/`, 88.8-90.4% vs fp32).

- **Quantize only the CNN part** (backbone, hybrid encoder). The decoder in int8 is either slower
  (its Linears: quantize/dequantize around fp16 neighbours) or broken (box refinement, reference
  points and the `Log`s of inverse-sigmoid in uint8). The ~6% loss is in the CNN itself: keeping
  AIFI in fp16 doesn't recover it.
- **The AIFI MLP's exact GELU** (`Div, Erf, Add, Mul, Mul`) stays fp16 as a whole. QNN EP fuses the
  float pattern into Gelu, but it has no Erf op of its own, float or quantized. An int8 Erf node
  unit, or a float Erf between quantized neighbours, lands on the CPU.
- **The uint8 NHWC input.** `full_qdq.quantized_io`'s `uint8 -> Transpose -> DQ` gets a lone uint8
  Transpose refused here (0xc26). `quantize.py` makes it a data-movement QDQ unit instead
  (`DQ -> Transpose -> Q -> DQ`, the same scale/zero point, exact).

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
