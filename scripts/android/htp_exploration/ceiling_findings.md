# How close is the backbone to the HTP's ceiling? 18% as shipped, 60% after five graph rewrites

Measured on the test phone (Snapdragon 8+ Gen 1, Hexagon V69 HTP, device `239dbd8f`) with the
`qnn_shell/` harness from `qnn_shell_findings.md` (ORT 1.26 + QNN EP 2.6.0 + Qualcomm
`qnn-runtime` 2.50.0, unsigned PD, strict: no CPU fallback, burst mode unless noted). All numbers
are steady-state medians; scripts are in `ceiling/`. Qualcomm publishes no HTP-only peak for
this SoC, so "ideal" here is the **practical ceiling measured on the same phone**, not a spec
sheet.

## 1. The practical ceiling: ~16 TMAC/s int8

`ceiling/gen_ceiling_models.py` builds chains of L identical layers in the backbone's own QDQ
convention (uint8 activations, int8 weights, int32 bias, per-tensor); `run_ceiling.sh` times
L=2 and L=6 and `summarize_ceiling.py` differences them, so graph launch, the fp32 input quantize
/ output dequantize and host copies cancel out. A Q->DQ-only model (fixed per-inference cost)
runs in 0.14 ms.

| layer | variant | GMAC/layer | per-layer ms | TMAC/s |
|---|---|---:|---:|---:|
| 1x1 conv 256->256 @64x64 | u8 | 0.27 | 0.022 | 11.9 |
| 1x1 conv 512->512 @64x64 | u8 | 1.07 | 0.073 | 14.8 |
| 1x1 conv 1024->1024 @64x64 | u8 | 4.29 | 0.285 | 15.1 |
| 1x1 conv 2048->2048 @64x64 | u8 | 17.18 | 1.265 | 13.6 |
| 1x1 conv 1024->1024 @128x128 | u8 | 17.18 | 1.337 | 12.8 |
| 1x1 conv 2048->2048 @128x128 | u8 | 68.72 | 5.345 | 12.9 |
| **3x3 conv 256->256 @128x128** | u8 | 9.66 | 0.607 | **15.9** |
| 3x3 conv 512->512 @64x64 | u8 | 9.66 | 0.675 | 14.3 |
| MatMul 4096x4096x4096 | u8 | 68.72 | 6.465 | 10.6 |
| 1x1 conv 1024->1024 @64x64 | per-channel weights | 4.29 | 0.317 | 13.5 |
| 3x3 conv 256->256 @128x128 | per-channel weights | 9.66 | 0.613 | 15.8 |
| 1x1 conv 1024->1024 @64x64 | uint16 activations | 4.29 | 0.555 | 7.7 |
| 3x3 conv 256->256 @128x128 | uint16 activations | 9.66 | 1.145 | 8.4 |

- **Plateau: 13-16 TMAC/s for u8 x s8; the best, 15.9 TMAC/s, is used as the ceiling below.**
  Throughput peaks at mid-size layers and sags slightly for the largest (activations far beyond
  on-chip memory).
- **Data-independent**: the 3x3 case with random input instead of zeros measured 0.612 ms/layer
  (15.8 TMAC/s), same as zeros.
- **Per-channel weight scales cost nothing measurable** (within noise), so the backbone's
  per-tensor weights are not a speed choice -- per-channel would be free if accuracy wanted it.
- **16-bit activations halve throughput**, as expected.

At the ceiling, the backbone's 159.2 GMAC would take **10.0 ms**.

## 2. Where the shipped backbone's 54 ms goes

QNN's own profiler (`profiling_level=basic` for calibrated totals, `detailed` for per-node cycles;
`ceiling/profile_backbone.sh`, `ceiling/analyze_profile.py`):

- Steady state: **54 ms wall = 43.2 ms on the accelerator + ~10 ms host-side** inside QNN's
  execute (graph I/O handling around the HTP run). 159.2 GMAC / 54 ms = 2.95 TMAC/s =
  **18.5% of the ceiling** (3.7 TMAC/s = 23% counting accelerator time only).
- Detailed profiling serializes ops (total accelerator cycles rise ~1.8x; 48 of 220 QNN nodes
  report 0 cycles, so it isn't a flat per-node tax), so per-node cycles are used only as **relative
  shares**, scaled to the basic-mode accelerator time. Per-conv TMAC/s derived that way comes out
  above the ceiling for many convs, so it is **not** reported; only class-level shares are.

Accelerator time by op class, as shipped:

| op class (QNN nodes) | share |
|---|---:|
| standalone QuantizeLinear | 36% |
| standalone DequantizeLinear | 21% |
| Conv (quantized conv units) | 13% |
| Relu | 8% |
| graph output handling, input quantize, layout transposes, RPN reshapes | ~20% |

**Convs are 13%**; the rest is data movement and float glue. The cause is in the model, not QNN:
each bottleneck ends `DQ(conv) + shortcut -> Add -> Relu -> Q`, and in 12 of the 16 blocks the
Relu output *also* feeds the next block's Add **in float** (the identity shortcut is never
quantized). Neither the Add (its output goes to Relu, not Q) nor the Relu (its input isn't a DQ)
forms a QDQ node unit, so QNN runs Add, Relu and the Q/DQ around them as float ops on
full-resolution tensors (e.g. `37_38_QuantizeLinear`, 256x200x272, costs 40x one of the block's
convs).

## 3. Five rewrites, measured one at a time

All are ONNX-level rewrites of `backbone.onnx` (`ceiling/make_optimized.sh` applies them in
order and reproduces the measured model byte-for-byte):

| step | rewrite | numerics | wall ms | accel ms | TMAC/s (wall) | % of ceiling |
|---|---|---|---:|---:|---:|---:|
| 0 | as shipped | -- | 54.3 | 43.2 | 2.93 | 18% |
| 1 | residual Adds as int8 QDQ units (`quantize_residuals.py`) | shortcut quantized to uint8 | 30.4 | 21.5 | 5.24 | 33% |
| 2 | uint8 graph outputs (`quantized_outputs.py`) | lossless | 21.5 | 17.9 | 7.41 | 47% |
| 3 | raw RPN conv outputs, no per-anchor layout chain (`raw_rpn_outputs.py`) | lossless* | 20.8 | 17.4 | 7.65 | 48% |
| 4 | uint8 NHWC image input (`quantized_input.py nhwc`) | lossless | 19.4 | 16.6 | 8.21 | 52% |
| 5 | NHWC FPN outputs (`nhwc_fpn_outputs.py`) | lossless | 17.1 | 14.4 | 9.33 | 59% |
| 5 | same, EP-context cache, no profiler attached | | **16.7** | | **9.52** | **60%** |
| 5 | same, default performance mode | | 18.7 | | 8.50 | 53% |

\* scores become logits; applying sigmoid in float afterwards matches the old quantized sigmoid
output to within half a uint8 step (max 0.002).

- **Step 1 is the big one (-24 ms).** Relu is dropped into the following QuantizeLinear, which is
  exact (every one is uint8 with zero point 0, so Q already clamps at 0), and the next block's
  Add reads `DQ(Q(block output))` instead of the float Relu output. After this all 19 Adds are
  `DQ,DQ -> Add -> Q` units that QNN runs as int8 adds. Quantizing the shortcut is the only
  numeric change in the whole series, and it is what any int8 backend must do anyway (TVM's
  `FakeQuantizationToInteger` did the same).
- **Steps 2-5 are all interface changes: the backbone hands out what the HTP already has.**
  Every output was `DQ(Q(x))` in NCHW; the input was fp32 NCHW quantized on the HTP. Emitting
  uint8, raw RPN conv outputs and NHWC removes ~13 ms of dequantize, layout transposes and
  fp32 traffic (the P2 FPN map alone is 55.7 MB as fp32). This project's DSP kernels already take
  these forms: the fast RoiAlign kernel wants NHWC FPN maps (`../tinygrad_hexagon_bridge/`
  RoiAlign Stage 3) and the fused RPN kernel reads raw uint8 conv deltas (PR #1830).
- The 1-2 image-prep steps this moves to the consumer (quantize the image; dequantize/lay out the
  outputs) are cheap on their own and mostly fold into ops that already exist downstream; they
  are *not* timed here.

**QNN knobs, measured on the step-3 model (all negative or neutral):**

| option | wall ms (vs 20.7-20.8) |
|---|---:|
| `htp_graph_finalization_optimization_mode=3` | **24.4** (slower, reproduced twice) |
| `vtcm_mb=8` | 20.6 (noise) |
| `offload_graph_io_quantization=0` / `=1` | 20.7 / 20.8 (noise) |

## 4. Accuracy of the optimized model

Final model's HTP outputs (converted back to the original output form) vs the **original**
`backbone.onnx` on host ORT CPU, next to the unmodified model on the HTP (`cats.jpg`,
`ceiling/compare_optimized.py`, `qnn_shell/compare.py`, `qnn_shell/detect_compare.py`):

| | FPN max abs | FPN mean abs | deltas max abs | scores max abs |
|---|---:|---:|---:|---:|
| unmodified model on HTP | 0.81-1.08 | 0.093-0.114 | 0.043-0.188 | 0.094-0.129 |
| optimized model on HTP | 0.93-1.15 | 0.096-0.119 | 0.036-0.188 | 0.097-0.163 |

Detections through `rest.onnx` (host ORT) vs the all-ORT pipeline: unmodified 3/3 matched, box
IoU 0.985, mask IoU 0.985, score diff 0.011; optimized 3/3 matched, box IoU 0.969, mask IoU 0.965,
score diff 0.007. Same int8-noise class, slightly larger box/mask deviation from the quantized
shortcut; **one image only** (the six COCO images of `../maskrcnn_e2e/README.md` aren't on this
machine).

## 5. The remaining gap: 16.7 ms vs 10.0 ms ideal

Detailed-profile shares for the step-4 model (relative, scaled to its 16.6 ms accelerator time;
host-side ~2 ms on top). Step 5 then removed ~2.2 ms of accelerator time, almost all of it from
the output/transpose row (one QNN node, an NHWC->NCHW transpose of an FPN map, was 4% alone):

| op class | share | ~ms (step 4) |
|---|---:|---:|
| Conv | 53% | 8.8 |
| quantized Add (residual + FPN merge) | 26% | 4.3 |
| graph output write + layout transposes | 17% | 2.8 |
| MaxPool + Resize | 5% | 0.8 |

Ranked levers, **not measured here**:
1. **The 19 quantized Adds (~4 ms)** are memory-bound elementwise passes over full-resolution
   tensors. Fusing each residual Add into the preceding conv's output stage would remove them;
   that's a QNN op-fusion question (no ONNX-level form found) or a different export of the model.
2. **The 7x7 stem conv** (cin=3) is by far the least efficient conv relative to its MACs in both
   profiles (it maps poorly onto a matrix engine built for wide channel dims; the same effect as
   `hex_stem7x7_kernel.py`'s cin padding story). A space-to-depth reformulation (e.g. 2x2
   pixel-unshuffle to cin=12 with a 4x4-ish kernel) is the standard fix; needs retraining-free
   weight rearrangement and a re-check of numerics.
3. **Host-side ~2 ms** (ORT session run + FastRPC round trip + input/output copies): an
   ION/rpcmem-backed I/O binding (`enable_htp_shared_memory_allocator`) would skip the copies.
4. Batching doesn't apply to single-image inference; burst vs default is already covered above.
