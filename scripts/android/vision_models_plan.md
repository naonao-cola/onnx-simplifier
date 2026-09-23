# Bringing BEVFormer and other recent vision models to this phone's Hexagon

A roadmap. For each model it says what would run where on this phone (Xiaomi 12S, Snapdragon 8+
Gen 1 / SM8475, Hexagon V69), what blocks it, which already-built pieces transfer, and the first
concrete step. **Measured facts and estimates are kept apart**: every number is tagged
*measured* (phone run, with the script that reproduces it) or *estimate* (with the reasoning).
Probe scripts are in `vision_models_probe/`; they download models rather than committing them.

## Ranked roadmap

| # | Model | Where it runs | Measured on the phone so far | Main blocker | Effort | First step |
|---|---|---|---|---|---|---|
| 1 | **YOLO11n** (detect; `-seg` for masks) | whole net on HTP; box decode + NMS on CPU or our HVX NMS kernel | fp32 graph: **0 ops refused, 10.0 ms strict all-HTP** (fp16) | none known; needs static int8 PTQ | small | int8 QDQ PTQ + COCO accuracy + FPS through the new deploy pipeline (`deploy/`) |
| 2 | **Depth Anything V2 Small** | whole net on HTP | fp32: **0 refused, 111 ms (fallback allowed) / 129 ms strict** (fp16) | int8 PTQ of a ViT (LayerNorm/Softmax/GELU accuracy); int16 activations halve throughput | medium | int8 QDQ with fp16/int16 islands for LayerNorm+Softmax; compare depth error vs fp32 |
| 3 | **RT-DETR r18vd** | backbone + encoder on HTP; decoder too once rank is fixed; TopK on HTP/HVX | fp32: **39 of 695 ops refused, all rank-6** (deformable-attention decoder); 90.9 ms with CPU fallback, strict fails | HTP rank-5 limit in the decoder's deformable attention | medium | the same rank-5 rewrite as BEVFormer (shared work), then strict all-HTP |
| 4 | **BEVFormer-tiny** | ResNet-50 backbone on HTP; encoder/decoder on HTP after rank-5 rewrite; deformable sampling possibly on HVX; TopK on HVX | one encoder layer at real tiny dims: rank-6 formulation **18 of 91 ops refused**, 190–215 ms with CPU fallback; rank-5 formulation **0 refused, 107 ms strict all-HTP** (fp16), but output only cos 0.986 vs fp32 | encoder speed (~15 GMAC/s effective) and an unexplained fp16 precision loss; needs a real exported checkpoint | large | per-op QNN profile of one encoder layer; bisect the precision loss |
| 5 | SAM-family small encoder (MobileSAM/EfficientSAM), SegFormer, DINOv2-S | ViT/Mix-transformer: likely whole net on HTP | not probed | same ViT int8 issues as #2 | medium | reuse #2's recipe; probe with `partition_report.sh` |

The order is value per unit of effort. YOLO11n is a complete, useful demo that needs nothing
new. Depth Anything is a clean ViT and teaches the int8-transformer recipe that SAM, SegFormer
and DINOv2 need too. RT-DETR and BEVFormer share the deformable-attention rank problem, so fixing
it once (item 3) pays for both.

## What the Mask R-CNN work established (the evidence this plan builds on)

All *measured* on this phone.

| Fact | Source |
|---|---|
| QNN HTP runs from a plain `adb shell` process: ORT 1.26 + `onnxruntime-android-qnn` 2.6.0 + Maven `com.qualcomm.qti:qnn-runtime` 2.50.0, unsigned PD, no root | `htp_exploration/qnn_shell_findings.md` (#1829) |
| Practical int8 ceiling ~16 TMAC/s (best case 15.9, 3x3 conv 256→256 @128²); per-channel weight scales cost nothing; **int16 activations halve throughput** | `htp_exploration/ceiling_findings.md` (#1833) |
| A QDQ model can run far below that ceiling for model-level reasons: the Mask R-CNN backbone was 18% as shipped (float residual Adds), 60% after 5 graph rewrites (int8 residuals, uint8/NHWC I/O, raw head outputs) | `ceiling_findings.md` |
| QNN rejects dynamic-shape regions: `Cannot get shape` (data-dependent RoI count), `NonZero` output not static, `ConstantOfShape` unsupported. Cutting static subgraphs out and pinning shapes lets them run 100% on HTP (Mask R-CNN mask head 18x vs 4-thread CPU) | `htp_exploration/rest_htp_findings.md` (#1832) |
| HTP is bad at data-dependent gathers that re-upload a big feature map per call (RoiAlign as its own HTP graph: 2.4x *slower* than CPU) | `rest_htp_findings.md` |
| Exact HVX DSP kernels exist for RoiAlign (channels-last bilinear gather, 2.1x ORT), NMS (bitmask, 3.2x per-level), TopK (select + counting sort, 2.3x), proposal decode, and a fused RPN post-proc (2.8–3.4x). V69 HVX has **no IEEE fp32**, only qfloat; exactness needs threshold rechecks and an asm barrier against LLVM folding qf32 conversions. ~0.25 ms fixed cost per FastRPC round trip | `tinygrad_hexagon_bridge/README.md`, `dynamic_ops_survey.md` |
| Full Mask R-CNN end to end: 131–158 ms/image (vs 1.0–1.5 s all-CPU ORT), accuracy on par with the reference, using the pattern **HTP for static dense parts, HVX for dynamic/gather ops, CPU for the remainder** | `e2e_pipeline/README.md` (#1841) |
| A precompiled EP-context model loads in ~0.5 s instead of ~6 s but made the box head ~1.7x slower (not root-caused) | `e2e_pipeline/README.md` |

### New, general finding from these probes: the HTP's rank-5 limit

*Measured.* Every op QNN refused in both transformer probes touches a rank-6 or rank-7 tensor, and
no op of rank ≤ 5 was refused (`summarize_partition.py` maps logcat's
`Failed to validate op <name> with error 0xc26` back to the graph):

| Model | Ops | Refused | Refused ops' max tensor rank |
|---|---:|---:|---|
| BEVFormer-tiny encoder layer (mmdeploy op → `rewrite_msdeformattn_to_gridsample`) | 91 | 18 | 15 at rank 6, 3 at rank 7 |
| RT-DETR r18vd (Hugging Face export, onnxsim with fixed 640x640) | 695 | 39 | 39 at rank 6 |
| YOLO11n, Depth Anything V2 Small | 320, 660 | 0 | — |

The cause is the standard multi-scale deformable-attention layout
`(batch, queries, heads, levels, points, 2)`: six dims before any batching. Both onnxsim's own
`rewrite_msdeformattn_to_gridsample` pass and the Hugging Face RT-DETR export produce it. Folding
`levels` or `heads` into a neighbouring dim keeps the math identical and gets under the limit:
`bevformer_tiny_encoder.py --rank5` does this by hand, gives output identical to the rank-6
formulation in torch (max abs diff 0.0), and runs **strict all-HTP**. The reusable fix is a
rank-reducing mode for `rewrite_msdeformattn_to_gridsample`, plus a generic "keep rank ≤ 5"
rewrite for exported graphs like RT-DETR's.

### Other finding: pre-quantized ONNX exports are not HTP-ready

*Measured.* The onnx-community `model_int8.onnx` files for RT-DETR and Depth Anything are
**dynamic** quantization (`DynamicQuantizeLinear` + `ConvInteger`/`MatMulInteger`), not the
static QDQ the HTP runs. Every model here needs our own static PTQ, with per-channel weights (free
on the HTP) and uint8 activations.

### How the estimates below are made

Latency *estimate* = (model int8 GMAC) / (16 TMAC/s × utilization) + non-MAC/overhead terms.
Utilization is bracketed by what Mask R-CNN actually got: **18%** (a QDQ model as shipped) to
**60%** (after model-level rewrites). Transformer blocks are expected at or below the low end
until measured, because their MatMuls are small per token and Softmax/LayerNorm do no MACs.
None of these estimates replaces a phone run.

## 1. YOLO11n (implement first)

- **Source.** `aaurelions/yolo11n.onnx` on the Hub (an Ultralytics export: opset 19, 320 nodes,
  2.7 M params, input `1x3x640x640`, output `1x84x8400` raw box+class scores, no NMS in the
  graph). `vision_models_probe/fetch_models.sh`.
- **Measured.** fp32 graph through the QNN EP: **0 ops refused, 1 QNN graph**, 10.0 ms strict
  all-HTP (fp16 on the HTP), 11.6 ms with fallback allowed.
- **Plan.**
  - The network is 88 Conv + SiLU (`Sigmoid`*`Mul`) + a small attention block (2 MatMul, 2 Softmax
    in the C2PSA layer), so it's the same shape of work as the Mask R-CNN backbone.
  - Static int8 QDQ PTQ on real images, per-channel weights, uint8 I/O.
  - Box decode + class max + NMS on the CPU first. The output is only 84x8400, so this is cheap.
    The HVX NMS kernel is an option if the CPU part dominates.
  - `-seg` (masks) is the same plus a prototype-mask MatMul, also static.
- **Estimate.** ~3.3 GMAC at 640² → 0.3–1.2 ms of MAC time at 18–60% of 16 TMAC/s, so latency
  will be dominated by fixed per-inference overhead (the ~2.5 ms FastRPC/graph launch seen for tiny
  models in `ceiling_findings.md`) and I/O. Expect **~3–6 ms int8** (estimate), i.e. well over
  100 FPS for the network alone.
- **Effort/risk.** Small. The main risk is int8 accuracy on the detection head's
  class/box outputs; the fallback is keeping the last head convs in fp16.
- **First step.** Deploy through `scripts/android/deploy/` (the reusable pipeline):
  fetch → onnxsim fixed shape → PTQ on COCO val images → partition report → EP-context → phone
  benchmark + mAP-style agreement vs fp32 ORT.

## 2. Depth Anything V2 Small

- **Source.** `onnx-community/depth-anything-v2-small` (fp32 `model.onnx`; the `model_int8.onnx`
  there is dynamic quantization, see above). Fixed to 518x518 with onnxsim: 660 nodes (ViT-S/14
  DINOv2 encoder + DPT head).
- **Measured.** fp32: **0 refused, 1 QNN graph**; 111 ms with fallback allowed, 129 ms strict
  (fp16).
- **Estimate.** ViT-S at 37x37 = 1369 tokens: ~22 GMAC for the encoder at 518², plus the DPT head
  (~8 GMAC), about 30 GMAC total. Int8 at 18–60% would be 3–10 ms of MAC time, but attention and
  LayerNorm are non-MAC work and int16 halves throughput, so **~20–50 ms int8** (estimate).
- **Plan / risk.** PTQ int8 for Linear/Conv, keeping LayerNorm, Softmax and GELU in fp16 or int16
  where accuracy needs it (measure per-block). Accuracy metric: relative depth error / δ<1.25 vs
  fp32 on real images. Medium effort: this is the recipe SAM, SegFormer and DINOv2 reuse.

## 3. RT-DETR r18vd

- **Source.** `onnx-community/rtdetr_r18vd` (Hugging Face export, opset 16, dynamic input; fixed to
  640x640 with onnxsim: 695 nodes).
- **Measured.** fp32: **39 of 695 ops refused, every one rank 6** (the decoder's multi-scale
  deformable attention: `Gather`, `Slice`, `Mul`, `Add`, `Sub`, `Div`, `Reshape`, `Unsqueeze`
  around the 9 `GridSample`s). 4 QNN graphs, **90.9 ms with CPU fallback; strict all-HTP fails**.
- **Plan.**
  - Apply the rank-5 rewrite (the shared work item with BEVFormer), which should give strict
    all-HTP.
  - The final `TopK(300)` is static (fixed k) and already runs on the HTP.
  - NMS-free, so there's no dynamic tail at all.
- **Estimate.** r18 backbone + hybrid encoder + 3-layer decoder, ~30 GMAC at 640² → **~15–40 ms
  int8** once all-HTP (estimate; the decoder's GridSample speed on the HTP is the unknown, see
  BEVFormer's 107 ms layer).
- **Effort.** Medium. The rank rewrite is the only new piece, and it also unblocks BEVFormer.

## 4. BEVFormer-tiny (primary target, hardest)

- **Source situation.** There's no public deployable BEVFormer ONNX (Hub search: none). Exporting
  the official checkpoint needs mmdet3d + mmcv + the BEVFormer repo, and mmdeploy's custom op for
  deformable attention.
  - This repo already handles that op: `rewrite_msdeformattn_to_gridsample` (#1157) decomposes it
    to `GridSample`, with Z3 proofs and torch end-to-end tests.
  - The `~/bev-tmp/bevformer_tiny.onnx` from earlier QNN checks is **not** BEVFormer-tiny. It's a
    1208-node reconstruction of the Qwen-Drive BEV head at 4x4 images with random weights
    (`check_qnn_bev.py`).
  - Its "exact on qnn-htp" result came from the **host** QNN HTP emulator (the x86 QNN EP has no
    HTP device; #1807 showed that path falls back), so it says nothing about the phone.
- **Measured (this plan).** `vision_models_probe/bevformer_tiny_encoder.py` rebuilds one
  BEVFormer-tiny encoder layer at the real config sizes, with random weights:
  - Sizes: BEV 50x50 = 2500 queries, embed 256, 8 heads, one image level of 15x25 (stride 32 of
    480x800), 6 cameras, 8 spatial points (4 pillar anchors x 2), 4 temporal points.
  - Formulation: temporal self-attention + spatial cross-attention + FFN, in the deploy-style
    all-queries-plus-mask form. Upstream selects visible queries per camera with `.nonzero()`,
    which gives a data-dependent query count the HTP can't take.

  On the phone:

  | Variant | QNN refused | Result |
  |---|---:|---|
  | mmdeploy op → onnxsim GridSample rewrite | 18 of 91, all rank 6/7 | 190–215 ms with CPU fallback, 3 QNN graphs |
  | rank-5 formulation (identical output in torch) | **0** | **107 ms strict all-HTP** (fp16), cos 0.986 / mean abs err 0.13 vs fp32 ORT |

- **Open problems.**
  - **Speed.** ~1.6 GMAC per layer (estimate) in 107 ms is ~15 GMAC/s effective, ~0.1% of the
    int8 ceiling. The likely cost is `GridSample` on the HTP, to be confirmed by a per-op profile.
    Its access pattern (per query, per head: bilinear sample of a channels-last `(H, W, 32)` map)
    is exactly what `roialign_fast/` already does on HVX, so an HVX deformable-sampling kernel is
    the likely fix if the profile agrees.
  - **Precision.** cos 0.986 is far worse than fp16 should give; something in the layer loses
    precision on the HTP. Bisect per op before trusting any encoder output.
  - **Scale.** BEVFormer-tiny has 3 encoder layers + a 6-layer decoder (900 queries), and the
    temporal loop feeds `prev_bev` back each frame (host-side buffer swap).
- **Estimate.**
  - **Backbone:** ResNet-50 on 6 cameras at 480x800 is ~188 GMAC, comparable to Mask R-CNN's
    159 GMAC backbone (16.7–54 ms measured), so **~20–65 ms**.
  - **Encoder/decoder:** at today's measured 107 ms/layer in fp16, that's ~1 s. With the sampling
    moved to HVX and int8 Linear layers, **~50–150 ms** (estimate, low confidence).
  - **Post-processing:** a fixed-k TopK over 900x10, which the existing HVX TopK kernel covers.
- **Steps, in order.**
  1. Per-op QNN profile of the rank-5 encoder layer (`htp_exploration/ceiling/profile_backbone.sh`
     pattern).
  2. Bisect the fp16 precision loss.
  3. Rank-reducing mode for `rewrite_msdeformattn_to_gridsample` (shared with RT-DETR).
  4. Real checkpoint export (mmdeploy route, under the 16 GB memory cap) and a nuScenes-mini frame
     for accuracy.
  5. An HVX deformable-sampling kernel if step 1 says `GridSample` dominates.
- **Effort/risk.** Large. The real risks are exporting the official model and the encoder's
  speed and precision on the HTP.

## 5. Others (not probed yet)

- **MobileSAM / EfficientSAM encoder.** TinyViT/ViT encoders plus a small prompt decoder, all
  static. Same int8-ViT recipe as Depth Anything. The prompt decoder is tiny and can stay on the
  CPU.
- **SegFormer-B0.** Mix-transformer with efficient attention plus an all-MLP head. Static; likely
  the whole net on the HTP. A good segmentation demo after YOLO11n-seg.
- **DINOv2-S.** The Depth Anything encoder without the DPT head, so it's covered by #2.

Probe any of them with `vision_models_probe/partition_report.sh` (fix the input shape with onnxsim
first; QNN needs static shapes).

## Appendix: probe method

- `fetch_models.sh`: downloads the Hub ONNX files into `models/` (ignored by git).
- `bevformer_tiny_encoder.py [--rank5] [--layers N]`: exports the encoder probe. The default goes
  through the mmdeploy custom op and onnxsim's rewrite; `--rank5` uses the rank-5 formulation and
  checks it matches the default in torch.
- `partition_report.sh <model.onnx> <manifest>`: runs with CPU fallback allowed (verbose ORT log
  to logcat), then strict all-HTP, then prints `summarize_partition.py`'s summary: refused ops by
  type, error code and tensor rank, QNN graph count, and both medians.
- Every heavy host step (torch export, onnxsim on large graphs) was run under
  `systemd-run --user -p MemoryMax=16G`.
- Phone: device `239dbd8f`, `QNN_PERF=burst`, fp32 models, so the HTP runs fp16. No int8 numbers
  in this doc yet; those come from the deploy pipeline.
