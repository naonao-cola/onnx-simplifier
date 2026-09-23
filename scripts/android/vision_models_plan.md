# Bringing BEVFormer and other recent vision models to this phone's Hexagon

A roadmap, not an implementation. It says, per model, what would run where on this phone
(Xiaomi 12S, Snapdragon 8+ Gen 1, Hexagon V69), what blocks it, what already-built pieces
transfer, and what the first concrete step is. **Measured facts and estimates are kept apart**:
every number is either tagged *measured* (with the file it came from) or *estimate* (with the
reasoning). Probe scripts are under `vision_models_probe/`; they download models rather than
committing them.

## What the Mask R-CNN work established (the evidence this plan builds on)

All *measured* on this phone unless noted.

| Fact | Source |
|---|---|
| QNN HTP runs from a plain `adb shell` process: ORT 1.26 + `onnxruntime-android-qnn` 2.6.0 + Maven `com.qualcomm.qti:qnn-runtime` 2.50.0, unsigned PD, no root | `htp_exploration/qnn_shell_findings.md` (#1829) |
| Practical int8 ceiling ~16 TMAC/s (best case 15.9, 3x3 conv 256→256 @128²); per-channel weight scales cost nothing; **int16 activations halve throughput** | `htp_exploration/ceiling_findings.md` (#1833) |
| A QDQ model can run far below that ceiling for model-level reasons: Mask R-CNN backbone was 18% as shipped (float residual Adds), 60% after 5 graph rewrites (int8 residuals, uint8/NHWC I/O, raw head outputs) | `ceiling_findings.md` |
| QNN rejects dynamic-shape regions: `Cannot get shape` (data-dependent RoI count), `NonZero` output not static, `ConstantOfShape` unsupported. Cutting static subgraphs out and pinning shapes lets them run 100% on HTP (Mask R-CNN mask head 18x vs 4-thread CPU) | `htp_exploration/rest_htp_findings.md` (#1832) |
| HTP is bad at data-dependent gathers that need a big feature map re-uploaded per call (RoiAlign: 2.4x *slower* than CPU as its own graph) | `rest_htp_findings.md` |
| Exact HVX DSP kernels exist for RoiAlign (channels-last bilinear gather, 2.1x ORT), NMS (bitmask, 3.2x per-level), TopK (select + counting sort, 2.3x), proposal decode, and a fused RPN post-proc (2.8–3.4x). V69 HVX has **no IEEE fp32** — only qfloat; exactness needs threshold rechecks + an asm barrier against LLVM qf32 folding. ~0.25 ms fixed cost per FastRPC round trip | `tinygrad_hexagon_bridge/README.md`, `dynamic_ops_survey.md` |
| Full Mask R-CNN end to end: 131–158 ms/image (vs 1.0–1.5 s all-CPU ORT), accuracy on par with the reference, using the partition pattern **HTP for static dense parts, HVX for dynamic/gather ops, CPU for the remainder** | `e2e_pipeline/README.md` (#1841) |
| A precompiled EP-context model loads in ~0.5 s instead of ~6 s but made the box head ~1.7x slower (not root-caused) | `e2e_pipeline/README.md` |

### How estimates below are made

Latency *estimate* = (model int8 GMAC) / (16 TMAC/s × utilization) + non-MAC/overhead terms, with
utilization bracketed by what Mask R-CNN actually got: **18%** (a QDQ model as shipped, float
side paths intact) to **60%** (after model-level rewrites). Transformer blocks are expected at or
below the low end until measured, because their MatMuls are small per token and softmax/LayerNorm
are non-MAC work. None of these estimates replaces a phone run.

## Ranked roadmap (summary)

_Filled in as each model is probed; see the per-model sections._

## 1. BEVFormer

_In progress._

## Appendix: probe method

_In progress._
