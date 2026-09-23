# Mask R-CNN end to end on the phone, one process: 131-158 ms per image

This replaces the "~191 ms per image (sum of separately measured pieces)" estimate in
`../htp_exploration/rest_htp_findings.md` with a real measurement. One native process on the test
phone (Snapdragon 8+ Gen 1, Hexagon V69, device `239dbd8f`) runs the whole model on real images:
each region runs on its fastest proven engine, and the tensors between regions stay in that
process.

| region | engine | from |
|---|---|---|
| backbone (ResNet-50 + FPN + RPN head) | HTP via ORT + QNN EP | PR #1829, optimized per PR #1833 |
| RPN post-processing (TopK, decode, NMS, merge -> 1000 proposals) | HVX DSP, one FastRPC call | PR #1830 (`rpn_fused/`) |
| RoiAlign (4 box + 4 mask calls) | HVX DSP | PRs #1819/#1824 (`roialign_fast/`) |
| box head, mask head | HTP (static batches 1000 / 32 or 100) | PR #1832 |
| everything else (level assignment, merges, box post-processing, 80 per-class NMS, mask paste) | ORT CPU, 4 threads | |

All numbers below are steady-state medians of 7 runs per image, after 1 untimed and 2 warm-up runs.
ORT CPU uses 4 intra-op threads. The phone was shared with other agents' jobs, so individual steps
jitter by a few ms between runs.

## Result

Per-image wall time in ms, six real images: five COCO val2017 (000000000139, 632, 724, 785, 1000)
and `cats.jpg`. Every stage runs on all six in one process.

| stage | what runs where | 139 | 632 | 724 | 785 | 1000 | cats |
|---|---|---:|---:|---:|---:|---:|---:|
| ref | whole model, ORT CPU | 1358 | 1441 | 1131 | 1374 | 1463 | 1012 |
| a | HTP backbone (#1829, unmodified) + `rest.onnx` on CPU | 847 | 852 | 463 | 523 | 608 | 458 |
| b | + ScatterND rewrite of the RoiAlign level merge | 627 | 628 | 295 | 346 | 419 | 285 |
| c | + box/mask heads on HTP | 239 | 241 | 200 | 209 | 228 | 203 |
| d | + fused RPN on DSP | 233 | 236 | 193 | 202 | 219 | 189 |
| c_opt | as c, **optimized backbone** (#1833) | 215 | 218 | 169 | 172 | 187 | 178 |
| d_opt | + fused RPN on DSP, fed the raw uint8 RPN convs | 203 | 208 | 158 | 157 | 178 | 173 |
| **e_opt** | **+ RoiAlign on DSP, fed the NHWC FPN maps** | **158** | **156** | **131** | **132** | **143** | **130** |
| e_opt_ctx | e_opt with every HTP session loaded from an EP-context model | 188 | 179 | 156 | 160 | 168 | 159 |

The final stage (`e_opt`) runs **7.8-10.4x faster than ORT alone on the phone's CPU**, and
**3.5-5.5x faster than stage (a)**, the HTP backbone with `rest.onnx` on the CPU. The first run of the first
image (cold, after session creation) is 198 ms (240 ms with EP-context models). Session setup takes 6.0 s JIT-compiled, or ~0.5 s
from EP-context models; see the EP-context section.

Every stage improves on the one before it, so the final pipeline is `e_opt`. Keep (a)-(d) only as
baselines with the unmodified backbone.

### Where e_opt's time goes (image 139, 158 ms)

| step | ms | engine |
|---|---:|---|
| quantize image to uint8 NHWC (preprocessing) | 15.1 | CPU |
| backbone | 16.9 | HTP |
| score adapter (sigmoid/flatten of the 5 raw score convs) | 0.7 | CPU |
| dequantize the 4 NHWC FPN maps to fp32 for RoiAlign | 13.1 | CPU |
| fused RPN (DSP 1.7) | 2.1 | DSP |
| box RoiAlign level assignment (`seg1`) | 1.8 | CPU |
| box RoiAlign, 4 calls (DSP 19.9) | 21.4 | DSP |
| box level merge, ScatterND of 1000x256x7x7 (`seg2`) | 16.6 | CPU |
| box head | 30.6 | HTP |
| box post-processing, 80 per-class NMS, mask level assignment (`seg3`) | 10.0 | CPU |
| mask RoiAlign, 4 calls (DSP 6.8) | 8.0 | DSP |
| mask level merge (`seg4`) | 6.5 | CPU |
| mask head, 100-RoI bucket | 13.0 | HTP |
| mask paste (`seg5`) | 0.2 | CPU |

The biggest remaining items:
- **Box head, 31 ms on the HTP.** One 1000x12544x1024 GEMM. PR #1832 measured it at 1.3x faster
  than 4 CPU cores.
- **Image quantize and FPN dequantize, 28 ms of CPU loops.** A real app's preprocessing would emit
  uint8 NHWC directly, and a uint8-input RoiAlign would remove the dequantize.
- **Box level merge, 17 ms.** It copies the four per-level RoiAlign outputs into the 1000-row
  tensor. The RoiAlign calls could write straight into those rows instead.

## Accuracy vs all-ONNX-Runtime

The reference is the whole `maskrcnn_sim.onnx` on host ORT CPU. The metric is the one
`../maskrcnn_e2e/README.md` uses: detections above score 0.5, matched by label and box IoU > 0.5.
Totals below cover the 5 COCO images plus `cats.jpg` (61 reference detections); the reference
counts per COCO image (23/17/2/2/14) match that README's table exactly.

| pipeline (phone) | matched | detections | mean box IoU | mean score \|Δ\| | mean mask IoU |
|---|---:|---:|---:|---:|---:|
| ORT CPU on the phone (`ref`) | 61/61 | 61 | 1.000 | 0.000 | 0.983 |
| **unmodified HTP backbone**, a = b | 57/61 | 60 | 0.924 | 0.019 | 0.852 |
| **unmodified HTP backbone**, c = d (heads on HTP) | 57/61 | 60 | 0.915 | 0.022 | 0.830 |
| **optimized HTP backbone** (c_opt, d_opt, e_opt: identical) | **58/61** | 63 | **0.943** | 0.019 | **0.861** |
| e_opt_ctx (EP-context models) | 58/61 | 62 | 0.944 | 0.018 | 0.868 |

For comparison, the TVM int8 pipeline on the DSP matched 58/61, with box IoU ~0.93 and mask IoU
~0.87 (`../maskrcnn_e2e/README.md`, 6 COCO images).

- **Optimized backbone:** it quantizes the residual shortcuts, the only numeric change in PR #1833.
  That lands slightly *closer* to the float reference here than the unmodified model does. PR
  #1833's one-image check (`cats.jpg`) had seen a small drop; over six images the sign flips, so
  the difference is noise-sized either way.
- **Unmodified backbone, stage by stage:** a->b is exact (ScatterND rewrite) and c->d is exact on
  the phone (fused RPN). b->c moves the box/mask heads to the HTP, which changes numerics a
  little: box IoU 0.924 -> 0.915, mask IoU 0.852 -> 0.830.
- **Fused RPN is bit-exact on the phone:** c_opt and d_opt outputs are identical on all six images.
- **DSP RoiAlign (d_opt -> e_opt):** boxes, labels and scores are identical on every image; mask
  probabilities differ by at most 0.033. RoiAlign's ~1e-4 fp32 difference from ORT gets amplified
  by one uint8 step inside the HTP mask head.
- **ORT on the phone vs on the host:** boxes, labels and scores are identical, mask IoU 0.983
  (CPU float differences somewhere in the mask path; not investigated). That is the floor any row
  above is measured against.
- **EP-context models change the HTP numerics a little** (e_opt_ctx vs e_opt: 62 vs 63
  detections, same matches), on top of being slower (below).

## EP-context vs JIT-compiled HTP sessions

PR #1832 saw the box head run ~2x slower when loaded from an EP-context model (`Ort::CompileModel`,
embed mode) than when compiled at session creation. It reproduces here in the full pipeline:

| | JIT (e_opt) | EP-context (e_opt_ctx) |
|---|---:|---:|
| HTP session creation, backbone + box head + mask head | 1.8 + 1.3 + 2.6 s | **0.33 + 0.07 + 0.06 s** |
| backbone, per image | 16.9 ms | 16.9 ms |
| box head, per image | 30.6 ms | **51.1 ms** |
| mask head, per image | 13.0 ms | 16.5 ms |

The backbone is unaffected; the box head is 1.7x slower and the mask head somewhat slower. So the
choice is ~6 s of startup against ~24 ms per image. Still not root-caused.
`htp_graph_finalization_optimization_mode` is set per session: 3 for the heads (#1832), default
for the backbone, where #1833 measured mode 3 as 3.6 ms slower.

## How the pipeline is cut, and checked

- **`build_models.py`** builds everything the phone needs from `backbone.onnx`, `rest.onnx` and
  `maskrcnn_sim.onnx` (`../maskrcnn_e2e/prepare.py`), PR #1833's `ceiling/make_optimized.sh`
  output, and the fused-RPN constants. It produces the heads, their NHWC-row variants, the adapters
  and the CPU segments, plus one `pipe_<stage>.txt` per stage.
  - **Cutting the CPU segments.** It cuts `rest.onnx` (after the ScatterND rewrite) *automatically*
    around the regions that run elsewhere. Each CPU node is tagged with the last external region it
    depends on, and each tag becomes one segment. A segment's inputs and outputs are exactly the
    tensors crossing its boundary, so nothing is computed twice. Nodes that depend only on
    constants (e.g. weight DequantizeLinears) are copied into whichever pieces read them.
  - **Adapters for the optimized backbone.** PR #1833's backbone emits raw uint8 tensors, NHWC FPN
    maps and raw RPN convs. The adapters cut the removed dequantize/layout chains out of PR #1833's
    step-2 model. The fused RPN kernel reads the raw uint8 delta convs directly (its source-1 mode),
    and RoiAlign reads the NHWC FPN maps directly.
  - **Heads that take RoiAlign's output as-is.** The DSP RoiAlign writes (R, OH, OW, C) rows, so the
    box head's fc6 int8 weight rows are permuted to match. The mask head gets a leading Transpose.
    No layout conversion runs anywhere in the pipeline.
- **`host_emulate.py`** runs each pipeline file on the host with CPU stand-ins: HTP models on ORT
  CPU, the DSP RPN as the 604-node region it replaces, RoiAlign as ORT's own op on the NCHW view.
  Stages a-d reproduce the whole-model reference **exactly**, and d_opt/e_opt reproduce c_opt
  exactly, on all six images. That check ran before anything ran on the phone.
- **`e2e_run.cpp`** is the phone driver. It holds one ORT environment with one global 4-thread CPU
  pool shared by every CPU session. HTP sessions are strict (no CPU fallback). The two FastRPC skels
  are built from `../tinygrad_hexagon_bridge/rpn_fused` and `roialign_fast` unchanged.
  DSP-visible buffers are `rpcmem`, so the 55.7 MB P2 map is mapped, not copied.
  `rpn_glue.c` loads the RPN constants with the unchanged `rpn_model_io.h`.
- **`compare_results.py`** computes the detection metric above.

## Reproduce

```bash
# models: ../maskrcnn_e2e/prepare.py -> W/{backbone,rest,maskrcnn_sim}.onnx
../htp_exploration/ceiling/make_optimized.sh W/backbone.onnx W/opt
# RPN constants: ../tinygrad_hexagon_bridge/rpn_fused/capture_rpn_fused.py (+ rpn_host_check) -> R
python build_models.py --work W --rpn R --out O
python prepare_inputs.py O/maskrcnn_sim.onnx IMGS img_000000000139.jpg ...   # + all-ORT reference
python host_emulate.py O e_opt IMGS/img_000000000139.bin EMU                  # optional host check
HEXAGON_SDK_ROOT=... HEXAGON_TOOLCHAIN=... OUT=O IMGS=IMGS ./build.sh ref a b c d c_opt d_opt e_opt e_opt_ctx
python compare_results.py IMGS/ref results a e_opt
```

`build.sh` downloads the ORT/QNN libraries through `../htp_exploration/qnn_shell/fetch_libs.sh`
(Maven Central, not committed). It builds the driver and both skels, pushes changed files only, and
runs each stage with `WARMUP`/`REPS` (defaults 2/7).

## Not done

- **Per-class NMS** (80 calls, `seg3`) stays on the CPU: PR #1825 measured it slower on the DSP
  because of the call overhead.
- **uint8 RoiAlign** reading the backbone's uint8 NHWC maps would remove the 13 ms dequantize.
- **The box level merge could be removed** by having RoiAlign write into the final rows.
- **Pinning the image to the phone**: the driver reads preprocessed fp32 images; JPEG decode and
  resize aren't timed.
- **More images:** this is 6 images, not a COCO evaluation.
