# Deploy pipeline: vision models to the phone's Hexagon HTP from one spec file

```sh
./deploy.py models/<name>.yaml --device 239dbd8f [--work ~/.cache/onnxsim-deploy] [--stages ...] [--force ...]
```

One command takes a model from a URL to a benchmarked, accuracy-checked pipeline on the phone.
Each stage caches its output under `<work>/<name>/<stage>/` together with a `stamp.json` key. The
key covers the spec keys the stage reads, the stage's own code, and the previous stage's key. A
re-run therefore redoes only what changed: editing `bench:` re-runs bench and nothing before it.

| stage | what | output |
|---|---|---|
| fetch | pinned model (url + sha256, a `script`, or a local `path`); COCO val2017 images by id | `model.onnx`, `_images/` |
| simplify | onnxsim with the spec's fixed input shapes (QNN needs static shapes) | `model.onnx` |
| quantize | ORT `quantize_static` QDQ: per-channel int8 weights, uint8 activations, real calibration images, MinMax | `model.onnx` |
| rewrite | passes from `passes/` (`uint8_input`, `uint8_outputs`, `script` = any existing `in.onnx out.onnx` rewrite, e.g. `../htp_exploration/ceiling/*.py`) | `model.onnx`, `rewrite_meta.json` |
| post | CPU post-processing graph from standard ONNX ops (`yolo_detect`: decode + NonMaxSuppression) | `post.onnx` |
| pipe | `pipe.txt` for `runtime/pipe_run` + preprocessed eval inputs | `pipe.txt`, `inputs/*.bin`, `pipe_meta.json` |
| partition | HTP model alone, CPU fallback allowed (QNN's refusals from logcat, with tensor rank) then strict | `report.json` |
| push | builds `runtime/pipe_run` (NDK) and pushes it, the ORT/QNN libs, the models and inputs (changed files only) | |
| bench | `pipe_run` over every eval input: per-step and total median, FPS, cold run, session creation | `bench.json`, `outputs/` |
| accuracy | fp32 ORT (simplified model + the same post graph) vs the host int8 graph and vs the phone outputs | `accuracy.json` |

simplify, quantize, rewrite, post and pipe run in a child process under
`systemd-run --user -p MemoryMax=16G -p MemorySwapMax=0` (`--mem` changes the cap; `--no-cap`
runs them in-process). They run one at a time, and nothing runs in the background.

## Runtime: `runtime/pipe_run.cpp`

It is `../e2e_pipeline/e2e_run.cpp` generalized, with the same step grammar (`ort`, `ortpad`,
`quant_in`, `dq`, `rpn`, `roialign`) plus these additions:
- **Header lines:** `input <name> <f32|u8> <dims>` and `output <names>`. A pipe file without them
  gets Mask R-CNN's defaults, so `../e2e_pipeline/pipe_*.txt` run unchanged.
- **`htp-fallback` engine:** used only for partition reports.
- **Reporting:** per-input `fps` and an `overall` line.

The DSP kernels (`rpn`, `roialign`) are compiled in only for `pipe_run_dsp` (`PIPE_DSP_KERNELS=1`,
linked against `../e2e_pipeline/build.sh`'s stubs). All HTP sessions are strict (no CPU fallback).
`pipeline.qnn` in the spec holds per-session QNN options, e.g. `htp_graph_finalization_optimization_mode`.

## Results (phone 239dbd8f, Snapdragon 8+ Gen 1 / V69, shared with other jobs)

**YOLO11n** (`models/yolo11n.yaml`: int8 QDQ on 64 COCO images, uint8 NHWC input, 20 eval images):
- **Partition:** strict all-HTP. QNN refuses one input Transpose only in its pre-layout form (see below).
- **Speed:** HTP 2.8-3.0 ms. With decode + NMS on the CPU (1.2-1.8 ms), the total is **4.2-4.9 ms = 206-239 FPS**.
- **Accuracy vs fp32 ORT** (class + IoU > 0.5, score >= 0.25): **80/92 matched (87%)**, mean IoU 0.856. Host int8 also matches 80/92, so the whole gap is PTQ; the HTP adds nothing.

**Mask R-CNN** (`models/maskrcnn.yaml`, #1841's `e_opt` pipeline through `prebuilt:`):
- **Accuracy vs all-ORT:** 58/61 matched, box IoU 0.943, score |Δ| 0.019, mask IoU 0.861. This is
  identical to #1841, per image included.
- **Speed:** 160-189 ms per image vs #1841's 130-158 ms. This run shared the phone with the demo
  app (running Mask R-CNN itself, ~47% CPU) and Photos. Per step, the largest deltas vs #1841 are:
  - `quant_in`: 20 vs 15 ms
  - backbone: 18.0 vs 16.9 ms
  - box head: 34.4 vs 30.6 ms
  - `seg3`: 15.8 vs 10.0 ms

  All of these share the phone's CPU/HTP with the other apps.

## Adding a model

1. **Write `models/<name>.yaml`** (copy `yolo11n.yaml`):
   - `fetch.url` plus `sha256`. Run once without the sha256 and paste the value it prints.
   - `inputs: {<name>: {shape: [...]}}`.
   - `preprocess`: `letterbox` or `resize_normalize`, and the size.
   - `calibration` / `eval`: COCO val2017 ids, or `files:`.
   - `rewrites`, `postprocess`, `accuracy.kind`: `detection_match` for detectors, `tensor` for anything else (cosine / max error per output).
2. **Run `./deploy.py models/<name>.yaml` and read the stages in order:**
   - **partition:** a refused op with max rank 6+ means the HTP's rank-5 limit; reformulate it (`../vision_models_plan.md`).
   - **accuracy:** "host_int8" isolates PTQ error, "phone" adds HTP numerics.
3. **If int8 hurts**, try `quantize.exclude_nodes` for the nodes whose output mixes ranges. YOLO's
   final Concat joins 0..640 pixels with 0..1 scores: quantized, that zeroed every score (0
   detections); excluded, it gives 87% matched.
4. **If the model's uint8 input is plain pixels** (scale 1/255, zero point 0), set
   `pipeline.host_quantize: true`. The camera frame then goes in as-is: YOLO11n went from 30 to 4.8 ms,
   because the fp32->uint8 `quant_in` on the phone took 15-20 ms.
5. **Pipelines built elsewhere** (several models, DSP kernels) use `prebuilt:` (`dir`, `pipe`,
   `inputs`, `dsp`, `dsp_build`). Only pipe -> push -> bench -> accuracy run for them.

## Gotchas found while building this

- **Refusals that don't stick.** QNN also validates ops *before* ORT's NHWC layout transform, so a
  Transpose can show up as "refused" in logcat even though strict all-HTP passes (it cancels
  against QNN's own). The report says so when strict passes.
- **Percentile/Entropy calibration** keeps histograms of every activation. At 64 images they were
  OOM-killed at the 16G cap, and the cap did its job. MinMax is the default.
- **Missing `u8` input support.** `../htp_exploration/qnn_shell/qnn_run_multi` has no `u8` inputs,
  so the partition stage runs the model through `pipe_run` instead.
