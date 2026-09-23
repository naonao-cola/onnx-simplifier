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
| quantize | `onnxsim.quantize_static(full_graph=True)`: QDQ on every activation (uint8), per-channel int8 weights, int32 biases, real calibration images; `calibration_method` minmax / mse / percentile / entropy (streamed histograms: memory doesn't grow with #images), or `auto` (picks one by the spec's accuracy metric, below) | `model.onnx`, `quantize_meta.json` |
| rewrite | passes from `passes/` (`uint8_input`, `uint8_outputs`, `script` = any existing `in.onnx out.onnx` rewrite, e.g. `../htp_exploration/ceiling/*.py`) | `model.onnx`, `rewrite_meta.json` |
| post | CPU post-processing graph from standard ONNX ops (`yolo_detect`: decode + NonMaxSuppression; `yolo_end2end`: YOLO26's NMS-free top-k) | `post.onnx` |
| pipe | `pipe.txt` for `runtime/pipe_run` + preprocessed eval inputs | `pipe.txt`, `inputs/*.bin`, `pipe_meta.json` |
| partition | HTP model alone, CPU fallback allowed (QNN's refusals from logcat, with tensor rank) then strict | `report.json` |
| push | builds `runtime/pipe_run` (NDK) and pushes it, the ORT/QNN libs, the models and inputs (files whose md5 changed) | |
| bench | `pipe_run` over every eval input: per-step and total median, FPS, cold run, session creation | `bench.json`, `outputs/` |
| accuracy | fp32 ORT (simplified model + the same post graph) vs the host int8 graph and vs the phone outputs | `accuracy.json` |

On a phone shared between several users, set `DEPLOY_DEVICE_ROOT=/data/local/tmp/<you>` (default
`/data/local/tmp/deploy`) so nobody loads another run's files, and wrap the phone stages
(partition, push, bench, accuracy) in the host's phone lock (`~/.cache/android-phone/phone-run`).

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

**YOLO11n** (`models/yolo11n.yaml`: int8 QDQ, MSE calibration on 64 COCO images, uint8 NHWC input, 20 eval images):
- **Partition:** strict all-HTP. QNN refuses one input Transpose only in its pre-layout form (see below).
- **Speed:** HTP 2.5-3.0 ms. With decode + NMS on the CPU (0.9-1.8 ms), the total is **3.3-4.9 ms = 206-306 FPS**.
- **Accuracy vs fp32 ORT** (class + IoU > 0.5, score >= 0.25): 79/92 matched on the 20 eval images
  (ORT's MinMax quantizer: 80/92). 20 images (92 boxes) are too few to rank calibration methods,
  so they were compared on Ultralytics' coco128 (128 images, 659 fp32 boxes) instead:

  | calibration | phone matched | host int8 matched | phone score \|Δ\| | HTP strict | calib time (64 img) |
  |---|---|---|---|---|---|
  | ORT `quantize_static` MinMax (before) | 585 (88.8%) | 583 | 0.058 | 2.66 ms | 4 s |
  | onnxsim minmax | 585 (88.8%) | 583 | 0.061 | 2.60 ms | 5 s |
  | **onnxsim mse** (default) | **596 (90.4%)** | **615** | **0.054** | 2.48-2.62 ms | 25 s |
  | onnxsim percentile 99.999 | 585 (88.8%) | 585 | 0.051 | 2.56 ms | 22 s |
  | onnxsim percentile 99.99, head at minmax (\*) | 563 (85.4%) | 563 | 0.073 | 2.83 ms | 20 s |

  (\*) measured before value-preserving ops shared their input's scale (see `onnxsim/qdq_full_graph.py`),
  which took ~0.2-0.5 ms of requantizes off the HTP for every method.

  Entropy and percentile 99.99 clip YOLO's class logits. Those are almost all background, so the
  rare large logits that *are* the detections sit above the clip, and every score is capped at
  ~0.5, or at 0 when the Sigmoid output itself is clipped (now never: bounded-op outputs keep
  their exact range). `quantize.minmax_tensors: [<fnmatch patterns>]` keeps chosen tensors, e.g.
  the head's score path, at their exact range; with the head kept, 99.99 recovers to 563 and
  entropy to ~63/92 on the 20-image set, both still below minmax. Per-tensor weights
  (`per_channel: false`) drop to 63/92: per-channel it is. HTP times vary by ~0.3 ms run to run
  (shared phone).
- **`calibration_method: auto`** (`onnxsim.pick_calibration`) quantizes once per candidate
  (minmax, mse, percentile 99.999 / 99.99, entropy, and onnxsim's per-tensor `auto`; set
  `auto_candidates` to change them) and keeps the best by the spec's own accuracy kind on host ORT
  against fp32: matched detections through the spec's postprocess for `detection_match`, worst
  per-output SQNR otherwise. It never sees the eval images. It cross-fits over the calibration
  images (`auto_folds`, default 4): each quarter is scored by candidates calibrated on the other
  three, then the winner is calibrated on all 64. One held-out quarter alone is not enough. On
  YOLO11n it ranked percentile 99.99 first, which coco128 and the phone both rank below mse:

  | candidate | 4 folds over the 64 calibration images | 16-image holdout | coco128 host (659 boxes) | coco128 phone |
  |---|---|---|---|---|
  | minmax | 0.884 | 0.865 | 559 | 585 |
  | **mse** | **0.909** | 0.888 | **607** | **596** |
  | percentile 99.999 | 0.877 | 0.843 | 593 | 585 |
  | percentile 99.99 | 0.888 | **0.921** | 594 | 575 |
  | entropy | 0.759 | 0.742 | 512 | - |
  | per-tensor auto | 0.879 | 0.876 | 589 | - |

  (coco128 host: calibrated on 48 of the 64 images, ORT at its basic level, which keeps every QDQ pair.)

  `auto` picks mse, byte-identical to the default model above, in 386 s (5 calibration runs,
  6 candidates x 64 images; cached after that) at 1.7 GB peak RSS. The per-tensor `auto`
  minimizes each tensor's own expected uint8 error and never clips graph outputs, Sigmoid/Softmax
  outputs or score logits (SiLU's gate excepted). That is not the task's error: it trails mse
  (589 vs 607), which is why the model-level pick scores with the task metric.
- **Calibration memory** (peak RSS; each tensor's histogram is 4096 int64 counts, streamed batch by batch):

  | model | images | peak RSS (any method) | the old keep-every-value entropy/mse would hold |
  |---|---|---|---|
  | YOLO11n 640, 333 tensors | 16 / 64 | 1.17 / 1.18 GB | 4.3 / 17.3 GB |
  | Mask R-CNN backbone 800x1088, 191 tensors | 16 / 64 | 6.3 / 6.3 GB | 34 / 137 GB |

  The remaining ~5 GB for Mask R-CNN is one image's activations, all fetched as outputs of one ORT run.

**YOLO26n / YOLO26s** (`models/yolo26{n,s}.yaml`: Ultralytics' NMS-free end-to-end YOLO, AGPL-3.0 weights
sha256-pinned and exported by `models/export_yolo26.py`; set `ULTRALYTICS_PYTHON` if the deploy
python has no `ultralytics`). YOLO26 has no DFL (`reg_max: 1`, boxes regressed directly) and uses a
one-to-one head, so there is no NMS: Ultralytics' export ends in a two-stage TopK (top 300 anchors by
best class score, then the top 300 (anchor, class) pairs) and GatherElements, output (1, 300, 6).
- **Split:** the export stops at the one-to-one head's decoded output (1, 84, 8400: x1,y1,x2,y2 +
  sigmoid scores), which runs strict all-HTP with uint8 NHWC input. The top-k runs on the CPU as
  `postprocess: {kind: yolo_end2end}`, which is bit-identical to Ultralytics' own end2end ONNX (max abs diff 0
  on 10 COCO images; `tests/test_deploy_yolo_end2end_post.py` checks it against a numpy transcription).
  Putting the top-k on the HTP too (Ultralytics' end2end export, tail kept float) also runs strict
  all-HTP, but at **11.8 ms**, vs 2.6 ms HTP + 0.6 ms CPU with the top-k on the CPU.
- **Calibration differs from YOLO11n.** Host int8 vs fp32 on coco128 (565 fp32 boxes): mse 458,
  minmax 461, percentile 99.999 506 but with scores inflated (\|score Δ\| 0.22), percentile + the
  class-logit path (`one2one_cv3.*.2` convs through the score Concat) at its exact range via
  `minmax_tensors` 500 (\|Δ\| 0.08, the default). Keeping one block group float at a time recovers
  at most ~30 boxes (class head, first backbone stages, box decode); the loss is spread out, not
  one bad layer.
- **Results** (median under the phone lock, 128 coco128 images; the 20-image set gives the same ms):

  | model | HTP | CPU post | total | FPS | phone vs fp32 | host int8 vs fp32 | fp32 vs coco128 GT (recall / precision) | phone vs GT |
  |---|---|---|---|---|---|---|---|---|
  | YOLO11n (mse, NMS) | 2.58 ms | 0.92 ms | 3.47 ms | 289 | 596/659 (90.4%) | 615 | 484/929 (52.1%) / 73% | 468 (50.4%) |
  | **YOLO26n** (percentile, top-k) | 2.57 ms | 0.57 ms | **3.18 ms** | **314** | 482/565 (85.3%) | 500 | 439/929 (47.3%) / 78% | 409 (44.0%) |
  | YOLO26s (percentile, top-k) | 3.75 ms | 0.57 ms | 4.29 ms | 233 | 589/693 (85.0%) | 603 | 549/929 (59.1%) / 79% | 509 (54.8%) |

  GT columns are recall/precision at score >= 0.25, IoU > 0.5, same class -- not mAP. YOLO26's
  scores are more conservative at 0.25 (fewer, more precise boxes). The NMS-free head saves
  ~0.3 ms of CPU per frame vs YOLO11n's NonMaxSuppression; the HTP time of the n models is the same.
  YOLO26n loses more to int8 than YOLO11n does (85% vs 90% of fp32 boxes kept on the phone).

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
3. **If int8 hurts**, try `quantize.exclude_nodes` (or `exclude_op_types`) for the nodes whose
   output mixes ranges. YOLO's final Concat joins 0..640 pixels with 0..1 scores: quantized, that
   zeroed every score (0 detections); excluded, it gives 87% matched. Then compare
   `calibration_method`s (`minmax`, `mse`, `percentile` + `percentile: 99.999`, `entropy`) on
   more eval images than you think you need, and keep a detector's score path at its exact range
   with `minmax_tensors` if a clipping method loses detections. `calibration_method: auto` runs
   that comparison for you on the calibration images (cross-fitted), and writes every candidate's
   score to `quantize/quantize_meta.json`.
4. **If the model's uint8 input is plain pixels** (scale 1/255, zero point 0), set
   `pipeline.host_quantize: true`. The camera frame then goes in as-is: YOLO11n went from 30 to 4.8 ms,
   because the fp32->uint8 `quant_in` on the phone took 15-20 ms.
5. **Pipelines built elsewhere** (several models, DSP kernels) use `prebuilt:` (`dir`, `pipe`,
   `inputs`, `dsp`, `dsp_build`). Only pipe -> push -> bench -> accuracy run for them.

## Gotchas found while building this

- **Refusals that don't stick.** QNN also validates ops *before* ORT's NHWC layout transform, so a
  Transpose can show up as "refused" in logcat even though strict all-HTP passes (it cancels
  against QNN's own). The report says so when strict passes.
- **ORT's Percentile/Entropy calibration** keeps every activation value. At 64 images it was
  OOM-killed at the 16G cap, which is why this stage now uses onnxsim's streaming calibrator.
- **Push by size missed re-quantized models.** A model with new scales has the same byte size, so
  the phone kept running the old one. Push now compares md5.
- **The quantize cache key covers onnxsim's quantizer** (`calibration.py`, `qdq_full_graph.py`,
  from the onnxsim the child imports), so editing it re-runs quantize and everything after.
- **Missing `u8` input support.** `../htp_exploration/qnn_shell/qnn_run_multi` has no `u8` inputs,
  so the partition stage runs the model through `pipe_run` instead.
