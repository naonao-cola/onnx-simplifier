# `rest.onnx` on the HTP: the heads run there, the rest can't, and the biggest cost was a scatter

Follow-up to `qnn_shell_findings.md` (the whole int8 backbone on the HTP, 57 ms). This is the
other half of Mask R-CNN: `rest.onnx`, 2405 nodes of proposal decode, TopK, 85x NMS, RoiAlign,
the box head (4 MatMuls), the mask head (5 Conv + ConvTranspose) and post-processing, which ran
at 300 ms per image on the phone's CPU (ORT, 4 threads). Same phone, same QNN setup
(`com.qualcomm.qti:qnn-runtime` 2.50.0, unsigned PD, adb-shell harness), with a multi-input
harness (`qnn_shell/qnn_run_multi.cpp`, `run_multi.sh`). All numbers are medians of 7-20 runs on
the phone after two warm-up runs.

## 1. The whole graph with CPU fallback allowed: 6 tiny partitions, slower and less accurate

Running all of `rest.onnx` through the QNN EP with CPU fallback allowed (ORT verbose logs, which
on Android go to logcat, not stdout):

- **6 QNN partitions** vs. 1808 nodes left on the CPU EP (after ORT's own graph optimizations).
  What lands on the HTP is the static-shape per-FPN-level RPN arithmetic after TopK (Slice,
  Gather, Add/Sub/Mul, Exp, Clip, ...), split into six islands by the dynamic ops between them.
- **Every head node is rejected**, box head and mask head alike, with `REASON : Cannot get
  shape`: inside the full graph the RoI/detection count is a dynamic dimension (it comes out of
  NonZero/NMS), and the QNN HTP backend needs static shapes. The other rejection reasons are
  `NonZero output shape must be static` (75) and `Operators of type ConstantOfShape are not
  supported by QNN EP` (69).
- It is **slower** than CPU only: 364 ms vs 284 ms (4 threads), because the six islands add
  CPU<->HTP transitions without taking away any real work.
- And **less accurate**: the RPN arithmetic running on the HTP (non-IEEE fp) perturbs the
  proposals. On `cats.jpg`: 32 raw detections instead of 21, 4 above score 0.5 instead of 3, box
  IoU 0.913, mask IoU 0.885 on the matched ones.

So "let ORT partition it" is the wrong way to use the HTP here.

## 2. The static heads, cut out and pinned to a static batch: both run fully on the HTP

`qnn_shell/rest_split.py make` cuts `rest.onnx` at fixed tensor names:

| Head | input | outputs | nodes |
|---|---|---|---:|
| box head | `2788` `[1000,256,7,7]` (1000 RoIs, fixed at this input size) | `2811` softmax scores `[1000,81]`, `2810` box deltas `[1000,324]` | 26 |
| mask head | `6833` `[ndet,256,14,14]`, pinned to a bucket (32 / 100) | `6857` mask probs `[ndet,81,28,28]` | 22 |

and the dynamic remainder into three CPU pieces (`cpu_a`: RPN post-processing + box RoiAlign;
`cpu_b`: box post-processing + mask RoiAlign; `cpu_c`: mask paste/post-processing). One tensor
leaked across the mask-head boundary: a `Shape` node reads the pre-sigmoid logits `6856`; it only
needs the shape, which `6857` shares, so the remainder points it there. Run on the host with ORT
on both sides, the split is **bit-exact** against the unsplit graph (all four outputs, 0 diff),
including the mask head zero-padded to 100.

With `session.disable_cpu_ep_fallback=1` (session creation fails if any node would run on the
CPU) **both heads are accepted in full**, including the mask head's fp32 `ConvTranspose` (the only
non-QDQ op in either head; QNN runs it on the HTP without complaint).

| Head, on the phone | HTP | ORT CPU, 4 threads | ORT CPU, 1 thread |
|---|---:|---:|---:|
| box head, 1000 RoIs, default | 39.9 ms | 36.3 ms | 131.4 ms |
| box head, `htp_graph_finalization_optimization_mode=3` | 31.3 ms | | |
| box head, opt mode 3 + `htp_performance_mode=burst` | **27.6 ms** (1.3x) | | |
| mask head, 32-RoI bucket, opt 3 + burst | **4.4 ms** (18x) | 79.9 ms | |
| mask head, 100 RoIs (max detections), opt 3 + burst | 12.7 ms (21x) | 266.9 ms | |

The mask head is a big win (small spatial convs over a batch of RoIs are exactly what the HTP is
for); the box head is only slightly faster than four big CPU cores: it is one 1000x12544x1024
GEMM (12.8 GMAC) plus a few small ones, and the 50 MB fp32 input has to be quantized and moved
to the HTP on every call. Graph finalization mode 3 is worth 20-25% on it; burst another ~10%.

**EP-context caveat:** with an EP-context model (`Ort::CompileModel`, embed mode) the box head
runs consistently **~2x slower** (52-55 ms, with or without burst/opt 3, across repeated
processes) than the same model compiled at session creation (27-40 ms). PR #1829 saw no such
difference for the backbone. Not root-caused here; the numbers above are JIT-compiled (session
creation costs ~2.9 s for the box head, ~0.6 s for the mask head).

**RoiAlign itself** is accepted by the QNN HTP backend too (strict mode, one real P2-level call:
395 RoIs over the `[1,256,200,272]` map, max abs err 6.8e-3 on a +-14.5 range), but as its own
graph it is **2.4x slower than the CPU** (41.9 ms vs 17.4 ms, 4 threads): each call has to
quantize and upload the 55.7 MB fp32 feature map. It would only pay off inside the same QNN
graph as the backbone that produced the map, which the dynamic per-level RoI assignment between
them prevents.

### Accuracy

Heads on the HTP vs. ORT CPU on the same real inputs: box-head class argmax agrees on 999/1000
RoIs (softmax max abs err 0.064, box deltas 0.116 on a -3.6..5.0 range); mask probabilities mean
abs err 0.015, 98.4% of pixels agree on the >0.5 mask threshold.

End to end on `cats.jpg`, stitching the three CPU pieces (host ORT) with the heads run on the
phone's HTP (`rest_split.py stitch ... phone`), against the all-ORT pipeline on the same input
(same matching as `../maskrcnn_e2e/README.md`, score > 0.5):

| Pipeline | matched | box IoU | score \|Δ\| | mask IoU |
|---|---:|---:|---:|---:|
| HTP backbone + all-CPU `rest.onnx` (PR #1829's path) | 3/3 | 0.985 | 0.011 | 0.985 |
| CPU backbone + **heads on HTP** | 3/3 | 0.990 | 0.002 | 0.986 |
| **HTP backbone + heads on HTP**, dynamic ops on CPU | 3/3 | 0.986 | 0.006 | 0.980 |
| (whole `rest.onnx` with automatic CPU fallback, section 1) | 3/3 (+1 extra) | 0.913 | 0.014 | 0.885 |

One image only; the six COCO val images from `../maskrcnn_e2e/README.md` aren't on this machine.
(Reproducibility note: re-preprocessing `cats.jpg` with this machine's PIL gives int8-step
differences from the cached backbone features in the scratch data, e.g. 28 vs 21 raw detections;
every comparison above uses one consistent input on both sides.)

## 3. Where the CPU time actually goes: a scatter, not the dynamic ops

ORT's profiler on the phone (`ORT_PROFILE=...`, all of `rest.onnx`, 4 threads, profiled total
340 ms), by region:

| Region | ms/image | share |
|---|---:|---:|
| **`ScatterElements`, RoiAlign level merge (8)** | **123.5** | 36% |
| mask head (4 Conv, ConvTranspose, 1x1 Conv) | 69.3 | 20% |
| RoiAlign (8 calls) | 60.0 | 18% |
| box head (fc6, fc7, cls, bbox) | 33.6 | 10% |
| box post-processing + mask RoI selection | 32.2 | 9% |
| RPN post-processing (decode, per-level loop) | 6.9 | 2% |
| NMS (85) | 5.1 | 1.5% |
| TopK (7) | 0.9 | 0.3% |
| other | 8.6 | 3% |

The genuinely dynamic ops the survey worried about (NMS, TopK, decode) are ~13 ms together.
The largest single cost is torchvision's `MultiScaleRoIAlign` export: each FPN level's RoiAlign
output is merged into the `[nroi,256,7,7]` result with an element-wise `ScatterElements` whose
index tensor is the RoI index `Expand`ed to the full output shape, i.e. 12.5M indexed element
writes per level. Since those indices are constant along every non-batch axis, it is exactly a
row scatter: `ScatterND(data, Reshape(idx, [-1,1]), level_out)`.

`qnn_shell/scatter_rewrite.py` makes that rewrite (8 scatters, 48 dead index-broadcast nodes
dropped). The rewritten graph is **bit-exact** against the original on three different feature
sets (host ORT, all four outputs), and on the phone:

| `rest.onnx` on the phone's CPU | 4 threads | 1 thread |
|---|---:|---:|
| original | 300.6 ms | 955.5 ms |
| **ScatterND rewrite** | **187.4 ms** (1.6x) | 655.7 ms |

This is a pure graph rewrite, the kind onnxsim itself could apply; it needs no HTP at all.

## 4. Per-image pipeline on the phone

Measured per piece on the phone (4 CPU threads, `cats.jpg`, heads with opt mode 3 + burst,
mask head in the 32 bucket; each piece timed separately, so this is a sum, not one process):

| Pipeline | backbone | rest | total |
|---|---:|---:|---:|
| all ORT CPU (4 threads) | 635 | 300.6 | ~936 ms |
| PR #1829: backbone on HTP, `rest.onnx` on CPU | 57 | 300.6 | ~358 ms |
| + ScatterND rewrite | 57 | 187.4 | **~244 ms** |
| + box and mask heads on HTP: `cpu_a` 66.4 + box 27.6 + `cpu_b` 21.1 + mask 4.4 + `cpu_c` 14.9 | 57 | 134.4 | **~191 ms** |

What the HVX kernels from `../tinygrad_hexagon_bridge/` would add on top (projection, not
measured end to end): per-level NMS (3.66 -> 1.13 ms incl. RPC), proposal decode (1.04 -> 0.66),
TopK (0.86 -> 0.72) are worth ~3 ms together. The fast RoiAlign (58.3 -> 27.8 ms for all 8
calls) needs channels-last feature maps; the HTP backbone writes NCHW, so it pays the separate
13.3 ms transpose measured in PR #1824 (the fused channels-last FPN conv from that PR is an HVX
conv, not the HTP backbone), for a net ~17 ms. That projects to roughly **170 ms** per image.

What's left on the CPU after that is mostly the RoiAlign level assignment and the box
post-processing (`cpu_a`/`cpu_b` minus the ops above), the per-class NMS (which stays on the CPU,
PR #1825), and the mask paste.

## Not done

- One process running the whole split pipeline on the phone (the pieces are timed separately;
  the head inputs/outputs cross the host here, not in a real deployment).
- Mask-head batch buckets beyond 32/100, and the RoI count at other input sizes (1000 is fixed by
  `post_nms_top_n` at 800x1088).
- Root-causing the EP-context slowdown on the box head.
- Wiring the HVX kernels into the same process; the ~170 ms figure is a projection.
