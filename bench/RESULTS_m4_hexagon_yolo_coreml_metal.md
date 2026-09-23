# Hexagon-deployed YOLO models on Core ML and tinygrad Metal

The repository's three deployable YOLO detectors all run through the generic
Core ML + tinygrad Metal benchmark. On the M4, Core ML FLOAT16 is faster than
tinygrad Metal JIT for each model. JIT makes tinygrad Metal 8–20x faster than
its eager ONNX runner, but does not beat Core ML for these convolution-heavy
graphs.

These are 8-run medians after 3 warm-ups on an M4 Mac mini. The input was one
representative 640x640 letterboxed photo with RGB values in [0, 1]. Core ML
used `compute_units=CPU_AND_NE` and `compute_precision=FLOAT16`; tinygrad used
`METAL`. The table times the model stage only: image preprocessing and
detection postprocessing are excluded. Core ML's compute plan was not traced,
so `CPU_AND_NE` indicates allowed units, not verified Neural Engine placement.

| model | Core ML FP16 | tinygrad Metal JIT | tinygrad eager Metal | Core ML output vs ORT | thresholded candidates / median box IoU |
| --- | ---: | ---: | ---: | --- | --- |
| YOLO11n | 5.40 ms | 16.91 ms | 280.16 ms | score p99 error 0.000040; box p99 error 4.50 px | 4 / 4; 0.968 |
| YOLO26n | 4.92 ms | 16.36 ms | 318.33 ms | score p99 error 0.000014; box p99 error 5.09 px | 2 / 2; 0.995 |
| YOLO26s | 12.18 ms | 41.85 ms | 332.93 ms | score p99 error 0.000010; box p99 error 7.78 px | 4 / 4; 0.998 |

The candidate check thresholds each anchor's highest class score at 0.25 and
matches the same anchor and class between Core ML and ONNX Runtime. It is a
single-image sanity check, not a COCO evaluation; the models' NMS/top-k
postprocessing is not included. Core ML's largest raw box errors were 37–67
pixels, on lower-confidence candidates; assess task-level quality on a full
validation set before selecting reduced precision. Tinygrad JIT outputs had
cosine similarity 1.0 to ONNX Runtime on this sample and maximum absolute
errors of 0.0051, 0.0082, and 0.0060 respectively.

For context, the existing Snapdragon 8+ Gen 1 / Hexagon V69 results use
quantized HTP models: YOLO11n is 2.58 ms on the HTP, YOLO26n 2.57 ms, and
YOLO26s 3.75 ms. Their CPU postprocessing adds 0.92, 0.57, and 0.57 ms. Those
are separate hardware and precision runs; treat them as recorded references,
not a controlled cross-device benchmark. The M4 measurements use floating
point ONNX models.

## What this says about hybrid deployment

For a single YOLO graph, splitting operations between Core ML and tinygrad
Metal has no demonstrated latency advantage: Core ML handles the whole graph
in less time than tinygrad JIT. Hybrid execution is useful when a Core ML
conversion gap blocks deployment or a small operation has a specialized Metal
kernel. The SAM measurements show the cost of such a boundary on these graphs:
Core ML encoder plus Metal JIT decoder took 1–7 ms longer than running both
stages on Core ML.

The current bridge materializes stage outputs as NumPy arrays, so it
synchronizes and copies data at each boundary. A worthwhile next optimization
is to partition only unsupported or custom-kernel subgraphs, then reduce
boundary copies and launches. tinygrad JIT already removes most eager dispatch
overhead; model quality and transfer cost are now more limiting than Python
per-op launch overhead.

The M4 runs used the manifest-driven runner in
[`scripts/apple/benchmark_onnx_pipeline.py`](../scripts/apple/benchmark_onnx_pipeline.py).
The Hexagon baselines are in
[`scripts/android/deploy/README.md`](../scripts/android/deploy/README.md).
