# SAM hybrid inference on an M4 Mac mini

The first M4 measurements show that full-precision Core ML is fastest overall
for these SAM variants. Running the prompt decoder on tinygrad Metal with
`TinyJit` costs only 1–7 ms more end to end, while eager `OnnxRunner` is much
slower. This gives a practical hybrid path when keeping the decoder on the
Metal backend is useful.

All numbers below are medians of 8 runs after 3 warm-ups on an M4 Mac mini.
Core ML used `compute_units=ALL` and `compute_precision=FLOAT32`; tinygrad used
`METAL`. The test used one image (`mask_point.jpg` from MobileSAM's public
assets) and one positive center-point prompt. End-to-end timing includes both
predictions and the encoder-to-decoder handoff. It excludes image resizing and
mask postprocessing.

| model | input | Core ML → Core ML | Core ML → Metal JIT | Core ML mask IoU vs. ORT | Hexagon HTP reference |
| --- | ---: | ---: | ---: | ---: | ---: |
| EdgeSAM | 1024² | 32.0 ms | 39.5 ms | 0.99996 | 92.2 ms |
| MobileSAM | 1024² | 66.6 ms | 67.5 ms | 1.00000 | 330.7 ms |
| EfficientViT-SAM-L0 | 512² | 30.8 ms | 36.7 ms | 0.99994 | 52.8 ms |

The Hexagon column is the existing first-mask sum from the repo's phone
measurements: image encoder plus prompt decoder. It is included as context,
not a new run on the same device or test harness. Relative to those recorded
figures, full Core ML was 2.9x faster for EdgeSAM, 5.0x for MobileSAM, and
1.7x for EfficientViT-SAM-L0. The hybrid path was 2.3x, 4.9x, and 1.4x
faster, respectively.

## Why tinygrad JIT matters

The eager ONNX runner dispatches graph operations separately. `TinyJit`
captures the execution and replays it as a graph, which changes Metal timing
substantially:

| model | encoder eager → JIT | decoder eager → JIT |
| --- | ---: | ---: |
| EdgeSAM | 327.1 → 34.5 ms | 190.2 → 12.4 ms |
| MobileSAM | 437.8 → 147.2 ms | 189.3 → 11.8 ms |
| EfficientViT-SAM-L0 | 262.5 → 53.2 ms | 186.8 → 11.4 ms |

The Core ML translator gained the `Resize`, `Not`, and `DepthToSpace`
lowerings needed by these graphs. For Core ML, the encoder's uint8 raw-pixel
input is presented as float32 because its public model interface does not
accept uint8. The ONNX graph's boundary dequantization is identity
(`scale=1`, `zero_point=0`), so this keeps the pixel values unchanged.

## Accuracy and limits

The shown mask IoUs compare the hybrid model output with ONNX Runtime CPU on
the one sample and prompt above. They establish numerical parity for the
benchmark path, but do not replace the repo's multi-image SAM quality check.
Default Core ML precision was faster on EdgeSAM (11.7 ms for both stages with
`CPU_AND_NE`) but produced 0.86 thresholded mask IoU on this sample. Full
precision is the setting used for all three models in the comparison table.

Hexagon reference timings and the broader mask-quality protocol are in
[`scripts/android/vision_models/sam/README.md`](../scripts/android/vision_models/sam/README.md).
The runner and reproduction command are documented in
[`scripts/apple/README.md`](../scripts/apple/README.md). The machine-readable
summary is in [`m4_sam_hybrid_results.json`](m4_sam_hybrid_results.json).
