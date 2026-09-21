# NVIDIA (TensorRT / CUDA) validation

Real-hardware checks of onnxsim output against the TensorRT builder, complementing the
hand-built-graph tests in `tests/test_tensorrt_*.py` (which never invoke TensorRT).

| file | interpreter | role |
|---|---|---|
| `qdq_pairs.py` | onnxsim (Python >= 3.11) | writes `<name>.orig.onnx` / `<name>.sim.onnx` pairs |
| `trt_harness.py` | system Python with `tensorrt` | builds engines, dumps per-layer tactic/precision, times inference, compares orig vs sim outputs |
| `modelopt_pipeline.py` | onnxsim + `nvidia-modelopt[onnx]` | fixes batch, runs `simplify()` and ModelOpt INT8 quantization on a real model, with an un-simplified control |
| `bench_trtexec.py` | any (stdlib) | builds/times every variant with `trtexec` and prints a table |
| `imagenette_data.py` | any with Pillow | preprocesses Imagenette (real ImageNet images, 10 classes) into val + calibration `.npy` sets |
| `ort_cpu_check.py` | any with onnxruntime | top-1 of ONNX models on a val subset via ORT CPU: separates "quantized model is inaccurate" from "TensorRT mishandles it" |
| `eval_accuracy.py` | system Python with `tensorrt` | streams the val set through each variant's TensorRT engine; top-1 and agreement with fp32 |

They are split because JetPack 6's TensorRT Python bindings are cp310-only while onnxsim
needs Python >= 3.11; models are exchanged as `.onnx` files.

```sh
python3.12 scripts/nvidia/qdq_pairs.py /tmp/pairs                      # onnxsim venv
python3.10 scripts/nvidia/trt_harness.py compare /tmp/pairs --int8 --fp16   # tensorrt venv
python3.10 scripts/nvidia/trt_harness.py build model.onnx --int8            # per-layer detail
```

The TensorRT venv needs `onnx` and `numpy<2` (`uv venv --system-site-packages` picks up
the apt-installed `tensorrt`). CUDA is reached via ctypes on `libcudart.so.12`.
`/usr/src/tensorrt/bin/trtexec` (package `libnvinfer-bin`) is useful for cross-checking:
on the 32-channel INT8 Conv it reported 0.049 ms mean GPU compute vs 0.062 ms wall-clock
from the harness (the harness includes launch overhead).

## Findings (Jetson Orin Nano 8GB, JetPack 6 / L4T R36.4.7, TensorRT 10.3.0, CUDA 12.6, sm_87)

`simplify()` on TensorRT's documented explicit-quantization Q/DQ conventions:

- Per-channel weight Q/DQ (axis 0) + per-tensor activation Q/DQ on a Conv: the real engine
  fuses both into one INT8 `sm80_xmma_..._i8f32_i8i32` convolution (2 layers), before and
  after `simplify()`.
- Symmetric zero-point MatMul: builds and runs in INT8; identical engine before/after.
- Residual `Add` with an independent Q/DQ per branch: engine contains
  `PWN(Add)` with two INT8 inputs, i.e. the whole `Add` runs in INT8, before and after.
- For all four models (three Q/DQ + a Conv/BN/ReLU control) in both INT8 and FP16:
  identical layer counts and **bit-identical outputs** (`max|diff| = 0`) between original
  and simplified. `simplify()` leaves these Q/DQ graphs node-for-node unchanged, so the
  engines are the same.
- Control (Conv+BN+ReLU, no Q/DQ): `simplify()` folds BN (3 -> 2 nodes) but TensorRT
  already folds BN itself (1 fused layer either way), so no engine-level gain here.

Caveats: tiny models, latencies are dominated by launch overhead and are not benchmark
numbers. The harness only enables `--int8` for graphs that contain Q/DQ nodes (TensorRT
fails with "no scaling factors" otherwise, since no calibrator is supplied).

Memory note: the Orin's 7.4 GB is shared CPU/GPU. `trtexec` failed with CUDA OOM while a
parallel C++ build was running; retry on an idle machine.

## ModelOpt + TensorRT: latency and accuracy on 3 ImageNet classifiers

ONNX model zoo `resnet18-v1-7`, `resnet50-v1-7`, `mobilenetv2-12`, all pretrained
ImageNet-1k. The pipeline lifts each to opset 17, pins the batch, then runs
`onnxsim.simplify()` (`sim`; ResNet-18 69 -> 49 nodes, ResNet-50 175 -> 122, MobileNetV2
106 -> 100) and ModelOpt INT8 (explicit Q/DQ, entropy or max calibration, mixed FP16/INT8
output, built with `--int8 --fp16`). `raw` is the same model without onnxsim.

```sh
python3.12 scripts/nvidia/imagenette_data.py imagenette2-320 /tmp/data      # needs Pillow
python3.12 scripts/nvidia/modelopt_pipeline.py resnet18.onnx /tmp/pl_r18 --batch 1 8 \
    --calib-npy /tmp/data/calib_x.npy --methods entropy max
python3 scripts/nvidia/bench_trtexec.py /tmp/pl_r18 --glob '*.sim*.onnx'       # latency
python3.10 scripts/nvidia/eval_accuracy.py /tmp/pl_r18 /tmp/data              # accuracy
```

**Accuracy** is 1000-way top-1 on the **Imagenette val split**: 3,925 real ImageNet images
of 10 ImageNet-1k classes (the gated ImageNet val set was not available). These are easy
classes, so absolute numbers run high; compare variants, not against published top-1.
Calibration uses 128 *train* images of the same 10 classes, disjoint from val, which is
in-distribution and likely flatters INT8 compared with a proper 1000-class calibration
set. Standard error on 3,925 images is ~0.65 points; "agree" is top-1 agreement with the
fp32 engine (two fp32 builds of one model agree only 99.9-100%: TensorRT tactic noise).

| model (sim, batch 8 eval) | fp32 | fp16 | int8 entropy | int8 max | agree w/ fp32 (entropy) |
|---|---|---|---|---|---|
| ResNet-18 | 75.18 | 75.16 | 75.46 | 75.44 | 95.95% |
| ResNet-50 | 80.99 | 80.97 | 80.82 | 80.74 | 97.38% |
| MobileNetV2 | 79.03 | 79.03 | **77.76** | **76.99** | 90.96% |

**Latency** (mean GPU ms, `trtexec --noDataTransfers`, MAXN_SUPER, simplified model,
INT8 = entropy calibration; max calibration is within 3% of it everywhere):

| model | batch | fp32 | fp16 | int8+fp16 | int8 vs fp16 |
|---|---|---|---|---|---|
| ResNet-18 | 1 | 1.578 | 0.758 | 0.485 | 1.56x |
| ResNet-18 | 8 | 8.357 | 3.537 | 1.927 | 1.84x |
| ResNet-50 | 1 | 3.596 | 1.814 | 1.190 | 1.52x |
| ResNet-50 | 8 | 19.969 | 8.816 | 4.930 | 1.79x |
| MobileNetV2 | 1 | 1.386 | 0.850 | 0.781 | **1.09x** |
| MobileNetV2 | 8 | 8.147 | 3.925 | 2.446 | 1.60x |

Takeaways:
- ResNets: INT8 is 1.5-1.8x faster than FP16 (3.0-4.3x vs fp32) at no measurable accuracy
  cost (within the ~0.65-point standard error).
- MobileNetV2 is the counter-example: INT8 loses 1.3 (entropy) to 2.0 (max) points and
  only 91% of predictions match fp32, while batch-1 INT8 is just 1.09x faster than FP16
  (depthwise convs get little from INT8 and add many Q/DQ reformats). On this board
  MobileNetV2 is better left at FP16 unless batch >= 8. Entropy beats max calibration.
- **onnxsim did not change accuracy or latency** for any of the three: raw and simplified
  models agree within noise (MobileNetV2's quantized raw and sim models score identically),
  since TensorRT and ModelOpt already fold BN/constants themselves. The earlier ResNet-18
  latency check (raw vs sim, batch 1/8) likewise showed <1% differences.

## Transformer: ViT-B/16 (`Xenova/vit-base-patch16-224`, HF `google/vit-base-patch16-224`)

```sh
python3.12 scripts/nvidia/imagenette_data.py imagenette2-320 /tmp/data_vit vit    # resize 224, mean=std=0.5
python3.12 scripts/nvidia/modelopt_pipeline.py vit_base.onnx /tmp/pl_vit --batch 1 --chw 3 224 224 \
    --calib-npy /tmp/data_vit/calib_x.npy --calib-n 32 --methods entropy \
    --variants default hp32 noattn hp32+noattn only-linear      # see variant_options()
python3 scripts/nvidia/bench_trtexec.py /tmp/pl_vit
python3.10 scripts/nvidia/eval_accuracy.py /tmp/pl_vit /tmp/data_vit --glob 'b1.*.onnx' \
    --ref b1.raw.onnx --mean .5 .5 .5 --std .5 .5 .5
```

The HF export is opset 11 with symbolic H/W and 1,375 nodes once lifted to opset 17
(385 `Constant`, 111 `Shape`, hand-decomposed LayerNorm/GELU); `simplify()` takes it to 420
nodes (standard `LayerNormalization`, `Gemm`, `Split`). Batch 1 only: a batch-8 ModelOpt run
was OOM-killed on the 8 GB board even with process isolation and 32 calibration images.
Calibration is 32 train images, same-10-class caveat as above. Standard error is ~0.6 points.

| batch 1 | latency ms | top-1 % |
|---|---|---|
| fp32, raw | 13.06 | 85.22 |
| fp32, onnxsim | 13.65 | 85.22 |
| fp16, raw | 5.59 | **18.96** |
| fp16, onnxsim | 5.68 | 85.27 |
| ModelOpt int8, default (all ops), raw | 4.30 | **0.03** |
| ModelOpt int8, default (all ops), onnxsim | 4.58 | **4.99** |
| ModelOpt int8, onnxsim, `only-linear` (fp16 remainder) | **4.92** | **84.51** |
| ModelOpt int8, onnxsim, `only-linear` (fp32 remainder) | 5.07 | 84.51 |

Findings, and here onnxsim does matter:

1. **Raw FP16 is broken; onnxsim fixes it.** The raw graph's hand-decomposed LayerNorm does
   `Pow(x, 2)`. On a real image, 13 of its 25 LayerNorms receive values up to |x| = 1418
   (the residual-stream outliers), so x^2 ~ 2e6 exceeds FP16's 65,504 and overflows: top-1
   collapses to 19%. onnxsim collapses the pattern into `LayerNormalization`, which TensorRT
   accumulates in fp32 (extra `Cast`s in the engine), and FP16 stays at 85.27%. The
   correct engine is ~2-7% *slower* than the broken one (92 vs 89 layers: per-encoder-layer
   extra small kernels and unfused `Gemm`s), a price worth paying. This was read from engine
   layer names, not profiled.
2. **ModelOpt's default INT8 collapses ViT-B** (0.03% raw, 5.0% simplified; max calibration
   and excluding attention MatMuls / head / FP16 remainder change nothing, all 0.4-5%).
   ORT CPU shows the same collapse (4.7% vs 78.7% fp32 on 150 images), so it is the
   quantized model, not TensorRT.
3. **Cause: ModelOpt quantizes the residual-stream `Add`s** (the ~46 Q nodes beyond the
   Linears in the `noattn` model), where those |x| ~ 1000 outliers live. Quantizing any one
   Linear group alone (QKV, attn-out, FC1, FC2) is harmless (77-79% vs 78.7% on ORT-150),
   and all 48 Linears together (`only-linear`, 96 Q nodes, `Add`/attention left in
   FP16/FP32) give **84.51% (-0.7 vs fp32, ~ within noise)**.
4. That correct INT8 model is **1.15x faster than FP16 (4.92 vs 5.68 ms) and 2.8x faster than
   FP32**, far less than the 1.5-1.8x the ResNets get, because attention, LayerNorm, GELU
   and the residual stream stay in higher precision. The fully-quantized 4.58 ms model is
   only ~7% faster and produces garbage.
5. The raw-graph counterpart of `only-linear` could not be built: the group matcher found no
   weight Linears in the raw graph (0 Q nodes, so those rows are plain FP16 at 19%). The
   only raw INT8 result is the default one (0.03%).

## DLA (NVDLA): not available on this board

The Orin Nano has no DLA. `tensorrt.Builder().num_DLA_cores == 0`, there is no
`/dev/nvdla*`, and `trtexec --useDLACore=0` (with or without `--allowGPUFallback`, or
`--buildDLAStandalone`) fails with `Cannot create DLA engine, 0 not available` even
though `nvidia-l4t-dla-compiler` is installed. DLA compilation and latency need an
Orin NX or AGX Orin; none of the numbers above involve a DLA.
