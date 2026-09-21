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

## DLA (NVDLA): not available on this board

The Orin Nano has no DLA. `tensorrt.Builder().num_DLA_cores == 0`, there is no
`/dev/nvdla*`, and `trtexec --useDLACore=0` (with or without `--allowGPUFallback`, or
`--buildDLAStandalone`) fails with `Cannot create DLA engine, 0 not available` even
though `nvidia-l4t-dla-compiler` is installed. DLA compilation and latency need an
Orin NX or AGX Orin; none of the numbers above involve a DLA.
