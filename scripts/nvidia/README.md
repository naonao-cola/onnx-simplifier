# NVIDIA (TensorRT / CUDA) validation

Real-hardware checks of onnxsim output against the TensorRT builder, complementing the
hand-built-graph tests in `tests/test_tensorrt_*.py` (which never invoke TensorRT).

| file | interpreter | role |
|---|---|---|
| `qdq_pairs.py` | onnxsim (Python >= 3.11) | writes `<name>.orig.onnx` / `<name>.sim.onnx` pairs |
| `trt_harness.py` | system Python with `tensorrt` | builds engines, dumps per-layer tactic/precision, times inference, compares orig vs sim outputs |
| `modelopt_pipeline.py` | onnxsim + `nvidia-modelopt[onnx]` | fixes batch, runs `simplify()` and ModelOpt INT8 quantization on a real model, with an un-simplified control |
| `bench_trtexec.py` | any (stdlib) | builds/times every variant with `trtexec` and prints a table |

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

## ModelOpt + TensorRT GPU performance: ResNet-18 (ONNX model zoo `resnet18-v1-7`)

```sh
python3.12 scripts/nvidia/modelopt_pipeline.py resnet18.onnx /tmp/r18 --batch 1 8
python3 scripts/nvidia/bench_trtexec.py /tmp/r18 --duration 5
```

The zoo model is opset 8 / IR 3 with weights declared as graph inputs and a dynamic
batch; the pipeline lifts it to opset 17, pins the batch, then (`sim`) runs
`onnxsim.simplify()` (69 -> 49 nodes). `raw` is the same model without onnxsim.
ModelOpt's INT8 output is mixed FP16 + INT8 explicit Q/DQ, built with `--int8 --fp16`.
Mean GPU compute time from `trtexec` (`--noDataTransfers`), MAXN_SUPER power mode:

| batch | precision | raw (ms) | onnxsim (ms) | vs fp32 |
|---|---|---|---|---|
| 1 | fp32 | 1.566 | 1.591 | 1.0x |
| 1 | fp16 | 0.758 | 0.759 | 2.1x |
| 1 | ModelOpt int8+fp16 | 0.498 | 0.497 | 3.2x |
| 8 | fp32 | 8.131 | 8.022 | 1.0x |
| 8 | fp16 | 3.527 | 3.541 | 2.3x |
| 8 | ModelOpt int8+fp16 | 1.931 | 1.929 | 4.2x |

- INT8 is ~1.5x (batch 1) and ~1.8x (batch 8) faster than FP16 on the Orin Nano GPU.
- onnxsim makes **no measurable latency difference** here: TensorRT's own graph
  optimizer already folds BN and constants, so raw and simplified engines run at the
  same speed (differences are within run-to-run noise). onnxsim's value for this
  workflow is a clean, fixed-shape input (and ModelOpt accepts both variants).
- Output sanity (batch 8, 64 random inputs, engine outputs vs the fp32 engine): fp16
  cosine 0.99999 / top-1 agree 64/64; ModelOpt int8 (simplified) cosine 0.9956 / 59/64;
  (raw) 0.9956 / 56/64. **This is not an accuracy measurement**: calibration and
  these inputs are random noise, not ImageNet images. It only shows the INT8 engines
  are not degenerate. Real accuracy needs a real calibration/eval set.

## DLA (NVDLA): not available on this board

The Orin Nano has no DLA. `tensorrt.Builder().num_DLA_cores == 0`, there is no
`/dev/nvdla*`, and `trtexec --useDLACore=0` (with or without `--allowGPUFallback`, or
`--buildDLAStandalone`) fails with `Cannot create DLA engine, 0 not available` even
though `nvidia-l4t-dla-compiler` is installed. DLA compilation and latency need an
Orin NX or AGX Orin; none of the numbers above involve a DLA.
