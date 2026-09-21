# NVIDIA (TensorRT / CUDA) validation

Real-hardware checks of onnxsim output against the TensorRT builder, complementing the
hand-built-graph tests in `tests/test_tensorrt_*.py` (which never invoke TensorRT).

| file | interpreter | role |
|---|---|---|
| `qdq_pairs.py` | onnxsim (Python >= 3.11) | writes `<name>.orig.onnx` / `<name>.sim.onnx` pairs |
| `trt_harness.py` | system Python with `tensorrt` | builds engines, dumps per-layer tactic/precision, times inference, compares orig vs sim outputs |

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
