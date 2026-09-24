# Native WebNN through rustnn, and tuning against tinygrad

**Status: experimental.** rustnn is marked "do not use in production"
upstream, and its Python API is still changing between releases.

[`docs/webnn.md`](webnn.md) covers WebNN inside a browser. There, onnxruntime-web's
WebNN execution provider runs the model, and it only works on a Chromium page with the
WebNN flag enabled. This page covers running the same kind of WebNN graph natively, from
Python, through [rustnn](https://github.com/rustnn/rustnn). It also covers timing it
against tinygrad on the same machine.

## Native WebNN implementations (as of September 2026)

| Implementation | Language / API | Backends | Usable outside a browser? |
| --- | --- | --- | --- |
| **[rustnn](https://github.com/rustnn/rustnn)** + [pywebnn](https://github.com/rustnn/pywebnn) | Rust; C ABI + C++ wrapper (`capi` feature); Python (`pip install pywebnn`) | ONNX Runtime (CPU/GPU/NPU EPs), TensorRT-RTX, Core ML, LiteRT, Huawei CANN | Yes. This is what onnxsim integrates. It runs the upstream WebNN WPT conformance tests in CI. |
| Chromium's `services/webnn` | C++, behind Mojo IPC | Windows ML / ONNX Runtime (earlier DirectML) on Windows, Core ML on macOS, LiteRT (TFLite) elsewhere | No. It only ships inside Chromium-based browsers and Electron, behind `--enable-features=WebMachineLearningNeuralNetwork`. rustnn's own docs use it as the reference lowering. |
| [webnn-native](https://github.com/webmachinelearning/webnn-native) | C/C++ (`webnn.h`, Dawn-style), Node.js binding | DirectML, OpenVINO, oneDNN, XNNPACK, MLAS | Yes in principle, but it is inactive: the last commit was in April 2023, and it tracks an old version of the spec. |
| [webnn-polyfill](https://github.com/webmachinelearning/webnn-polyfill) | JavaScript on TensorFlow.js | TF.js backends | Node.js only. It is a polyfill, not a native implementation. |

onnxruntime-web's WebNN EP (used by `scripts/convertmodel`) is a WebNN *client*, not an
implementation. It hands graphs to whichever implementation the browser has.

So for an out-of-browser, maintained WebNN today, rustnn is the only real option. Its
`webnn-graph` / `onnx2webnn` tools (`.webnn` text/JSON graphs) are separate Rust CLIs.

## `onnxsim.rustnn_runtime`: run a simplified model as a WebNN graph

```python
import onnx, onnxsim
from onnxsim import RustnnSession, find_unsupported_webnn_ops

model, ok = onnxsim.simplify(onnx.load("model.onnx"), overwrite_input_shapes={"x": [1, 3, 224, 224]})
print(find_unsupported_webnn_ops(model))  # {} when every op type has a lowering
session = RustnnSession(model, device_type="cpu")  # "auto" | "cpu" | "gpu" | "npu"
outputs = session.run({"x": x})
timing, _ = session.benchmark({"x": x}, warmup=2, runs=10)
```

pywebnn has no ONNX importer, so `build_webnn_graph` lowers the ONNX graph onto
`MLGraphBuilder` calls itself. The lowering has the same constraints a browser WebNN
graph has:

- **Static shapes only.** Pass `input_shapes=` for symbolic inputs, or simplify with
  `overwrite_input_shapes`.
- **Shape-like inputs must be constants.** This covers `Reshape`/`Expand` shape,
  `Slice` starts/ends/axes/steps, `Clip` bounds, `Pad` pads, and `Squeeze`/`Unsqueeze`
  and `Reduce*` axes. `onnxsim.check_webnn_support` flags the same thing for the
  browser. Running `onnxsim.simplify` first is what folds them. A graph whose `Reshape`
  shape comes from `Shape → Gather → Concat` fails to lower until it is simplified
  (see `tests/test_rustnn_runtime.py`).
- **Covered op types** are listed in `rustnn_runtime.WEBNN_SUPPORTED_OPS`:
  - elementwise unary, binary and comparison ops, `MatMul`, `Gemm`;
  - 2-D `Conv`/`ConvTranspose`/`MaxPool`/`AveragePool` and the global pools;
  - `BatchNormalization`/`InstanceNormalization`/`LayerNormalization`;
  - `Softmax`, `Gelu`, `Clip`, `Where`, `Cast`;
  - the reshape family, `Transpose`, `Concat`, `Split`, `Slice`, `Gather`, `Pad`;
  - `Reduce*` and `ArgMax`/`ArgMin`.
- **Rejected attribute combinations** raise `WebnnLoweringError`. Examples: 3-D
  convolution, `AveragePool(count_include_pad=1)` with padding, `Slice` with negative
  steps, and `ceil_mode=1` where it changes the output shape on a pywebnn build without
  `output_shape_rounding`.

pywebnn's ONNX Runtime backend `dlopen`s the `libonnxruntime` from the pip `onnxruntime`
package, or from `ORT_DYLIB_PATH`. `probe_rustnn(device_type, backend)` runs a canary and
returns `(ok, reason)`.

The released pywebnn 0.5.12 chooses its backend from `device_type` alone. `backend=`
(`"onnx"`, `"trtx"`, `"coreml"`, `"litert"`, `"cann"`) needs a pywebnn built from rustnn
`main`.

## `onnxsim.webnn_tinygrad_tuning`: WebNN vs. BEAM-tuned tinygrad, per node

onnxsim's existing tinygrad tuning (`onnxsim.webgpu_kernel_tuning` plus
`scripts/convertmodel/webgpu_kernel_tuner.mjs`) generates tinygrad kernel candidates in
Python and times them in a browser. It cannot use tinygrad's own `BEAM=N` search,
because that search compiles and times on a local `Device`.

With rustnn, both runtimes run natively in the same process, so:

```python
from onnxsim import webnn_tinygrad_tuning as t

result = t.tune_node(model, "conv_3", webnn_device_types=("cpu",), beams=(0, 2))
for timing in result.timings:
    print(timing.backend, timing.config, timing.median_ms, timing.max_abs_error, timing.error)
print("winner:", result.winner)            # fastest backend within tolerance of the reference
t.read_tuning_result(model, "conv_3")      # stored on node.metadata_props
t.tune_model(model)                        # every named Conv/ConvTranspose/MatMul/Gemm
t.benchmark_model(model)                   # whole graph, so tinygrad can fuse across nodes
```

For each node, `tune_node` does the following:

1. It isolates the node as a one-node model (`extract_node_model`). Constant inputs
   become initializers, and shapes come from shape inference.
2. It times that model on rustnn for each requested WebNN device type.
3. It times it on tinygrad (`OnnxRunner` under `TinyJit`) for each of
   `tinygrad_devices` (`CPU`, `METAL`, `CUDA`, `WEBGPU` via Dawn, ...; `None` is
   tinygrad's default) at each BEAM width. `0` is tinygrad's untuned kernels. `N > 0`
   runs tinygrad's own kernel search on that device.
4. It checks every output against `onnx.reference.ReferenceEvaluator`.
5. It marks each timing `within_tolerance` of `atol + rtol·max|ref|` and picks the
   fastest one that is.
6. It writes the result as JSON to the node's `metadata_props["onnxsim.webnn_tinygrad_tuning"]`.

Both sides are timed host to host: inputs are copied in and outputs read back, the same
span `MLContext.compute` covers. A missing pywebnn or tinygrad, or a node WebNN can't
lower, is recorded as an errored timing with its reason rather than raised.

Caveats:

- **The result is advisory.** Nothing in onnxsim or onnxruntime acts on the stored
  metadata yet.
- **Isolated per-node timing ignores cross-node fusion**, which tinygrad relies on.
  Compare with `benchmark_model` too.
- **tinygrad's native timing is not a browser WebGPU timing.** A tinygrad `WEBGPU`
  device through Dawn comes closest, but a winner here is evidence, not a guarantee, for
  the browser flow.

## On Apple silicon (Core ML / Neural Engine, Metal GPU)

With pywebnn 0.5.12 on macOS, the device types map as follows:

| WebNN `device_type` | What rustnn runs |
| --- | --- |
| `cpu` | ONNX Runtime, CPU execution provider |
| `npu` | Core ML, `MLComputeUnits.cpuAndNeuralEngine`, falling back to `.all` |
| `gpu` | **Not a GPU.** rustnn reports `onnx_gpu`, but only registers ONNX Runtime's CPU EP |

The GPU is covered by tinygrad's `METAL` device instead:

```python
t.tune_node(model, "conv_3", webnn_device_types=("cpu", "npu"),
            tinygrad_devices=("CPU", "METAL"), atol=2e-2, rtol=2e-2)
```

rustnn 0.5.12's Core ML backend has several bugs. CI on an M1 runner showed them, and
the rustnn source confirms them. So when `backend_info()["backend"] == "coreml"`,
`build_webnn_graph` works around them:

| rustnn 0.5.12 Core ML bug | What onnxsim does |
| --- | --- |
| Float32 outputs are read as contiguous, ignoring `MLMultiArray.strides`. Outputs the ANE produces have 64-byte-aligned rows, so every row after the first is scrambled. | Flattens every output to 1-D in the graph; `RustnnSession.run` reshapes it back. |
| The fused `bias` of `conv2d` / `convTranspose2d` / `gemm`'s `C` is dropped. | Emits the bias as an explicit `add`. |
| `argMax`/`argMin` can't return int64 (Core ML has none). | Requests int32; `run` casts back to ONNX's int64. |
| Integer outputs read back as ~0: rustnn only recognizes type code `3` for Int32, but Core ML reports `0x20020`, so ints fall into the Float32 branch. | Casts every non-float output to float32 in the graph; `run` casts back to the ONNX dtype. This is exact for integers up to 2^24, or 2048 if Core ML does the cast in float16. |
| `pad` (MIL `mode` never emitted, plus `constant_val` only when a value is set), `where` (MIL `select` rejects the uint8 condition), `layerNormalization` (wrong values) and strided `slice` (strides ignored) are broken. | Raises `WebnnLoweringError`, so these nodes are reported as not lowerable instead of returning wrong numbers. |

`tests/test_rustnn_runtime.py` forces this Core ML path on the ONNX Runtime CPU backend,
to check that the rewrites themselves are exact, independently of Core ML.

Core ML computes in float16 on the Neural Engine and GPU. The default `1e-3` tolerance
would therefore rule out `npu` results, so loosen `atol`/`rtol` when you compare it.

Core ML decides per op where it actually runs. A GitHub-hosted macOS runner is a virtual
machine and may not expose the Neural Engine at all, so an `npu` timing there shows that
the Core ML path works, not that the ANE was used.

`scripts/apple/benchmark_webnn_tinygrad.py` builds a few small models, simplifies them
and prints a per-node Markdown table of all four configurations. `--require npu,METAL`
makes it fail if either one produced no valid timing.

## CI

Tests: `tests/test_rustnn_runtime.py` and `tests/test_webnn_tinygrad_tuning.py`.
`ONNXSIM_RUSTNN_DEVICE_TYPES` (default `cpu`) and `ONNXSIM_TINYGRAD_DEVICES` (default:
tinygrad's default device) widen the device coverage; device types whose canary fails are
skipped.

- **Linux:** the `rustnn` job of `.github/workflows/backend-integration.yml` runs the
  tests on `cpu`.
- **macOS:** the `rustnn-webnn` job of `.github/workflows/apple-integration.yml` (runner
  `macos-15`) runs them on `cpu,npu` and `CPU,METAL`. It then runs the benchmark script
  and appends the table to the job summary.
