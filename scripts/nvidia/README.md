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
| `sparsity_check.py` | onnxsim (`gen`) then system Python with `tensorrt` (`trt`) | checks whether TensorRT's builder gives 2:4-pruned `MatMul`/`Gemm` weights sparse tactics |
| `trt_nms_check.py` | onnxsim (`gen`), system Python with `tensorrt` (`trt`), any (`compare`) | differentially checks `rewrite_trt_batched_nms`'s output against the real `BatchedNMSDynamic_TRT` plugin |
| `run_cuda_feature_notebook.py` | onnxsim + a GPU `onnxruntime` | runs `examples/cuda_feature_tests/cuda_feature_tests.ipynb`'s tests as a plain script |
| `llm_pipeline.py` | onnxsim (Python >= 3.11) | pins shapes on a decoder-with-KV-cache LLM export and runs `simplify()` on it |

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

## 2:4 sparsity: does `convert_matmul_to_gemm` matter on TensorRT 10.3?

`scripts/nvidia/sparsity_check.py` checks `onnxsim/tensorrt_sparsity.py`'s claim (citing
[NVIDIA/TensorRT#2271](https://github.com/NVIDIA/TensorRT/issues/2271), filed against an
older TensorRT) that N:M sparse math only ever applies to `Gemm`, not `MatMul`, so
2:4-pruned transformer FFN/attention weights (which are wired through `MatMul`, since
their activation is 3-D) need `convert_matmul_to_gemm` first. **Scope note**: that claim
is specifically about *ONNX Runtime's* TensorRT execution provider
(`ORT_TENSORRT_SPARSITY_ENABLE=1`); this checks the underlying TensorRT builder directly
(same as `trt_harness.py`), which ORT's EP delegates to -- informative, not a direct test
of the literal claim (see `run_cuda_feature_notebook.py`'s findings for why ORT-TRT-EP
itself could not be run on this board).

```sh
python3.12 scripts/nvidia/sparsity_check.py gen /tmp/sp --layers 6      # onnxsim venv
python3.10 scripts/nvidia/sparsity_check.py trt /tmp/sp --runs 3        # tensorrt venv
```

Built a 6-layer ViT-B-shaped MLP stack (768&rarr;3072 ReLU 3072&rarr;768) at three
activation shapes -- `2d197` `[197,768]`, `3d197` `[1,197,768]` (the realistic
batched/transformer case), `2d2048` `[2048,768]` (larger, 2-D) -- each as dense and
`apply_magnitude_pruning(n=2, m=4)`-pruned weights (correctly 50% zero, valid 2:4-along-K
pattern, confirmed programmatically), both as plain `MatMul` and after
`convert_matmul_to_gemm` (value-preserving: `max_abs_diff = 0.0` against the un-converted
model). Built each of the 12 resulting models with `trtexec --fp16
--sparsity={disable,enable,force}` (36 engines) and timed the successful ones (median of
2 rounds; TensorRT's internal timing-based tactic autotuner was still re-run per engine).

**Eligibility** (`enable` mode; `force` ignores actual weight content and is a sanity
check, not a real signal):

| shape | dense (either op) | pruned `MatMul` | pruned `Gemm` |
|---|---|---|---|
| `2d197` (2-D) | 0/12 eligible | **12/12** | 12/12 |
| `3d197` (3-D, batched) | 0/12 eligible | **12/12** | 12/12 |
| `2d2048` (2-D, large) | 0/12 eligible | 11/12 | 12/12 |

**`MatMul` does get sparse-tactic eligibility on TensorRT 10.3** -- at small/medium
shapes, identically to `Gemm`; at the largest shape tested, nearly so (11 vs 12). This
updates the premise behind issue #2271 for current TensorRT: it is no longer categorically
true that `MatMul` never gets N:M sparse math. The one clean exception found: `force`
mode on the 3-D-activation *dense* `MatMul` got 0/12 eligible (vs `Gemm`'s 12/12 via its
reshape scaffold) -- but `force` on dense weights isn't the real-world case either way.

**Latency** (median GPU ms, `enable` mode -- the real-world setting):

| shape | dense | pruned `MatMul` | pruned `Gemm` | pruning speedup |
|---|---|---|---|---|
| `2d197` | 1.33 ms | 1.061 ms | 1.052 ms | ~1.26x, both ops tied (&lt;1% apart) |
| `3d197` | dense `MatMul` 1.34 ms / `Gemm` 1.48 ms | **1.059 ms** | 1.212 ms | `MatMul` 1.27x; converting to `Gemm` is **13% slower**, not faster |
| `2d2048` | ~12.2 ms | **10.17 ms** | 10.84 ms | `MatMul` 1.20x; converting to `Gemm` is **7% slower** |

At every shape tested, `convert_matmul_to_gemm` gave **no latency benefit** -- and at the
two shapes where it isn't a zero-overhead rewrite (`3d197`'s reshape/unflatten scaffold
around the batched activation; `2d2048`, plain 2-D, where the extra overhead is less
obvious), the *converted* engine was measurably **slower** than leaving it as `MatMul`.
Pruning itself is worth it either way (~1.2-1.3x over dense at `fp16`); the conversion
pass is not, at least at these shapes on TRT 10.3.

Numeric correctness: `convert_matmul_to_gemm` is exact (checked in `gen`, fp32); the fp16
engines' relative error vs the fp32 reference stayed in the same range regardless of form
(`~0.1%-1%`), and the pruned+`enable` engines were consistently *more* accurate than their
dense fp16 counterparts (e.g. `2d197`: `1.25e-3` vs `1.05e-2`) -- plausibly because a sparse
kernel accumulates over fewer (only nonzero) terms, though this isn't a guaranteed property
and is incidental to the claim under test.

**Caveat -- tactic-selection nondeterminism**: two independent full `trt`-stage runs gave
the *same* eligibility pattern above both times, but a different `chosen` count for one
config (`2d2048 pruned_matmul enable`: 6/11 chosen in run 1, 11/11 in run 2 -- TensorRT's
timing-based autotuner re-profiles tactics per build and can land on a different one).
The eligibility table is corroborated across both runs; the latency table is from one
full run only (each number's own 2-sample spread was tight, 0.0-1.1%, but a re-build
could plausibly pick different tactics and shift the exact numbers, especially at
`2d2048` where `MatMul`'s eligibility was already partial). Given this, treat the
qualitative result -- pruning helps, conversion doesn't, on TRT 10.3 -- as the reliable
takeaway rather than the exact percentages.

## CUDA feature notebook (`examples/cuda_feature_tests/`): blocked on this board

`scripts/nvidia/run_cuda_feature_notebook.py` runs `cuda_feature_tests.ipynb`'s 7 tests
(Tests A-G: `backend.run_model` CPU/CUDA parity, `simplify(providers=CUDA)` GPU constant
folding, the `(name, options)` device-pinning tuple form, CLI `--cuda`, the
unavailable-provider `ValueError`, DLPack zero-copy with a CUDA `torch.Tensor`, and
`measure_accuracy_drop(providers=CUDA)`) as a plain script, so they can run outside
Jupyter/Colab -- the notebook is explicitly hand-run-only and has never executed on real
hardware.

```sh
python3.12 scripts/nvidia/run_cuda_feature_notebook.py
```

**Could not actually run any of the 7 tests on this board.** onnxsim requires Python >=
3.11 (its wheel is `cp312-abi3`), but the only real GPU-capable `onnxruntime` for JetPack
6/CUDA 12.6 -- the Jetson AI Lab index
(`--index-url https://pypi.jetson-ai-lab.io/jp6/cu126`) -- ships `onnxruntime-gpu` (1.24.0)
for `cp310` only (confirmed with `uv pip install --dry-run`, not by guessing: it resolves
cleanly against `/usr/bin/python3.10` and fails ABI resolution against 3.12). No single
interpreter on this board can import both `onnxsim` and a working GPU `onnxruntime`.

PyPI's plain `onnxruntime-gpu==1.30.0` does have a `cp312`/aarch64 wheel and installs
without error, so it is tempting to reach for as a workaround -- but it requires CUDA
13.x/cuDNN 9.x, and this board runs CUDA 12.6. Concretely reproduced (not just inferred
from the version requirement): `rt.get_available_providers()` lists
`CUDAExecutionProvider` regardless -- that check is static metadata, not a real capability
probe -- but creating an `InferenceSession` with it requested fails to `dlopen
libcublasLt.so.13` and **silently falls back to `CPUExecutionProvider`**, with no
exception, only a stderr warning:

```
Failed to load library .../libonnxruntime_providers_cuda.so with error:
  libcublasLt.so.13: cannot open shared object file: No such file or directory
Failed to create CUDAExecutionProvider. Require cuDNN 9.* and CUDA 13.*.
```

**This is worth a maintainer's attention beyond this board's mismatch**: onnxsim's own
provider validation (`onnxsim/backend.py:178`, `available = set(rt.get_available_providers())`)
checks exactly the same static list that just lied above. So `onnxsim.simplify(providers=
["CUDAExecutionProvider"])` or `backend.run_model(..., providers=CUDA)` on a
version-mismatched `onnxruntime-gpu` install raises nothing and silently returns a
CPU-computed result -- indistinguishable from a real GPU run to the caller, including
Test A's "CPU vs CUDA parity" check, which would trivially pass either way (both sides
would be CPU). Not fixed here (a behavior change to a core runtime path deserves its own
review, not a bundled verification-script PR); the fix would compare each requested
provider against the *session's actual* `sess.get_providers()` after construction (which
does reflect real fallback) rather than trusting `get_available_providers()` alone, and
warn or raise on mismatch.

Not attempted: building `onnxruntime-gpu` from source for `cp312`/CUDA 12.6/sm_87 (a
multi-hour build disproportionate to this check). The script itself is unaffected by any
of this and should work as-is on a board where a GPU `onnxruntime` matching onnxsim's
Python floor actually exists (e.g. an x86 box with `onnxruntime-gpu`, or a future JetPack
release on CUDA 13).

## Decoder LLM: Qwen2.5-0.5B-Instruct (KV cache, RoPE, GQA, RMSNorm)

`scripts/nvidia/llm_pipeline.py` checks onnxsim on a real decoder-with-KV-cache export --
a different shape of graph than any CNN/ViT above: RoPE and RMSNorm hand-decomposed into
primitive ops, grouped-query attention (14 query heads sharing 2 KV heads), and 48
`past_key_values.N.{key,value}` inputs / 48 `present.N.{key,value}` outputs alongside
`input_ids`/`attention_mask`/`position_ids`. Model:
[`onnx-community/Qwen2.5-0.5B-Instruct`](https://huggingface.co/onnx-community/Qwen2.5-0.5B-Instruct)
`onnx/model_fp16.onnx` (opset 14, IR 10, 2759 nodes, 24 layers, head_dim 64; no `If` node --
`past_key_values` are required inputs, a length-0 cache standing in for prefill).

```sh
python3.12 scripts/nvidia/llm_pipeline.py model_fp16.onnx /tmp/llm --seq 1 --past 31   # decode step
```

Pinning `batch_size`/`sequence_length`/`past_sequence_length` to concrete values (batch 1,
1 new token, 31 cached) and running `simplify()`:

**2759 -> 1703 nodes (-38%)**, reproduced across two independent runs (bit-identical node
counts and, per ORT CPU, bit-identical logits: `max_abs_diff = 0.0` between the raw and
simplified fixed-shape models, argmax token matches). `RMSNorm`'s decomposition (`Pow`/
`ReduceMean`/`Sqrt`/`Div`, 49 each) and RoPE's (`Neg`, 48) are **untouched** -- no fusion
pass recognizes either pattern here -- so the whole reduction comes from constant-folding
now-static shape/dtype bookkeeping (`Concat` 267 -> 96, `Expand` 54 -> 48, `Cast` 99 -> 98):
once every KV-cache dimension is a fixed number instead of a symbolic
`past_sequence_length`, the `Shape`/`Gather`/`Range`/`Where`/`Concat` chains that computed
those dimensions at graph-run time become foldable constants outright. ORT CPU session
load+run was also faster on the simplified model (1.9s vs 3.1s, one sample, not a
controlled benchmark).

**Could not complete a TensorRT engine build for either variant on this board.**
TensorRT compiles this attention pattern as a single large fused ("Myelin") subgraph
rather than decomposable layers, and its constant-weight staging buffer needs
**physically contiguous** GPU memory: the raw model requested a 988 MB contiguous
allocation, the *simplified* model (fewer nodes, same weights) needed only 272 MB at
first -- a real ~3.6x reduction from `simplify()`:

```
NvMapMemAllocInternalTagged: ... error 12
Error Code 1: Cuda Runtime (out of memory)
Requested amount of GPU memory (272573440 bytes) could not be allocated.
```

The board's CMA (contiguous memory allocator) pool -- `CmaTotal` in `/proc/meminfo` --
was originally capped at **256 MB**, well under 272 MB. Raising it (`cma=` in
`/boot/extlinux/extlinux.conf`, needs `sudo` + reboot) turned out to be more involved
than a size fix, and **did not unblock the build**:

- `cma=1024M` **failed to reserve at boot** (`dmesg`: `cma: Failed to reserve 1024 MiB`)
  -- this board's physical memory layout has other fixed carveouts (framebuffer, VPR,
  camera debug, PVA, ...) that a 1 GB contiguous request can't fit around -- leaving CMA
  at **0 MB**, worse than the default.
- `cma=512M` **did** reserve cleanly (`dmesg`: `cma: Reserved 512 MiB`, confirmed in
  `/proc/meminfo`) -- but with CMA actually available, TensorRT's tactic autotuner
  stopped picking the 272 MB strategy at all and consistently requested **988 MB**
  instead (matching the raw model's original request almost exactly), which still
  exceeds the 512 MB pool. `--noBuilderCache --noCompilationCache`,
  `--builderOptimizationLevel=0/1/3`, and dropping the page cache before the build made
  no difference -- this looks like a genuine TensorRT tactic-selection interaction with
  CMA availability (a bigger contiguous pool makes a more memory-hungry fusion strategy
  look viable to the cost model, which then doesn't fit either), not a simple
  size-threshold problem. Not investigated further (would need TensorRT internals access
  this write-up doesn't have); a CMA size between 256 MB and 512 MB, never tried, might
  keep the cheaper tactic while still fitting it, but that needs another reboot per
  attempt and was not pursued past this point.

CMA was left at 512 MB (a reasonable general-purpose increase for future GPU work on
this board) rather than reverted. A dynamic-INT8 variant of the same model
(`model_int8.onnx`, `MatMulInteger`/`DynamicQuantizeLinear`) was fetched as a
smaller-weights workaround attempt but not pursued: it is a substantially different graph
shape (dynamic per-token activation quantization) that risks confounding TensorRT parser
compatibility with the actual question, for uncertain payoff given the same ceiling.

**Takeaway**: on hardware with enough contiguous GPU memory, this would be worth finishing
(TensorRT build/latency/correctness comparison, matching the CNN/ViT sections above).
Here, the graph-level finding stands on its own: `simplify()` is exact and shrinks a real
decoder LLM export by more than a third, entirely by folding now-static KV-cache shape
arithmetic, without touching (or needing to touch) RoPE/RMSNorm/GQA at all.

