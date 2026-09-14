# RESULTS: roofline benchmark on AMD Strix Halo NPU (`bench/amd_npu_roofline.py`)

Machine: AMD Ryzen AI Max+ 395 (Strix Halo, XDNA2, 6x8 AIE array), 128 GB
LPDDR5X. Software: XRT 2.25.37, NPU fw 1.1.2.65, Ryzen AI Software 1.8,
`onnxruntime-vitisai` 1.27.0. EP options: `cacheDir`/`cacheKey` set (first
compile 2--300 s depending on size, cached after); random inputs; 2 warmup +
10 timed runs.

Theory columns come from `theory` (FLOPs and ideal traffic via
`onnxsim.model_info`; `ideal@50T` assumes the XDNA2 50 INT8 TOPS peak at
100% utilization). `achieved` is measured FLOPs / measured latency.

## Conv towers (Conv+Bias+Relu, INT8 QDQ)

| model | GFLOP | ideal@50T | NPU | achieved | CPU (fp32 path ref) |
|---|---|---|---|---|---|
| conv_base: 16x 256ch/56^2 | 59.2 | 1.18 ms | 6.3 ms | **9.4 TOPS (19%)** | 51.9 ms |
| conv_wide: 16x 512ch/28^2 | 59.2 | 1.18 ms | 5.4 ms | **11.0 TOPS (22%)** | -- |
| conv_big: 12x 512ch/56^2 | 177.6 | 3.55 ms | 15.9 ms | **11.2 TOPS (22%)** | -- |
| conv_base fp32 (BF16 flow) | 59.2 | -- | 15.2 ms | 3.9 TFLOPS | 51.9 ms |

NPU runs above use the ORT-quantized models (`--quantizer ort`): every one
compiles to a single DPU subgraph. The `onnxsim`-quantized twins hit a
separate EP 1.27 tail-partitioning abort on deep +bias towers (the last
block falls back to CPU, then `Attrs doesn't contain attribute data`);
see PR #1421 for the zero-point fix that unblocked the ORT-CPU and
fuser-compatibility half of this. Wider channels tile slightly better
(11.2 vs 9.4 TOPS); batch scaling is perfectly linear (batch-4 twin of
conv_base: 25.2 ms for 4x FLOPs, same 9.4 TOPS), i.e. negligible dispatch
overhead -- the NPU is at its sustained rate.

## MatMul towers (MatMul+Add, INT8 QDQ, ORT-quantized)

| model | GFLOP | NPU | achieved |
|---|---|---|---|
| mm_1024: 32x 1024^2 | 68.7 | 10.1 ms | **6.8 TOPS (14%)** |
| mm_2048: 8x 2048^2 | 137.4 | 32.5 ms | 4.2 TOPS (8%) |
| mm_skinny: 8x M=256,K=N=4096 | 68.7 | 113 ms | 0.6 TOPS (1%) |
| mm_4096: 1x 4096^2 | 137.4 | 282 ms | 0.5 TOPS (1%) |

GEMM maps worse than convolution on the 4x8 CMC overlay: 1024-wide deep
chains are the sweet spot; a single giant GEMM and skinny-M shapes fall off
a cliff (weight streaming / tiling limits). A 4x1024^2 bare tower (no bias,
`onnxsim`-quantized) does execute on the NPU, but at 0.5 TOPS -- small
models are launch/transfer-bound.

## Reading the gap

~11 TOPS sustained vs 50 peak is normal for a single fused chain: peak
assumes dense MACs on every AIE tile every cycle; real runs pay LPDDR
weight streaming, PDI swaps, Quantize/Dequantize boundary ops on CPU, and
single-stream execution. The towers' value is comparative (conv vs GEMM,
width vs depth, batch scaling), not the absolute number.

Reproduce: `python bench/amd_npu_roofline.py prep [--quantizer ...]`
in this checkout, then `run <model> --provider vitisai` inside the Ryzen
AI venv (see the script header).
