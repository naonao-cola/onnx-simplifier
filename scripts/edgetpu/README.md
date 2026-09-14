# Edge TPU peak-performance benchmark

Measures how close a Coral Edge TPU gets to its **4 TOPS INT8** spec peak
(2 TOPS/W) across workloads spanning the roofline, using models built,
quantized and compiled through onnxsim's own TFLite/Edge TPU export
(`io_layout="nhwc"`, full-int8, uint8 I/O).

## This one needs real Edge TPU hardware (for timing only)

Everything except the timed loops runs without a device: `models.py` builds
the suite with plain `onnx`, and `peak_benchmark.py` quantizes/compiles it
with TensorFlow + `edgetpu_compiler`, printing exact MACs, mapping stats,
on-chip/off-chip memory and roofline predictions. Only
`benchmark_device.py`'s `invoke()` loop needs a plugged-in Coral device with
readable USB nodes (see its docstring for the one-time `libedgetpu` + udev
setup) — like `scripts/amd`, there is no host emulator, so nothing here is
wired into CI.

## What the suite covers

`models.py` (all int8, channel-last, single Edge TPU subgraph when compiled):

| model | workload | GMAC | params | role |
|---|---|---|---|---|
| `pointwise-8` | 8x 1x1, 256ch/16px | 0.134 | 0.53 MB | link-bound reference |
| `dense3x3-6` | 6x 3x3, 128ch/16px | 0.226 | 0.89 MB | near compute-bound on USB3 |
| `mbblock-4` | 4x inverted residuals, 32px | 0.038 | 0.04 MB | realistic mobile workload |
| `fc-4k` | flatten + 4096x1024 Gemm | 0.004 | 4.20 MB | memory-bound anchor |
| `cliff-64x32` | 1x 3x3, 64ch/32px | 0.038 | 0.04 MB | refuses NCHW compilation, maps fully as NHWC |
| `pointwise-48` | 48x 1x1, 256ch/16px | 0.805 | 3.16 MB | compute-bound probe |
| `big3x3-8x128` | 8x 3x3, 128ch/64px | 4.832 | 1.18 MB | large-spatial efficiency probe |
| `big3x3-4x256` | 4x 3x3, 256ch/64px | 9.664 | 2.36 MB | peak sustained probe (1.03 TOPS at max clocks) |

Conv weights use He scaling so activations stay bounded through deep stacks
(like trained nets); unscaled random weights explode and break int8
quantization mid-graph, which would benchmark the converter's float fallback
instead of the TPU.

## How peak performance breaks down (measured + predicted)

Spec peak is unattainable sustained (systolic fill + SRAM traffic); the
roofline below assumes the literature best-CNN ceiling (~40% of peak,
≈1.6 TOPS) for compute and 40/350 MB/s effective USB bulk throughput, with
params cached on-chip after the first inference (steady state transfers I/O
only). Predicted sustained TOPS (compute = 2 ops/MAC):

| model | MAC/B | compute | USB2 xfer | USB3 xfer | USB2 TOPS | USB3 TOPS | USB3 regime |
|---|---|---|---|---|---|---|---|
| pointwise-8 | 1024 | 0.17 ms | 3.28 ms | 0.37 ms | 0.08 | 0.72 | link |
| dense3x3-6 | 3456 | 0.28 ms | 1.64 ms | 0.19 ms | 0.28 | 1.60 | compute |
| mbblock-4 | 584 | 0.05 ms | 1.64 ms | 0.19 ms | 0.05 | 0.41 | link |
| fc-4k | 819 | 0.01 ms | 0.13 ms | 0.02 ms | 0.07 | 0.57 | link |
| cliff-64x32 | 288 | 0.05 ms | 3.28 ms | 0.37 ms | 0.02 | 0.20 | link |
| pointwise-48 | 6144 | 1.01 ms | 3.28 ms | 0.37 ms | 0.49 | 1.60 | compute |

Reading it:

- **Achievable peak, ideal dense compute-bound models: ~1.6 TOPS (~40%)**
  on USB3 at standard frequency (up to ~2x on the max-frequency runtime).
- **Realistic mobile CNNs: 0.1–0.4 TOPS** — Google's own USB3 numbers agree:
  MobileNetV2 0.6 GOPs @ 2.6 ms = 0.23 TOPS (5.8%); ResNet-50 ~4.1 GOPs @
  49 ms = 0.08 TOPS (2.1%, its 25.6 MB of params stream over USB).
- **On a USB 2.0 host link everything is link-bound** (0.02–0.49 TOPS
  predicted here) — moving the accelerator to USB3 is worth ~7x on small
  models. First inference additionally pays the model load (pointwise-48's
  3.16 MB ≈ 80 ms USB2 / 9 ms USB3).

## Measured on-device (USB3, Python LiteRT, median of 500)

Timed with `benchmark_device.py` on a Coral USB Accelerator (SuperSpeed
5 Gbps link), standard-frequency runtime unless noted, 50 warmup + 500 timed
invokes per model:

| model | GMAC | median | TOPS | %peak |
|---|---|---|---|---|
| pointwise-8 | 0.134 | 1.212 ms | 0.22 | 5.6% |
| dense3x3-6 | 0.226 | 0.901 ms | 0.50 | 13.0% |
| mbblock-4 | 0.038 | 1.008 ms | 0.08 | 1.9% |
| fc-4k | 0.004 | 0.276 ms | 0.03 | 0.8% |
| cliff-64x32 | 0.038 | 1.006 ms | 0.08 | 1.9% |
| pointwise-48 | 0.805 | 2.521 ms | 0.65 | 16.2% |
| big3x3-8x128 | 4.832 | 13.284 ms | 0.73 | 18.4% |
| big3x3-4x256 | 9.664 | 25.974 ms | 0.75 | 18.7% |

Max-frequency runtime (`libedgetpu1-max`, fresh delegate per model):

| model | median | TOPS | %peak |
|---|---|---|---|
| big3x3-4x256 (9.664 GMAC) | 18.685 ms | 1.03 | 25.9% |
| pointwise-48 | 1.681 ms | 0.96 | 24.0% |
| dense3x3-6 | 0.689 ms | 0.66 | 16.4% |

Takeaways:

- **Peak measured: 1.03 TOPS sustained (26% of the 4 TOPS peak)** on the
  9.7 GMAC dense model at max clocks — in line with the literature best-CNN
  ceiling once the ~1.4x max/std clock ratio is accounted for.
- Small models sit on a **~0.3–1.0 ms fixed floor** (Python + delegate
  scheduling + USB round trip; fc-4k at 0.276 ms is the purest probe), so
  anything under ~0.5 GMAC reports single-digit %peak regardless of the ASIC.
- Throughput scales linearly once compute dominates (S8 is exactly 2x S7's
  MACs at 2x the latency, at higher efficiency than the 16px models — larger
  spatial dims fill the 64x64 systolic array better).
- Caution: at max frequency, sharing one delegate across models back-to-back
  segfaulted here (likely thermal); `benchmark_device.py` uses a fresh
  delegate per model — keep cooldowns between runs at max clocks.

## Running it

```bash
# no hardware needed: build + quantize + compile + roofline table
python peak_benchmark.py --out-dir peak_models
python peak_benchmark.py --models pointwise-48 cliff-64x32 --samples 10

# with a device: timed loops, TOPS vs the 4 TOPS peak
python benchmark_device.py --dir peak_models --runs 500
```
