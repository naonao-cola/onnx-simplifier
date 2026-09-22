# Mask R-CNN through TVM's code generator on a Hexagon DSP

An end-to-end evaluation of `onnxmodelzoo/MaskRCNN-12-qdq` (torchvision Mask R-CNN R50-FPN, int8
QDQ, 3574 nodes) with Apache TVM 0.17 generating the code, plus an honest answer to "is it
deployable". Measured on a Xiaomi 12S (Hexagon V69; kernels compiled for `v73`), fixed
800x1088 input, 6 COCO val2017 images.

```bash
python prepare.py --workdir work                       # download, simplify, split
python e2e_eval.py --workdir work --model <MaskRCNN-12-qdq.onnx> --images img1.jpg img2.jpg ...
```

## What runs where

`prepare.py` simplifies the model with the input fixed to 800x1088 (onnxsim constant-folds the
`Shape`/`Gather`/`Resize`-scale arithmetic, 3574 -> 2983 nodes, outputs bit-identical) and splits
it automatically:

| Part | Nodes | Contents | Runs on |
|---|---:|---|---|
| `backbone.onnx` | 578 | ResNet-50, FPN, RPN head: 76 of 81 convs, **159.2 GMAC**, 14 outputs (4 FPN maps, RPN scores/deltas for 5 levels) | **TVM on the Hexagon DSP** (int8) |
| `rest.onnx` | 2405 | proposal decode, TopK, 85x NMS, RoiAlign, 4 MatMuls, mask head, post-processing (dynamic shapes) | ONNX Runtime (CPU) |

The backbone's QDQ pattern is converted to real integer convolutions with Relay's
`FakeQuantizationToInteger` (76 `qnn.conv2d` with uint8 activations and int8 weights,
`qnn.requantize` between layers, only 14 dequantizes left at the outputs), then compiled with
`relay.build` for Hexagon. Both halves reproduce the full model bit-exactly in ONNX Runtime.

## Results (TVM DSP backbone + ORT rest vs. full ORT model)

Detections with score > 0.5, matched by label and box IoU > 0.5:

| Image | Ref dets | TVM dets | Matched | Mean box IoU | Mean score |Δ| | Mean mask IoU | DSP backbone | ORT rest |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 000000000139 | 23 | 22 | 20 | 0.924 | 0.026 | 0.877 | 3.67 s | 0.48 s |
| 000000000632 | 17 | 20 | 17 | 0.869 | 0.014 | 0.684 | 3.70 s | 0.54 s |
| 000000000724 | 2 | 2 | 2 | 0.969 | 0.013 | 0.978 | 3.68 s | 0.21 s |
| 000000000785 | 2 | 3 | 2 | 0.933 | 0.001 | 0.836 | 3.72 s | 0.19 s |
| 000000001000 | 14 | 14 | 14 | 0.943 | 0.009 | 0.915 | 3.72 s | 0.31 s |
| 000000039769 | 3 | 3 | 3 | 0.956 | 0.001 | 0.936 | 3.72 s | 0.16 s |
| **total** | 61 | 64 | **58 (95%)** | ~0.93 | ~0.01 | ~0.87 | ~3.7 s | 0.2-0.5 s |

The three unmatched reference detections and six extra TVM detections (64 - 58) sit near the
0.5 threshold:
the int8 pipeline differs from ONNX Runtime's `QLinearConv` kernels by quantization-step-sized
noise (feature-map mean abs error ~0.06-0.11 against a +-14 range), which flips borderline
scores. This is agreement with the reference model, **not COCO mAP** (no annotations were used).

Backbone only, same input: ONNX Runtime CPU (32-core desktop) 0.22 s; TVM x86 int8 build,
untuned, 2.0 s; **TVM Hexagon int8, untuned, 3.7 s = 43 GMAC/s (~86 int8 GOPS)**; TVM Hexagon
fp32 did not finish in 27 minutes (untuned scalar-ish schedules for ~160 GMAC).

## Findings that matter for deployment

1. **TVM cannot compile the whole graph.** Relax's ONNX frontend rejects 11 operator types
   (NonZero, RoiAlign, NonMaxSuppression, TopK, ConvTranspose, QuantizeLinear, DequantizeLinear,
   ScatterElements, Floor, Not, And). Relay's frontend imports everything (about 10 minutes for
   the full graph), but `relay.vm.compile` on the dynamic remainder **segfaults inside VM
   lowering** (TVM 0.17, LLVM 17 and 19), also with `qnn.CanonicalizeOps`, `DynamicToStatic`
   and `FoldConstant`, and with an unlimited stack. At opt_level 1 the x86 conv strategy fails
   on a symbolic channel dimension, i.e. the importer loses static channel counts on the mask
   head. So the dynamic remainder (about 80% of the nodes, but far fewer MACs than the backbone:
   roughly 13 GMAC for the fc6 box-head layer on 1000 proposals plus the mask head on the kept
   detections) stays in another runtime. It is not negligible there: 0.2-0.5 s in ONNX Runtime,
   comparable to ORT's 0.22 s for the whole backbone.
2. **`Session.get_executor_from_factory` does not upload constants.** In TVM 0.17 it calls
   `get_graph_executor(json, lib)` only; the model's weights stay uninitialised on the DSP and
   every embedded constant reads as garbage (int8, int32 and float32 alike). This looked like a
   kernel miscompile until a NumPy comparison of single convolutions and a constant-only graph
   showed the pattern. Fix: `executor.load_params(tvm.runtime.save_param_dict(lib.get_params()))`.
3. Everything else needed for the int8 path works: `qnn.conv2d`/`qnn.requantize`/`qnn.add`/
   `qnn.sigmoid` lowering on Hexagon, uint8/int8 tensors over RPC, 55 MB activations within the
   DSP heap. `requantize` with `TONEAREST` rounding mismatched the host on a synthetic test
   (72% of elements); the default `UPWARD` mode used by the FQ2I pass is correct.

## Is it deployable?

**The dense part, yes as a prototype; the whole model, not yet.**

- Deployable today: a static-shape int8 backbone/FPN/RPN on Hexagon through TVM, numerically
  faithful to the reference, chained to a CPU runtime for the dynamic remainder. It runs on a
  dev phone over the ADB/TVM-RPC flow with an unsigned protection domain.
- Not deployable as is:
  - **Latency.** 3.7 s for the backbone is untuned: 43 GMAC/s, about 45% of the ~96 GMAC/s
    measured earlier on an isolated `vrmpy` NCHWc conv, and far from HTP-class throughput. It
    needs MetaSchedule/AutoTVM tuning or hand schedules for the 3x3/1x1 int8 shapes, and the
    remainder's 0.2-0.5 s on the CPU is a second cost centre.
  - **Whole graph in TVM.** Blocked by the Relay-VM segfault and Relax's missing operators;
    fixed-K, static-shape post-processing would sidestep dynamic shapes but is a model change.
  - **Fixed input size.** The backbone is compiled for 800x1088; other sizes recompile (or tile).
  - **Packaging.** The RPC skeleton needed a statically linked libc++ (the SDK 6.4 dynamic one
    fails on this DSP, see `../relink_hexagon_skel_static_libcxx.sh`), and the session helper
    bug above; a real app would use AOT/FastRPC packaging instead of the RPC server.
  - **Accuracy sign-off.** Six images vs. the reference model, not a COCO evaluation.

## The same model in tinygrad

tinygrad 0.14 (`pip install tinygrad`) loads the whole graph from ONNX in about 5 s (Relay needs
~10 minutes) and implements 41 of the model's 43 operator types out of the box; TVM's Relax
frontend lacks 11. `tinygrad_ops.py` adds the two missing ones, `NonMaxSuppression` (85 nodes)
and `RoiAlign` (8), on NumPy, plus a guard for zero-size `ScatterElements` (an FPN level that
received no RoIs). `tinygrad_eval.py` runs everything, backbone included, in tinygrad:

```bash
DEV=NV python tinygrad_eval.py --workdir work --model <MaskRCNN-12-qdq.onnx> --images ...
```

- **Correctness of the added operators.** Feeding ONNX Runtime's backbone features into the
  remainder graph in tinygrad reproduces ONNX Runtime's remainder output: labels exactly, box
  coordinates within 1.2e-4, scores within 3.6e-7 (`--rest-only`).
- **Full model on an RTX 5050 (NV backend), same six images** (QDQ emulated in fp32 by the
  frontend): 57 of 61 reference detections matched (93%), mean box IoU ~0.96, mean score
  difference ~0.01, mean mask IoU ~0.92, i.e. on par with the TVM int8 DSP pipeline above
  (58/61, ~0.93). It is slow, 72-160 s per image: an op-by-op Python interpreter, NumPy NMS and
  RoiAlign, and per-image kernel compilation.
- **One tinygrad quirk.** `OnnxRunner` caches Python constants across calls, which goes stale
  when shapes are data-dependent (the second image raised a shape mismatch), so the script builds
  a fresh runner per image.

### tinygrad's Hexagon path

`tinygrad_dsp_qemu.py` drives tinygrad's Hexagon renderer through its mock mode (clang
`--target=hexagon -mcpu=hexagonv65 -mhvx=v65`, executed under `qemu-hexagon-static`):

| Convolution (fp32) | max abs error vs NumPy | instructions per MAC |
|---|---:|---:|
| 1x1 64->64 @28x28 | 5.8e-7 | 3.2 |
| 3x3 64->64 @14x14 | 4.3e-6 | 4.7 |
| 3x3 128->128 @14x14 | 1.1e-5 | 4.0 |

Numerically correct, but the default schedule emits a single-thread scalar accumulator loop per
output element (no output-channel vectorization, no int8/`vrmpy`, no qfloat), about 16x the
instructions a vectorized HVX kernel needs. And it cannot reach this phone's DSP: tinygrad's
runtime opens `/dev/ion` and `/dev/adsprpc-smd` directly, which the ADB shell is denied (TVM's
RPC path works because libadsprpc opens the device through the HAL). Running tinygrad kernels on
the phone would need a FastRPC bridge, and competitive speed would need renderer work (HVX vector
accumulators with qfloat/`vrmpy`, multiple hardware threads), the same ground the TVM pass in
`../hexagon_qfloat.py` covers.

**Where that leaves the two stacks.** tinygrad has the better frontend and a robust
op-at-a-time execution model (a clear `NotImplementedError` per missing operator, no
whole-graph compiler to crash), and a QEMU test path that works out of the box. TVM has the
working int8 Hexagon backend. A practical combination is tinygrad as the graph executor
and correctness reference, with TVM-compiled Hexagon kernels behind it.

## Follow-up: does the int8 backbone actually use the DSP's hardware threads?

A quick disassembly check (grepping for the literal string `TVMBackendParallelLaunch` in
`hexagon-llvm-objdump -d` output) found zero matches in the compiled int8 backbone and
concluded parallel scheduling was silently dropped on Hexagon. **That conclusion was wrong** --
`objdump -d` does not print a symbol name for an indirect call through a GOT-resolved function
pointer, which is exactly how Hexagon RPC modules import `TVMBackendParallelLaunch` (a weak
`OBJECT` symbol resolved at load time, the same mechanism as `__TVMBackendAllocWorkspace` in
`../hexagon_sim_harness.py`). `hexagon-readelf --dyn-syms` / `-r` on the same `.so` show the
relocation is present and correctly wired:

```
18: 002e20d8   4 OBJECT WEAK DEFAULT 13 __TVMBackendParallelLaunch
```

Checked on the real phone with a controlled A/B (`relay.transform.FakeQuantizationToInteger` +
`relay.build`, same backbone, `te.schedule.Stage.parallel` monkeypatched to a no-op for the
"forced serial" row):

| Build | Median wall time |
|---|---:|
| default (parallel schedule, as shipped) | 3.68 s |
| target `num_cores=1` vs `num_cores=4` | 3.68 s vs 3.69 s -- no measurable difference |
| every `.parallel()` call replaced with a no-op (truly serial) | 12.56 s |

So parallel dispatch is genuine and contributes a real **3.4x** on this workload -- the earlier
"0 ParallelLaunch calls" claim was a tooling mistake (checking disassembly text instead of
relocations), not a finding about the runtime. The `num-cores` target attribute, however, has no
measurable effect: `CreateParallelLaunch` in TVM's Hexagon codegen path calls
`TVMBackendParallelLaunch(body, /*num_task=*/0, ...)` for a plain `.parallel()`-scheduled loop
(no explicit task count), so the actual worker-thread count is a Hexagon-runtime-side decision,
not something the compiled kernel's target string controls. With about 3.4x already realized,
thread count is not the remaining bottleneck for this workload; the leads noted above (per-shape
vrmpy tiling, MetaSchedule/AutoTVM tuning) remain the likely next gains.

