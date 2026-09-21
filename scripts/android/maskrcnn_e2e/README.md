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
