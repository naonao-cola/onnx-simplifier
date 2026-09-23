# Record-level emitters on the AX8850: device check

The record-level emitters were validated against native Pulsar2 builds,
record for record:

- `reshape_record_emit`
- `misc_op_record_emit`
- `matmul_record_emit.recalibrate`
- the Relu zero-point retarget in `elementwise_scale_emit`

That check only covers calibrations Pulsar2 has built. This one runs models
emitted at calibrations Pulsar2 **never built** on the card: the AX8850 in
`axcl-vm`, AXCL V3.6.5_P1. Where the emitter serves the step node, the
target is the ResNet18 step's predicted calibration
(`docs/axera-step-real-calibration.md`). Elsewhere it is a synthetic
calibration.

**Result:**
- 24 cases ran. Every emitted model is within 1 LSB of the quantized
  reference, except one element in 802,816 of the 3x3 Conv chain, which is
  at 2 LSB.
- Every health run afterwards was clean.
- The device found **two emitter bugs** that the native-build validation
  could not have caught. Both are fixed here:
  1. Log did not move its output scale.
  2. `recalibrate` accepted a Conv calibration that its program cannot
     represent.

Harness: `scripts/axera/emitter_device_check.py`. Tests:
`tests/test_axera_emitter_device_check.py`. The device tests are skipped
without the card. Raw results are in
`scripts/axera/fixtures/emitter_device_check/device_results.json`.

## Method

- **Reference.** The reference is the float graph with a
  QuantizeLinear/DequantizeLinear pair at every quantized
  `(consumer op, tensor)` and on every graph output, run in onnxruntime. For
  the MatMul chains the float graph is the build's own `t.onnx` (committed
  next to the results) and the placement comes from the template's
  `quant_axmodel.json`. For single ops it is written with `onnx.parser`.
- **Error.** Error is `|device - reference| / s_out`, in output LSBs.
- **Inputs.** Inputs span each input's quantization range. For Reduce and
  MatMul outputs they are narrowed until at most 5% of the reference output
  clamps, so the output is exercised inside its range.
- **Runs.** Every case makes three device runs, each holding
  `/tmp/axcl-device.lock` for that run only:
  1. The native template at its own calibration, as a control against its
     own reference. This gives the baseline.
  2. The emitted model at the target calibration.
  3. The native template again, as a health run. It must reproduce the
     control bit for bit.

## Results

"Emitted" is the max error in LSB; "> 1 LSB" is the fraction of elements
over 1 LSB. All health runs matched their controls.

| emitter | case | target | control | emitted | > 1 LSB |
| --- | --- | --- | ---: | ---: | ---: |
| Relu zp retarget | `[16,512,7,7]` | step `stage4_relu0` (zp 169) | 0 | 1 | 0 |
| Relu zp retarget | `[16,256,14,14]` | step `stage3_relu0` (zp 137) | 0 | 0 | 0 |
| Relu zp retarget | `[16,64,56,56]` | step `stage1_relu0` (zp 173) | 0 | 0 | 0 |
| Reshape | `64x64x3x3 -> 1x64x64x9` | step `Reshape_357` | 0 | 0 | 0 |
| Reshape | `16x512x7x7 -> 16x1x512x49` | step `Reshape_47` (s = 1.9e-6) | 1 | 0 | 0 |
| Reshape, tiled | `16x64x56x56 -> 16x1x64x3136` | step `Reshape_327` | 1 | 1 | 0 |
| Reshape, zero-point-0 template | `1024x28224 -> 1024x9x3136` (115 MB) | step `Reshape_442` | 1 | 1 | 0 |
| misc | Softmax `[16,1000]` | step `Softmax_2` | 0 | 1 | 0 |
| misc | Log `[16,1000]` (after the fix) | step `Log_3` | 0 | 0 | 0 |
| misc | MaxPool `[16,64,112,112]`, shared zp | step `pool0` | 0 | 1 | 0 |
| misc | ReduceSum `[16,1000]` axes 0,1 | step `ReduceSum_6` | 0 | 0 | 0 |
| misc | ReduceSum `[16,1,512,49]` axes 0,3 (packed zp_x) | step `ReduceSum_64` | 0 | 0 | 0 |
| misc | ReduceSum `[16,1,128,64]` axis 0 | step `ReduceSum_347` | 0 | 0 | 0 |
| misc | ReduceSum `[1024,9,3136]`, zero-point-0 template | step `ReduceSum_452` | 0 | 0 | 0 |
| misc | ReduceSum `[16,1,128,64]` moved to zp_y = 0 (records removed) | synthetic | 0 | 1 | 0 |
| misc | Neg, large program | step `Neg_8` (x255, y0) | 0 | 0 | 0 |
| misc | Neg, small program | synthetic s = 0.012 < 1/64 (x200, y55) | 0 | 0 | 0 |
| misc | ReduceMean `[16,512,7,7]` | synthetic scales | 0 | 0 | 0 |
| misc | Greater -> Cast (no calibration) | template as built | exact | exact | 0 |
| MatMul | fc dW `MatMul_38` | step calibration | 1 | 0 | 0 |
| MatMul | dX `MatMul_121` (Gather/Mul/Reshape chain) | step calibration | 1 | 0 | 0 |
| MatMul | dW `MatMul_142` | step calibration | 1 | 1 | 0 |
| MatMul | 3x3 Conv `stage3_conv1` (fused bias Add, per-tap requantize) | perturbed | 1 | 2 | 1.2e-6 |
| MatMul | dense-head Gemm (fused bias) | perturbed | 1 | 1 | 0 |

Notes on the table:

- **Perturbed targets.** The step calibration refuses the Conv and Gemm
  templates (`docs/axera-step-real-calibration.md`), so their targets are
  perturbed instead. Every template scale is multiplied by a random factor
  in [0.7, 1.4]. Scales that are a power of two apart share one factor
  (see "Conv" below).
- **Degenerate ReduceSum case.** `[16,1000]` over axes 0,1 has a
  one-element output, so it checks that single value only.

## Bug 1: Log did not move its output scale

At the step's calibration, the first emitted Log was off by up to 14.8 LSB
on **every** element. The decoded records showed the cause:

- The table lookup is dequantized by eight `0x0f50..0x0fc0` lanes holding
  `s_y`.
- `lane_values("Log")` knew only `1/s_x`, so those lanes kept the
  template's `s_y`: 0.036119 instead of 0.038625.
- A 6.9% scale error on outputs up to about 8.8 is about 15 LSB, which
  matches.

The emitter's native-build validation could not see this. Both native Log
builds have the same input calibration, so they share one `s_y`.

After adding `s_y` to Log's lanes, the emitted model is exact (0 LSB).

## Bug 2: a Conv calibration the program cannot carry

A 3x3 Conv legalized to a MatMul concatenates its taps as int8. For each
tap, Pulsar2 sets the Concat input's scale to exactly 2x its uint8
source's (`s_xcat = 2 s_x`), and the weight taps' to 1x.

The first perturbed target scaled `xcat` independently of `x`.
`recalibrate` accepted it, and the emitted model was off by up to 189 LSB
on every element. The same perturbation with the ratios kept (the
table's row) is within 1 LSB, apart from one element at 2 LSB.

The program has no record that carries the x-to-xcat ratio. So
`recalibrate` now refuses any calibration that changes an exact
power-of-two ratio (other than 1) between two template tensors
(`_check_fixed_ratios`). A calibration Pulsar2 produces never does.

The step coverage totals are unchanged.

## A card death during the run (not caused by this check)

Partway through, the card died while another agent's session was using it:

- Symptoms: `device 3: dead!` heartbeats, and `request ports ... Operation
  not permitted` on this check's *native control* runs.
- This check's previous case had ended with a clean health run.
- The other agent reloaded the full guest driver stack
  (`scripts/axera/vm/README.md`), and the card came back.
- The MatMul cases were re-run afterwards, each starting with a native
  control.
