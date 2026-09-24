# Live-operand MatMul templates for every ResNet18 step node

`docs/axera-matmul-record-emit.md` (#1870) showed that a live-operand MatMul
recalibrates from its scales alone, once one Pulsar2 build of its shape
exists. This adds those builds for the ResNet18 training step
(`t6-r18fold/step.onnx`) and a manifest that says which template serves which
step node.

## Status

Built so far, each as a template plus a held-out build, recalibrating onto
each other exactly in both directions (records and `npu_params`):

| batch | templates | step nodes served |
| --- | ---: | ---: |
| fc (dX, dW, forward Gemm) | 3 | 3 |
| dX | 7 of 9, plus the existing 512->512 template | 14 of 19 |
| dW | 6 of 11 | 8 of 20 |
| Conv | 6 of 11 | 8 of 20 |

Coverage (`tinygrad_ax_backend.coverage_report` on `step.onnx`, master's
backend): MatMul 41 refused -> 24 conditional, 17 refused; Gemm 1 -> 1
conditional; Conv 20 refused -> 8 conditional, 12 refused; totals
82 / 324 / 698 -> 82 / 357 / 665 (covered / conditional / refused).

The remaining builds are in progress, largest last.

## Templates at the step's real calibration

`coverage_report(..., calibration=)` (#1897) recalibrates each template onto
the step's *predicted* scales. The first round of templates failed that: they
had been calibrated on made-up ranges, so their zero-point classes, int8/uint8
boundaries and scale ties did not match the step. This round
(`~/npu-scratch/t_step_real/make_real.py`) builds every chain from the step's
own predicted quantization instead:

- **Ranges from the prediction.** Each chain input's range is rebuilt from its
  predicted `(scale, zero point)` in the step (scale x1.12 and zp +6 for the
  template, x0.83 and zp -9 for the held-out build), with the two extremes
  pinned in every sample so MinMax sees exactly that range. The zero-point
  class (symmetric int8 / uint8 zp 0 / uint8 zp > 0) is the step's by
  construction.
- **Side outputs keep a mixed-use input uint8.** A chain input that has other
  consumers in the step (the SGD update reads every weight; a residual Add
  reads a block's input) is uint8 there and the chain requantizes it. Alone
  in a template its only consumer would be the MatMul, so Pulsar2 would make
  it int8. An `Identity` to an extra graph output (`<t>__side`) keeps it
  uint8. The template-only tensors are listed as `aliases` in the manifest
  and take their source's scale.
- **A Relu prefix for an unfused Relu output.** A Conv input that is a
  non-fused Relu's output shares its pre-activation's quantization in the
  step (nonnegative data, nonzero zero point). The template feeds it from a
  `<t>__pre` graph input through a Relu, which reproduces that.
- **Predicted ties are rejected before building.** `make_real.ties` runs
  `step_calibration.assign` on the template's own data and rejects any two
  differently-scaled tensors that share a zero point, and any zero point in a
  different class than the step's. It then widens the bias, shifts the zero
  points, or zero-centres the weights (a sum over Relu outputs with a
  one-signed weight comes out one-signed: zero point 255 downstream).
- **dX kernels include their Reshape/Transpose path.** The kernel operand is
  cut back to the trainable weight, which is uint8 (the Conv and the SGD
  update read it) while the rearranged kernel is int8, so the requantize is
  inside the chain.

Each template must pass two checks: exact in both directions against its
held-out build, and `recalibrate` onto the predicted step scales for every
node it serves.

| template | served | |
| --- | ---: | --- |
| dW 128, 61, 265, 367, 471 | 12 | exact, and all served nodes recalibrate |
| dX 121, 156 (extended kernel path) | 4 | exact, and all served nodes recalibrate |
| Conv stem 7x7, stage2 conv0, stage3 conv0, stage2 conv2, stage3 conv2 | 5 | exact, and all served nodes recalibrate |

At the predicted calibration (`tests/test_axera_step_calibration.py`), totals
go from 467 covered / 637 refused to **481 / 623**. MatMul goes from 24 to
36 of 41, Conv from 0 to 5 of 20. Reshape goes from 153 to 150 covered.
That is because the new int8 rule (below) makes the rearranged-kernel
Reshapes int8. Inside a covered extended dX chain they are covered again;
the other dX chains still use first-round templates.

What the emitter had to learn for these:

- **A MatMul's int8 view of a uint8 input is its own scale.**
  `quant_axmodel.json` lists a mixed-use tensor once per consumer: uint8 for
  the side output, int8 at its own scale for the MatMul. The dW chains'
  activation lanes hold `1/s` of the int8 view. `quant_scales` keeps it as
  `<t>#i8` (`I8`). The backend supplies it from the calibration's
  `consumer_int8_scale`, or from the int8 tensor it overlaps when no MatMul
  consumes it directly (the weight at the head of a dX kernel path).
- **The Concat header** (`cat15`): a 3x3 Conv chain stores the activation
  taps' and the weight taps' ratio into their Concat as two Q15 halves in
  `npu_params`. Every earlier build stored 32768 in the weight half, so a
  missing role went unnoticed until stage2 conv1's pair stored 32768 vs 32767.
- **Zero-point bytes in `npu_params`** (`zp8`): a signed-input Conv chain
  stores its input zero point as a run of equal bytes (9 or 32 bytes, not
  word-aligned).
- **`0x1eb0` as a zero-point register**, when its value is a zero point (it
  also carries `0x80000000`).
- **The requantize offset uses a float32 product**, `zp_x * f32(s_x/s_y)`
  rounded to float32 before the subtraction. With a double product, 2 of 6
  signed-input offsets came out one too high.
- **One stand-in per shared quantization** in the role searches, so the stem's
  147 taps don't make them cubic (5 minutes -> 4 s). `recalibrate` refuses
  when tensors that share a quantization in the template would part at the new
  scales.

Still refused, and why:

- **3x3 Conv stage2/3/4 conv1 (9 nodes):** the Concat header is not found.
  With the Relu prefix, the taps' ratio into the Concat is above 1, and the
  header then has a different form. That needs decoding.
- **stage1 conv0 template (serves 4):** it fits the 2 nodes whose input is
  the max pool. For the 2 whose input is an unfused Relu, the Concat ratio is
  above 1 (a different program). Those need a Relu-prefix template of their
  own.
- **stage4 conv0 / stage4 conv2:** held-out pair mismatch. For conv2 the
  `npu_params` DMA table differs by one tile rotation (#1836-style noise),
  and a second build is needed to confirm. For conv0 the record structure
  differs between the two calibrations.
- **The extended dX chains 54, 138, 223, 240, 258, 342, 360:** each
  recalibrates onto the step, but its two builds differ in record structure,
  so the pair check can't pass yet. 240 also has a ratio tie. These nodes keep
  master's first-round templates.
- **Gemm (with its ReduceMean):** one lane (`0x3d84030e`) matches no formula
  yet. The mean pool also writes `zp * 49` to a zero-point register, which is
  decoded (`zpk`).
- **dX 325:** Pulsar2 does not finish it within 40 minutes (tried 4 times; the
  memory cap killed the first attempt).

## A fused bias Add has three more calibration words

The first Gemm template recalibrated with 4 records and 2 `npu_params`
bytes wrong. `recalibrate` had left them alone because they are neither
float lanes nor zero-point registers. They are the fused bias `Add`'s words,
the same ones `binary_op_scale_emit` decoded for a standalone Add (#1869):

- registers `0x1ef0..0x1f20`: the int32 zero-point offset,
  `int((zp_y - zp_x*r_x - zp_z*r_z) * 2**(15-k))`, with `r = s/s_y` rounded
  to float32;
- register `0x1ea0`: `15 - k`;
- one `npu_params` word: `round(r_x * 2**(15-k))` and
  `round(r_z * 2**(15-k))` as two uint16s.

`k` is the smallest shift that brings both ratios below 1. The standalone
Add builds all had `k = 0`. The Gemm has `k = 1`, because its MatMul output
scale is a hair above the Add's output scale, so the offset is in Q14 there.
The formula fits both Gemm builds and #1870's `mm_add` template exactly.

`mm_add` has the same offset lanes (`-119879`). #1870 checked that template
only by identity and round trip, which cannot catch lanes that are never
touched, so its "Gemm covered offline" was wrong until this fix. The held-out
pair is what caught it. `locate` now finds these words (`zpoff`, `qshift`
and `q15` roles), so the Gemm and every Conv chain with a bias recalibrate
them too.

## A 3x3 Conv chain also requantizes its weight

The same offset lanes appear a second time in a 3x3 Conv's legalized chain,
behind a shift of `0x8f` or `0x8e` (bit 7 set). That group requantizes the
weight taps, which are asymmetric `uint8`, into their `Concat`, which is
symmetric: `rqoff = int((zp_y - zp_x*r) * 2**(15-k))` with one ratio
`r = s_x/s_y`, and `rqshift = 0x80 | (15 - k)`. Here `k` stops at
`r <= 1`, not `r < 1`: a slice at exactly its Concat's scale keeps `k = 0`
(stage4 conv1's held-out build). `locate` pairs each offset group with the
shift write before it, and picks Add triples or requantize pairs by bit 7.

Two data choices keep the formulas apart in a Conv template:

- **Per-tap weight ranges.** Each 3x3 tap is scaled differently, so no two
  tap slices tie on scale or zero point.
- **A bias comparable to the output range**, so the MatMul output, the
  biased sum and the bias get distinct zero points. With a small bias they
  all came out at 128, and `recalibrate` refused the tie.

## What a template is

- **One build per distinct step chain, not per shape.** Each template is the
  node's real chain cut out of `step.onnx` with `onnx.utils.Extractor`. That
  matters because Pulsar2 fuses the chain into one program:
  - **dX** (a Conv's input gradient): `Gather -> Mul(mask) -> [Reshape] ->
    MatMul`, with the rearranged kernel as the other operand. The same form
    as `gather_aggregate_real`, whose build serves the three 512->512 nodes.
  - **dW** (weight gradient): `[Reshape] -> Transpose -> MatMul`.
  - **fc**: `MatMul` (dX), `Squeeze -> Transpose -> MatMul` (dW), and the
    forward Gemm as `legalize.gemm_to_matmul` rewrites it (`Squeeze`,
    `Transpose(W)`, `MatMul`, `Add(bias)`). The Gemm itself fails Pulsar2's
    quantizer with a live weight.
  - **Conv**: the single live-weight Conv after
    `legalize.act_weight_conv_to_matmul` (Pad, Transpose, per-tap Slice,
    Concat, one MatMul, bias Add), because Pulsar2's own live-weight Conv
    path fails.
- **A node is served only if its chain is identical to the template's.**
  `manifest.json` lists a node only when its op sequence, attributes, input
  shapes and every constant (Gather indices, masks, reshape shapes, slice
  bounds) hash the same as the template's. Every same-shape node in the
  step passes this check.
- **Tensor names map positionally.** `manifest.json` stores each chain's
  non-constant tensors in structural order: first use, node by node.
  `step_template(node)` pairs them up, and `step_node_scales` re-keys a
  node's scales onto the template's names for `recalibrate`. Constants keep
  the template's own scale. Any other tensor without a scale raises.

## Calibrating a template

- **Two builds per chain.** The template, and a held-out build at a
  different calibration. The test recalibrates each onto the other and
  compares records and `npu_params`.
- **The two operands get clearly different ranges**, so no two scale
  formulas tie (#1870).
- **The zero-point class has to match the step's.** `recalibrate` refuses a
  zero point that moves between zero and nonzero. Two things decide it:
  - *Nonnegative inputs.* A tensor that is a Relu output, or only pooled,
    gathered or reshaped from one, is calibrated in `[0, hi]`. In the step
    those are the dW activation operands, the forward Conv inputs (except the
    stem's image) and the pooled features. (MatMul operands themselves are
    quantized symmetrically, so their zero point is 0 either way. The
    chain-start tensors of a Gather are not.)
  - *Signed outputs.* The first Gemm build used a signed range that was not
    zero-mean. Over a 512-long contraction against nonnegative features,
    every output then had one sign: the output zero point was 0 in one
    build and 255 in the other, and the pair did not recalibrate. dW, Conv
    and the Gemm are now built with zero-mean signed ranges, so their
    outputs are signed, as in the step.

## Using it

```
entry = matmul_record_emit.step_template("MatMul_367")
tq = matmul_record_emit.load_scales(entry["quant"])
new = matmul_record_emit.step_node_scales(entry, tq, step_scales)
model, report = matmul_record_emit.recalibrate(
    matmul_record_emit.load_model(entry["axmodel"]), tq, new)
```

`tinygrad_ax_backend.plan_node` reports a MatMul, the Gemm or a live-weight
Conv as `conditional` when the manifest lists it. The condition is the zero-
point class: the step's calibration is not in the graph.

## Reproduction

The build scripts live in `~/npu-scratch/t_step_templates/` and are not
committed, because they read `step.onnx`: `make.py` (chains, datasets,
configs), `run.sh` (one build at a time), `validate.py`,
`gen_manifest.py` and `export.py`.
