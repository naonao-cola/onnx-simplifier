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
| dW | 7 of 11 | 8 of 20 |
| Conv | 6 of 11 | 8 of 20 |

Coverage (`tinygrad_ax_backend.coverage_report` on `step.onnx`, master's
backend): MatMul 41 refused -> 24 conditional, 17 refused; Gemm 1 -> 1
conditional; Conv 20 refused -> 8 conditional, 12 refused; totals
82 / 324 / 698 -> 82 / 357 / 665 (covered / conditional / refused).

The remaining builds are in progress, largest last.

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
