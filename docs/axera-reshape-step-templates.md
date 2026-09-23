# AX650 step Reshape templates: one build per shape, calibration retargeted

`docs/axera-reshape-decompressed.md` covered the step's non-fused Reshapes
with polynomials in the channel count, fitted across a record-structure
group. For the step's actual shapes that approach stalls: the groups around
them are too small. This note gives up on shape generality. It builds each
distinct step shape once and shows that everything calibration-dependent in
the program can be rewritten offline.

**Short version.**

- **One template per shape.** Each of the step's distinct non-fused
  Reshapes (`in -> out` shape pair) is built as `Reshape -> Relu`, at two
  symmetric calibration ranges (±0.9 and ±0.5).
- **Scale lanes.** Within a pair, the only records that differ are the
  scale lanes (0x0f50..0x0fc0). Rewriting them with
  `reshape_record_emit.retarget_scale` turns either build into the other.
  That holds record for record, in the tail words, and in blob length, in
  both directions.
- **Zero points.** A nonzero zero point is retargetable too. Asymmetric
  builds (zero point 64 and 113) differ from the symmetric template only in
  the scale lanes and three zero-point records, 0x1b10, 0x1eb0 and 0x1a90.
  Retargeting reproduces them. A zero point of 0 (range `[0, hi]`) compiles
  to a different, shorter program, so `retarget_scale` refuses it.
- **Coverage.** Step coverage (`tinygrad_ax_backend.coverage_report`) goes
  from 18 to **146** conditional Reshapes out of 170: the 18 fused bias
  flattens plus 128 templated nodes. Totals go from 82 covered, 357
  conditional and 665 refused to **82 / 485 / 537**. "Conditional" here
  means covered if the node's calibrated zero point is nonzero.
- **Not done:** nothing ran on the device. The 24 remaining Reshapes are the
  large activation reshapes (27 to 441 MB inputs), and their builds are
  queued.

## Record-level facts

MinMax on these builds gives `s = (hi - lo) / 255` and zero point
`round(-lo / s)`, a uint8 value. Symmetric ranges give 128.

| records | untiled program (e.g. `[1,128,64] -> [128,64,1,1]`) | tiled program (e.g. `[16,64,56,56] -> [16,1,64,3136]`) |
| --- | --- | --- |
| scale lanes | 8 × `1/s`, then 8 × `s` | groups of 8, repeated per tile; each group is `1/s` or `s`, first group `1/s` |
| zero point | one nonzero write each to 0x1b10, 0x1eb0, 0x1a90 | the same registers, rewritten per tile |
| other writes to those registers | value 0 (the Relu's own records) | value 0 |

The existing `scale_lanes` (16 lanes, one segment) was written for the
untiled family fits. `retarget_scale` accepts any number of 8-lane groups,
as long as each group holds `1/s` or `s` of a single calibration.

## Why not polynomials for these shapes

Batch 1 swept the two weight-fold families adjacent to the step's `C=64`
build, `[1,C,C,9] -> [1,1,C,9C]` (unfold) and `[1,C,9C] -> [C,C,3,3]`
(flatten), at C = 40..72 step 4 with fixed calibration. In both families:

- C = 40..60 is one record-structure group, and C = 64, 68 and 72 are
  each a group of one. The step's C = 64 therefore cannot be fitted from
  neighbours.
- Inside the 40..60 group, some fields are rounded rather than polynomial.
  Unfold segment 2, slot 218 takes 22, 24, 26, 29, 31, 33 at C = 40..60
  (`round(0.55·C)`). Flatten slot 124 alternates between 7 and a
  C-dependent value. `fit_rules` rightly refuses both.

A direct build per step shape avoids all of this, because the step has only
59 distinct non-fused shapes.

## Using it

```python
import reshape_record_emit as rre
m = rre.emit_step_reshape([16, 64, 56, 56], [16, 1, 64, 3136],
                          scale=s, zero_point=zp, output_path="r.axmodel")
```

`step_template(in, out)` raises `ValueError` for a shape pair outside
`fixtures/reshape_step_templates/manifest.json`. `retarget_scale` raises for
a zero point of 0, or when the lane or zero-point records don't have the
shape described above.

Tests: `tests/test_axera_reshape_record_emit.py`. Every template retargets
onto its second build and back, and the zero-point probes retarget from and
to their templates. All checks use committed fixtures and need no device.

## Reproduce the builds

The builds use the `Reshape -> Relu` model and `fixed_range_samples` from
`reshape_calibration_isolation.py`, with `MinMax` calibration, 4 samples,
NPU1, and the `pulsar2:7.0-lite` image. Every build ran through the
shared 3-slot runner, one at a time.
