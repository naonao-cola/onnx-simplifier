# A 3x3 Conv's Concat header above 1, and its two programs

The legalized 3x3 Conv (`act_weight_conv_to_matmul`) slices its input into
taps and concatenates them (`<conv>_xcat`), and does the same to the weight
(`<conv>_wcat`). `npu_params` stores both taps' ratios into their Concat as
one word, the *Concat header* (`cat15` in `matmul_record_emit`). Until now
every decoded header had both ratios at most 1, stored as `round(r * 2**15)`.

## The header when a ratio is above 1

A Conv whose input is an unfused Relu's output (stage2/3/4 `conv1`) takes
its quantization from the Relu's pre-activation: the range reaches below 0
while the data does not. The taps' own Concat then calibrates to the narrower,
nonnegative data. So the activation ratio is `127.5 / (255 - zp)`, which is
above 1 whenever the shared zero point is above 127.

That half is stored with the requantize's shift `k` (the smallest `k` with
`r * 2**-k <= 1`) as `round(r * 2**(15-k))`:

| build | activation ratio | `k` | stored | `r * 2**(15-k)` |
| --- | ---: | ---: | ---: | ---: |
| stage2 conv1 | 1.0991 | 1 | 18008 | 18008.4 |
| stage2 conv1, held out | 1.0625 | 1 | 17408 | 17408.0 |
| stage4 conv1 | 1.7000 | 1 | 27853 | 27853.1 |
| stage4 conv1, held out | 1.3421 | 1 | 21989 | 21989.3 |

The weight half stays Q15 (0.87931 into 28813). `locate` now accepts a
shifted half and records both `k` in the role. One misaligned match appeared
in the stem, 2 bytes before the real header: a ratio in one half and its
inverse in the other. It is rejected.

## `k = 0` and `k >= 1` are two programs

A sweep of the stage2 conv1 chain moved the Relu's zero point from 70 to
225, giving an activation ratio from 0.71 to 4.25. Eight builds were
recalibrated onto each other with the shift pin lifted:

| ratio | 0.71 | 0.91 | 1.04 | 1.05 | 1.21 | 1.99 | 2.32 | 4.25 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `k` | 0 | 0 | 1 | 1 | 1 | 1 | 2 | 3 |
| segment 2 records | 4308 | 4308 | 4312 | 4312 | 4312 | 4312 | 4312 | 4312 |

- **The 1.04..2.32 builds are one program.** They recalibrate onto each
  other record for record, including across `k = 1 -> 2`.
- **`k = 0` has fewer records.** The activation taps' requantize writes its
  shift to `0x1ea0`. At `k = 0` that value (`0x8f`) is already in the
  register and the write is elided, the same elision as the zero-point
  registers. At `k >= 1` it writes `0x8e` and the rest of the segment moves.
- **A second toggle is allocator noise, not calibration.** The last segment
  differs by one record: a buffer address in `0x0240` of 0 or `0x80`. In the
  sweep it showed up at the extremes (0.71 and 4.25), but equal-input
  rebuilds flip it as well (see below). An emitted model keeps its
  template's allocation, which is self-consistent.

So `recalibrate` refuses a header half that crosses 1 in either direction
("a ratio on the other side of 1 is a different program"). The step's 3x3
convs fall into both classes by their input zero point: 7 of the Relu-fed
ones are above 127, and 4 of the 3x3 stride-1 convs are at or below it. Each
chain shape therefore gets one template per class it needs, and the
manifest assigns each step node to the template that recalibrates onto it
(`concat_shift_class`).

## Equal-input rebuilds pick one of two program variants

The stage3 conv1, stage4 conv0 and stage4 conv2 pairs had failed the exact
pair check. Rebuilding each build from identical inputs gives a
*different* program about half the time. It is either a different record
count, or, for stage4 conv2, the same tile table rotated by two entries
(the #1836 table-order flip). Each held-out build has an exact partner
among its template's rebuilds:

| chain | exact pair |
| --- | --- |
| stage4 conv0 | template x held-out rebuild, and template rebuild x held-out |
| stage3 conv1 | template rebuild x held-out, and x held-out rebuild |
| stage4 conv2 | template rebuild x held-out, and x held-out rebuild |

So these were not calibration effects. The committed pair is one exact
pairing; the other variant is an equally valid program.

Reproduce: `tests/test_axera_matmul_record_emit.py`
(`test_concat_header_*`) and `tests/test_axera_matmul_step_templates.py`
(every committed pair, both directions). The sweep and rebuilds live in
`~/npu-scratch/t_step_convA/` (`sweep.py`, `sweep_check.py`, `mk_classes.py`).

## Result on the step

With the shifted header, the two shift classes and the rebuild pairings, all
20 Conv nodes recalibrate onto the predicted step calibration:

| chain template | class | step nodes |
| --- | --- | --- |
| stage1 conv0 | k0 | stage1 conv0, conv2 |
| stage1 conv1 | k1 | stage1 conv1, conv3 |
| stage2 conv1 | k1 | stage2 conv1, conv3 |
| stage2 conv4 | k0 | stage2 conv4 |
| stage3 conv1 | k1 | stage3 conv1, conv4 |
| stage3 conv3 | k0 | stage3 conv3 |
| stage4 conv1 | k1 | stage4 conv1, conv4 |
| stage4 conv3 | k0 | stage4 conv3 |
| stage4 conv0, stage4 conv2 | - | one node each |

Totals at the predicted calibration go from 481 covered / 623 refused to
**496 / 608**. Conv goes from 5 to 20 of 20.

On the device (`emitter_device_check.py`, native control run before and
health run after), emitted at the step's predicted calibration:

| case | class | max error (LSB) |
| --- | --- | ---: |
| stage2 conv1 | k1 | 1 |
| stage2 conv4 | k0 | 1 |
| stage3 conv1 | k1 | 2, on 1 of 802,816 elements (the native control is also 2) |
| stage4 conv3 | k0 | 1 |

### What picks the program variant is not decoded

For stage4 conv3 the two variants differ inside segment 0, the weight-tap
part (4,719,680 vs 4,390,560 decoded bytes). Across four builds, no single
calibration value separates them: not the weight zero point (93/99 vs
73/103), the MatMul or output zero points, the taps' ratio into the Concat,
or scale ties (27 tied taps in every build). Rebuilding from identical inputs
keeps the variant for this chain but flips it for others (stage3 conv1,
stage4 conv0), so part of the choice is not in the inputs at all. This does
not affect correctness: each variant is a complete program, and an emitted
model keeps its template's variant. Each template's held-out build is simply
chosen from the builds that came out as the same variant.

## Open: dX MatMul 121 on the device

The device check's float reference for `dX_MatMul_121` was still the chain
before #1942 extended it back to the trainable weight. The native template
now takes `resnetv15_stage4_conv0_weight`, so the control run could not be
fed ("Feed stimulus failed"). With the reference refreshed to the chain the
template was built from, the native control is within 1 LSB. But the model
emitted at the step's predicted calibration is up to 3 LSB off, with 1.1% of
elements above 1 LSB (health run clean). The extended dX template passes the
record-exact pair check, so this is not yet explained.
