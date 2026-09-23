# Can a compiled training step be recalibrated by patching instead of recompiling?

**Short answer: not in place, and the reason is narrow and specific.** A
calibration change touches only three things in a compiled step: the TENG
compute queue, the Conv/MatMul requant blocks in `npu_params`, and (on a real
CNN) activation zero-point bytes in the two conv queues. The CV and DMA queues
never change. Everything outside TENG is fixed-width and formula-level. TENG
is the blocker: new values change the byte length of its compressed register
writes, and some ops carry integer tables whose entry count depends on the
input range. So there is no recalibration patcher here; there is a classifier
(`scripts/axera/step_recalib_diff.py`) that measures all of this and says why.

Why it matters: multi-phase on-device training swaps to a recalibrated model
when gradients die, and each swap is a full recompile, ~15 s on a small step
and 70-450 s+ on resnet18 (`docs/axera-on-device-training-handoff.md`,
"Multi-phase calibration swap" and "Reducing the per-phase recompile cost").
Composition rewrites MCode, so per-op templates can't be stitched into a step
(`docs/axera-transpose-compose-real.md`, `docs/axera-conv-compose-real.md`);
patching the step's own calibration-dependent values was the remaining route.

Everything here comes from our own Pulsar2 7.0-lite builds. No device run was
made: nothing produced a patched model to run.

## Method

Build the same step graph several times, changing only the calibration data,
then compare the builds engine queue by engine queue (`mcode.segments`) at the
record level (`mcode.decode`). Records are aligned by shape (kind plus register
address), so a changed value shows up as a value change and an added, removed
or re-encoded record shows up as a structural change. Changed 4-byte literals
are checked against each build's own recorded scales (`quant_axmodel.json`):
a literal counts as explained when both its old and new float values are a
scale, a reciprocal, a ratio, or a product of two scales.

Graphs, both real pipeline outputs already on disk:

- **Toy step**, `toy-npu2/step.onnx`: 118 nodes, a 2-layer MLP student with a
  distillation loss (Softmax, Log) and Adam. It uses its original config,
  which pins Mul/Add/Sub/Div to FP32. The all-int8 config crashes in
  `TileFailException("AxQuantizedAdd, tuple index out of range")` on the
  4-element Adam tensors, the same tiny-tensor crash the handoff reported.
  About 26 s per build.
- **ResNet18 dense-head step**, `t8-r18head2/wd`: the real resnet18 training
  step with the dense head trainable, 237,368 B of MCode and 11.35 MB of
  `npu_params`, the same step `docs/axera-step-attribution.md` profiled. About
  40 s per build. The full conv-trainable resnet18 step (`t6-r18fold`) still
  doesn't compile (the known PPQ calibrator crash), so it couldn't be used.

Calibration variants (the dataset tars scaled; nothing else changed):

| build | change |
| --- | --- |
| `toyf_A1`, `toyf_A2` | original calibration, built twice |
| `toyf_Bx2` | `input` x2 |
| `toyf_Cw001` | `fc1.weight`, `fc2.weight` x0.01 (the handoff's "late training" 100x shift) |
| `toyf_Dmom100` | all eight Adam `__m`/`__v` states x100 |
| `toyf_Eteach2` | `teacher_logits` x2 |
| `r18_A1`, `r18_A2` | original calibration, built twice |
| `r18_Bx2` | `data` (the image) x2 |

## Rebuild noise is always segment 0's tail, and these steps are deterministic

Two builds with identical calibration differ only in the last few dozen bytes
of segment 0: bytes 3124-3140 of the toy step (segment 0 ends at 3160),
25830-25851 of the ResNet18 head step (ends at 25884), and one byte in the
FlatBuffers tail of the toy step. The long-used "noise window 301-325" is the
same thing in a single-op model, whose segment 0 is 64 bytes.
`step_recalib_diff.noise_bytes` uses segment 0's last 64 bytes.

Both steps are otherwise byte-identical across rebuilds, including the
ResNet18 head's full 11.35 MB `npu_params`. That is much more deterministic
than the 1.9% prefix difference the handoff measured on a conv-trainable
resnet18 step. It holds for these two graphs and was not checked on a
conv-trainable one.

## What a calibration change touches

| queue / table | toy `Bx2` | toy `Cw001` | toy `Dmom100` | toy `Eteach2` | r18 head `Bx2` |
| --- | --- | --- | --- | --- | --- |
| seg 0-1, conv queues | unchanged | unchanged | unchanged | unchanged | **value-only**: 15 + 11 zero-point words |
| seg 2, TENG | 17216 -> 17344 B, 26 structural | 17216 -> 17120 B, 19 structural | 17216 B, 5 structural | 17216 -> 17248 B, 9 structural | 72800 -> 72704 B, 68 structural |
| seg 3, CV | unchanged | unchanged | unchanged | unchanged | unchanged |
| seg 4, DMA | unchanged | unchanged | unchanged | unchanged | unchanged |
| `npu_params` | 7 runs, 92 B | 8 runs, 104 B | unchanged | 6 runs, 80 B | ~300 runs of ~256 B |

Every difference falls in one of five classes.

**(a) Float literals that are scale functions: patchable arithmetic.** TENG
writes quantize and dequantize multipliers as float32 register values, as
`docs/axera-teng2-register-decode.md` (#1831) found for a standalone Relu. In a
real step they are spread across the whole queue, in full register-write
records and in short compressed units. Explained fraction of changed literals:
140/181 (`Bx2`), 142/152 (`Cw001`), **27/27** (`Dmom100`), **103/103**
(`Eteach2`), 574/765 (r18). In `Dmom100` they are two 4-lane register groups,
`0x0f50-0x0f80` (#1831's group plus lane 0) and a second group
`0x0fd0-0x1000`. In `Bx2` the 41 unexplained ones are the (d) table entries,
whose 4-byte payloads merely read as tiny floats; r18's 191 unexplained were
not checked one by one.

**(b) `npu_params` requant blocks: formula known.** In the r18 head step the
changed runs recur at a 37,120-byte stride (292 of about 300 gaps), which is
the per-output-channel-tile `(bias, M)` scaffold layout
`docs/axera-conv-scaffold-arithmetic.md` (#1784) and
`docs/axera-conv-256to512-tiled-fix.md` (#1790) decoded for Conv. The toy's 7-8
short runs are its MatMul requant blocks. The per-block B/A ratios are not
constant, as expected from #1784's bias term (`y_zero - x_zero*sum(q)*M +
bias/y_scale`). Recomputing these needs the new scales and zero points, which
the formula already takes.

**(b') Conv-queue zero points: fixed-width fields.** The r18 head's conv
queues change only in words whose high part is fixed and whose low byte moves
(`0x54 -> 0x4c`, `0x8e -> 0x92`, `0x75 -> 0x63`, ...). Every changed low byte is
a recorded activation zero point of its own build, 6 of 6 on each side. Fixed
width, so patchable once each field is mapped to its tensor.

**(c) Value-dependent re-encodings in TENG: the first blocker.** A short
compressed register write carries only as many value bytes as the value needs,
so a new value can change the record's length. From `Dmom100`, same register
`0x0fd0`, old then new:

```
04 d0 0f eb dc 8b    82 fc   02 eb dc 8b    82 fc  ...   (3 value bytes)
05 d0 0f fb 4e 20 3a 81 fc   03 fb 4e 20 3a 81 fc  ...   (4 value bytes)
```

The first write carries the address, and the following lanes drop it. All
five structural changes in `Dmom100` are of this kind, and so are the
variable-width zero-point bytes #1831 found in a standalone Relu. Patching a
value into a slot of a different width shifts everything after it, so the
segment would have to be re-encoded, and its padded size and the tail's
per-segment word counts rewritten.

**(d) Range-dependent integer tables in TENG: the second blocker.** Two regions
of TENG hold runs of small integer pairs (`17 00 00 01`, `17 02 00 02`,
`17 03 00 04`, ...) that change entries and entry count when the input range
changes (`Bx2`: `17 13 00 13`, `17 14 00 15`, ...; `records` 4506 -> 4526).
They change whenever a variant changes a tensor feeding Softmax/Log (`Bx2`,
`Cw001`, `Eteach2` -- teacher logits feed the teacher Softmax) and not in
`Dmom100`, whose change doesn't reach them. Softmax and Log are dispatched
int8 in this step. The likely reading is a piecewise approximation whose
breakpoints are placed in quantized input codes; that is not confirmed, and
no generator for these tables is known.

## Verdict

The step is **not recalibratable by in-place patching**:
`step_recalib_diff.classify` reports `not-patchable` for every calibration
variant, and `equivalent` for both same-calibration rebuild pairs. The whole
obstacle is inside TENG: (c) the value-length-dependent compressed register
writes and (d) the range-dependent tables. Everything else is known formulas
or fixed-width bytes, and two of the five engine queues never change at all.

What would turn this into a working recalibrator:

1. **An encoder for TENG's compressed register writes**, so a new value can be
   written at whatever width it needs and the segment re-encoded, padded and
   re-counted. The `Dmom100` case would then be fully patchable: its 27
   literals are all scale functions, and its only other changes are (c).
2. **The generator for the (d) tables**, or a way to keep them fixed. One
   untested route: pin the calibration range of the tensors that feed table
   ops (the logits), so a phase swap that only rescales weights, gradients
   and optimizer state never moves them.
3. **A tensor-to-field map** for the (b') zero-point bytes and the (b)
   scaffold blocks. The formulas exist; the mapping from each block to its
   layer does not.

## Reproduction

`scripts/axera/step_recalib_diff.py REF NEW [REF_SCALES NEW_SCALES]` prints the
per-queue comparison and the verdict; scales can be a Pulsar2
`quant_axmodel.json` or `{"scales": [...]}`.
`tests/test_axera_step_recalib_diff.py` pins the toy findings against four
committed builds in `scripts/axera/fixtures/step_recalib/` (15 KB each,
gzipped, with their recorded scales). The ResNet18 builds (11.6 MB each) are
not committed; they are in `/home/takecheeze/npu-scratch/t_step_recalib/`.
