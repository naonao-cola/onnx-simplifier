# Does Gather index retargeting survive real graph neighbours and real scale?

`docs/axera-compose.md` (PR #1732) found that inside a composed graph,
retargeting a Gather's indices (rewrite the index words in `npu_params`,
leave MCode and scales alone) is *not* equivalent to a rebuild: the emitted
model keeps the reference's calibrated ranges, and a downstream `MatMul`'s
output range depended on which elements the indices selected, so a
retargeted index vector could saturate outside the reference's range. That
was measured only at toy scale: `x[1,1,4,16]`, 8 indices. This checks whether
the same risk, and the same fix (calibrate on the full range), hold at real
ResNet18-stem scale with the real graph neighbours, using the actual index
vector and mask the training step compiles.

## The real neighbours

Loading `/home/takecheeze/npu-scratch/t6-r18fold/step.onnx` and looking at
its stem Gather (`Gather_467`, `[16,1,3,50176] -> [16,1,3,614656]`, the 7x7
stride-2 im2col unroll `docs/axera-stem-gather.md` already characterizes)
confirms the neighbours both handoff docs only described in prose:

- **Before:** `Reshape_466`, `[16,3,224,224] -> [16,1,3,50176]` -- a pure
  dim-merge, `data -> distill__reshape_919`.
- **After:** `Mul_468`, `distill__gather_921 * distill__mask_922 ->
  distill__mul_923`, where the mask is a real constant tensor,
  `[1,1,1,614656]` float32, values in `{0.0, 1.0}` only, about 1.5% zero (the
  stem's padding positions).

That second fact matters: an elementwise multiply by a `{0,1}` mask can only
zero elements out, never scale them up, so `Mul`'s output range is bounded by
`[min(0, gather_min), max(0, gather_max)]` regardless of which elements the
indices selected -- unlike the toy test's `MatMul`, which recombines selected
elements into new values whose range genuinely depends on the selection.
This is the concrete difference the real neighbour introduces, and the
reason the toy-scale finding does not have to transfer unchanged.

## What was built

Real index vector and mask extracted directly from `step.onnx`
(`distill__idx_920`, `distill__mask_922`), Pulsar2 7.0-lite, AX650, MinMax
calibration, 4 samples. Two scales attempted:

- **Full scale, 7-chunk `Gather`+`Concat`+`Mul`** (`docs/axera-stem-gather.md`'s
  own chunking, now with the real `Mul` consumer instead of a bare output):
  compiled past quantization and into native build (`calc input
  dependencies: 929143/929143` after about 22 minutes), then ran for another
  40+ minutes with no further progress or error before this session stopped
  it. **Not completed.** Whether it eventually succeeds, and what its MCode
  looks like, is unknown. A single 87,808-index chunk's own standalone
  `Mul` (no `Concat`) failed outright with `integer 87807 does not fit
  'uint16_t'` in `AxQuantizedMul` -- the elementwise engine indexes a
  tensor's last axis with a 16-bit offset, so any single axis above 65,536
  elements needs the compiler's own further tiling (as `dma_tile_predict.py`
  already found for a large *total* size, but this is a hard per-axis limit,
  not the earlier soft byte budget). This may be why the full 7-chunk build
  is so much slower than either the single-op templates or the mid-scale
  build below: the compiler has to solve a harder tiling problem to keep
  every `Mul` job's addressed axis under 65,536 at 614,656 total elements.
- **Mid scale, single `Gather`+`Mul`, 32,768 indices** (under the 65,536
  per-axis limit, so no extra tiling problem): a real, contiguous slice of
  the stem's own index vector and mask (offset 200,000, chosen to include a
  realistic mix of masked and unmasked positions -- 1.4% zero in this slice
  too), skipping the leading `Reshape` (graph input is already
  `[16,1,3,50176]`, matching what `docs/axera-stem-gather.md`'s own template
  does). Three builds: the reference (calibrated on synthetic
  ImageNet-normalization-range data, uniform `[-2.5, 2.5]`, an assumption --
  no real calibration images were available in this environment), a native
  rebuild with a shuffled index vector (same range), and a reference
  recalibrated on a wider range, uniform `[-6, 6]`. This is 4,096x the toy
  test's element count and is where all the results below come from.

`scripts/axera/fixtures/gather_compose_real/` holds the three mid-scale
compiled models (gzipped); `scripts/axera/gather_compose_real_check.py` and
`tests/test_axera_gather_compose_real_check.py` check the index layout.

## Confirmed: `npu_params` layout survives composition and real scale

The first 32,768 little-endian uint32 words of `npu_params` are exactly the
Gather's index vector, contiguous, at word offset 0 -- the same layout
`memory_emit.py`'s standalone emitters already rely on, now confirmed with a
real `Mul` consumer present and at 4,096x the previously-tested scale.
`patch_indices()` (this module) rewrites only those words, the same
operation `emit_gather_last_axis_axmodel` performs standalone.

## Device results: the toy-scale risk does not reproduce at this scale, for this consumer

AX8850 in `axcl-vm`, one lock-serialized session, health checks between
groups, ground truth is `numpy.take(x, indices, axis=3) * mask` computed
per model. All errors are versus that ground truth for the model's own
indices.

| test | model | input range | max err | mean err | elements &gt;0.1 err |
| --- | --- | --- | --- | --- | --- |
| 1 (control) | reference (own indices) | in calib range `[-2.5,2.5]` | 0.0098 | 0.0048 | 0 |
| 2 (native) | native rebuild, shuffled indices | in calib range | 0.0098 | 0.0048 | 0 |
| 3 (emitted) | reference patched to shuffled indices | in calib range | 0.0098 | 0.0048 | 0 |
| 4 (control) | reference (own indices) | wide `[-5,5]` (exceeds calib) | 2.51 | 0.62 | 744,648 / 1,572,864 (47%) |
| 4b (emitted) | reference patched to shuffled indices | wide `[-5,5]` | 2.51 | 0.62 | 743,259 (47%) |
| 5 (control) | wide-calib reference (own indices) | wide `[-5,5]` | 0.024 | 0.012 | 0 |
| 5b (emitted) | wide-calib reference patched to shuffled indices | wide `[-5,5]` | 0.024 | 0.012 | 0 |

Three findings, all clean at this element count (no borderline cases the way
the toy test had):

1. **Tests 1-3 are numerically identical.** At real scale, retargeting the
   Gather's indices to an unrelated shuffled vector produces exactly the
   same error as a fresh native rebuild with those indices, for in-range
   input. The reference's calibrated ranges did not need to change. This is
   a real, different result from the toy test, and the mechanism explains
   why: `Mul`-by-`{0,1}`-mask cannot expand the value range past the
   Gather's own output range, and the Gather's own output range is bounded
   by the *source* tensor's range for any index choice (it selects, it does
   not combine). Whether index selection itself shifts the calibrated range
   at large N was not tested in isolation (the toy test's `MatMul`-specific
   mechanism is absent here by construction, so this experiment cannot
   separate "large N washes out range sensitivity" from "no aggregating op
   means no range sensitivity to begin with" -- see Not covered).
2. **Tests 4 and 4b are numerically identical too** (both badly wrong, matching
   each other to within the last measured digit). Exceeding the narrow
   reference's calibration range breaks the retargeted model exactly as much
   as it breaks the reference computing its own indices -- confirming this
   is a generic "input exceeds calibration" failure, not something
   index-retargeting adds on top.
3. **Tests 5 and 5b confirm the fix, and that it composes with retargeting.**
   Calibrating on the wider `[-6,6]` range fixes the wide-input case for
   both the reference and the index-retargeted model, to the same small
   residual (0.024 max, consistent with ordinary quantization noise at a
   wider range).

## Consequence for `memory_emit.py`'s existing Gather emitters

None of the 19 templates in `docs/axera-memory-op-generator.md` or the stem
template in `docs/axera-stem-gather.md` need to change on the strength of
this finding: they are all standalone Gathers (their only consumer is the
graph output), so there is no downstream op whose calibrated range could
depend on the indices at all -- the toy test's whole failure mode requires
an aggregating consumer between the Gather and the output. This real-scale
check adds evidence for a narrower, different point: *if* a future emitter
retargets a Gather that feeds a real elementwise consumer (a mask, a bias
add, anything that cannot expand the value range past its inputs') inside a
composed reference, this data suggests -- at 32,768 elements, for this one
mask -- that no special recalibration is needed. That is *not* a license to
skip recalibration for an aggregating consumer (`MatMul`, `Conv`, a
reduction): `docs/axera-compose.md`'s finding there stands, untested at
real scale, and remains the thing to recalibrate for if it is ever
retargeted.

## Not covered

- The full 614,656-element, 7-chunk, real-`Mul` composition did not finish
  compiling in this session (see above). The mid-scale result does not
  prove the full stem composition behaves the same; the compiler's own
  tiling response to the 65,536-per-axis `Mul` limit is unknown at that
  scale.
- The leading `Reshape` was not included (graph input matches
  `docs/axera-stem-gather.md`'s own scope cut). `reshape_dma_emit.py`'s
  findings suggest most 4-D reshapes are *not* metadata-only, so this is a
  real, not merely formal, gap.
- Only one mask (the stem's own, 1.4-1.5% zero) and one consumer op (`Mul`)
  were tested. A mask that is not `{0,1}`, or any other elementwise op whose
  own scale is itself index-dependent in some other way, was not tried.
- The calibration range (`[-2.5,2.5]`) is an assumption about realistic
  normalized-image activation range, not measured from a real image or the
  model's own recorded calibration data (unavailable in this environment).
- Whether large-N *index selection itself* (not the input range) can still
  shift a Gather's own calibrated range enough to matter was not isolated
  from the "no aggregating consumer" explanation -- see finding 1.
