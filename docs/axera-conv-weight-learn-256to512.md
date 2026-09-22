# The widest Conv shapes in the ResNet18 step: weight codes hold, the scaffold does not

The final, widest-channel pair in this project's Conv weight-learn series
(`docs/axera-conv-weight-learn-{stem,downsample,wide}.md`, and the parallel
`docs/axera-conv-weight-learn-128-and-widegap.md`): ResNet18's stage-3-to-
stage-4 downsample, `Cout=512`, the largest channel count of any real Conv in
the training step.

Real shapes (from `/home/takecheeze/npu-scratch/t6-r18fold/step.onnx`,
confirmed via `onnx.shape_inference`, batch=16):

- **1x1 downsample**: `x[16,256,14,14]`, `w[512,256,1,1]`, stride 2, pad 0, bias.
- **3x3 strided**: `x[16,256,14,14]`, `w[512,256,3,3]`, stride 2, pad 1, bias.

## Weight-code region: byte-exact for both, at k=65

`emitter.learn()` over 65 real Pulsar2 7.0-lite builds per shape (i.i.d.
Gaussian weights and bias, MinMax calibration, `[16,256,14,14]` uniform
+/-0.9 calibration input) converges to **zero collisions** for both shapes,
mapping exactly `Cout*Cin*K*K*8` code bits (1,048,576 for 1x1, 9,437,184 for
3x3). This is the widest channel count this project's Conv-learn series has
validated -- matching, not breaking, the pattern every other real shape
(64/64, 128/128, 64->128, 256/256, 512/512) has already shown: the bit-
permutation technique does not care about channel count, kernel size, or
stride, for this half of the table.

At k=40 the 3x3 shape still had 39 collisions; both shapes needed the full
k=65 to reach zero, consistent with `docs/axera-conv-weight-learn-128-and-
widegap.md`'s finding that `k` needs headroom past the ~47-48 the original
256/256 and 512/512 campaigns used.

## Scaffold region: does not close for either shape, in two different ways

A second-pass `emitter.learn()` (`requant_block_biased`'s own bytes as the
"code", the first pass's ambiguous table bytes as the "table") was applied to
both shapes at k=65, the same technique that resolved 79-80% of the 128/128
and refreshed-256/256 scaffolds.

### 3x3: partial resolution, and a real device-confirmed failure

- **Scaffold layout**: 1024 scattered runs of mostly 3-4 bytes, 7,168
  ambiguous bytes of 1,188,080 total (0.60%) -- the same "wide conv splits
  into slices" signature the 256/256 and 128/128 shapes already showed, not
  the 64/64 shape's single contiguous block.
- **Second-pass result**: 73.4% resolved (40,790 of 57,344 scaffold bits),
  286 collisions remaining -- in the same range as 128/128 (80.4%) and fresh
  256/256 (79.9%) at a comparable k, but a bit lower and not yet converged.
- **Held-out emission**: weight-code region confirmed byte-exact against a
  native rebuild (0 of 1,188,080 `npu_params` bytes differ outside the
  scaffold). All 1,720 differing bytes are inside the still-unresolved 26.6%
  of the scaffold.
- **Device check (AX8850, `axcl-vm`, serialized under the shared lock,
  control run before and after, both clean)**: the emitted holdout is
  **substantially wrong**, not the "numerically harmless" outcome
  `docs/axera-conv-weight-learn-128-and-widegap.md` found for 128/128's
  comparable residual:

  | build | max err vs numpy | mean err vs numpy |
  | --- | --- | --- |
  | control (reference, own weights) | 0.072 | 0.014 |
  | native holdout | 0.076 | 0.015 |
  | **emitted holdout** | **11.79** | **0.220** |

  The error is not a uniform offset the way 128/128's was (which corrected
  to noise-floor with one additive `y_scale` term once its unpatched zero-
  point was identified). Per-channel breakdown of 512 output channels: most
  channels carry a small, broadly consistent ~0.03-0.07 mean diff (plausibly
  the same kind of small unpatched-scalar effect 128/128 had), but **17
  channels differ by up to 11.8** -- two orders of magnitude larger, and 3.3%
  of all output elements exceed 1.0 absolute error. Those 17 channel indices
  (46, 59, 69, 110, 148, 157, 166, 170, 205, 279, 354, 383, 398, 422, 447,
  468, 499) show no periodicity checked (not `mod 32`, not `mod 128`, not
  evenly spaced) -- consistent with specific scaffold bytes for those
  channels' bias/scale landing inside the still-unresolved 26.6%, at values
  where that particular channel's bias or scale term happens to matter a
  lot. **Conclusion: this shape's scaffold gap is load-bearing, unlike
  128/128's.**

### 1x1: does not even benefit from more data, and the emitted holdout faults the device

- **Scaffold layout**: 4 clean contiguous runs of exactly 1,024 bytes each
  (4,096 bytes total -- exactly `2*4*Cout` for `Cout=512`, i.e. the same
  total byte budget a single contiguous bias+scale block would need, just
  split into 4 equal, evenly-spaced pieces). Hand-checked three plausible
  channel-group orderings (contiguous 128-channel groups with bias-then-
  scale layout, per-channel interleaved bias/scale within a 128-channel
  group, and a stride-4 channel grouping) against one reference build's raw
  bytes directly -- **none matched**, so the actual channel ordering inside
  each 1,024-byte run was not found by inspection.
- **Second-pass learn does not improve with k, unlike every other shape
  checked in this whole series**:

  | k | resolved | collisions |
  | --- | --- | --- |
  | 40 | 71.0% | 1,136 |
  | 65 | 70.2% | 1,119 |

  Every other shape's second pass (this file's own 3x3, and 128/128,
  256/256 in `docs/axera-conv-weight-learn-128-and-widegap.md`) showed
  collisions dropping and resolution rising with more builds -- the
  classic "needs more samples" signature the weight-code region itself
  also showed before converging. 1x1's numbers went the *wrong* direction
  (resolution down half a point, collisions essentially flat) between k=40
  and k=65. That is a different failure mode: not under-sampling, but a
  sign the "scaffold is literally a bit-permutation of
  `requant_block_biased`'s byte layout, in *some* order" hypothesis itself
  may not hold cleanly at this shape -- the manually-checked orderings above
  already failed to find that order directly, and more data does not appear
  to be finding it empirically either.
- **Held-out emission and device check**: weight-code region again
  byte-exact (0 of 151,792 `npu_params` bytes differ outside the scaffold;
  960 differing bytes are all inside the unresolved 29.8%). Run on the
  AX8850 under the same lock and control/health discipline as the 3x3 case
  above: **`axcl_run_model` returned `0x8030070C`** -- a hardware fault, not
  a wrong-output run. `scripts/axera/mcode.py`'s `check()` found no
  structural violation in the emitted mcode, so the fault most likely comes
  from the *values* left in the unresolved ~30% of scaffold bytes (garbage
  relative to the new scale/zero-point context after the mcode's own scale
  literals were patched to the holdout's quantisation), not from a malformed
  instruction stream. A control run of the unmodified reference build, and a
  standalone health check, were both confirmed clean immediately before and
  after this fault -- the device itself was not wedged, only this one
  emitted model faulted it, following this project's established
  device-safety discipline
  (`axera-device-patch-experiments-need-health-checks` in the project's own
  session memory).

## What this means for the series

- **11 of 11 real ResNet18 Conv shapes now have a byte-exact weight-code
  region**, across every fork in this Conv-learn series. That half of the
  technique is fully general, confirmed at the widest channel count
  attempted (512).
- **The scaffold half is not uniformly a "needs more data" problem.** 3x3
  here matches that pattern (comparable resolution to 128/128 and 256/256 at
  comparable k) but is device-confirmed *wrong*, not harmless, at its
  current resolution -- a materially different outcome from 128/128's case
  despite a similar-looking resolved percentage. 1x1 here does not even show
  the "more data helps" signature at all, and its emitted artifact is
  actively unsafe to run, not just inaccurate.
- **A resolved-percentage number alone does not predict whether a partial
  scaffold emitter is safe.** Two shapes at ~70-73% resolved gave two
  different, both-bad outcomes (visibly wrong output; a hardware fault) at
  this channel width, while 128/128's ~80% gave a harmless, fully-diagnosed
  one-instruction fix. Anyone extending this series should device-check
  every new shape's holdout individually rather than assuming a resolution
  percentage in the 70-80% range from one shape transfers to another.
- **Not attempted**: pushing either shape's second pass past k=65 (128/128
  and 256/256 needed exactly this kind of headroom); locating the 1x1
  scaffold's real channel ordering (the manually-checked orderings all
  failed, and the automatic second pass does not appear to be finding it
  either -- a real, currently-blocking open question); locating any specific
  byte in either shape's scaffold (no zero-point or scale byte was
  identified individually, unlike the 1x1 downsample at 64->128 or the
  128/128 shape); the 7x7 stem (still separately unattempted, per
  `docs/axera-conv-weight-learn-stem.md`).

## Reproduction

`scripts/axera/conv_learn_256to512.py` has `learn_weight_codes`,
`learn_scaffold`, and `emit_holdout` -- the same first-pass/second-pass/emit
pipeline used here, parameterised over a work root of fresh same-shape
builds. `load_maps("c1x1" | "c3x3")` returns this file's own committed maps.

Fixtures under `scripts/axera/fixtures/conv_learn_256to512/`: a reference and
native holdout build (gzipped `.axmodel`s) for both shapes, the holdout
weight/bias arrays, the emitted (scaled, partially-scaffolded) holdout
`.axmodel`s for both shapes -- the exact artifacts the device check above
ran -- and both shapes' first- and second-pass learned maps.
`tests/test_axera_conv_learn_256to512.py` reproduces the weight-code-exact
result and the scaffold resolution percentages with no Docker or device
needed. The raw 65+2-build campaigns themselves (scratch under
`/home/takecheeze/npu-scratch/t_conv_learn_256to512`, not committed -- 5.6 GB
across 132 build directories) are not reproducible from the repo alone; the
committed fixtures are what the doc's own numbers were computed from.
