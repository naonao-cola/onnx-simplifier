# The 128/128 Conv shape, and does more data close the 256/256 scaffold gap?

Two follow-ups to `docs/axera-conv-weight-learn-{stem,downsample,wide}.md`,
which validated `emitter.py`'s bit-permutation weight learner at five real
ResNet18 Conv shapes with mixed success on the per-channel scaffold/
requantisation region.

## Part A: `Conv(128,128,3,3,stride=1)`, the untested stage-2 block

This shape sits between the fully-working 64/64 shape
(`docs/axera-conv-weight-learn-stem.md`) and the struggling 256/256 shape
(`docs/axera-conv-weight-learn-wide.md`). 49 real Pulsar2 7.0-lite builds
(48 for learning, 1 held out; `[16,128,28,28]` input, real bias, i.i.d.
Gaussian weights, MinMax calibration, uniform +/-0.9 calibration input --
recipe identical to the prior campaigns).

**Weight-code region: byte-exact, as at every other shape tried.**
`emitter.learn()` mapped all `Cout*Cin*3*3*8 = 1,179,648` code bits with
**zero collisions** at k=48.

**The scaffold is scattered, like 256/256 -- not contiguous, like 64/64.**
6,330 ambiguous bits in **128 runs** of length 3 or 131 bytes, spanning byte
offsets 36,864-148,478 of a 149,120-byte table -- the same run-length
signature `docs/axera-conv-weight-learn-wide.md` found at 256/256 (runs of 3
or 131 bytes). This settles which of the two prior findings 128 channels
follows: **128 already behaves like the wide-channel family, not the narrow
one.** Combined with the 1x1 downsample (also `Cout=128`) staying
contiguous (`docs/axera-conv-weight-learn-downsample.md`), the split looks
governed by kernel size at this channel count, not channel count alone: 3x3
scatters at Cin=Cout>=128, 1x1 does not.

**Second-pass `learn()` on the scaffold: 80.4% resolved**, clearly better
than 256/256's 51% at a comparable build count:

```
scaffold total ambiguous bits: 6330
second-pass mapped:            5090  (80.4%)
second-pass const:             1246
second-pass still-ambiguous:   1240  (19.6%)
second-pass collisions:        20
```

**Held-out weight-code + 80.4%-scaffold table: 200 of 149,120 bytes differ
from native (0.134%)**, all inside the still-unresolved 19.6% of scaffold.

### Device verification: the entire residual is the (separately known, unpatched) zero-point -- not the unresolved scaffold

On the AX8850 (`axcl-vm`, serialized under the shared device lock, control
run first):

| build | max abs err vs. numpy | mean abs err vs. numpy | vs. |
| --- | --- | --- | --- |
| control (reference build, its own weights) | 0.0465 | 0.0090 | numpy conv2d |
| native holdout (unmodified Pulsar2 build) | 0.0454 | 0.0092 | numpy conv2d |
| **emitted holdout** (weight codes + 80.4% scaffold, scale patched via `patch_scales.patch_model`, **zero-point left at the reference's value**) | 0.0770 | 0.0316 | numpy conv2d |

Emitted vs. native directly: mean abs diff 0.0316, essentially uniform
across 99.6% of output elements (not a sparse, scattered pattern the way an
unresolved-bits error would look) -- the signature of a **constant additive
offset**, not noise. The reference build's `y_zero` is 128; the holdout's
own is 127. Correcting the emitted output by exactly `+1 * y_scale_holdout`
(`0.0316`, matching the observed mean error to four decimal places) drops
the emitted-vs-native mean error to **0.00021** -- indistinguishable from
the ~0.045 baseline device noise every build in this project's Conv work
shows.

**This is a precise, proven diagnosis, not a guess**: the entire measurable
gap between the emitted model and a real Pulsar2 build is the single
unpatched zero-point code, confirmed by directly applying and checking the
correction. The unresolved 19.6% of the scaffold region, whatever it
encodes, is **numerically harmless at this precision** -- a materially
different conclusion from 256/256's own device check, where the (then)
unresolved ~49% was confirmed load-bearing (`docs/axera-conv-weight-learn-wide.md`:
0.63 mean error against a 0.015 baseline). Resolving *more* of a wide-Conv
scaffold does not uniformly matter -- how much of it is missing, and what
that missing fraction is worth in output-code units, both matter, and 128/128's
remaining 19.6% happens to be worth less than a quantisation step.

**The zero-point byte itself was not found.** A brute-force per-offset
search (does byte `i` equal `y_zero` across the 49 builds, for `i` in the
full mcode) found only one candidate above 70% agreement, and it did not
actually track `y_zero` for the specific reference/holdout pair used here
(a coincidence in the wider sample, not a real field). Correlating candidate
bytes against `y_zero` directly (the two bytes nearest the four already-
found `y_scale` float32 slots) found no linear relationship either. This
needs the same kind of careful, non-bulk-search investigation
`docs/axera-conv-weight-learn-downsample.md` used for the 1x1 downsample's
zero-point (there, at offset 4010, found by manual cross-referencing, not a
sweep) -- not attempted further here for time. What *is* established is
that the fix, once found, is exactly one additive correction, precisely
quantified above.

## Part B: does more data close 256/256's scaffold gap, or is it a plateau?

`docs/axera-conv-weight-learn-wide.md` resolved only 51% of `Conv(256,256,3,3)`'s
scaffold at k=47-48 and left "closing the remaining 179 collisions with more
builds" as untried. The original campaign's individual weight tensors were
not committed (only the reference/holdout builds and the final map), so this
reruns the campaign fresh rather than extending it: **k=65** new builds
(`Conv(256,256,3,3)`, `[16,256,14,14]`, same recipe), scratch under
`/home/takecheeze/npu-scratch/t_conv256_fresh`.

**Weight-code region: byte-exact, zero collisions at k=65**, matching the
original campaign (mapped bit count exactly `4,718,592 = Cout*Cin*3*3*8`).

**Second-pass scaffold learn: 79.9% resolved, up from 51%:**

```
k = 65
scaffold total ambiguous bits: 25498
second-pass mapped:            20370  (79.9%)
second-pass const:              5134
second-pass still-ambiguous:    5128  (20.1%)
second-pass collisions:          132
```

**Answer: more builds help substantially -- this is not a plateau.** Going
from k~47 to k=65 (1.4x) took the scaffold's resolved fraction from 51% to
80%, landing almost exactly on Part A's 128/128 number (80.4%) at a
comparable k. The remaining 132 collisions and 20.1% unresolved bits look
like the same "needs more samples" shape the weight-code region itself
showed before it converged (`docs/axera-conv-weight-learn-wide.md`'s own
k=8/16/24/32/40/47 collision table dropping from millions to zero) -- not
evidence of a different, non-bit-permutation encoding. Whether it reaches
zero collisions at, say, k=90-100 was not tested (build budget), and no new
device check was run for conv256 specifically (Part A's device evidence,
that a ~20% residual at this same run-length/byte-count signature is
numerically harmless, is the closest available evidence but was not
re-confirmed at this exact shape).

## What this means for coverage

- **128/128 is materially closer to a working emitter than 256/256 or
  512/512 were**, and its one remaining gap (the zero-point byte) is fully
  characterized, just not yet located.
- **The wide-channel scaffold problem is a sample-size problem, not (at
  least not entirely) a structurally different encoding.** Whoever revisits
  256/256 or 512/512 next should budget for k well past 48, not assume the
  bit-permutation technique has hit a hard wall there.
- **Not covered**: locating 128/128's zero-point byte; pushing conv256's
  second pass to zero collisions; any device check of the improved conv256
  map; the 512/512 shape's own scaffold at a larger k; the 7x7 stem; any of
  the four remaining, entirely untested real Conv shapes (128->256 and
  256->512 downsample pairs).

## Reproduction

Fixtures under `scripts/axera/fixtures/conv_learn_128_widegap/`: a reference
and held-out native build (gzipped `.axmodel`s) for both the 128/128 shape
and the fresh 256/256 campaign, the held-out weight/bias pairs, and both
shapes' first- and second-pass learned maps (`conv*_map.npz`,
`conv*_scaffold_map.npz`). `tests/test_axera_conv_learn_128_widegap.py`
reproduces the weight-code-exact result, the scaffold resolution
percentages, and `emit_conv_128`'s small residual, with no Docker or device
needed. The device numbers in Part A and the raw 49/70-build campaigns
themselves (scratch under `/home/takecheeze/npu-scratch/t_conv_learn_128`
and `/home/takecheeze/npu-scratch/t_conv256_fresh`, not committed) are not
reproducible from the repo alone.
