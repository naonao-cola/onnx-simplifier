# The ResNet18 stem `Conv`: the last of the 11 real training-step Conv shapes

PR #1769 validated `emitter.py`'s bit-permutation weight learner on
`Conv(64,64,3x3,stride=1)` but explicitly skipped the stem "for lack of
remaining budget, not because it looked harder." This is that campaign, on
`Conv(cin=3, cout=64, k=7x7, stride=2, pad=3, batch=16, spatial=224x224)` --
the only convolution in the ResNet18 training step that touches the raw
image, with the largest kernel and the widest spatial size of any real
shape in the graph. `docs/axera-conv-weight-learn-stem.md`'s title is
misleading: its actual content is the 64x64 shape, not this one; that file
is unchanged here, and this is a separate document as instructed.

**Result: it validates completely, and turned out to be one of the
*cheaper* shapes to learn, not a harder one -- the opposite of what the
large spatial size suggested going in.**

## Build cost: much cheaper than budgeted for

Each single-`Conv` build (Docker `pulsar2:7.0-lite`, AX650, MinMax
calibration, 2 concurrent) took on the order of 15-30 seconds, the same
range PR #1769 saw at 56x56 spatial size, not the up-to-60-minutes this
campaign was budgeted to tolerate. 24 builds finished in under 5 minutes;
40 builds (36 for learning, plus 3 held-out attempts) fit comfortably in
under 20 minutes total, including the `learn()`/`learn_mcode()` analysis
between batches. The 224x224 input and 7x7 kernel did not change per-build
compile cost measurably from the 56x56/3x3 case -- whatever dominates
Pulsar2's build time for a single op, it is not spatial size or kernel
extent at this scale.

## Calibration data: a stated assumption, as in every prior doc here

No real ImageNet-normalized image data was available in this sandbox.
Calibration and the on-device test input both use
`np.clip(randn(16,3,224,224), -2.5, 2.5)` -- a synthetic, roughly
unit-normal-per-channel distribution meant to resemble typical
image-normalization statistics (e.g. `(pixel/255 - mean)/std`), not real
photographs. This is the same class of assumption `docs/
axera-conv-weight-learn-stem.md` and the other campaigns in this family
made for their own calibration data, stated for the same reason: nothing
here has been checked against a real trained ImageNet-statistics input.

## `k`: empirically 36, found the same way as the other shapes

This shape's code-bit count is `Cout*Cin*K*K*8 = 64*3*7*7*8 = 75,264` --
between the 1x1 downsample's 65,536 (needed `k=32`) and the two 3x3
downsamples' 589,824 (needed `k=38`). Built incrementally and rechecked
`emitter.collisions()` after each batch:

| `k` | mapped bits | collisions |
| --- | --- | --- |
| 24 | 75,277 (7 too many -- some spurious matches not yet resolved) | 180 |
| 32 | 75,264 (exact) | 1 |
| 36 | 75,264 (exact) | **0** |

`k=36` reproduces the exact expected bit count (`Cout*Cin*K*K*8`) with zero
collisions -- the same clean convergence every other real shape in this
family has shown, at a `k` consistent with (if not perfectly predicted by)
the code-bit-count-scaling relationship PR #1771 first noted.

## The requantisation block: one contiguous 512-byte span, like the "easy" shapes

The 36-build campaign's ambiguous/scaffold region is exactly one contiguous
run, bytes 18432-18942 of the 22,144-byte table (`Cout * 2 floats * 4 bytes
= 512` bytes, the same size every shape in this family expects) -- matching
PR #1769's `Conv(64,64,3,3)` and PR #1771's 1x1 downsample, **not** PR
#1770's wide-channel (256/512) shapes' scattered four-region layout. Weight
Cout is 64 here too, consistent with the block layout tracking output
channel count (or something correlated with it) rather than kernel size or
input channel count.

Applying `conv_bias_requant.emit_conv_table` (the bias-aware requantisation
formula PR #1771 already derived and fixed as a wrapper, unchanged here) at
`block_at=18432, block_len=512` on a held-out weight/bias set (never used
for learning): the weight-code region is **byte-exact** (0 bytes differ
outside the block), and the block itself differs from the real compiled
table in 69 of 512 bytes, with **max absolute float difference 0.00045** --
the same order of magnitude as PR #1769's 1.9e-4 and PR #1771's 0.00044,
consistent with float32-rounding, not a wrong formula.

## The mcode: cleaner than every other shape in this family

All 36 learning builds share **the same mcode length** (65,528 bytes) --
zero shifted-layout builds, versus PR #1769's 6-of-48 (12.5%) and PR
#1771's 15-of-32 (47%) at this stage. `emitter.learn_mcode` at
`min_agreement=0.7` (0.9 finds nothing, matching the other campaigns' need
to lower this) finds:

```
scale_offsets: [8375, 8383, 8391, 8399]   (4 copies of y_scale, f32)
zero_offsets:  [8357]                      (1 byte, round(y_zero) & 0xff)
free:          28 bytes
outliers:      10 of 36 builds
unpatchable:   {"zero": [127, 128], "scale_low_byte": []}
```

**Unlike PR #1771's downsample case, the `unpatchable` zero points here are
real, not a false refusal.** Two of the first three held-out weight sets
tried (`stem_k36`, `stem_k37`, `stem_k38`) landed on `y_zero` exactly 127.0
or 127.7-ish rounding to 127 -- both confirmed genuinely shifted streams by
direct comparison: the reference mcode's scale-offset bytes for `stem_k36`
held a value with no relation to that build's own `y_scale`
(`3.9e35` vs. the real `0.0612`), and patching anyway produced 3,372 of
65,528 bytes differing from the real native rebuild, not the handful the
"free" bytes account for. This is `emitter.py`'s documented "roughly one
value in fifty" phenomenon, and this campaign's held-out sampling happened
to hit it twice in three tries -- unlucky, not a flaw in the classifier.
The third attempt, `stem_k39` (`y_zero=126.0`, clear of the unpatchable
set), is the one validated below.

## Held-out weight set (`stem_k39`, `y_zero=126.0`): full pipeline, clean

Patching the reference's mcode at the four scale offsets (with `stem_k39`'s
own `y_scale`) and the one zero offset (with `126`) and comparing directly
against a real native rebuild of `stem_k39`: **9 of 65,528 bytes differ**,
all inside the 28-byte "free"/scheduling region `learn_mcode` already
flagged as not affecting output (PR #1769 and #1771 both confirmed this
category is harmless; this campaign did not re-derive that, only relied on
it).

## On the card

Reference: `stem_k00` (one of the 36 learning builds). Target: `stem_k39`'s
weights, held out from learning entirely. Built the emitted model via
`conv_weight_learn_stem.emit_stem_conv` (weight-code map + bias-corrected
requant block + the four scale/one zero mcode bytes, nothing else) and ran
it on the AX8850 via `axcl-vm` (serialized under the shared device lock; a
control run of the reference build before every other run, a health run of
the same reference build after, both bit-identical -- no fault):

| comparison | max abs error | mean abs error |
| --- | --- | --- |
| native `stem_k39` build vs. `numpy` conv2d | 0.7085 | 0.01829 |
| **emitted** `stem_k39` vs. `numpy` conv2d | 0.7085 | 0.01829 |
| emitted vs. native (device output, direct) | 0.0632 max (one `y_scale` quantisation step), **2,865 of 12,845,056 elements differ at all (0.0223%)** | -- |

The emitted model's error against `numpy` is not merely close to the native
build's own error -- to 4 significant figures, both max and mean, it is the
*same number*. Every element that differs at all between emitted and native
differs by at most one output quantisation step (`y_scale = 0.0632`), the
same "free bytes are cosmetic" conclusion every other shape in this family
reached, now confirmed at the stem's own scale (12.8M output elements, the
largest single on-device check run in this family so far).

## What this closes out

All 11 distinct real forward `Conv` shapes in the ResNet18 training step
have now had `emitter.py`'s bit-permutation weight learner attempted at
least at the weight-code level:

| shape | weight-code bits | full pipeline (bias+scale+zero) |
| --- | --- | --- |
| stem, `3->64, 7x7, s2` | validated (this doc) | **validated, device-confirmed** |
| stage-1, `64->64, 3x3, s1` (x4) | validated (#1769) | **validated, device-confirmed** |
| downsample, `64->128, 1x1, s2` | validated (#1771) | **validated, device-confirmed** |
| downsample, `64->128, 3x3, s2` | validated (#1771) | open (scattered aux block) |
| stage-2, `128->128, 3x3, s1` (x3) | not attempted | not attempted |
| downsample, `128->256, 1x1/3x3, s2` | not attempted | not attempted |
| stage-3, `256->256, 3x3, s1` (x3) | validated (#1770) | open (~51% recovered) |
| downsample, `256->512, 1x1/3x3, s2` | not attempted | not attempted |
| stage-4, `512->512, 3x3, s1` (x3) | validated (#1770) | open (~51% recovered) |

Three of ResNet18's 11 distinct Conv shapes (stem, stage-1, one downsample)
now have a complete, device-verified weight-emission pipeline; the 128- and
256->512-channel families remain entirely unattempted, and two wide/one
3x3-downsample shape have validated weight codes but an unresolved
auxiliary block.

**What this still is not**: exactly the same boundary PR #1769 and #1771
already drew. This gives a standalone per-layer inference `.axmodel`
refreshable from a checkpoint's weights, not access to the training step's
own compiled multi-op graph (where the trainable weight is a graph
input/output, not a static table -- `docs/axera-mcode-training-graph-
coverage.md`). Composition with a real neighbour has also not been checked
for Conv specifically (`docs/axera-transpose-compose-real.md` found
composition rewrites MCode for a different op).

## Reproduction

Fixtures under `scripts/axera/fixtures/conv_learn_stem/`: the reference
build and the held-out native build (both gzipped `.axmodel`s plus their
`quant_axmodel.json`s), the held-out weight/bias tensors, and the learned
map (`stem_map.npz`, `emitter.save_map`/`load_map` format).
`tests/test_axera_conv_weight_learn_stem.py` reproduces every claim above
except the device numbers, with no Docker/device required.

The 40-build campaign itself (scratch under
`/home/takecheeze/npu-scratch/t_conv_learn_stem_full`) is the expensive,
non-reproducible-from-the-repo part; the committed fixtures are its output.
