# Does `emitter.py`'s weight-learning generalize to real, wide-channel Conv shapes?

`scripts/axera/emitter.py` learns an AX650N weight table's encoding as a bit
permutation from `k` same-shape reference builds, instead of deriving the
seven hand-found layout rules `README.md` recovers. Its own docstring claims
the method "does not care which of the seven layouts the shape happens to
use." That claim had only ever been checked at toy shapes -- the README's own
48-build campaign used a 32-channel `Conv`; `docs/axera-step-attribution.md`'s
coverage table records the measured scope as "Conv fixtures at 1-8 channels,
8x8." This is the first run at real ResNet18 training-step shapes, the two
widest same-channel 3x3 blocks in `/home/takecheeze/npu-scratch/t6-r18fold/
step.onnx`:

- **`conv256`**: `Conv(x[16,256,14,14], w[256,256,3,3], stride=1, pad=1)`, with bias.
- **`conv512`**: `Conv(x[16,512,7,7], w[512,512,3,3], stride=1, pad=1)`, with bias.

These are wide enough to hit the "convolution wider than 128 input channels
is split into slices" bit-sliced/polyphase layout `README.md` documents at
length -- a materially different on-chip format from the narrow shapes the
method was validated on before. That was the real thing worth checking, not
assumed.

## The weight-code bit permutation: validated, byte-exact, and the same `k`

Random-weight `k`-build campaigns (Pulsar2 7.0-lite, AX650, MinMax
calibration, i.i.d. Gaussian weights/biases, uniform +/-0.9 calibration
input) at both shapes, tracking `emitter.learn()`'s own collision count as
`k` grows:

| shape | k=8 | k=16 | k=24 | k=32 | k=40 | k=47/48 |
| --- | --- | --- | --- | --- | --- | --- |
| `conv256` (4,718,592 code bits) | 4,681,142 | 4,652,941 | 604,700 | 2,693 | 7 | **0** |
| `conv512` (18,874,368 code bits) | -- | -- | 7,546,387 | 41,319 | 169 | **0** |

Both converge to **zero collisions** with all 47-48 available builds, and in
both cases `learn()`'s `mapped` count lands on exactly `Cout*Cin*3*3*8` --
`4,718,592` for `conv256`, `18,874,368` for `conv512` -- the same clean
signature the README's own 32-channel campaign found. This is the headline
result: **the number of reference builds needed did not grow with the
~192x/~768x larger code-bit count.** The toy shape needed roughly the same
`k` (the README's own table shows 48) as these two real, much wider ones.
Naive birthday-paradox math over the raw code-bit count would predict `k`
needs to grow with the log of that count; it did not, in practice, on this
evidence -- whatever governs how many builds are needed, it is not simply the
code-bit count.

A held-out weight set (a 48th/47th build, native seed unseen in the learning
set, natively Pulsar2-compiled as ground truth) confirms this is not just a
collision-count artifact: emitting its table through the learned `conv256`
map and diffing against the native build's own table gives **100% byte match
everywhere outside the scaffold region described below, and zero mismatches
outside it** (`tests/test_axera_conv_learn_wide.py`). The weight-code
permutation itself is solid at this shape.

## The requantisation/scaffold region: real, load-bearing, and only ~51% recovered

Here the "does not care which layout" claim breaks, precisely where the
README's own prose predicts it might: the bytes `learn()` cannot explain from
weight-code bits alone are not one contiguous run the way the toy 32-channel
shape's is (`README.md`: "bytes 4608 to 4864 ... 32 channels x 8 bytes").
At `conv256` they are **512 scattered runs** totaling 3,584 bytes (28,672
bits) -- runs of 3 or 131 bytes, at periods of 4 and 132, spanning byte
offsets 36,864 to 593,918 of a 594,240-byte table. That is also
substantially more than the narrow-shape formula's `2*4*Cout = 2048` bytes
would predict, so whatever is stored here is not simply the requantisation
block padded out -- there is real, additional, wide-channel-specific
scaffolding this project has not previously characterized.

Applying `emitter.learn()` a **second time** -- this time treating
`emitter.requant_block()`'s computed bytes (the closed-form per-channel
bias/multiplier pair the narrow-shape format already explains) as the "code"
and the scattered scaffold bytes as the "table" -- recovers **14,626 of
28,672 bits (51%)** as a clean permutation, with only 179 collisions at the
full 47-build sample (likely resolvable with a handful more builds, not
attempted here for time). The remaining **10,766 bits (37.5%) are genuinely
unexplained** by the requantisation formula: not weight-code bits, not
`requant_block()` bytes, present in the confirmed-scaffold region, and
varying across builds in a way this session did not identify. `conv512`'s
same 47-build sample gives comparable structure (14,360 scattered scaffold
bytes; a second-pass learn was not run for it, for time -- see "What was not
attempted").

## `learn_mcode` fails outright at this shape -- a real, undecoded compiler form-split

The toy shape's mcode is "otherwise a function of shape alone" per
`learn_mcode`'s own docstring, with the output scale/zero point written as a
handful of literal bytes findable automatically. At `conv256` this broke in
a new way: of 47 builds, mcode length itself splits into two groups (46 at
14,456 bytes, 1 at 14,488) -- expected, `learn_mcode` already anticipates one
outlier build shifting layout. But **within the 46-build majority-length
group, `learn_mcode`'s automatic scale-offset search still found nothing**:
manually checking whether `y_scale` appears as a literal float32 at the
position it occupies in one reference build (offset 5527, repeated at
5535/5543/5551, matching the "four copies" the README documents) found it
present, byte-exact, in only **24 of the 46** majority-length builds. The
other 22 use some different byte layout for the same field -- a second,
undecoded split this project has not seen at the toy shape, where
`learn_mcode`'s `min_agreement=0.9` threshold cleanly separated one
true outlier from 47 agreeing builds. Restricting to the 24-build "form-A"
subgroup, the zero-point byte was found the same way, manually, at offset
5514.

**`scripts/axera/conv_learn_wide.py`'s `MCODE_SCALE_OFFSETS_FORM_A`/
`MCODE_ZERO_OFFSET_FORM_A` are therefore specific to one reference build's
form, not shape-general** -- unlike everything else `emitter.py` exposes,
this is not something `learn_mcode` validated across the whole sample.

## Device measurement: how much the unexplained ~49% actually costs

A held-out weight set (`conv256_holdout_weights.npz`, native seed 9999,
natively compiled as ground truth) was emitted two ways and run on the AX8850
(`axcl-vm`, serialized under the shared device lock, a native-build control
run first):

| variant | mean abs error | max abs error | fraction >0.5 from native |
| --- | --- | --- | --- |
| native build vs. numpy conv2d (baseline device noise) | 0.0149 | 0.085 | -- |
| emitted, reference's own (mismatched) scale/zero, no mcode patch | 0.827 | 11.26 | 52.7% |
| emitted, held-out's true scale/zero patched into "form-A" offsets | 0.633 | 11.01 | 33.5% |

Peak output magnitude is ~6.2. Patching the output scale/zero point measurably
helps (mean error 0.827 -> 0.633) but the result is still nowhere near the
0.015 baseline device noise -- **confirming the unexplained ~49% of the
scaffold region is load-bearing, not padding or unused precision.** Whatever
it encodes materially affects the convolution's output.

## Bottom line

- **Validated and reusable**: the weight-code bit-permutation method itself,
  at real 256- and 512-channel shapes, with the same build-count economics
  the toy shape showed. `emitter.learn()`/`emitter.emit_table()` need no
  change to work here.
- **Not validated, and the actual blocker**: about half of the
  scaffold/requantisation region at wide-channel shapes, plus the mcode
  scale/zero-point fields' own undecoded form-split. Neither closes with the
  47-48 builds gathered here.
- **This is a characterization result, not a working emitter for these
  shapes.** `scripts/axera/conv_learn_wide.py` exists to make the byte-exact
  and not-yet-explained portions inspectable and extendable, not to be
  dropped into a pipeline expecting correct output -- its own device
  measurement above says plainly that it is not yet correct.

## What was not attempted

- A second-pass `learn()` on `conv512`'s scaffold region, or any held-out
  device check for `conv512` at all -- only the weight-code convergence
  table above was confirmed there, for time.
- Closing the 179 second-pass collisions at `conv256` with more builds.
- Finding what the remaining unexplained ~37.5% of the scaffold region
  actually encodes (per-slice partial sums, duplicated precision, tiling
  metadata -- not investigated).
- Resolving the mcode form-split (what separates the ~24-build "form-A"
  group from the rest), or finding `learn_mcode`-style automatic detection
  robust to it.
- Any shape besides these two (e.g. the step's `[16,128,28,28]`/
  `[16,64,56,56]` blocks, or 1x1/stride-2 downsample convs).

## Reproduction and fixture sizes

Builds ran in `/home/takecheeze/npu-scratch/t_conv_learn_wide` (not
committed -- 96 compiled `.axmodel`s, several MB each). Committed fixtures
under `scripts/axera/fixtures/conv_learn_wide/` total about 11 MB: a
gzipped reference build (~0.6 MB), a gzipped held-out oracle build (~0.6 MB),
the held-out weight/bias pair (~2.2 MB, compressed), and the learned maps
themselves (~7.2 MB, dominated by one `int32` per table bit -- 4,753,920 of
them). This is larger than most fixtures elsewhere in this project; noted
here rather than trimmed further, since the maps are the actual reusable
artifact and shrinking them below one entry per bit was not attempted.
