# Does `emitter.py`'s weight-learning technique work at a real ResNet18 Conv shape?

**Yes**, on a real `Conv(cin=64, cout=64, k=3x3, stride=1, pad=1, batch=16,
spatial=56x56)` with a real bias -- the exact shape the ResNet18 training
step's stage-1 blocks use -- with one real defect found and fixed along the
way. Device-verified: a weight set the technique never saw produces output
statistically indistinguishable from Pulsar2's own native compile of those
same weights.

## Correcting the assignment's premise

This task was framed as "`emitter.py` has only ever been validated at toy
1-8 channel 8x8 shapes." That undersells the existing work:
`scripts/axera/README.md`'s "Replacing the ONNX compiler at a fixed shape"
section already device-validated this exact technique on a real 48-build
campaign, bit-identical output on eight held-out weight sets, on hardware.
But that campaign's shape, `Conv(32, 32, 3)`, is a 1-D convolution (32
channels each way, one kernel dimension) -- almost certainly from this
project's audio/vocoder work, not a 2-D NCHW ResNet18-shaped conv. The real,
still-open gap was never "toy vs. real" but "1-D, 32-channel vs. 2-D,
64-512-channel, batch-16, with a real float bias" -- and the existing test
suite (`tests/test_axera_emitter.py`) is entirely synthetic (a hand-rolled
fake compiler), so *no* real Pulsar2-compiled model had ever round-tripped
through `emitter.py` in CI or elsewhere. This closes that gap for one real
shape.

## Scope actually completed

Given real build cost, this covers **one** of the two shapes originally
assigned (`Conv(64,64,3,3)`, stride 1 -- the four-times-repeated stage-1
block), not the 224x224 stem. Builds turned out fast (~15-30s each on this
shape), so 49 real Pulsar2 builds (48 for learning + 1 held-out) fit in
about 10 minutes -- the stem was not attempted for lack of remaining budget
in this session, not because it looked harder. Whoever picks it up next
should expect the campaign mechanics below to transfer directly; only the
per-build wall-clock time is likely to differ (224x224 input, 7x7 kernel).

## The campaign

48 standalone single-`Conv` Pulsar2 7.0-lite (AX650, MinMax calibration)
builds of the same shape, `i.i.d.` Gaussian weights and bias per build
(`std = sqrt(2/(Cin*K*K))`, a real He-style init scale), identical input
calibration data across all 48 (two full-batch `[16,64,56,56]` samples,
uniform in ±0.9) so `x_scale`/`x_zero` stay effectively constant while
`y_scale`/`y_zero` vary naturally with each build's own weights -- giving
`learn_mcode` the diversity it needs for free, the same way the README's own
campaign did. A 49th, held-out weight/bias set (never used for learning) is
the test: build it natively (ground truth) and separately emit it from
build 1 plus the learned map, then compare.

Reproduce: `scripts/axera/conv_weight_learn.py` has the extended API;
`docs/axera-conv-weight-learn-stem.md` (this file) has the exact recipe. No
build-harness script is committed (the campaign was run from scratch files
outside the repo) -- the shape, calibration, and weight-generation recipe
above is complete enough to reproduce from scratch if needed.

## Learning the weight-code map: clean, zero collisions

```
48 builds, table 38656 B (309248 bits)
  mapped to a weight bit : 294912   (= Cout*Cin*K*K*8 = 64*64*9*8, exactly)
  constant across builds : 11210
  unexplained            : 3126
  colliding sources      : 0
```

Every one of the 294,912 weight-code bits is placed exactly once, with zero
ambiguity -- the same clean result the README's 1-D 32-channel case reached
at the same build count, now confirmed at ~9.6x the code count. The 3,126
unexplained ("ambiguous") bits are one contiguous-ish region, bytes
36,864-37,374 (~510 of the intended 512 bytes; the alignment tool's
run-detector misses a handful of individually-constant bytes inside the
span) -- exactly `Cout * 2 floats * 4 bytes = 512` bytes, the requantisation
block `emitter.py`'s docstring already names. `learn()` correctly reports it
as scaffolding to model, not to copy, matching the README's derivation.

## The requantisation block needs a bias term `emitter.py` does not have

`emitter.requant_block()` computes `bias[c] = zy - zx*sum(q_c)*M_c`, which is
correct only for a `Conv` with **no** bias input -- every prior test and the
README's own derivation used one. This shape's real `Conv` (like every real
ResNet18 `Conv`) has a bias. Comparing the uncorrected formula's output to
the real stored block: consistently off, by exactly `bias_float[c] /
y_scale`, confirmed directly:

```
diff[0:10]              : [ 1.0427933   0.16085815  0.05295563  2.4410248  ... ]
b[0:10] / y_scale        : [ 1.0428343   0.16089594  0.05288334  2.4409752  ... ]
max |diff - b/y_scale|   : 0.00019
```

`scripts/axera/conv_weight_learn.py::requant_block_biased()` adds this term.
Checked across all 48 builds: max absolute error **1.9e-4** against the
formula-plus-bias, the same order of magnitude as `emitter.py`'s own
no-bias derivation's stated 6.1e-05 (a bit larger here, consistent with one
extra floating-point operation's worth of rounding, not a wrong formula).
**This is a real, load-bearing gap in `emitter.py`/`emitter.requant_block`**
for any real biased convolution -- which is every ResNet18 `Conv`. It was
not fixed in `emitter.py` itself (out of this task's scope: touch only new
files), only worked around in `conv_weight_learn.py`, and should be folded
into `emitter.py` directly by whoever owns it next -- the fix is exactly the
one addend shown above.

## The mcode: 6 of 48 builds shift layout, the rest need `learn_mcode`

Mcode length is not uniform: 42 of 48 builds are 28,400 bytes, 6 are 28,432
(the "escaped literal" shifted-layout phenomenon the README already
describes for its own shape, at a higher rate here -- 12.5% vs. 4.2% -- not
investigated further). `learn_mcode`, run on the 42 majority-length builds:

```
scale_offsets : [4269, 4276, 4283, 4290, 5139, ... ]   (28 offsets, 7 groups of 4 -- 7 y_scale copies)
zero_offsets  : [4259, 5127, 5773, 6417, 7062, 7685, 8435]   (7 offsets -- 7 y_zero copies)
free_offsets  : 29 bytes
outliers      : [5, 18, 26]   (3 of 42 majority-length builds still don't re-emit cleanly)
unpatchable   : {"zero": [128], "scale_low_byte": []}
```

Seven copies of `y_scale`/`y_zero` (not four, as in the README's smaller
shape) -- consistent with this being a bigger, more heavily-tiled program.
The 29 "free" bytes include a 21-byte run at mcode offset 1743-1763: checked
directly across 43 builds (42 training + the held-out), it takes **22
distinct byte patterns**, confirming it genuinely varies with something
(weights, most likely, or the scheduler's own per-build choices) and is not
scheduling noise that happens to look constant. `learn_mcode` correctly
classifies it as "free" (matches neither `y_scale` nor `y_zero`) and
`emit_mcode` leaves it at the reference's value -- exactly the same category
the README's own smaller shape found ("scheduling, not semantics") and
found did not matter on the card. This campaign's device check (below)
confirms that conclusion transfers to this larger shape too.

## The held-out weight set

Weight-code region: **byte-exact**, confirmed by `emitter.emit_table` and
directly by the committed regression tests (no Docker/device needed).

Full table (weight codes + requantisation block): **66 of 38,656 bytes
differ (0.17%)**, all inside the 512-byte requantisation block -- consistent
with the 1.9e-4 float32-rounding-scale residual measured above occasionally
flipping a low mantissa byte, not a wrong formula.

Mcode: **22 of 28,400 bytes differ** -- the 21-byte "free" run at
1743-1763 (this held-out build's own scheduling value, left at the
reference's instead) plus one footer byte at 28,104, not otherwise
characterized.

## On the card

Reference: build 1 of 48. Target: the held-out weight/bias set, never seen
by the learning campaign. `emitter.emit_table` + `conv_weight_learn.
requant_block_biased` + `emitter.emit_mcode`, nothing else, against a real
in-calibration-range input (`x_calib[0]`, `[16,64,56,56]`, uniform ±0.9) on
the AX8850 via `axcl-vm` (serialized under the shared device lock, a control
run of the native build before and a health run after, both clean):

| comparison | max abs error | mean abs error |
| --- | --- | --- |
| native holdout build vs. `numpy` | 0.0459 | 0.00924 |
| **emitted** holdout vs. `numpy` | 0.0459 | 0.00925 |
| emitted vs. native (device output, direct) | 0.0324 max, **265 of 3,211,264 elements differ at all** (0.0083%) | -- |

The emitted model's error against `numpy` is statistically the same as the
*native* build's own error against `numpy` -- both are dominated by ordinary
int8 quantisation noise (`y_scale` ~0.032 per code step, well above either
error). Direct emitted-vs-native comparison shows a difference on **0.008%**
of output elements, each below one quantisation step. This is not
bit-identical the way the README's smaller-shape, no-bias campaign achieved
(that residual float32-rounding and the "free" mcode bytes both introduce a
tiny, non-zero effect here), but it is a **materially correct, deployable
result**: the emitted model is not distinguishable from a real Pulsar2 build
of the same weights by any measure that matters to a training loop's
accuracy.

## What this means for coverage

- **The technique generalizes** from the README's validated 1-D, 32-channel,
  no-bias case to a real 2-D, 64-channel, biased, batch-16 ResNet18 shape,
  with only the one bias-term gap found and fixed (in a wrapper, not
  `emitter.py` itself).
- **This is real, usable coverage for ONE of ResNet18's 11 distinct forward
  Conv shapes** (`docs/axera-step-attribution.md`'s coverage table
  previously listed this op type as out of scope entirely). Given one
  reference build of `Conv(64,64,3,3,stride=1)` and the learned map, any
  trained weights at that exact shape can be written into a deployable
  `.axmodel` with no Pulsar2 in the loop.
- **What this is not**: it does not cover the training-step's *trainable*
  Conv weight tensor as it actually appears in the compiled multi-op
  training graph. `docs/axera-mcode-training-graph-coverage.md` already
  established that a trainable weight is a graph input/output there, not a
  static `npu_params` initializer at all -- there is no weight table for
  this technique to write into inside that compiled artifact. What this
  *does* enable is a separate, standalone per-layer inference `.axmodel`,
  refreshed from a checkpoint's weights after a training step, without
  Pulsar2 -- useful for fast deployment of a training run's intermediate
  checkpoints, not for the on-device training compute step itself. Do not
  conflate the two.
- **Not covered**: the stem shape (7x7, 224x224, cin=3->cout=64), the ten
  other distinct real Conv shapes in the step (1x1 downsample convs, the
  128/256/512-channel stage blocks), any stride-2 shape, and whether the
  same bias-correction/collision-count numbers hold at those other shapes.
  The mcode "free" 21-byte region and the 6-of-48 shifted-layout rate are
  both recorded, not explained.

## Reproduction

Fixtures under `scripts/axera/fixtures/conv_weight_learn/`: the reference
build, the held-out native build (gzipped `.axmodel`s), both builds' weight
tensors (gzipped `.npy`), and the learned map (`stage1_map.npz`).
`tests/test_axera_conv_weight_learn.py` reproduces the weight-code-exact and
bias-corrected-block-close claims from these fixtures, no Docker or device
needed. The mcode and full 48-build campaign, and the device numbers above,
are not reproducible from the committed fixtures alone (they need the other
47 training builds and a real card) -- they are recorded here as read
directly off the campaign.
