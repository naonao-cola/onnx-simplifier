# Gather index retargeting into a real aggregating consumer: the toy-scale range risk reproduces, and wide input calibration fixes it

`docs/axera-compose.md` (PR #1732) found, on a toy `Gather(x[1,1,4,16],
idx[8]) -> Reshape -> MatMul -> Transpose -> Add` chain, that retargeting a
composed Gather's indices without recalibrating inherits the reference
build's calibration *range*: the emitted model clipped at the reference's
own calibrated bound rather than the new indices' true range.
`docs/axera-gather-compose-real-scale.md` (PR #1759) then checked a real,
ResNet18-scale `Gather -> Mul`(mask) chain and found no such problem --
because a `{0,1}` mask cannot expand a value's range, only zero or pass it
through. Both left "an aggregating consumer" (the actual case that failed at
toy scale) unverified at real scale. This does that check.

## The real chain

The real op that follows a stem-style Gather in the ResNet18 training step
(`/home/takecheeze/npu-scratch/t6-r18fold/step.onnx`, nodes `Gather_48`,
`Mul_49`, `Reshape_50`, `MatMul_54`) is:

```
Gather(x[16,1,512,49], idx[441], axis=3)   # out [16,1,512,441]
  -> Mul(mask[1,1,1,441])                  # {0,1} padding mask, 18.1% zero
  -> Reshape([16,1,4608,49])
  -> MatMul(w[1,1,512,4608], reshaped)     # out [16,1,512,49]
```

`w` is itself a *live* tensor in the real graph (not a constant), matching
`legalize.py`'s `act_weight_conv_to_matmul`-style live-weight matmuls
elsewhere in this project. Note this chain is **not** produced by
`legalize.py`'s `act_weight_conv_to_matmul`/`dilated_conv_to_taps` -- both of
those build taps with `Slice`, not `Gather`. The real `Gather`s in the step
are already present in the graph `graph_grad` (torch autograd) produces,
before `legalize.py` runs; the MatMul here contracts over the reshaped
4608-element axis, which is exactly the "aggregation over the gathered axis"
shape that matters for this check.

The real index array (`idx`, extracted from the model) is 441 int64 values
in `[0, 48]`; the value `0` appears 84 times (the clamped-and-masked
out-of-bounds/padding positions), every other value 6-9 times.

## Build

Three Pulsar2 7.0-lite (AX650, MinMax) compiles of this exact 4-op graph,
`x`/`w` both graph inputs (own calibration datasets), scratch under
`/home/takecheeze/npu-scratch/t_gather_aggregate_real`:

- **`a_reference_narrow`**: real `idx`. Calibration data for `x` is
  *structured*, not uniform: positions 0-39 of the last axis are drawn from
  `[-0.3, 0.3]`, positions 40-48 from `[-0.9, 0.9]` -- so which positions an
  index set selects actually changes the aggregate's likely magnitude,
  matching how real conv activation statistics vary by spatial/channel
  position.
- **`b_native_adversarial`**: a fresh native build with an adversarial index
  set, `adv_idx = tile([40..48], 50)[:441]` -- every one of the 441 gathered
  positions lands in the "large" 40-48 band, unlike the real `idx` (mostly
  position `0`, in the "small" band). Same narrow calibration as (a). This is
  the ground truth for "what should the retargeted model compute".
- **`c_wide_reference`**: real `idx`, but `x`'s calibration data is uniform
  `[-0.9, 0.9]` at *every* position (not just 40-48).

All three compiled to one `neu mode` node each (853 ops, max_cycle
30,724,188). `npu_params` is **10,861 bytes** in every build -- not a whole
number of uint32 words. The first `441*4 = 1,764` bytes are exactly that
build's own index array as little-endian uint32 words (confirmed by direct
comparison, both for the real and the adversarial index set): the
"leading words are the indices" layout `memory_emit.py` and
`gather_compose_real_check.py` rely on holds for this real, 4-op, live-weight
composed graph too, not just the 2-op mask case.

**Bug found in `gather_compose_real_check.patch_indices`** (PR #1759): it
computes `len(table.raw_data) // 4` and repacks that many words, silently
dropping the trailing byte when the table isn't a whole number of words --
`struct.error` here (10861 // 4 * 4 = 10860, one byte short). This module's
`patch_indices_bytesafe` rewrites only the leading index bytes in place and
never touches anything after them, so it isn't affected by the table's total
length. Not fixed in place: `gather_compose_real_check.py` is shared with
other work; the fix is a new function in a new module instead.

## Device result: the aggregation failure mode reproduces, and wide calibration fixes it

Emitted `a_reference_narrow` retargeted to `adv_idx` (`patch_indices_bytesafe`,
keeping (a)'s scales), and separately `c_wide_reference` retargeted to the
same `adv_idx`. Ran all three -- `b_native_adversarial` (ground truth),
`emitted_a` (narrow calibration + new indices), `emitted_c` (wide calibration
+ new indices) -- against the same adversarial test input (positions 40-48 at
`[0.7, 0.9]`, matching the "large" band) on the AX8850 via `axcl-vm`, under
the shared device lock, control run first (`a_reference_narrow` against its
own in-range input matched a numpy reference to 0.0287 max error, in line
with ordinary int8 quantization noise for this shape).

| comparison | max abs error | mean abs error |
| --- | --- | --- |
| native vs numpy (ground truth's own quantization noise) | 1.361 | -- |
| **emitted_a (narrow) vs native** | **1.220** | 0.0237 |
| **emitted_c (wide) vs native** | **0.146** | 0.0122 |

For scale: `numpy`'s true output range for this input is `[-4.84, 5.91]`.
`native`'s range is `[-4.52, 4.55]` (quantization pulls it in slightly, as
expected). `emitted_a`'s range is **`[-3.30, 3.54]`** -- visibly clipped, and
it has 5,771 output elements within 1% of its own max magnitude against
native's 1,477 (a classic saturation signature, not just diffuse noise).
`emitted_c`'s range is `[-4.66, 4.63]`, matching native closely.

So: **the toy-scale finding generalizes to this real op and shape.**
Retargeting `a_reference_narrow`'s indices without recalibration produces an
error (max 1.22) of the same order as the entire signal (native's own range
is about 4.5) -- not usable. Retargeting `c_wide_reference` instead reduces
that to 0.146, comfortably inside the noise floor `native`'s own quantization
already sets against the true numpy value (1.36) -- **usable**.

## What "wide calibration" means operationally

The fix here calibrated `x` (the *input* to the Gather) with data spanning
its full valid range **uniformly across every last-axis position** -- not
data crafted to cover the aggregate's (post-MatMul) output range directly,
and not data that specifically visits the indices you intend to retarget to.
Pulsar2's own calibration evidently propagates a wide-enough `x` calibration
forward through `Gather -> Mul -> Reshape -> MatMul` to give the output a
correspondingly wide MinMax estimate, without needing that propagation
spelled out by hand. This is the practical, checkable recipe for anyone
recalibrating a reference before using `memory_emit.py`'s (or `compose_emit`'s
or `gather_compose_real_check`'s) index-retargeting emitters ahead of an
aggregating consumer: **calibrate the gathered tensor's full input range at
every position the indices could ever select, not just the reference's own
index pattern's positions.** This was not derived from first principles here,
only confirmed empirically on this one shape/op/consumer combination.

## Scope and what is still open

- One real op sequence (`Gather -> Mul -> Reshape -> MatMul`), one real shape
  (`[16,1,512,49] -> 441 -> [16,1,4608,49]`), one adversarial index pattern,
  one test input. Not swept over other shapes, index patterns, or the other
  consumer types in the step (plain `Reshape`-into-`MatMul` without the mask,
  or chains ending in `Add` rather than `MatMul`).
- Does not check the *stem* Gather specifically (614,656 indices,
  `docs/axera-stem-gather.md`'s 7-chunk model) -- that one is additionally
  blocked by the uint16 last-axis addressing limit
  `docs/axera-gather-compose-real-scale.md` found, independent of this
  calibration-range issue.
- Does not check whether Pulsar2's calibration-propagation behavior (wide `x`
  calibration -> wide aggregate estimate, without hand-computing the
  aggregate's range) holds for a different aggregation shape or op (e.g. a
  much larger contraction, or an `Add`-sum rather than `MatMul`).
- `emitted_c`'s residual 0.146 error against `native` is itself unexplained
  beyond "comparable to ordinary quantization noise" -- it was not decomposed
  further.

Scratch builds and raw device output arrays are at
`/home/takecheeze/npu-scratch/t_gather_aggregate_real`.
