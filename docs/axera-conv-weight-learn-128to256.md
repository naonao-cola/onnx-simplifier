# The 128->256 downsample pair: both shapes fully working end to end

The last untested pair among ResNet18's four stage-transition downsample
paths (`64->128`: `docs/axera-conv-weight-learn-downsample.md`; `128->256`:
here; `256->512`: `docs/axera-conv-weight-learn-256to512.md` and its
follow-up fix `docs/axera-conv-256to512-tiled-fix.md`). Two real shapes:
`Conv(x[16,128,28,28], w[256,128,1,1], stride 2, pad 0)` and
`Conv(x[16,128,28,28], w[256,128,3,3], stride 2, pad 1)`, both with a real
trained bias.

**Both shapes are validated end to end and confirmed on the AX8850** -- the
first `Cout>Cin` downsample pair in this project's Conv work where *both*
the 1x1 and the 3x3 member fully close, not just one of the two (contrast
`docs/axera-conv-weight-learn-downsample.md`'s `64->128` pair, where the 3x3
member was left with an open scaffold gap).

## Weight-code region: byte-exact, as at every shape tried

`emitter.py`'s `learn`/`emit_table` bit-permutation, unmodified:

| shape | code bits (`Cout*Cin*K*K*8`) | `k` for zero collisions |
| --- | --- | --- |
| 1x1 (`[256,128,1,1]`) | 262,144 | 46 |
| 3x3 (`[256,128,3,3]`) | 2,359,296 (9x more) | 54 |

Both found empirically (build a batch, check `emitter.collisions()`, add more
if nonzero), continuing the pattern `docs/axera-conv-weight-learn-
downsample.md` and `docs/axera-conv-weight-learn-128-and-widegap.md` already
established: `k` tracks the code-bit count, not a fixed small number. (One
early campaign run wasted a few builds past convergence because of a
convergence check bug on my own side -- `ambiguous < 8`, when the ambiguous
count is dominated by the scaffold region and never gets that small; the
actual convergence signal is `collisions == 0` alone, which both shapes hit
cleanly at the `k` above.)

## Scaffold placement: 1x1 stays contiguous, 3x3 tiles in 8

Continuing the pattern `docs/axera-conv-scaffold-arithmetic.md` found for
`Conv(128,128,3,3)`: the per-output-channel `(bias, M)` requantisation block's
placement depends on `Cin*K*K`, via a fixed 36,864-byte weight-code-per-tile
budget.

**1x1** (`Cin*K*K=128`): the tile-width budget (`36864 // 128 = 288`) exceeds
`Cout=256`, so it should be one contiguous tile -- and empirically, it is.
The first-pass `learn()`'s own `ambiguous` bit range pins it directly: byte
36,864 to 38,910 of the 39,392-byte table, a single flat
`[bias(256 x f32)][M(256 x f32)]` block, **not** per-tile-interleaved. This
was checked directly against the real compiled table (not assumed from the
formula) and matches, byte-exact for `M`, within the usual ~4e-4 tolerance
for `bias`.

This differs from `docs/axera-conv-256to512-tiled-fix.md`'s "Fact 1"
(K=1 shapes use a 128-channel tile cap independent of `Cin`, confirmed at
`Cin=64` and `Cin=256`) -- at `Cin=128` exactly, there is no splitting at all,
just one 256-wide block. This is a genuine third K=1 data point that refines
Fact 1's "independent of Cin" claim: the real condition governing whether a
K=1 shape splits looks tied to `Cin` after all, plausibly the same `Cin>128`
threshold Fact 2 (below) uses for its own, structurally different
duplication behaviour -- at `Cin<=128` neither effect triggers. Only one
shape was checked here, so this is a data point, not a swept rule.

**3x3** (`Cin*K*K=1152`): `tile_width = 36864 // 1152 = 32`, giving 8 tiles of
32 output channels, each `[32 x weight codes][bias(32 x f32)][M(32 x f32)]`
= 37,120 bytes, headerless (tile 0's weight codes start at table byte 0).
Confirmed three independent ways, not just formula-trusted:

1. **Direct byte-search** for the computed `M` value of channel 0 in each
   tile finds it at exactly `tile*37120 + 36992` for all 8 tiles -- i.e.
   `bias` at `+36864`, `M` at `+36992`, matching the headerless formula
   exactly.
2. **The first-pass `learn()`'s own ambiguous bits** land almost entirely
   inside those same 8 `[bias][M]` spans (768-834 of 1024 bits per span) and
   nowhere else.
3. **A 480-byte trailing region** the pure `8*37120=296,960` tile-formula
   prediction misses (`297,440` is the real table length) carries **zero**
   ambiguous bits across every build in the campaign -- constant, unrelated
   to weights, safe to leave at the reference's own value. Not resolved
   further what it is; likely the same kind of fixed footer/metadata this
   project's Transpose and DMA-queue tile-table decodes also found outside
   their own tiled regions.

## A real bug found in already-merged code

`scripts/axera/conv_scaffold_arithmetic.py`'s `emit_conv_table_tiled`
(merged to `origin/master`, `docs/axera-conv-scaffold-arithmetic.md`) assigns
a raw `bytes` object directly into a `numpy.uint8` array slice:
`table[a:b] = float32_array[...].tobytes()`. This does not perform a byte
copy -- NumPy tries to cast the `bytes` object itself into the destination
dtype and raises `ValueError: invalid literal for int() with base 10: b'...'`
for any real-sized assignment. Confirmed with a two-line minimal repro,
independent of this project's shapes entirely. The function was never
actually exercised against a real build before merging -- its own docstring
says so ("No end-to-end emission was device-checked") -- which is how this
survived. The fix is `np.frombuffer(...tobytes(), dtype=np.uint8)` in place
of the bare `.tobytes()`; `scripts/axera/conv_128to256_tiled.py`'s
`emit_p3_table` reimplements the tiling logic directly with this fix rather
than depending on the broken function (this project's established pattern
for not editing another module in place); whoever owns
`conv_scaffold_arithmetic.py` should apply the one-line fix there too.

## Device verification: both shapes, full pipeline, on the AX8850

`axcl-vm`, serialized under the shared device lock, a native-build control
run before and a health-check rerun after each shape, both clean. The
emitted model uses the reference build's own `x_scale`/`x_zero` (the
realistic emission scenario -- an emission has no separately-recalibrated
holdout to read scale from) against a **holdout native build compiled with
the reference's exact calibration dataset**, isolating weight/bias-dependent
correctness from independent input-quantisation drift between builds (see
the methodology note below).

| shape | native max err vs. numpy | emitted max err vs. numpy | emitted vs. native, direct |
| --- | --- | --- | --- |
| 1x1 | 0.3814 | **0.3814** (identical) | 0.0362 |
| 3x3 | 0.2139 | **0.2139** (identical) | 0.1018 |

The emitted model's error against numpy is *exactly* the native build's own
quantisation noise for both shapes, not a larger or systematic error -- the
strongest available signal that the table (weight codes + scaffold) is
correct, not merely close. `docs/axera-conv-weight-learn-downsample.md`'s
1x1 case showed the same identical-error signature; this confirms it holds
again for both members of a wider, asymmetric-channel downsample pair,
including the tiled 3x3 case that no prior 3x3 downsample shape reached
end-to-end.

### A methodology note, worth stating precisely

An earlier attempt at this same comparison used a holdout built with an
*independently* seeded calibration dataset (different random samples, not
just different weights) and found a spurious 1.1-unit bias residual --
traced to the holdout's own asymmetric-MinMax calibration landing on a
different input zero-point (127 vs. the reference's 128) purely from
calibration-sample noise, unrelated to weights at all. Using the *same*
calibration data for the reference and the holdout (varying only the
weights/bias, which is what the emitter is actually meant to vary) removes
this confound and reproduces the formula's documented ~1e-4-to-4e-4 rounding
tolerance cleanly. This is a real methodological hazard for anyone else
testing this pipeline: a "wrong"-looking large residual may be an
uncontrolled calibration-seed difference between reference and holdout, not
a formula bug -- check `x_zero` matches before concluding otherwise.

## What this means for coverage

- **Both `128->256` downsample shapes now have a complete, device-confirmed
  emission pipeline** -- weight-code bit-permutation (standard, shape-general)
  plus scaffold placement (shape-specific, but now understood for both the
  contiguous 1x1 case and the 8-tile 3x3 case).
- **This is the first `Cout>Cin` downsample pair where the 3x3 member also
  fully closes**, not just the 1x1 member -- `64->128`'s 3x3 was left open,
  `256->512`'s 3x3 needed a separate two-fact fix
  (`docs/axera-conv-256to512-tiled-fix.md`) for a `Cin>128` duplication this
  shape's `Cin=128` does not trigger.
- **A real, previously-unexercised bug in merged code** (`conv_scaffold_
  arithmetic.emit_conv_table_tiled`) is found, precisely diagnosed, and
  worked around in a new module rather than fixed in place.
- **Not covered**: composition with a real neighbour
  (`docs/axera-conv-compose-real.md` and `docs/axera-conv-compose-tiled-
  fix.md` cover this question for other Conv shapes, not these two); the
  refined K=1 tiling hypothesis above rests on one data point, not a sweep.

## Reproduction

Fixtures under `scripts/axera/fixtures/conv_learn_128to256/`: a reference and
held-out native build (gzipped `.axmodel`s plus their `quant_axmodel.json`s)
for both shapes, the held-out weight/bias tensors, and both shapes' learned
maps (`emitter.save_map`/`load_map` format). `tests/test_axera_conv_128to256_
tiled.py` reproduces every table-level claim above (not the device numbers)
with no Docker or device required, plus a standalone repro of the
`conv_scaffold_arithmetic.py` bug.

The `k`-build campaigns themselves (46 and 54 Pulsar2 builds respectively,
scratch under `/home/takecheeze/npu-scratch/t_conv_learn_128to256` and
`/home/takecheeze/npu-scratch/t_conv_learn_128to256_p3`) are the expensive,
non-reproducible-from-the-repo part; the committed fixtures are their output.
