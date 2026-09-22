# The wide-Cout Conv scaffold is a deterministic per-tile layout, not a bit-permutation-learned region

Five prior docs (`docs/axera-conv-weight-learn-{stem,downsample,wide}.md`,
`docs/axera-conv-weight-learn-128-and-widegap.md`,
`docs/axera-conv-weight-learn-256to512.md`) established that `emitter.py`'s
bit-permutation weight learner reproduces every real ResNet18 Conv shape's
weight-code region byte-exact, but the per-output-channel `(bias, M)`
requantisation block ("the scaffold") stops being one contiguous span once
`Cout` grows past a threshold, and a *second* bit-permutation learning pass
on that scattered region only ever recovered 51-80% of it -- and for the
widest shape tried (`Conv(512,256,1,1)`), the values it converged to were
confirmed **wrong** on real hardware, including one that faulted the AX8850
runtime outright (`docs/axera-conv-weight-learn-256to512.md`).

This finds the scaffold's actual layout by direct search against real
compiled tables, instead of statistically inferring a bit permutation for it.

## The premise this started from was half wrong, and worth correcting

`emitter.requant_block()` (extended by `conv_bias_requant.py`'s
`requant_block_with_bias()`) already computes the scaffold's *values*
arithmetically -- `M_channel = x_scale * weight_scale_channel / y_scale`,
`bias_channel = y_zero - x_zero * sum(q_channel) * M_channel + bias_channel /
y_scale` -- and this formula was already confirmed byte-exact (M) and within
~6e-5 float32-rounding tolerance (bias) for the *contiguous*-scaffold shapes
(`docs/axera-conv-weight-learn-stem.md`, the 1x1 downsample in
`docs/axera-conv-weight-learn-downsample.md`). The prior forks' "second
learn()" on the *scattered* shapes was not disputing this formula -- it was
trying to find *where in the table* the formula's output bytes land, using
`requant_block()`'s computed bytes as the bit-permutation "code" instead of
guessing the placement directly. The actual open problem was placement, not
arithmetic -- and placement turns out to be fully deterministic.

## The layout: per-output-channel-tile, budget-limited, no learning needed

Searching a real compiled `Conv(128,128,3,3)` table for the exact byte
sequence of each computed `M_channel` and (loosely, allowing the same ~1.5e-4
tolerance the contiguous-shape formula already carries) `bias_channel` value
finds them at a clean, structured set of offsets, not scattered randomly:

```
channel 0-31:   weight codes [0:36864)      bias [36864:36992)   M [36992:37120)
channel 32-63:  weight codes [37120:73984)  bias [73984:74112)   M [74112:74240)
channel 64-95:  weight codes [74240:111104) bias [111104:111232) M [111232:111360)
channel 96-127: weight codes [111360:148224) bias [148224:148352) M [148352:148480)
```

Each 32-channel **output-channel tile** stores its own weight codes
immediately followed by its own `[bias(32 x f32)][M(32 x f32)]` pair --
256 bytes -- before the next tile's weight codes begin. This explains the
"scattered" signature every prior doc found: the bit-permutation learner,
working on the whole table as one flat bit vector, saw 128 small
scaffold-shaped islands (one bias+M pair per channel) sitting between much
larger weight-code regions, not one contiguous block.

**The tile width is set by a fixed per-tile weight-code byte budget, not a
fixed channel count.** `tile_width = 36864 // (Cin * K * K)`, clamped to
`Cout`:

| shape | `Cin*K*K` | predicted tile width | matches |
| --- | --- | --- | --- |
| `Conv(64,64,3,3)` | 576 | 64 (= Cout, one tile) | contiguous scaffold, already confirmed working (`docs/axera-conv-weight-learn-stem.md`) |
| `Conv(128,64,1,1)` | 64 | 128 (one tile) | contiguous, `block_at=9216` already confirmed working (`docs/axera-conv-weight-learn-downsample.md`) -- see caveat below |
| `Conv(128,128,3,3)` | 1152 | 32 | **directly verified here** |
| `Conv(256,256,3,3)` | 2304 | 16 | run-count matches the independently-published finding, see below |
| `Conv(512,512,3,3)` | 4608 | 8 | not independently checked |

`36864` is not an arbitrary fit: it is exactly `64 * 576` (the 64/64 shape's
*entire* weight-code size) and exactly `32 * 1152` (one 128/128 tile) --
consistent with a real fixed on-chip buffer size for weight loading, the same
kind of byte-budget tiling this project's Transpose (`docs/axera-transpose-
tiled.md`, 131,072-byte tile budget) and DMA queue
(`docs/axera-dma-queue.md`, 524,288-byte tile budget) decodes already found
for other engines -- a third independent instance of the same mechanism.

## Verification

**Blind, twice, no search in the final check.** Two standalone
`Conv(128,128,3,3)` Pulsar2 7.0-lite builds, independent random weights and
biases (different RNG seeds), independent calibration data. For each,
`quant_axmodel.json` (Pulsar2's own build-output artifact, not something
this project had read before -- it holds the real `x_scale`/`x_zero`/
`y_scale`/`y_zero`/per-channel `w_scale` values directly, sidestepping the
need to re-derive them from calibration statistics) gives the true
quantisation parameters. Computing `bias_channel`/`M_channel` from those plus
the known weights, then reading the real compiled table **directly at the
formula's predicted offsets, with no search**:

- `M`: byte-exact for all 128 channels, both builds.
- `bias`: within 1.6e-4 absolute of the table's real value for every channel,
  both builds -- the same tolerance class `conv_bias_requant.py`'s own
  formula already documents (order-of-summation float32 rounding, not a
  formula error).

**Structural cross-check against already-published data, no new build.**
`docs/axera-conv-weight-learn-wide.md` reported `Conv(256,256,3,3)`'s
scattered scaffold as exactly **512 runs**, found independently via
bit-permutation learning with no reference to tile geometry. This module's
formula predicts 16 tiles of 16 channels each; 16 tiles x 16 channels x 2
runs/channel (one for `bias`, one for `M`) = **512**, matching exactly.

## What is not verified here

- **No end-to-end emission was device-checked.** This module supplies the
  scaffold's placement and value formula (`emit_conv_table_tiled`); combined
  with the already-validated weight-code bit-permutation (`origin`/
  `emitter.emit_table`, unchanged), it should produce a complete table, but
  that combination was not run against a native Pulsar2 build or real
  hardware in this session. The two things it replaces (weight-code
  placement, scaffold value formulas) are each independently already
  validated elsewhere; the specific claim not yet checked is only that they
  compose correctly.
- **`Conv(512,512,3,3)` and the two `Conv(*,*,256->512,*)` shapes** --
  the actual shapes where prior bit-permutation learning was shown *wrong*,
  not just incomplete (`docs/axera-conv-weight-learn-256to512.md`) -- were
  not rebuilt and checked against this formula directly, only structurally
  corroborated via the 256/256 run-count match above. This is the highest-
  value thing to check next: does deterministic placement actually fix the
  256->512 1x1 shape's device fault, where 65 builds of statistical learning
  did not?
- **`K=1` shapes may have a header this formula doesn't account for.** The
  already-working 1x1 downsample's real scaffold offset is `block_at=9216`
  (`docs/axera-conv-weight-learn-downsample.md`), but this formula (with no
  header, tile 0 at byte 0, which matched perfectly for every `K=3` shape
  checked) predicts `8192` for the same `Cin*Cout` -- a 1024-byte
  discrepancy, exactly one scaffold block's worth. Whether 1x1 kernels
  always carry a header 3x3 kernels don't, or something else, is not
  resolved.
- **Cin's own tiling** (this project's README documents a separate ~16-wide
  input-channel tile) was not investigated for its effect on the *weight
  code* byte layout within a tile -- irrelevant to this module, since it only
  places the scaffold and defers weight-code bytes entirely to the
  already-validated `origin`/`emit_table` machinery, but worth noting as an
  open question for whoever extends this further.

## Reproduction

`tests/test_axera_conv_scaffold_arithmetic.py` checks the offset formula
against the two committed blind-build fixtures
(`scripts/axera/fixtures/conv_scaffold_arithmetic/`) and the 256/256
run-count cross-check; no Docker or device required.
