# Fixing the two Conv shapes bit-permutation learning got wrong, with the tiled scaffold formula

`docs/axera-conv-weight-learn-256to512.md` (PR #1777) found the ResNet18
stage-3-to-4 downsample pair -- `Conv(x[16,256,14,14], w[512,256,1,1], stride 2)`
and `Conv(x[16,256,14,14], w[512,256,3,3], stride 2)` -- has a byte-exact
weight-code region but a requantisation "scaffold" that a second bit-permutation
learning pass never fully resolved: the 3x3 shape's emitted output was
device-confirmed **wrong** in 17 of 512 channels (up to 11.8 error), and the 1x1
shape's emitted holdout **faulted the AX8850 runtime** outright.
`docs/axera-conv-scaffold-arithmetic.md` (PR #1784) found the real placement
mechanism behind this scaffold -- a deterministic per-output-channel-tile layout,
not something that needs learning -- but explicitly left these two shapes
"structurally corroborated" only, not rebuilt and device-checked. This is that
follow-through, and it finds two real layout facts #1784's validated case
(`Conv(128,128,3,3)`) didn't need.

## Fact 1: K=1 has its own tile geometry

`conv_scaffold_arithmetic.py`'s formula (`tile_width = budget // (Cin*K*K)`,
scaffold immediately after weight codes, zero gap) is validated only for `K=3`.
For `K=1`, applying it directly is wrong. Searching a real compiled
`Conv(512,256,1,1)` table for the exact bytes of a computed `M`/`bias` value
(the same direct-search technique #1784 used) finds the true tile width is
**128** output channels -- not the budget-derived 144 -- and the scaffold
starts, from that tile's own weight-code start, at:

```
scaffold_start = weight_bytes_per_tile * 9 // 8
```

This is confirmed exactly, to the byte, at **two** shapes: the already-working
`Conv(128,64,1,1)` (`Cin=64`: `8192*9//8 == 9216`, matching
`docs/axera-conv-weight-learn-downsample.md`'s independently-reported
`block_at=9216`) and this new `Conv(512,256,1,1)` (`Cin=256`:
`32768*9//8 == 36864`), across 3 independent builds each with different random
weights. `128` as the K=1 tile-width cap, independent of `Cin`, is itself
notable -- it's the same width the already-working `Conv(128,64,1,1)` case uses
for its single tile.

Why `*9/8`: not resolved here. It's clean and exact, not approximate, so it's
likely a real encoding property (perhaps a reserved 1-in-9 byte lane), not
padding-to-a-round-number.

## Fact 2: Cin > 128 duplicates the scaffold

For `Conv(512,256,3,3)`, direct byte-search for `M`/`bias` values finds each
computed value at **two** locations in the real table, 37,120 bytes apart.
The two 128-wide input-channel sub-tiles (`Cin=256` splits at the same 128 cap
Fact 1 shows for `K=1`) each carry their **own copy** of the identical
per-output-channel `(bias, M)` block -- plausibly because a per-input-group
partial sum gets its own requantisation step before the two groups' partials
are combined. `origin`/`emit_table`'s bit-permutation map correctly leaves
both copies `CONST` (neither is weight-code data), so writing only one copy --
what the earlier, non-tile-aware scaffold learning implicitly did -- silently
leaves the other holding stale reference-build values. **This is the direct
cause of the 3x3 shape's 17-bad-channel failure**: those channels' errors
came from whichever of the two copies the old approach happened not to touch
for that particular channel.

The real per-32-output-channel tile is `74,240` bytes: two `37,120`-byte
input-channel-group halves, each `36,864` bytes of weight code (`32 channels *
128 Cin * 9` -- the same per-tile weight-code size the already-validated
`Conv(128,128,3,3)` uses for its own, non-split, 32-channel tile) plus `256`
bytes of scaffold (`bias(32) + M(32)`, `4` bytes each).

## Verification

**Placement and values, direct against real compiled tables** (3 builds for
1x1, 2 for 3x3, independent random weights each): `M` byte-exact for every
channel, `bias` within `1.2e-4`-`3.7e-4` absolute -- the same rounding-tolerance
class `conv_bias_requant.py`'s formula documents everywhere else in this
project.

**A held-out weight set, emitted and diffed against a real native rebuild**:
zero `npu_params` bytes differ outside the scaffold region for either shape
(the weight-code region was already known byte-exact); the ~500-1,000
remaining differing bytes are inside the scaffold, at the expected bias
rounding tolerance.

**On the AX8850** (`axcl-vm`, serialized under the shared lock, control run
before and health run after every variant, all clean):

| shape | before this fix (PR #1777) | after |
| --- | --- | --- |
| `Conv(512,256,3,3)` | 17 channels wrong, up to 11.8 error, mean 0.22 | **0 channels over 1.0 error**, max 0.153, mean 0.048 (vs. a 0.014-0.015 control/native baseline -- close to, but not exactly at, noise floor) |
| `Conv(512,256,1,1)` | emitted holdout **faulted the runtime** (`0x8030070C`) | scaffold-only (mcode scale literals left at the reference's own values) emission runs cleanly, giving a bounded 0.11-0.17 error consistent with the stale `y_scale`; the fully scale-patched emission still faults -- see below |

## The remaining 1x1 fault is a separate, pre-existing limitation, not this fix

Isolating the fault: the scaffold-only emission (correct `npu_params`, but
`scripts/axera/patch_scales.py` not yet applied, so the mcode still declares
the *reference* build's own `x_scale`/`y_scale`) **runs cleanly** on the
device. Only the additional `patch_scales.py` step -- which patches scale
*literals* inside the mcode instruction stream, a completely separate region
from `npu_params` -- causes the fault. Diffing the scale-patched mcode against
a real native rebuild's own mcode finds **1,548 differing bytes**, clustered
from offset ~608, far beyond `patch_scales.py`'s own documented scope (a
handful of scale slots) or the project's known ~25-byte compiler-noise window.
`mcode.check()` reports the patched mcode as structurally clean, consistent
with this project's established understanding that `0x8030070C` is a runtime
validity check, not something the static checker catches.

`patch_scales.py`'s own docstring already flags this precisely: "Do not train
on a patched artifact that has not run cleanly on device at least once per
graph shape." `Conv(512,256,1,1)` is the first shape in this whole series
where that check was actually run, and it fails -- Pulsar2's real scheduling
for this specific (widest-channel, 1x1, stride-2) shape apparently diverges
between same-shape builds by more than scale literals alone, in a way
`patch_scales.py`'s slot-based patch cannot reach. This is a real, precisely
diagnosed limitation in that separate tool, not a gap in the scaffold fix this
document is about.

## What this changes

- **Both 256->512 shapes' `npu_params` scaffold is now correct and
  device-confirmed**, closing the actual bug (wrong values, not just an
  incomplete resolution percentage) PR #1777 found.
- **`Conv(512,256,3,3)` now has a working, device-verified full pipeline**
  (weight codes + scaffold + mcode scale patch), joining the stem, `64/64`,
  and the `1x1` `64->128` downsample as shapes with complete coverage.
- **`Conv(512,256,1,1)`'s scaffold is fixed and verified**, but the shape's
  full pipeline is still blocked -- now by a specific, isolated limitation in
  `patch_scales.py`'s mcode patching, not an unknown.
- **`conv_scaffold_arithmetic.py`'s formula needs the two facts above to
  reach `K=1` or `Cin>128` shapes.** Any future shape with `Cin>128` should be
  checked for the same duplicate-copy structure directly (byte-search for a
  known value, don't assume single-copy); any `K=1` shape should use the
  `*9/8` tile geometry from Fact 1, not the `K=3` budget formula.

## Reproduction

`scripts/axera/conv_256to512_tiled_fix.py`'s `emit_holdout_table(prefix,
reference_table, origin, w, b, x_scale, x_zero, y_scale, y_zero)` implements
both facts for `prefix in ("c1x1", "c3x3")`. `tests/test_axera_conv_256to512_tiled_fix.py`
reproduces both shapes' held-out emission against a committed oracle
(`scripts/axera/fixtures/conv_256to512_tiled_fix/`) with no Docker or device
required. The raw build campaign and device-check scripts used above are
scratch, under `/home/takecheeze/npu-scratch/t_conv_learn_256to512` (the
original campaign, reused read-only) and
`/home/takecheeze/npu-scratch/t_conv_256to512_tiled_fix` (this fix's own
emitted artifacts), not committed.
