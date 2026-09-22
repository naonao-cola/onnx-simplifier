# Isolating calibration from shape in the `teng2` sweep: the confound is real but small, and doesn't explain the opacity

Five prior `teng2` decode attempts (`docs/axera-dma-queue.md`,
`docs/axera-teng2-sqrt-blocks.md`, `docs/axera-teng2-tiled-repeat.md`,
`docs/axera-teng2-add-two-input.md`) all swept *shape* while calibrating each
build from `rng = np.random.RandomState(0)` (or `(0), (1)` for a second
input) drawing `rng.uniform(-0.9, 0.9, shape)`. Because the array size
depends on `shape`, a fixed seed still produces a different observed
min/max at every shape -- a larger draw from the same uniform distribution
gets closer to the true `±0.9` bound. `docs/axera-teng2-add-two-input.md`
(PR #1756) separately proved `teng2`-adjacent regions embed
calibration-derived Q15 literals directly. So shape and calibration range
were confounded in every prior sweep, just not for the "random seed"
reason -- for a "sample-count-dependent extremum" reason. This isolates the
two variables for the first time, to see whether removing the confound
reveals the template earlier sweeps missed.

Method: `scripts/axera/teng2_calibration_isolation.py` builds a standalone
float32 `Relu`'s calibration array by drawing the interior uniformly at
random (fixed seed, for reproducibility) and then overwriting element 0 and
element 1 with an *exact*, chosen `(lo, hi)` bound -- so the resulting
`y_scale`/`y_zero` is a function of the chosen bounds alone, not of the
array's size. Pulsar2 7.0-lite, AX650, MinMax calibration throughout.

## Part 1: calibration-only sweep, shape fixed at `[1,64,56,56]`

Four builds of the identical shape, varying only the pinned bounds:

| bounds | vs. `(-1,1)` | differing bytes (segment 2, 1408 B, outside the 301-325 noise window) |
| --- | --- | --- |
| `(-0.5, 0.5)` | | 16 |
| `(-1.0, 1.0)` | (reference) | -- |
| `(-2.0, 2.0)` | | 8 |
| `(-0.1, 0.9)` (asymmetric) | | **678** |

**Symmetric range changes touch a small, clean, positionally-repeated set of
bytes**: 8 two-byte fields at offsets `698, 706, 714, 722` and
`1201, 1209, 1217, 1225` -- two clusters of 4, each spaced 8 bytes apart,
matching this shape's 4-entry DMA tile count (`dma_tile_predict.py`), i.e.
one field per channel-tile.

**Asymmetric range changes almost everything**: 678 of 1408 bytes, spanning
nearly the whole segment. Symmetric ranges keep `y_zero = 0`; an asymmetric
range needs a nonzero zero-point, and that appears to select a structurally
different compute path through the whole segment, not just different
literal values within one path. This is a real, previously unknown
"symmetric vs. asymmetric is a different program, not a different constant"
finding, and by itself explains why prior sweeps (which never controlled
this) saw such large, inconsistent diffs -- but see Part 2: it is not the
dominant effect once shape also varies.

### The confound is decodable, log-linear in the range, not fully solved

Reading the two clusters' 4 repeated 16-bit values as little-endian at each
bound:

| bounds | cluster 1 (register-like value, all 4 tiles) | cluster 2 |
| --- | --- | --- |
| `(-0.5, 0.5)` | 17278 | 15232 |
| `(-1.0, 1.0)` | 17150 | 15360 |
| `(-2.0, 2.0)` | 17022 | 15488 |

Each doubling of `hi` (0.5→1.0→2.0) moves cluster 1 by exactly **-128** and
cluster 2 by exactly **+128** -- linear in `log2(hi)`, not in `hi` or
`1/hi` as a plain Q15 scale ratio would be. The two clusters' sum is
constant across all three bounds: `17278+15232 = 17150+15360 = 17022+15488
= 32510`. This looks like a biased-exponent or shift+mantissa style scale
representation (common in fixed-function requantizers) rather than the
direct `round(scale_ratio * 32768)` Q15 form PR #1756 found for `Add`'s
two-input header -- but the exact bit-field semantics (what the additive
constants 17150/15360 and the invariant sum 32510 represent physically)
were not further decoded here; this is a lead for future work, not a closed
formula.

## Part 2: shape-only sweep, calibration exactly pinned at `(-1.0, 1.0)`

Nine builds of `Relu(x[1,C,56,56])`, `C` a multiple of 8 from 32 to 96,
every one calibrated with the identical pinned bounds (only the array's
filler content differs, which does not affect MinMax's observed min/max):

| pair | differing bytes (segment 2, outside noise window) |
| --- | --- |
| C=32 vs 40 | 513 |
| C=40 vs 48 | 795 |
| C=48 vs 56 | 493 |
| C=56 vs 64 | 291 |
| C=64 vs 72 | length differs (1408 vs 1376) |
| C=72 vs 80 | length differs (1376 vs 1408) |
| C=80 vs 88 | 49 |
| C=88 vs 96 | 15 |
| C=32 vs 96 (endpoints) | 714 |

**This is the answer to the directive's central question: no, isolating
calibration does not reveal a materially cleaner signal.** These numbers
(15-795 bytes differing per adjacent step) are the same magnitude as the
confounded sweep in `docs/axera-dma-queue.md` reported for the same shape
family (15-799 bytes). The confound identified in Part 1 explains at most
16 bytes of any given diff -- a small fraction of the 291-795 bytes shape
alone produces. Shape's own effect on `teng2` dominates completely and
remains exactly as irregular and un-templated as every prior sweep found:
no consistent block structure, no clean small-diff neighbour pairs beyond
the two already known (`C=88` vs `96`, `C=80` vs `88`), both of which
reproduce here unchanged.

One secondary, smaller effect is real: `C=72`'s segment length (1376 B, the
minority form) is unchanged by pinning calibration, so length/form
selection at a borderline shape is a property of the shape alone here, not
something the earlier sweep's calibration noise introduced.

## Part 3 (cross-check): not done

Directly comparing a jointly-varying (shape and calibration both change, as
every prior sweep did) build against the union of Parts 1 and 2's
explained bytes was not attempted -- Part 2 already answers the directive's
core question on its own (the confound is not the reason `teng2` looked
opaque), and the marginal value of confirming that with a third sweep did
not seem to justify the additional build time in a single-shot task.

## Conclusion

The calibration/shape confound in prior `teng2` sweeps is real (proven
directly here, not assumed) and it is genuinely interesting -- the
symmetric/asymmetric split is a new, previously unreported structural
finding, and the log-linear paired-field encoding is a concrete lead. But
it is **not** what made `teng2` resist decoding: with calibration
perfectly controlled, shape alone still produces 15-795-byte diffs between
adjacent shapes with no consistent block structure. `teng2` remains
opaque for the same reason the five prior attempts found -- it needs a
real instruction-level model of the encoder, not a cleaner sweep.

## Reproduction

```
scripts/axera/teng2_calibration_isolation.py build CASES.json WORK_ROOT
scripts/axera/teng2_calibration_isolation.py diff WORK_ROOT NAME_A NAME_B
```

No device work was done: nothing here reached a validated predictor or
emitter to run.
