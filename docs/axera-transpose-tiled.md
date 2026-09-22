# AX650 tiled Transpose: the `npu_params` tile table is predictable, the MCode is not

Follow-up to [`axera-transpose-mcode.md`](axera-transpose-mcode.md), which found that
every Transpose in the ResNet18 training step is tiled (its `npu_params` grows past one
40-byte entry) and left the tile table undecoded. This note decodes it for part of the
space and tests the result on builds made after the rule was fixed.

**Result.** For a measured family of shapes the whole `npu_params` table of a standalone
float32 `Transpose` can be computed from the shape, byte for byte
(`scripts/axera/transpose_tiled_params.py`). **The MCode still cannot**, so there is no
emitter and no device run: a predicted table without its program is not a model. Of the
41 Transposes in the ResNet18 step, 9 have a predicted table; the other 32 (all 20
`[16,1,R,C]` activations, 11 of the weight transposes and `[512,1000]`) are outside the
validated domain and the predictor refuses them.

## Data

About 450 Pulsar2 7.0-lite builds (AX650, MinMax calibration, one uniform +/-0.9
calibration set) of a standalone one-node `Transpose`, in three families:

- `A`: `x[N,1,R,C]`, `perm=0,1,3,2` (activations),
- `B`: `x[1,R,C,K]`, `perm=0,2,1,3` (the `[1,C,C,9]` weight transposes: an `[R, C]` grid of
  `4*K`-byte elements),
- `C`: 2-D `x[R,C]`, `perm=1,0`.

The tables were read from `npu_params` of each `.axmodel`. Scratch:
`/home/takecheeze/npu-scratch/t_transpose_sweep` (earlier) and `t_transpose_tiled`.
`python scripts/axera/transpose_tiled_params.py check WORK_ROOT` compares the predictor with
every `A_`, `B_` and `C_` build under a root.

## The table

`npu_params` is always **five repetitions of one base period**. The base period is an
input-offset group followed by an output-offset group; every offset is a byte offset into
the input or output tensor.

**Untiled.** `e*R*C <= 65536` gives ten zero words, where `e` is the element size (4 for
`A` and `C`, `4*K` for `B`). In every alignment class where `R` or `C` is a multiple of 8,
the largest untiled build is at most 65,536 bytes and the smallest tiled one is at least
65,600 (`[1,1,1025,16]`), which confirms the 64 KiB guess of the earlier note there. Shapes
with neither `R` nor `C` a multiple of 8 stay untiled far past 64 KiB (81,204 bytes for
`101x201`) and then tile with a different table (see below).

**Column tiles.** For `C` a multiple of 8, the transpose is cut along `C` into
`k = ceil(C / tc)` column tiles of `tc = max(8, (C // 4) // 8 * 8)` columns. Tile `j`
reads at input offset `j*tc*e` and writes at output offset `j*tc*e*R`. For example
`[1,1,272,64]` has `tc = 16`, `k = 4`: input `0, 64, 128, 192`, output `0, 17408, 34816,
52224` (`16 * 4 * 272 = 17408`).

**Row chunks.** A tile of `rr` rows holds at most 131,072 bytes (32,768 elements of 4
bytes). When a whole column of `R` rows does not fit, the rows are cut into `m` chunks of
`ceil(R / m)` rows, which adds `i * ceil(R/m) * e * C` to the input offsets. Pulsar2's
search, as reproduced: for `tc` = `C/4` and then halved down to 8, try `m = 1, 2, 3` and take
the first that fits; only at 8 columns can `m` keep growing. Chunks can be uneven:
`[1,1,1300,256]` uses three chunks of 434, 434 and 432 rows (input offsets `i * 444416`).

**Order.** Each group is written in the iteration order of a **CPython `set`** of its
offsets, built by inserting them column-major (`j` outer, `i` inner). Offsets that are all
multiples of the table size collide in the hash table and so keep insertion order; others
land in slot order. This is what made the tables look non-monotonic; it reproduced the order
of all 70 single-level groups I checked and every two-level input group. Sorted insertion
matches only single-level tables. It depends on CPython's integer-hash set layout, which is
stable across 3.x (the tests pass on 3.12 and 3.14; 3.11 and 3.13 were not available
offline).

**Unaligned C, tiles along R.** For `C` not a multiple of 8 and `R` a multiple of 8, small
shapes tile along `R`: `tr = max(8, (R // 4) // 8 * 8)` rows per tile, input offset
`i*tr*4*C`, output offset `i*tr*4`. This holds while `tr * C <= 15680`, checked at
`C = 49, 133, 164, 196, 300, 421, 561, 612, 644`; above that Pulsar2 tiles along `C` (below).

**Wide elements (`B` family).** The offsets use the real element size, and the tile cap
counts each element padded to a power of two (36 bytes counts as 64). With that,
`[1,64,64,9]` gives `tc = 16`, `k = 4`, input offsets `j * 576`, output offsets
`j * 36864`, which is exact. But it only holds when no row chunking is needed (below).

## Validation

Fitting and testing were kept apart with fresh random shape sets. "Blind" means built after
the rule it tests was written down; a rule change after a failure is noted, and the set is
then no longer blind for that rule.

| set | shapes | rule status when built | exact | wrong | refused |
| --- | --- | --- | --- | --- | --- |
| `cases_blind` | 24 `[1,1,R,C]`, 65-400 KB | single-level rule (C aligned: tile C, else tile R) | 20 | 0 | 4 |
| `cases_blind2` | 22, 0.5-3.4 MB, C a power of two | first two-level rule | 20 | 0 | 2 |
| `cases_blind3` | 26: 10 two-level, 8 single, 8 R-tiled | **frozen** | **26** | **0** | **0** |
| `cases_blindB` | 21 `B` shapes | padded-cap rule | 10 | 0 | 11 |
| `cases_blindB2` | 16 `B` shapes | rule restricted after `blindB` | 4 | 0 | 12 |

Read the table carefully:

- The only clean blind result is `cases_blind3`: 26 of 26 exact, all tiled, on shapes the
  predictor accepts (it never has to guess: the set was drawn from shapes it does not refuse).
- `cases_blind` is not blind for the R-tiled branch: its limits (`C <= 644`, 15,680
  elements) were set after that set's 4 misses, which are now refusals. `cases_blind2` failed once (`1300x256`: Pulsar2 chose `m = 3` with
  64-column tiles where I predicted `m = 2` with 32-column tiles), which changed the search
  order to the one above; the 20 exact were re-scored under it.
- The `B` sets are where the rule broke. Of the 45 wide-element plans I could recover from
  builds, the padded-cap rule disagreed on 5 (Pulsar2 used fewer chunks than predicted:
  `[1,364,64,9]`, `[1,364,32,9]`, `[1,375,32,9]`, `[1,427,128,9]`, `[1,608,512,9]`). A search over caps
  (real bytes, padded bytes, 64 KiB to 256 KiB), chunk limits and preference orders found no
  simple rule that fits all 147 recovered plans; the padded-cap rule was best (142 of 147).
  So the predictor accepts wide elements **only when one chunk fits**, and `blindB2` is
  the check of that restriction (4 exact, 0 wrong; the 12 refusals are two-level shapes).
  The restriction was made after seeing `blindB`, so its evidence is `blindB2` alone.
- Overall: 443 builds in the two scratch roots, **0 wrong**, 368 exact (151 of them tiled
  tables, the rest untiled zero tables), 75 refused.

## What is not decoded

- **Unaligned C with tiles along C.** `[1,1,1536,49]` tiles 7 column tiles of 8, but
  `[1,1,1280,49]` tiles 4 row tiles of 320. The switch is consistent with a row tile of about
  15,680 elements (`320*49` R-tiled, `384*49 = 18816` C-tiled; `48*164` R, `96*164 = 15744`
  C; `32*196` R, `128*196` C) but `[1,1,72,969]` violates it (row tile 15,504, yet C-tiled),
  and the C-tile size is sometimes not `(C // 4) // 8 * 8`: `[1,1,128,774]` uses 160 columns
  where that gives 192, and 2-D `[512,1000]` uses 104-column tiles (10 tiles, two chunks)
  where it gives 248. Predictions there are refused.
- **`m >= 4` row chunks.** `[1,1,12800,64]` has a 16-word output group (offsets
  `j*409600 + {0, 25600}`) where three chunks have 8; not decoded.
- **Both `R` and `C` unaligned, above the (higher) threshold.** They tile with a different
  table: `[1,1,201,301]` is `(0, 121604, 0)` repeated five times (a plain two-way row split
  at row 101), and `[1,1,301,401]` has five 96-column tiles. Untiled up to 84,000 bytes
  (`70x300`) and tiled by 242,004 (`201x301`); not bracketed further.
- **Batch `N > 1`, i.e. the ResNet18 activations.** A different table form. Observed for
  `[N,1,256,49]`: N=2 has two groups of 4 (input unit half a batch, output unit 512), N=4 and
  N=6 a single group of N offsets (one batch each), N=8 a single group of 4 (two batches
  each), N=12 groups of 8 and 4 (one batch and three batches), N=16 groups of 8 and 4 (two and
  four batches). `[16,1,128,196]` is 16 + 8, `[16,1,64,784]` is 32 + 16 (half batches). The
  full-size ones (`[16,1,4608,49]`: 208 + 112 words per period, `[16,1,576,3136]`: 1,056 +
  528) are not decoded.

## The MCode is not predictable

Within one tile plan the MCode is still not a template plus a few immediates. For shapes
with the same axis, tile size, tile count, chunk count and `C`, and the same segment sizes,
the bytes that differ between shapes are (of 5 segments):

| plan `(tc, k, m)`, `C` | shapes | segment sizes | varying bytes per segment |
| --- | --- | --- | --- |
| `(16, 4, 1)`, 64 | 8 | 64, 32, 800, 512, 736 | 8, 0, 188, 374, 16 |
| `(32, 4, 1)`, 128 | 7 | 64, 32, 800, 512, 736 | 8, 0, 191, 387, 24 |
| `(64, 4, 1)`, 256 | 3 | 64, 32, 800, 512, 736 | 6, 0, 146, 340, 17 |
| `(8, 4, 2)`, 32 | 5 | 64, 32, 832, 416, 960 | 7, 0, 175, 171, 37 |
| `(16, 4, 2)`, 64 | 4 | 64, 32, 832, 416, 960 | 8, 0, 182, 109, 30 |
| `(16, 8, 2)`, 128 | 2 | 64, 32, 928, 512, 1536 | 5, 0, 75, 144, 17 |

Only segment 1 (the 32-byte one) is constant. Segments 2 to 4 vary in between about 8% and
73% of their bytes across shapes that share a plan (the per-tile segments most), and the
segment sizes themselves change between neighbouring shapes with the same plan (for
`C = 64` the MCode is 2,856 or 2,888 bytes depending on `R`). Rebuilds of the same shape
change only the 301-325 noise window, so the variation is real dependence on `R`. This is
the same picture as the untiled program ([`axera-transpose-mcode.md`](axera-transpose-mcode.md)
third pass): shape-dependent immediates re-flow a variable-length stream, and no
segment-level template survives a change of `R`. A byte-exact generator would need the
instruction encoder itself.

## Consequences for the generator

- Nothing new is emitted. The table predictor is a characterization result, not an
  emitter: an `.axmodel` needs both the table and the MCode, and only the table is known.
- It does establish that the tile table is **derived, not allocator noise**: earlier notes
  called the offset order "non-monotonic, allocator-like". It is the hash-order of a Python
  set, and the tile count and sizes follow a capacity rule. That removes one of the two
  unknowns a Transpose generator would have needed.
- Coverage of the ResNet18 step: 9 of 41 Transposes have a predicted table (four
  `[1,64,64,9]`, `[1,512,256,1]`, `[1,256,128,1]`, `[1,128,64,9]`, `[1,128,64,1]`, and the
  untiled `[16,512]`). The 32 others: 20 batch-16 activations; 11 weight transposes
  (`[1,128,128,9]` x3, `[1,256,256,9]` x3, `[1,512,512,9]` x3, `[1,512,256,9]`,
  `[1,256,128,9]`), whose measured tables the padded-cap rule does reproduce but which I
  refuse because that rule is not trustworthy for two-level wide elements; and `[512,1000]`.
- Next, in order of value: (1) the batch form, since it covers the 20 activation
  Transposes and may be the same construction with the batch as another split axis (not
  tested);
  (2) the wide-element chunk rule (needs a denser `B` sweep at fixed `K`); (3) the
  unaligned-C axis switch; (4) only then the MCode, which needs the encoder.

## What was not tested

Element types other than float32, Transposes next to other ops (a consumer changes the
MCode, and here the table too may differ), permutations other than the three families, and
any device run (nothing is emitted). The CPython set-order dependence was not checked on
the Python that Pulsar2's container runs; it is inferred from 70 matching groups.
