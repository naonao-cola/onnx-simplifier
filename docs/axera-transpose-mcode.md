# AX650 standalone Transpose: is the MCode predictable from shape?

**Answer: not to the byte, so there is no emitter.** The shape dependence is
deterministic and partly decoded, but reproducing a Pulsar2 build from
`(shape, perm)` alone would need the instruction encoder, and that is not
reverse-engineered. Nothing here is a validated predictor, and no device run
was made because nothing was emitted. What follows is what the sweep did and
did not establish.

Method: 124 Pulsar2 7.0-lite (AX650, MinMax calibration, one uniform +/-0.9
calibration set) builds of a standalone float32 `Transpose`, scratch
`/home/takecheeze/npu-scratch/t_transpose_sweep`. Reproduce with
`scripts/axera/transpose_sweep.py` (`build CASES.json ROOT`, then `table ROOT`).
Families, from the ResNet18 training step: swap the last two axes of `[N,1,R,C]`
(perm `0,1,3,2`), swap axes 1 and 2 of `[1,C,C,K]` (perm `0,2,1,3`), and 2-D
perm `1,0`. Sizes run from 8x8 up to `[16,1,576,3136]` and `[1,512,512,9]`.

## What is deterministic

- **Same shape, same program.** Three builds of each of four shapes had
  identical `npu_params` and MCode that differs only inside the known 301-325
  compiler-noise window (0-8 bytes; one pair was fully identical). So a
  shape-to-MCode map exists; the question is whether it is computable.
- **No data to retarget.** Below about 64 KiB the `npu_params` table is a single
  40-byte all-zero entry. Above it, it holds 40-byte entries (always a multiple
  of 40) that are cyclic rotations of tile byte offsets (multiples of the shape
  strides). All four builds that were rebuilt had identical tables, so rotation
  order was stable here, unlike the MatMul/Gemm tables.

## Layout of the MCode: what is constant and what is not

MCode = 284-byte head, five segments listed in the FlatBuffers tail
(`mcode.segments`), and a tail. For every small shape (one 40-byte params
entry), the segment sizes are `[64, 32, S2, 32, S4]`.

| part | across 70 grid shapes (R,C in 8..64) |
| --- | --- |
| head, bytes 0-283 | identical when S2 is the same size; 9-13 bytes differ when S2 changes size (segment word counts) |
| segment 0 (64 B) | noise window only (2-8 bytes) |
| segments 1 and 3 (32 B each) | identical for every untiled shape checked (grid, `13x17`, `[1,16,16,9]`, 2-D `[16,512]`); segment 3 grows once tiled |
| segment 4 (512-576 B) | 3-9 bytes differ; length is 512 or 544 (576 for one `[1,C,C,9]` case) |
| **segment 2 (bulk program)** | **all 70 shapes have a different program** |

So only segment 2 (and the two length fields that describe it) carries the
shape. Its first 507 bytes are shared by every small shape; the differences
begin at byte 507 to 569 and run to the end of the segment.

## Where the shape appears

Varying one dimension at a time (R at C=32, C at R=16) shows single bytes of
segment 2 that are affine in the shape:

| offset in segment 2 | value (checked by varying R and C separately) |
| --- | --- |
| 549 | 4C (row bytes: 96, 128, 160, 192, 224 for C = 24..56) |
| 569 | R-1 (0x0f, 0x17, 0x27, 0x2f, 0x37, 0x3f for R = 16..64) |
| 619 and 631 | R*C/8 - 1 (63, 79, 95 at R=16 for C = 32, 40, 48; 159 at R=40; 255 at R=64) |
| 668 | R*C/64 |
| 673 | R*C/8 |
| 682 | R*C/256 |

These are natural quantities (row byte stride, index of the last row, counts of
8-, 64- and 256-element groups). Values of 128 or more are written differently
(`0x7f 83` for 127 versus `0xff 82` for 255), which is a variable-length coding
of the immediate. It is also why a value crossing a threshold inserts or removes
a byte and shifts everything after it. For example `R=40` versus `R=56`
re-encodes `81 66 81` as `06 84`, one byte shorter, and the rest of the segment
moves by one byte.

## Why the length looks irregular, and why that is not a search heuristic

Segment sizes are multiples of 32, so the stream is padded. The length only
changes when the exact byte count crosses a 32-byte boundary, which depends on
the immediates' encoded widths. That makes the length a step function of
value-dependent encodings, not a smooth function of R and C.

MCode length for `[1,1,R,C]`, perm `0,1,3,2`, R rows and C columns:

```
        C=8   16    24    32    40    48    56    64
R=8    2080  2112  2112  2112  2112  2112  2112  2112
R=16   2112  2144  2112  2112  2112  2112  2112  2112
R=24   2112  2112  2112  2112  2112  2112  2112  2112
R=32   2112  2112  2112  2112  2112  2112  2112  2112
R=40   2080  2112  2112  2112  2112  2112  2112  2112
R=48   2080  2112  2112  2112  2112  2144  2112  2112
R=56   2080  2112  2112  2112  2112  2112  2144  2112
R=64   2112  2112  2112  2112  2144  2144  2144  2112
```

Non-multiples of 8 cost more (`13x17` and `17x13`: 2240, `12x20` and `20x12`:
2272, `49x9` and `9x49`: 2240). The length is not symmetric in R and C:
`64x40` is 2144 but `40x64` is 2112, and `48x8` is 2080 but `8x48` is 2112.
Several non-square pairs do match, and equal length never meant equal bytes.

## Regime change with tensor size

A tiled program (segment 3 grows from 32 B, `npu_params` grows past 40 B)
appears once the tensor is large enough. For `[1,1,R,C]`: `192x49` (37,632 B)
and `256x49` and `128x98` (50,176 B) are untiled; `384x49` (75,264 B) and
`128x196` (100,352 B) are tiled. The threshold is between 50,176 and 75,264 B;
it was not bracketed further and 64 KiB is only a guess.

| shape | MCode | `npu_params` | segments |
| --- | --- | --- | --- |
| `[1,1,128,196]` | 3048 | 160 | 64,32,896,544,768 |
| `[1,1,384,49]`, `[1,1,512,49]` | 3048 | 160 | 64,32,896,544,768 |
| `[1,1,128,392]`, `[1,1,128,784]`, `[1,1,64,784]` | 3144 | 200 | 64,32,832,480,992 |
| `[16,1,256,49]` | 3336 | 240 | 64,32,928,448,1120 |
| `[16,1,64,784]` | 4904 | 960 | 64,32,1088,832,2144 |
| `[16,1,4608,49]` | 29,000 | 6400 | 64,32,3008,3072,22080 |
| `[16,1,2304,196]` | 54,856 | 12,160 | 64,32,6752,6848,40416 |
| `[16,1,1152,784]` | 80,712 | 18,880 | 64,32,7136,10048,62688 |
| `[16,1,576,3136]` | 132,104 | 31,680 | 64,32,9120,16608,105536 |
| `[16,1,147,12544]` | 139,688 | 30,720 | 64,32,9824,14880,114144 |
| `[1,512,512,9]` (perm 0,2,1,3) | 12,808 | 3840 | 64,32,1984,2400,7584 |
| `[512,1000]` (perm 1,0) | 4008 | 580 | 64,32,992,640,1536 |

At training scale the MCode is 29-140 KB and the tile table 6-32 KB: for the
five largest rows, 167-182 bytes of MCode per 40-byte table entry, not a
constant (and the small tiled shapes are far off that ratio). `[128,196]`,
`[384,49]` and `[512,49]` are three different shapes with the same segment
sizes and MCode length, and `[128,392]`, `[128,784]` and `[64,784]` are three
more. Same length does not mean same bytes.

## What would be needed for a generator

1. The instruction encoder for segment 2 (and the tiled segments 3-4): the
   variable-length forms for the immediates, and the rules for the extra
   instructions that appear at the small-shape thresholds. The `mcode.py`
   tokenizer parses the stream but does not decode what a verb computes.
2. The tile decomposition that decides the number of table entries and their
   cyclic order above the size threshold.
3. A per-shape oracle comparison on held-out shapes, then a device run; this
   sweep is the training set for that, not the validation.

## What was not tested

- Only float32, one calibration set, one perm per family.
- Transposes with a consumer: earlier probes showed a `Relu` consumer changes
  the MCode by 72 bytes, so a standalone template would not drop into a real
  training step even if the standalone case were solved.
- The exact size threshold between untiled and tiled programs, and shapes that
  are not multiples of 8 beyond the few listed.
- Any device execution. A standalone Transpose template would be a lookup of a
  Pulsar2 build, not a generator, so it was not made.

The enrichment check below is a weak signal and is included only for
completeness: a shape's own six formula values (R-1, 4C, R*C/8-1, R*C/64,
R*C/8, R*C/256, when below 256) appear as bytes in the suffix of its own
segment 2 for 90% of the 64 multiple-of-8 grid shapes, versus 61% for other
shapes' values. Bytes 0-255 are dense in a 200-byte suffix, so this is much less
convincing than the controlled one-dimension variations above.

## Second pass: decode attempt (what held and what did not)

Re-examined the 76 untiled `[1,1,R,C]` builds in `t_transpose_sweep` with the
`mcode.decode` codec and raw byte diffs. No predictor came out of it; these are
corrections and measurements for whoever continues.

**The fixed-offset formulas do not generalize.** The table above (offsets 549,
569, 619/631, 668, 673, 682) was read off two one-dimensional sweeps (R at
C=32, C at R=16). Checking all seven offsets at once against the six formulas
holds for 3 of the 76 untiled shapes. Read those offsets as "where the field
sits in that neighbourhood", not as a layout.

**What is solid, at R=16 with C varying.** The immediates are directly visible
as single bytes: `01 02 X 83 92` with X = 4C (0x60, 0x80, 0xa0, 0xc0, 0xe0 for
C = 24..56), `0c X 83 14` and `0c X 83 62` with X = R*C/8-1 (0x2f..0x7f), and
`88 7c 00 X 82 1a` with X = C/4 for C = 32, 40, 56, 64.

**Two things that need a real encoder.**
- *Discrete micro-program switches.* C/4 is written `00 X 82 1a` for C = 32, 40,
  56 and 64, but as `82 56 02` (C=24) and `82 98 02` (C=48). Both are C/8
  divisible by 3. The trigger is not established.
- *Value-dependent form changes that insert or remove a byte.* `16x32 -> 16x40`
  changes 27 bytes: five immediates, then from byte 692 on the same
  instructions shifted by one, because 3 bytes (`82 8c 8a`) become 4
  (`07 83 3c 88`). Equal-length blobs therefore differ by anywhere from 27 to
  143+ bytes for the same R and near C, and `16x40 -> 16x56` shifts the whole
  stream from byte 572.

**Cluster count says "not a few templates".** Grouping the 76 shapes whose
segment-2 streams have equal length and differ in at most 14 bytes gives 61
clusters. The largest has five shapes (16x32, 24x40, 24x48, 40x16, 40x32); 47
clusters are singletons. So a template-plus-immediates model would need dozens
of templates, and the choice among them is the unknown encoder.

**The tokenizer is not reliable enough on these segments to align by record.**
A record-level diff (`mcode.decode(..., **FULL_RULE)`, `difflib` on record
signatures) of `16x16` against `16x24` splits the same bytes into different
forms and emits raw bytes and B/S units that do not line up. A byte diff is no
worse. A usable alignment needs a grammar validated on these segments first.

**Tiled regime (training sizes) is more regular, but only partly.**
- Segment 4 has high self-similarity at a fixed period: 184 bytes for
  `[16,1,4608,49]` and `[16,1,2304,196]` (64% and 69% of bytes equal to the byte
  one period later), and 1064 for `[16,1,576,3136]` (73%), with about 54% zero
  bytes. That fits repeated per-tile blocks with varying immediates.
- The `npu_params` tile offsets are not monotonic. The first entry of
  `[16,1,4608,49]` is `0, 451584, 1354752, 903168, 1806336, 2257920, 4515840,
  4064256, 2709504, 3612672` (multiples of 451584 = 4608*49*2). The order looks
  like an allocator's, so even with the block structure decoded, the offset
  order would need its own model. Rebuilds of four shapes were identical here,
  but only four were tried.

**Next steps if this is continued.**
1. A one-dimensional sweep of every R (or C) from 8 to 64 in steps of 1 at a
   fixed other dimension, cataloguing each form change against the value that
   caused it (is it a threshold, a divisibility rule, or both?).
2. Build the grammar for segment 2 from those catalogues instead of the
   tokenizer's forms, and validate it by round-tripping every sweep build.
3. Only then attempt the tiled segments, starting from the 184-byte block.

## Third pass: the step-1 sweep (alignment classes and the size immediate)

Every `R` and every `C` from 8 to 64 at a fixed other dimension (`C=32` and
`R=16`), 113 builds, reproduced by `scripts/axera/transpose_decode.py` from the
sweep root. This corrects one claim above and pins down several fields.

**Alignment to 8 is the first-order structure, not value thresholds.** With `R=16`
and `C` varying, the untiled segment-2 content is 800 bytes for every `C` that is
not a multiple of 8 and 736 for multiples of 8 (768 at `C=16`). Between
neighbouring unaligned `C` values only 1-2 bytes change, with no insertions or
deletions, except at the block edges (`C = 8k+1`) where a whole new form
appears (`ins` of 60-90 bytes), and at `C=12` (one-byte shift). With `C=32` and
`R` varying, multiples of 8 give 736 bytes and other `R` give 768 (`R=55` is the one
exception, at 736), with small form changes (3-byte inserts) between neighbours. So the rule that a
"shape change re-flows the stream" is really a per-block change: streams are
stable inside a block of 8 and change form at block edges.

**A 16-bit size immediate, by alignment class.** Every untiled stream holds the
tensor's byte size as a little-endian uint16, `4*R*C - 1` or `4*R*C` depending on
alignment (only tested where the value is below 65536):

| C | R | field | present |
| --- | --- | --- | --- |
| unaligned | aligned | `4*R*C - 1` | 54 of 54 |
| aligned | aligned | `4*R*C` | 61 of 64 |
| aligned | unaligned | `4*R*C` | 43 of 50 |
| unaligned | unaligned | neither | 0 of 6 (`9x49`, `12x20`, `13x17`, `17x13`, `20x12`, `49x9`) |

The ten aligned-`C` shapes where `4*R*C` is absent are `8x24`, `11x32`, `13x32`,
`15x32`, `16x8`, `24x8`, `25x32`, `55x32`, `57x32` and `61x32`; the cause is
unknown (a different form of the same value is one possibility, and several of
them have a zero byte in the value, such as 512 and 768). Inside a block of
unaligned `C` at `R=16`, exactly these two bytes are the only ones that vary:
`C=17..23` gives `3f 04, 7f 04, bf 04, ff 04, 3f 05, 7f 05, bf 05`, which is
`64*C - 1`, and the same holds for the blocks at `C=25, 33, 41, 49, 57`.

**Block templates, at R=16.** Masking the size field and diffing block against
block shows 7-13 differences, nearly all of them immediates that are simple
functions of `q = ceil(C/8)` (rows of 8 elements) and `R`. Observed, not fitted
across all shapes:

| field (approx. byte offset) | value | for `q` = 3..8 |
| --- | --- | --- |
| 549 | `32*q`, the row size padded to 8 elements | 0x60, 0x80, 0xa0, 0xc0, 0xe0 (q=3..7) |
| 617 and 629 | `R*q - 1` | 0x2f, 0x3f, 0x4f, 0x5f, 0x6f, 0x7f |
| before `82 1a` | `2*q` | 8, 10, 14, 16 (q = 4, 5, 7, 8) |
| before `83 34 93` | `4*q` | 16, 20, 24, 28, 32 (q = 4..8) |

So the earlier "4C" reading at offset 549 was right only for aligned `C`; the
padded row size is what is stored. The exceptions line up with `q` divisible by
3: for `q=3` and `q=6` the same slots use a different form (`82 56 02` and
`82 98 02` instead of `00 2q 82 1a`, and `82 cc 95` instead of `00 4q 83 34 93`).
`2q` and `4q` take the values 6, 12 and 24 there, so the trigger looks like a
value that is a multiple of 3 (or of 6). It is a pattern over six data points,
not a rule.

**What this does not establish.**
- No dependence on `R` beyond the two sweeps; the `R` blocks (`R = 8k+1..8k+7`)
  were only seen through the `C=32` sweep, where the changes are larger and
  shifted, and they were not decoded.
- Nothing for the both-unaligned class, or the ten aligned-`C` exceptions.
- Nothing in the tiled regime, and every Transpose in the ResNet18 training step
  (for example `[16,1,576,3136]`, `[1,64,64,9]` at 147 KB) is tiled. The untiled
  decode is a stepping stone: the same quantities (byte size, `ceil(C/8)` row
  counts, `R*q - 1`) should reappear per tile, but that has not been checked.
- No predictor. A generator would still need templates per `(ceil(R/8), ceil(C/8))`
  class (or the rule that produces them), the form-switch rule for `q` divisible
  by 3, and validation on held-out shapes with a device run.
