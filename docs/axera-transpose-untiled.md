# AX650 Transpose, untiled regime: what was decoded and what is emitted

Continues `docs/axera-transpose-mcode.md` ("Third pass"). Scope is the same:
standalone float32 `Transpose(x[1,1,R,C], perm=[0,1,3,2])`, tensors small enough
that Pulsar2 7.0-lite leaves them untiled (one 40-byte all-zero `npu_params`
entry, five MCode segments of sizes `[64, 32, S2, 32, S4]`). Builds are under
`/home/takecheeze/npu-scratch/transpose_untiled` and `t_transpose_sweep`.

**Result.** There is now a validated emitter for a limited set of shapes, and
still no predictor from `(R, C)` alone. `scripts/axera/transpose_emit.py`
patches a committed compiled template for another `C` inside the same
`(R, q = ceil(C/8))` block. 20 blocks are indexed (R = 8, 16, 32, 48, 64; C
unaligned), covering 140 shapes. The field map of each block was fitted on five
of its seven shapes; the other two were built by Pulsar2 and compared byte for
byte (modulo the 301-325 noise window): **40 of 40 held-out shapes are
byte-exact.** 13 emitted models from seven blocks ran on the AX8850 and matched
`numpy.transpose` exactly. What decides which form a new `(R, q)` takes is not
found, so a shape without a template is rejected, not guessed.

## Within a block, only the tensor sizes change

For every complete block (`C = 8q-7 .. 8q-1`, `q >= 3`) built at R=16
(q = 3..8, 37..41 and 10, 16, 24, 33) the compiled MCode of the seven shapes
has the same length, and the only bytes that differ (outside 301-325) are the
low and high bytes of four quantities (`4*R*C - 1`, `4*R*C`, `4*C - 1`, `4*C`):

| quantity | sites (R=16, MCode length 2208) |
| --- | --- |
| `4*R*C - 1` | one in segment 2 (1086 for q=3, 1094 for q=41), and 1588, 1652 |
| `4*R*C` | 2000, 2160 |
| `4*C - 1` | 1244, 1428 |
| `4*C` | 1276 |

Each site is a low byte and the next byte as its high byte, so a fitted map has
16 positions (11 at R=64, where the low byte of `4RC-1 = 256C-1` is always 255).
The high bytes only move once a value crosses 255. The five sites after segment 2
sit at the same offsets in every R=16 form class that was checked; the segment-2
site moves with the class. At R=8 the tail sits at different offsets (1212, 1244,
1396, 1556, 1620, 1968, 2128, MCode length 2176), so the map is per block.

`scripts/axera/transpose_fields.py` fits the map from training builds and
`make_transpose_untiled_index.py` writes the index, fixtures and held-out
oracles. A carry rule extends a fitted low byte to its neighbouring high byte
(needed when a value only crosses 255 outside the training shapes, e.g. `32q`
at q=40); held-out scoring is what validates it.

Two features can coincide on a whole block (the high byte of `4RC-1` and `4RC`
whenever `4RC` is not a multiple of 256, and `4RC-1`'s high byte equals `C-1`
at R=64). The fit resolves that with an explicit tie-break, and the held-out
score, not the fit, is the evidence that the choice is right outside training.

Six blocks did not fit and are not emitted: R=8 q=3, R=24 q=3 and q=5, R=40 q=3,
R=56 q=3, and R=16 q=2. In the R=24, 40 and 56 blocks one segment-2 position
takes the values 142 or 143 depending on `C` in a way none of the fields above
explains (for R=24: 143 at C=19 only). R=8 q=3 has a similar single byte that
varies. R=16 q=2 mixes two forms inside one block (C=12 differs in form from the other six).

## What varies with R and q, and why it is not predictable

**q at fixed R=16.** 145 builds, at the time of the clustering (every C from 9 to 63, C=8q-4 up to q=40,
289..330, 32 random C up to 504) fall into 31 classes under complete-linkage
clustering (at most 34 differing bytes inside a class, roughly the number of
fields). A block of seven consecutive `C` is always one class. Between blocks the
class changes nearly every step for q < 34, and it then recurs:

| class | q values |
| --- | --- |
| A | 34, 35, 41, 42, 43, 44 |
| B | 36, 37, 38, 39, 40, 45, 46, 47 |
| C | 11, 12, 14 |
| D | 19, 20, 22 |
| E | 17, 21 |
| F | 24, 27 |
| G | 28, 30 |
| H | 50-55, 63 |

A fit trained on q=37..40 (class B) is byte-exact for 29 of the 147 other
unaligned R=16 builds, and every one of those is in class B. No simple
rule for the recurrence was found: inspecting the class next to `q mod 3`,
`mod 4`, `mod 5`, the binary form of `q`, and `4q`, `16q-1` showed no pattern
(this was read off a table, not searched exhaustively). The earlier guess that
`q` divisible by 3 switches form is not supported (q=12 shares a class with
q=11 and 14, q=15 and q=18 have their own, q=21 shares with q=17). The form
looks like the output of a scheduler decision over the whole tile plan, which is
why the template is the unit and not a formula.

**R at fixed C.** A block of unaligned `R` (`R = 8k+1 .. 8k+7`, C=32) differs
from neighbour to neighbour in 100-160 positions of segment 2, with insertions
and deletions of 2-11 bytes at most steps, so each `R` is its own form. Among the
aligned `R` at a fixed unaligned `C` the pairwise MCode distances are 26 to 213
bytes, with some pairs close (R=24 and R=64 at C=20 differ in 26 bytes; R=16 and
R=40 at C=28 in 27; R=16 and R=48 at C=36 in 31) and others far. Only aligned
`R` blocks are emitted for that reason.

**Same MCode everywhere else.** The `extra_data` metadata and `npu_graph_info`
attributes are identical across shapes (only `outputs_info`, and the `x`/`y`
value info and graph input/output dims, change); the emitter rewrites those.

## Loose ends from the previous pass

- **Both-unaligned shapes** (9x49, 12x20, 13x17 and permutations). They carry no
  size field and were not decoded further; no template exists for them.
- **The ten aligned-C shapes without `4*R*C`.** All have `4RC` divisible by 128
  (low byte 0x00 or 0x80), but so does every aligned-C shape in the data, so this
  does not separate them. Unexplained.
- **The `q` multiple-of-3 form switch** from the earlier pass is not supported;
  see the class table above.

## Device check

AX8850 in `axcl-vm`, runs serialized with the shared device lock, a control run
of a compiler-built `16x18` model first and a health run after, inputs uniform
in +/-0.8. `numpy.transpose` was the reference.

| emitted shape | block | max error |
| --- | --- | --- |
| 16x19, 16x22, 16x23 | R16 q3 | 0 |
| 16x74 | R16 q10 | 0 |
| 16x322, 16x326 | R16 q41 | 0 |
| 32x18, 32x22 | R32 q3 | 0 |
| 48x34, 48x38 | R48 q5 | 0 |
| 64x19 | R64 q3 | 0 |
| 8x35, 8x38 | R8 q5 | 0 |

Exact match is expected: the op is pure data movement on float32.

## What this is and is not

It is a real, validated way to produce a complete `.axmodel` for a Transpose
without invoking Pulsar2, for 140 shapes across 20 blocks. It is not a model of
the compiler: adding a block means building it once (five training builds plus
held-out checks) and running `make_transpose_untiled_index.py`. The remaining
gap to the ResNet18 step is unchanged: every Transpose there is tiled (`[16,1,
576,3136]`, `[1,64,64,9]`, ...), which this work does not touch, and every one of
its 41 Transposes lies outside the emitted shapes.

## Reproduce

```
scripts/axera/transpose_sweep.py build CASES.json WORK_ROOT
scripts/axera/make_transpose_untiled_index.py WORK_ROOT [WORK_ROOT ...]
uv run --with onnx --with pytest python -m pytest tests/test_axera_transpose_emit.py
```
