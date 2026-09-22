# Add's `cv3` segment: content is calibration-invariant; size resists this project's existing formulas

`docs/axera-teng2-add-two-input.md` (PR #1756) decoded Add's `npu_params`
table but explicitly left `cv3` alone -- it found the segment's length
"sometimes coincides" with Relu's at the same shape and its content "matches
by chance" 21-27% of the time, and did not attempt a real decode. This is
that first attempt.

## What `cv3` is

`cv3` is `mcode.segments(mc)[1][3]` -- the fourth of the five per-engine
segments (`conv0`, `conv1`, `teng2`, `cv3`, `sdma4`, per
`docs/axera-step-attribution.md`'s profile-trace naming) every compiled
model's MCode splits into, regardless of op type. For a Relu or Add model
with no `Conv`, `conv0`/`conv1` collapse to small fixed 64/32-byte headers,
but all five segments are still present.

## Finding 1: content does not depend on calibration at all

Three builds of the identical `Add([1,16,16,16], [1,16,16,16])` shape, with
three different calibration configurations -- symmetric `x`/`z` both in
`[-0.9,0.9]`; the asymmetric `x` in `[-0.9,0.9]` / `z` in `[-0.1,0.1]` range
PR #1756's `npu_params` Q15-header formula was itself derived from; and a
second, differently-asymmetric `z` in `[-0.3,0.05]` -- produced **byte-
identical `cv3` segments, 0 of 256 bytes differing, across all three**.

This decisively rules out the natural hypothesis (raised but not tested in
PR #1756) that `cv3` might hold additional quantization literals -- a second
scale ratio, a zero-point, a clip bound -- beyond the ones `npu_params`'s
header already carries. Whatever `cv3` encodes, it is not calibration-
derived, at least not at this shape.

## Finding 2: length is not explained by any single shape variable, nor by `dma_tile_predict.py`'s tile model

`npu_params`'s length and content are a clean function of `N*C`, `H*W`, and
the derived tile `entries` count (`scripts/axera/dma_tile_predict.py`).
`cv3`'s length is not a function of any one of those, each falsified by a
direct counter-example in the measured table below:

- **Not `N*C` alone**: two shapes share `N*C=16` (`[1,16,8,8]` vs.
  `[1,16,64,64]`) with `cv3_len` 256 vs. 32.
- **Not `H*W` alone**: two shapes share `H*W=3136` (`[1,64,56,56]` and
  `[4,64,56,56]`, differing only in batch) with `cv3_len` 256 vs. 288.
- **Not total tensor bytes (`N*C*H*W*4`) alone**: `[1,16,8,8]` (4,096 B) and
  `[1,32,32,32]` (131,072 B) are both far from any shared multiple, giving
  256 and 32 respectively, while `[1,32,8,8]` (8,192 B) gives 256 again --
  non-monotonic in total bytes.
- **Not `dma_tile_predict.py`'s `entries` count alone**: three shapes all
  predict `entries=4` (`[1,64,56,56]`, `[1,128,56,56]`, `[8,64,28,28]`), but
  measure `cv3_len` 256, 256, and 288 respectively.

**One real, narrower structure found**: within `dma_tile_predict.py`'s
*untiled* regime (`entries=1`, i.e. every shape whose `npu_params` is the
same 40-byte all-zero table `dma_tile_predict.py` already predicts), `cv3`
has its own, finer size threshold that model has no concept of at all --
every untiled shape measured with total tensor bytes `<=16,384` gives
`cv3_len=256`; every one with total tensor bytes `>=65,536` gives
`cv3_len=32`. The boundary is bracketed to `(16384, 65536)` and not narrowed
further. This threshold is real (confirmed on 6 shapes: 3 give 256, 3 give
32, all `entries=1`) but it is a second, independent fact about `cv3`, not a
consequence of anything `npu_params`'s own model predicts -- two shapes with
*identical* (all-zero, 40-byte) `npu_params` tables can have different `cv3`
lengths.

Once tiled (`entries>1`), the picture is murkier still: `entries=4` shapes
split 256 vs. 288 depending on shape (not on `entries`, not on `N*C`, not on
`H*W` alone -- see the table), and `entries=8`/`16`/`32` give 288/256/512
with no pattern connecting them to the `entries=4` values. This matches the
same kind of instruction-stream re-flow this project's `teng2` and non-fused
Transpose/Reshape segment decodes have repeatedly hit: real, measurable
structure, but not one that reduces to a formula from the shape alone with
the effort spent here.

## Measured table

| shape | `N*C` | `H*W` | `entries` (predicted) | `cv3_len` |
| --- | --- | --- | --- | --- |
| `[1,16,8,8]` | 16 | 64 | 1 | 256 |
| `[1,16,16,16]` (x3 calibrations) | 16 | 256 | 1 | 256 |
| `[1,32,8,8]` | 32 | 64 | 1 | 256 |
| `[1,16,32,32]` | 16 | 1024 | 1 | 32 |
| `[1,32,32,32]` | 32 | 1024 | 1 | 32 |
| `[1,16,38,38]` | 16 | 1444 | 1 | 32 |
| `[1,16,32,64]` | 16 | 2048 | 1 | 32 |
| `[1,16,56,56]` | 16 | 3136 | 1 | 32 |
| `[1,16,64,64]` | 16 | 4096 | 1 | 32 |
| `[1,32,56,56]` | 32 | 3136 | 4 | 256 |
| `[1,64,56,56]` | 64 | 3136 | 4 | 256 |
| `[1,128,56,56]` | 128 | 3136 | 4 | 256 |
| `[8,64,28,28]` | 512 | 784 | 4 | 288 |
| `[4,64,56,56]` | 256 | 3136 | 8 | 288 |
| `[16,128,28,28]` | 2048 | 784 | 16 (split regime) | 256 |
| `[16,64,56,56]` | 1024 | 3136 | 32 | 512 |

## What this means for a future attempt

`cv3`'s role in the compute is still unknown -- this establishes what it is
*not* (calibration-derived, or predictable from `npu_params`'s own tile
model) more than what it *is*. A real decode would need the same kind of
dense, fine-grained shape sweep the `teng2` and Transpose segment-2 attempts
used, specifically inside the two now-identified regimes (the untiled
256-vs-32 boundary, and the tiled `entries=4` 256-vs-288 split) rather than
across them, plus whatever caused `docs/axera-teng2-tiled-repeat.md`'s
negative self-similarity result for `teng2` to be checked separately here
(this doc did not test whether `cv3` repeats per DMA tile the way `teng2`
was shown not to).

## Reproduction

`scripts/axera/add_cv3_decode.py`'s `MEASURED` table and `load_cv3()` read
the fixtures directly; `tests/test_axera_add_cv3_decode.py` checks both
findings against them with no Docker/device required. New fixtures were
built with `scripts/axera/add_tile_sweep.py build CASES.json WORK_ROOT`
(Pulsar2 7.0-lite, AX650, MinMax calibration) and gzip'd into
`fixtures/add_cv3_decode/`.
