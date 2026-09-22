# AX650 DMA-queue decode: the elementwise tile table, and why the compute program still resists a template

`docs/axera-step-attribution.md` found the `sdma4` DMA queue is 40.6-42.5% of a
real training step's MCode -- the single largest engine, and the recommended
next target. This is a first pass at it, on the simplest carrier available: a
standalone float32 `Relu`.

## The tile table generalizes beyond Transpose

`transpose_tiled_params.py` (from the Transpose tiled-regime decode) already
found the tiling rule for a data-movement op: a byte-size threshold, a
per-tile budget, and offsets written in the iteration order of a Python
`set`. `scripts/axera/dma_tile_predict.py` finds the same *kind* of rule for a
one-input elementwise op's tile table -- structurally simpler (one axis
group, not Transpose's input+output pair) and governed by a different
threshold and budget, because an elementwise op moves one tensor rather than
transposing it:

- Untiled (`npu_params` all zero) up to 262,144 measured bytes; tiled from
  401,408 bytes. The exact boundary in between was not bracketed.
- Once tiled, fold batch and channel into `L = N*C` and treat `H*W` as one
  atomic row. Entry count is the smallest of 4, 8, 16, 32, ... such that
  `ceil(L/n) * H*W*4 <= 524288` (512 KiB) -- a different per-tile budget than
  Transpose's 131,072 bytes, consistent with an elementwise op needing less
  working memory per tile than a transpose. The offsets are the usual `set`
  iteration order.

This was validated against 41 tiled and several untiled `Relu` builds
(`scripts/axera/dma_tile_sweep.py`, standalone `Relu(x[N,C,H,W])`, Pulsar2
7.0-lite, AX650, MinMax calibration): **37 of 41 tiled shapes are byte-exact**,
covering every batch in {1, 2, 4, 8, 16} with any channel count a multiple of
4, from `[1,32,56,56]` up to `[16,128,28,28]` and `[16,64,56,56]` (32
entries). Rebuilds of the same shape (`relu_1x64x56x56_rb1/2`) reproduced the
table exactly.

**Not decoded**, and the predictor raises `ValueError` rather than guessing:

- A batch that is not a power of two. `[3,64,56,56]`, `[6,64,56,56]` and
  `[12,64,56,56]` picked entry counts of 6, 12 and 48 -- values this rule's
  4/8/16/32 doubling sequence can never produce, so batch tiling is not
  simply folded into channel tiling the way the rule assumes.
- A channel count not a multiple of 4 (`[1,33,56,56]`, `[1,42,56,56]`,
  `[1,50,56,56]`): `L % n != 0` for the required entry count.
- `[16,1,64,3136]`: it tiles as if `L=1024, row=3136` (folding H into the
  leading axis), not this rule's `L=16, row=200704`. The predictor rejects
  any `C=1` shape rather than silently mispredicting this case.
- A two-input op. Every one of 10 `Add` builds at the same shapes mismatched
  this rule; `Add`'s table is structurally different (it has its own `cv3`
  segment of nontrivial size where `Relu` has none) and was not decoded.

This is real generalization of the tiling mechanism -- a second, independently
confirmed instance of "byte-size threshold, per-tile budget, `set`-order
table" beyond Transpose -- but it is still narrower than the training step
needs: none of the 37 exact shapes is one of the step's own tensors, and nine
op types with non-trivial DMA traffic (`Add`, `Sub`, `Mul`, `Div`, quantize,
dequantize, clip, ...) are not yet checked against it.

## The compute program (`teng2`) does not reduce to a template here either

`npu_params` is not the whole story: the segment that actually encodes what a
Relu computes (segment 2, `teng2` in the step-attribution doc's naming) was
swept the same way the Transpose decode swept segment 2, and it fails the
same way.

At a fixed shape family `[1,C,56,56]`, `C` a multiple of 8 from 32 to 160
(all producing the same tile-table entry count, 4, and the same segment
sizes `[64,32,1408 or 1376,256,480]`): the pairwise byte distance between any
two different `C`'s segment-2 content is never zero and rarely small --
15-799 bytes differing out of ~1408, with a handful of near-neighbour pairs
(`C=88` vs `96`: 15 bytes; `104` vs `160`: 33 bytes) that do not form a
consistent block structure the way the Transpose decode's blocks did. So
there is no `(shape) -> template + small patch` reduction here, at least not
one visible at this granularity; a real generator for `teng2` would need the
same kind of instruction-level decode the Transpose write-ups (`docs/axera-
transpose-mcode.md`, `docs/axera-transpose-tiled.md`) already found elusive,
applied to a different, and differently-encoded, segment.

One field outside segment 2 resisted explanation even after the segment-2
puzzle was set aside: a single byte at a fixed offset in the loader tail
varies with `C` (`0x62, 0x0c, 0x62, 0x61, 0x61, 0x0c, ...` for `C =
32,40,48,56,64,72,...`) with no linear, modular, or tile-count-based formula
found, yet it is deterministic on rebuild (unlike the known 301-325 noise
window). It is left unexplained.

Five other loader-tail/segment-3/segment-4 fields *are* linear in `C`
(confirmed on 7 shapes, held out from the 2-point fit that produced the
formula): two operand fields at `21*C` and `35*C`, three at `35/2*C`,
`63/2*C` and `77/2*C` (half-integer, consistent with fields addressing
2-byte units), and three single-byte counters at `C/4 - 1`. These are
real, useful fields for a future `teng2`-adjacent generator, but they cover
under 30 of segment 2's ~800 varying bytes at each step -- not enough to
reconstruct the segment.

## What this changes about the priority order

`docs/axera-step-attribution.md` ranked the DMA queue first by MCode share.
That share turns out to split into the same two pieces every other memory-op
decode in this project has found: a movement/addressing table (tile offsets,
byte sizes, counts) that *is* tractable and generalizes across op families,
and a compute/addressing microprogram (`teng2`, and Transpose's segment 2)
that is not, at least not without a real instruction-level model of the
encoder. The first piece is now decoded a second time, independently, with a
different budget and axis-folding rule than Transpose's. The second piece
remains the actual blocker to generating any of this step's MCode, whichever
op family it is approached from next.

## Reproduction

```
scripts/axera/dma_tile_sweep.py build CASES.json WORK_ROOT
scripts/axera/dma_tile_sweep.py check WORK_ROOT
```

`tests/test_axera_dma_tile_predict.py` checks the predictor against eight
committed fixtures under `scripts/axera/fixtures/dma_tiles/` (gzipped
compiled `.axmodel`s spanning untiled, 4/8/16/32-entry tiled, and a
rebuild). No device run was made: this module predicts `npu_params` only,
and without the `teng2` segment there is no complete model to run.
