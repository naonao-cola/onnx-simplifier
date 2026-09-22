# AX650 two-input elementwise (Add) DMA table: the header is Q15 requant scales, the compute segment is not decoded

`docs/axera-dma-queue.md` decoded the single-input `Relu` tile table and
explicitly left `Add` out of scope: "every one of 10 `Add` builds mismatched
this rule; `Add`'s table is structurally different (it has its own `cv3`
segment of nontrivial size where `Relu` has none) and was not decoded." This
picks that up. The `npu_params` table is now decoded for a real (if bounded)
domain; the compute segment (`teng2`) and `cv3`'s content remain undecoded,
consistent with every other memory-op decode in this project.

## `npu_params`: a Q15 quantization header plus Relu's own tile cycle, 15 times

Unlike `Relu`'s tile table, `Add`'s is **not** purely a function of shape --
it depends on the compiled model's calibration data too, through the first 2
or 4 bytes:

- **The header is a per-input requantization multiplier**, not a tile
  offset: `round(x_scale/y_scale * 32768)` and
  `round(z_scale/y_scale * 32768)`, each a little-endian `uint16` (a Q15
  fixed-point fraction of the output scale). This was found by building an
  `Add` with deliberately non-overlapping input ranges (`x` in `[-0.9,0.9]`,
  `z` in `[-0.1,0.1]`) so the two scales, and thus the two header words,
  would clearly diverge: the compiled model's real scales
  (`x_scale=0.0070582`, `z_scale=0.0007842`, `y_scale=0.0077911`) predict
  `0x73f5` and `0x0ce2` by that formula, and the compiled table holds exactly
  those two bytes-for-bytes.
- **A dedup rule halves the header when the two ratios coincide.** Every
  earlier build used the same input distribution for both `x` and `z`
  (different RNG seeds, so close but not identical scales), and in every one
  of those, `round(x_scale/y_scale*32768)` equals `round(z_scale/y_scale*32768)`
  to the bit -- and the compiled table stores that value **once** (a 2-byte
  header), not twice. This is what looked, before the ratio formula was
  found, like an unexplained "1-value vs 2-value" structural difference: it
  is the same formula in both cases, just sometimes producing equal words.
- **The body is Relu's own tile cycle for the shape, repeated 15 times**
  instead of Relu's 10: below a size threshold (below), the offset table
  after the header is byte-identical to `dma_tile_predict.predict_words`'s
  output for the same `[N,C,H,W]`, truncated to one cycle and repeated 15x.
  Read and write apparently share one tiling here, rather than each getting
  their own.

`scripts/axera/add_tile_predict.py` implements this: `predict_params(shape,
x_scale, z_scale, y_scale)`. Validated **byte-exact against every one of 7
builds checked** (`scripts/axera/add_tile_sweep.py`, Pulsar2 7.0-lite, AX650,
MinMax calibration), including one held out end to end -- a shape
(`[8,64,28,28]`), calibration ranges (`x` in `[-0.3,0.6]`, `z` in
`[-1.5,0.2]`), and RNG seeds not used anywhere in developing the formula,
built and checked only after the formula was fixed.

**Not decoded, and the predictor raises `ValueError`:**

- **`N*C >= 2048`.** Two shapes at exactly that threshold
  (`[16,128,28,28]` and, to separate "large `N*C`" from "this specific
  shape", a same-total-size `[1,2048,28,28]`) both produced a **three-way
  mix**: the plain write-side cycle (identical to what `Relu`'s own rule
  predicts for the shape) interleaved with two halves of a finer,
  double-entry-count read-side cycle, in an irregular repeat pattern (three
  rows, then repeating blocks of five) that was not reverse-engineered. Both
  were confirmed deterministic across a rebuild, so this is a real regime
  change, not noise -- just not one with a decoded rule. Everything below
  `N*C=1024` (five distinct `N*C` values checked, up to
  `[16,64,56,56]`/`[16,1,64,3136]`) stayed in the simple 15x-single-cycle
  form, so the boundary is somewhere in `[1024, 2048)`, not bracketed
  further.
- **`C=1`**, inherited directly from `dma_tile_predict.py`'s own guard (the
  `[N,1,H,W]`-shaped tensors that fold `H` into the leading axis
  differently); not a new gap this module introduces.
- Everything `dma_tile_predict.predict_words` itself rejects (non-power-of-
  two batch, channel count not a multiple of 4) is inherited the same way,
  since this module calls it directly for the tile cycle.

This predicts `npu_params` only, and it needs the compiled model's own
calibration scales as input -- unlike the single-input tile tables, there is
no way to predict it from shape alone. That also means there is no natural
"retargeting" emitter here the way `memory_emit.py`'s Gather/Slice ones
work: those preserve a reference's MCode and only rewrite indices or offsets
that don't change what the MCode computes. Here, changing the *scale
values* (as opposed to keeping the same calibration and only changing which
data flows through it) is exactly the kind of change PR #1732's compose
decode found does **not** reduce to a params-only patch for Gather; it is
even less likely to for Add, since these Q15 words are literally
requantization multipliers the compute segment's arithmetic depends on, and
nothing here establishes that segment doesn't also encode them elsewhere.
No device run was attempted for the same reason `dma_tile_predict.py` made
none: there is no way to test "does the emitted model compute the right
answer" without first establishing the compute segment is unaffected, which
is not shown.

## `cv3`: content is op-specific; length is shape-dependent but not shared with Relu in general

Comparing `Relu` and `Add` at the same input shape (same tiling, same
`npu_params` shape-dependent length):

| shape | Relu `cv3` bytes | Add `cv3` bytes | same length? |
| --- | --- | --- | --- |
| `[1,64,56,56]` | 256 | 256 | yes |
| `[4,64,56,56]` | 288 | 288 | yes |
| `[16,128,28,28]` | 32 | 256 | **no** |

Where the lengths match, the *content* mostly does not -- diffing `cv3`
byte for byte between `Relu` and `Add` shows about a quarter of bytes
matching by chance (69/256 = 27% at `[1,64,56,56]`, 60/288 = 21% at
`[4,64,56,56]`), clustered early in the segment (mostly within the first
~30 bytes) rather than spread evenly, but not identical. So `cv3`'s size
is shape-dependent in a way that
*sometimes* coincides between op types (plausibly both computing some tiling
metric from the same shape), but its content is op-specific, and the length
coincidence itself breaks at `[16,128,28,28]` -- the same shape whose
`npu_params` enters the undecoded split regime, which is suggestive but not
confirmed as the same underlying cause. No further `cv3` decode was
attempted.

## `teng2` (segment 2): not attempted beyond the negative evidence already on record

`docs/axera-dma-queue.md` already established that `Relu`'s own compute
segment does not reduce to a per-shape template in the size range this
project can search. Add's `teng2` is certainly no simpler (it has two live
inputs' worth of dequantization plus the add itself to encode), and no
sweep was run here to characterize it: the one free comparison available
(`[16,1,64,3136]` vs `[16,64,56,56]`, two literal shapes that
`add_tile_predict.py` and Relu's own rule both fold to the same `(L, row)` =
`(1024, 3136)` tiling) shows their `teng2` segments differ in *length*
(4160 vs 9216 bytes), so even that coincidence Relu's tile table exploits
does not carry over to the compute segment. This is not a new finding, just
confirmation that the same wall applies here; a real characterization would
need the same kind of dense shape sweep the Transpose and Relu decodes did,
which this session's time went to the `npu_params` formula instead.

## Reproduction

```
scripts/axera/add_tile_sweep.py build CASES.json WORK_ROOT
scripts/axera/add_tile_sweep.py check WORK_ROOT
```

`tests/test_axera_add_tile_predict.py` checks the predictor against 7
committed fixtures under `scripts/axera/fixtures/add_tiles/` (gzipped
compiled `.axmodel`s: untiled dedup and non-dedup headers, tiled at 4/8/32
entries, the asymmetric-range build, and the held-out build), plus explicit
rejection tests for the split regime, `C=1`, and non-rank-4 shapes.
