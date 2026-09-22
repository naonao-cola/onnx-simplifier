# AX650 ReduceSum: a first decode, and a split verdict

The ResNet18 training step has 44 `ReduceSum` nodes across 21 distinct
shapes (from `/home/takecheeze/npu-scratch/t6-r18fold/step.onnx`, mostly
gradient-accumulation reductions over the batch axis, plus a few full
per-channel-bias reductions and the FC layer's `(0,)`/`(1,)` reductions).
Nobody in this project had looked at `ReduceSum`'s compiled artifact before.
This is that first look, on standalone float32 builds (Pulsar2 7.0-lite,
AX650, MinMax calibration, uniform +/-0.9 calibration data).

**Compiles standalone with no chunking or neighbour needed**, unlike
`Reshape` (needs a consumer) or the stem `Gather` (needed splitting for an
OCM budget). Six shapes were built: the four real shapes
`[16,1,64,576]->axis=(0,)`, `[16,1,64,3136]->axis=(0,3)`,
`[16,1000]->axis=(0,)`, `[16,1000]->axis=(1,)`, plus two synthetic
axis-`(0,)`-only variants of real shapes (`[16,1,64,3136]` and
`[16,1,128,784]`) built specifically to isolate the batch-reduction tiling
rule from the real graph's more complex `axis=(0,3)` case.

## The input tile table substantially reuses the already-decoded elementwise mechanism -- this is real, validated progress

`npu_params` for a single-axis (`axis=0` only) reduction is governed by the
**same entries/stride/set-order rule** `scripts/axera/dma_tile_predict.py`
already decoded for standalone `Relu`'s DMA input-loading table: with
`L = N` (the reduced axis) and `row` = the product of every kept dimension,
find the smallest `entries` in `4, 8, 16, ...` such that
`ceil(L/entries) * row * 4 <= 524288`, and the offsets are
`k * ceil(L/entries) * row * 4` for `k` in `range(entries)`, in the
iteration order of a Python `set` built by inserting them in that order.

Confirmed exactly, byte-for-byte on the core offset values, for two of three
tested shapes:

| shape | `L`, `row` | predicted `entries`, stride | confirmed |
| --- | --- | --- | --- |
| `[16,1,64,576]`, axis 0 | 16, 36864 | 8, 294912 | yes -- all 8 offsets present, correct stride |
| `[16,1,128,784]`, axis 0 (synthetic) | 16, 100352 | 16, 401408 | yes -- all 16 offsets present, correct stride |
| `[16,1,64,3136]`, axis 0 (synthetic) | 16, 200704 | 16, 802816 | no -- see below |

This is a genuine extension, not a coincidence: it is the *load* side of
the computation (how the compiler tiles the input tensor's DMA transfer into
OCM-sized chunks) working identically whether the consumer is an elementwise
op or a reduction, because tiling the input load is upstream of whatever the
consumer computes with each chunk.

**What is not explained even in the confirmed cases**: a small trailing
group of extra offset words follows the main block -- one word (`0`) for the
8-entry case, four words (`0, row_bytes/4, 2*row_bytes/4, 3*row_bytes/4`)
for the 16-entry case. Neither count nor pattern was decoded; it does not
block reproducing the *values* that matter (the main offset block), but it
means the *whole* table is not yet byte-for-byte reproducible, only its
dominant part.

**What breaks the pattern**: `[16,1,64,3136]` axis-0-only crosses a second
threshold the existing single-level formula does not predict -- even
`entries=16, chunk=1` leaves a single row (`200704` elements, `802816`
bytes) larger than the `524288`-byte budget the DMA-tile work established
elsewhere, and the real table shows evidence of that single row being
further split (offsets that are non-uniform fractions of the row, not
multiples of one clean stride). This is a second, nested tiling level this
project has not looked at before, on any op. Not decoded here.

**The real graph's `axis=(0,3)` shape** (`[16,1,64,3136] -> [1,64]`, reducing
both the batch axis and the last spatial axis at once) has an even more
different structure: a 36-word block (32 large offsets plus a 4-word tail),
repeated 5 times for 720 bytes total. The 32-word block mixes several
different magnitudes, consistent with tiling *two* axes at once rather than
one. Not decoded; reducing more than one axis is out of scope for this
first pass.

Two real shapes reduce to a tiny output with `npu_params` all zero (`[16,1000]`,
axis 0 or axis 1) -- these are below the untiled threshold, matching the
existing convention.

## The compute segment is the same wall as everywhere else in this project

Segment 2 (the compute microprogram -- `teng2`'s counterpart for this op)
has a **different length for every one of the three shapes tested**: 1568,
11776, and 1888 bytes for `[16,1,64,576]`, `[16,1,64,3136]`, and
`[16,1,128,784]` respectively (all axis-0-only). No pair shares a length, so
there is nothing to byte-diff directly, let alone a template. This matches
every other op's compute segment this project has looked at (Transpose,
Relu/Sqrt, non-fused Reshape): real per-shape re-flow, no rule found.

## Verdict

Split, and worth stating precisely: **the input-loading data table is
tractable and substantially already-solved** (it reuses decoded machinery,
not a new wall), for the single-axis-reduction, small-row case specifically.
**The compute segment has the same opacity as every other op** -- ReduceSum
does not get a pass on that wall just because its output is smaller than its
input. No emitter is possible from this alone: even a byte-exact input table
does not produce a runnable model without the compute segment, and even the
input table has an unexplained tail and an unexplained large-row regime.

## Reproduction

Fixtures (gzipped compiled `.axmodel`s) are committed under
`scripts/axera/fixtures/reducesum_decode/`:

- `a_16x1x64x576_ax0.axmodel.gz`, `c_16x1000_ax0.axmodel.gz`,
  `d_16x1000_ax1.axmodel.gz`, `b_16x1x64x3136_ax03.axmodel.gz`: the four
  real training-step shapes.
- `e_16x1x64x3136_ax0_synth.axmodel.gz`,
  `f_16x1x128x784_ax0_synth.axmodel.gz`: synthetic axis-0-only variants
  built to isolate the single-axis tiling rule from the real graph's
  `axis=(0,3)` case -- these do not correspond to any real node in the
  training step.

`tests/test_axera_reducesum_decode.py` checks the confirmed offset
values/entries/stride against the committed fixtures directly (no
Docker/device required).
