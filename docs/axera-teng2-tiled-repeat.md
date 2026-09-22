# AX650 `teng2`: not organized per DMA tile -- and `sdma4`'s emptiness is the real surprise

Follow-up to [`axera-dma-queue.md`](axera-dma-queue.md), which decoded a
standalone `Relu`'s `npu_params` tile table (`dma_tile_predict.py`) but left
segment 2 (`teng2`, the compute program) undecoded, having found no template
across an *untiled* channel sweep. `docs/axera-transpose-tiled.md` had found
that a tiled *Transpose*'s segment 4 has a fixed-period, ~65-73%
self-similar repeating block -- one per `npu_params` tile entry. This note
tests the equivalent hypothesis for `teng2` on a *tiled* `Relu` (`n` = 4 to
32 tile entries): **rejected**, with both byte- and record-level evidence.
Segment 4 (`sdma4`) turns out to be the more interesting segment here, for a
reason unrelated to the original hypothesis.

## The hypothesis, and why it fails

If `teng2` were `n` copies of one per-tile compute block, splitting it into
`n` equal pieces (`period = segment_size / n`) should show high
self-similarity -- bytes at offset `i` and `i + period` should agree far more
often than chance, the way Transpose's tiled segment 4 does (64-73%).
Checked on 7 tiled `Relu` builds (`n` = 4, 8, 8, 16, 16, 16, 32; `scripts/
axera/teng2_tile_repeat_check.py`, data at `/home/takecheeze/npu-scratch/
t_dma` and `t_teng2_tiled_repeat`):

| shape | `n` | `teng2` (seg 2) size | self-sim at `size/n` |
| --- | --- | --- | --- |
| `[1,64,56,56]` | 4 | 1408 | 0.055 |
| `[1,192,56,56]` | 8 | 1472 | 0.046 |
| `[4,64,56,56]` | 8 | 1568 | 0.046 |
| `[1,384,56,56]` | 16 | 1760 | 0.039 |
| `[16,128,28,28]` | 16 | 1920 | 0.043 |
| `[16,64,56,56]` | 32 | 2752 | 0.042 |

All six sit at 4-6%, indistinguishable from an equal-length random-byte
baseline (~0.4%) and nowhere near a real repeat. A record-level check
(decode segment 2 with `mcode.FULL_RULE`, split by record count into `n`
blocks, compare each block's sequence of `(kind, verb/tag, field, bank)`
tuples -- i.e. ignore operand values, which is where a per-tile address or
size would live, and compare only instruction *shape*) agrees: where the
record count happens to divide by `n`, exactly 1 of `n` blocks matches the
first (the trivial self-match). Where it doesn't divide evenly (`4`, `8`,
`16`, `32` records did not divide 774, 528, 473, and more record counts than
not), that alone rules out an exact per-tile split. The first ~14 records of
every one of these builds are byte-identical regardless of shape or `n` --
`teng2` opens with a fixed preamble -- but nothing past that recurs at any
tile-aligned period.

**Conclusion: `teng2` does not decompose by DMA tile.** Whatever drives its
size and content, it isn't the same axis `npu_params`' tiling walks. This
doesn't rule out *some* periodicity at a different, undiscovered period
(only `size/n` was tested, since that's what the hypothesis specifically
claimed), but the record-count-not-divisible-by-n evidence makes even a
different tile-aligned period unlikely: a genuine per-tile split should keep
the record count divisible by tile count.

## What was found instead: `sdma4` is sometimes empty

Segment 4 (`sdma4`, the actual DMA queue -- distinct from `teng2`) was not
this note's target, but checking it alongside `teng2` surfaced a real,
cleanly-thresholded finding. For every `Relu` shape already built for
`dma_tile_predict.py`, plus two new ones to bracket the boundary:

| total tensor bytes | `sdma4` (seg 4) |
| --- | --- |
| up to 4,816,896 (`[1,384,56,56]`, `n=16`) | real content: 480-832 bytes, `0xa3`-terminated task groups (2-6 of them, with `0xa1`/`0xa2`/`0xa8`/`0xa9` load/store verbs around each) |
| 5,218,304 (`[1,416,56,56]`, `n=16`) | still real: 832 bytes |
| 5,619,712 (`[1,448,56,56]`, `n=16`) | **empty**: 32 bytes, no verb records at all |
| 6,422,528 and above (`[16,128,28,28]`, `[1,512,56,56]`, `[16,64,56,56]`, `n=16..32`) | empty |

The switch is a clean **total-byte-count threshold between 5,218,304 and
5,619,712 bytes**, not bracketed further. It does not depend on `n`:
`[1,384,56,56]` and `[16,128,28,28]` are both `n=16` and disagree (832 bytes
vs. empty); `[16,64,56,56]` at `n=32` is also empty. `teng2`'s own size keeps
growing past this threshold too (1920 to 2752 bytes for the empty-`sdma4`
shapes above, larger than most full-`sdma4` shapes) -- consistent with, but
not proof of, the DMA descriptors moving into `teng2` itself once `sdma4`
goes empty, rather than disappearing. That would also explain why `teng2`
never showed a clean per-`n` period: for the empty-`sdma4` shapes it would be
carrying two different jobs (compute and addressing) at once, and even for
the full-`sdma4` shapes there's no reason to expect its own structure to
follow the DMA tile count rather than, say, a compute-engine lane count.

This was not decoded further (what determines the threshold precisely, what
`teng2` is actually doing differently past it, or whether `sdma4`-emptying
happens for other op types) -- it is reported as a lead, not a result.

## Consequences

- **The Transpose-derived hypothesis does not transfer.** Tiled Transpose's
  segment 4 repeating per tile was specific to that op (or at least not
  shared by standalone `Relu`'s `teng2`). Generalizing a decode result from
  one op family to another needs its own check every time, not an
  assumption.
- **`teng2` remains the blocker** `docs/axera-dma-queue.md` described. This
  note narrows what it is *not* (tile-periodic) without narrowing what it
  is; a real decode still needs the instruction encoder.
- **The `sdma4`-emptying threshold is a new, real, narrow finding**,
  independent of the original hypothesis, and the most promising lead this
  note produced: unlike `teng2`'s reflow, it is a clean binary switch at a
  bracketed byte count, which is the kind of fact a formula can plausibly be
  found for.

## Reproduction

```
scripts/axera/teng2_tile_repeat_check.py scripts/axera/fixtures/dma_tiles/relu_1x64x56x56.axmodel.gz
scripts/axera/teng2_tile_repeat_check.py /home/takecheeze/npu-scratch/t_dma   # every built Relu shape
```

`tests/test_axera_teng2_tile_repeat.py` checks both measurements (segment-2
self-similarity, segment-4 emptiness) against the 7 tiled + 1 untiled
`dma_tiles` fixtures already committed for `dma_tile_predict.py`. No new
fixtures were added: the two threshold-bracketing builds
(`[1,416,56,56]`, `[1,448,56,56]`) are described here by their byte counts
only, not committed, since they support a single narrow prose fact rather
than a reusable regression.

## What was not tested

Other op types (`Add`, `Sqrt`, quantize/dequantize/clip -- `docs/axera-
elementwise-coverage.md`'s next targets), periods other than `size/n`,
whether the `sdma4`-emptying threshold holds for a different `H*W`, and any
device run (nothing is emitted; this is characterization only, same as
`docs/axera-dma-queue.md`).
