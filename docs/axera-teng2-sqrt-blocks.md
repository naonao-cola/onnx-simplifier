# Does teng2 (the compute microprogram) template within a q=ceil(C/8) block, the way Transpose's segment 2 did? Partly -- and the real boundary is finer and unpredictable

`docs/axera-dma-queue.md`'s byte-level sweep of a standalone `Relu`'s
`teng2` segment (MCode segment 2, the compute program -- as opposed to
`dma_tile_predict.py`'s DMA-load tile table) found no template: every
channel count it tried was a distinct program, with no shared structure.
But that sweep only used channel counts that are multiples of 8, so every
one was its own `q = ceil(C/8)` group -- it never actually tested whether
grouping by `q` (the way `docs/axera-transpose-mcode.md`'s "Third pass"
found Transpose's own segment 2 to be block-stable) works for `teng2` too.
This tests that, on a different op (`Sqrt`, so the finding isn't specific
to `Relu`), and finds a real but much finer, and not fully predictable,
version of the same phenomenon.

## Method

Three `q`-blocks of 8 consecutive channel counts each (`C=33..40` -> `q=5`,
`C=73..80` -> `q=10`, `C=153..160` -> `q=20`) of standalone float32
`Sqrt(x[1,C,56,56])`, Pulsar2 7.0-lite, AX650, MinMax calibration. Then,
having found small-diff clusters narrower than a whole `q`-block inside
`q=20`, a **fresh, previously unbuilt** block (`C=161..168`, `q=21`) to
check whether that finer pattern holds out of sample. 32 builds total,
scratch under `/home/takecheeze/npu-scratch/t_teng2_sqrt_blocks`.

## The whole-`q`-block hypothesis fails

Pairwise byte-distance matrices within each block (`q=5`, `q=10`, `q=20`)
show no block is uniformly template-stable. Segment-2 length itself often
changes *inside* a block (`q=5`: lengths 1344/1376/1344/1376/1344 across
`C=33..40`), and even same-length pairs can differ almost completely
(`q=10`: `C=75` vs `C=77`, both length 1312, 458 of ~1300 bytes differ).

## A finer pattern: some runs of 2-3 consecutive C share a template

Within `q=20`, three tight clusters appear: `{153,154,155}` (8-9 differing
bytes), `{156}` alone, `{157,158,159}` (8 differing bytes), `{160}` alone.
Decoding the differing bytes in the `{153,154,155}` and `{157,158,159}`
clusters shows each is an exact affine function of `C`, and critically the
**same small set of slopes recurs in both, independently discovered**
clusters: two 1-byte fields step by exactly 1 per unit `C`, one steps by
14 (mod 256), and several by 28 (mod 256, including one genuine 2-byte
field, `28*C + 546`). These look like natural derived quantities (`14 =
3136/224`, `28 = 2*14`, plausibly row/element counts related to `H*W`),
though their exact semantics were not pinned down.

## Fresh out-of-sample check (`q=21`, never built before this)

The pattern recurs, confirming clusters of this kind are real and not a
`q=20` coincidence, but the cluster **boundaries** do not repeat the same
shape: `q=20`'s split was 3+1+3+1 (isolated singles at `C=156,160`);
`q=21`'s is 2+2+3+1 (`{161,162}`, `{163,164}`, `{165,166,167}`, `{168}`
-- the last with entry count 8, a distinct tile plan). The same slopes
(1, 14, 28) reappear in the `{161,162}` and `{165,166,167}` clusters.

**What does *not* explain the boundaries, checked and rejected in turn:**

- **`C mod 4`.** Fit `q=20`'s split (isolate multiples of 4) but not
  `q=21`'s (`163,164` don't cluster despite differing residues 3,0; `165,
  166,167` cluster despite spanning residues 1,2,3).
- **The DMA tile-table entry count** (`dma_tile_predict.py`'s `entries`).
  `q=21` has three different entry counts inside one `q`-block (4, 4, 4,
  4, **7**, 7, 7, 8) -- itself new information: the existing predictor's
  4/8/16/32-doubling rule does not cover `entries=7`, seen here for
  `C=165,166,167`.
- **Matching `(entries, segment-2 length)`.** `C=163` and `C=164` share
  both (`entries=4`, length 1344) with `C=157,158,159`, yet `163` vs `164`
  differ in >99% of segment 2's bytes -- a real counter-example, not
  measurement noise.
- **"Exact division, no last-tile padding" vs not** (`C` divisible by
  `chunk*entries`). Explained `q=20`'s isolated `156,160` (both exact
  divisions) but not `q=21` (`164` is the only exact division among
  `161..164`, yet pairs with `163`, not isolated).

No rule tried predicts cluster membership from `C` alone. Only building
(or having already built) two candidates and comparing reveals whether
they share a template.

## A working, but not fully reliable, extrapolation tool

`scripts/axera/teng2_cluster_patch.py` takes two already-compiled,
consecutive-channel-count references the caller has confirmed are in the
same cluster, and extrapolates the immediate next (or previous) member by
applying each differing field's observed delta. It scans the whole MCode
blob (not just segment 2 -- a few of the differing fields are in segments
3/4 and the loader tail, matching what the Relu sweep in
`docs/axera-dma-queue.md` found there) and merges adjacent differing bytes
into little-endian fields before computing a delta, so that a byte which
only changes because a lower byte's value overflowed is handled as one
wider quantity rather than two unrelated bytes.

**It is not always exact.** Checked against Pulsar2's own build for all
three fresh triples:

| prediction | result |
| --- | --- |
| `extrapolate(ref157, ref158, target=159)` | **exact** (MCode outside the noise window, `npu_params`, everything) |
| `extrapolate(ref153, ref154, target=155)` | 2 residual differing bytes (of ~3000) |
| `extrapolate(ref165, ref166, target=167)` | 20 residual differing bytes |

The failures are a real, understood limitation, not noise: a field whose
value is identical between the two given references (so it shows no
slope at all) can still be wrong at the target if its low byte overflows
exactly between the nearer reference and the target -- a carry that two
points cannot see. `npu_params` was exact in all three cases (every
cluster member shares one tile-table entry).

## Device check

The one exact prediction (`C=159` from `C=157,158`) was run on the AX8850
in `axcl-vm`, under the shared device lock, with an in-range input
(uniform 0.05-0.85, inside the Sqrt domain and the calibration range).
It matched `numpy.sqrt` with maximum error 0.0081 -- the same error as a
real, independently compiled `C=158` control run on the same input slice,
consistent with ordinary int8 quantization noise rather than anything
wrong with the extrapolation. A repeat control run afterward was clean.
The two inexact predictions (155, 167) were not run on the device --
their bytes are known wrong, so a device run would only tell us if the
wrong bytes happen to be harmless, which is not the question this project
is trying to answer.

## What this means for the "decode teng2" goal

- **The blocker is confirmed to generalize**, not just to `Relu`: `Sqrt`
  shows the identical shape of problem (real templates exist locally, but
  their boundaries are unpredictable from the shape alone).
- **The recurring small-slope fields are a genuine, reusable clue** (the
  same 1/14/28/49 constants across three independently-found clusters,
  two different ops now), but are not enough by themselves: predicting
  *which* shapes share a template is the actual open problem, and no
  candidate explanation (mod-4, tile entries, exact-division, matching
  plan signature) survived a fresh out-of-sample check.
- **A full generator remains out of reach** without either (a) the actual
  instruction encoder (what `docs/axera-transpose-mcode.md` already
  concluded would be needed for Transpose's own segment 2), or (b) an
  exhaustive, expensive per-shape build-and-compare approach that is not
  meaningfully cheaper than just compiling the shape you want.

## Reproduction

Builds: `scripts/axera/dma_tile_sweep.py build CASES.json WORK_ROOT` (the
existing harness; it supports arbitrary single-input ops, `Sqrt` included).
Cluster tool: `scripts/axera/teng2_cluster_patch.py`'s `cluster_positions`
and `extrapolate`; `tests/test_axera_teng2_cluster_patch.py` covers both
the one exact case and the documented carry-byte failure mode as a
regression baseline, against six committed fixtures under
`scripts/axera/fixtures/teng2_sqrt_blocks/`.
