# Sub/Mul/Div's `npu_params` tile table: Sub matches Add exactly, Mul and Div drop the header entirely

`docs/axera-teng2-add-two-input.md` (PR #1756) decoded `Add`'s `npu_params`
table -- a quantization header (`round(x_scale/y_scale*32768)`,
`round(z_scale/y_scale*32768)` as Q15 `uint16`s, deduped when equal) followed
by the single-input tile cycle repeated 15 times instead of Relu's 10 -- and
left the other two-input elementwise ops (`Sub`, `Mul`, `Div`) untested. This
checks all three against real training-step-relevant shapes.

## Method

18 standalone Pulsar2 7.0-lite builds (AX650, MinMax calibration), 6 shapes
per op: `[1,32,32,32]` and `[1,16,8,8]` (untiled, the second with a
deliberately asymmetric input range), `[1,64,56,56]`/`[4,64,56,56]`/
`[16,64,56,56]` (tiled at 4/8/32 entries), and a held-out `[8,64,28,28]` with
calibration ranges and RNG seeds not used to derive anything below. Scales
were read from each build's own `out/quant/quant_axmodel.json`, the same
convention `add_tile_predict.py`'s tests use.

## Result: two formulas, not one

**`Sub` reuses `Add`'s formula unchanged.** All 6 builds, including the
held-out one, matched `add_tile_predict.predict_params` byte-for-byte with no
modification -- same header rule, same 15x body repeat, same dedup logic.

**`Mul` and `Div` carry no header at all.** Their real `npu_params` length
never includes the 2 or 4 bytes `Add`'s formula would predict -- e.g.
`Mul(x[1,32,32,32])`'s table is 60 bytes (the body alone: an untiled 1-entry
cycle times 15, 4 bytes/word), not 62 (Sub's own untiled-symmetric case,
header deduped to 2 bytes). This was found by comparing predicted vs. real
*lengths* first, not assumed from any prior expectation about symmetry with
`Add`. Once noticed, "body only, no header" matched all 6 shapes for each op
byte-for-byte, including both held-out cases.

A plausible reading, offered as a guess and not confirmed: `Add`/`Sub` can be
computed by rescaling one quantized input into the other's units (the
header's ratio) and then doing a single elementwise op on raw int8 codes --
`Mul`/`Div` cannot be expressed that way (multiplying or dividing two
differently-scaled int8 codes doesn't reduce to one ratio the same way), so
there may be no equivalent header value to write in the first place. This is
not derived from anything beyond the shape of the evidence.

## Verification

`tests/test_axera_elementwise_two_input_tile_predict.py`: 25 tests against 18
committed fixtures (`scripts/axera/fixtures/elementwise_two_input_tiles/`),
covering all 18 builds above plus rejection tests for an unknown op, the
inherited `N*C >= 2048` split threshold, and non-rank-4 shapes. No
Docker/device required to reproduce; predicting `npu_params` alone doesn't
need a device run any more than `Add`'s own predictor did.

## What isn't covered

- The `N*C >= 2048` split-regime boundary (where `Add`'s own table becomes an
  undecoded three-way read/write mix) is inherited as a rejection threshold
  here, not independently reconfirmed for `Sub`/`Mul`/`Div` -- it may differ
  per op.
- `Div`'s `y_scale` reads as an implausible constant (`78431376.0`) in every
  build here except the differently-ranged holdout, where it's a normal
  value. This looks like the same class of pitfall
  `axera-quant-model-input-scales-overwrite-bug` (a memory note in this
  project) already flagged elsewhere -- a naive tensor-hash lookup in
  `quant_axmodel.json` picking up an unrelated tensor's scale. It doesn't
  affect the results here, since `Div`'s predictor never reads `y_scale` (it
  has no header), but it's worth knowing about for anyone reusing the
  `_scales()`-style extraction on `Div` models for a different purpose.
- As with `Add`, this predicts only `npu_params`. The compute-engine `teng2`
  segment is not decoded for any of these ops (`docs/axera-dma-queue.md`,
  `docs/axera-teng2-calibration-isolation.md` found the same opacity persists
  even once the calibration confound both those docs investigated is
  controlled for), so this remains a characterization tool, not a full
  emitter, and no device run was made.

## Reproduction

The sweep harness used to produce the 18 builds was a local, uncommitted
script mirroring `scripts/axera/add_tile_sweep.py`'s structure (same
`build_one`/`Docker pulsar2:7.0-lite`/calibration-tar pattern, parametrized by
op). It isn't checked in; `add_tile_sweep.py` itself is easy to adapt for a
fresh sweep at other shapes if needed.
