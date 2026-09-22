# Does MatMul/Gemm have a Conv-Wbt-style arithmetic scaffold? No -- and the existing evidence already shows why, precisely

`docs/axera-conv-scaffold-arithmetic.md` (PR #1784) found that Conv's
per-output-channel requantization scaffold (`M_channel = input_scale *
weight_scale_channel / output_scale`, plus a quantized-bias array) is a
**passive data table** whose *content* formula was already known and whose
only remaining problem was *placement* -- where that table's bytes sit once
`Cout` gets wide enough to tile. Once the tiling rule was found, the known
content formula worked unmodified. This asks whether the same move --
"the value formula is already known, only the offset/tiling needs solving"
-- applies to MatMul/Gemm's own quantization fields. It does not, and the
project's own prior work already demonstrates precisely why, so this is a
synthesis of existing evidence rather than a new build campaign.

## What MatMul/Gemm's quantization fields actually are: per-tensor values, not a per-channel table

Searching `scripts/axera/tiny_emit.py` for every already-implemented
MatMul/Gemm-touching patcher turns up:

- `patch_matmul_a_scale` / `matmul_a_quad_form` -- **one** rank-3 batched `A`
  scale, stored as 4 stride-6-or-7 copies of a single value (not one value
  per row/column).
- `patch_matmul_gemm_output_zero_point` -- **one** output zero point, an S
  record at a fixed `reg=120, tag=132`.
- `patch_output_quad` -- **one** output scale, shared across Gemm/Conv/MatMul/
  Mul as the same 4-copy quad mechanism.
- `bank81_field192_operand` -- the one field this project found whose value
  genuinely depends on shape (`K`, the contraction dimension), not on
  calibration data.

None of these is indexed per output channel/row the way Conv's Wbt is
(Conv's own two float32 arrays are `N` entries long, once per output
channel, confirmed by `docs/axera-conv-scaffold-arithmetic.md`'s and the
original README correlation study's channel-indexed layout). Every MatMul/
Gemm field found so far is a **single scalar value for the whole op**. So
the premise this task started from -- "a Wbt-analogous per-channel scale/
zero-point scaffold for MatMul/Gemm" -- does not have a target to decode:
there is no per-channel table in what this project has found in these ops'
mcode. (This does not rule out a per-channel field existing somewhere
un-searched; it means none of the ~150 prior PRs' worth of characterization
work surfaced one, despite extensive searching of exactly this kind of
structure on Conv.)

## The one shape-dependent field was already tested the way Conv's tiling fix was tested -- and failed, for a different, more fundamental reason

`bank81_field192_operand(k)` is the closest MatMul/Gemm analogue to Conv's
`M_channel`: a **known, closed-form value formula** (`4c 05 <1024 // k -
1>`), decoded from real Gemm builds and independently reconfirmed
byte-identical on `MatMul`. Its own docstring already records the exact
experiment this task was asked to run: patch a real `K=512` reference's
field to the formula's `K=256` prediction, and diff against a real,
independently-built `K=256` reference.

**Result (2026-09-18, `tests/test_axera_bank81_field192_patch_verify.py`,
re-run here against the current repo state -- all 19 related tests still
pass, no regression):**

- The two real builds' raw stream lengths already differ (4368 vs 3792
  bytes) -- a length gap no in-place patch can close.
- Over the shared 3792-byte prefix, 2776 bytes (73%) differ, with the first
  mismatch at byte 36 -- nowhere near either copy of the patched field
  itself (bytes 533 and 1172).
- The reverse direction (`K=256` source patched to `K=512`'s predicted
  value, diffed against a real `K=512` build) shows the identical picture.

This is **scattered, not confined** -- the opposite of what a tiling
correction (à la Conv's `Cin>128` duplicate-scaffold fix, PR #1790) would
predict. A tiling fix presupposes the diff is confined to *where* a passive
data table's copies live; here, changing `K` rewrites the surrounding
**instruction stream** itself (the tiling/scheduling of the actual compute
verbs), not just a data table's placement within an otherwise-fixed
program. That is a categorically different kind of shape-dependence from
Conv's Wbt, and it is the same wall this project's `teng2`/DMA-queue
decode work and the Transpose segment-2 decode independently hit: MCode
that encodes real scheduling decisions does not reduce to a value formula
plus an offset rule, however well the value itself is understood.
`bank81_field192_operand`'s own docstring already states the conclusion
precisely: "this field is understood, not generatable."

## What this leaves usable

The per-tensor value patchers (`patch_matmul_a_scale`,
`patch_matmul_gemm_output_zero_point`, `patch_output_quad`) are already
emission-validated and require no new work here: for a **fixed shape**
(same `K`, `M`, `N` as the reference build), they can rewrite scale/
zero-point values without recompiling, the same class of capability
Conv's weight-learning emitter provides for Conv weights at a fixed shape.
That is real, already-shipped coverage -- just not a new discovery, and
not extensible to a different `K`/`N` the way this task hoped.

## Bottom line

No new field moved from "characterized" to "emission-validated" in this
task. The one candidate that looked most like Conv's Wbt story
(`field=192`) was already moved from "characterized" to "**proven
non-generatable across shape**" before this task began, by the exact
experiment this task specifies -- confirmed still valid by re-running its
tests against the current repository. The Conv Wbt tiling breakthrough
does not transfer to MatMul/Gemm because the underlying problems are not
the same kind of problem: Conv's was a data-table placement problem;
MatMul/Gemm's shape-dependent field sits inside the instruction stream
itself.

## Reproduction

No new builds or device runs. Re-run the existing suite:

```
uv run --offline --no-project --with onnx --with pytest --with numpy \
  python -m pytest -q tests/test_axera_bank81_field192_patch_verify.py \
  tests/test_axera_matmul_quad_form_switch.py \
  tests/test_axera_output_scale_quad_generalizes.py \
  tests/test_axera_gemm_output_quad.py
```
