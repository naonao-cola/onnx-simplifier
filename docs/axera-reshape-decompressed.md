# AX650 non-fused Reshape, re-read on decompressed MCode

`docs/axera-reshape.md`, `docs/axera-reshape-dma.md` and
`docs/axera-reshape-calibration-isolation.md` concluded that non-fused
`Reshape -> Relu` programs have no closed form: `Co=36/60` "changed layout" in
the `Cin=8` weight fold, and `C=44/48/52` looked like "a second undecoded
layout" in the square fold. All three diffed the **compressed** MCode bytes.
PR #1850 (`scripts/axera/short_unit_codec.py`,
`docs/axera-short-unit-encoding.md`) showed that the segments are
LZ77-compressed 8-byte register records. This note redoes the same sweeps on
the decompressed records, offline, with no new builds and no device.

**Short version.**

- **The old walls were compression artifacts.** `Co=36` and `Co=60` share
  one record structure with every other `Co` from 16 to 80 (odd `Co`
  included). `C=44/48/52` share one with `C=40..64`. Only the compressor's
  match choices differed.
- **The rule.** Group builds by record structure, meaning the same
  `(verb, register)` sequence in every segment. Within a group, every record
  value is an exact polynomial of degree 2 or less in the channel count, and
  so are the two tensor-size words in the tail tables. The only exceptions are
  the segment-0 rebuild-noise records and the calibration scale lanes.
- **Emitter.** `scripts/axera/reshape_record_emit.py` fits these rules from
  committed fixtures. It predicts the decompressed segments, re-encodes them
  with the codec's native-compatible encoder, and rewrites the size words. The
  output matches every held-out native build byte for byte, except the known
  301-326 noise window. That includes the real ResNet18 `[64,64,3,3] ->
  [1,64,64,9]` build, predicted from `C=40..60` only.
- **ResNet18 step:** the Reshapes that are generatable and validated offline go
  from 17 to **21 of 170**. The 4 new ones are the `[64,64,3,3]` weight folds.
  The tiled activation families are still single builds, so no rule can be fit
  for them yet. The held-out builds they need are listed at the end.

Method: `short_unit_codec.decode_segments` → `records` on existing builds only:
`t_reshape_dma` (115 sweep builds), `t_reshape_calib` (17 fixed-calibration
square folds), `t_reshape` (the real-shape `R01`..`R13`, `G`, `H` builds),
`t_transpose_sweep`. For each fit I required one more build than the
polynomial's degree needs, as a check point. I also held out each build in
turn (leave-one-out).

## 1. What varies between builds of one family

Three kinds of records change with shape:

| kind | registers | behaviour |
| --- | --- | --- |
| shape/size fields | `0x0320`, `0x0330`, `0x0560`, `0x0790`, `0x07a0`, `0x0180`, `0x01e0`, `0x03b0`, `0x0410`, ... | exact polynomials: `C-1`, `8C-1`, `C/4-1`, `C²/2-1`, `4C²-1`, `16C`, `36C`, `k·C²` |
| buffer addresses | `0x0730`, `0x0500`, `0x02c0`, `0x0240`, `0x0350` (+ flags in `0x0100`) | polynomial offsets inside a group; at the smallest sizes the allocator places buffers in a second region based at `0x2f7020` and sets bits in `0x0100` (`97281 → 97377`), which is a different group in practice |
| calibration | scale lanes `0x0f50..0x0fc0` (8 × `1/s`, then 8 × `s`, float32); `0x1b10`/`0x1eb0`/`0x1a90` = 127/128 | follow the calibration, not the shape; constant across builds with one calibration |

Segment 0 also has four `0xa2` records whose value is `(n << 20) | 19`, where
`n` rotates between rebuilds of the *same* graph. These records are the byte
window 301-326 that earlier notes ignored as noise.

The tail tables outside the segments hold two uint32 words equal to the fp32
tensor byte size, `4·numel`: `36C²` for the square fold and `288·Co` for the
`Cin=8` fold. The only other words that change are each segment's compressed
length (table key 5), which `replace_segment` rewrites itself.

## 2. Sweeps, record level

Groups are sets of builds with identical record structure. "LOO" means each
build was predicted from the rest of its group, excluding the noise records.
For families with free calibration it also excludes the calibration lanes.

| family | builds | groups (members) | LOO |
| --- | --- | --- | --- |
| `[Co,8,3,3] -> [1,Co,8,9]` | 39 | {8, 16..80 incl. 36, 60, odd Co} (8: address-region flip), {12} | 37/37 exact (Co 16..80) |
| `[8,Ci,3,3] -> [1,8,Ci,9]` | 15 | {8, 16..64 step 4}, {12} | 13/13 exact |
| `[C,C,3,3] -> [1,C,C,9]`, fixed calibration | 17 | {8..32}, **{40,44,48,52,56,60,64}**, {68}, {72,76,84,88}, {80} | 7/7 exact for 40..64 |
| `[1,C,7,7] -> [1,1,C,49]` | 15 | {8,12,16,28,40..64}, {20,24,32,36} | 8/8 and 4/4 exact (C ≥ 20) |
| `[N,8,7,7] -> [N,1,8,49]` | 7 | {2,5,6,7,8}, {3,4} | 4/4 exact (N 5..8) |
| `[1,C,C,9] -> [1,1,C,9C]` | 7 | {8,24}, {16}, {32}, {40,48}, {64} | too few per group |
| `[1,C,9C] -> [C,C,3,3]` | 7 | {8,16,24}, {32}, {40,48}, {64} | too few per group |
| `[1,1,8,H²] <-> [1,8,H,H]`, k-merge | 10+10+6 | many 1-2 member groups | too few per group |

Notes:

- The `Cin=8` rule is 12 shape records: `Co-1` (×6), `8·Co-1` (×2),
  `128·Co` (×2) and `256·Co` (×2). The 4-byte compressed patch in
  `reshape_dma_emit.py` was a lucky special case of it. `EXACT_CO` excluded
  36, 60, the odd values and ≥ 66 only because their compressed streams
  realigned.
- In the square fold, `{68}`, `{72..88}` and `{80}` are one program. They
  differ only in the order of two register writes that carry equal values
  (`0x0500`/`0x0700`, or `0x0300`/`0x0500`). This is the same kind of
  ordering coin-flip seen in AX650 MatMul tables. A structure key that
  canonicalizes the order would merge them. I did not do that here, because
  I have no device run showing that the order does not matter.
- Group boundaries sit at size thresholds (`C=32→40`, `C=64→68`): the
  `[7,2,299,2,2]` → `[7,2,286,38,2]` → `[7,2,363,39,89]` record counts show
  work moving into segments 3 and 4 as the tensor grows, so the boundaries are
  tiling decisions, not noise.

## 3. Transpose (secondary check)

`t_transpose_sweep` `[1,1,M,N]` families, record level:

- `[1,1,16,N]` for N 8..64 has three groups: {8}, N not a multiple of 8 (49
  builds), and N a multiple of 8 (7 builds). `[1,1,M,32]` for M 9..63 splits
  the same way. The "per-block templates" in earlier Transpose notes are
  this alignment split, not per-shape programs.
- Inside the multiple-of-8 groups every varying record is polynomial. The
  exceptions are groups whose fixed dimension is 8 or 16 (small tensors):
  there `0x0100`/`0x0730`/`0x02c0`/`0x0240`/`0x0350` show the same
  address-region flip as Reshape. The unaligned groups have 6-11 fields that
  are not polynomial. They look like `ceil`/remainder terms and need a
  piecewise fit, which is not attempted here.

## 4. The real ResNet18 Reshape families

Reshape counts in `t6-r18fold/step.onnx` (1104 nodes, 170 Reshapes), grouped by
what the decompressed evidence supports:

| family (count in step) | evidence | status |
| --- | --- | --- |
| `[1,C] -> [C]` bias (17) | fused (`reshape_emit.py`) | covered, device-checked earlier |
| `[64,64,3,3] -> [1,64,64,9]` (4) | real build `R10` has the `C=40..64` group structure; `C=40..60` rule predicts it exactly, scale lanes aside | **offline-validated** (`reshape_record_emit.py`) |
| `[1,64,64,9] -> [1,1,64,576]` (4), `[1,64,576] -> [64,64,3,3]` (4) | real builds `R11`/`R12` are record-identical to the sweep builds at C=64, but C=64 is a group of one in both sweeps (48 → 64 inserts a `0x0780` write) | not yet, needs sweep points |
| `[C,C,3,3]`, `[1,C,C,9]`, `[1,C,9C]` at C = 128, 256, 512 (27) and the non-square `[2C,C,k,k]` folds (18) | only `[512,512,3,3]` built (`R13`, tiled, 1531 records in seg 2) | not yet |
| tiled activation folds `[16,C,H,W] <-> [16,1,C,HW]`, `[16,1,C,9HW] -> [16,1,9C,HW]`, plus the stem, max-pool and head reshapes (96) | one build per activation family (`G`, `H`, `R01`..`R07`). Segments 3 and 4 are a prologue plus K repeats of a 13-85 record block, whose values are mostly constant or affine in the repeat index. Segment 2 (1-9k records) does not repeat. Stem, max-pool and head shapes are not built | not yet |

So the step goes from 17 to 21 of 170 Reshapes offline-validated. The rule
covers the whole square-fold group, `C` 40 to 64. The step only uses `C=64`
from that range.

## 5. Emitter

`scripts/axera/reshape_record_emit.py`:

- `fit_rules(fixtures)` takes native MCode of one group. It requires the same
  blob layout, decompressed lengths and `(verb, register)` bytes in every
  fixture. Each varying record and each tail word must be an exact polynomial
  with one fixture to spare. Otherwise it raises.
- `predict_mcode(rules, C, scale=None)` starts from the nearest fixture, sets
  every fitted record, and re-encodes each changed segment with
  `short_unit_codec.replace_segment`. The encoder reproduces native streams
  byte for byte on all these builds, so the padded size never changes. It
  then sets the tail words. `scale` rewrites the 16 scale lanes to `(1/s, s)`.
- `emit_axmodel(family, C, path, scale=None)` also relabels the input and
  output dims, `value_info`, `inputs_info` and `outputs_info`. It refuses any
  `C` outside the group measured for that family.

Families: `square_weight_fold` has `C ∈ {40..64 step 4}`. Its fixtures are
`reshape_calib_isolation/fixed_C{40,44,56,60,64}`, which share one fixed
calibration. `cin8_weight_fold` has `Co ∈ {16..80}` as measured. Its fixtures
are the committed `reshape_dma/w1_co{34,40,42,58,62}`.

Offline validation (`tests/test_axera_reshape_record_emit.py`, committed
fixtures in `scripts/axera/fixtures/reshape_record_emit/`). "Byte-identical"
below always means outside offsets 301-326:

- Leave-one-fixture-out, both families: byte-identical.
- Held-out natives `C=48, 52`, `Co=36, 60` (the old "layout change" cases):
  byte-identical.
- Real `[64,64,3,3] -> [1,64,64,9]` (`R10`), fitted without `C=64`: only the
  16 scale-lane records differ, because that build used a different
  calibration. With `scale=` set to its `s`, the output is byte-identical.
  `Co=20` with its own calibration behaves the same way.
- Against scratch builds not committed: all 37 measured `Co` values, and all 7
  square-fold `C` values, record-identical apart from the scale lanes. They
  were also byte-identical once the scale was matched.

Not validated: running an emitted model on the device. The scale passed to
`scale=` was read back from native builds, not derived from calibration data.
MinMax gives `s = amax/127`, which matches these builds (`amax ≈ 0.896`), but
predicting `amax` itself is the calibration emitter's job.

## 6. Held-out builds wanted (Pulsar2 only, no device)

These are ordered by how many step Reshapes each would unlock. All are
`Reshape -> Relu` with the same MinMax recipe.

1. `[1,C,C,9] -> [1,1,C,9C]` and `[1,C,9C] -> [C,C,3,3]` for C = 52, 56, 60,
   64 and 68, 72 (8 builds). This finds the group that holds C=64 and would
   cover 8 more step ops.
2. Square fold `[C,C,3,3] -> [1,C,C,9]` for C = 96, 112, 120, 128, 136, 144
   and 240..272 step 8 (about 11 builds). This finds the groups that hold 128
   and 256, and the same again for the flatten and unfold pair (about 22
   builds). That would cover about 18 more ops.
3. Tiled activation fold `[16,C,56,56] -> [16,1,C,3136]` for C = 48, 56, 72,
   80, plus `[N,64,56,56]` for N = 8, 12, 20 (7 builds). Each is a large
   build, so run them one at a time. This tests whether the per-tile block
   repeat is polynomial in C and N. If it is, it opens most of the 96
   remaining activation and stem Reshapes.

## Reproduce

```
uv run --no-project --with onnx --with pytest \
  python -m pytest tests/test_axera_reshape_record_emit.py
python scripts/axera/reshape_record_emit.py square_weight_fold 52 /tmp/w52.axmodel
```
