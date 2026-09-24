# Whole-step recalibration when a segment outgrows its slot

**Short answer: it works, on device, bit for bit.** `toyf_A1` was
recalibrated to `toyf_Bx2`'s calibration without Pulsar2. Segment 2 grows by
128 bytes, the blob was re-laid out, and the result ran on the AX8850 with
output identical to the native `toyf_Bx2` build. The same code moves the
ResNet18 dense-head step `r18_A1` to `r18_Bx2`, and that result is also
bit-identical on device. The full conv-trainable ResNet18 step is still out
of reach, for reasons unrelated to this code (see the last section).

Code: `scripts/axera/step_recalibrate.py`. Tests:
`tests/test_axera_step_recalibrate.py` (no device). Device results:
`scripts/axera/fixtures/step_recalib/relayout_device_results.json` and
`relayout_r18_head_device_results.json` in the same directory.

## Where this starts from

`docs/axera-step-recalibrate.md` (#1836) found that a calibration change
touches the TENG queue's scale records and the `npu_params` requant blocks.
#1850 then decompressed the segments (`short_unit_codec.py`). Seen through
the codec, a recalibration is a pure value edit. `toyf_A1` -> `toyf_Bx2`
rewrites 570 of segment 2's 5,464 records, and every one is a float32
scale-derived value such as `1/s`, `s` or a scale ratio. No record is added
or removed.

The "structural" changes that #1836 counted were LZ77 artifacts. But the
recompressed stream is 108 bytes longer, and segment 2's 32-byte-padded
slot grows from 17,216 to 17,344 bytes. #1855's in-place transplant stops
at that point, because `replace_segment` refuses to change the slot size.

## What a segment's size is recorded in

These rules come from native Pulsar2 builds of the same graph at two sizes,
diffed field by field: `toyf_A1`/`toyf_Bx2` (+128 bytes, no gap before the
tail) and `relu_16x64x56x56_x0_y0`/`x128_y128` (+288 bytes, 4-byte gap).
When segment `k` grows by `delta` bytes, a multiple of 32:

| where | field | change |
| --- | --- | --- |
| tail table of segment `k` | key 2, padded size in 8-byte words | `+delta/8` |
| tail table of segment `k` | key 5, exact token-stream length | new length |
| tail tables of segments after `k` | key 3, word offset from the header end | `+delta/8` |
| word just before segment 0 | `[uint64]` vector length prefix, total words | `+delta/8` |
| header FlatBuffers tables | uoffsets whose target moved (the tail vector pointer, strings and vectors in the tail) | `+delta` |
| header FlatBuffers tables | soffsets whose vtable moved (one header table keeps its vtable in the tail) | `-delta` |
| `.axmodel` | `*_neu` initializer `dims`, and the `value_info` of the same name | new blob length |

Nothing else changes:

- No decoded register record in any segment holds a blob offset.
- Every segment is a multiple of 32 bytes, so the 0/4-byte gap before the
  tail vector stays the same.

`relayout` applies exactly these rules. Given the native streams, it
rebuilds `toyf_Bx2` from `toyf_A1`, and `x128_y128` from `x0_y0`, byte for
byte. It also rebuilds the smaller build from the larger one, byte for byte.

**Finding the header references.** A blind scan of every header word for
"looks like an offset into the moved region" is wrong. `toyf_A1` has four
header words that are string bytes (`"ms\0\0"` and similar), and read as
uoffsets they land past segment 2. Rewriting them made the first attempt
miss the native build by 4 bytes. `header_refs` instead walks the header's
FlatBuffers tables from the root. It follows fields that point at other
header tables, and only touches table soffsets and 4-byte fields whose
target is a valid table, string or vector.

Across all 328 fixture blobs, the walk finds every reference that a blind
scan with target validation finds, except three scalar fields in the large
Gather fixture that the blind scan gets wrong.

## Recalibration: `recalibrate(step, reference)`

There is still no map from per-tensor scale or zero point to register
record. `scales.json` is a sorted list with no tensor names, and #1836's
explainer only classifies records. So the target calibration comes from a
**reference build**: the same graph compiled with the target calibration.
`recalibrate`:

1. decodes both builds' segments and copies every differing record into the
   step;
2. re-encodes the changed segments and applies `relayout` to any whose
   padded size changed;
3. copies `npu_params` from the reference.

It refuses anything that is not a same-shape value edit:

- a different segment count, or a decoded segment length that differs;
- a verb or register that differs at the same record index;
- a zero point that goes between zero and nonzero. Pulsar2 emits extra
  records for a nonzero zero point, so Relu x0 -> x128 is refused;
- an `npu_params` length that differs.

Afterwards it checks that the re-laid-out blob decodes to exactly the
reference's records and passes `mcode.check()`.

For `toyf_A1` -> `toyf_Bx2`, the result is 570 records in segment 2, 92
bytes of `npu_params`, and an MCode that grows from 31,032 to 31,160 bytes.
The model is byte-identical to native `toyf_Bx2` except for one byte, where
our encoder picks a different token than Pulsar2's.

## Device check (AX8850, axcl-vm)

`step_recalibrate.py device OUT.json`. Each run holds
`/tmp/axcl-device.lock` for that run only. Inputs come from
`short_unit_codec_device_check.inputs_for` (seed 7).

| run | vs native `toyf_Bx2` |
| --- | --- |
| native `toyf_A1` (control) | differs, max abs diff 2789.4 |
| **recalibrated `toyf_A1`** | **bit-identical** |
| recalibrated MCode, but A1's `npu_params` | differs, max abs diff 309.1 |
| native `toyf_A1` again (health) | identical to the control run |

There were no faults. The third row shows that `npu_params` is not
optional: the MatMul requant blocks carry calibration too, so patching MCode
alone gets part of the way and then diverges.

## How far this is from the real ResNet18 training step

- **ResNet18 dense-head step (`t8-r18head2`, the `r18_A1`/`r18_Bx2` builds
  from #1836; 237,368 B MCode, 11.35 MB `npu_params`).**
  - Offline, `recalibrate` works. It changes 92 + 91 conv-queue zero-point
    records and 2,189 TENG records, which is a pure value edit. It copies
    66,333 bytes of `npu_params`.
  - Segment 2 shrinks by 96 bytes (the relayout shrink path), and the MCode
    goes from 237,368 to 237,272 bytes. The result differs from native
    `r18_Bx2` in 1 byte.
  - On the AX8850 the recalibrated step's outputs are **bit-identical to
    native `r18_Bx2`**. For reference, native A1 vs native Bx2 differ by a
    max abs of 0.78. The health run after it matched the control, and there
    were no faults.
  - Harness: `scripts/axera/step_recalibrate_r18_device.py`. Results:
    `scripts/axera/fixtures/step_recalib/relayout_r18_head_device_results.json`.
  - These builds live in `~/npu-scratch/t_step_recalib`. They are not
    committed, because `npu_params` alone is 11 MB.
- **Full conv-trainable ResNet18 step (`t6-r18fold`): not reached.** Pulsar2
  still crashes in its PPQ calibrator on this graph (#1836), so there is no
  reference build at any calibration and nothing to recalibrate or compare
  against.
- **Every useful recalibration still needs a Pulsar2 build of the target
  calibration.** Without a scale -> record map and the #1784/#1790
  `npu_params` formula applied per tensor, this is a way to *reuse* a
  reference build's calibration, not to *compute* a new one. It avoids no
  compile. What it proves is that the layout side, including segment growth
  and shrinkage, is no longer a blocker. The remaining gap is:
  - mapping each tensor's scale and zero point to its TENG records and conv
    zero-point fields;
  - the requant formula for `npu_params`, keyed by tensor;
  - handling a zero point that goes between zero and nonzero, which changes
    the record count.
