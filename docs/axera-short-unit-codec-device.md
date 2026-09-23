# Re-encoded MCode runs on the AX8850: `short_unit_codec` device check

`scripts/axera/short_unit_codec.py` (#1850) decompresses AX650 MCode segments
into 8-byte register records and re-encodes them. Until now that was only
checked in software. This checks it on hardware: the AX8850 in `axcl-vm`,
V3.6.5 firmware, with committed fixtures only and no Pulsar2 builds.

**Result: every variant ran and produced output bit-identical to the native
reference, and every health run afterwards was clean.** No faults, no card
recoveries; the card stayed at 67–68 °C.

Harness: `scripts/axera/short_unit_codec_device_check.py`. Raw results:
`scripts/axera/fixtures/short_unit_codec_device/device_results.json`. Each
device run holds `/tmp/axcl-device.lock` for that run only, with a control run
of the native model first and a health run after every variant.

## 1. Pure round trip

Every compressed segment is decoded and re-encoded with the codec's own
encoder, then written back in its slot (`replace_segment`). The fixtures were
chosen so the encoder does **not** reproduce Pulsar2's bytes for some
segments; otherwise the test would be vacuous.

| fixture | segments whose re-encoded stream differs from native | output vs native |
| --- | --- | --- |
| `binary_op_registers/add_c2` | 2, 3 | bit-identical |
| `binary_op_registers/sub_c3` | 2, 3 | bit-identical |
| `step_recalib/toyf_A1` (distillation step with Adam, 19 inputs) | 0, 2, 3 | bit-identical |

So the hardware decompressor accepts the encoder's token choices, not just
Pulsar2's.

## 2. Edit round trip: one scale record

In `dma_tiles/relu_1x64x56x56`, segment 2 (TENG) writes `s_y` (0.0070588) to
`0x0f50`..`0x0f80`, one register per output lane (`docs/axera-teng2-register-decode.md`).
Each lane's copy was doubled in the decoded records, re-encoded and run. The
lane-0 copy was not touched; #1831 found that patching it killed the card.

| register | lane changed | elements changed | ratio | other lanes |
| --- | --- | --- | --- | --- |
| `0x0f60` | 1 | 12,491 / 50,176 | exactly 2.0 | unchanged |
| `0x0f70` | 2 | 12,449 / 50,176 | exactly 2.0 | unchanged |
| `0x0f80` | 3 | 12,471 / 50,176 | exactly 2.0 | unchanged |

The unchanged elements in each lane are Relu's zeros. This is the predicted
effect exactly, now reached through decode → edit → re-encode rather than a
byte patch of the compressed stream.

## 3. Record transplant: zero points and step recalibration

For two native builds of the same graph, the records that differ in the
decoded segments are copied from the target into the template, the segments
are re-encoded into the template's slots, and the result is compared with the
**target's** native output. The two native builds really do differ on the
device, so an unchanged template would fail.

| transplant | records moved (segment: count) | native template vs native target | patched template vs native target |
| --- | --- | --- | --- |
| Relu zero point 127 → 100 | 0: 4, 2: 19 | max diff 0.190 | bit-identical |
| Relu zero point 127 → 64 | 0: 2, 2: 19 | max diff 0.049 | bit-identical |
| Relu zero point 127 → 200 | 0: 4, 2: 19 | max diff 0.582 | bit-identical |
| `toyf_A1` → `toyf_Dmom100` step recalibration | 0: 4, 2: 128 | max diff 0.598 | bit-identical |

The three zero-point targets are native Relu builds at the same shape
(`fixtures/short_unit_codec_device/relu_1x64x56x56_z{64,100,200}`), taken from
the elementwise scale-emitter's scratch sweep. Segment 0's moved records are
the known rebuild-noise records (`docs/axera-step-recalibrate.md`); the
functional changes are in segment 2.

The step recalibration is the case `docs/axera-step-recalibrate.md` had to call
`not-patchable`: the momentum-recalibrated step changes 128 TENG records, and
their compressed length differs. Decoded, edited and re-encoded, it fits the
template's padded slot and runs exactly like the native recalibrated build. Two
MCode bytes outside the segment streams also differ between these two builds
(offsets 27,928 and 28,008); they were not transplanted and the output still
matched, so they do not affect execution here.

## What this establishes, and what it does not

Established on hardware:

- The codec's re-encoded streams are accepted by the device decompressor, both
  when they match Pulsar2's bytes and when they don't.
- Editing a decoded register value and re-encoding has exactly the effect the
  register decode predicts.
- Nonzero → nonzero zero-point changes and a real step recalibration can be
  produced from a template by editing records, with device output identical to
  Pulsar2's own build.

Not covered:

- **Growing segments.** Every variant here fit its padded slot.
  `replace_segment` refuses a stream that outgrows it (e.g. `toyf_Bx2`, 108 bytes
  over), and moving later segments to make room is not implemented.
- **Record insertion/removal.** Zero ↔ nonzero zero points and Add's extra
  8-record block change the decoded length; transplant refuses that.
- **Values not taken from a native target.** The transplants copy records from
  a real build. Computing those records from scales alone is the emitter's job;
  this check shows the container side is sound for it.
- **Larger models.** The largest program here is `toyf_A1`'s 43,712-byte TENG
  segment; the full ResNet18 step was not run.

## Reproducing

```sh
# no device: every variant must round-trip exactly in software first
python -m pytest tests/test_axera_short_unit_codec_device_check.py
# device (axcl-vm + AX8850)
python scripts/axera/short_unit_codec_device_check.py device results.json
```
