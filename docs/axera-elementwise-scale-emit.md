# Elementwise scale-retarget emitter (work in progress)

`scripts/axera/elementwise_scale_emit.py` rewrites a compiled standalone AX650
elementwise model for a new quantization scale without Pulsar2.
`scripts/axera/elementwise_scale_sweep.py` builds the validation references
with *controlled* calibration: every sample is `linspace(lo, hi)`, so MinMax
sees exactly `lo` and `hi`.

## What the scale depends on

At a fixed zero point, a standalone Relu's whole compiled model depends on the
calibration scale through float32 copies in the TENG segment (`segments()[2]`)
and nothing else:

- quant multiplier `f32(1 / s)`: one copy per output lane in the
  `0x0f50..0x0f80` register-write group (more copies in tiled programs);
- dequant multiplier `f32(s)`: one copy per lane in the second group.

`s` is Pulsar2's own float32 scale. Both formulas matched all 35 batch-1
builds bit for bit, including scales that differ in mantissa bytes, not only
exponent. `npu_params`, the node attributes and every other MCode byte stay
the same.

`minmax_params(lo, hi)` reproduces Pulsar2's MinMax `(scale, zero_point)` for
all 35 builds (the zero point uses the unrounded range, round-half-to-even).

## Validated so far

### Relu: every shape in the ResNet18 training step

Templates at zero point 0 and 128 are committed for all five Relu shapes in the
step (`[16,64,56,56]`, `[16,128,28,28]`, `[16,256,14,14]`, `[16,512,7,7]`, and
the stem's `[16,64,112,112]`; 17 of the step's 1,104 nodes). Retargeting each
template to held-out, non-power-of-two scales reproduced the native Pulsar2
build byte for byte, outside the known 301-325 noise window:

| shapes | zero point | held-out calibrations | byte-exact |
| --- | --- | --- | --- |
| 5 real step shapes | 0 | 3 each | 15/15 |
| 5 real step shapes | 128 | 2 each | 10/10 |
| `[1,64,56,56]` | 128 | 6 | 6/6 |
| `[1,64,56,56]` | 0 | 1 | 1/1 |

Five more held-out builds used a symmetric range (`±0.685`) whose MinMax zero
point came out 127, not 128, through float rounding; the emitter correctly
refused them (zero point is the template key, not range symmetry).
`tests/test_axera_elementwise_scale_emit.py` re-checks one held-out native
build per template with no Docker or device.

On the AX8850 (`axcl-vm`, device lock per run, a native control run first and a
native health run after each emitted model), four emitted models matched
`numpy.maximum(x, 0)` within half an output LSB -- the expected rounding bound --
and every health run passed. Two of the four used calibrations Pulsar2 never
built (`[0, 2.9]` and `[-0.4, 0.4]`):

| shape | zero point | scale | max error | 1 LSB |
| --- | --- | --- | --- | --- |
| `[16,64,56,56]` | 0 | 0.0121569 | 0.0061 | 0.0122 |
| `[16,512,7,7]` | 128 | 0.0121569 | 0.0061 | 0.0122 |
| `[16,128,28,28]` | 0 | 0.0113725 | 0.0057 | 0.0114 |
| `[16,256,14,14]` | 128 | 0.0031373 | 0.0016 | 0.0031 |

An emitted model differs from a native build of the same calibration only
inside the noise window, whose bytes come from the template (itself a native
build), so these runs never put an untested byte pattern on the card. No lane-0
or `W`-companion value is patched away from what Pulsar2 itself writes there.

## Zero point: not handled yet

A zero-point change is refused. The zero point is written through the stream's
compressed register writes (#1836's blocker). A controlled zero-point sweep
(22 values at a fixed scale) shows these are LZ-style: a record's bytes are
either literal or copied from an earlier position, at a distance in 4-byte
units. For zero points 129, 191, 200 and 254 the stream is byte-identical
except for the single literal zero-point byte, but a value that already
occurs earlier (e.g. 100) is back-referenced instead, which changes lengths
and shifts later distances. Details to follow.
