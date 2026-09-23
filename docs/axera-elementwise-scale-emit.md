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

Byte-exact against native Pulsar2 builds, outside the known 301-325 noise
window, at held-out calibrations:

| shape | zero point | held-out scales | result |
| --- | --- | --- | --- |
| `[1,64,56,56]` | 128 | 6 (incl. 4 non-power-of-2) | 6/6 |
| `[1,64,56,56]` | 0 | 1 | 1/1 |
| `[16,64,56,56]` | 0 | 3 (non-power-of-2) | 3/3 |

Still to validate: the other real training-step Relu shapes, Sqrt, the binary
ops, and device runs.

## Zero point: not handled yet

A zero-point change is refused. The zero point is written through the stream's
compressed register writes (#1836's blocker). A controlled zero-point sweep
(22 values at a fixed scale) shows these are LZ-style: a record's bytes are
either literal or copied from an earlier position, at a distance in 4-byte
units. For zero points 129, 191, 200 and 254 the stream is byte-identical
except for the single literal zero-point byte, but a value that already
occurs earlier (e.g. 100) is back-referenced instead, which changes lengths
and shifts later distances. Details to follow.
