# Register census of the ResNet18 step's elementwise ops (offline)

The ResNet18 training step's elementwise ops -- Mul 397, Add 144, Div 52,
Sub 46, ReduceSum 44, Sqrt 42, Cast 19, Greater 18, Relu 17 -- run on the AX650's
TENG unit. #1831 (`docs/axera-teng2-register-decode.md`) found Relu's
quantization scales as register writes, and #1837
(`docs/axera-binary-op-register-semantics.md`) device-confirmed the map for
Add/Sub/Mul/Div at `[1,16,8,8]`. This census covers the ops those didn't
(Sqrt, ReduceSum, Greater->Cast), and re-checks #1837's map at other shapes
and calibrations. It is **offline**: Pulsar2 builds and byte analysis only, no
device runs, so everything here is "where the value is stored", not "what the
hardware does with it". No license- or obfuscation-protected material was used.

Reproduce: `scripts/axera/teng_register_census.py build WORK_ROOT` (36 builds,
Pulsar2 7.0-lite, AX650, MinMax), then `... census WORK_ROOT`.
`tests/test_axera_teng_register_census.py` checks the Sqrt, ReduceSum, Div and
Greater->Cast findings against eleven committed builds (no Docker, no device).

## Method

- **Shapes** follow the step: Relu/Add/Mul/Greater->Cast at `[1,64,56,56]`
  (the step's stage-1 activation, batch 1), Sub/Div/Sqrt at `[64,64,3,3]` (the
  Adam update's parameter shape), ReduceSum over axis 2 of `[1,1,64,3136]`
  (`keepdims=0`), Greater against a scalar 0 followed by `Cast(to=FLOAT)` (the
  step's Relu-mask pattern).
- **Calibration is engineered**: every calibration sample is the same
  `linspace(lo, hi)`, so each input's min/max -- and hence MinMax's scale and
  zero point -- is known exactly. Variants per op: symmetric ±1 and ±4, an
  asymmetric range, mixed ranges between the two inputs, a non-power-of-two
  scale ratio (`r3`, z = 3·x's range), and zero-point-only changes (`zp0`, `zp`:
  same scale, different zero point). Sub's z is rolled by a third of the array:
  with identical x and z arrays, x − z is exactly 0 and MinMax gives
  `y_scale = 0`.
- **The true scales** come from each build's `quant/quant_axmodel.onnx`
  (`AxQuantizeLinear` outputs for inputs, `AxDequantizeLinear` inputs for the
  output, read per op type).
- **Parsing**: every segment is read as value-carrying records -- 8-byte `V`
  register writes and 7-byte `W` records (#1818), plus compressed short units
  (`S`, `[p][payload][tag][reg]`), which #1837 showed are register writes too.
  Records in the known compiler-noise window (bytes 301-325) are dropped.
- **Locate**: for each variant, every product `s_x^a · s_z^b · s_y^c`
  (`a,b,c ∈ {-1,0,1}`) is searched as a float32 in every 4-byte value record and
  as a Q15 word in the first four `npu_params` words. A hit only counts as
  **discriminating** in a variant where no other product has the same value (for
  Relu, `s_x = s_y` in every build, so nothing there discriminates).

**Segment 0 is build noise, not calibration.** Greater->Cast has no
quantization at all (its quantized model is just `AxGreater`/`AxCast`), yet its
segment 0 (a CONV queue) differs between builds, while everything else is
byte-identical. The census therefore treats segment-0 differences as noise.

## Per-op map

"Copies" is the number of matching records per build (four = one per output
lane, #1831/#1837). Encodings: `V8`/`W7` = 8-/7-byte register write, `S7` =
7-byte compressed short unit (`p=3`, 4-byte payload). Registers are in the TENG
program (segment 2, TENG EU9).

| op | value | where | copies, encoding | verified (discriminating / builds) |
| --- | --- | --- | --- | --- |
| **Sqrt** | `1/s_x` (quantize) | `0x0f50`..`0x0f80` | 4, `W7`+`V8` | 2/3 (3/3 found) |
| | `s_x` (dequantize the input) | short units | 3, `S7` | 2/3 |
| | `s_y` | `0x0fd0`..`0x1000` | 4, `W7`+`V8` | 2/3 |
| **ReduceSum** | `1/s_x` | `0x0f60`..`0x0f80` | 3, `V8` (lane-0 copy not tokenized) | 3/3 |
| | `s_x/s_y` (requant after the sum) | short units | 4, `S7` | 3/3 |
| | `s_y` | `0x0f50`..`0x0f80` (second pass) | 4, `V8` | 3/3 |
| **Greater->Cast** | -- | nothing calibration-dependent | -- | whole mcode identical outside segment 0 in 3/3 |
| Div | `1/s_x` | `0x0f50`..`0x0f80` | 4, `W7`+`V8` | 4/5 |
| | `s_y` | short units | 4, `S7` | 4/5 |
| Mul | `1/s_x` | `0x0f50`..`0x0f80` | 4, `W7`+`V8` | 2/6 |
| | `s_y/(s_x·s_z)` (#1837's divisor) | `0x0fd0`..`0x1000` | 4, `W7`+`V8` | 3/6 (5/6 found) |
| | `s_y` | short units | 4, `S7` | 1/6 (6/6 found) |
| Add, Sub | `1/s_x` | `0x0f50`..`0x0f80` | 4, `W7`+`V8` | 3/6 |
| | `s_y` | `0x0f60`..`0x0f80` (second pass) | 3, `V8` | 3/6 |
| | `s_x/s_y`, `s_z/s_y` | `npu_params` words 0, 1 (Q15) | 1 each | 3/6 |
| Relu | `1/s_x`, `s_y` (`= s_x`) | `0x0f50`..`0x0f80`, occurrences 0 and 1 | 4 / 3 | 0/4 (always `s_x = s_y`; #1831 device-confirmed) |

Low discriminating counts for Mul/Add/Sub are a property of the calibrations,
not missing matches: symmetric ranges make several products coincide (e.g. for
Mul, `s_y/(s_x·s_z)` is exactly 255/4 whenever both inputs are symmetric, which
is why the `pos` variant was added).

**Unified reading of the `0x0fd0`..`0x1000` block.** #1837 device-confirmed it
as Mul's per-lane *divisor* (`out = q_x·q_z / R`, `R = s_y/(s_x·s_z)`). Sqrt
stores `s_y` in the same block, which is exactly the divisor that turns
`sqrt(q_x·s_x)` into the output code `q_y`. So Sqrt appears to use the same
divide-by-register path; this is an inference from placement, not a device
result.

**ReduceSum** follows the elementwise pattern with the requant in the middle:
quantize (`1/s_x`), sum, rescale by `s_x/s_y` (four short units), dequantize
(`s_y`).

**Greater->Cast needs no per-model values.** Pulsar2 keeps it unquantized
(`AxGreater`, `AxCast`, no Quantize/Dequantize), and the compiled program is
byte-identical across the three calibrations outside segment 0. A compiled
reference at a given shape is therefore already the complete program for any
calibration.

## Cross-check of #1837's binary-op map at other shapes

#1837's reader (`binary_op_register_decode.decode`) was run on all 23 Add/Sub/
Mul/Div builds here (`[1,64,56,56]` and `[64,64,3,3]`, versus #1837's
`[1,16,8,8]`):

- **Holds**: `1/x_scale` at `0x0f50..0x0f80` and `y_scale` recovered exactly in
  23/23; Mul's `y/(x·z)` divisor at `0x0fd0..0x1000` exact in 5/6 (absent in
  `mul_asym`, where no `0x0fd0` block is emitted); Add/Sub's Q15 `x/y`, `z/y`
  `npu_params` words, collapsing to one word when the ratios are equal.
- **Does not hold at these shapes: `1/z_scale` is not stored.** Whenever
  `z_scale ≠ x_scale` (Add/Sub `mix`, `asym`, `r3`; Mul `mix`, `asym`, `r3`,
  `pos`; Div `s1`, `s4`, `mix`, `r3`), no record in any segment holds
  `1/z_scale` -- searched as a 4-byte float32 and as its top 3 and top 2 bytes.
  The reader's `z_quant` is `None` in 22 of 23 builds, and in `div_asym` it
  returns a value 13% off.
  How z is scaled at these shapes is not determined here; for Add/Sub the Q15
  `z/y` word is present.
- **The CV-program rule is shape-dependent.** #1837 found a real CV program
  (`segments()[3]`) only when Add's ratios differ. Here segment 3 is populated
  for every Add and Mul build, and even for a single-input Relu at
  `[1,64,56,56]` (256 bytes), so `has_cv_program` reports `True` regardless;
  at larger shapes segment 3 carries other work.

## Zero points: reflow vs fixed width

Pairs that change only the zero point (same scales):

| op | zero points | segment 2 content length | differing bytes | first difference |
| --- | --- | --- | --- | --- |
| Relu | 128 → 64 | 1406 → 1406 | 597 | +668: `84 22 01 10 1b 83 4a` → `... 83 e6` |
| Relu | 128 → 0 | 1406 → 1374 | 671 | +664: `84 22 01 10 1b ...` → `84 22 00 10 84 24` |
| Sub | 128 → 64 (x, z) | 1728 → 1728 | 1023 | +664: `84 22 01 10 1b 83 4a` → `84 22 02 10 1b 40 83 36` |
| Add | 128 → 64 (x, z, y) | 1854 → 1854 | 1657 | +33: register `0x03d0` value `...02` → `...04` |
| ReduceSum | 128 → 64 in, 128 → 0 out | 2144 → 2144 | 1449 | +659: same `84 22 02 10 1b 40 83 36` form as Sub |

So in every op the zero point is handled by a variable-width unit near the
`84 22` marker (#1831's location), and a change **re-encodes the stream from
that point on** (hundreds to ~1,700 differing bytes). Whether the allocated
segment length changes depends on the value: Relu's zero point 0 shortens it by
32 bytes, while 64 and 128 keep it (the extra bytes fit in the 32-byte
padding). No op stores its zero point as a fixed-width field. Greater->Cast has
no zero point.

## What remains unexplained

The census also lists calibration-dependent records that match no scale
product. Most are reflow artifacts: once a variable-width unit changes length,
records after it are re-tokenized and keyed differently. A few are stable
register writes whose values change without a matching formula: `0x03d0` (Add,
Sub, Mul, Div; low byte `02`/`04` not tracking the zero point cleanly), `0x0730`
(Add, Sub, ReduceSum), `0x02b0` (Sub, Div), `0x04e0` (Mul, Div) and `0x8430`
(Sqrt). These are the next candidates for device patching.
