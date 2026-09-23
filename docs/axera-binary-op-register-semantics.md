# Binary elementwise ops (Add, Sub, Mul, Div): device-confirmed register semantics

`Mul`, `Add`, `Div` and `Sub` are 397, 144, 52 and 46 of the ResNet18 training
step's 1,104 nodes. This extends the Relu register decode
(`docs/axera-teng2-register-decode.md`, #1831) to them, using the same method:
find candidate registers offline, then patch ONE value of a compiled model by a
known factor, run it on the AX8850, and see exactly what changes. Everything
here is from our own Pulsar2 builds and black-box device runs; no license or
obfuscation protection was used or bypassed.

Reproduce with `scripts/axera/binary_op_register_confirm.py` (build + patch
harness); the reader is `scripts/axera/binary_op_register_decode.py`. Raw device
results are committed as `scripts/axera/fixtures/binary_op_registers/
device_patch_results.json`, and `tests/test_axera_binary_op_register_decode.py`
checks both the reader and every conclusion below against them (no device).

## Method

- **Builds.** `Op(x[1,16,8,8], z[1,16,8,8])` for each op, Pulsar2 7.0-lite,
  AX650, MinMax, at three engineered calibrations with the range extremes
  planted so MinMax hits them exactly: `c1` x,z in [-1,1]; `c2` x in [-2,2],
  z in [-0.5,0.5]; `c3` x in [-0.5,0.5], z in [-3,3] (Div: z in [b/2, b]). True
  scales come from each build's `quant/quant_axmodel.json`.
- **Offline census.** Diff every segment's register writes across the three
  calibrations, and search the mcode and `npu_params` for float32/Q15 encodings
  of `1/x`, `1/z`, `y`, and ratios/products of the scales.
- **Device confirmation.** Patch one value of a `c3` build; run under three
  stimuli: x only (z = 0, or a constant for Mul/Div), z only, and both; fit
  `patched = a * control + b` per output lane (flat element index `mod 4`).
  Lock per run, an unpatched health run after every patched run. **All 23
  device experiments ran without a fault and every health run matched its
  control.** The unpatched controls match numpy within int8 noise (max error
  0.015-0.025 on outputs up to ~3 for Add/Sub, 0.013 for Mul, 0.0025 for Div).

## Shared layout: four per-lane copies

In the TENG program (`segments()[2]`, TENG EU9) every scale-like quantity is
written as **four copies 0x10 apart, one per output lane**. Lanes 1-3 are `V`
register writes (e.g. `0x0f60/0x0f70/0x0f80`) or 7-byte compressed short units
(`[03][value32][tag][reg]`); lane 0 is a `W` record or a short unit that also
carries the register address (e.g. payload `50 0f` + value = register
`0x0f50`). So the compressed short units are register writes too -- confirmed on
device below. Only lane-1 copies were patched (#1831 found a lane-0 `W` patch can
kill the card).

## Add

| value | where | device result (patch lane-1 copy) |
| --- | --- | --- |
| `1/x_scale` | TENG `0x0f50..0x0f80` (V; first pass) | x0.5 -> **x's** output in **lane 1** x0.500; z's output and lanes 0/2/3 unchanged |
| `1/z_scale` | TENG, same block, second pass (compressed short units) | x0.5 -> **z's** output in lane 1 x0.500; x's unchanged |
| `y_scale` | TENG `0x0f50..0x0f80`, last pass (V) | x2 -> lane 1 x2.000 for every stimulus |
| `x_scale/y_scale` | `npu_params` bytes 0-1, Q15 | x0.5 -> x's slope 0.50 in **all four lanes**, offset -0.252; z's slope 1.00 |
| `z_scale/y_scale` | `npu_params` bytes 2-3, Q15 | x0.5 -> z's slope 0.50 in all lanes, offset -1.505; x's slope 1.00 |

The Q15 offsets confirm the arithmetic exactly. The requant multiplies the raw,
zero-point-included int8 codes, `y_q = C + (x/y)*q_x + (z/y)*q_z`, with the
zero-point correction `C` a separate constant the patch doesn't touch. Halving
`x/y` therefore also shifts the output by `-0.5 * (x/y) * 128 * y_scale`:
predicted -0.251 (x/y = 0.1429) and -1.506 (z/y = 0.8571), measured -0.252 and
-1.505.

**Equal ratios use a different path.** When `x_scale/y_scale == z_scale/y_scale`
(`c1`), the header holds one shared word and `cv3` (`segments()[3]`, CV EU6) is
only a two-write sync/end stub. Halving the shared word halves **both** inputs'
contributions in all lanes with offset -1.004, matching `-0.5 * 0.5 * (128 + 128)
* y_scale` for one multiplier applied to `q_x + q_z`. When the ratios differ
(`c2`, `c3`), the header holds two words and the compiler adds a real CV
program (13 register writes). So the per-input requant runs through the CV unit.

**cv3's `0x0170` write** (verb `0xa8`, value `0x00088207`, identical across
calibrations): setting byte 2 (`0x08`->`0x04`) changed nothing. Setting byte 0
(`0x07`->`0x03`) left x's slope at 1.00 but scaled **z's contribution to 0.579x**
in every lane (offset -1.266). That equals z's effective coefficient changing
from the header's 0.857 to about 0.497 (~0.5), since `(0.497 - 0.857) * 128 *
y_scale = -1.267`. So byte 0 governs how z's requant coefficient is applied on
the CV path. Unexplained: the x-only stimulus got a -3.01 offset instead of
-1.27.

## Sub

Identical layout to Add: `1/x`, `1/z`, `y` in the same per-lane TENG registers
(same per-lane, per-input device results), and `x/y`, `z/y` as Q15 in
`npu_params` bytes 0-3, both stored **positive**. The subtraction shows up in
the offset sign: halving the `z/y` word halves z's slope in all lanes with
offset **+1.506** (predicted `+0.5 * 0.8672 * 128 * y_scale = +1.506`), versus
-1.505 for Add. The `x/y` word gives -0.251. The sign is applied somewhere not
identified here, so Add and Sub can't be told apart from these values alone.

## Mul

| value | where | device result (patch lane-1 copy) |
| --- | --- | --- |
| `1/x_scale` | TENG `0x0f50..0x0f80` (V) | x0.5 -> lane 1 x0.500 (a product halves with either factor) |
| `1/z_scale` | TENG, second pass (short units) | x0.5 -> lane 1 x0.500 |
| `y_scale / (x_scale * z_scale)` | TENG `0x0fd0..0x1000` (W + V) | **x2 -> lane 1 x0.500**: a per-lane **divisor**, `out = q_x * q_z / R` |
| `y_scale` | TENG, last pass (four plain short units) | x2 -> lane 1 x2.000 |

No `npu_params` header (60 zero bytes) and no CV program. Mul's requant is a
float divisor in TENG registers, not a Q15 word in `npu_params` like Add/Sub.
With four plain short units (no address-carrying first copy), copy k is lane
k-1.

## Div

| value | where | device result (patch lane-1 copy) |
| --- | --- | --- |
| `1/x_scale` | TENG `0x0f50..0x0f80` (V) | x0.5 -> lane 1 x0.500 |
| `1/z_scale` | TENG, second pass (short units) | x0.5 -> lane 1 **x2.00** (reciprocal) |
| `y_scale` | TENG, last pass (short units) | x2 -> lane 1 x2.000 |

No header, no CV program, and no ratio register was found. A search for float32
values within 1e-4 of every product `x^i y^j z^k` (i,j,k in {-1,0,1}) found only
these three in segment 2, yet the output is correct (max error 0.0025). Where
the `x_scale / (z_scale * y_scale)` correction is applied is not identified.

## What the reader gives

`binary_op_register_decode.decode(mcode, npu_params, op)` returns the per-lane
values above and the scales they imply, and recovers `x`, `z` and `y` scales for
all six committed builds within 5e-4 relative of the builds' own quantization
(`tests/test_axera_binary_op_register_decode.py`). When `1/z_scale` is stored in
a shorter compressed short-unit form (1-byte payload, e.g. `add_c2`'s
`00 43 84 5c`) it can't be read directly, and `z_scale` is derived from the Q15
`z/y` word (Add/Sub) or the Mul divisor instead.

## Not confirmed

- The zero-point correction constant `C` of Add/Sub (it is 0 for all builds
  here, because the engineered calibration makes the two ratios sum to 1).
- Where Sub's minus sign and Div's `x/(z*y)` correction live.
- The shorter compressed short-unit value forms (e.g. `00 43` with tag `0x84`);
  only the full 4-byte-payload form (`p = 3`) is read.
- The CV program's other registers (`0x04d0`, `0x0150`, `0x0160`, `0xe1a0..`,
  `0x0210`, `0xe120/0xe130`), and `0x0170` beyond the two bytes tested.
- TENG's `0x03d0` value (`0x00088202`/`0x00088204`) changes with calibration
  across builds, but it wasn't patched here.
- Other shapes, and whether the register blocks move for larger tensors.
- Lane 0 (`W` / address-carrying records) was not patched, by design.

This is a reader plus a device-confirmed semantic map, not an emitter.
