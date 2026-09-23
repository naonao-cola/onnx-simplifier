# teng2 as register writes: device-log visibility + a scale-register decode

This continues the `teng2` (elementwise compute segment) reverse-engineering
with a genuinely different method than the shape/calibration sweeps that hit a
wall in `docs/axera-teng2-sqrt-blocks.md`, `-tiled-repeat.md`,
`-add-two-input.md`, `-calibration-isolation.md`, and `-cluster-patch-refine.md`.
It builds on two just-landed findings: `docs/axera-pulsar-v1-mining.md` (#1818,
`V` records are 8-byte register writes `[verb][unit][reg_lo][reg_hi][value32]`)
and `docs/axera-axcl-runtime-mining.md` (#1821, the device kernel log carries NPU
driver diagnostics). Everything here is from our own builds and black-box device
testing; no license/obfuscation protection was used or bypassed.

## Phase 1 -- execution-visibility levers

### `axcl-smi sh` + `/proc/ax_proc/logctl`: real driver trace, not a disassembler

`axcl-smi sh <cmd>` runs a shell **on the card**. `/proc/ax_proc/logctl` is a
per-module log-level control (`echo klog|ulog <id> <level> > logctl`; NPU is id
6, ENGINE id 29, levels 0-7). Raising NPU+ENGINE to 7, running one Relu, then
restoring level 4, the device kernel log (`axcl-smi log`, files under
`/opt/data/AXSyslog/`) emits the NPU driver's full debug trace. What it gives:

- **The engine/EU map, read directly off the silicon** (init-time version
  banner): `CONV EU[0..5]`, `CV EU[6..8]`, `TENG EU[9..11]`, `MAU EU[12]`,
  `SDMA EU[13..15]`, and `vnpu_info_config` counts (`potato_num:6, warp_num:3,
  dma_num:3, sdma_num:3, mau_num:1`). "potato" is the internal name for the
  CONV engine. This confirms, not infers, that `teng2` runs on **TENG** (the
  fault-injection sweeps saw EU9).
- **Per-run task orchestration**: `model_create_common` prints each segment's
  size (`mcode[0..4] size is 60/32/176/4/8` for a tiny model), the RTV input
  names (`_ocm_base, x, x_offset, y, y_offset, params`), `ocm_usase is
  3141632` (the OCM budget this project already knew), the selected
  `eu_mask` (a Relu uses `0x2243` = CONV0, CONV1, CV6, TENG9, SDMA13), and the
  `rtv write pbuf[i] paddr` load addresses.

What it does **not** give: any per-instruction execution, register-write, or
mcode-decode trace. It is task-level orchestration, one generation above the
instruction stream. Useful for confirming structure and attributing faults, not
a shortcut to the opcode semantics.

### The driver's dumpmcode flag is present but not reachable read-only

`ax_npu.ko` exports `is_npu_debug_dumpmcode_enalbe`, `dump_mcode_with_handle`,
and `npu_set_debug_conf` (matching `docs/axera-bsp-mining.md`'s
`AX_NPU_Set_debug_conf`). But there is no `/proc` or `/sys` node to toggle it
and no module parameter; it is gated behind the `AX_NPU_Set_debug_conf` API with
an argument struct that ships in no header. Reaching it would need calling that
API with a reverse-engineered struct -- not attempted. Recorded so the next
person doesn't re-hunt for a knob that isn't exposed.

## Phase 2 -- register-level decode of the Relu compute segment

Method: single-value patching of one already-verified `Relu(x[1,64,56,56])`
reference, run on hardware, output compared per output *lane* (element index
`mod 4`). `scripts/axera/teng2_register_decode.py` reads the segment's register
writes; `scripts/axera/teng2_fault_injection.py` (#1822) does the patching.

### The scale registers: quant and dequant multipliers, per lane

Building a calibration sweep (input range 1.0/1.8/4.0; and a zero-point sweep
at fixed range 1.0) and diffing segment 2 register-by-register: **only three
registers carry calibration-dependent values -- `0x0f60`, `0x0f70`, `0x0f80`,
each written twice.** Everything else is calibration-invariant.

| occurrence | float value | identity | confirmed against |
| --- | --- | --- | --- |
| 0 | `1 / input_scale` | quantize multiplier | 255 / 141.67 / 63.75 for scale 1/255, 1.8/255, 4/255 |
| 1 | `output_scale` | dequantize multiplier | 0.00392 / 0.00706 / 0.01569 for the same |

Each of the three registers drives **one output lane**: `0x0f60`->lane 1,
`0x0f70`->lane 2, `0x0f80`->lane 3 (lane 0 is carried by a companion `W` record
just before the group). Device-confirmed: doubling the dequant copy of `0x0f60`
changed **exactly** the `index mod 4 == 1` elements, ratio exactly 2.0, nothing
else; doubling all three changed lanes 1/2/3 (75% of elements) and left lane 0
untouched. Halving the quant copy scaled its lane's output by ~0.5 through the
int8 requant, as `int8 = round(acc / input_scale); out = int8 * output_scale`
predicts.

`teng2_register_decode.py` reads these back: `implied_scales()` recovers
`input_scale` and `output_scale` from a compiled model with no device, verified
on committed fixtures at two ranges (`tests/test_axera_teng2_register_decode.py`).

### The clamp registers: per-lane activation floor

`0x1c20 / 0x1c30 / 0x1c40` (default value `0x0000ffff`) are the per-lane
activation clamp lower bound (the ReLU floor), lane-mapped 1/2/3 the same way.
Device-confirmed: patching the high half of `0x1c20`'s value raised lane 1's
floor -- a nonzero patch made every lane-1 output clamp up to a constant
(e.g. 0.9035), while lanes 0/2/3 were untouched; `0x1c30` did the same to lane
2, `0x1c40` to lane 3. So the runtime's ReLU is a clamp whose bound is a
programmable per-lane register, not a hardwired zero.

### Why byte-diffing defeated every prior attempt: the zero point reflows the stream

The zero point is **not** in a register write. It appears as a single byte
`round((-min)/scale)` inside a variable-length short unit right after the
`0x84 0x22` marker (~offset 1044): 0x00, 0x1a, 0x40, 0x4c, 0x80, 0x99, 0xbf,
0xff for zero points 0, 26, 64, 76, 128, 153, 191, 255. Its encoding width
depends on its value (a zero-point of 0 omits it; small values take a 2-byte
form, larger a 3-byte form), so changing the calibration **inserts or removes
bytes** and shifts everything after it. That is exactly the whole-segment reflow
(398-795 differing bytes) the confounded sweeps blamed on shape/calibration:
the scale *values* live at stable register-write offsets, but the zero-point
*byte* re-flows the stream around them. Reading the segment as register writes
(this module) sidesteps the reflow; byte-diffing never could.

## Scope and what's next

Decoded, device-confirmed: the quant/dequant scale registers and the per-lane
clamp registers of a standalone Relu's `teng2`, plus the zero-point byte's
location and reflow behaviour. Not decoded: the majority of segment 2's
register writes (calibration-invariant addresses/tile counts whose individual
meanings are unknown), the variable-length short units beyond the zero-point
byte, and whether the same registers carry over to other elementwise ops
(Add's `teng2` writes to a different-looking `0x03b0/0x03c0/0x03d0` group per
#1823 -- a natural next comparison). This is a reader and a decode, not an
emitter: it does not synthesize a `teng2` segment.

### Device note

Two card firmware deaths occurred during this work (guest `device 3: dead!`);
one during a patch of a scale register's companion `W` slot. Recovery is the
full four-module guest reload in `scripts/axera/vm/README.md`, verified with a
control Relu run each time. Patching the 7-byte companion `W` scale slot (as
opposed to the three `V` copies) is the operation that wedged the card -- avoid
it; the three `V` copies are safe to patch and cover lanes 1-3.
