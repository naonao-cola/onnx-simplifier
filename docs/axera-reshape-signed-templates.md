# AX650 step Reshapes on a signed input: `Reshape -> Identity` templates

The step Reshape templates of #1891 are `Reshape -> Relu` builds. They
retarget record-exactly, but the emitted model still applies the Relu. The step
runner (#1919) found this on the device: on a signed input, Reshape_42 was up to
164 LSB off. This note replaces them, for every signed input, with a form that
does not change any value.

**Short version.**

- **Which nodes.** A nonzero uint8 zero point means the calibrated range goes
  negative. At the step's predicted calibration, 100 non-fused Reshapes across
  46 distinct shapes had one and were planned onto a Relu template. The other 19
  are nonnegative (zero point 0), where the Relu changes nothing; they keep the
  zero-point-0 templates.
- **The form.** Pulsar2 refuses a standalone Reshape and a Reshape followed by
  an identity Transpose (both fail with `ZeroDivisionError` in the NPU backend).
  `Reshape -> Identity` compiles: a requantizing copy with the same scale lanes
  as the Relu form, `1/s` then `s`, and the zero point written to
  `0x1b10`/`0x1a90` only. The Relu form also writes it to `0x1eb0`, its clamp.
  `Reshape -> Add(0)` compiles too but is a longer, different program.
- **Retarget.** `reshape_record_emit.retarget_scale` with
  `zp_regs=IDENTITY_ZP_REGS` turns each Identity build into its second
  calibration, and back, record for record: every record, every tail word, and
  the blob length. See the per-shape results below.
- **Device.** On the AX650, Identity templates retargeted to calibrations Pulsar2
  never built, including signed ones, match a plain-`Reshape` fake-quant
  reference to within 1 LSB, the same as the native builds. See the results
  below.
- **Planning.** `emit_step_reshape` now takes the Identity template for any
  nonzero zero point and never emits a Relu template for one. It refuses when no
  Identity template exists. `plan_at_calibration` requires the Identity
  template for a nonzero zero point, and `step_runner` no longer marks these
  Reshapes unsafe.
- **Why #1910's device check passed.** Its Reshape cases compared against a
  `Reshape -> Relu` float reference, so a clipping template matched. The cases now
  use a plain `Reshape` reference.

## Probe: which Reshape forms compile

`[1,128,64] -> [128,64,1,1]`, calibration ±0.9, Pulsar2 7.0-lite:

| form | result |
| --- | --- |
| `Reshape` alone | build fails: `NPUBackendError`, `ZeroDivisionError` |
| `Reshape -> Transpose` (identity perm) | build fails, same error |
| `Reshape -> Identity` | compiles: 228 records in segment 2, lanes `1/s`, `s`; zero point on `0x1b10`/`0x1a90` |
| `Reshape -> Add(0)` | compiles: 292 records, a different program |
| `Reshape -> Relu` (#1891) | compiles: 264 records; also writes the zero point to `0x1eb0` |

## Templates

- 47 shapes, one per distinct shape pair a signed step Reshape takes. 46 come
  from the step runner's list (100 nodes). The 47th is `Reshape_440`
  (`[16,64,112,112] -> [1024,12544]`): its input range is nonnegative, but it
  shares its producer's nonzero zero point (108), so it needs the Identity form
  too.
- Each shape was built twice: at ±0.9 (zero point 128), and at
  `[-0.3, 1.7]` (zero point 38).
- All 94 builds compiled, one at a time through the shared build slots.
- All 47 pairs retarget exactly, in both directions (records, tail and length),
  with the zero point read from Pulsar2's `quant_axmodel.json`. The check is
  `tests/test_axera_reshape_record_emit.py::test_identity_template_retargets_onto_its_other_calibration`.
- One `1024x28224` pair re-encodes to another 32-byte padding. `retarget_scale`
  now relayouts the blob (`binary_op_scale_emit.relayout_segment`) instead of
  refusing.

## Device (AX650, `emitter_device_check.py`)

The reference is float `Reshape` with a Q/DQ pair at the target calibration,
run in onnxruntime. Each case ran as a native control, then the emitted model,
then a health run. Every health run matched its control bit for bit.

| case | target | control max LSB | emitted max LSB |
| --- | --- | ---: | ---: |
| `[1,128,64]->[128,64,1,1]` Identity | synthetic `s=0.0213, zp=93` | 0 | 0 |
| same | synthetic `s=0.0041, zp=201` | 0 | 0 |
| `[16,64,56,56]->[16,1,64,3136]` Identity (tiled) | synthetic `s=0.0213, zp=93` | 1 | 1 |
| same, from the zero-point-38 build | `s=0.00706, zp=128` | 0 | 1 |
| `16x512->16x512x1x1` | step `Reshape_42` (`zp=172`) | 0 | 0 |
| `64x64x3x3->1x64x64x9` | step `Reshape_357` | 0 | 0 |
| `16x512x7x7->16x1x512x49` | step `Reshape_47` | 1 | 0 |
| `16x64x56x56->16x1x64x3136` | step `Reshape_327` | 1 | 1 |
| `1024x28224->1024x9x3136` zero-point-0 | step `Reshape_442` | 1 | 1 |

Reshape_42 had been up to 164 LSB off with the Relu template; it is now exact.

## Coverage at the predicted step calibration

Totals are unchanged: 467 covered / 0 conditional / 637 refused. The 100
Reshapes counted as covered before were emitted from the Relu templates and
were wrong on the device. They are now emitted from the Identity templates, and
the check above supports them. Without a calibration, `plan_node` still lists
all 170 Reshapes as conditional, and names the forms each shape has.

## The fused ReduceSum -> flatten chain at extreme scale ratios

The step runner (#1919) saw the C64 chain
`ReduceSum:16x1x64x3136:axes0,3:k0:reshape64` misbehave on the device at the
step's `s_y/s_x` of about 460. Reshape_371 was up to 144 LSB off, and
Reshape_416 faulted with `0x8030070C`. This was not a range overflow in a scale
register:

- **Native builds at the step's own calibrations.** Built at ratios 478 and 443
  (Reshape_371 and Reshape_416), the emitted MCode matches them record for
  record in every segment past segment 0.
- **Device.** The native builds were exact (0 LSB). The emitted ones were 48 to
  71 LSB off, even at a template-like calibration.
- **The difference.** Header byte 72 holds 0x1000 in the template and in every
  native build. The emitted blob had 0x0fe0.
  - At these scales the ratio lanes re-encode 32 bytes shorter.
  - `binary_op_scale_emit.relayout_segment` then shifts every header word whose
    "target" lies past the segment. The word at 72 is a scalar (72 + 0x1000
    lands 2 bytes before a vtable), and Pulsar2 keeps it.
- **Fix.** `relayout_segment` now shifts a header word only if its target is a
  plausible FlatBuffers object (`_plausible_object`: an aligned table with a
  sane vtable, or a vector whose length fits).
  - The emitted chains are now byte-identical to native outside segment 0's
    slot permutation.
  - On the device: 0 LSB at ratios 478, 443 and the template-like one, with
    health runs matching.
  - The regression test is `test_fused_chain_at_an_extreme_ratio_matches_native_bytes`.

Other emitters used the same path. A scan of the 1010 committed axmodels finds
header words the old rule would have shifted, whenever a retarget changes the
edited segment's padded size:

- 0x1000 at byte 72 in small binary-op, Conv and Gather templates;
- ASCII string words (`"ms"`, `"et"`) in MatMul, MaxPool, Transpose and Conv
  templates.

`Add_976`, the runner's other failing segment, does not relayout; its 115 LSB
error is a different defect.
