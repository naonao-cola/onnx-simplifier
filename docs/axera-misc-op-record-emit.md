# Record-level emission for ReduceSum, Greater→Cast, Sqrt [512,512,3,3] and the tail ops

This page covers the ResNet18 training step's (`t6-r18fold/step.onnx`) non-memory ops
that no emitter handled yet. All of the work was offline: existing compiler builds were
decompressed with `short_unit_codec` (#1850), their records were compared, and the
comparison was checked against held-out builds. No new Pulsar2 builds and no device runs
were used.

- Code: `scripts/axera/misc_op_record_emit.py`.
- Tests (no device): `tests/test_axera_misc_op_record_emit.py`.
- Fixtures: `scripts/axera/fixtures/misc_op_record_emit/` (the index, the held-out
  calibrations and the Sqrt builds), plus the existing `reducesum_decode/` and
  `teng_register_census/` builds.

## Method

The method is the one from `docs/axera-reshape-decompressed.md` (#1862):

1. Decompress every segment.
2. Group builds by record structure, meaning the `(verb, register)` sequence.
3. Within a group, sort each varying record into either a shape record or a
   calibration record.

For these ops, every build with a different shape turned out to have its own record
structure. The 1×N Greater→Cast sweep does share structures, but its SRAM addresses
switch regime inside a group, so no shape polynomial fits. That leaves the emitters with
exact-shape templates plus calibration edits. Calibration edits touch whole records only.

## ReduceSum

Each tile of a compiled ReduceSum runs three stages. Each stage writes two zero-point
registers and then one run of 8 scale lanes:

| stage | `0x1a90` | `0x1b10` | lanes (`0x0f50..0x0fc0`) |
| --- | --- | --- | --- |
| dequantize | `0` | `zp_x` | `1/s_x` |
| requantize | `zp_x * 2**k` | `zp_y` | `s_x/s_y` |
| output | `zp_y` | `0` | `s_y` |

- **Lane values.** Each lane value is float32 scales combined in float64, then rounded
  to float32.
- **The `2**k` factor.** It depends only on shape. Measured values:
  - `2**4` for 16-row reductions;
  - `2**6` for the census `[1,1,64,3136]` axis 2;
  - `2**8` for `[16,1000]` axis 1 and for `[16,1,64,3136]` axes (0,3).

  The emitter reads it from the template, as the template's own value divided by the
  template's `zp_x`.
- **Omitted writes.** Pulsar2 omits a `0x1b10` write when the register already holds the
  value.
  - `reducesum_s1` has `zp_x = zp_y = 128`. It has no requantize `0x1b10` write, and it
    writes `0x1b10 = 0` after the output stage's `0x1a90`.
  - `reducesum_asym` has `zp_x = 64` and `zp_y = 0`, and is the other way round.
  - An omitted write goes directly after its stage's `0x1a90` record.
- **Validation.** The retarget turns each of `reducesum_s1`, `_s4` and `_asym` into the
  other two record-for-record, in all 6 directions. Each of the 5 committed templates
  also reproduces itself exactly from its own calibration.
- **Tiled programs.** Tiled builds (`[16,1,64,3136]` axes (0,3): 4 tiles) repeat the
  three stages per tile, and the same rule holds.

Refused (`ValueError`):

- `zp_x = 0`, because an omitted dequantize `0x1b10` write was never observed;
- a `0x1a90` write that would equal the register's current value;
- a segment that doesn't end in the `0xa2` terminator plus 0 to 3 zero pad records.

**Record-count changes (added after the step-shape builds).** With `zp_y = 0` the elision
rule removes records overall: the output stage's `0x1b10 = 0` write goes (the requantize
stage already left 0 there), and on a tiled program the next tile's requantize
`0x1b10 = zp_y` write goes too. `[16,1000]` axes (0,1) loses 1 record and
`[16,1,128,784]` axes (0,3) (2 tiles) loses 4. Pulsar2 then re-pads the segment: every
decompressed segment is a whole number of 4-record (32-byte) groups, with all-zero
records after the `0xa2` terminator (0 to 3 of them in every one of 20 ReduceSum builds).
The emitter strips the pad, applies the same state machine, and re-pads. Both pairs (step
calibration ↔ `zp_y = 0`) reproduce the other build record for record in both
directions.

**Packed `zp_x` lanes.** Multi-axis programs (`[16,1000]` axes (0,1), `[16,1,256,196]`
and `[16,1,512,49]` axes (0,3)) also write `zp_x * 0x01010101` (the zero point in all
four bytes) on eight lanes `0x0d30..0x0da0`. The emitter rewrites them with the zero
points, and refuses if those lanes don't hold the template's own packed `zp_x`.

## Sqrt [512,512,3,3]

#1840 left this shape out because only 1 of 3 held-out builds matched its patch of the
compressed bytes. Decompressed, the four builds (`_t`, `_h1`, `_h2`, `_h3`) share one
record structure of 2000 segment-2 records, and only the scale lanes differ:

- `1/s_x` and `s_x` on `0x0f50..`;
- `s_y` on `0x0fd0..` and `0x0f50..`.

The rewrite of those lanes matches all 12 ordered pairs record-for-record. The old miss
was the LZ77 re-encoding, not the program. Sqrt is now fully emitted for the step:
42/42 nodes, 39 from `elementwise_scale_emit` and 3 from this module.

## Greater→Cast

Greater→Cast is not quantized. `gtcast_s1` and `gtcast_asym` are record-identical except
segment 0's slot table (records 2 to 5, a per-build permutation that is also present
across rebuilds of the same model). So a build at the exact shape is the complete
program, and `emit_model("GreaterCast:<shape>")` returns it as is.

No step shape has a build yet. The committed template is the census shape
`[1,64,56,56]`, and the `pgtcast_1x*` sweep only goes up to 131072 elements.

## Coverage on the real step

Run `python scripts/axera/misc_op_record_emit.py step.onnx` for the report.

| op | nodes | covered now | still needs |
| --- | --- | --- | --- |
| Sqrt | 42 | 42 (39 via `elementwise_scale_emit`, 3 here) | -- |
| ReduceSum | 44 | 43 | `[16,1,64,12544]` axes (0,3): Pulsar2 can't compile it standalone (below) |
| Greater | 18 | 18 | -- |
| Cast | 19 | 19 (each one follows a Greater or Less) | -- |
| Less | 1 | 1 | -- |

Tail ops. None of these has an emitter or a build at a step shape.

| op (nodes) | step shape | compiles standalone? | status |
| --- | --- | --- | --- |
| Softmax (3) | `[16,1000]` axis −1 | yes: one `AxQuantizedSoftmax` node (`test_axera_softmax_probe_hardware.py`, small shapes) | needs a step-shape build plus a second calibration to locate its lanes |
| Log (2) | `[16,1000]` | yes, `[1,8]` battery (`pulsar2_ops.AX650_CONFIRMED_WORKING_OPS`) | same |
| Neg (2) | `[1,1]` | yes. `tiny_emit.emit_neg` retargets the output scale on a same-shape reference, and builds exist for `[1,4]`..`[1,32]` (`t9-neg`) | needs one `[1,1]` build; then template plus `emit_neg` |
| MaxPool (1) | `[16,64,112,112]` k3 s2 p1 | yes, single-node battery | needs a step-shape build plus a second calibration |
| ReduceMean (1) | `[16,512,7,7]` axes (2,3) k1 | yes, `[1,8]` probe (`test_axera_reducemean_probe_hardware.py`) | needs a build. It probably follows ReduceSum's three-stage rule; unchecked |
| Squeeze (1) | `[16,512,1,1]→[16,512]` | **no**: a standalone Squeeze hits the scheduler's ZeroDivisionError (`AX650_CONFIRMED_BROKEN_OPS`). It compiles when fused into the following Gemm (FullyConnected). | belongs to the Gemm/FC template, not a standalone emitter |

## Builds still needed (run one batch at a time)

Each build is a single-op model at the step shape, using the step's calibration class
(random normal inputs give `zp_x ≈ 127..128` and `zp_y ∉ {zp_x, 0}`).

Adding a second calibration for the first build of each new group gives a held-out check
of the lane and zero-point rule at that shape. Batches marked (+held-out) should include
one.

- **A: Greater→Cast `x > 0`** (5 builds; covers 17 Greater and 17 Cast). One build per
  shape is complete.
  - `[16,64,56,56]`
  - `[16,128,28,28]`
  - `[16,256,14,14]`
  - `[16,512,7,7]`
  - `[16,64,112,112]`
- **B: MaxPool backward masks** (2 builds; covers 1 Greater, 1 Less and 2 Cast). Both
  compare `[1024,9,3136]` against a live broadcast `[1024,1,3136]` input:
  - Greater→Cast
  - Less→Cast
- **C1: ReduceSum axis 0 on `[16,1,C,K]`, keepdims 0** (4 builds, 6 nodes, +held-out):
  - `C=64`: `K=147`
  - `C=128`: `K=64`, `576`, `1152`
- **C2: the same, larger `C`** (6 builds, 10 nodes):
  - `C=256`: `K=128`, `1152`, `2304`
  - `C=512`: `K=256`, `2304`, `4608`
- **D: ReduceSum axes (0,3), keepdims 0** (4 builds, 13 nodes, +held-out):
  - `[16,1,128,784]`
  - `[16,1,256,196]`
  - `[16,1,512,49]`
  - `[16,1,64,12544]`
- **E: ReduceSum, the rest** (3 builds, 4 nodes):
  - `[16,1000]` axes (0,1) k1
  - `[1024,9,3136]` axis 1 k1
  - `[1024,9,12544]` axis 1 k0
- **F: tail ops** (5 builds, two calibrations each to locate lanes):
  - Softmax `[16,1000]`
  - Log `[16,1000]`
  - Neg `[1,1]`
  - MaxPool `[16,64,112,112]`
  - ReduceMean `[16,512,7,7]`

After A to E, ReduceSum, Greater, Less and Cast reach 44/44, 18/18, 1/1 and 19/19,
provided each new template passes the self-identity check (`test_own_calibration_is_identity`
pattern).

## Step-shape builds (batches A to E)

Built from `step.onnx` one at a time (`fixtures/misc_op_step_templates/`), registered in
`index.json` only after the checks pass:

- **Greater→Cast and Less→Cast** (batches A and B, 7 builds): registered as built, since
  they are not quantized.
- **ReduceSum** (batches C to E, 16 templates): each passes the self-identity check
  (retarget away and back). Four have a second build at another calibration and
  reproduce it record for record in both directions:
  - `[16,1,512,4608]` axis 0 and `[16,1,64,147]` axis 0 (both `zp_y != 0`);
  - `[16,1000]` axes (0,1) k1 and `[16,1,128,784]` axes (0,3), each paired with a
    `zp_y = 0` build that changes the record count. Before the re-pad rule above,
    these two were refused.
- **Not buildable:** `[16,1,64,12544]` axes (0,3) k0 (1 node). Pulsar2 fails with
  `TileFailException: AxQuantizedReduceSum, Can not tile` (a 12.8 MB U8 input against a
  3 MB memory limit). In the step, the reduction presumably fuses with neighbouring ops.
  Covering it standalone would need a different decomposition, which isn't a
  template of this op.
