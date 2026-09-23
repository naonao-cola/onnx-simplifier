# Live-operand MatMul: per-shape templates, calibration from scales

**Short answer.**

- **Calibration can be computed without a build.** Every calibration value in
  a compiled live-operand MatMul is an exact, bit-for-bit function of the
  tensor scales and zero points. `scripts/axera/matmul_record_emit.py` finds
  these values in a template build and recomputes them for new scales. No
  Pulsar2 build at the target calibration is needed.
  - It reproduces native builds record for record, with `npu_params` byte
    for byte, on all 650 same-shape pairs in the small sweeps whose
    calibrations differ.
  - It does the same in both directions at one real training-step shape:
    `[1,1,512,4608] x [16,1,4608,49]`, inside its real
    Gather -> Mul -> Reshape -> MatMul chain.
- **Shape needs one build per shape.** The register program does not
  extrapolate across shapes, so every distinct step shape needs one Pulsar2
  build as its template.
- **Offline coverage of the step today:**
  - 4 of the 62 nodes are covered: the Gemm, and the three
    `dX, 512->512 3x3` MatMuls.
  - 4 more have a template that has to be rebuilt first.
  - The rest need 33 builds, batched below.

Everything here is offline. No device run was made, and no new builds.

## Why this route

- **The step's weights are live.** All 20 Conv weights, the Gemm's weight
  and all 41 MatMul operands in `t6-r18fold/step.onnx` are graph inputs or
  activations. The weight-code tables `emitter.py` learns cannot reach them
  (`docs/axera-conv-compose-real.md`), so every one is a matrix multiply of
  two live operands.
- **The whole step doesn't compile.** Pulsar2 still fails on
  `t6-r18fold/step.onnx` (`docs/axera-mcode-training-graph-coverage.md`).
  So `step_recalibrate.recalibrate` (#1860) has no reference build to copy
  from, and it needs one at the target calibration anyway.
- **One build per shape is the minimum.** A template is built once per shape
  at any calibration. After that, calibration comes from the scales alone.

## Calibration: what it touches, and the formulas

The segments are decoded with `short_unit_codec` (#1850). A calibration
change touches three places. Everything else in the blob is identical across
calibrations.

| where | what | formula (float32 bits unless noted) |
| --- | --- | --- |
| TENG (seg 2) lanes `0x0f50..0x0fc0`, 8 per group | one group per operand | `1/s_B`; at larger shapes also `1/s_A`, `s_Y` |
| same lanes, fused chain | per fused op | `1/s`, `s`, `s_i/s_j`, `s_i*s_j/s_k`, `-zp` |
| `0x1a90` / `0x1b10` records | zero points | `int(zp)` |
| `npu_params` | one lane per output column, padded to 16 or per output tile | `float(zp_Y)`, then `s_A*s_B/s_Y` (float64 product, then rounded) |

- **The formula set is closed and small.** It is `float_roles`, plus the `zp` role:
  `1/s` (float32 or float64 division), `s`, `±float(zp)`, `int(zp)`, pairwise
  ratios, and `s_i*s_j/s_k`. Across 179 builds (the sweeps, the step-shape
  ladder builds, the Gather and Transpose chains), every nonzero lane
  matched one of these formulas.
- **Where a single MatMul's program lives.** At `[16,512]x[512,1000]` (the
  step's classifier) the program uses three lane groups: `1/s_b`, `1/s_a` and
  `s_y`. The `npu_params` lanes come in 256/256/224/224/40-column runs, one
  run per output tile, and the tile's DMA descriptor words sit between them.
- **Lanes are not always word-aligned.** In the Gather chain, the
  `npu_params` lanes start at byte 4511, so all four byte phases are scanned.
  A lane is only taken as one when at least 4 equal words sit in a row,
  because genuine requant lanes come one per output column. The descriptor
  words are shape data and are left as they are.

### Two things that look like calibration but are not

- **Rebuild noise.** Verb `0xa8` writes to `0x02b0`/`0x03d0` flip between
  `1/2` and `3/4` even across builds whose calibration and graph are
  identical, for example `t17 batched_r0` and `r3`. They differ in 58 of 63
  same-shape pairs in the sweeps. Together with the known segment-0 tail rotation
  (verb `0xa2`), this is the whole rebuild noise. `compare` counts both as
  noise, not as a difference.
- **`b_native_adversarial`.** This build differs from `a_reference` in its
  Gather *indices*, not in its calibration. The first 1,761 bytes of
  `npu_params` are the index table, and 1,028 `0x02b0` flags change with it.
  It is not a recalibration pair.

### Where the formulas tie

- **Tied formulas never disagreed.** `1/s` in float32 and in float64 always
  gave the same bits. So did `s` and `s*t/t`. No build tells either pair
  apart, and `PRECEDENCE` takes the simplest.
- **A real tie makes `recalibrate` refuse.** Two formulas of the same kind
  can explain one value. In `t_transpose_compose_real`, for example,
  `s_mul_out*s_mul_out/s_y` and `s_mul_out*s_other/s_y` round to the same
  float32. If they give different values at the new scales, `recalibrate`
  raises instead of guessing.
- **What that means for new builds.** A template should be built with its
  two inputs calibrated to clearly different ranges.

### Other refusals

`recalibrate` also refuses a zero point that crosses zero, in either
direction: Pulsar2 then emits a different record count (#1860). It refuses
an unexplained lane and a missing tensor too.

### Validation (all offline)

| check | result |
| --- | --- |
| every calibration-differing same-shape pair of the 151 small sweep builds (`t13`-`t27`) | 650 / 650 record-exact, `npu_params` byte-exact |
| `gather_aggregate_real` `a_reference` <-> `c_wide_reference`, step shape `[1,1,512,4608]x[16,1,4608,49]` | both directions record-exact (890 lanes, 98 `npu_params` words); only the 4 segment-0 rotation records differ, as between any two rebuilds |
| `ladder/mm_add` (step Gemm shape, MatMul + live bias) | all 34 lanes and 1,040 `npu_params` words explained; identity and round-trip exact |

`tests/test_axera_matmul_record_emit.py` pins these results against small
fixtures in `scripts/axera/fixtures/matmul_record_emit/`. Those fixtures
reuse the committed `gather_aggregate_real` builds.

## Shape: why every step shape needs its own build

**Vendor corpus (152 BSP models, #1820).** Decompressed and grouped by
`(verb, register)` sequence, the 152 models fall into **139 distinct
structures**.

- **npu1:** none of the 40 models shares a structure with another.
- **Record count grows with the tile count.** An npu1 s8 conv queue has
  about 7,200 records at N=10,000 and about 71,000 at N=100,000. The program
  is unrolled per tile, not parameterized.
- **Only two npu3 groups have more than one member.**
  - The group with M = 32/48/64/96 at K=256, N=10,000 (4 models) fits
    exactly. 386 of its 400 varying values are exact polynomials in M under
    leave-one-out. The other 14 are the segment-0 rotation.
  - The 11-member group varies in both M and K.

**Small sweeps (151 Pulsar2 builds, `B<=64, M<=64, K<=128, N<=16`).**

- **They fall into 30 structures.** A structure change is a new tiling or a
  new record form, and it happens at thresholds in M, K and batch.
- **Most values are exact functions of the swept dimension.** Inside the
  four structures with enough points, nearly every varying shape value fits
  exactly under leave-one-out. The fitted forms are `p`, `ceil(p/4)`,
  `p mod 4`, `(p/4) mod 2` or `2^p - 1`.
- **These did not fit:**
  - Two pairs of same-register records swap values at a threshold
    (M >= 24: `0x0500`, `0x0730`).
  - A bitmask spills into a second word once K > 32.

So a shape emitter would be possible inside one small structure. It is not
implemented, because none of these structures reaches a step shape. The
step's operands run up to 12,544 x 4,608 x 16, which is deep in the
unrolled-tile regime. There, a new shape means a new structure.

**npu1 vs npu3.** The two compilers share nothing at the record level.

- npu1 emits 5 queues and npu3 emits 15 (three per engine).
- The npu3 s8 TENG queue is a fixed 100-record program per core, and
  npu1's has 132-140 records.
- Our Pulsar2 7.0-lite builds are `npu_mode: NPU1`.

A template therefore has to come from the same compiler generation and NPU
mode as the step. Nothing transfers between them.

## Coverage of the step's 62 live-operand nodes

The step has 41 MatMuls, 1 Gemm and 20 Convs. The shape tables are
`STEP_SHAPES` and `STEP_CONV_MATMULS`.

| nodes | shape(s) | status |
| ---: | --- | --- |
| 1 | Gemm `[16,512]x[512,1000]` + bias | **covered offline**: `t_step_attr/ladder/mm_add` template, every lane explained. The Gemm's `transB` Transpose is a separate op |
| 3 | dX 512->512 3x3, `[1,1,512,4608]x[16,1,4608,49]` in its Gather chain | **covered offline**, cross-validated both ways |
| 4 | dW 64->64 3x3, `[16,1,64,3136]x[16,1,3136,576]` | template exists (`t_transpose_compose_real`, all 612 lanes explained). Its two input scales tie in float32, so it refuses as soon as they separate. **Rebuild** with distinct input ranges |
| 2 | fc dX `[16,1000]x[1000,512]`, fc dW `[512,16]x[16,1000]` | needs builds (batch 1) |
| 16 | the other 9 dX shapes (Gather chain form) | needs builds (batch 2) |
| 16 | the 10 other dW shapes | needs builds (batch 3) |
| 20 | Conv, as `act_weight_conv_to_matmul` MatMuls: 11 shapes | needs builds (batch 4). Pulsar2's own `ActWeightConv` path fails (`legalize.act_weight_conv_to_matmul`) |

### Build batches

Each item is one template build. Every template uses any calibration, but
with its two operands calibrated to clearly different ranges.

1. **fc:** 2 shapes.
2. **dX:** 9 shapes, each built in the chain form it takes in the step
   (Gather -> Mul -> Reshape -> MatMul). Composition rewrites MCode, so a
   standalone MatMul is not a template for a fused chain.
3. **dW:** 11 shapes. That is the 10 remaining, plus the 64->64 rebuild.
4. **Forward Conv:** 11 legalized-MatMul shapes.

That is 33 builds in total. Once they exist, no further build is needed for
any recalibration of these nodes.

## Reproduction

```
matmul_record_emit.py explain BUILD_DIR            # every lane and its formula
matmul_record_emit.py check TEMPLATE_DIR TARGET_DIR
matmul_record_emit.py emit TEMPLATE_DIR NEW_quant_axmodel.json OUT.axmodel
```

The shape-fit and corpus-grouping scripts used for the numbers above are in
`~/npu-scratch/t_matmul_record_emit/`: `group.py`, `fitgroups.py`,
`small.py`, `fitbasis.py`, `all_pairs.py` and `role_stats.py`. They are not
committed, because they read uncommitted builds.
