# AX650 elementwise, reduction and activation ops: what the emitters reach

Goal: extend byte-level generation of Pulsar2 `.axmodel` files toward the ResNet18
training step, whose non-memory ops are Mul 397, Add 144, Div 52, Sub 46,
ReduceSum 44, Sqrt 42, MatMul 41, Conv 20, Cast 19, Greater 18, Relu 17, Softmax 3,
Log 2, Neg 2, MaxPool 1, ReduceMean 1, Gemm 1, Less 1. This records an audit of the
existing emitters, a shape sweep of the highest-count uncovered ops, and one new
emitter (`scripts/axera/elementwise_emit.py`) with its evidence and limits.

**Short answer.** Every existing emitter needs a compiler-built reference at the
target's own shape, and the fixtures cover only tiny shapes. This work adds a
shape-and-range retargeter for standalone `Relu` and `Sqrt` on `x[1, n]` for
`49 <= n <= 63`, `n % 8 != 0`. It is byte-exact on 8 of 12 blind builds and refuses
the rest of what was measured. No training-scale shape is generated: those programs
are tiled and were characterised only by size.

## Audit: what the code and tests reach today

Read from `scripts/axera/tiny_emit.py`, `emitter.py`, the fixtures and tests; nothing
here is guessed. "Needs same-shape reference" is the docstring's own condition.

| op (count in step) | emitter / patcher | shapes with fixtures | training shapes |
| --- | --- | --- | --- |
| Neg (2) | `emit_neg`: rewrites the four stride-7 output-scale copies | `[1,8]` | no; needs a same-shape reference |
| Add (144) | `emit_add` (output scale), `patch_add_output_zero_point` | `[1,16]`, `[1,32]`, `[1,64]`, `[2,32]` two-live-input | no; docstring: "reference must already have the target's shape" |
| Sub (46) | `patch_sub_output_scale`, `patch_sub_output_zero_point` | `[1,16]`, `[2,32]` | no |
| Mul (397) | `patch_mul_scales`, `patch_mul_output_quad`, `patch_mul_zp_x` (hardware-confirmed bit-exact on one build pair) | `[1,8]`, `[1,16]`, `[2,32]` | no |
| Div (52) | none: fixtures and field-decode tests only (`grep -i div tiny_emit.py` is empty) | `[1,16]`, `[2,32]` | no |
| Relu (17), Sqrt (42) | none before this work: `trace_relu` is a tinygrad matcher, and the Relu hardware test only runs `[1,8]` | `[1,8]` | no; **this work** adds `49 <= n <= 63` |
| Greater (18), Less (1), Cast (19), ReduceSum (44), ReduceMean (1), Log (2), Softmax (3), MaxPool (1) | none | none (Softmax only inside `attn_qkv_softmax`) | no |
| MatMul (41), Conv (20), Gemm (1) | many small-shape patchers (`patch_site_a`, `emit_*_reg8_*`, ...) documented elsewhere | many | not audited here |

Two facts that bound everything else:

- Three training-step MCode streams are in the corpus (`adam_update_fp32`,
  `toy_training_step`, `w2v2fe_training_step`); they are used for structural
  decoding, not generation.
- The existing patchers rewrite float words or fields *by value* inside a
  reference stream. None derive a stream from a shape.

## Shape sweep of standalone ops

Pulsar2 7.0-lite, AX650, MinMax, Numpy calibration. `x[1,n]` for 31 values of `n`
from 8 to 131072, five ops, calibration endpoints pinned so scales do not vary
with the random data (`scripts/axera/elementwise_sweep.py`, names starting with
`p`). 1x200704 and Sqrt above 65536 elements are separate rows below.

| op | distinct programs over 31 shapes | `npu_params` | notes |
| --- | --- | --- | --- |
| Relu | 31 (noise window masked) | 40 B, 100 B at `n` = 65535..65536, 220-240 B above | length classes for `n <= 72`: 1952 (`n <= 16`), 1984, +32 for `n % 8 == 0` above 32 |
| Sqrt | 29 | 40 B, 100 B at 65535..65536 | **fails above 65536 with `integer N does not fit 'uint16_t'`** in `AxQuantizedSqrt` |
| Greater -> Cast | 31 | 44 B, 64 B, 164 B | MCode length 2368 for every `n <= 16384`, but the bytes still differ per shape; not decoded |
| Add (two live) | 31 | 64 B, 182 B, 362 B, 402 B | two coin-flip forms, see below |
| Mul (two live) | 31 | 60 B, 180 B, 360 B, 400 B | same |

Every one of the 31 shapes gives a different program, so the answer to "is it
shape-templated?" is "yes, but a distinct template per shape unless the
n-dependent bytes are decoded". For Relu and Sqrt they can be, below.

**Rebuild noise.** Three rebuilds of `n=64` per op:

| op | rebuild differences outside 301-325 |
| --- | --- |
| Relu, Sqrt, Greater -> Cast | none (2-8 bytes, all inside the window) |
| Mul | one pair differs in 1018 bytes, the other rebuild in 2 |
| Add | two rebuilds of the same shape had different lengths (2808 and 2776) |

That is the same coin-flip behaviour the README documents for Mul, Gemm and MatMul
tables. It is why the decode below used Relu and Sqrt, and why Mul and Add were not
attempted: any diff between two Mul or Add builds is ambiguous without a per-shape
census of both modes.

**Sqrt's element limit is per dimension.** `[1,200704]` and `[1,65537]` fail (the
stored `n-1` must fit 16 bits), but `[512,512,3,3]` (2.36M elements),
`[64,64,3,3]` and `[1000,512]` build. So the Adam update's Sqrt is not blocked by
element count, only by a last dimension above 65536.

## Training-scale points (sizes only)

| build | MCode | `npu_params` | segment sizes |
| --- | --- | --- | --- |
| Relu `[16,64,56,56]` | 3648 | 1280 | 64, 32, 2752, 32, 32 |
| Mul `[16,64,56,56]`, two live | 12,320 | 1920 | 64, 32, 8544, 512, 2240 |
| Add `[16,64,56,56]`, two live | 13,120 | 1922 | 64, 32, 9216, 512, 2368 |
| Mul `[512,512,3,3]`, two live | 9632 | 1920 | 64, 32, 6272, 512, 1824 |
| Sqrt `[64,64,3,3]` | 2080 | 40 | 64, 32, 1184, 32, 32 |
| Sqrt `[512,512,3,3]` | 3328 | 1280 | 64, 32, 2432, 32, 32 |
| Sqrt `[1000,512]` | 2856 | 160 | 64, 32, 1312, 224, 480 |

`npu_params` grows in multiples of 40 B (tile offset entries) once the tensor is
large, as in the Transpose and Gather work. None of these was diffed against a
neighbouring shape, so no claim is made about how they vary.

## The Relu and Sqrt emitter

For `49 <= n <= 63`, `n % 8 != 0`, and the same calibration style, the streams of
`Relu` and `Sqrt` are **byte-identical apart from the 301-325 noise window and two
kinds of field**:

- **Five size immediates**: the low bytes of `4n-1` (offsets 878 and 1412, 1393 for
  Sqrt), `n-1` (offset 1000), and `4n` (offsets 1776 and 1936). In the sweep, blocks
  `49..55` and `57..63` differ only there. Below 49 the streams reflow by hundreds
  of bytes per shape (blocks 9..47) and above 63 they change form; not decoded.
- **Float32 scale words** that depend on the calibration range, four copies each:
  `1/x_scale` and `x_scale` for Relu (offsets 1069 stride 8 and 1368 stride 7);
  `1/x_scale` (1068, stride 8), `x_scale` (1180, stride 7) and the output scale
  `r_scale` (1224, stride 8) for Sqrt. Fitted formulas, exact on every matching
  build: `x_scale = f32(f32(hi - lo) / 255)` with the range widened to include 0,
  `recip = f32(1 / x_scale)`, and `r_scale = f32(f32(sqrt(hi)) / 255)`.

`elementwise_emit.py` copies the committed template model (`n = 49`, Relu
`[-0.9, 0.9]`, Sqrt `[0.1, 0.9]`), patches the scale words (8 for Relu, 12 for Sqrt) and the five bytes, updates
the input, output and `outputs_info` shapes, and refuses everything else.

### Evidence, in the order it was gathered

| set | builds | result |
| --- | --- | --- |
| fit: `n` = 49..55, 57..63, template ranges | 28 | exact (derived from these) |
| A: new seeds and two rebuilds at the template range (9), plus Relu +/-2, Sqrt `[0.05,2]` and `[0.3,0.6]` (3) | 12 | 12 exact. The nine were predicted before any scale formula existed; the three with other ranges are where the scale families were found, so they are not blind |
| A': asymmetric Relu ranges `[-0.5,1.5]`, `[0,1]`, `[-1,0.25]` | 3 | reflow ~365 bytes each; now refused |
| B: eight more symmetric Relu and five Sqrt ranges | 13 | exact after fixing two formula errors (one-ulp scale rounding) found here; **not blind** |
| C: untouched shapes and ranges, formulas frozen | 12 | **8 exact, 4 failed** |
| D: rebuilds and neighbouring ranges of C's failures | 18 | failures are deterministic |

Set C is the honest test. Its four failures are:

- Relu `+/-0.77` and `+/-0.333` reflow (~368 bytes), each on three separate builds,
  while `+/-0.76`, `+/-0.78` and eighteen other symmetric ranges match the
  template. No rounding formula for the zero point separates them (checked in
  float32 and float64); the trigger is unexplained.
- Sqrt `[0.7, 0.8]`: same structure, but `r_scale` is one ulp above the formula, on
  three builds. All matching builds had `lo/hi <= 0.5`.
- Sqrt `hi = 1.0`: a different program (length 1920, five builds; here
  `r_scale == x_scale`).

The refusal list (`_measured_range_problem`) was written **after** seeing these, so
it is not itself held-out validated. Its limit is stated in the code: a range that
nobody built may still reflow, and only measured failures can be refused. Relu
symmetric ranges: 20 of 22 built match.

### Device check

Ten emitted models ran on the AX8850 (`axcl-vm`, firmware V3.6.5) under
`flock /tmp/axcl-device.lock`, with a control run before and a health run after,
inputs inside the calibration range, compared to `numpy`:

- Relu `n` = 49, 55 (`[-0.9,0.9]`), 61 (`+/-2`), 52 (`+/-0.3`), 63 (`+/-7.5`):
  maximum error 0.47-0.5 of an input LSB (the quantisation step of `x_scale`).
- Sqrt `n` = 49, 57 (`[0.1,0.9]`), 53 (`[0.1,1.25]`), 61 (`[0.02,0.3]`), 52
  (`[0.5,3.7]`): 0.73-0.91 of an output LSB (`sqrt(hi)/255`).
- The refused ranges, emitted with `allow_unmeasured=True` only to see whether the
  predicted bytes still work: Relu `+/-0.77` 0.47 LSB, `+/-0.333` 0.44, Sqrt
  `[0.7,0.8]` 0.68, `[0.1,1.0]` 0.79, `[0,1.0]` 1.36. They stay refused because the
  bytes are not what Pulsar2 would build, but they do not look functionally wrong.

Tests: `tests/test_axera_elementwise_emit.py` (34 cases): seven compiler-built
oracles compared outside the noise window, template self-consistency, refusals,
and a guard that the known Relu `+/-0.77` mismatch still exists.

## What this does not do

- No training-scale shape is emitted. `n` is limited to 49..63 on a single row.
  The size immediates appear to be `4n-1`, `n-1` and `4n`, but the ResNet18 shapes
  are 4-D and tiled, and how their programs relate to these is unknown.
- Relu and Sqrt only, float32, standalone, one calibration style. In a fused graph
  the MCode changes (the Transpose work saw 72 bytes from a `Relu` consumer).
- Mul, Add, Sub, Div, Greater, Cast, ReduceSum, Log and the rest are unchanged: no
  new emitter, and Div still has no patcher. Add and Mul need the two-mode census
  first. ReduceSum and Softmax were not swept.
- Two decode results are patterns, not rules: the `4n-1`/`n-1`/`4n` immediates (five
  values of `n` per block, two blocks, then held-out builds) and the scale formulas
  (13 to 15 builds each).

## Next steps

1. Repeat the block analysis of `n` between 8 and 48 for Relu (the reflowing region)
   to see whether it is the same five fields plus a form switch, as in the
   Transpose alignment classes.
2. Diff Relu `[16,64,56,56]` against `[16,64,56,28]` and other 4-D neighbours
   to see whether the size immediates generalise to tiled shapes.
3. Census the two modes of Add and Mul rebuilds across shapes before decoding them;
   Mul and Add are 541 of the step's 1104 nodes.
