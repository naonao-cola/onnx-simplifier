# Does `emitter.py`'s bit-permutation weight learner work at real ResNet18 Conv shapes? Yes for a 1x1 downsample, end to end; partially for a 3x3

`docs/axera-step-attribution.md` found Conv is 29-35% of a real training
step's MCode -- the second-largest share after the DMA queue -- and flagged
`emitter.py`'s weight-table machinery as validated only at toy 1-8 channel
8x8 shapes. This checks it, for the first time, against two real ResNet18
training-step Conv shapes: the two downsample paths between stage 1 and
stage 2 (`[16,64,56,56] -> [16,128,28,28]`, both with a real trained bias).

## What worked unmodified

`emitter.py`'s core claim -- compile the same shape `k` times with different
weights, read each `npu_params` bit's origin off the k-way signature, then
write arbitrary new weights into a reference build without Pulsar2 --
generalises to both real shapes' **weight-code bits**, the large majority of
the table:

| shape | code bits (`Cout*Cin*K*K*8`) | `k` for zero collisions |
| --- | --- | --- |
| 1x1 downsample (`[128,64,1,1]`) | 65,536 | 32 |
| 3x3 downsample (`[128,64,3,3]`) | 589,824 (9x more) | 38 |

Both `k` values were found empirically -- build a batch, check
`emitter.collisions()`, add more if nonzero -- not assumed from the
docstring's own rough estimate ("`k`=15" for a much smaller reference shape).
The real, previously-uncharacterized cost this adds: **`k` scales with the
layer's code-bit count**, roughly doubling the collision-free threshold for a
$9\times$ larger layer rather than needing only a few more builds. For a
project already running production ResNet Conv shapes (64-512 channels, up
to $1000\times512\times9\times8 \approx 4.6\text{M}$ code bits for the widest
real layer), a build budget in the dozens per distinct shape, not the 4-6 the
docstring's own toy-shape validation used, should be budgeted for up front.

## The real bug: `requant_block()` silently assumes bias-free convolutions

The first full emission attempt reproduced the weight-code region exactly
but left the per-channel bias/scale block (`emitter.requant_block()`'s
output) wrong by up to 6.88 in output-code units -- large enough to visibly
corrupt the result. Every real ResNet18 conv has a trained bias input;
`requant_block()`'s formula, `bias[c] = zy - zx*sum(q_c)*m_c`, has no term
for it at all. Deriving the missing piece directly from the quantised-conv
math (`y_code = zy + m_c * sum_i (x_code_i - zx)*q_i + bias_c/y_scale`) and
adding `+ bias_c/y_scale` reproduces the real compiled table to a max
absolute difference of 0.00044 -- the same order as `requant_block()`'s own
documented ~6e-5 float32-rounding tolerance, against the existing formula's
6.88. `scripts/axera/conv_bias_requant.py`'s `requant_block_with_bias()` is
this fix, as a new function rather than an edit to the shared `emitter.py`;
whoever owns that module should fold it in, since every real biased Conv
this project's own README documents hits this.

## The other bug: `emit_mcode`'s zero-point detector needs a much lower agreement threshold at real scale, and is still over-conservative

Beyond the table, the compiled mcode also bakes in the output quantisation
(`y_scale`/`y_zero`) as a literal. Two separate problems here, both real:

1. **`learn_mcode`'s default `min_agreement=0.9` misses the field entirely**
   at real scale. The wide weight-magnitude sweep needed to get good
   code-bit disambiguation (weight scale from 0.02 to 0.6, to spread
   `y_scale` enough for `learn_mcode` to have anything to key on) also pushes
   15 of 32 builds (47%) into the "unpatchable/shifted stream" case
   `emitter.py`'s own docstring already names as a real, if supposedly rare
   ("roughly one value in fifty"), phenomenon. At `min_agreement=0.5` it
   finds the right offset (`zero_offsets=[4010]`, `scale_offsets=[4032,
   4040, 4048, 4056]`); at 0.9, 0.83, and 0.7 it finds nothing.
2. **Even found, `emit_mcode()` refuses this exact holdout.** The specific
   held-out weight set's zero point (135) is in the learned `unpatchable`
   set -- because *some other* training build sharing that zero point showed
   a shifted layout, not because 135 is inherently unwritable at *this*
   offset. A direct comparison against a real native rebuild of the holdout
   shows offset 4010 patches cleanly (confirmed byte-exact against the real
   rebuild, and confirmed correct on device below) -- `emit_mcode`'s
   per-value unpatchable classification is coarser than the actual
   constraint and produces a false refusal here. `patch_scales.py`'s own
   `find_scale_slots` mechanism (bf16/bf16-reciprocal/float32 literal
   search) independently handles the *scale* half correctly (4 `x` slots as
   `bf16recip`, 4 `y` slots as `f32`) without hitting this problem at all,
   since it patches by byte-pattern search rather than by
   per-build-outlier classification; only the zero-point half needed a
   separate, manually-confirmed byte offset here.

Neither of these is fixed in the codebase -- `docs/`/`scripts/axera/
conv_bias_requant.py` document and pin the offset (`test_shape1_zero_point_
byte_is_at_the_documented_offset`), not patch `emitter.py` itself.

## Device verification: 1x1 downsample, full pipeline

Combining the bias-corrected block, `patch_scales.patch_model()` for the
scale literals, and a direct single-byte overwrite of the zero-point offset,
against the real held-out weight/bias tensor, on the AX8850 (`axcl-vm`,
serialized under the shared device lock, a native-build control run before
and after, both clean):

| check | max abs error | vs. |
| --- | --- | --- |
| control (native reference build) | 0.0062 | numpy conv2d |
| native holdout (unmodified Pulsar2 build) | 0.0384 (1.04% of peak 3.70) | numpy conv2d |
| **emitted holdout (this pipeline)** | **0.0384** (identical to native) | numpy conv2d |
| native vs. emitted, device output directly | 0.0293 | -- |

The emitted model's error against numpy is *exactly* the native build's own
quantisation noise, not a larger, systematic error -- the 17 mcode bytes
still unexplained after the zero-point fix (a cluster near offset 1242-1256
plus four scattered single bytes) left at the reference's stale value are
confirmed numerically harmless at this precision. This is the first
end-to-end, real-shape, device-confirmed Conv weight emission this project
has produced outside the toy 1-8 channel fixtures.

## 3x3 downsample: the weight codes generalise; the auxiliary block does not have a simple layout

The weight-code bit-permutation portion is independently validated the same
way: `k=38`, zero collisions, every table byte the learned map names a code
bit for matches the real compiled table exactly
(`test_shape2_3x3_downsample_weight_codes_match_native`).

The per-channel bias/scale block is where 1x1 and 3x3 diverge. What looked
at first like a second confirmation -- searching for each of the 128
computed `m = x_scale*w_scale/y_scale` values anywhere in the table found
most or all of them present -- turned out to be a **false positive from
searching without requiring contiguity**. Reading the table at the offset
the search reported (`[18560:19072)`, the 1x1 shape's analogous span)
directly: only the first 32 entries (channels 0-31) are valid `m` values;
the rest is unrelated data (values up to $10^{38}$ in magnitude, clearly not
scale floats). The 128 output channels are split into (at least) four
32-channel groups, at table regions starting near byte offsets 18432,
37120, 55808, and 74496 -- consistent with this project's own repeatedly
found 32-channel Conv tiling boundary (`README.md`'s "cout"/"cin"
tiling-granularity sections) and the same *class* of finding the Transpose
and DMA-queue tile-table decodes made elsewhere in this project: a real,
present, but non-trivially-ordered per-group structure, not a flat array.
Only one channel's value (channel 0, the first of the first group) is
confirmed at a fixed, reliable offset; the ordering within and across the
other three groups was not decoded further here (time-boxed: this was a
secondary check within a two-shape directive, not this task's main
question).

**Per the directive's own instruction not to force a uniform conclusion:**
the two shapes behave differently in exactly the place `README.md`'s prior
Conv-format work already suggested they would (1x1 vs. wider-kernel/
multi-channel convolutions use different underlying layouts) -- the
weight-code bit-permutation technique itself is shape-agnostic and confirmed
working on both, but the smaller, per-channel auxiliary data needs its
layout characterized separately per format family, the same way the rest of
this project's Conv-format reverse engineering has always found necessary.

## What this means for the emitter's coverage

- **Validated, end to end, on real hardware:** a 1x1, stride-2, biased,
  128x64-channel Conv -- the exact shape of one of ResNet18's four
  stage-transition downsample paths.
- **Validated, weight-code portion only:** a 3x3, stride-2, biased,
  128x64-channel Conv -- the other stage-transition downsample path. Not yet
  a working end-to-end emitter; the auxiliary block's layout is an open,
  scoped gap.
- **Not attempted here:** the plain 3x3 stride-1 same-channel blocks (the
  majority of ResNet18's 20 real Conv nodes), the 7x7 stem, and whether this
  pipeline's outputs survive composition with a real neighbour the way
  `docs/axera-transpose-compose-real.md` found composition rewrites MCode
  for other ops -- this project's Conv weight table is patched in place on a
  standalone reference, and whether that reference's own MCode needs to
  match a *composed* build's has not been checked for Conv specifically.

## Reproduction

Fixtures under `scripts/axera/fixtures/conv_learn_downsample/`: one
reference build and one held-out native build per shape (both gzipped
`.axmodel`s plus their `quant_axmodel.json`s), a `k`-build-learned map per
shape (`emitter.save_map`/`load_map` format), and the held-out weight/bias
tensors. `tests/test_axera_conv_weight_learn_downsample.py` reproduces every
claim above except the device numbers, with no Docker/device required.

The `k`-build campaigns themselves (32 and 38 Pulsar2 builds respectively,
scratch under `/home/takecheeze/npu-scratch/t_conv_learn_downsample`) are
the expensive, non-reproducible-from-the-repo part; the committed fixtures
are their output, not the campaigns themselves.
