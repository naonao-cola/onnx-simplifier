# Does isolating calibration from shape clean up the non-fused Reshape sweep?

[`axera-reshape-dma.md`](axera-reshape-dma.md) swept the ResNet18 `Cin = Cout`
weight-fold family (`[C,C,3,3] -> [1,C,C,9]`, then `Relu`) and found "no rule":
even equal-length programs at `C=40` and `C=64` differed in 398-519 bytes. This
checks one thing that sweep never controlled for -- calibration.

**Short version.** The premise that motivated this ("every prior build used
freshly-random-per-build calibration data") is wrong: `reshape_dma_sweep.py`
seeds `np.random.RandomState(0)` the same way for every build. But `shape_in`
varies across the sweep, so each build draws a different number of samples
from that fixed-seed stream, and the realised min/max -- hence the compiled
`y_scale`/`y_zero` -- drifts slightly with shape. That drift is real (confirmed
below) but small. Holding it exactly fixed does not fix the "no rule" finding
in general, but it cleanly separates two things the noisy sweep conflated:
some shape pairs turn out to be very close once calibration is controlled
(`C=40` vs `C=64`: 29 bytes, not 398-519), revealing three clean affine fields;
other shape pairs in the same length bucket remain hundreds of bytes apart
regardless, because they use a different, undecoded byte layout at the exact
same offsets. Calibration was hiding which shapes belong to which of those two
groups, not the underlying rule itself.

Method: float32 `Reshape([C,C,3,3] -> [1,C,C,9]) -> Relu` (standalone `Reshape`
does not compile), `pulsar2 build --target_hardware AX650` (Pulsar2 7.0-lite),
`*_neu` MCode compared with the 301-325 noise window ignored. Calibration
datasets are engineered per `fixed_range_samples()`: four samples per build,
each with its first two elements pinned to exactly -0.9 and +0.9 so MinMax
calibration derives the identical `y_scale`/`y_zero` regardless of `C`.
Harness `scripts/axera/reshape_calibration_isolation.py`, scratch
`/home/takecheeze/npu-scratch/t_reshape_calib`.

## 1. Calibration alone changes real bytes, but few of them

Same shape (`[32,32,3,3] -> [1,32,32,9]`), same seed-0 RNG draw pattern, two
calibration ranges (`-0.9..0.9` vs `-9.0..9.0`): the compiled MCode is the same
length (2208 bytes) and differs in exactly 28 bytes outside the noise window,
at two clusters (1079-1106, 1578-1605 -- the `Relu`-side scale/zero-point
literals, consistent with `axera-dma-queue.md`'s tail-field findings for other
elementwise ops). 28 of 2208 bytes is real, but it is an order of magnitude
smaller than the 398-519 bytes the prior sweep attributed to a shape step from
`C=40` to `C=64` -- so calibration cannot be the dominant cause of that
sweep's irregularity by itself.

## 2. With calibration fixed, two shapes can be nearly identical

`C` in `{40, 44, 48, 52, 56, 60, 64}`, segment 2 pairwise byte differences
(`matrix` subcommand; `x` marks a different segment length):

```
       40   44   48   52   56   60   64
 40     0  353  672  673  203  200   29
 44   353    0  401  677  536  537  360
 48   672  401    0  624  672  671  672
 52   673  677  624    0  672  673  674
 56   203  536  672  672    0   16  217
 60   200  537  671  673   16    0  211
 64    29  360  672  674  217  211    0
```

`C=40` vs `C=64`: 29 bytes (was 398-519 in the confounded sweep). `C=56` vs
`C=60`: 16 bytes. Every other pair is 200-680 bytes apart. So `{40, 56, 60,
64}` cluster together (16-217 bytes) while `{44, 48, 52}` sit far from
everything (350-680+ bytes), including each other. Within the first cluster
there is a further split: `{40, 64}` and `{56, 60}` are each internally very
tight (16-29 bytes) but about 200-217 bytes apart from each other -- a nested
structure, not one flat group.

## 3. Three affine fields, validated across the whole `{40,56,60,64}` cluster

Diffing `C=40` vs `C=64` and `C=56` vs `C=60` byte-for-byte and checking each
differing offset against simple functions of `C` finds three fields that hold
exactly for all four members of the cluster (`decode_fields`/
`fields_match_formula`; absolute offsets into segment 2):

| offset | width | value | check (C = 40, 56, 60, 64) |
| --- | --- | --- | --- |
| 537 | uint16 LE | `36*C` | 1440, 2016, 2160, 2304 |
| 550 | uint8 | `C-1` | 39, 55, 59, 63 |
| 555 | uint8 | `C-1` (repeated) | 39, 55, 59, 63 |
| 825 | uint16 LE | `C*C//2 - 1` | 799, 1567, 1799, 2047 |

`36*C` is the weight row's byte count (`4 bytes * 9 (K*K)` per channel `* C`,
i.e. a `[C, 9]` slice at 4 bytes/element). `C*C/2 - 1` looks like an
element-index into half of the `[C,C]` tensor, the same "count minus one"
convention this project has found everywhere else (Gather's `4*R*C-1`, the
Transpose size immediates). These are the first clean, validated affine
fields found for any `Cin=Cout` ResNet18 Reshape shape; the prior sweep's
"17-90 blocks change per step" conclusion did not separate them from the
unmatched-cluster noise.

The remaining 29 (or 16) bytes between cluster members are **not** explained
by these three fields -- they sit at other offsets (e.g. 756-767, 841-845,
975-1088 for the `40` vs `64` pair) and were not decoded further.

## 4. `{44, 48, 52}` do not fit -- a second, undecoded layout

At the identical four offsets, `C=44/48/52` hold different values that do not
match any of the three formulas (`fields` subcommand):

```
44 row_bytes_36c=33328 c_minus_1_a=3   c_minus_1_b=3   half_c_sq_minus_1=50947
48 row_bytes_36c=33472 c_minus_1_a=3   c_minus_1_b=3   half_c_sq_minus_1=32515
52 row_bytes_36c=10945 c_minus_1_a=32  c_minus_1_b=1   half_c_sq_minus_1=800
```

Segment 2's total length is identical (1248 bytes) for all of `C=40..64`, so
this is not a length-driven layout switch the way the untiled/tiled boundary
elsewhere in this project is -- it is a same-length, different-byte-layout
split, the same character as the discrete "form" switches
`docs/axera-transpose-mcode.md` and `docs/axera-teng2-sqrt-blocks.md` found
and could not predict either.

**No rule found for cluster membership.** Checked and rejected: `C mod 4`
(all seven values are `0`), `C mod 8` (`{40,48,56,64}` are `0` but only three
of those four are in the matched cluster; `60` is `4 mod 8` and *is* in the
cluster), `C mod 16`, `C mod 24`. None separate `{40,56,60,64}` from
`{44,48,52}`.

## What this changes

- The calibration-confound hypothesis was directionally right but for a
  different, more limited reason than assumed: it did not free the whole
  family, only revealed which shapes already shared a layout.
- Three affine fields are now decoded and validated for the
  `{40,56,60,64}` sub-cluster -- new content, not in
  `axera-reshape-dma.md`.
- The core blocker is unchanged: which `C` gets which discrete layout is
  still not predictable, so there is still no emitter. Building one would
  need either the actual layout-selection rule or an explicit lookup table
  of measured-safe `C` values, the same shape `reshape_dma_emit.py`'s
  `EXACT_CO` set already takes for the `Cin=8` family -- but here even that
  needs many more builds to map cluster membership before it would be safe.

## Not done

- Mapping which `C` values (beyond the seven built) belong to the matched
  cluster, to see if cluster membership has any pattern at all or is
  effectively unpredictable per-shape.
- Decoding the remaining ~16-217 bytes that still separate matched-cluster
  members, or the `{44,48,52}` layout.
- The activation-fold families (`[N,C,H,W] -> [N,1,C,H*W]`) and the real
  ResNet18 shapes (batch 16) -- this only reprises the small `Cin=Cout`
  weight-fold sweep `axera-reshape-dma.md` already used.
- Any device run: nothing here reaches a validated, emittable case.

## Reproduce

```
uv run --no-project --with onnx --with numpy python \
    scripts/axera/reshape_calibration_isolation.py sensitivity WORK_ROOT
uv run --no-project --with onnx --with numpy python \
    scripts/axera/reshape_calibration_isolation.py sweep WORK_ROOT 40 44 48 52 56 60 64
uv run --no-project --with onnx --with numpy python \
    scripts/axera/reshape_calibration_isolation.py matrix WORK_ROOT 40 44 48 52 56 60 64
uv run --no-project --with onnx --with numpy python \
    scripts/axera/reshape_calibration_isolation.py fields WORK_ROOT 40 44 48 52 56 60 64
```

`tests/test_axera_reshape_calibration_isolation.py` checks the decode logic
against seven committed fixtures under
`scripts/axera/fixtures/reshape_calib_isolation/` -- no Docker/device needed.
