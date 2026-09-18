"""Continues `tests/test_axera_conv_zpx_binary_cluster_collision.py`
(PR #1636)'s own explicitly flagged next step: that file traced Conv's
`reg=60` "binary path switch" to the SAME register as `patch_conv_zp_x`'s
own literal zero-point field (`reg=54`), and offered a precise but
explicitly UNPROVEN hypothesis -- if `reg=54` really is Conv's literal
input zero point, the observed non-determinism at `RandomState(0)`
calibration may be a genuine floating-point ROUNDING near-tie in the
zero-point calculation itself, not an unrelated weight statistic. PR
#1636 said confirming this "would require recomputing calibration
statistics from the original build scripts' own RNG-seeded data, not
attempted here."

This file does exactly that recomputation, independently, and the
result is a strong, quantitative confirmation of both halves of the
hypothesis.

## The formula, cited from this project's own established source

`scripts/axera/pulsar2_quantizer.py`'s own module docstring documents,
from a real `pulsar2 build` output's own `AxQuantizedConv` node
attributes, that Conv's input activation is quantized U8 (uint8),
per-tensor, asymmetric, with `quant_method = 0` matching this project's
own `"calibration_method": "MinMax"` build config -- and that
`onnxsim.quantize_static(..., method="minmax")` is this project's own
authoritative (not reimplemented) numeric match for that scheme. The
actual C++ formula, `ComputeAsymmetricUint8QuantParams`
(`onnxsim/passes/static_quantize_matmul.h`, read directly, not
paraphrased from memory):

```
lo = min(0.0f, min_val)
hi = max(0.0f, max_val)
if hi <= lo: hi = lo + 1.0f
scale = (hi - lo) / 255.0f
zp = round(-lo / scale)          # std::round: half away from zero
zero_point = clamp(zp, 0.0f, 255.0f)
```

`min_val`/`max_val` are the calibration range `onnxsim.calibration.calibrate()`
computes for `"minmax"` method: a running `(min(prev_min, batch_min),
max(prev_max, batch_max))` over every calibration batch, with no
outlier-clipping step (that only applies to `"entropy"`/`"mse"`, per
`calibrate()`'s own source, read directly) -- so for the `"minmax"`
method this project's build config actually uses, it reduces to the
plain min/max of every calibration sample.

`tests/test_axera_mcode_structure.py`'s own `_build_and_get_mcode_bytes`
(the function every `Conv(dilation=3,...)` fixture in this session's
own corpus was built through) generates calibration data as
`[RandomState(seed).randn(1, 4, 16, 16).astype(np.float32) for _ in
range(4)]` -- 4 samples, matching `x`'s own `[1, cin=4, insz=16, insz=16]`
shape exactly. `x` is the graph's only input and feeds `Conv` directly,
so it is exactly the tensor `patch_conv_zp_x`'s own name (zp_**x**)
already implies is being quantized.

## Result: exact match on every deterministic seed, and seed=0 is the
## uniquely close-to-a-tie outlier among all six seeds tested

| seed | computed `zp` (this formula) | computed `zp_f` (pre-round) | observed `reg=54`/`reg=60` (mcode) | match |
| --- | --- | --- | --- | --- |
| 0 | 126 (rounds down from 126.4593) | 126.4593 | `0x7e`=126 (group A, 5/8) / `0x7f`=127 (group B, 3/8) | matches group A exactly; group B is +1 |
| 1 | 113 | 113.0639 | `0x71`=113 | **exact** |
| 7 | 124 | 123.8465 | `0x7c`=124 | **exact** |
| 42 | 115 | 115.3153 | `0x73`=115 | **exact** |
| 100 | 116 | 115.8108 | `0x74`=116 | **exact** |
| 999 | 133 | 132.7820 | `0x85`=133 | **exact** |

**All 5 seeds this project has already confirmed fully deterministic
(1, 7, 42, 100, 999) match this independently-recomputed zero point
EXACTLY, zero exceptions.** This is direct, strong confirmation that
`reg=54` genuinely encodes the standard MinMax-calibrated asymmetric
uint8 zero point for `x`, computed by the formula above from the same
RNG-seeded calibration data every build script in this project already
uses -- not merely a plausible reading, a reproducible one.

**Seed=0 -- the ONLY seed this whole project has ever found non-
deterministic -- is quantitatively the closest of all six tested seeds
to a rounding half-boundary**, measured as the distance from `zp_f`'s
own fractional part to `0.5`:

| seed | `frac(zp_f)` | distance to `0.5` |
| --- | --- | --- |
| **0** | **0.4593** | **0.0407** |
| 42 | 0.3153 | 0.1847 |
| 999 | 0.7820 | 0.2820 |
| 100 | 0.8108 | 0.1892 |
| 7 | 0.8465 | 0.3465 |
| 1 | 0.0639 | 0.4361 |

Seed=0's own distance to the rounding boundary (`0.0407`) is more than
**4x smaller** than the next-closest seed (`42`, at `0.1847`), and
almost 11x smaller than the seed farthest from any boundary (`1`, at
`0.4361`). This is exactly the signature PR #1636's own hypothesis
predicted: if Pulsar2's real internal computation (which this project's
own `pulsar2_quantizer.py` module docstring already documents is NOT a
bit-exact reproduction of this onnxsim-based recomputation -- different
internal IR, different accumulation order, different fusion) lands
close enough to `126.5` for its own rounding decision to be sensitive
to compiler-internal floating-point accumulation order, a genuine
non-deterministic tie-break between `126` and `127` is exactly what a
real build would show -- and no OTHER tested seed comes remotely close
to that same knife's edge.

## What this establishes, precisely, and what it does not

**Established**: `reg=54` is very likely Conv's real, calibration-
derived input zero point (not merely a byte-pattern label this project
inherited from older work) -- the formula this project already has,
applied to the exact calibration data every relevant fixture's own
build script used, reproduces its value exactly for every seed known
to be stable, and correctly flags the one unstable seed as an outlier
in exactly the direction (proximity to a rounding tie) the earlier
hypothesis predicted.

**NOT established**: that this onnxsim-based recomputation is a
bit-exact reproduction of Pulsar2's own real internal computation
(explicitly documented elsewhere in this project as NOT the case);
that `126.4593`'s own specific fractional distance is "close enough"
to actually explain non-determinism by some externally-verified
threshold (no such threshold is established here, only a strong
RELATIVE comparison against the other five seeds); or why the true
Pulsar2-internal near-tie's own accumulation order varies
non-deterministically in the first place (still, as PR #1586/#1598/
#1600 already found, outside what this project's own visible mcode
bytes or offline recomputation can observe directly).
"""

import math
import os
import sys
import unittest

import numpy as np

_AXERA_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts", "axera")
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

# EXPECTED_REG60/REG54 values, cited directly from already-merged PRs
# (not recomputed from fixtures here -- this file's own job is the
# independent zero-point calculation, not re-decoding mcode; those
# values are already established with zero exceptions by
# tests/test_axera_conv_zpx_binary_cluster_collision.py and
# tests/test_axera_conv_binary_cluster_seed_survey.py).
DETERMINISTIC_SEEDS = {1: 0x71, 7: 0x7C, 42: 0x73, 100: 0x74, 999: 0x85}
SPLIT_SEED = 0
SPLIT_SEED_GROUP_A = 0x7E  # 126, 5/8 samples
SPLIT_SEED_GROUP_B = 0x7F  # 127, 3/8 samples


def compute_asymmetric_uint8_zero_point(seed, shape=(1, 4, 16, 16), n_samples=4):
    """Reproduces `ComputeAsymmetricUint8QuantParams`
    (`onnxsim/passes/static_quantize_matmul.h`) exactly, in float32
    arithmetic (matching C++ `float`), against the same
    `RandomState(seed)`-derived calibration data
    `tests/test_axera_mcode_structure.py`'s own `_build_and_get_mcode_bytes`
    generates for every `Conv(dilation=3,...)` fixture in this
    project's corpus. Returns `(zero_point_int, zp_f_pre_round)`.
    """
    rng = np.random.RandomState(seed)
    samples = [rng.randn(*shape).astype(np.float32) for _ in range(n_samples)]
    lo = np.float32(0.0)
    hi = np.float32(0.0)
    for s in samples:
        lo = min(lo, np.float32(s.min()))
        hi = max(hi, np.float32(s.max()))
    if hi <= lo:
        hi = lo + np.float32(1.0)
    scale = (hi - lo) / np.float32(255.0)
    zp_f = float(np.float32(-lo / scale))
    # std::round is round-half-away-from-zero, not Python's
    # round-half-to-even -- replicated explicitly, not via `round()`.
    zp = math.floor(zp_f + 0.5) if zp_f >= 0 else math.ceil(zp_f - 0.5)
    zp = max(0, min(255, zp))
    return zp, zp_f


class TestFormulaMatchesEveryDeterministicSeedExactly(unittest.TestCase):
    """The core finding: the standard MinMax asymmetric-uint8 zero-point
    formula, applied to the exact calibration data this project's own
    build scripts use, reproduces `reg=54`'s real observed value
    exactly for all 5 seeds this project has already confirmed fully
    deterministic -- zero exceptions."""

    def test_all_five_deterministic_seeds_match_exactly(self):
        for seed, expected in DETERMINISTIC_SEEDS.items():
            zp, zp_f = compute_asymmetric_uint8_zero_point(seed)
            self.assertEqual(zp, expected, (seed, zp_f))


class TestSplitSeedRoundsToGroupA(unittest.TestCase):
    """Seed=0's own computed zero point rounds to Group A's real
    observed value (0x7e=126, the more common of the two observed
    states, 5/8 samples) -- Group B (0x7f=127) is exactly +1, the
    single-count rounding-direction flip a genuine near-tie would
    produce."""

    def test_seed_zero_rounds_to_group_a(self):
        zp, zp_f = compute_asymmetric_uint8_zero_point(SPLIT_SEED)
        self.assertEqual(zp, SPLIT_SEED_GROUP_A)
        self.assertEqual(zp + 1, SPLIT_SEED_GROUP_B)


class TestSplitSeedIsTheClosestToARoundingTieAmongAllSixSeeds(unittest.TestCase):
    """The quantitative confirmation of the near-tie hypothesis: among
    all six tested seeds, seed=0 -- the ONLY one this whole project has
    ever found non-deterministic -- has its own computed zp_f closest
    to a rounding half-boundary, by a wide margin (>4x closer than the
    next-closest seed)."""

    def _dist_to_half(self, seed):
        _, zp_f = compute_asymmetric_uint8_zero_point(seed)
        frac = zp_f - math.floor(zp_f)
        return abs(frac - 0.5)

    def test_seed_zero_has_the_smallest_distance_to_a_half_boundary(self):
        all_seeds = [SPLIT_SEED] + list(DETERMINISTIC_SEEDS)
        distances = {seed: self._dist_to_half(seed) for seed in all_seeds}
        closest_seed = min(distances, key=distances.get)
        self.assertEqual(closest_seed, SPLIT_SEED, distances)

    def test_the_margin_is_at_least_four_x(self):
        all_seeds = [SPLIT_SEED] + list(DETERMINISTIC_SEEDS)
        distances = {seed: self._dist_to_half(seed) for seed in all_seeds}
        seed0_dist = distances.pop(SPLIT_SEED)
        next_closest = min(distances.values())
        self.assertGreater(next_closest, 4 * seed0_dist, (seed0_dist, distances))


class TestFormulaHandlesTheDegenerateRangeGuardCorrectly(unittest.TestCase):
    """Sanity check on the ported formula itself: the `hi <= lo`
    degenerate-range guard (all-zero calibration data) does not fire
    for any of this file's own real seeds (their calibration data is
    genuinely two-sided, spanning both negative and positive values),
    confirmed directly rather than assumed."""

    def test_no_seed_hits_the_degenerate_guard(self):
        for seed in [SPLIT_SEED] + list(DETERMINISTIC_SEEDS):
            rng = np.random.RandomState(seed)
            samples = [rng.randn(1, 4, 16, 16).astype(np.float32) for _ in range(4)]
            lo = min(np.float32(0.0), min(np.float32(s.min()) for s in samples))
            hi = max(np.float32(0.0), max(np.float32(s.max()) for s in samples))
            self.assertGreater(hi, lo, seed)


if __name__ == "__main__":
    unittest.main()
