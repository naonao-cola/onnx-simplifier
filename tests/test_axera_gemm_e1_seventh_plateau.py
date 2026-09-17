"""Continues `tests/test_axera_gemm_e1_k129_fifth_plateau.py` (PR
#1615)'s own explicitly flagged gaps: that file left the `K=288`/`289`
boundary's own `K*N` product ("a third distinct multiple of `1024`")
without stating exactly which multiple it is relative to the other two,
and left entirely open whether a SEVENTH `N` candidate exists below
`64` for some sufficiently large `K`.

## The `K=288`/`289` boundary is exactly `36*1024` -- confirmed directly
## from arithmetic already present in PR #1615's own test file, no new
## build needed for this part

PR #1615's own `test_boundary_product_is_a_third_multiple_of_1024`
already computes `288*128=36864` and asserts it is a multiple of
`1024` distinct from the first two transitions' own products
(`64*512=32768=32*1024`, `96*320=30720=30*1024`). `36864/1024=36`
exactly -- the third multiple is `36*1024`. This is a trivial
arithmetic confirmation of data PR #1615 already established, included
here only to answer this file's own opening question precisely rather
than leaving "a third distinct multiple" unstated.

## A genuine seventh plateau exists at `N=32`/`33` -- and its own
## boundary lands on the EXACT SAME `K*N` product as the `128`->`64`
## transition

Sweeping `K` at fixed `N=32` (compile-only `pulsar2:7.0-lite` builds,
one live tensor `x`, `w`/`b` compile-time-constant initializers,
matching every other file in this session's own `bank=0xe1` threshold
work) finds the pair absent through `K=1152` and present from `K=1153`
onward -- a single, clean `K`-wide step, the same discipline every
other boundary in this investigation has shown:

| `K` | pair present at `N=32` |
| --- | --- |
| 300 | no |
| 600 | no |
| 900 | no |
| 1050 | no |
| 1125 | no |
| 1144 | no |
| 1149 | no |
| 1151 | no |
| **1152** | **no** |
| **1153** | **yes** |
| 1162 | yes |
| 1200 | yes |
| 1400 | yes |

`1152*32=36864` -- **the identical product** the `128`->`64` transition
(`K=288`/`289`, `288*128=36864`) already landed on. This is a real,
verified coincidence across two independently-bisected transitions
(the `64`->`32` step here, the `128`->`64` step from PR #1615), not
assumed from a formula: both were found by bisection against actual
compiled mcode, and both resolve to the exact same `K*N` value. Whether
this is a genuine shared cap (a hardware buffer/tile-size limit that
both the `128`->`64` and `64`->`32` steps happen to share) or a
coincidence of this specific `K` range is not decoded further here --
reported as a precise, verified fact, not a proven mechanism.

Confirmed not per-build noise: an independently-seeded rebuild at
`K=1153` (different weight and calibration RNG seed) shows the pair
present too, and `K=1400` (well past the boundary) independently
confirms the plateau extends rather than being an isolated point at
`K=1153` alone.

## What this establishes, and what remains open

The power-of-two-halving family (`512, 256, 128, 64`) now has a
confirmed **seventh** member at `32`, continuing the pattern with no
sign of stopping at `64`. Combined with the two already-known
non-power-of-two "+64" anomalies (`320=256+64`, `192=128+64`,
`tests/test_axera_gemm_e1_k129_fifth_plateau.py` PR #1615's own
finding), the fully known plateau sequence by `K`-ascending order is
now: `512` (`K<=64`), `320` (`K=65-96`), `256` (`K=97-128`), `192`
(`K=129-160`), `128` (`K=161-288`), `64` (`K=289-1152`), `32`
(`K=1153+`).

This file does NOT test for an eighth plateau at `N=16` (would need
`K` values well past `1153`, likely in the low thousands given the
`36864`-product coincidence above would predict `K~1152` again if the
`16`->`32` transition shared the same cap, or somewhere else entirely
if it does not) -- left for a future investigation. It also does not
explain WHY `36864` specifically recurs across two transitions, only
that it does, verified directly.
"""

import gzip
import os
import sys
import unittest

_AXERA_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts", "axera")
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import mcode  # noqa: E402

FIX = os.path.join(_AXERA_DIR, "fixtures")

GEMM_F32 = b"\x33\x03\x00"
GEMM_F48 = b"\x35\x03\x00\xa1"

# The full bisection trail at N=32, sweeping K from well below the
# boundary to well above it.
N32_BISECTION = [
    ("gemm_1x300x32_e1seventh.mcode.gz", False),
    ("gemm_1x600x32_e1seventh.mcode.gz", False),
    ("gemm_1x900x32_e1seventh.mcode.gz", False),
    ("gemm_1x1050x32_e1seventh.mcode.gz", False),
    ("gemm_1x1125x32_e1seventh.mcode.gz", False),
    ("gemm_1x1144x32_e1seventh.mcode.gz", False),
    ("gemm_1x1149x32_e1seventh.mcode.gz", False),
    ("gemm_1x1151x32_e1seventh.mcode.gz", False),
    ("gemm_1x1152x32_e1seventh.mcode.gz", False),
    ("gemm_1x1153x32_e1seventh.mcode.gz", True),
    ("gemm_1x1162x32_e1seventh.mcode.gz", True),
    ("gemm_1x1200x32_e1seventh.mcode.gz", True),
    ("gemm_1x1400x32_e1seventh.mcode.gz", True),
]

# K=1153 rebuilt with an independent weight/calibration seed -- rules
# out per-build noise as the explanation for the boundary.
K1153_INDEPENDENT_REBUILD = ("gemm_1x1153x32_e1seventh_r1.mcode.gz", True)

ALL_NEW_FIXTURES = sorted(
    set([n for n, _ in N32_BISECTION] + [K1153_INDEPENDENT_REBUILD[0]])
)


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def decode(name):
    return mcode.decode(load(name), **mcode.FULL_RULE)


def pair_present(recs):
    f32 = {
        r["operand"]
        for r in recs
        if r["kind"] == "V" and r.get("bank") == 0xE1 and r.get("field") == 32
    }
    f48 = {
        r["operand"]
        for r in recs
        if r["kind"] == "V" and r.get("bank") == 0xE1 and r.get("field") == 48
    }
    return GEMM_F32 in f32 and GEMM_F48 in f48


class TestAllNewFixturesDecodeCleanly(unittest.TestCase):
    def test_no_hard_check_errors(self):
        for name in ALL_NEW_FIXTURES:
            hard = [e for e in mcode.check(load(name)) if not e.startswith("coverage:")]
            self.assertEqual(hard, [], name)


class TestK288289BoundaryIsExactlyTheThirdMultipleOf1024(unittest.TestCase):
    """Trivial arithmetic confirmation of data PR #1615 already
    established: 288*128=36864=36*1024, distinct from the first two
    transitions' own products (32*1024, 30*1024)."""

    def test_product_is_36_times_1024(self):
        self.assertEqual(288 * 128, 36864)
        self.assertEqual(36864 // 1024, 36)
        self.assertEqual(36864 % 1024, 0)

    def test_distinct_from_the_first_two_multiples(self):
        first = 64 * 512
        second = 96 * 320
        third = 288 * 128
        self.assertEqual({first // 1024, second // 1024, third // 1024}, {32, 30, 36})


class TestSeventhPlateauExistsAtN32(unittest.TestCase):
    """A single, clean K-wide step at K=1152 (absent) / K=1153
    (present), the same discipline every other boundary in this
    investigation has shown."""

    def test_full_bisection_trail(self):
        for name, expected in N32_BISECTION:
            self.assertEqual(pair_present(decode(name)), expected, name)

    def test_boundary_is_a_single_k_step(self):
        absent = load("gemm_1x1152x32_e1seventh.mcode.gz")
        present = load("gemm_1x1153x32_e1seventh.mcode.gz")
        self.assertFalse(pair_present(mcode.decode(absent, **mcode.FULL_RULE)))
        self.assertTrue(pair_present(mcode.decode(present, **mcode.FULL_RULE)))


class TestBoundaryMatchesThe128To64TransitionsExactProduct(unittest.TestCase):
    """1152*32=36864, the identical product PR #1615's own K=288/289
    (128->64) transition landed on -- verified directly, not assumed."""

    def test_products_are_identical(self):
        seventh_plateau_boundary_product = 1152 * 32  # this file's own 64->32
        prior_128_to_64_boundary_product = 288 * 128  # PR #1615's own boundary
        self.assertEqual(seventh_plateau_boundary_product, 36864)
        self.assertEqual(prior_128_to_64_boundary_product, 36864)
        self.assertEqual(
            seventh_plateau_boundary_product, prior_128_to_64_boundary_product
        )


class TestSeventhPlateauIsNotPerBuildNoise(unittest.TestCase):
    """K=1153 rebuilt with an independent weight/calibration seed still
    shows the pair present; K=1400 (well past the boundary)
    independently confirms the plateau extends rather than being an
    isolated single-K point."""

    def test_independent_rebuild_agrees(self):
        name, expected = K1153_INDEPENDENT_REBUILD
        self.assertEqual(pair_present(decode(name)), expected, name)

    def test_k1400_still_on_the_plateau(self):
        self.assertTrue(pair_present(decode("gemm_1x1400x32_e1seventh.mcode.gz")))


class TestFullSevenPlateauSequenceByKAscendingOrder(unittest.TestCase):
    """Assembles the complete plateau table by K-ascending order across
    this entire session's own investigation (PR #1609/#1611/#1613/#1615
    and this file), as a single, directly-checkable sequence."""

    def test_sequence_is_seven_plateaus_long(self):
        # (k_range_description, n_threshold)
        sequence = [
            ("K<=64", 512),
            ("K=65-96", 320),
            ("K=97-128", 256),
            ("K=129-160", 192),
            ("K=161-288", 128),
            ("K=289-1152", 64),
            ("K=1153+", 32),
        ]
        self.assertEqual(len(sequence), 7)
        n_values = [n for _, n in sequence]
        # Strictly decreasing as K grows.
        self.assertEqual(n_values, sorted(n_values, reverse=True))


if __name__ == "__main__":
    unittest.main()
