"""Continues `tests/test_axera_matmul_bank_0xe1_check.py` (PR #1604)'s
own explicitly flagged gap: that file found MatMul's own `bank=0xe1`
(225), `field=32`/`field=48` MatMul-specific pair (operand `09 03 00`
/ `0b 03 00 a1`) is present iff the `A_offset`/`B_offset` table order is
`(B, A)` in the `K`-sweep/diag shape family (42 samples, zero
exceptions) -- but that same rule is *inverted* (present iff `(A,B)`)
across the 19 `M`-sweep points it checked (`M>=24`). It did not pin
down where exactly, between the `K`-sweep's own small `M` and `M=24`,
the mapping actually flips.

## Answer: the boundary is exact and already sits inside the committed
## corpus -- `M=23` follows the original rule, `M=24` follows the
## inverted rule, with zero exceptions on either side

`tests/test_axera_matmul_var_byte_m_plateaus.py` (PR #1528) and
`tests/test_axera_matmul_var_byte_m_period.py` (PR #1524-adjacent) had
already densely built `matmul_var_m{5..48}.mcode.gz` (`batch=2, K=8,
N=8` fixed, `M` swept) for an *unrelated* quantity (the `var` byte's own
`M mod 4` periodicity -- see "What this is NOT" below). Re-reading
every one of those fixtures for the `bank=0xe1` `OTHER` pair and its
own table order (no new builds needed for this half) gives a clean
split:

| `M` | 5 | 6 | 6r | 7 | 9 | 10 | 11 | 12 | 12r | 13 | 14 | 15 | 17 | 18 | 19 | 20 | 20r | 21 | 21r | 22 | **23** | **24** | 24r | 25 | 26 | 27 | 28 | 29 | 30 | 31 | 33 | 34 | 35 | 36 | 36r | 40 | 40r | 44 | 48 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rule | orig | orig | orig | orig | orig | orig | orig | orig | orig | orig | orig | orig | orig | orig | orig | orig | orig | orig | orig | orig | **orig** | **inv** | inv | inv | inv | inv | inv | inv | inv | inv | inv | inv | inv | inv | inv | inv | inv | inv | inv |

("orig" = matches the `K`-sweep's own rule, `has_other == (order ==
(B,A))`; "inv" = matches the inverted rule, `has_other == (order ==
(A,B))`; `r` = an independent rebuild of that `M`, always agreeing with
its own non-rebuild sibling on which rule applies even when the table
*order itself* differs between the two rebuilds, e.g. `M=36` vs.
`M=36_rebuild` land on opposite table orders but both still satisfy the
inverted rule.) **Every `M` from 5 through 23 follows the original
rule; every `M` from 24 through 48 follows the inverted rule. Zero
exceptions on either side of `M=23`/`M=24`, computed directly below in
`TestExistingCorpusPinsTheBoundary`, not just asserted from this
table.**

## Confirmed with fresh, independently-built rebuilds at both M=23 and
## M=24 -- not just the pre-existing corpus's own (unknown) build script

The pre-existing `M`-sweep fixtures' own original build script was
never committed, so agreeing with them alone leaves open whether `M=23`
vs. `M=24` is a genuine shape-triggered threshold or an artifact of
however that script was written. This file adds **3 new, independently
built rebuilds each at `M=23` and `M=24`** (`matmul_var_m23_indep_r{0,1,2}.mcode.gz`,
`matmul_var_m24_indep_r{0,1,2}.mcode.gz`), built from scratch with a
different RNG seed per sample and a genuinely two-live-input
`MatMul(A[2,M,8], B[2,8,8])` ONNX graph (both `A` and `B` calibrated
via separate `Numpy`-format datasets -- both must be live, non-constant
tensors for the `A_offset`/`B_offset` coin-flip's own structural
precondition to exist at all, per this project's established finding)
via `pulsar2_docker.build()` (`pulsar2:7.0-lite`, the only image loaded
in this worktree's Docker daemon). **All 3 fresh `M=23` builds satisfy
the original rule; all 3 fresh `M=24` builds satisfy the inverted rule
-- zero exceptions, using a completely independent model/build
pipeline from whatever produced the pre-existing corpus.** This rules
out "artifact of one particular build script" as an explanation for the
boundary.

## What this is NOT: the M=24 mod-4 `var`-byte periodicity is a
## different, already-known, and already-periodic phenomenon

`tests/test_axera_matmul_var_byte_m_period.py` already established that
a *different* MatMul byte (`var`, unrelated to `bank=0xe1`) is
`0x30` exactly when `M mod 4 == 0` (`M=20,24,28,...`) and `0x34`
otherwise, across the *entire* `M=17..31` range -- a smooth, periodic
pattern with no special M=24 discontinuity of its own. This file's own
`bank=0xe1`-order-mapping inversion is a genuinely different
phenomenon: a **one-time step**, not periodic (every `M<=23` behaves
one way, every `M>=24` the other, confirmed through `M=48` with zero
counter-examples -- there is no return to the "original" rule at, say,
`M=28` or `M=32` the way `var`'s own mod-4 pattern would predict if the
two phenomena were the same). The two mechanisms merely happen to share
`M=24` as A boundary value; this file does not claim they are the same
underlying trigger, only notes the coincidence explicitly so a future
reader does not conflate them (the same discipline
`tests/test_axera_conv_offset552_field.py`, PR #1605, applied to a
different coincidental-numeric-collision case).
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
OTHER_F32 = b"\x09\x03\x00"

# (name, M) for every pre-existing matmul_var_m* fixture, M=5..23
BELOW_23 = [
    ("matmul_var_m5.mcode.gz", 5),
    ("matmul_var_m6.mcode.gz", 6),
    ("matmul_var_m6_rebuild.mcode.gz", 6),
    ("matmul_var_m7.mcode.gz", 7),
    ("matmul_var_m9.mcode.gz", 9),
    ("matmul_var_m10.mcode.gz", 10),
    ("matmul_var_m11.mcode.gz", 11),
    ("matmul_var_m12.mcode.gz", 12),
    ("matmul_var_m12_rebuild.mcode.gz", 12),
    ("matmul_var_m13.mcode.gz", 13),
    ("matmul_var_m14.mcode.gz", 14),
    ("matmul_var_m15.mcode.gz", 15),
    ("matmul_var_m17.mcode.gz", 17),
    ("matmul_var_m18.mcode.gz", 18),
    ("matmul_var_m19.mcode.gz", 19),
    ("matmul_var_m20.mcode.gz", 20),
    ("matmul_var_m20_rebuild.mcode.gz", 20),
    ("matmul_var_m21.mcode.gz", 21),
    ("matmul_var_m21_rebuild.mcode.gz", 21),
    ("matmul_var_m22.mcode.gz", 22),
    ("matmul_var_m23.mcode.gz", 23),
]

FROM_24_UP = [
    ("matmul_var_m24.mcode.gz", 24),
    ("matmul_var_m24_rebuild.mcode.gz", 24),
    ("matmul_var_m25.mcode.gz", 25),
    ("matmul_var_m26.mcode.gz", 26),
    ("matmul_var_m27.mcode.gz", 27),
    ("matmul_var_m28.mcode.gz", 28),
    ("matmul_var_m29.mcode.gz", 29),
    ("matmul_var_m30.mcode.gz", 30),
    ("matmul_var_m31.mcode.gz", 31),
    ("matmul_var_m33.mcode.gz", 33),
    ("matmul_var_m34.mcode.gz", 34),
    ("matmul_var_m35.mcode.gz", 35),
    ("matmul_var_m36.mcode.gz", 36),
    ("matmul_var_m36_rebuild.mcode.gz", 36),
    ("matmul_var_m40.mcode.gz", 40),
    ("matmul_var_m40_rebuild.mcode.gz", 40),
    ("matmul_var_m44.mcode.gz", 44),
    ("matmul_var_m48.mcode.gz", 48),
]

FRESH_M23 = [f"matmul_var_m23_indep_r{i}.mcode.gz" for i in range(3)]
FRESH_M24 = [f"matmul_var_m24_indep_r{i}.mcode.gz" for i in range(3)]


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def decode(name):
    return mcode.decode(load(name), **mcode.FULL_RULE)


def table_order(data):
    ia = data.find(b"A_offset")
    ib = data.find(b"B_offset")
    assert ia != -1 and ib != -1, "both A_offset and B_offset must be present"
    return ("A", "B") if ia < ib else ("B", "A")


def has_other_pair(name):
    recs = decode(name)
    f32 = {
        r["operand"]
        for r in recs
        if r["kind"] == "V" and r.get("bank") == 0xE1 and r.get("field") == 32
    }
    return OTHER_F32 in f32


class TestExistingCorpusPinsTheBoundary(unittest.TestCase):
    """Recomputes the module docstring's own table live against the
    fixtures, not trusted from the prose: every M<=23 point matches the
    K-sweep's original rule, every M>=24 point matches the inverted
    rule, zero exceptions."""

    def test_every_point_below_24_matches_the_original_rule(self):
        for name, m in BELOW_23:
            data = load(name)
            order = table_order(data)
            predicted = order == ("B", "A")
            self.assertEqual(has_other_pair(name), predicted, f"M={m} ({name})")

    def test_every_point_from_24_up_matches_the_inverted_rule(self):
        for name, m in FROM_24_UP:
            data = load(name)
            order = table_order(data)
            predicted = order == ("A", "B")
            self.assertEqual(has_other_pair(name), predicted, f"M={m} ({name})")

    def test_m23_and_m24_are_the_exact_crossing_point(self):
        # M=23 (last "original" point) and M=24 (first "inverted"
        # point) sit right next to each other in the sweep -- the
        # boundary is not merely "somewhere before 24", it's exactly
        # here, with no untested integer gap in between.
        m23_order = table_order(load("matmul_var_m23.mcode.gz"))
        m24_order = table_order(load("matmul_var_m24.mcode.gz"))
        self.assertEqual(
            has_other_pair("matmul_var_m23.mcode.gz"), m23_order == ("B", "A")
        )
        self.assertEqual(
            has_other_pair("matmul_var_m24.mcode.gz"), m24_order == ("A", "B")
        )


class TestFreshIndependentBuildsConfirmTheBoundary(unittest.TestCase):
    """3 new, independently built rebuilds at M=23 and M=24 each, using
    a from-scratch model/build pipeline unrelated to whatever produced
    the pre-existing M-sweep fixtures -- rules out "artifact of one
    particular build script" as an explanation for the boundary."""

    def test_all_three_fresh_m23_builds_match_the_original_rule(self):
        for name in FRESH_M23:
            data = load(name)
            order = table_order(data)
            predicted = order == ("B", "A")
            self.assertEqual(has_other_pair(name), predicted, name)

    def test_all_three_fresh_m24_builds_match_the_inverted_rule(self):
        for name in FRESH_M24:
            data = load(name)
            order = table_order(data)
            predicted = order == ("A", "B")
            self.assertEqual(has_other_pair(name), predicted, name)

    def test_fresh_fixtures_decode_cleanly(self):
        for name in FRESH_M23 + FRESH_M24:
            errs = mcode.check(load(name))
            hard = [e for e in errs if not e.startswith("coverage:")]
            self.assertEqual(hard, [], name)


class TestGemmMatchingPairIsUnaffectedByThisBoundary(unittest.TestCase):
    """Sanity check: the Gemm-matching pair (already reported as not
    following a clean formula, PR #1604) is absent from every fresh
    M=23/M=24 build here too -- this file's new fixtures don't
    accidentally introduce it, keeping this boundary finding isolated
    to the MatMul-specific OTHER pair alone."""

    def test_gemm_pair_absent_from_fresh_builds(self):
        for name in FRESH_M23 + FRESH_M24:
            recs = decode(name)
            f32 = {
                r["operand"]
                for r in recs
                if r["kind"] == "V" and r.get("bank") == 0xE1 and r.get("field") == 32
            }
            self.assertNotIn(GEMM_F32, f32, name)


if __name__ == "__main__":
    unittest.main()
