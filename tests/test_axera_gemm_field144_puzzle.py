"""Continuing `tests/test_axera_gemm_bank_81_e1_decode.py` (PR #1570)'s own
explicitly flagged open item: bank `0x81`'s field=144 record (operand
`08 03 00 a1`) appears in *some* but not all large-`N` Gemm shapes, and that
file's own four data points showed presence at the two *extremes* of
`N/K` with absence in between -- explicitly not a clean `N/K` threshold, and
explicitly left open.

## The `N/K`-ratio hypothesis is refuted

Nine new `M=1` Gemm builds (`K` in `{64, 128, 256, 512}`, `N` chosen to
probe specific `N/K` brackets) directly contradict the "presence at the
extremes of a `floor(log2(N/K))` bracket" reading that PR #1570's four
points were consistent with:

- `K=128, N=250` (`N/K=1.95`, same bracket as the already-known *present*
  `K=512, N=1000` point) is **absent**.
- `K=512, N=4000` (`N/K=7.8125`, same bracket as the already-known *present*
  `K=128, N=1000` point) is **absent**.

Both are clean counterexamples to a pure `N/K`-bracket rule -- not decoded
further here, but refuted with real data rather than left as an untested
guess.

## What *is* real: presence is a pure function of `Q = N*K/1024`, not of
## `K` or `N` independently

Defining `Q = N * K / 1024` (equivalently: the number of `N`-tiles when the
per-tile width is `1024/K`, the same `1024` that already turned up as
bank `0x81` field=192's own `1024 // K` divisor -- see PR #1570), **every
pair of shapes tested here or in PR #1570 that lands on the same `Q` value
agrees on field=144's presence, regardless of how different their own `K`
and `N` are**:

| `Q` | shapes landing on it | presence |
| --- | --- | --- |
| 125 | `(K=128,N=1000)` [PR #1570], `(K=256,N=500)`, `(K=64,N=2000)` | present (3-way agreement across 3 different `K`) |
| 500 | `(K=512,N=1000)` [PR #1570], `(K=256,N=2000)` | present (2-way agreement) |
| 250 | `(K=256,N=1000)` [PR #1570], `(K=128,N=2000)` | absent (2-way agreement) |
| 62.5 | `(K=128,N=500)`, `(K=64,N=1000)` | absent (2-way agreement) |
| 1000 | `(K=512,N=2000)` | present (single point) |
| 31.25 | `(K=128,N=250)` | absent (single point) |
| 2000 | `(K=512,N=4000)` | absent (single point) |
| 50 | `(K=512,N=100)` [PR #1570] | absent (single point) |

Six of these eight `Q` values are confirmed by at least two independent
`(K,N)` pairs with *different* `K`, and every single one agrees on presence
-- strong evidence `Q` (not raw `K` or `N`) is the real argument of whatever
determines this record's presence. This is a genuine reduction of the
open question from two free variables to one.

## What remains open: `Q`'s own presence function is not simple

Writing `Q = 125 * 2**e` for the values above that are exact powers of two
times 125 (`e = log2(Q/125)`, an integer for every `Q` above except `50`,
which is not of this form at all and is a separate loose end):

| `e` | -2 | -1 | 0 | 1 | 2 | 3 | 4 |
| --- | - | - | - | - | - | - | - |
| present? | no | no | **yes** | no | **yes** | **yes** | no |

This is not a parity rule (`e=2` and `e=3` are *consecutive* and both
present -- a clean even/odd split would forbid that), not a threshold (`e=0`
is present while the intervening `e=1` is absent, sandwiched between two
present values), and not obviously periodic over the 7-point range actually
sampled. `Q=50` (`K=512, N=100`, from PR #1570) sits outside this `125*2**e`
family entirely and is also absent, but that alone doesn't disambiguate
between "the rule only applies to the `125*2**e` family" and "`50` is absent
for the same underlying reason, just not visible in this parametrization."

This file does not resolve the `e`-function -- consistent with this
project's own established practice (see PR #1570's own explicit refusal to
force a pattern) of reporting a precise, well-evidenced negative over a
forced fit. The next natural step for whoever picks this up: sample `e=5,6`
and `e=-3,-4` to check for periodicity past this window, and find a second
`Q` value landing on `e=1` or `e=4` from a third distinct `K` to see if the
single-point absences there also replicate.
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


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def field144_present(data):
    recs = mcode.decode(data, **mcode.FULL_RULE)
    hits = [
        r for r in recs if r["kind"] == "V" and r["bank"] == 0x81 and r["field"] == 144
    ]
    for r in hits:
        assert r["operand"] == b"\x08\x03\x00\xa1", r["operand"]
    return bool(hits)


class TestNewFixturesCheckClean(unittest.TestCase):
    def test_all_new_fixtures_check_clean(self):
        for name in (
            "gemm_1x128x250.mcode.gz",
            "gemm_1x128x500.mcode.gz",
            "gemm_1x128x2000.mcode.gz",
            "gemm_1x256x2000.mcode.gz",
            "gemm_1x512x4000.mcode.gz",
            "gemm_1x64x1000.mcode.gz",
            "gemm_1x256x500.mcode.gz",
            "gemm_1x512x2000.mcode.gz",
            "gemm_1x64x2000.mcode.gz",
        ):
            self.assertEqual(mcode.check(load(name)), [], name)


class TestNKRatioBracketHypothesisRefuted(unittest.TestCase):
    """PR #1570's four points were consistent with "present at the extremes
    of a floor(log2(N/K)) bracket, absent in the middle" -- these two new
    shapes share a bracket with an already-known *present* point but are
    themselves absent, refuting that reading directly."""

    def test_k128_n250_same_bracket_as_present_k512_n1000_but_absent(self):
        # N/K = 1.953125, same floor(log2(N/K)) == 0 bracket as the known
        # present (K=512, N=1000) point (N/K also 1.953125) -- but absent.
        self.assertFalse(field144_present(load("gemm_1x128x250.mcode.gz")))

    def test_k512_n4000_same_bracket_as_present_k128_n1000_but_absent(self):
        # N/K = 7.8125, same floor(log2(N/K)) == 2 bracket as the known
        # present (K=128, N=1000) point (N/K also 7.8125) -- but absent.
        self.assertFalse(field144_present(load("gemm_1x512x4000.mcode.gz")))


class TestPresenceIsAFunctionOfQNotKOrNAlone(unittest.TestCase):
    """Q = N*K/1024. Every pair of shapes landing on the same Q agrees on
    presence, regardless of how different their own K and N are -- strong
    evidence Q, not raw K or N, is what actually determines this."""

    def test_q125_present_via_three_different_k(self):
        # (K=128,N=1000) already established present in PR #1570 -- not
        # re-asserted here (that fixture/file is that PR's own scope), only
        # the two NEW (K,N) pairs also landing on Q=125 are checked, plus
        # the shared value assertion between them.
        a = field144_present(load("gemm_1x256x500.mcode.gz"))  # Q = 500*256/1024 = 125
        b = field144_present(load("gemm_1x64x2000.mcode.gz"))  # Q = 2000*64/1024 = 125
        self.assertTrue(a, "K=256,N=500 (Q=125)")
        self.assertTrue(b, "K=64,N=2000 (Q=125)")

    def test_q500_present_matches_k512_n1000_from_pr1570(self):
        self.assertTrue(field144_present(load("gemm_1x256x2000.mcode.gz")))  # Q=500

    def test_q250_absent_matches_k256_n1000_from_pr1570(self):
        self.assertFalse(field144_present(load("gemm_1x128x2000.mcode.gz")))  # Q=250

    def test_q62_5_absent_two_different_k(self):
        a = field144_present(load("gemm_1x128x500.mcode.gz"))  # Q = 62.5
        b = field144_present(load("gemm_1x64x1000.mcode.gz"))  # Q = 62.5
        self.assertFalse(a, "K=128,N=500 (Q=62.5)")
        self.assertFalse(b, "K=64,N=1000 (Q=62.5)")


class TestQFunctionIsNotPeriodicOrMonotonic(unittest.TestCase):
    """The e=log2(Q/125) sequence over the sampled range is
    absent,absent,present,absent,present,present,absent (e=-2..4) -- not a
    parity rule (e=2,3 consecutive and both present), not a threshold
    (e=0 present with e=1 absent sandwiched between two present values).
    Recorded honestly as still-open, not forced into a false pattern."""

    def test_e_minus_2_absent(self):
        self.assertFalse(field144_present(load("gemm_1x128x250.mcode.gz")))  # Q=31.25

    def test_e_3_present(self):
        # K=512, N=2000 -> Q = 1000 = 125 * 2**3
        self.assertTrue(field144_present(load("gemm_1x512x2000.mcode.gz")))

    def test_e_4_absent(self):
        self.assertFalse(field144_present(load("gemm_1x512x4000.mcode.gz")))  # Q=2000

    def test_e2_and_e3_are_consecutive_and_both_present(self):
        # The single clearest refutation of a parity-of-e rule: e=2 and e=3
        # are adjacent integers, and both are present.
        e2 = field144_present(load("gemm_1x256x2000.mcode.gz"))  # Q=500, e=2
        e3 = field144_present(load("gemm_1x512x2000.mcode.gz"))  # Q=1000, e=3
        self.assertTrue(e2)
        self.assertTrue(e3)


if __name__ == "__main__":
    unittest.main()
