"""Continuing `tests/test_axera_gemm_field144_puzzle.py` (PR #1573)'s own
explicitly flagged gap: that file reduced bank `0x81` field=144's presence
question from two free variables (`K`, `N`) to one (`Q = N*K/1024`), but
left `Q`'s own presence rule unsolved -- the sampled `e = log2(Q/125)`
sequence (`-2..4`) was neither a parity rule, a threshold, nor obviously
periodic, and explicitly asked for closer sampling plus an `M` control.

## `M` does not matter -- `Q` alone (not `K`, `N`, or `M` individually)
## still determines presence

Two new builds hold `Q` fixed at already-known values while changing `M`
from `1` (every prior probe) to `4`:

- `Q=125` (`M=4, K=128, N=1000`): **present**, matching the known
  `M=1, Q=125` result exactly.
- `Q=250` (`M=4, K=256, N=1000`): **absent**, matching the known
  `M=1, Q=250` result exactly.

This directly rules out the concern that PR #1570/#1573's own `M=1`-only
sampling coincidentally produced the `Q` pattern -- `Q` genuinely
generalizes across `M`, strengthening (not just repeating) that finding.

## Twelve new `M=1` builds at finer `Q` resolution -- no clean rule found

| `Q` | `K`, `N` | presence |
| --- | --- | --- |
| 15.625 | K=128, N=125 | absent (`*`) |
| 31.25 | [PR #1573] | absent |
| 50 | [PR #1570] | absent |
| 62.5 | [PR #1573] | absent |
| 90 | K=256, N=360 | absent |
| 105 | K=256, N=420 | **present** |
| 125 | [PR #1570/#1573] | present |
| 180 | K=256, N=720 | **present** |
| 210 | K=256, N=840 | absent |
| 250 | [PR #1570/#1573] | absent |
| 300 | K=256, N=1200 | absent |
| 400 | K=256, N=1600 | absent |
| 500 | [PR #1570/#1573] | present |
| 700 | K=256, N=2800 | **present** |
| 850 | K=256, N=3400 | absent |
| 1000 | [PR #1570/#1573] | present |
| 1300 | K=256, N=5200 | absent |
| 1700 | K=256, N=6800 | **present** |
| 2000 | [PR #1573] | absent |
| 4000 | K=256, N=16000 | absent |

(`*` `Q=15.625`'s fixture is the one place `mcode.check()` reports
sub-100% byte coverage on this whole corpus -- see "A coverage caveat"
below; the field=144 absence reading itself is unaffected.)

This is a genuine expansion of the sampled range (20 points spanning
`Q=15.625` to `4000`, vs. #1573's 8) and it refutes every candidate rule
tried:

- **Not a `floor(log2 Q)` (power-of-two band) rule**: the band
  `Q in [128, 256)` (`floor(log2 Q) = 7`) contains both a present point
  (`Q=180`) and two absent points (`Q=210`, `Q=250`) -- presence is not
  constant within a band.
- **Not a rule on `Q`'s fractional position within its band**
  (`frac = Q / 2**floor(log2 Q)`, i.e. `Q` normalized to `[1, 2)`):
  the `125 * 2**e` family (`Q=31.25, 62.5, 125, 250, 500, 1000, 2000,
  4000`) all share the *same* `frac = 1.953125` by construction, yet
  their presence alternates (absent, absent, present, absent, present,
  present, absent, absent) purely as a function of the band -- so `frac`
  alone can't be the rule either, since it's constant across all eight
  while presence isn't.
- **Not a joint (band, frac) rule with a simple closed form**: no
  monotonic or periodic pattern in either dimension fits the 20-point
  table above under manual inspection. Adjacent present points do not
  share `floor(log2 Q)` (`105` and `125` do: both band 6; but `180`
  (band 7) is present while its band-mates `210`/`250` are absent) and
  the gap between present bands is not constant (present at bands
  6, 7, 8, 9, 10 -- i.e. nearly every band from 6 to 10 has *at least
  one* present point, which weakens the original "banded" framing
  further rather than clarifying it).

## What remains genuinely open

Twelve new data points plus the eight already on record (20 total) are
not enough to pin down a rule, and no simple function of `K`, `N`, `M`,
or `Q` (raw, banded, or fractional) checked here explains the full
pattern. This file does not force a fit -- consistent with this
project's established practice (`tests/test_axera_gemm_field144_puzzle.py`'s
own explicit refusal, and `tests/test_axera_gemm_bank_81_e1_decode.py`'s
before it) of reporting a precise, well-evidenced negative over a forced
pattern. Two observations that might help whoever picks this up next:

- Presence looks like it could plausibly track a genuinely different
  quantity that happens to correlate with `Q` at the coarse granularity
  #1573 originally sampled, but does not reduce to `Q` cleanly at this
  file's finer resolution -- something in the compiler's own internal
  tiling/scheduling decision that isn't a pure arithmetic function of the
  shape at all (e.g. a search/heuristic outcome), rather than a decodable
  closed-form field.
- All fine-resolution probes here fixed `K=256`; deliberately varying
  `K` while holding `Q` near one of these newly found present/absent
  boundaries (e.g. re-probing `Q=105`, `Q=180`, `Q=700`, or `Q=1700`
  with a different `K`) would test whether *this* denser data still
  reduces to `Q` alone the way the original 8-point, coarser sample did,
  or whether the apparent `Q`-only dependence itself starts to break
  down under closer inspection -- not attempted here.

## A coverage caveat on the smallest new fixture

`gemm_1x128x125.mcode.gz` (`Q=15.625`) is the only fixture in this file's
new set -- and, as far as this file checked, in the whole corpus -- where
`mcode.check()` reports sub-100% non-zero-byte coverage (94.0%, four
small unexplained byte ranges). This is a coverage gap in the grammar's
explanation of *other* parts of this small fixture, not a defect in the
field=144 read itself (that read is a simple, unambiguous
"is this exact (bank, field) record present" query, independent of
whatever the unexplained bytes elsewhere encode) -- recorded here for
honesty rather than silently ignored.
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


NEW_FIXTURES = (
    "gemm_1x256x360.mcode.gz",
    "gemm_1x256x420.mcode.gz",
    "gemm_1x256x720.mcode.gz",
    "gemm_1x256x840.mcode.gz",
    "gemm_1x256x1200.mcode.gz",
    "gemm_1x256x1600.mcode.gz",
    "gemm_1x256x2800.mcode.gz",
    "gemm_1x256x3400.mcode.gz",
    "gemm_1x256x5200.mcode.gz",
    "gemm_1x256x6800.mcode.gz",
    "gemm_1x256x16000.mcode.gz",
    "gemm_1x128x125.mcode.gz",
    "gemm_4x128x1000.mcode.gz",
    "gemm_4x256x1000.mcode.gz",
)


class TestNewFixturesDecodeCleanly(unittest.TestCase):
    def test_all_new_fixtures_have_no_hard_check_errors(self):
        # gemm_1x128x125 (Q=15.625) is the one exception with a soft
        # "coverage" warning (see module docstring) -- everything else
        # is fully clean. A "coverage" message is informational (the
        # grammar doesn't yet explain every byte); anything else would
        # be a real structural problem.
        for name in NEW_FIXTURES:
            errs = mcode.check(load(name))
            hard_errs = [e for e in errs if not e.startswith("coverage:")]
            self.assertEqual(hard_errs, [], name)

    def test_q15_625_has_the_one_known_coverage_gap(self):
        errs = mcode.check(load("gemm_1x128x125.mcode.gz"))
        self.assertEqual(len(errs), 1, errs)
        self.assertTrue(errs[0].startswith("coverage:"), errs[0])


class TestMDoesNotMatter(unittest.TestCase):
    """Q=125 and Q=250 already had known presence at M=1; these two new
    M=4 builds land on the exact same Q values and agree -- M is not a
    hidden third variable, Q (or whatever it's a proxy for) really is
    M-independent too."""

    def test_q125_present_at_m4_matches_m1(self):
        self.assertTrue(field144_present(load("gemm_4x128x1000.mcode.gz")))

    def test_q250_absent_at_m4_matches_m1(self):
        self.assertFalse(field144_present(load("gemm_4x256x1000.mcode.gz")))


class TestFinerQResolutionRefutesBandAndFracRules(unittest.TestCase):
    """Twelve new K=256 (plus one K=128) points at finer Q resolution than
    PR #1573's original 8-point sample. See module docstring for the full
    table and the band/frac hypotheses this refutes."""

    def test_q90_absent(self):
        self.assertFalse(field144_present(load("gemm_1x256x360.mcode.gz")))

    def test_q105_present(self):
        self.assertTrue(field144_present(load("gemm_1x256x420.mcode.gz")))

    def test_q180_present(self):
        self.assertTrue(field144_present(load("gemm_1x256x720.mcode.gz")))

    def test_q210_absent(self):
        self.assertFalse(field144_present(load("gemm_1x256x840.mcode.gz")))

    def test_q300_absent(self):
        self.assertFalse(field144_present(load("gemm_1x256x1200.mcode.gz")))

    def test_q400_absent(self):
        self.assertFalse(field144_present(load("gemm_1x256x1600.mcode.gz")))

    def test_q700_present(self):
        self.assertTrue(field144_present(load("gemm_1x256x2800.mcode.gz")))

    def test_q850_absent(self):
        self.assertFalse(field144_present(load("gemm_1x256x3400.mcode.gz")))

    def test_q1300_absent(self):
        self.assertFalse(field144_present(load("gemm_1x256x5200.mcode.gz")))

    def test_q1700_present(self):
        self.assertTrue(field144_present(load("gemm_1x256x6800.mcode.gz")))

    def test_q4000_absent(self):
        self.assertFalse(field144_present(load("gemm_1x256x16000.mcode.gz")))

    def test_q15_625_absent(self):
        self.assertFalse(field144_present(load("gemm_1x128x125.mcode.gz")))


class TestBandRuleRefuted(unittest.TestCase):
    """floor(log2(Q))==7 (Q in [128,256)) contains both a present point
    (Q=180) and absent points (Q=210, Q=250, all K=256) -- a single band
    is not uniformly present or absent."""

    def test_q180_and_q210_share_a_band_but_disagree(self):
        import math

        self.assertEqual(math.floor(math.log2(180)), math.floor(math.log2(210)))
        self.assertTrue(field144_present(load("gemm_1x256x720.mcode.gz")))
        self.assertFalse(field144_present(load("gemm_1x256x840.mcode.gz")))


if __name__ == "__main__":
    unittest.main()
