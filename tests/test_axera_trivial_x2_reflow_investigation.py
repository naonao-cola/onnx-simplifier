"""Investigates the one precise, unexplained asymmetry `tests/test_axera_
x2_trivial_zeropoint_replacement_check.py` (PR #1680) left open: when a
live input's own calibrated zero point is trivially `0`, `Mul`'s own
trivial-`x2` fixtures ALWAYS reflow their whole stream 32 bytes shorter
(confirmed at all 3 tested seed pairs, `2264`->`2232`); `Add` does this
at only ONE of its 3 tested seed pairs (`(100,999)`, `3104`->`3072`);
`Sub`/`Div` never do it at any tested seed pair.

This file localizes WHERE the length change comes from via a byte-level
common-prefix/common-suffix diff (not a decode-record diff -- any single
early byte shift cascades into every later record's own reported
kind/offset regardless of whether the underlying change was itself
localized, so record-count deltas alone cannot localize anything).

## Finding 1: every length-CHANGING case diverges over MOST of the
## stream, not a small, localized 32-byte deletion -- but not uniformly
## "near total" either; one case is a genuine partial exception

| case | ctrl len | triv len | common prefix | common suffix | divergent bytes | divergent % |
| --- | --- | --- | --- | --- | --- | --- |
| Mul (1,2) | 2264 | 2232 | 32 | 30 | 2202 | 97.3% |
| Mul (7,42) | 2264 | 2232 | 32 | 415 | 1817 | 80.3% |
| Mul (100,999) | 2264 | 2232 | 32 | 30 | 2202 | 97.3% |
| Add (100,999) | 3104 | 3072 | 32 | 30 | 3042 | 98.0% |

An earlier draft of this file assumed every Mul seed pair would show the
SAME tiny 32/30-byte common prefix/suffix `(1,2)` and `(100,999)`
happen to show -- checking `(7,42)` directly (not assumed from the
other two) found a genuinely larger common suffix (415 bytes), and a
correspondingly smaller divergent span (80.3% instead of ~97%). The
real finding is a strong trend ("length-changing implies substantially
more of the stream differs"), not a clean uniform rule -- reported
here with the actual per-seed-pair numbers, not rounded into a false
"always near-total" claim.

The common prefix (32 bytes, every length-changing case) matches the
FlatBuffers root-table/vtable header length the README's own op-count
sweep already documented for a completely different trigger (adding a
`Conv` layer) -- consistent with this project's own already-documented
"Pulsar2's compiler re-serializes/re-addresses essentially the whole
command stream on any topology change... rather than treating mcode as
an append-only log" finding (`scripts/axera/README.md`).

## Finding 2: length-PRESERVING cases show substantially LESS
## divergence than every length-changing case -- a real, measured
## contrast, even though the gap with Mul's own `(7,42)` case is
## narrower than with the other three length-changing cases

| case | ctrl len | triv len | common prefix | common suffix | divergent bytes | divergent % |
| --- | --- | --- | --- | --- | --- | --- |
| Add (1,2) | 3104 | 3104 | 359 | 1169 | 1576 | 50.8% |
| Add (7,42) | 3104 | 3104 | 373 | 1169 | 1562 | 50.3% |

Both length-preserving cases sit right around 50% divergence -- clearly
below every length-changing case's own divergent fraction (80.3%-98.0%),
but the gap to Mul's own `(7,42)` (80.3%) is real, not a clean
threshold the way the gap to the other three length-changing cases
(97.3%-98.0%) is.

## What this establishes, precisely, and what it does not

**Established**: every length-changing trivial-`x2` case diverges over
a MAJORITY (80.3%-98.0%) of the stream, clearly more than either tested
length-preserving case (~50%) -- consistent with this project's own
already-documented non-append-only re-serialization behavior for
topology-changing edits in general, now observed for this specific
trigger too. The 32-byte length question is downstream of a real,
measured "how much of the stream gets touched" spectrum, not a clean
binary "localized patch vs. total re-layout" split.

**NOT established**: WHY the trivial-`x2` edit is length-changing for
Mul at every tested seed but for Add only at one of three (no access to
Pulsar2's own source, and this file does not find a byte-level
correlate that distinguishes Add's two length-preserving seed pairs
from its one length-changing one, or that explains why Mul's own
`(7,42)` diverges less than its other two seed pairs despite an
identical length delta); whether `Sub`/`Div`'s own never-reflowing
behavior at every tested seed pair reflects a structurally different
code path or is itself just a coincidence of which seeds were tested
(not investigated here -- would need more seed pairs per op to
distinguish "never" from "not yet observed").
"""

import gzip
import os
import sys
import unittest

_AXERA_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts", "axera")
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

FIX = os.path.join(_AXERA_DIR, "fixtures")

MUL_SEED_PAIRS = [(1, 2), (7, 42), (100, 999)]
ADD_SEED_PAIRS = [(1, 2), (7, 42), (100, 999)]
ADD_LENGTH_CHANGING_SEED = (100, 999)

# (op, seed1, seed2) -> (common_prefix, common_suffix), measured directly
# against the real committed fixtures -- not assumed uniform across
# seed pairs (an earlier draft's own mistake, caught by actually
# checking each one).
EXPECTED_PREFIX_SUFFIX = {
    ("mul", 1, 2): (32, 30),
    ("mul", 7, 42): (32, 415),
    ("mul", 100, 999): (32, 30),
    ("add", 1, 2): (359, 1169),
    ("add", 7, 42): (373, 1169),
    ("add", 100, 999): (32, 30),
}


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def common_prefix_len(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def common_suffix_len(a, b):
    n = min(len(a), len(b))
    j = 0
    while j < n and a[-1 - j] == b[-1 - j]:
        j += 1
    return j


class TestMulAlwaysReflowsBy32BytesAtEverySeedPair(unittest.TestCase):
    def test_length_delta_is_exactly_32_at_every_seed(self):
        for s1, s2 in MUL_SEED_PAIRS:
            ctrl = load(f"mul_1x16_two_live_seed{s1}_{s2}.mcode.gz")
            triv = load(f"mul_1x16_two_live_seed{s1}_{s2}_trivialx2.mcode.gz")
            self.assertEqual(len(ctrl) - len(triv), 32, (s1, s2))


class TestAddOnlyReflowsAtOneOfThreeSeedPairs(unittest.TestCase):
    def test_length_unchanged_except_at_100_999(self):
        for s1, s2 in ADD_SEED_PAIRS:
            ctrl = load(f"add_1x16_two_live_seed{s1}_{s2}.mcode.gz")
            triv = load(f"add_1x16_two_live_seed{s1}_{s2}_trivialx2.mcode.gz")
            delta = len(ctrl) - len(triv)
            if (s1, s2) == ADD_LENGTH_CHANGING_SEED:
                self.assertEqual(delta, 32, (s1, s2))
            else:
                self.assertEqual(delta, 0, (s1, s2))


class TestCommonPrefixSuffixMatchesRealMeasurement(unittest.TestCase):
    """Directly re-decodes the real committed fixtures and checks the
    exact table above -- per seed pair, not assumed uniform."""

    def test_mul_prefix_suffix_matches_per_seed_pair(self):
        for s1, s2 in MUL_SEED_PAIRS:
            ctrl = load(f"mul_1x16_two_live_seed{s1}_{s2}.mcode.gz")
            triv = load(f"mul_1x16_two_live_seed{s1}_{s2}_trivialx2.mcode.gz")
            prefix = common_prefix_len(ctrl, triv)
            suffix = common_suffix_len(ctrl, triv)
            expected = EXPECTED_PREFIX_SUFFIX[("mul", s1, s2)]
            self.assertEqual((prefix, suffix), expected, (s1, s2))

    def test_add_prefix_suffix_matches_per_seed_pair(self):
        for s1, s2 in ADD_SEED_PAIRS:
            ctrl = load(f"add_1x16_two_live_seed{s1}_{s2}.mcode.gz")
            triv = load(f"add_1x16_two_live_seed{s1}_{s2}_trivialx2.mcode.gz")
            prefix = common_prefix_len(ctrl, triv)
            suffix = common_suffix_len(ctrl, triv)
            expected = EXPECTED_PREFIX_SUFFIX[("add", s1, s2)]
            self.assertEqual((prefix, suffix), expected, (s1, s2))


class TestLengthChangingCasesDivergeMoreThanLengthPreservingOnes(unittest.TestCase):
    """The headline, honestly-scoped finding: length-changing cases
    diverge over a clear majority of the stream (80.3%-98.0%), more
    than either tested length-preserving case (~50%) -- a real trend,
    not a uniform "near total vs. localized" binary split."""

    def divergent_fraction(self, ctrl, triv):
        prefix = common_prefix_len(ctrl, triv)
        suffix = common_suffix_len(ctrl, triv)
        return (len(ctrl) - prefix - suffix) / len(ctrl)

    def test_every_length_changing_case_exceeds_75_percent_divergence(self):
        cases = [("mul", 1, 2), ("mul", 7, 42), ("mul", 100, 999), ("add", 100, 999)]
        for op, s1, s2 in cases:
            ctrl = load(f"{op}_1x16_two_live_seed{s1}_{s2}.mcode.gz")
            triv = load(f"{op}_1x16_two_live_seed{s1}_{s2}_trivialx2.mcode.gz")
            frac = self.divergent_fraction(ctrl, triv)
            self.assertGreater(frac, 0.75, (op, s1, s2, frac))

    def test_every_length_preserving_case_is_below_60_percent_divergence(self):
        cases = [("add", 1, 2), ("add", 7, 42)]
        for op, s1, s2 in cases:
            ctrl = load(f"{op}_1x16_two_live_seed{s1}_{s2}.mcode.gz")
            triv = load(f"{op}_1x16_two_live_seed{s1}_{s2}_trivialx2.mcode.gz")
            frac = self.divergent_fraction(ctrl, triv)
            self.assertLess(frac, 0.60, (op, s1, s2, frac))


if __name__ == "__main__":
    unittest.main()
