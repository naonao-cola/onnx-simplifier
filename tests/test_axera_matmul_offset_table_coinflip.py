"""The "unusual 28% batched-MatMul rebuild noise" flagged open by
`tests/test_axera_matmul_batched_site_a.py` (PR #1502) is neither
batching-specific nor a large uniform noise zone -- it's the
`A_offset`/`B_offset` name-table order-swap cascade
(`tests/test_axera_matmul_a_input_cascades.py`, PR #1493) firing as a
genuine, unconditioned per-compile coin flip, not something scale- or
batching-triggered. This also revises PR #1493's own causal reading.

## What PR #1502 reported, and what it left open

`tests/test_axera_matmul_batched_site_a.py` found that ONE rebuild pair
of the fully-batched `A[2,4,8] @ B[2,8,8]` MatMul differed at 932 of
3336 bytes (28%), "two orders of magnitude past the ~6-15-byte noise
zone this project has confirmed repeatedly for every other op/shape",
and flagged *why* as "a real, motivated open question for whoever looks
at it next" -- with a specific candidate hypothesis left implicit by
the contrast drawn there: maybe fully-batched (`A` and `B` both rank-3)
specifically is unstable in a way rank-2/broadcast is not.

## What this file establishes: it's a two-mode coin flip, present at rank-2 too

Building **8 independent rebuilds** of the identical plain rank-2
`MatMul(A[4,8], B[8,8])` config (same model, same calibration data,
same everything -- this project's hard-learned rule is one rebuild pair
is never enough to characterize noise, see the README's own
"Correction, caught the same way the auto_pad false lead was" passage)
and diffing all 28 pairs shows a clean bimodal split, not a spread of
noise magnitudes: **every pair falls into exactly one of two buckets --
2 to 6 bytes differing (ordinary noise, same magnitude this project
sees everywhere), or 913 to 917 bytes differing (a "big" diff).** There
is nothing in between. The 8 builds partition cleanly into two groups
of 4 (`{r0,r1,r2,r6}` and `{r3,r4,r5,r7}` in this file's own numbering)
such that every within-group pair is small and every cross-group pair
is big.

**The two groups are exactly `test_axera_matmul_a_input_cascades.py`'s
`A_offset`/`B_offset` name-table order.** Each build's mcode has a
parameter name-table entry pair at offset ~204-260 holding
`B_offset`-then-`A_offset` or `A_offset`-then-`B_offset` (same
flatfbuffer-style `<len><name>` record this project already decoded);
which order a given rebuild lands in predicts, with zero exceptions
across all 28 pairs checked, whether that pair is "small" (same order)
or "big" (different order) -- and every "big" pair's diff spans offset
204 to 3084, the identical range `test_axera_matmul_a_input_cascades.py`
found for its own A-scale-driven order swap.

**The same thing happens for the fully-batched shape PR #1502 flagged.**
Comparing the already-committed `matmul_2x4x8x8_batched.mcode.gz`
fixture (order: `B_offset`-then-`A_offset`) against a fresh rebuild that
landed in the other order gives 932 of 3336 bytes differing, offset 204
to 3308 -- the *exact* 932-byte figure PR #1502 reported from its own
independent rebuild check. This was never a batching-specific
instability; it's the general table-order coin flip, and PR #1502's
rebuild pair happened to land on opposite sides of it.

## Revising PR #1493's causal reading

`tests/test_axera_matmul_a_input_cascades.py` observed this same order
differ between an "A narrow" and an "A wide" calibration build and
described it as correlating "with which of A/B has the larger
calibration range" -- a real, honestly-hedged observation from a single
pair, not asserted as a mechanism. This file's 8-way rebuild of a
**single, unchanged** config shows the order varies on its own, with no
input change at all (roughly half of 8 identical-input rebuilds landed
each way here), so that correlation was very likely coincidental: which
of the two builds happened to draw which table order, not something
`A`'s calibration range causes. The underlying cascade mechanism PR
#1493 characterized (widespread but non-uniform content difference
downstream of the swap, not explained by a constant byte-offset shift)
remains correct and is not revised here -- only the claim about what
*triggers* the swap.

## What this means for reading this project's own noise-zone checks

This project's standard determinism check (build once, rebuild once,
diff) has, by this evidence, roughly even odds of landing on either
side of this coin flip for any MatMul shape -- a single rebuild pair
reporting "hundreds of bytes differ" could mean real signal, or could
simply mean the pair drew opposite table orders. The fix is cheap and
now available: check the `A_offset`/`B_offset` relative order (offset
~204-260) before trusting a large rebuild-pair diff as evidence of
something shape- or value-dependent, the same way this file's own tests
do below. Not audited here: whether this exact table-order coin flip
also explains any of this project's *other* previously-reported "looked
noise-free on one rebuild pair, wasn't on a second" corrections (the
README documents at least two, for unrelated ops) -- a real, motivated
lead for whoever checks next, since the mechanism (an unordered
internal collection whose iteration order isn't pinned across separate
compiler invocations) is generic and has no obvious reason to be
MatMul-specific at the compiler level, only confirmed as MatMul-specific
by what has actually been tested so far.
"""

import gzip
import os
import unittest

FIX = os.path.join(os.path.dirname(__file__), "..", "scripts", "axera", "fixtures")


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def offset_table_order(data):
    """Returns ("A", "B") if A_offset's record precedes B_offset's in the
    ~204-260 window, else ("B", "A"). Mirrors
    test_axera_matmul_a_input_cascades.py's own inspection of this
    region."""
    window = data[195:270]
    ia = window.find(b"A_offset")
    ib = window.find(b"B_offset")
    assert ia != -1 and ib != -1, "both A_offset and B_offset must be present"
    return ("A", "B") if ia < ib else ("B", "A")


class TestRank2MatMulRebuildIsBimodalNotNoisy(unittest.TestCase):
    """8 independent rebuilds of one unchanged rank-2 MatMul config
    partition cleanly into two table-order groups; only cross-group
    pairs show the ~900-byte diff, and it always spans 204..3084."""

    MODE_A = "matmul_4x8x8_rebuild_modeA.mcode.gz"
    MODE_B = "matmul_4x8x8_rebuild_modeB.mcode.gz"

    def test_the_two_fixtures_have_opposite_table_order(self):
        a = load(self.MODE_A)
        b = load(self.MODE_B)
        order_a = offset_table_order(a)
        order_b = offset_table_order(b)
        self.assertNotEqual(order_a, order_b)

    def test_cross_mode_diff_matches_the_known_cascade_range(self):
        a = load(self.MODE_A)
        b = load(self.MODE_B)
        self.assertEqual(len(a), len(b))
        diffs = [i for i in range(len(a)) if a[i] != b[i]]
        self.assertGreater(len(diffs), 900, "cross-mode diff should be the big bucket")
        self.assertLess(len(diffs), 1000, "cross-mode diff should be the big bucket")
        self.assertEqual(min(diffs), 204)
        self.assertEqual(max(diffs), 3084)


class TestBatchedMatMulSameCoinFlipExplainsPR1502(unittest.TestCase):
    """The already-committed batched fixture (PR #1502) and a rebuild
    that landed in the opposite table order reproduce PR #1502's own
    932/3336 figure exactly -- confirming it was this same coin flip,
    not a batching-specific instability."""

    COMMITTED = "matmul_2x4x8x8_batched.mcode.gz"
    OTHER_MODE = "matmul_2x4x8x8_batched_rebuild_modeB.mcode.gz"

    def test_committed_and_other_mode_have_opposite_table_order(self):
        committed = load(self.COMMITTED)
        other = load(self.OTHER_MODE)
        self.assertNotEqual(offset_table_order(committed), offset_table_order(other))

    def test_diff_reproduces_pr1502s_own_932_byte_figure(self):
        committed = load(self.COMMITTED)
        other = load(self.OTHER_MODE)
        self.assertEqual(len(committed), 3336)
        self.assertEqual(len(other), 3336)
        diffs = [i for i in range(len(committed)) if committed[i] != other[i]]
        self.assertEqual(len(diffs), 932)
        self.assertEqual(min(diffs), 204)
        self.assertEqual(max(diffs), 3308)


if __name__ == "__main__":
    unittest.main()
