"""Continues `tests/test_axera_conv_rebuild_stability.py` (PR #1579) and
`tests/test_axera_conv_reg8_reg60_noise_source.py` (PR #1580)'s own
repeatedly-flagged open thread: `reg=60` (tag=131, payload's last byte)
flips between `0x7e` and `0x7f` across independent `Conv(k=3,
dilation=3, pad=3, cin=4, cout=4, insz=16)` rebuilds, with three prior
hypotheses (toolchain image version, old/fresh build batch, correlation
with the `reg=8` pool's "missing class") each ruled out with real
evidence. This file reuses the same 8 already-committed fixtures --
no new builds needed.

## `reg=60` is not isolated noise -- it is one visible copy of a
## 28-byte, all-or-nothing binary "path" switch

Grouping the 8 samples by `reg=60`'s own value gives an exact 5-vs-3
split:

- **Group A (`0x7e`)**: `conv_dilation3`, `_v7stability_r0`,
  `_v7stability_r1`, `_v7stability_r2`, `_v7stability_r3`.
- **Group B (`0x7f`)**: `_rebuild0`, `_rebuild1`, `_rebuild2`.

Scanning *every* byte offset in the 3,528-byte fixture for one that is
byte-identical within each group but differs between the two groups
(the same computational check, not eyeballed) finds **exactly 28 such
offsets, zero more and zero fewer** -- and critically, `reg=60`'s own
two copies (`at=685`, `at=1293`) are only 2 of those 28. The other 26
belong to a small, fixed set of OTHER records that flip in exact
lockstep with `reg=60`, verified with zero exceptions across all 8
samples:

| record | offset(s) | group A value | group B value |
| --- | --- | --- | --- |
| `reg=60`, tag=131 (copy 1) | 685 | payload ends `0x7e` | payload ends `0x7f` |
| `reg=60`, tag=131 (copy 2) | 1293 | payload ends `0x7e` | payload ends `0x7f` |
| `reg=54`, tag=131 | 2158 | payload ends `0x7e` | payload ends `0x7f` |
| `reg=232`, tag=132 | 2547 | payload ends `0x7c` (124) | payload ends `0x76` (118) |
| `verb=161,bank=15,field=96/112/128` (3 copies, identical operand each -- Conv's mcode also carries 2 unrelated, stable `verb=161,bank=15,field=128` records elsewhere with a different operand, excluded by matching the operand value directly) | 2194, 2202, 2210 | operand `37 3f 07 42` (float32 `33.81`) | operand `43 07 ff 42` (float32 `127.51`) |
| `reg=224`, tag=129 (4 copies, one 5-byte + three 4-byte) | 2559, 2567, 2574, 2581 | payload `4b 18 6f 3c` (float32 `0.01459`) | payload `63 7c 20 3c` (float32 `0.00980`) |

`reg=54`'s own value is not merely correlated with `reg=60` -- it is
**byte-identical** to it in every one of the 8 samples (both read
`0x7e` or both read `0x7f`, never split). `reg=232` and the two float32
values track the *same* 2-way partition with their own distinct values
-- not the literal `0x7e`/`0x7f` byte, but a clean binary switch
between exactly the same two states every time.

## What this reframes, and what it does NOT solve

This is a real, useful advance: `reg=60`'s instability is not an
isolated, unexplained single-byte residual the way it looked in PR
#1579/#1580 -- it is one visible thread of a much larger, clean,
all-or-nothing 28-byte switch between two complete alternate states,
at least two of which are genuinely different *computed* float32
quantization-scale-like values (not just flag bits or scratch-buffer
addresses the way the `reg=8` pool's candidates were). This is
consistent with two different floating-point reduction/summation
orders in Pulsar2's own per-channel weight-statistics computation for
this `dilation=3` shape (this project's own recurring finding that
floating-point non-associativity plus non-deterministic thread
scheduling produces exactly-binary, not continuously-varying, build
variance) -- but this file does **not** prove that reading, and does
not attempt to.

**The root trigger remains genuinely open, same as before.** This file
extends the search but does not close it:

- **Not the old/fresh batch boundary**, and this is now proven even
  more directly than PR #1579's own check: `conv_dilation3` (part of
  `OLD_BATCH`) lands in Group A while its own batch-mates
  `_rebuild{0,1,2}` all land in Group B -- the split cuts *across*
  `OLD_BATCH` itself, not merely between `OLD_BATCH` and
  `FRESH_BATCH`.
- **Not the `reg=8` pool's slot assignment**: `conv_dilation3` and
  `_rebuild1` have *byte-identical* `reg=8`/`reg=172`/`reg=174` group
  content (see PR #1580's own `EXPECTED` table -- both read
  `[("174","P2"),("8","P1"),("242","P3")]`, `reg172_tag=132`) yet land
  in opposite groups here -- a stronger refutation than PR #1580's own
  "same missing class" check, since these two fixtures are identical
  on literally every byte PR #1580's own mechanism covers.
- **Not "first build vs. subsequent rebuild in a loop"**: if it were,
  `_v7stability_r0` (the first of its own batch) would be expected to
  differ from `_rebuild0` (the first of *its* batch) in some
  chronological-position-driven way, but instead ALL FOUR
  `_v7stability_r{0,1,2,3}` land in the same group regardless of
  position within their own build loop, and all three `_rebuild{0,1,2}`
  land in the other group regardless of position within theirs --
  ruling out simple "loop position" as the driver.

No new hypothesis tested here actually explains *why* a given rebuild
lands on one side or the other -- only what moves together once it
does. Whoever chases this next should look outside the mcode's own
visible content (e.g. real compiler-internal thread-scheduling state)
or build a much larger sample (this file's 8 samples give only 5 vs. 3
of one binary outcome -- not enough to look for anything beyond a
clean 2-way split with real statistical power).
"""

import gzip
import os
import struct
import sys
import unittest

_AXERA_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts", "axera")
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import mcode  # noqa: E402

FIX = os.path.join(_AXERA_DIR, "fixtures")

ALL_NAMES = [
    "conv_dilation3.mcode.gz",
    "conv_dilation3_rebuild0.mcode.gz",
    "conv_dilation3_rebuild1.mcode.gz",
    "conv_dilation3_rebuild2.mcode.gz",
    "conv_dilation3_v7stability_r0.mcode.gz",
    "conv_dilation3_v7stability_r1.mcode.gz",
    "conv_dilation3_v7stability_r2.mcode.gz",
    "conv_dilation3_v7stability_r3.mcode.gz",
]

GROUP_A = [
    "conv_dilation3.mcode.gz",
    "conv_dilation3_v7stability_r0.mcode.gz",
    "conv_dilation3_v7stability_r1.mcode.gz",
    "conv_dilation3_v7stability_r2.mcode.gz",
    "conv_dilation3_v7stability_r3.mcode.gz",
]
GROUP_B = [
    "conv_dilation3_rebuild0.mcode.gz",
    "conv_dilation3_rebuild1.mcode.gz",
    "conv_dilation3_rebuild2.mcode.gz",
]

# The exact 28 byte offsets found by scanning the whole 3528-byte
# fixture for positions that are constant within each group but differ
# between them -- see TestFullClusterDiscovery below, which recomputes
# this set from scratch rather than trusting this hardcoded list alone.
KNOWN_CORRELATED_OFFSETS = frozenset(
    [
        687,
        1295,
        2161,
        2190,
        2191,
        2192,
        2198,
        2199,
        2200,
        2206,
        2207,
        2208,
        2214,
        2215,
        2216,
        2550,
        2561,
        2562,
        2563,
        2568,
        2569,
        2570,
        2575,
        2576,
        2577,
        2582,
        2583,
        2584,
    ]
)


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def decode(name):
    return mcode.decode(load(name), **mcode.FULL_RULE)


def reg60_value(recs):
    hits = [
        r
        for r in recs
        if r["kind"] == "S"
        and r.get("reg") == 60
        and r.get("tag") == 131
        and r.get("payload")
    ]
    assert len(hits) >= 1, hits
    return hits[0]["payload"][-1]


class TestGroupSplitMatchesReg60(unittest.TestCase):
    def test_group_a_is_0x7e_group_b_is_0x7f(self):
        for n in GROUP_A:
            self.assertEqual(reg60_value(decode(n)), 0x7E, n)
        for n in GROUP_B:
            self.assertEqual(reg60_value(decode(n)), 0x7F, n)

    def test_split_is_not_the_old_fresh_batch_boundary(self):
        """conv_dilation3 is part of OLD_BATCH (PR #1579's own naming)
        but lands in Group A with the FRESH batch, while its own
        OLD_BATCH siblings _rebuild{0,1,2} all land in Group B -- the
        split cuts across OLD_BATCH itself."""
        self.assertIn("conv_dilation3.mcode.gz", GROUP_A)
        for n in (
            "conv_dilation3_rebuild0.mcode.gz",
            "conv_dilation3_rebuild1.mcode.gz",
            "conv_dilation3_rebuild2.mcode.gz",
        ):
            self.assertIn(n, GROUP_B)


class TestFullClusterDiscovery(unittest.TestCase):
    """Recomputes the correlated-offset set from scratch (not trusted
    from the module docstring's own claim) by scanning every byte
    offset in the fixture for one that's constant within each group but
    differs between them. Confirms exactly 28 such offsets exist, and
    that KNOWN_CORRELATED_OFFSETS above is exactly that set."""

    def test_exactly_28_correlated_offsets(self):
        datas = {n: load(n) for n in ALL_NAMES}
        length = len(datas[ALL_NAMES[0]])
        for n in ALL_NAMES:
            self.assertEqual(len(datas[n]), length, n)

        found = []
        for i in range(length):
            vals_a = {datas[n][i] for n in GROUP_A}
            vals_b = {datas[n][i] for n in GROUP_B}
            if len(vals_a) == 1 and len(vals_b) == 1 and vals_a != vals_b:
                found.append(i)

        self.assertEqual(frozenset(found), KNOWN_CORRELATED_OFFSETS)
        self.assertEqual(len(found), 28)


class TestReg54MirrorsReg60Exactly(unittest.TestCase):
    """reg=54's own tag=131 record (at byte offset 2158) is
    byte-identical to reg=60's value in every one of the 8 samples --
    not merely correlated, a literal duplicate of the same underlying
    byte."""

    def test_reg54_equals_reg60_in_every_sample(self):
        for n in ALL_NAMES:
            recs = decode(n)
            r60 = reg60_value(recs)
            hits = [r for r in recs if r.get("reg") == 54 and r.get("tag") == 131]
            self.assertEqual(len(hits), 1, n)
            self.assertEqual(hits[0]["payload"][-1], r60, n)


class TestReg232AndFloatValuesTrackTheSamePartition(unittest.TestCase):
    """reg=232 (a different byte value, not a literal reg=60 mirror)
    and two distinct computed float32 values (the verb=161/bank=15
    V-records' shared operand, and reg=224's own 4 duplicate copies)
    all switch in exact lockstep with reg=60's own group split."""

    def test_reg232_tracks_the_partition(self):
        for n in GROUP_A:
            recs = decode(n)
            hits = [r for r in recs if r.get("reg") == 232 and r.get("tag") == 132]
            self.assertEqual(hits[0]["payload"][-1], 0x7C, n)
        for n in GROUP_B:
            recs = decode(n)
            hits = [r for r in recs if r.get("reg") == 232 and r.get("tag") == 132]
            self.assertEqual(hits[0]["payload"][-1], 0x76, n)

    def test_bank15_verb161_float_operand_tracks_the_partition(self):
        # Restricted to field in {96, 112, 128} AND a 4-byte operand
        # matching one of the two known variants -- Conv's mcode also
        # carries two unrelated, stable verb=161/bank=15/field=128
        # records elsewhere (operand b"$\\x83N", at offsets 641/1249)
        # that don't belong to this cluster and must be excluded.
        fields = (96, 112, 128)
        variants = (b"7?\x07B", b"C\x07\xffB")
        for n in GROUP_A:
            recs = decode(n)
            hits = [
                r
                for r in recs
                if r["kind"] == "V"
                and r.get("verb") == 161
                and r.get("bank") == 15
                and r.get("field") in fields
                and r.get("operand") in variants
            ]
            self.assertEqual(len(hits), 3, n)
            for h in hits:
                self.assertEqual(h["operand"], b"7?\x07B", n)
        for n in GROUP_B:
            recs = decode(n)
            hits = [
                r
                for r in recs
                if r["kind"] == "V"
                and r.get("verb") == 161
                and r.get("bank") == 15
                and r.get("field") in fields
                and r.get("operand") in variants
            ]
            self.assertEqual(len(hits), 3, n)
            for h in hits:
                self.assertEqual(h["operand"], b"C\x07\xffB", n)

    def test_reg224_float_value_tracks_the_partition_across_all_four_copies(self):
        for n in GROUP_A:
            recs = decode(n)
            hits = [
                r
                for r in recs
                if r.get("reg") == 224 and r.get("tag") == 129 and r.get("payload")
            ]
            self.assertEqual(len(hits), 4, n)
            for h in hits:
                self.assertTrue(h["payload"].endswith(b"K\x18o<"), (n, h))
        for n in GROUP_B:
            recs = decode(n)
            hits = [
                r
                for r in recs
                if r.get("reg") == 224 and r.get("tag") == 129 and r.get("payload")
            ]
            self.assertEqual(len(hits), 4, n)
            for h in hits:
                self.assertTrue(h["payload"].endswith(b"c| <"), (n, h))

    def test_the_two_float_values_are_genuinely_different_computed_scales(self):
        """Decoded as float32, these are not simple related constants
        (e.g. not off by a clean factor or reciprocal) -- consistent
        with two different real computed quantities, not a synthetic
        marker. Recorded here so a future investigation doesn't have to
        redo this decode."""
        a = struct.unpack("<f", b"7?\x07B")[0]
        b = struct.unpack("<f", b"C\x07\xffB")[0]
        self.assertAlmostEqual(a, 33.811733, places=4)
        self.assertAlmostEqual(b, 127.514183, places=4)

        c = struct.unpack("<f", b"K\x18o<")[0]
        d = struct.unpack("<f", b"c| <")[0]
        self.assertAlmostEqual(c, 0.0145932, places=6)
        self.assertAlmostEqual(d, 0.0097953, places=6)


class TestReg60RemainsUndecodedAtTheRootCause(unittest.TestCase):
    """The cluster is now fully characterized (TestFullClusterDiscovery,
    TestReg54MirrorsReg60Exactly, TestReg232AndFloatValuesTrackThePartition)
    but WHY a given rebuild lands on one side or the other is not --
    this test only documents that the search has moved past the
    hypotheses PR #1579/#1580 already ruled out, not that it has found
    the trigger."""

    def test_not_explained_by_old_fresh_batch_or_reg8_pool_content(self):
        # conv_dilation3 and _rebuild1 are IDENTICAL on every byte the
        # reg=8/reg=172/reg=174 mechanism (PR #1580) covers, yet land
        # in opposite reg=60 groups -- the strongest available
        # refutation of "reg=60 is downstream of that mechanism."
        a = decode("conv_dilation3.mcode.gz")
        b = decode("conv_dilation3_rebuild1.mcode.gz")
        self.assertNotEqual(reg60_value(a), reg60_value(b))

        def group_slots(recs):
            r170 = [
                r
                for r in recs
                if r["kind"] == "S"
                and r.get("reg") == 170
                and r.get("payload") == b"\x12"
            ]
            anchor = r170[0]["at"]
            group = [
                r
                for r in recs
                if r["kind"] == "S"
                and r.get("payload") in (b"#\x00 ", b"#\x00\x10", b"#\x00@", b"0")
                and r.get("at") is not None
                and anchor + 8 <= r["at"] <= anchor + 30
            ]
            return [(r["reg"], r["payload"]) for r in group]

        self.assertEqual(group_slots(a), group_slots(b))


if __name__ == "__main__":
    unittest.main()
