"""Tests whether `tests/test_axera_conv_reg60_mechanism.py` (PR #1586)'s
own Conv binary-cluster finding -- `reg=60` (and 27 other bytes) flip
between two complete alternate states across independent rebuilds of
`Conv(k=3, dilation=3, pad=3, cin=4, cout=4, insz=16)` -- is specific to
that exact `dilation=3` shape, or a general Conv phenomenon.

`tests/test_axera_matmul_binary_cluster_search.py` (PR #1592) already
applied PR #1586's own brute-force scan method to MatMul (clean
negative) and Gemm (inconclusive, defeated by diffuse positional
reflow) -- but neither of those is Conv itself, and nobody had checked
whether Conv's OWN phenomenon shows up at a *different* Conv shape.

## Method

Built 8 independent rebuilds of `Conv(k=3, dilation=1, pad=1, cin=4,
cout=4, insz=16)` -- identical to PR #1586's own shape in every
parameter except `dilation`/`pad` (both reduced from 3 to 1, chosen the
same way `scripts/axera/README.md`'s own dilation-pair experiments do,
so the output shape -- and thus total serialized mcode length -- stays
identical: `insz + 2*pad - dilation*(k-1) - 1 + 1 = 16` for both
`(dilation=1, pad=1)` and `(dilation=3, pad=3)`). All 8 samples decode
to the identical 2984-byte length with zero `mcode.check()` errors, via
a standalone build script bypassing `pulsar2_docker.py`'s module-level
`onnxsim` import (a pre-built `onnxsim_cpp2py_export` extension borrowed
from a sibling checkout at import time, not committed -- gitignored,
confirmed via `git status`), using the `pulsar2:7.0-lite` image (the
only one loaded in this worktree's Docker daemon).

## Result: a clean, confident negative -- the binary-cluster switch does
## NOT appear at dilation=1

**All 9 near-universal banks are content-stable across all 8 samples.**
**35 of 36 near-universal registers are content-stable; only `reg=8`
varies** -- the already-known, separately-decoded pool mechanism
(`tests/test_axera_conv_reg8_reg60_noise_source.py` PR #1580), not a new
finding. Critically, `reg=60` itself -- PR #1586's own original anchor
-- reads the exact same byte (`0x7e`) in every one of the 8 samples,
zero exceptions.

The exact same brute-force method PR #1586 used (the union of every
pairwise byte-diff across all samples of a fixed-length fixture) finds
**only 8 distinct differing offsets total, all of them inside the
already-known `reg=8` pool group's own window** (bytes 297-325 -- the
group's stable `verb=162,bank=0,field=0` anchor at 297, two more fixed
V-kind slots at 305/313, and the `S`-kind `reg=8` slot at 322).
**Zero offsets outside that window differ in any pairwise comparison
across all 28 pairs of the 8 samples.** This is Conv's own version of
PR #1592's MatMul negative: once the already-decoded `reg=8` mechanism
is accounted for, there is no additional hidden source of rebuild-to-
rebuild variation at this shape -- in sharp contrast to PR #1586's own
`dilation=3` finding, where 26 offsets OUTSIDE `reg=60`'s own two
copies moved in lockstep.

## A secondary, honestly-flagged observation: this shape's `reg=8` group
## looks like Mul/MatMul's clean form, not Conv's own `dilation=3` form

Not the primary question this file was written to answer, but
noteworthy: at `dilation=3`, PR #1580 decoded Conv's `reg=8` group as a
"3-of-4, always-omits-one-class, variable-register-label" mechanism.
Here, at `dilation=1`, the group instead shows the exact same
**fixed 4-slot, always-a-complete-4-of-4-permutation** structure PR
#1583 (MatMul) and PR #1591 (Mul) decoded, at the identical relative
offsets (anchor +0/+8/+16/+24) -- all 4 slots always present, all 4
pool classes always used, no omission. Sample `_r7` also shows a
`\\x13\\x00` leading-byte variant PR #1591 found was Mul-specific, not
previously seen for Conv at `dilation=3`. This suggests the `reg=8`
mechanism's own exact slot/register-label structure (not just its
value pool, already shown shape-invariant by PR #1585) may itself
depend on the specific Conv shape/dilation, not just the op. Not
decoded further here -- flagged as a concrete lead for a future
investigation, not claimed as a general rule from one data point.

## What this establishes

Conv's PR #1586 binary-cluster phenomenon does **not** generalize to
`dilation=1` -- a clean, confident negative (not a methodologically-
limited one, unlike PR #1592's own Gemm attempt). Combined with PR
#1592's MatMul negative, this narrows the phenomenon: it is either
specific to `dilation=3` (or dilation >= some threshold -- this
project's own established dilation-trigger work, PR #1508/#1509/#1519,
already found several other Conv fields that only activate past a
specific dilation value), or specific to this one exact shape's own
tiling/scheduling decision. Distinguishing those two readings needs a
`dilation=2` or `dilation=4` probe, not attempted here.
"""

import gzip
import itertools
import os
import sys
import unittest

_AXERA_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts", "axera")
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import mcode  # noqa: E402

FIX = os.path.join(_AXERA_DIR, "fixtures")

ALL_NAMES = [f"conv_dilation1_r{i}.mcode.gz" for i in range(8)]

CORE_BANKS = (0x00, 0x01, 0x02, 0x03, 0x04, 0x0E, 0x0F, 0x1C, 0x1E)
CORE_REGS = (
    0,
    2,
    7,
    8,
    9,
    10,
    12,
    14,
    16,
    18,
    20,
    22,
    24,
    26,
    28,
    32,
    36,
    38,
    40,
    52,
    60,
    62,
    66,
    72,
    74,
    76,
    78,
    96,
    100,
    101,
    106,
    107,
    120,
    126,
    128,
    138,
)

# The known reg=8 pool group's own window (anchor at 297, 8-byte-spaced
# slots through the reg=8 slot at 322, plus its trailing L/raw framing
# bytes) -- see TestAllPairwiseDiffsAreConfinedToTheReg8Window below,
# which recomputes this from scratch rather than trusting this literal.
REG8_WINDOW = range(297, 326)


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def decode(name):
    return mcode.decode(load(name), **mcode.FULL_RULE)


def bank_records(recs, bank):
    return sorted(
        (r["field"], r.get("value"))
        for r in recs
        if r["kind"] == "V" and r["bank"] == bank
    )


def reg_records(recs, reg):
    return [
        (r["kind"], r.get("tag"), r.get("payload"))
        for r in recs
        if r["kind"] in ("S", "B") and r.get("reg") == reg
    ]


class TestAllEightSamplesAreTheSameLength(unittest.TestCase):
    def test_2984_bytes_every_time(self):
        for n in ALL_NAMES:
            self.assertEqual(len(load(n)), 2984, n)

    def test_all_decode_cleanly(self):
        for n in ALL_NAMES:
            self.assertEqual(mcode.check(load(n)), [], n)


class TestCoreBanksAreContentStable(unittest.TestCase):
    def test_nine_banks_stable_across_all_eight(self):
        all_recs = [decode(n) for n in ALL_NAMES]
        for bank in CORE_BANKS:
            sets = [bank_records(recs, bank) for recs in all_recs]
            self.assertTrue(
                all(s == sets[0] for s in sets),
                f"bank {bank:#04x} should be content-stable across rebuilds",
            )
            self.assertGreater(len(sets[0]), 0, f"bank {bank:#04x} should be present")


class TestOnlyReg8VariesAmongCoreRegisters(unittest.TestCase):
    """35 of 36 near-universal registers are stable; only reg=8 (the
    already-known, separately-decoded pool mechanism) varies. Critically,
    reg=60 -- PR #1586's own anchor register -- does NOT vary here."""

    def test_reg60_is_constant_0x7e_in_every_sample(self):
        for n in ALL_NAMES:
            recs = decode(n)
            hits = [r for r in recs if r.get("reg") == 60 and r.get("tag") == 131]
            self.assertEqual(len(hits), 1, n)
            self.assertEqual(hits[0]["payload"][-1], 0x7E, n)

    def test_only_reg8_is_unstable(self):
        all_recs = [decode(n) for n in ALL_NAMES]
        noisy = []
        for reg in CORE_REGS:
            sets = [reg_records(recs, reg) for recs in all_recs]
            if not all(s == sets[0] for s in sets):
                noisy.append(reg)
        self.assertEqual(noisy, [8])


class TestAllPairwiseDiffsAreConfinedToTheReg8Window(unittest.TestCase):
    """PR #1586's exact brute-force method, applied here: the union of
    every pairwise byte-diff across all 8 samples. Recomputes the
    diffing-offset set from scratch (not trusted from the module
    docstring's own claim) and confirms it is entirely contained inside
    the already-known reg=8 pool group's own byte window -- a clean
    negative, not a methodologically-limited one (unlike PR #1592's own
    Gemm attempt, this shape's record count and byte layout are fully
    constant across samples)."""

    def test_zero_diffs_outside_the_reg8_window(self):
        datas = {n: load(n) for n in ALL_NAMES}
        length = len(datas[ALL_NAMES[0]])
        for n in ALL_NAMES:
            self.assertEqual(len(datas[n]), length, n)

        all_diff_offsets = set()
        for a, b in itertools.combinations(ALL_NAMES, 2):
            da, db = datas[a], datas[b]
            all_diff_offsets.update(i for i in range(length) if da[i] != db[i])

        self.assertTrue(all_diff_offsets, "expected at least the known reg=8 diffs")
        outside = all_diff_offsets - set(REG8_WINDOW)
        self.assertEqual(
            outside, set(), f"found diffs outside the reg=8 window: {outside}"
        )

    def test_pairwise_diff_offsets_are_a_small_fixed_set(self):
        # Exact set found: {301, 303, 309, 311, 317, 319, 323, 325} --
        # the reg=8 group's 4 slots' own trailing (varying) bytes.
        datas = {n: load(n) for n in ALL_NAMES}
        length = len(datas[ALL_NAMES[0]])
        all_diff_offsets = set()
        for a, b in itertools.combinations(ALL_NAMES, 2):
            da, db = datas[a], datas[b]
            all_diff_offsets.update(i for i in range(length) if da[i] != db[i])
        self.assertEqual(all_diff_offsets, {301, 303, 309, 311, 317, 319, 323, 325})


class TestReg8GroupLooksLikeMulMatMulsCleanForm(unittest.TestCase):
    """Secondary observation (see module docstring): at dilation=1, the
    reg=8 group is a fixed 4-slot, always-a-complete-permutation
    structure -- like Mul (PR #1591) and MatMul (PR #1583) -- not
    Conv's own dilation=3 "3-of-4, variable register label" form
    (PR #1580)."""

    def test_anchor_group_has_four_fixed_offset_slots(self):
        POOL_TRAILING_BYTES = {0x10, 0x20, 0x30, 0x40}
        for n in ALL_NAMES:
            recs = decode(n)
            anchor_hits = [
                r
                for r in recs
                if r["kind"] == "V"
                and r.get("verb") == 162
                and r.get("bank") == 0
                and r.get("field") == 0
                and r.get("at") == 297
            ]
            self.assertEqual(len(anchor_hits), 1, n)

            v_slots = [
                r
                for r in recs
                if r["kind"] == "V"
                and r.get("verb") == 162
                and r.get("bank") == 0
                and r.get("field") == 0
                and r.get("at") in (297, 305, 313)
            ]
            self.assertEqual(len(v_slots), 3, n)

            reg8_hit = [
                r
                for r in recs
                if r["kind"] == "S"
                and r.get("reg") == 8
                and r.get("tag") == 130
                and r.get("at") == 322
            ]
            self.assertEqual(len(reg8_hit), 1, n)

            classes = {r["operand"][2] for r in v_slots} | {reg8_hit[0]["payload"][-1]}
            self.assertTrue(classes <= POOL_TRAILING_BYTES, (n, classes))
            self.assertEqual(
                len(classes), 4, f"{n}: expected a complete 4-of-4 permutation"
            )


if __name__ == "__main__":
    unittest.main()
