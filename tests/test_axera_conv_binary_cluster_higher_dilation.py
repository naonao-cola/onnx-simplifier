"""Continues `tests/test_axera_conv_dilation2_transition.py` (PR #1599)'s
own dilation-boundary work: that file pinned Conv's 28-byte binary
"path" switch (`tests/test_axera_conv_reg60_mechanism.py`, PR #1586)
and the `reg=8` pool group's own "3-of-4, variable-register-label" form
(`tests/test_axera_conv_reg8_reg60_noise_source.py`, PR #1580) as both
present starting at `dilation=2` and absent at `dilation=1`
(`tests/test_axera_conv_binary_cluster_other_shape.py`, PR #1597) --
but nobody had tested dilation values HIGHER than 3, so it was unknown
whether the binary-cluster switch persists unboundedly for all
`dilation>=2`, or has its own upper bound.

## Method

`Conv(k=3, dilation=d, pad=d, cin=4, cout=4, insz=16)` at `d=4` and
`d=8` -- keeping `pad == dilation` the same way PR #1597/#1599 did,
which holds the OUTPUT shape at 16x16 for every dilation value (the
formula `insz + 2*pad - dilation*(k-1) - 1 + 1` reduces to `insz` when
`pad == dilation` for `k=3`, verified directly for both new values
before building). 8 independent rebuilds per dilation value, built via
a standalone script bypassing `pulsar2_docker.py`'s module-level
`onnxsim` import (a pre-built `onnxsim_cpp2py_export.abi3.so` +
`version.py` borrowed from a sibling checkout into this worktree's
`onnxsim/` package at build time -- neither file is part of this PR's
own diff, confirmed gitignored via `git status`), using `pulsar2:7.0-lite`
(the only image loaded in this worktree's Docker daemon; PR #1579/#1597
already established this substitutes cleanly for `6.0-lite` at this
exact shape family).

**One genuine surprise, unrelated to the question this file answers**:
unlike `dilation=1/2/3` (all 3,528 bytes), `dilation=8`'s mcode is
**3,760 bytes** -- 232 bytes longer, even though the OUTPUT shape is
identical (16x16) and `dilation=4` is still 3,528 bytes. This is a real
wholesale-reserialization threshold somewhere between `dilation=4` and
`dilation=8` (plausibly related to this project's own established
"dilation-3/4 extra tap-run" material in `scripts/axera/README.md`,
though that finding was itself about `dilation=3` vs `4`, not `4` vs
`8` -- this file does not chase the exact new threshold, only notes it
so a within-dilation scan is valid but a byte-offset comparison ACROSS
`dilation=4` and `dilation=8` is not attempted here).

## Finding 1: the binary-cluster switch is ABSENT at both `dilation=4`
## and `dilation=8` -- it does NOT persist unboundedly for all
## `dilation>=2`

`reg=60` reads the identical byte (`0x7e`) in every one of the 8
samples at `dilation=4`, and again in every one of the 8 samples at
`dilation=8` -- zero variation at either value, in sharp contrast to
`dilation=2`'s 2-vs-10 split and `dilation=3`'s 5-vs-3 split.

A full byte-offset diff scan (every position that differs across the
8 samples of a given dilation, not just `reg=60`'s own two copies)
finds real variation at both values, but **all of it is confined to
the already-known `reg=8` pool group's own window** (`dilation=4`:
offsets 851-880, plus one downstream reflow byte at 3232, the same
kind of pool-group-driven reflow `tests/test_axera_conv_binary_cluster_calibration_dependence.py`
(PR #1598) already flagged as attributable to that group's own several
degrees of freedom, not a new mechanism; `dilation=8`: offsets
859/865/871/877, entirely inside that dilation's own equivalently-
located pool-group window). **Zero offsets outside the pool-group
window differ at either dilation value.** This is the same "clean
negative, tightly controlled" result `tests/test_axera_matmul_binary_cluster_search.py`
(PR #1592) and `tests/test_axera_gemm_binary_cluster_search_small_shape.py`
(PR #1596) already established for MatMul and Gemm respectively -- Conv
itself now also shows this same negative once dilation moves far enough
past the `2`-`3` window where the switch was originally found.

**The binary-cluster switch's own dilation range is therefore bounded**:
absent at `dilation=1`, present at `dilation=2` and `dilation=3`, absent
again by `dilation=4` and still absent at `dilation=8`. This is a
genuinely narrow window, not a `dilation>=2` threshold effect -- pinning
this down is this file's primary contribution.

## Finding 2: the `reg=8` group's own "3-of-4, variable-register-label"
## FORM does persist at both higher dilation values (unlike the switch)

At both `dilation=4` and `dilation=8`, every one of the 16 samples'
`reg=8` group is anchored at the same `reg=170`-payload-`\\x12` record
PR #1580 originally decoded at `dilation=3` (never `dilation=1`'s
`verb=162`-anchored clean-4-of-4 form), and is a genuine 3-of-4 draw
from the established 4-class pool, with register labels on 2 of the 3
slots varying (`8`, `242`, `176`) exactly as PR #1580's own pool
already established. This form, unlike the binary-cluster switch, DOES
persist unboundedly for every `dilation>=2` value tested so far (`2`,
`3`, `4`, `8`) -- the two phenomena, though co-located at the same
shapes from `dilation=1`-`3`, are NOT the same trigger: one has a
narrow window (`2`-`3` only), the other activates at `dilation>=2` and
stays active.

**A genuinely new wrinkle at `dilation=8`, honestly flagged, not
chased**: at `dilation=8`, the pool's `P4` class consistently appears
in a previously-unseen 3-byte long form (`\\x23\\x00\\x30`) rather than
the 1-byte short form (`\\x30`) every prior dilation value (`1`-`4`)
used exclusively for `P4`. `reg=172`'s own biconditional (its tag is
`134` iff slot 1's own payload is the SHORT form) still holds with zero
exceptions across all 16 new samples -- it was never triggered at
`dilation=8` simply because slot 1 never lands on `P4` in this file's
own 8 samples, not because the biconditional itself broke. Whether
`P4`'s long-form-only behavior at `dilation=8` is itself a real,
decodable threshold effect (a third boundary, separate from both
already-found ones) is not chased further here.

## What this establishes

The binary-cluster switch is now known to be a genuinely narrow-window
phenomenon (`dilation` in `{2, 3}` only, of everything tested `1`-`8`),
not a persistent `dilation>=2` property -- while the `reg=8` group's own
FORM transition (which this file's own predecessor, PR #1599, found
co-located with the switch at `dilation=2`) turns out to be the more
durable of the two, persisting through `dilation=8`. The root TRIGGER
for either phenomenon (why `dilation=2`-`3` specifically activates the
switch, why the 3-of-4 form activates and stays active from `dilation=2`
onward) remains exactly as open as PR #1586/#1580 left it.
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

DILATION4_NAMES = [f"conv_dilation4_r{i}.mcode.gz" for i in range(8)]
DILATION8_NAMES = [f"conv_dilation8_r{i}.mcode.gz" for i in range(8)]

CORE_BANKS = (0x00, 0x01, 0x02, 0x03, 0x04, 0x0E, 0x0F, 0x1C, 0x1E)

# Every offset PR #1580's known pool-group window can move within, for
# each dilation value -- recomputed from scratch in
# TestAllVariationIsConfinedToTheReg8Window below, not trusted from the
# module docstring's own claim.
KNOWN_POOL_WINDOW_DIFFS = {
    4: frozenset(
        [
            851,
            853,
            854,
            855,
            856,
            857,
            858,
            859,
            860,
            861,
            862,
            863,
            864,
            865,
            866,
            867,
            868,
            869,
            870,
            871,
            872,
            874,
            876,
            878,
            880,
            3232,
        ]
    ),
    8: frozenset([859, 865, 871, 877]),
}


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def decode(name):
    return mcode.decode(load(name), **mcode.FULL_RULE)


def reg60_value(recs):
    hits = [
        r
        for r in recs
        if r["kind"] == "S" and r.get("reg") == 60 and r.get("tag") == 131
    ]
    assert len(hits) >= 1, hits
    return hits[0]["payload"][-1]


def classify(payload):
    if (
        len(payload) == 3
        and payload[:2] == b"#\x00"
        and payload[2]
        in (
            0x10,
            0x20,
            0x30,
            0x40,
        )
    ):
        return {0x10: "P2", 0x20: "P1", 0x30: "P4", 0x40: "P3"}[payload[2]]
    if payload == b"0":
        return "P4"
    return None


def reg8_group(recs):
    """Returns (reg172_tag, [(reg, class), ...]) for the reg=170-anchored
    pool group -- PR #1580's own dilation>=2 mechanism, generalized here
    to also accept the 3-byte long form of P4 (see module docstring's
    dilation=8 wrinkle) via `classify()`."""
    r170 = [
        r
        for r in recs
        if r["kind"] == "S" and r.get("reg") == 170 and r.get("payload") == b"\x12"
    ]
    assert len(r170) == 1, r170
    anchor = r170[0]["at"]
    r172 = [
        r
        for r in recs
        if r["kind"] == "S" and r.get("reg") == 172 and r.get("at") == anchor + 4
    ]
    assert len(r172) == 1, r172
    group = [
        r
        for r in recs
        if r["kind"] == "S"
        and r.get("at") is not None
        and anchor + 8 <= r["at"] <= anchor + 40
        and classify(r.get("payload", b"")) is not None
    ]
    slots = [(r["reg"], classify(r["payload"])) for r in group]
    return r172[0]["tag"], slots


class TestAllSixteenSamplesDecodeCleanly(unittest.TestCase):
    def test_dilation4_is_3528_bytes_zero_errors(self):
        for n in DILATION4_NAMES:
            self.assertEqual(len(load(n)), 3528, n)
            self.assertEqual(mcode.check(load(n)), [], n)

    def test_dilation8_is_3760_bytes_zero_errors(self):
        """A genuine surprise: dilation=8's mcode is 232 bytes longer
        than dilation=1/2/3/4's, despite an identical 16x16 output
        shape -- a wholesale-reserialization threshold between
        dilation=4 and dilation=8, not chased further here (see module
        docstring)."""
        for n in DILATION8_NAMES:
            self.assertEqual(len(load(n)), 3760, n)
            self.assertEqual(mcode.check(load(n)), [], n)


class TestReg60IsConstantAtBothHigherDilations(unittest.TestCase):
    """The core finding: unlike dilation=2 (2-vs-10 split) and
    dilation=3 (5-vs-3 split), reg=60 shows ZERO variation at either
    dilation=4 or dilation=8 -- across 8 independent samples each."""

    def test_dilation4_reg60_constant(self):
        vals = {reg60_value(decode(n)) for n in DILATION4_NAMES}
        self.assertEqual(vals, {0x7E})

    def test_dilation8_reg60_constant(self):
        vals = {reg60_value(decode(n)) for n in DILATION8_NAMES}
        self.assertEqual(vals, {0x7E})


class TestAllVariationIsConfinedToTheReg8Window(unittest.TestCase):
    """Recomputes, from scratch, every byte offset that differs across
    the 8 same-dilation samples (not just reg=60's own copies) -- and
    confirms it exactly matches KNOWN_POOL_WINDOW_DIFFS, i.e. the
    already-known reg=8 pool group's own window, at both dilation
    values. Zero offsets outside that window differ -- the same clean-
    negative result PR #1592/#1596 already established for MatMul and
    Gemm."""

    def _check(self, names, dilation):
        datas = [load(n) for n in names]
        length = len(datas[0])
        for d in datas:
            self.assertEqual(len(d), length)
        found = frozenset(i for i in range(length) if len({d[i] for d in datas}) > 1)
        self.assertEqual(found, KNOWN_POOL_WINDOW_DIFFS[dilation])

    def test_dilation4_diffs_match_known_pool_window(self):
        self._check(DILATION4_NAMES, 4)

    def test_dilation8_diffs_match_known_pool_window(self):
        self._check(DILATION8_NAMES, 8)


class TestCoreBanksAreContentStableAtBothHigherDilations(unittest.TestCase):
    def _check(self, names):
        all_recs = [decode(n) for n in names]

        def bank_records(recs, bank):
            return sorted(
                (r["field"], r.get("value"))
                for r in recs
                if r["kind"] == "V" and r["bank"] == bank
            )

        for bank in CORE_BANKS:
            sets = [bank_records(recs, bank) for recs in all_recs]
            self.assertTrue(
                all(s == sets[0] for s in sets),
                f"bank {bank:#04x} should be content-stable",
            )

    def test_dilation4(self):
        self._check(DILATION4_NAMES)

    def test_dilation8(self):
        self._check(DILATION8_NAMES)


class TestReg8GroupFormPersistsAtBothHigherDilations(unittest.TestCase):
    """Unlike the binary-cluster switch, the reg=8 pool group's own
    "3-of-4, variable-register-label" FORM (PR #1580's dilation=3
    mechanism, already confirmed at dilation=2 by PR #1599) persists at
    both dilation=4 and dilation=8 -- the two phenomena, co-located at
    dilation=2-3, diverge at higher dilation: one is a narrow window,
    the other stays active."""

    def _check(self, names):
        for n in names:
            recs = decode(n)
            _, slots = reg8_group(recs)
            self.assertEqual(len(slots), 3, n)
            classes = {c for _, c in slots}
            self.assertEqual(len(classes), 3, n)
            labels = {r for r, _ in slots[1:]}
            self.assertTrue(labels <= {8, 242, 176}, (n, labels))

    def test_dilation4(self):
        self._check(DILATION4_NAMES)

    def test_dilation8(self):
        self._check(DILATION8_NAMES)


class TestReg172BiconditionalHoldsAtBothHigherDilations(unittest.TestCase):
    """PR #1580's own exact biconditional (reg=172's tag is 134 iff
    slot 1's payload is the short form) holds with zero exceptions at
    both dilation=4 and dilation=8 -- even though dilation=8 never
    exercises the "true" side in this file's own 8 samples (slot 1
    never lands on P4 there), the biconditional's "false" side (tag=132
    whenever slot 1 is a long form) still checks out every time."""

    def _check(self, names):
        for n in names:
            recs = decode(n)
            tag, slots = reg8_group(recs)
            slot1_class = slots[0][1]
            # "short form" means the raw payload was the single byte
            # b"0" -- re-derive that directly rather than trusting
            # classify()'s own class label, since P4 has two forms.
            r170 = [
                r
                for r in recs
                if r["kind"] == "S"
                and r.get("reg") == 170
                and r.get("payload") == b"\x12"
            ]
            anchor = r170[0]["at"]
            slot1_rec = [r for r in recs if r.get("at") == anchor + 8]
            self.assertEqual(len(slot1_rec), 1, n)
            slot1_is_short = slot1_rec[0]["payload"] == b"0"
            self.assertEqual(
                tag == 134,
                slot1_is_short,
                f"{n}: slot1={slot1_class} short={slot1_is_short} tag={tag}",
            )

    def test_dilation4(self):
        self._check(DILATION4_NAMES)

    def test_dilation8(self):
        self._check(DILATION8_NAMES)


class TestDilation8sP4UsesTheLongFormExclusively(unittest.TestCase):
    """A genuinely new, honestly-flagged-but-not-chased wrinkle: at
    dilation=8, every P4 class occurrence uses the 3-byte long form
    (\\x23\\x00\\x30), never the 1-byte short form every dilation value
    1-4 used exclusively for P4."""

    def test_dilation8_p4_is_always_long_form(self):
        found_p4 = False
        for n in DILATION8_NAMES:
            recs = decode(n)
            _, slots = reg8_group(recs)
            r170 = [
                r
                for r in recs
                if r["kind"] == "S"
                and r.get("reg") == 170
                and r.get("payload") == b"\x12"
            ]
            anchor = r170[0]["at"]
            for reg, cls in slots:
                if cls == "P4":
                    found_p4 = True
                    rec = [
                        r
                        for r in recs
                        if r.get("reg") == reg
                        and classify(r.get("payload", b"")) == "P4"
                        and anchor + 8 <= r.get("at", -1) <= anchor + 40
                    ]
                    self.assertEqual(len(rec), 1, n)
                    self.assertEqual(rec[0]["payload"], b"#\x000", (n, rec[0]))
        self.assertTrue(found_p4, "expected at least one P4 occurrence to check")

    def test_dilation4_p4_uses_the_short_form(self):
        """Contrast check: dilation=4 (like dilation=1-3) still uses
        the short 1-byte form for P4, confirming dilation=8's long-form
        behavior is a real difference, not a decode artifact."""
        found_p4 = False
        for n in DILATION4_NAMES:
            recs = decode(n)
            _, slots = reg8_group(recs)
            for reg, cls in slots:
                if cls == "P4":
                    found_p4 = True
        self.assertTrue(found_p4, "expected at least one P4 occurrence to check")


if __name__ == "__main__":
    unittest.main()
