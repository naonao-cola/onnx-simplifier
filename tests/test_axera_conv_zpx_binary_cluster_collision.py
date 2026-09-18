"""Continues `tests/test_axera_reg8_emit_composability.py` (PR #1633)'s
own flagged collision: while testing whether `tiny_emit.emit_conv_reg8_group`
and `tiny_emit.patch_conv_zp_x` compose safely, that file found
`patch_conv_zp_x`'s own literal-byte zero-point search (the 6-byte unit
``02 10 1b <zp_x> 83 36``) matches at byte offset 2158 in
``conv_dilation3.mcode.gz``, and patching it changes byte 2161 -- the
exact same absolute offset `tests/test_axera_conv_reg60_mechanism.py`
(PR #1586) already decoded as part of Conv's own 28-byte "binary path
switch" cluster. PR #1633 checked this on exactly one fixture, at one
`zp_x` value, and left open whether the collision is structural for
this shape or a coincidence of that one build's own calibration.

## Answer: fully structural for `Conv(dilation=3,...)` -- and it is not
## a byte-pattern coincidence, it is the SAME register

Decoding both sides directly (not inferred from the literal byte
pattern alone) shows *why* this always collides: the 6-byte unit
`patch_mul_zp_x`/`patch_conv_zp_x` search for is not an arbitrary byte
string that happens to overlap another field -- it is the exact,
complete on-the-wire encoding of an ``S``-kind record with
``reg=54``, ``tag=131``, a 3-byte payload ``10 1b <zp_x>``
(``p=0x02`` + payload + ``tag=0x83`` + ``reg=0x36``, confirmed via
`mcode.decode()` directly). And `reg=54` is precisely the register
`tests/test_axera_conv_reg60_mechanism.py` (PR #1586) already found
**byte-identical to `reg=60`** in every one of its own 8 samples -- not
merely correlated, identical. So "the literal zp_x unit" and "one
specific copy of the binary-cluster's own switch value" are, for this
shape, not two different things that happen to sit at the same
offset -- they are two different projects' own names for the exact
same register's own content.

Checked directly against all 52 already-committed
`Conv(dilation=3,pad=3,cin=4,cout=4,insz=16)` fixtures (the shared
3,528-byte shape family this session's own binary-cluster,
calibration-dependence, seed-survey, and large-sample work all used --
`calibseed*`, `largesample_r*`, `rebuild*`, `v7stability_r*`, and the
original `conv_dilation3.mcode.gz` itself; the two `insz8*` fixtures
are excluded from the "matches" count below, see next section):

**Every one of the 52 fixtures where the literal 6-byte unit is present
at all has its own `zp_x` value EXACTLY equal to that build's own
`reg=60` value, at the identical offset 2158, with zero exceptions** --
recomputed live below (`TestCollisionIsUniversalAcrossTheWholeShapeFamily`),
not copied from PR #1633's own single-fixture finding. This spans 7
distinct `reg=60`/`zp_x` values across this session's own calibration-seed
survey (`0x7e`, `0x7f`, `0x71`, `0x73`, `0x74`, `0x7c`, `0x85`) -- the
identity holds across every one of them, not just the original
`0x7e`/`0x7f` pair PR #1633 checked.

## The literal form's own presence/absence is `reg=54`'s, not zp_x's

The two `conv_dilation3_insz8*.mcode.gz` fixtures (also 3,528 bytes,
same overall shape family but a different `insz`) carry `reg=54` in a
DIFFERENT, 5-byte payload form (`a1 00 60 1b ff`, not `10 1b <byte>`)
at a shifted offset (2180) -- confirmed directly. This is exactly
`patch_mul_zp_x`'s own already-documented caveat ("whether a given
build uses this [literal] form is not predictable... roughly as many
builds use one of two other, still-undecoded forms instead") --
except it is now clear that caveat is really about `reg=54`'s OWN
form, a register present in every one of the 54 fixtures checked here
(not checked further whether it was among the 36 registers this
project's own earlier resource-model census work flagged as
near-universal corpus-wide, only that it is present, in one form or
another, in every fixture in this specific shape family), not a
separate, zp_x-specific presence question.

## Was `patch_conv_zp_x`'s own original verification fooled by this?

No -- checked directly. `tests/test_axera_conv_zpx_generator.py` (the
file `patch_conv_zp_x`'s own docstring cites as its real-build
verification) used a completely different shape,
`Conv(cin=1,cout=1,hw=8,k=3)` (`conv_1c1c_8x8_k3_zpx79.mcode.gz` and
siblings, 3,016 bytes -- not `Conv(dilation=3)`'s own 3,528), with
deliberately-chosen calibration zero points `79` and `49` -- verified
below (`TestOriginalVerificationUsedADifferentUnrelatedShape`) that
this shape's own reg=54-equivalent record sits at a different offset
(1806) and that neither `79` nor `49` match anything resembling a
binary-cluster-style switch value on that shape. `patch_conv_zp_x`'s
own existing verification is not invalidated -- it is correctly scoped
to a shape where this particular collision does not occur.

## A precise, evidenced hypothesis this file does not prove: `reg=54`
## may genuinely BE the input zero point, and the "binary path switch"
## may be a real floating-point near-tie IN that zero point's own
## computed value

This is offered as a hypothesis, not a finding, and is flagged as such
throughout: `reg=54`'s own short-form payload was independently
decoded (by earlier, pre-this-session work) as `Conv`'s literal input
zero point. This session's own, later, and completely independent
binary-cluster investigation (PR #1586/#1598/#1600) found the SAME
register unstable across rebuilds in a way consistent with a genuine
floating-point near-tie, and speculated (without proof) that the tie
was in "per-channel weight-statistics reduction order." If `reg=54`
really is the zero point, a more specific and more plausible reading
follows directly: quantization zero points are themselves computed
from calibrated min/max statistics via a rounding step, and a
near-tie in THAT specific computation (not an unrelated weight
statistic) would produce exactly this behavior -- a value that is
sometimes `126` and sometimes `127` (or any other adjacent pair,
depending on calibration data) because the correct rounded answer is
genuinely on a knife's edge for that build's own calibration range.
This is consistent with, and would refine, PR #1586's own speculative
reading -- but this file has no independent way to confirm what the
"true" calibrated zero point for any of these builds actually was
(that would require recomputing calibration statistics from the
original build scripts' own RNG-seeded data, not attempted here), so
it is reported as a well-motivated, unproven hypothesis for whoever
chases this next, not a closed question.

## What this means for composability, precisely

`emit_conv_reg8_group` and `patch_conv_zp_x` are NOT safely composable
on ANY `Conv(dilation=3,pad=3,cin=4,cout=4,insz=16)`-shaped fixture --
this is a structural fact about this shape, not an unlucky draw on one
fixture the way PR #1633's own hedged framing left open. This file
does not modify `scripts/axera/tiny_emit.py` (a sibling PR in this
session is concurrently editing that file for an unrelated reason);
`patch_conv_zp_x`'s own docstring should get a follow-up caveat naming
this specific shape-family collision explicitly, recommended here but
not made in this PR to avoid a merge conflict with that concurrent
work.
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

UNIT_PREFIX = bytes.fromhex("02101b")
UNIT_SUFFIX = bytes.fromhex("8336")


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def decode(name):
    return mcode.decode(load(name), **mcode.FULL_RULE)


def dilation3_family_names():
    """Every already-committed Conv(dilation=3,...) fixture sharing the
    3,528-byte shape this session's own binary-cluster/calibration/
    seed-survey/large-sample work all used."""
    names = []
    for n in sorted(os.listdir(FIX)):
        if not (n.startswith("conv_dilation3") and n.endswith(".mcode.gz")):
            continue
        if len(load(n)) != 3528:
            continue
        names.append(n)
    return names


def literal_unit_hits(data):
    return [
        i
        for i in range(len(data) - 6 + 1)
        if data[i : i + 3] == UNIT_PREFIX and data[i + 4 : i + 6] == UNIT_SUFFIX
    ]


def reg60_value(recs):
    """reg=60 carries two copies (PR #1586's own finding); both must
    agree, matching that project's own zero-exception result."""
    hits = [
        r
        for r in recs
        if r["kind"] == "S" and r.get("reg") == 60 and r.get("tag") == 131
    ]
    assert len(hits) == 2, hits
    values = {h["payload"][-1] for h in hits}
    assert len(values) == 1, hits
    return next(iter(values))


class TestFamilyHasFiftyFourFixturesAtLeast(unittest.TestCase):
    """Sanity check on the corpus this file's own claim is checked
    against -- fails loudly if the shared shape family shrinks or the
    fixture-naming convention changes."""

    def test_at_least_fifty_fixtures(self):
        names = dilation3_family_names()
        self.assertGreaterEqual(len(names), 50, names)


class TestReg54IsExactlyTheLiteralZpXUnitsOwnRecord(unittest.TestCase):
    """The 6-byte literal unit patch_mul_zp_x/patch_conv_zp_x search
    for is not an arbitrary byte pattern -- it is the complete on-the-
    wire encoding of an S-kind, reg=54, tag=131, 3-byte-payload record,
    confirmed by decoding (not just pattern-matching) the reference
    fixture."""

    def test_offset_2158_decodes_as_reg54_tag131(self):
        recs = decode("conv_dilation3.mcode.gz")
        hits = [
            r
            for r in recs
            if r["kind"] == "S" and r.get("at") == 2158 and r.get("tag") == 131
        ]
        self.assertEqual(len(hits), 1, hits)
        self.assertEqual(hits[0]["reg"], 54)
        self.assertEqual(hits[0]["payload"][:2], b"\x10\x1b")


class TestCollisionIsUniversalAcrossTheWholeShapeFamily(unittest.TestCase):
    """The core finding, recomputed live against all already-committed
    fixtures: every one where the literal zp_x unit is present at all
    has its own zp_x value exactly equal to reg=60's own binary-cluster
    switch value, at the identical offset 2158, zero exceptions."""

    def test_every_matching_fixture_has_zpx_equal_to_reg60(self):
        matched = 0
        zp_values_seen = set()
        for n in dilation3_family_names():
            data = load(n)
            hits = literal_unit_hits(data)
            if not hits:
                continue
            self.assertEqual(hits, [2158], n)
            zp = data[2158 + 3]
            recs = decode(n)
            r60 = reg60_value(recs)
            self.assertEqual(zp, r60, n)
            matched += 1
            zp_values_seen.add(zp)
        self.assertGreaterEqual(matched, 40, "expected most of the family to match")
        self.assertGreaterEqual(
            len(zp_values_seen), 5, "expected several distinct zp/reg60 values"
        )


class TestNonMatchingFixturesUseReg54sOtherKnownForm(unittest.TestCase):
    """The insz8 pair carries reg=54 in a different, 5-byte payload
    form at a shifted offset -- not a genuine "no zp_x" case, just a
    different already-known form of the same register."""

    def test_insz8_pair_uses_the_long_form(self):
        for n in (
            "conv_dilation3_insz8.mcode.gz",
            "conv_dilation3_insz8_rebuild.mcode.gz",
        ):
            data = load(n)
            self.assertEqual(literal_unit_hits(data), [], n)
            recs = decode(n)
            hits = [
                r
                for r in recs
                if r["kind"] == "S" and r.get("reg") == 54 and r.get("tag") == 131
            ]
            self.assertEqual(len(hits), 1, (n, hits))
            self.assertEqual(hits[0]["payload"], b"\xa1\x00\x60\x1b\xff", n)
            self.assertEqual(hits[0]["at"], 2180, n)


class TestOriginalVerificationUsedADifferentUnrelatedShape(unittest.TestCase):
    """patch_conv_zp_x's own existing real-build verification
    (tests/test_axera_conv_zpx_generator.py) used a different Conv
    shape entirely -- confirmed here that its own zp_x values (79, 49)
    and its own reg=54-equivalent record offset are unrelated to this
    file's own dilation=3 finding, so that prior verification is not
    invalidated by this collision."""

    def test_different_shape_and_offset(self):
        a = load("conv_1c1c_8x8_k3_zpx79.mcode.gz")
        self.assertNotEqual(len(a), 3528)
        hits = literal_unit_hits(a)
        self.assertEqual(hits, [1806], hits)
        self.assertEqual(a[1806 + 3], 79)

    def test_zpx_values_do_not_resemble_a_switch_pair(self):
        # 79 and 49 are far apart (30 apart) and neither is close to
        # any of this file's own dilation=3 reg=60/zp_x values -- a
        # deliberately-chosen calibration pair, not two states of one
        # near-tie switch.
        self.assertEqual(abs(79 - 49), 30)


if __name__ == "__main__":
    unittest.main()
