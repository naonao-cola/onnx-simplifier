"""Closes the last gap `tests/test_axera_conv_matmul_trivial_zeropoint.py`
(PR #1679) explicitly left open: PR #1679 corrected PRs #1663/#1667/
#1669/#1670/#1678's own framing of the "trivially-zero calibrated zero
point" phenomenon -- showing, for the literal-quad-style locators
(Gemm's/Add's own `x1`, Conv's own `reg=54`), that the value isn't
*unencoded* when the searched-for locator has zero hits; it is
*replaced*, at the identical byte offset, by a different, already-
documented wire form (`tests/test_axera_mcode_reciprocal.py`'s own
`TestZpXImmediateRegion` class: "zp_x=0: a fixed 3-byte tail
`00 10 84`"). PR #1679 only checked the literal-quad-style mechanism --
it explicitly left open whether the two-live-input cluster's OWN
`x2`-side mechanism (`reg=94,tag=132`, a decode record, architecturally
different from a raw literal-quad byte pattern) resolves the same way.

## Finding: yes -- `x2`'s own absence is ALSO a replacement, not an
## omission, via a fixed 6-byte tail (`00 10 85 62 03 0f`) -- distinct
## from the `00 10 84` tail the literal-quad locators use, but the
## identical *shape* of correction

Directly re-decoded every already-committed `reg=94,tag=132`-trivial
fixture across all FOUR two-live-input ops (`Add`/`Sub`/`Mul`/`Div`,
PRs #1663/#1667/#1669, 3 calibration seed pairs each -- 12 fixtures
total) and searched for a replacement byte sequence at the control
fixture's own known `reg=94,tag=132` offset. Every one of the 12
trivial fixtures contains the SAME fixed 6-byte pattern
(`00 10 85 62 03 0f`) exactly once, at (or immediately at) the byte
offset the control fixture's own `reg=94,tag=132` record occupies;
none of the 12 non-degenerate control fixtures ever contain it. This
is a real, deterministic, op-generic replacement -- not the SAME wire
form the literal-quad locators use (`00 10 84`, a 3-byte tail), but a
different, longer, equally fixed tail, confirming the same underlying
"replacement, not omission" story PR #1679 established for the
literal-quad mechanism now also holds for this architecturally
distinct decode-record mechanism.

Four of the twelve fixtures additionally reflow their own total stream
length when `x2` becomes trivial: `Mul` does so at all three tested
seed pairs (always exactly 32 bytes shorter, `2264`->`2232`); `Add`
does so only at seed pair `(100,999)` (`3104`->`3072`); `Sub` and `Div`
never do, at any of the three tested seed pairs. This is not a clean
per-op split -- `Add` sometimes reflows and sometimes does not,
depending on the seed pair -- so this file reports it precisely rather
than rounding it into a false "op X always reflows" summary (see
`TestStreamLengthChangesAreOpAndSeedSpecific` below). The replacement
pattern was located via a raw substring SEARCH across the whole
fixture in every case, not a fixed-offset assumption, so this
stream-length variation does not silently break the check; its own
located offset is additionally confirmed to always land within 4 bytes
of the control fixture's own `reg=94,tag=132` offset (see
`TestReplacementLandsNearTheControlsOwnReg94Tag132Offset` below), never
exactly at a fixed distance and never far from it.

## What this establishes, precisely, and what it does not

**Established**: the "`x2`'s own `reg=94,tag=132` locator goes
structurally absent" finding from PRs #1663/#1667/#1669 (the two-live-
input cluster's own trivial-zero-point discovery) is, like the
literal-quad case PR #1679 already corrected, a REPLACEMENT -- the
value is still encoded, via a fixed 6-byte tail (`00 10 85 62 03 0f`)
that appears in every one of the 12 tested trivial-`x2` fixtures across
all four ops and never in any of their own non-degenerate controls.
This closes PR #1679's own explicitly flagged open question: the
correction generalizes to the decode-record-style mechanism too, not
only the literal-quad-style one, though the specific replacement bytes
differ between the two mechanisms (`00 10 84` for the literal-quad
case, `00 10 85 62 03 0f` here).

**NOT established**: what the trailing 4 bytes (`62 03 0f`, after the
shared `00 10` prefix) actually encode -- this file only confirms the
sequence is fixed and op-generic, the same "raw bytes pinned without a
decode" scope `TestZpXImmediateRegion` itself already used for its own
still-undecoded forms; whether this is literally the SAME underlying
codec mechanism as the literal-quad case's own `00 10 84` tail (both
share a `00 10` prefix, suggestive but not confirmed to be the same
underlying field/opcode) or a coincidentally similar but distinct one;
why `Mul`'s own trivial-`x2` builds consistently reflow their whole
stream by exactly 32 bytes while `Add`'s/`Div`'s do not and `Sub`'s
does so only at one of three tested seed pairs (not investigated here).
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

REPLACEMENT_TAIL = bytes.fromhex("00108562030f")

SEED_PAIRS = [(1, 2), (7, 42), (100, 999)]

# (op, trivial_fixture_name_template)
OPS = {
    "add": "add_1x16_two_live_seed{s1}_{s2}_trivialx2.mcode.gz",
    "sub": "sub_1x16_two_live_seed{s1}_{s2}_trivialx2.mcode.gz",
    "mul": "mul_1x16_two_live_seed{s1}_{s2}_trivialx2.mcode.gz",
    "div": "div_1x16_two_live_seed{s1}_{s2}_trivialzp2.mcode.gz",
}
CONTROL_TEMPLATE = "{op}_1x16_two_live_seed{s1}_{s2}.mcode.gz"


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def hits(data, pat):
    return [i for i in range(len(data) - len(pat) + 1) if data[i : i + len(pat)] == pat]


class TestFixturesDecodeCleanly(unittest.TestCase):
    def test_no_decode_errors_on_any_trivial_x2_fixture(self):
        for op, template in OPS.items():
            for s1, s2 in SEED_PAIRS:
                data = load(template.format(s1=s1, s2=s2))
                errs = mcode.check(data)
                self.assertEqual(errs, [], (op, s1, s2, errs))


class TestReplacementTailAppearsInEveryTrivialX2Fixture(unittest.TestCase):
    """The headline correction: `reg=94,tag=132`'s own absence is a
    replacement, not an omission, for all four two-live-input ops."""

    def test_exactly_one_hit_in_every_trivial_fixture(self):
        for op, template in OPS.items():
            for s1, s2 in SEED_PAIRS:
                data = load(template.format(s1=s1, s2=s2))
                found = hits(data, REPLACEMENT_TAIL)
                self.assertEqual(len(found), 1, (op, s1, s2, found))

    def test_zero_hits_in_every_non_degenerate_control_fixture(self):
        for op in OPS:
            for s1, s2 in SEED_PAIRS:
                data = load(CONTROL_TEMPLATE.format(op=op, s1=s1, s2=s2))
                found = hits(data, REPLACEMENT_TAIL)
                self.assertEqual(found, [], (op, s1, s2, found))


class TestReplacementLandsNearTheControlsOwnReg94Tag132Offset(unittest.TestCase):
    """Confirms the replacement isn't just present SOMEWHERE in the
    stream by coincidence -- it lands within a small (<=4-byte) window
    of the exact byte offset the control fixture's own `reg=94,tag=132`
    decode record occupies, in every one of the 12 tested (op, seed
    pair) combinations. The offset is not always byte-for-byte
    identical (some builds shift it by exactly 4 bytes, independent of
    whether the stream's own total length also changed -- see
    `TestStreamLengthChangesAreOpAndSeedSpecific` below, which shows
    the length change and the offset shift are NOT the same
    phenomenon: `sub(100,999)` shifts the offset by 4 bytes with an
    UNCHANGED total length), but it is never far from it, ruling out a
    coincidental match to some unrelated part of the stream."""

    def test_offset_within_four_bytes_of_control_reg94_offset(self):
        for op, template in OPS.items():
            for s1, s2 in SEED_PAIRS:
                control = load(CONTROL_TEMPLATE.format(op=op, s1=s1, s2=s2))
                trivial = load(template.format(s1=s1, s2=s2))
                recs = mcode.decode(control, **mcode.FULL_RULE)
                control_hits = [
                    r
                    for r in recs
                    if r.get("kind") == "S"
                    and r.get("reg") == 94
                    and r.get("tag") == 132
                ]
                self.assertEqual(len(control_hits), 1, (op, s1, s2))
                control_at = control_hits[0]["at"]
                trivial_at = hits(trivial, REPLACEMENT_TAIL)[0]
                self.assertLessEqual(
                    abs(trivial_at - control_at),
                    4,
                    (op, s1, s2, control_at, trivial_at),
                )


class TestStreamLengthChangesAreOpAndSeedSpecific(unittest.TestCase):
    """Documents, as real assertions rather than only prose, exactly
    which (op, seed pair) combinations reflow their own total stream
    length when `x2` becomes trivial: `Mul` always does (32 bytes
    shorter, every seed pair); `Add` does only at `(100,999)`; `Sub`
    and `Div` never do (at any of the 3 tested seed pairs). This is
    NOT a clean per-op split -- `Add` is the one case that sometimes
    reflows and sometimes does not, depending on the seed pair -- and
    is reported precisely rather than rounded into a false "op X always
    reflows, op Y never does" summary."""

    def test_mul_always_reflows_by_32_bytes(self):
        for s1, s2 in SEED_PAIRS:
            control = load(CONTROL_TEMPLATE.format(op="mul", s1=s1, s2=s2))
            trivial = load(OPS["mul"].format(s1=s1, s2=s2))
            self.assertEqual(len(control) - len(trivial), 32, (s1, s2))

    def test_sub_and_div_never_reflow(self):
        for op in ("sub", "div"):
            for s1, s2 in SEED_PAIRS:
                control = load(CONTROL_TEMPLATE.format(op=op, s1=s1, s2=s2))
                trivial = load(OPS[op].format(s1=s1, s2=s2))
                self.assertEqual(len(control), len(trivial), (op, s1, s2))

    def test_add_reflows_only_at_the_100_999_seed_pair(self):
        expected_diff = {(1, 2): 0, (7, 42): 0, (100, 999): 32}
        for s1, s2 in SEED_PAIRS:
            control = load(CONTROL_TEMPLATE.format(op="add", s1=s1, s2=s2))
            trivial = load(OPS["add"].format(s1=s1, s2=s2))
            self.assertEqual(
                len(control) - len(trivial), expected_diff[(s1, s2)], (s1, s2)
            )


if __name__ == "__main__":
    unittest.main()
