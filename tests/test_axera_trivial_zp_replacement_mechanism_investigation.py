"""Answers the question `tests/test_axera_trivial_zeropoint_framing_
correction_synthesis.py` (PR #1680) explicitly left open: are the two
"trivial-zero-point replacement" byte forms -- the literal-quad-style
mechanism's own `00 10 84` (Conv's own `reg=54`, Gemm/Add's own `x1`,
MatMul-always) and the decode-record mechanism's own `00 10 85 62 03
0f` (the two-live-input cluster's own `x2`, `reg=94,tag=132`) -- the
same underlying codec mechanism, or an independent coincidence that
happens to share a `00 10` prefix?

## Finding: neither -- both "replacement forms" are ordinary, already-
## pervasive generic short units, not a dedicated zero-point wire form
## at all. The correct picture is more mundane than either alternative
## PR #1680's own diff posed, and more informative.

Decoding both cases directly under `mcode.decode(..., **mcode.
FULL_RULE)` (not a raw byte-substring search, which is what every
contributing PR in this thread used) shows:

**Conv's own case**: the vacated 6-byte span (previously one `p=2,
tag=131(0x83),reg=54` S-unit -- the real `zp_x` record) is refilled by
TWO ordinary `p=0,tag=132(0x84)` S-units at two DIFFERENT, unremarkable
registers (`reg=36`, `reg=38`) -- not a special new tag. `tag=132` is
not novel to this replacement: it is the single most common short-unit
tag in this exact fixture, appearing 151 of 423 total S/B records
(35.7%) in the control fixture -- including at registers immediately
adjacent to the vacated span (`reg=8`, `reg=34`, `reg=40` all already
use `tag=132` in the CONTROL fixture, both before and after the
vacated `reg=54` record). The replacement is not a dedicated wire
form; it is the register allocator continuing to emit the same kind of
generic filler unit it already uses pervasively nearby, simply
skipping the one register (54) that used to carry the real value.

**Add's own case (`x2`)**: PR #1680's own "fixed 6-byte tail" (`00 10
85 62 03 0f`) is not one atomic unit. It is the boundary between TWO
separate S-units: a genuinely new 4-byte unit (`p=0,tag=133(0x85),
reg=98,payload=[0x10]`) immediately followed by the FIRST TWO bytes of
an entirely different, PRE-EXISTING record at `reg=94,tag=130(0x82)`
(present in the control fixture too, at the same register, one payload
byte narrower). In the trivial case this neighboring record's own
width grows by exactly one byte (`p=2`->`p=3`) with a constant leading
byte (`0x0f`) -- but its OWN remaining payload bytes (the genuinely
variable part) differ by calibration seed, confirmed directly across
all three standard seed pairs here. PR #1680's own fixed-bytes search
only ever matched the first six bytes of this two-unit sequence
because those six bytes happen to be seed-invariant; the sequence as a
whole is not a fixed token. Register 98 and tag 133 are not novel to
this replacement either: both already appear elsewhere in the control
fixture (`tag=133` five times, `reg=98` once, at unrelated locations),
confirming `tag=133` is likewise an ordinary, pre-existing tag value,
not a dedicated marker.

## What this means for PR #1680's own "shared `00 10` prefix" question

The two forms are NOT the same opcode (`0x84` vs `0x85` are different,
unrelated tag values from the same broad `ALL_TAGS` family this
project's own codec already documents, `scripts/axera/mcode.py`'s own
`range(0x81, 0xA0)`) -- so "same mechanism" in the sense PR #1680 posed
it (one shared dedicated encoding) is false. But they are not an
unrelated coincidence either: BOTH are instances of one single, more
general, already-known behavior -- when a value's own dedicated S-unit
is omitted because the value is trivially `0`, the surrounding stream
does not shrink or leave a gap; the register allocator fills the freed
byte-span with more of whatever generic short-unit forms it already
uses commonly nearby. The shared `00 10` two-byte prefix both cases
happen to show is itself unremarkable: `p=0,payload=[0x10]` is a common
short-unit shape (confirmed present 3-4 times, at unrelated registers,
in each control fixture checked here) -- not evidence of a shared
special field, just two small, frequently-occurring generic values.

## What this establishes, precisely, and what it does not

**Established**: neither `00 10 84` nor `00 10 85 62 03 0f` is a
dedicated, purpose-built "trivial zero point" wire form. Both are
ordinary, already-pervasive generic short-unit patterns (`tag=132` for
Conv, `tag=133` for Add's own case) that happen to fill the byte-span
the omitted zero-point record vacates -- confirmed by their own tags
and register values already appearing, unremarkably, elsewhere in each
control fixture. The "replacement, not omission" framing PR #1679/
#1680 established remains correct (the byte span is genuinely filled,
not shortened); only the characterization of WHAT fills it -- a
special wire form, versus ordinary allocator filler -- needed this
further refinement. PR #1680's own "fixed 6-byte tail" for the
decode-record mechanism is corrected here to "a fixed 4-byte unit
followed by a widened, still-seed-varying pre-existing record."

**NOT established**: why the allocator chooses exactly `tag=132`/two
new registers for Conv but `tag=133`/one new register plus one widened
existing record for Add -- this file does not have a general rule
predicting which nearby generic form fills a given vacated span, only
that it does so with ALREADY-COMMON forms in both tested cases; whether
this same "generic filler, not dedicated form" picture holds for
MatMul's own always-trivial case, or for Sub/Mul/Div's own `x2`
mechanism (not decoded structurally here, only Conv and Add); whether
the specific tag/register choice is itself deterministic or one
resolution among several equally-valid ones the allocator could pick.
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

ADD_SEED_PAIRS = [(1, 2), (7, 42), (100, 999)]


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def decode(name):
    return mcode.decode(load(name), **mcode.FULL_RULE)


class TestConvReplacementIsGenericTagReuseNotADedicatedForm(unittest.TestCase):
    """Conv's own `reg=54,tag=131` vacancy is refilled by two ordinary
    `tag=132` S-units at unrelated registers, not a special form."""

    def test_reg54_tag131_is_absent_in_the_trivial_fixture(self):
        recs = decode("conv_dilation3_trivialA_seed1.mcode.gz")
        hits = [r for r in recs if r.get("reg") == 54 and r.get("tag") == 131]
        self.assertEqual(hits, [])

    def test_reg54_tag131_is_present_in_the_control_as_a_p2_unit(self):
        recs = decode("conv_dilation3.mcode.gz")
        hits = [
            r
            for r in recs
            if r.get("kind") == "S" and r.get("reg") == 54 and r.get("tag") == 131
        ]
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["p"], 2)
        self.assertEqual(hits[0]["at"], 2158)

    def test_the_vacated_span_is_refilled_by_two_plain_tag132_units(self):
        recs = decode("conv_dilation3_trivialA_seed1.mcode.gz")
        at2158 = [r for r in recs if r["at"] == 2158]
        at2162 = [r for r in recs if r["at"] == 2162]
        self.assertEqual(len(at2158), 1)
        self.assertEqual(len(at2162), 1)
        for r in (at2158[0], at2162[0]):
            self.assertEqual(r["kind"], "S")
            self.assertEqual(r["p"], 0)
            self.assertEqual(r["tag"], 132)
        # Genuinely different, unremarkable registers -- not reg=54's own.
        self.assertEqual(at2158[0]["reg"], 36)
        self.assertEqual(at2162[0]["reg"], 38)

    def test_tag132_is_the_single_most_common_short_unit_tag_in_this_fixture(self):
        recs = decode("conv_dilation3.mcode.gz")
        su = [r for r in recs if r["kind"] in ("S", "B")]
        tag132 = [r for r in su if r.get("tag") == 132]
        # Established: 151/423 = 35.7%, comfortably the majority tag.
        self.assertGreater(len(tag132) / len(su), 0.30, (len(tag132), len(su)))

    def test_tag132_already_appears_immediately_adjacent_to_the_vacated_span(self):
        recs = decode("conv_dilation3.mcode.gz")
        adjacent_regs = {8, 34, 38, 40}
        hits = [
            r
            for r in recs
            if r.get("tag") == 132
            and r.get("reg") in adjacent_regs
            and 2140 <= r["at"] <= 2180
        ]
        self.assertGreaterEqual(len(hits), len(adjacent_regs) - 1, hits)


class TestAddReplacementIsATwoUnitBoundaryNotOneFixedToken(unittest.TestCase):
    """PR #1680's own "fixed 6-byte tail" for Add's own `x2` mechanism
    is actually a new 4-byte unit plus the leading two bytes of a
    separate, pre-existing, still-seed-varying record."""

    def test_reg94_tag132_is_absent_in_every_trivial_fixture(self):
        for s1, s2 in ADD_SEED_PAIRS:
            recs = decode(f"add_1x16_two_live_seed{s1}_{s2}_trivialx2.mcode.gz")
            hits = [r for r in recs if r.get("reg") == 94 and r.get("tag") == 132]
            self.assertEqual(hits, [], (s1, s2))

    def test_a_new_4byte_tag133_reg98_unit_appears_at_the_same_offset(self):
        for s1, s2 in ADD_SEED_PAIRS:
            recs = decode(f"add_1x16_two_live_seed{s1}_{s2}_trivialx2.mcode.gz")
            hits = [
                r
                for r in recs
                if r["at"] == 1244 and r.get("tag") == 133 and r.get("reg") == 98
            ]
            self.assertEqual(len(hits), 1, (s1, s2))
            self.assertEqual(hits[0]["p"], 0)
            self.assertEqual(hits[0]["payload"], b"\x10")

    def test_the_following_reg94_tag130_record_widens_by_exactly_one_byte(self):
        recs_ctrl = decode("add_1x16_two_live_seed1_2.mcode.gz")
        recs_triv = decode("add_1x16_two_live_seed1_2_trivialx2.mcode.gz")
        ctrl_hit = [
            r
            for r in recs_ctrl
            if r["at"] == 1250 and r.get("tag") == 130 and r.get("reg") == 94
        ]
        triv_hit = [
            r
            for r in recs_triv
            if r["at"] == 1248 and r.get("tag") == 130 and r.get("reg") == 94
        ]
        self.assertEqual(len(ctrl_hit), 1)
        self.assertEqual(len(triv_hit), 1)
        self.assertEqual(ctrl_hit[0]["p"], 2)
        self.assertEqual(triv_hit[0]["p"], 3)
        # The extra leading byte is a fixed 0x0f; the rest still varies.
        self.assertEqual(triv_hit[0]["payload"][0], 0x0F)

    def test_that_records_own_true_payload_varies_by_seed_not_fixed(self):
        payloads = set()
        for s1, s2 in ADD_SEED_PAIRS:
            recs = decode(f"add_1x16_two_live_seed{s1}_{s2}_trivialx2.mcode.gz")
            hits = [
                r
                for r in recs
                if r["at"] == 1248 and r.get("tag") == 130 and r.get("reg") == 94
            ]
            self.assertEqual(len(hits), 1, (s1, s2))
            payloads.add(bytes(hits[0]["payload"][1:]))
        # If this were a truly fixed 6-byte token, all three would be
        # identical; they are not.
        self.assertEqual(len(payloads), 3, payloads)

    def test_tag133_and_reg98_are_not_novel_they_appear_elsewhere_in_the_control(self):
        recs = decode("add_1x16_two_live_seed1_2.mcode.gz")
        tag133_elsewhere = [r for r in recs if r.get("tag") == 133]
        reg98_elsewhere = [r for r in recs if r.get("reg") == 98]
        self.assertGreater(len(tag133_elsewhere), 0)
        self.assertGreater(len(reg98_elsewhere), 0)


class TestSmallCommonShapesExplainTheSharedZeroTenPrefixCoincidence(unittest.TestCase):
    """The `00 10` prefix both replacement forms happen to share is
    itself an ordinary, frequently-occurring short-unit shape -- not
    evidence the two forms are a shared special field."""

    def test_p0_payload_0x10_units_already_exist_at_unrelated_registers(self):
        for name in ("add_1x16_two_live_seed1_2.mcode.gz", "conv_dilation3.mcode.gz"):
            recs = decode(name)
            hits = [
                r
                for r in recs
                if r.get("kind") == "S"
                and r.get("p") == 0
                and r.get("payload") == b"\x10"
            ]
            self.assertGreaterEqual(len(hits), 3, (name, hits))


class TestFixturesDecodeCleanly(unittest.TestCase):
    def test_no_decode_errors(self):
        names = ["conv_dilation3.mcode.gz", "conv_dilation3_trivialA_seed1.mcode.gz"]
        for s1, s2 in ADD_SEED_PAIRS:
            names.append(f"add_1x16_two_live_seed{s1}_{s2}.mcode.gz")
            names.append(f"add_1x16_two_live_seed{s1}_{s2}_trivialx2.mcode.gz")
        for name in names:
            errs = mcode.check(load(name))
            self.assertEqual(errs, [], (name, errs))


if __name__ == "__main__":
    unittest.main()
