"""Answers `tests/test_axera_trivial_zp_replacement_mechanism_
investigation.py` (PR #1684)'s own explicitly flagged next step: does
its "the trivial-zero-point replacement is ordinary, already-pervasive
generic filler, not a dedicated wire form" finding -- confirmed there
for Conv's own `reg=54` (literal-quad-style mechanism) and Add's own
`x2` (`reg=94,tag=132`, decode-record-style mechanism) -- also hold for
MatMul's own always-trivial case, and for one of the OTHER two-live-
input ops' (Sub/Mul/Div) own `x2` mechanism?

## Finding: yes, cleanly, for both -- confirming the "generic filler
## reuse" picture is universal across this whole arc, not specific to
## the two cases PR #1684 happened to check first

**MatMul's own case**: MatMul's own zero point is unconditionally
forced to `0` (`tests/test_axera_zpx_generalizes.py`, PR #1497), so
EVERY already-committed MatMul fixture already is the "trivial" case
-- no new build needed, only a fresh decode of one already-committed
fixture (`matmul_4x8x8_asym_calib.mcode.gz`, PR #1497's own
"genuinely asymmetric calibration data" stress-test fixture). The
literal-quad prefix (`02 10 1b`) has zero hits, as already established
(PR #1679); the `00 10 84` tail has 6 hits. Decoding every one of
those 6 hits directly under `mcode.decode(..., FULL_RULE)` shows each
is a genuine, ordinary `p=0,tag=132(0x84),payload=[0x10]` S-unit, at
six DIFFERENT, unremarkable registers (14, 148, 74, 202, 36, 36) --
exactly the same generic-filler shape PR #1684 already found
pervasively in Conv's own control fixture (`tag=132` there: 151 of 423
S/B records, 35.7%). This fixture's own `tag=132` count (122 of 309 S
records, 39.5%) confirms `tag=132` is similarly the single most common
short-unit tag here too -- MatMul's own always-trivial `zp_x` is filled
by the exact same kind of pre-existing, pervasive generic filler unit
Conv's own conditionally-trivial `zp_x` is, not a MatMul-specific or
"always-trivial-specific" dedicated form.

**Sub's own `x2` case**: decoding `sub_1x16_two_live_seed1_2_
trivialx2.mcode.gz` against its own non-degenerate control
(`sub_1x16_two_live_seed1_2.mcode.gz`) the same way PR #1684 diffed
Add's pair finds an EXACT match to Add's own mechanism, not merely a
similar one: the control's own `reg=94,tag=132` record (`x2`'s real
zero point) is absent in the trivial fixture; in its place, at the
IDENTICAL byte offset (1248), is the identical new unit PR #1684 found
for Add -- `p=0,tag=133(0x85),reg=98,payload=[0x10]` -- and the
neighboring `reg=94,tag=130` record widens by exactly one byte
(`p=2->p=3`) with the identical leading byte (`0x0f`) and identical
constant trailing bytes (`88 c1 bd`) PR #1684 found for Add. This is
not merely the same MECHANISM -- the specific bytes are identical,
because `x2`'s own calibration data (`RandomState(2)`) is the same
regardless of which op consumes it, and neither `reg=98` (`tag=133`
count in the control: 2 of 224 S records, an ordinary rare-but-
pre-existing tag) is introduced as anything special.

## What this establishes, precisely, and what it does not

**Established**: PR #1684's own "generic filler reuse, not a dedicated
wire form" finding generalizes cleanly to MatMul's own always-trivial
case and to Sub's own `x2` mechanism -- in Sub's case, not merely the
same qualitative mechanism but byte-identical replacement content to
Add's own case, confirming the replacement is driven purely by `x2`'s
own shared calibration data, independent of which op consumes it.

**NOT established**: whether Mul's or Div's own `x2` mechanism shows
the identical byte-for-byte match (plausible, given Sub's own exact
match and the shared calibration-data mechanism, but not independently
verified for those two ops here); whether this pattern holds at shapes
other than `(1,16)`/MatMul's own tested shape; why the allocator
specifically reuses `tag=132`/`tag=133` rather than some other
pre-existing tag (no access to Pulsar2's own source, the same open
item PR #1684 already left).
"""

import gzip
import os
import sys
import unittest
from collections import Counter

_AXERA_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts", "axera")
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import mcode  # noqa: E402

FIX = os.path.join(_AXERA_DIR, "fixtures")


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def decode(name):
    return mcode.decode(load(name), **mcode.FULL_RULE)


def hits(data, pat):
    return [i for i in range(len(data) - len(pat) + 1) if data[i : i + len(pat)] == pat]


LITERAL_QUAD_PREFIX = bytes.fromhex("02101b")
LITERAL_QUAD_REPLACEMENT_TAIL = bytes.fromhex("001084")


class TestMatMulsAlwaysTrivialZpXIsFilledByOrdinaryTag132Filler(unittest.TestCase):
    """MatMul's own zero point is unconditionally 0 (PR #1497), so
    every already-committed fixture is already the trivial case --
    confirms the literal-quad locator never fires and the `00 10 84`
    replacement tail resolves to ordinary, pervasive `tag=132` S-units
    at unremarkable registers, matching Conv's own pattern (PR #1684)."""

    def test_literal_quad_prefix_never_fires(self):
        data = load("matmul_4x8x8_asym_calib.mcode.gz")
        self.assertEqual(hits(data, LITERAL_QUAD_PREFIX), [])

    def test_00_10_84_tail_resolves_to_ordinary_tag132_s_units(self):
        data = load("matmul_4x8x8_asym_calib.mcode.gz")
        recs = decode("matmul_4x8x8_asym_calib.mcode.gz")
        tail_offsets = hits(data, LITERAL_QUAD_REPLACEMENT_TAIL)
        self.assertGreater(len(tail_offsets), 0)
        by_at = {r["at"]: r for r in recs if r.get("kind") == "S"}
        for off in tail_offsets:
            rec = by_at.get(off)
            self.assertIsNotNone(rec, off)
            self.assertEqual(rec["tag"], 132, (off, rec))
            self.assertEqual(rec["p"], 0, (off, rec))
            self.assertEqual(rec["payload"], b"\x10", (off, rec))

    def test_tag132_is_the_single_most_common_short_unit_tag(self):
        recs = decode("matmul_4x8x8_asym_calib.mcode.gz")
        tag_counts = Counter(r.get("tag") for r in recs if r.get("kind") == "S")
        most_common_tag, most_common_count = tag_counts.most_common(1)[0]
        self.assertEqual(most_common_tag, 132)
        total = sum(tag_counts.values())
        self.assertGreater(most_common_count / total, 0.35)


class TestSubsOwnX2ReplacementIsByteIdenticalToAdds(unittest.TestCase):
    """Sub's own trivial-`x2` fixture (PR #1669) is diffed against its
    own non-degenerate control (PR #1661) the same way PR #1684 diffed
    Add's pair -- finding not merely the same mechanism but the exact
    same replacement bytes, since `x2`'s own calibration data is
    shared regardless of op."""

    CONTROL = "sub_1x16_two_live_seed1_2.mcode.gz"
    TRIVIAL = "sub_1x16_two_live_seed1_2_trivialx2.mcode.gz"

    def test_reg94_tag132_x2_zero_point_present_in_control_absent_in_trivial(self):
        ctrl = decode(self.CONTROL)
        triv = decode(self.TRIVIAL)
        ctrl_hits = [r for r in ctrl if r.get("reg") == 94 and r.get("tag") == 132]
        triv_hits = [r for r in triv if r.get("reg") == 94 and r.get("tag") == 132]
        self.assertEqual(len(ctrl_hits), 1)
        self.assertEqual(triv_hits, [])

    def test_new_reg98_tag133_unit_matches_adds_own_exactly(self):
        triv = decode(self.TRIVIAL)
        hits_ = [r for r in triv if r.get("reg") == 98 and r.get("tag") == 133]
        self.assertEqual(len(hits_), 1)
        rec = hits_[0]
        self.assertEqual(rec["p"], 0)
        self.assertEqual(rec["payload"], b"\x10")

    def test_reg98_is_absent_from_the_control_reg94_tag132_is_present(self):
        ctrl = decode(self.CONTROL)
        self.assertEqual([r for r in ctrl if r.get("reg") == 98], [])

    def test_neighboring_reg94_tag130_record_widens_by_one_byte_matching_add(self):
        ctrl = decode(self.CONTROL)
        triv = decode(self.TRIVIAL)
        ctrl_hits = [r for r in ctrl if r.get("reg") == 94 and r.get("tag") == 130]
        triv_hits = [r for r in triv if r.get("reg") == 94 and r.get("tag") == 130]
        self.assertEqual(len(ctrl_hits), 4)
        self.assertEqual(len(triv_hits), 4)
        # All four control copies share one constant payload.
        ctrl_payloads = {r["payload"] for r in ctrl_hits}
        self.assertEqual(len(ctrl_payloads), 1)
        # The first trivial copy is one byte wider (p=3 vs p=2) with a
        # leading 0x0f byte; the trailing 3 bytes match Add's own
        # already-established constant (88 c1 bd) exactly.
        first = sorted(triv_hits, key=lambda r: r["at"])[0]
        self.assertEqual(first["p"], 3)
        self.assertEqual(first["payload"], bytes.fromhex("0f88c1bd"))
        rest = sorted(triv_hits, key=lambda r: r["at"])[1:]
        for r in rest:
            self.assertEqual(r["p"], 2)
            self.assertEqual(r["payload"], bytes.fromhex("88c1bd"))


class TestFixturesDecodeCleanly(unittest.TestCase):
    def test_no_decode_errors(self):
        for name in (
            "matmul_4x8x8_asym_calib.mcode.gz",
            "sub_1x16_two_live_seed1_2.mcode.gz",
            "sub_1x16_two_live_seed1_2_trivialx2.mcode.gz",
        ):
            errs = mcode.check(load(name))
            self.assertEqual(errs, [], (name, errs))


if __name__ == "__main__":
    unittest.main()
