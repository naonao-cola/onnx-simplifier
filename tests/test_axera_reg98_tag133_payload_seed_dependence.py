"""Corrects a naming/framing imprecision in `tests/test_axera_mul_div_
x2_replacement_mechanism.py` (PR #1687): that file named the trailing
3 bytes of the widened `reg=94,tag=130` neighbor record
`CONSTANT_LEADING_PAYLOAD`/`CONSTANT_TRAILING_PAYLOAD` and asserted
them equal across Add/Sub/Mul/Div -- true, but only because every one
of those cross-op comparisons happened to use the SAME calibration
seed pair, `(1,2)`. This file checks the same bytes across MULTIPLE
seed pairs for the same op and finds they are NOT constant -- they are
genuinely seed-dependent, exactly as PR #1684's own original Add-only
analysis already, correctly, said ("its own remaining payload bytes...
differ by calibration seed, confirmed directly across all three
standard seed pairs here") before PR #1687's own cross-op file lost
that nuance in its own variable naming.

## Finding: `reg=98,tag=133`'s own payload IS a genuine seed-invariant
## constant (strengthening the "generic filler" story); the
## neighboring `reg=94,tag=130` record's own trailing bytes are NOT --
## they are real, seed-varying data that merely happens to match
## across ops at any one FIXED seed pair, because `x2`'s own
## calibration data is shared regardless of op

Decoding Add's own three already-committed trivial-`x2` fixtures
(seed pairs `(1,2)`, `(7,42)`, `(100,999)`, PRs #1667/#1670's own
fixture set):

| seed pair | `reg=98,tag=133` payload | `reg=94,tag=130`'s first copy payload |
| --- | --- | --- |
| `(1,2)` | `10` (fixed) | `0f 88 c1 bd` |
| `(7,42)` | `10` (fixed) | `0f 23 8e c7` |
| `(100,999)` | `10` (fixed) | `0f fb 40 c1` |

`reg=98,tag=133`'s own single payload byte (`0x10`) is IDENTICAL
across all three seed pairs -- a genuine constant, consistent with PR
#1684's own "ordinary generic filler" characterization for this
specific new unit. But `reg=94,tag=130`'s own trailing 3 bytes
(`88 c1 bd` / `23 8e c7` / `fb 40 c1`) are three DIFFERENT values, one
per seed pair -- only the single leading byte (`0x0f`) is actually
constant. Cross-checking Sub's own trivial fixtures at the SAME three
seed pairs finds the IDENTICAL trailing bytes at each corresponding
seed pair (`23 8e c7` at `(7,42)`, `fb 40 c1` at `(100,999)`) -- this
is why PR #1687's own cross-op comparison (Add vs. Mul vs. Sub vs.
Div, all at seed pair `(1,2)` only) found a match: it's the SAME
seed's own shared `x2` calibration data being reflected, not a
universal constant.

## What this corrects, precisely, and what it does not

**Corrected**: PR #1687's own `CONSTANT_LEADING_PAYLOAD`/
`CONSTANT_TRAILING_PAYLOAD` variable names are misleading for the
trailing 3 bytes specifically -- those bytes are real, per-build data
that varies with `x2`'s own calibration seed, not a fixed constant.
Only the single leading byte (`0x0f`) and `reg=98,tag=133`'s own
payload (`0x10`) are genuinely seed-invariant. PR #1687's own actual
TEST ASSERTIONS remain correct and are not invalidated by this
(they only ever compared ops at one matched seed pair, which is a
valid comparison), but the "constant" framing needed this correction.

**NOT established**: what the `reg=94,tag=130` record's own
seed-varying trailing bytes actually encode (plausibly a per-build
quantity of `x2`'s own -- its own real, non-degenerate scale computed
from the SAME all-positive calibration data that also determines its
own trivially-zero zero point -- but this file does not attempt a
first-principles recomputation to confirm that guess, the same
"pinned as raw bytes, not decoded" scope PR #1684's own
`TestZpXImmediateRegion`-citing analysis already used for its own
still-undecoded forms); whether Mul's/Div's own trailing bytes also
match Add's/Sub's at the two NEW seed pairs checked here (not
re-verified for those two ops, though PR #1687 already established
the mechanism is shared across all four ops at seed pair `(1,2)`).
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

SEED_PAIRS = [(1, 2), (7, 42), (100, 999)]

# reg=94,tag=130's own first-copy trailing 3 bytes (after the constant
# leading 0x0f), per seed pair -- independently re-decoded here, not
# cited from any prior PR's own table.
EXPECTED_TRAILING_BYTES = {
    (1, 2): bytes.fromhex("88c1bd"),
    (7, 42): bytes.fromhex("238ec7"),
    (100, 999): bytes.fromhex("fb40c1"),
}


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def decode(name):
    return mcode.decode(load(name), **mcode.FULL_RULE)


def first_reg94_tag130(recs):
    hits = [r for r in recs if r.get("reg") == 94 and r.get("tag") == 130]
    return sorted(hits, key=lambda r: r["at"])[0]


class TestReg98Tag133PayloadIsGenuinelySeedInvariant(unittest.TestCase):
    """The one part of PR #1687's own "constant" framing that holds up
    under a real multi-seed check: reg=98,tag=133's own payload byte
    is identical across all three tested seed pairs for Add."""

    def test_payload_is_0x10_at_every_seed_pair(self):
        for s1, s2 in SEED_PAIRS:
            recs = decode(f"add_1x16_two_live_seed{s1}_{s2}_trivialx2.mcode.gz")
            hits = [r for r in recs if r.get("reg") == 98 and r.get("tag") == 133]
            self.assertEqual(len(hits), 1, (s1, s2))
            self.assertEqual(hits[0]["payload"], b"\x10", (s1, s2))


class TestReg94Tag130TrailingBytesAreSeedDependentNotConstant(unittest.TestCase):
    """The correction: the trailing 3 bytes PR #1687 called
    "constant" genuinely differ across seed pairs for Add."""

    def test_three_seed_pairs_give_three_different_trailing_byte_values(self):
        observed = {}
        for s1, s2 in SEED_PAIRS:
            recs = decode(f"add_1x16_two_live_seed{s1}_{s2}_trivialx2.mcode.gz")
            rec = first_reg94_tag130(recs)
            self.assertEqual(rec["p"], 3, (s1, s2))
            self.assertEqual(rec["payload"][0], 0x0F, (s1, s2))
            observed[(s1, s2)] = rec["payload"][1:]
        self.assertEqual(observed, EXPECTED_TRAILING_BYTES)
        # All three values are genuinely distinct -- not a coincidence
        # of only checking two seed pairs that happen to differ.
        self.assertEqual(len(set(observed.values())), 3)


class TestSubsOwnTrailingBytesMatchAddsAtTheSameSeedPairs(unittest.TestCase):
    """Confirms the earlier "byte-identical across ops" finding
    (PR #1687) generalizes to seed pairs beyond (1,2) too -- the match
    tracks the SHARED calibration seed, not a universal constant."""

    def test_matches_at_seed_7_42(self):
        recs = decode("sub_1x16_two_live_seed7_42_trivialx2.mcode.gz")
        rec = first_reg94_tag130(recs)
        self.assertEqual(rec["payload"], bytes.fromhex("0f238ec7"))

    def test_matches_at_seed_100_999(self):
        recs = decode("sub_1x16_two_live_seed100_999_trivialx2.mcode.gz")
        rec = first_reg94_tag130(recs)
        self.assertEqual(rec["payload"], bytes.fromhex("0ffb40c1"))


class TestFixturesDecodeCleanly(unittest.TestCase):
    def test_no_decode_errors(self):
        names = [
            f"add_1x16_two_live_seed{s1}_{s2}_trivialx2.mcode.gz"
            for s1, s2 in SEED_PAIRS
        ] + [
            "sub_1x16_two_live_seed7_42_trivialx2.mcode.gz",
            "sub_1x16_two_live_seed100_999_trivialx2.mcode.gz",
        ]
        for name in names:
            errs = mcode.check(load(name))
            self.assertEqual(errs, [], (name, errs))


if __name__ == "__main__":
    unittest.main()
