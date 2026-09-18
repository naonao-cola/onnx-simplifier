"""Answers `tests/test_axera_matmul_and_x2_replacement_mechanism.py`
(PR #1686)'s own explicitly flagged next step: PR #1686 found Sub's
own trivial-`x2` replacement mechanism (`reg=94,tag=132` vacancy) is
not merely the same MECHANISM Add's own case showed (PR #1684) -- the
replacement bytes are IDENTICAL, because `x2`'s own calibration data
is shared regardless of which op consumes it. This was only checked
for Add and Sub. This file extends the check to Mul and Div.

## Finding: both match byte-for-byte too -- confirming the replacement
## is driven purely by `x2`'s own calibration data, across all four
## ops in this cluster, not just the two PR #1686 happened to check

**Mul's own case**: decoding `mul_1x16_two_live_seed1_2_
trivialx2.mcode.gz` against its own control
(`mul_1x16_two_live_seed1_2.mcode.gz`) finds the identical pattern
Add/Sub both showed: `reg=94,tag=132` (the real `x2` zero point)
present in the control, absent in the trivial fixture; in its place, a
new `reg=98,tag=133,payload=[0x10]` unit; the neighboring
`reg=94,tag=130` record widens by one byte with the SAME leading byte
(`0x0f`) and the SAME constant trailing bytes (`88 c1 bd`) Add/Sub
both showed.

**Div's own case**: `div_1x16_two_live_seed1_2_trivialzp2.mcode.gz`
(PR #1663's own trivial-`x2` fixture) shows the IDENTICAL pattern too
-- same `reg=98,tag=133,payload=[0x10]` unit, same `0f 88 c1 bd` /
`88 c1 bd` constant bytes at the widened `reg=94,tag=130` neighbor.
This is not a coincidence: PR #1663 established Div's own
"trivialzp2" fixtures use the SAME all-positive `2.0+0.3*randn()`
calibration design for `x2` that Add/Sub/Mul's own trivial-`x2`
fixtures use (only Div's own NON-degenerate baseline fixtures use a
different, sign-flip-safe divisor design) -- so `x2`'s own underlying
calibration data really is identical across all four ops in the
trivial case specifically, and the replacement bytes track that data,
not the op.

## What this establishes, precisely, and what it does not

**Established**: the trivial-`x2` replacement mechanism (`reg=98,
tag=133` new unit, `reg=94,tag=130` neighbor widening with constant
`0f 88 c1 bd`/`88 c1 bd` payload bytes) is now confirmed byte-
identical across ALL FOUR two-live-input ops (Add PR #1684, Sub/Mul/
Div here) -- closing PR #1686's own explicitly flagged open item.
Combined with PR #1686's own MatMul finding (the literal-quad-style
mechanism's own generic-filler reuse), this whole "trivial-zero-point
replacement is ordinary allocator filler tracking the vacated value's
own calibration data, not a dedicated wire form" picture is now
directly confirmed for every op and every locator type this arc has
ever characterized.

**NOT established**: whether this holds at shapes other than
`(1,16)`; whether the SAME `reg=98,tag=133` pair would still fire if
`x2`'s own trivial calibration data used a genuinely different
distribution (not tested -- only the one all-positive design this
whole project has ever built a trivial-`x2` fixture with).
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

CONSTANT_LEADING_PAYLOAD = bytes.fromhex("0f88c1bd")
CONSTANT_TRAILING_PAYLOAD = bytes.fromhex("88c1bd")


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def decode(name):
    return mcode.decode(load(name), **mcode.FULL_RULE)


class TrivialX2ReplacementByteMatchMixin:
    """Shared assertions applied to each op's own (control, trivial)
    fixture pair -- reused verbatim across Mul and Div below, following
    PR #1686's own established methodology."""

    CONTROL = None
    TRIVIAL = None

    def test_reg94_tag132_present_in_control_absent_in_trivial(self):
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

    def test_reg98_tag133_is_absent_from_the_control(self):
        # reg=98 can appear at unrelated tags in some controls (e.g.
        # Div's own control has a reg=98,tag=129 B-unit that has
        # nothing to do with this mechanism) -- what matters is that
        # the specific (reg=98, tag=133) pair the trivial fixture
        # introduces is genuinely new, not that register 98 itself
        # never appears anywhere in the control.
        ctrl = decode(self.CONTROL)
        self.assertEqual(
            [r for r in ctrl if r.get("reg") == 98 and r.get("tag") == 133], []
        )

    def test_neighboring_reg94_tag130_widens_with_adds_own_constant_bytes(self):
        ctrl = decode(self.CONTROL)
        triv = decode(self.TRIVIAL)
        ctrl_hits = [r for r in ctrl if r.get("reg") == 94 and r.get("tag") == 130]
        triv_hits = [r for r in triv if r.get("reg") == 94 and r.get("tag") == 130]
        self.assertEqual(len(ctrl_hits), 4)
        self.assertEqual(len(triv_hits), 4)
        first = sorted(triv_hits, key=lambda r: r["at"])[0]
        self.assertEqual(first["p"], 3)
        self.assertEqual(first["payload"], CONSTANT_LEADING_PAYLOAD)
        rest = sorted(triv_hits, key=lambda r: r["at"])[1:]
        for r in rest:
            self.assertEqual(r["p"], 2)
            self.assertEqual(r["payload"], CONSTANT_TRAILING_PAYLOAD)


class TestMulsOwnX2ReplacementMatchesAddAndSubExactly(
    TrivialX2ReplacementByteMatchMixin, unittest.TestCase
):
    CONTROL = "mul_1x16_two_live_seed1_2.mcode.gz"
    TRIVIAL = "mul_1x16_two_live_seed1_2_trivialx2.mcode.gz"


class TestDivsOwnX2ReplacementMatchesAddSubMulExactly(
    TrivialX2ReplacementByteMatchMixin, unittest.TestCase
):
    """Div's own NON-degenerate baseline uses a different, sign-flip
    divisor calibration design for x2 -- but its own trivial-x2
    fixture (PR #1663's own `trivialzp2` variant) uses the SAME
    all-positive design Add/Sub/Mul's trivial fixtures do, so the
    replacement content matches byte-for-byte despite Div's own
    baseline behaving differently elsewhere in this arc."""

    CONTROL = "div_1x16_two_live_seed1_2.mcode.gz"
    TRIVIAL = "div_1x16_two_live_seed1_2_trivialzp2.mcode.gz"


class TestFixturesDecodeCleanly(unittest.TestCase):
    def test_no_decode_errors(self):
        for name in (
            "mul_1x16_two_live_seed1_2.mcode.gz",
            "mul_1x16_two_live_seed1_2_trivialx2.mcode.gz",
            "div_1x16_two_live_seed1_2.mcode.gz",
            "div_1x16_two_live_seed1_2_trivialzp2.mcode.gz",
        ):
            errs = mcode.check(load(name))
            self.assertEqual(errs, [], (name, errs))


if __name__ == "__main__":
    unittest.main()
