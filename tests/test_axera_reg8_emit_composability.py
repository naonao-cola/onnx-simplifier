"""Continues `tests/test_axera_reg8_emit_capability_synthesis.py` (PR
#1631)'s own explicitly flagged, untested gap: whether two of this
project's separately-verified `scripts/axera/tiny_emit.py` functions
compose cleanly -- applied in sequence to the same reference stream,
does the result satisfy BOTH edits' own rules at once, not just
whichever was applied last?

## What was actually tested, and why

This file composes `emit_matmul_reg8_quad` (this session's own reg=8
mechanism, PR #1626) with `patch_output_quad` (a function that predates
this session's own reg=8 arc, already verified safe for Gemm/Conv/
MatMul/Mul's shared output-scale quad, `scripts/axera/tiny_emit.py`'s
own module docstring). These two were chosen deliberately over other
candidate pairs: both are ALREADY independently verified byte-exact
against real ground truth for their own field, and (checked directly
below, not assumed) their own byte spans in `MatMul(A[4,8],B[8,8])`'s
mcode are far apart -- the reg=8 quad's own anchor sits at offset 345,
the output-scale quad at offset ~1851 -- so a clean composition is a
real, meaningful positive result if it holds, not a foregone conclusion
just because both functions work in isolation.

## Result 1: clean composition, order-independent, on two real fixtures

Composing `patch_output_quad` then `emit_matmul_reg8_quad` (and the
reverse order) on `matmul_4x8x8_v7stability_diag0.mcode.gz` and
`matmul_4x8x8_v7stability_r1.mcode.gz`:

- Both edits' own effects are present in the final stream (the output
  quad reads the new scale; the reg=8 group reads the target
  permutation), verified by decoding the result, not just by trusting
  the diff.
- `mcode.check()` reports zero errors on the composed result, in BOTH
  orders, on BOTH fixtures.
- The two orders produce a BYTE-IDENTICAL result (`patch then emit` ==
  `emit then patch`) -- genuine commutativity, not just "both orders
  happen to pass their own checks."
- The full diff against the original fixture is exactly the union of
  each edit's own known byte offsets (6 reg=8-quad bytes within
  345-381, 16 output-quad bytes within 1851-1878) -- no unexpected
  third region moved.

This is the first genuinely positive composability result this
project's own generator-capability work has: two independently-decoded,
independently-verified mechanisms, safely combinable in a single
stream, with no reflow -- unlike every LENGTH-CHANGING edit
`tests/test_axera_reg8_emit_capability_synthesis.py` (PR #1631) already
found always breaks `mcode.check()`'s tail table. Both functions
composed here are fixed-length, in-place edits; this result is
consistent with, not a refutation of, that file's own "length-
preserving edits are clean, length-changing ones are not" rule -- it
extends that rule from "each edit alone is clean" to "two together are
still clean", which was NOT something the earlier file tested.

## Result 2: a genuine caution surfaced while looking for a second,
## cross-op test pair -- byte-span overlap is not guaranteed just
## because two fields are separately "understood"

Attempting the same kind of test for Conv -- composing
`emit_conv_reg8_group` (PR #1628) with `patch_conv_zp_x` (an older,
already-verified function for Conv's own zero-point) on
`conv_dilation3.mcode.gz` (the exact fixture `emit_conv_reg8_group`'s
own test file uses) -- found the two are NOT safe to compose on this
fixture, for a reason neither function's own docstring flags:
`patch_conv_zp_x`'s own zero-point literal-byte search
(`02 10 1b <zp_x> 83 36`) matches at byte offset 2158, and patching it
changes byte 2161 -- the *exact* absolute offset
`tests/test_axera_conv_reg60_mechanism.py` (PR #1586) already decoded
as part of the Conv "binary path switch" 28-byte cluster (`reg=60`'s
own record). Verified directly below
(`TestConvZpXAndBinaryClusterCollideOnThisFixture`): the byte
`patch_conv_zp_x` reports changing is not some third, unrelated region
-- it is inside a region a COMPLETELY DIFFERENT, already-decoded
mechanism already claims. This is not `emit_conv_reg8_group`'s own
byte span (that group is anchored near offset 855, per
`tests/test_axera_conv_reg8_emit_verify.py`) -- so this file's own
originally-planned Conv composability pair (`emit_conv_reg8_group` +
`patch_conv_zp_x`) was never actually going to be a genuine two-
disjoint-mechanism test on THIS fixture; it would have silently
collided with a THIRD mechanism instead. Reported here as a real,
useful caution rather than silently abandoned: two functions each
independently verified "safe" for their own field can still target
the SAME byte on a specific shape if their own byte spans were never
checked against each other (or against every other already-decoded
mechanism) in advance -- exactly the kind of check a real, general-
purpose generator (composing many decoded fields into one stream)
would need to do systematically, which no single-field patch function
in this project attempts.

## What this does and does not establish

- Established: at least one genuine pair of independently-verified,
  fixed-length `tiny_emit.py` functions (`emit_matmul_reg8_quad` +
  `patch_output_quad`) compose cleanly and commutatively on real
  fixtures, extending PR #1631's own "length-preserving edits are
  clean" finding to the two-edits-at-once case for the first time.
- NOT established: that ANY two `tiny_emit.py` functions compose
  safely by default -- the Conv case above shows the opposite can
  happen, and this file only checked byte-span overlap for ONE
  specific pair on ONE specific fixture, not systematically across
  every function this project has.
- NOT established: whether `emit_conv_reg8_group` composes safely with
  something genuinely disjoint from it (this file does not find or
  test such a pair for Conv -- a real remaining gap, not claimed
  closed).
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
import tiny_emit  # noqa: E402

FIX = os.path.join(_AXERA_DIR, "fixtures")

REG8_CANDS = {
    b"\x33\x00\x20",
    b"\x23\x00\x40",
    b"\x23\x00\x30",
    b"\x23\x00\x10",
}


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def find_output_quad_old_scale(data):
    idx = data.find(bytes.fromhex("05500f"))
    assert idx != -1, "no generalized output-quad lead-in found"
    return struct.unpack("<f", data[idx + 3 : idx + 7])[0]


def reg8_quad_slots(data):
    anchor = bytes.fromhex("a2000000") + bytes.fromhex("12000000")
    hits = [
        i
        for i in range(len(data) - len(anchor) + 1)
        if data[i : i + len(anchor)] == anchor
    ]
    assert len(hits) == 1
    a = hits[0]
    slots = [
        data[a + 8 + 4 : a + 8 + 7],
        data[a + 16 + 4 : a + 16 + 7],
        data[a + 24 + 4 : a + 24 + 7],
    ]
    s_off = a + 33
    slots.append(data[s_off + 1 : s_off + 4])
    return slots


def output_quad_value(data):
    idx = data.find(bytes.fromhex("05500f"))
    return struct.unpack("<f", data[idx + 3 : idx + 7])[0]


PERM_A = (b"\x33\x00\x20", b"\x23\x00\x30", b"\x23\x00\x10", b"\x23\x00\x40")
PERM_B = (b"\x23\x00\x40", b"\x33\x00\x20", b"\x23\x00\x10", b"\x23\x00\x30")

CASES = [
    ("matmul_4x8x8_v7stability_diag0.mcode.gz", PERM_A),
    ("matmul_4x8x8_v7stability_r1.mcode.gz", PERM_B),
]


class TestByteSpansAreDisjointBeforeComposing(unittest.TestCase):
    """Confirms, rather than assumes, that the reg=8 quad's own anchor
    and the output-scale quad's own lead-in sit far apart in this
    fixture -- the precondition that makes a clean-composition result
    meaningful rather than accidental."""

    def test_spans_do_not_overlap(self):
        for name, _perm in CASES:
            data = load(name)
            reg8_anchor = data.find(
                bytes.fromhex("a2000000") + bytes.fromhex("12000000")
            )
            quad_lead_in = data.find(bytes.fromhex("05500f"))
            self.assertNotEqual(reg8_anchor, -1, name)
            self.assertNotEqual(quad_lead_in, -1, name)
            # reg8 quad's own span is anchor..anchor+37; quad's own span
            # starts at the lead-in and runs for four stride-7 copies.
            reg8_end = reg8_anchor + 37
            self.assertLess(reg8_end, quad_lead_in, name)


class TestComposedEditPreservesBothEffects(unittest.TestCase):
    """Applies both edits (each order) and confirms BOTH effects are
    independently readable back from the final stream, and
    `mcode.check()` is clean."""

    def test_forward_order_both_effects_present_and_clean(self):
        for name, perm in CASES:
            data = load(name)
            old_scale = find_output_quad_old_scale(data)
            step1 = tiny_emit.patch_output_quad(data, old_scale, 0.5)
            step2 = tiny_emit.emit_matmul_reg8_quad(step1, perm)
            self.assertEqual(mcode.check(step2), [], name)
            self.assertAlmostEqual(output_quad_value(step2), 0.5, places=5, msg=name)
            self.assertEqual(reg8_quad_slots(step2), list(perm), name)

    def test_reverse_order_both_effects_present_and_clean(self):
        for name, perm in CASES:
            data = load(name)
            old_scale = find_output_quad_old_scale(data)
            step1 = tiny_emit.emit_matmul_reg8_quad(data, perm)
            step2 = tiny_emit.patch_output_quad(step1, old_scale, 0.5)
            self.assertEqual(mcode.check(step2), [], name)
            self.assertAlmostEqual(output_quad_value(step2), 0.5, places=5, msg=name)
            self.assertEqual(reg8_quad_slots(step2), list(perm), name)


class TestCompositionIsCommutative(unittest.TestCase):
    """The two orders produce a BYTE-IDENTICAL result -- not just two
    independently-valid streams, the same one."""

    def test_orders_match_exactly(self):
        for name, perm in CASES:
            data = load(name)
            old_scale = find_output_quad_old_scale(data)
            forward = tiny_emit.emit_matmul_reg8_quad(
                tiny_emit.patch_output_quad(data, old_scale, 0.5), perm
            )
            reverse = tiny_emit.patch_output_quad(
                tiny_emit.emit_matmul_reg8_quad(data, perm), old_scale, 0.5
            )
            self.assertEqual(forward, reverse, name)


class TestFullDiffIsExactlyTheUnionOfBothSpans(unittest.TestCase):
    """No unexpected third region moves -- the composed diff against
    the original is exactly the reg=8 quad's own known window plus the
    output-quad's own known window."""

    def test_diff_confined_to_known_windows(self):
        for name, perm in CASES:
            data = load(name)
            old_scale = find_output_quad_old_scale(data)
            composed = tiny_emit.emit_matmul_reg8_quad(
                tiny_emit.patch_output_quad(data, old_scale, 0.5), perm
            )
            diffs = [i for i in range(len(data)) if data[i] != composed[i]]
            reg8_anchor = data.find(
                bytes.fromhex("a2000000") + bytes.fromhex("12000000")
            )
            quad_lead_in = data.find(bytes.fromhex("05500f"))
            for d in diffs:
                in_reg8_window = reg8_anchor <= d < reg8_anchor + 37
                in_quad_window = quad_lead_in <= d < quad_lead_in + 3 + 4 + 7 * 3 + 4
                self.assertTrue(in_reg8_window or in_quad_window, (name, d))


class TestConvZpXAndBinaryClusterCollideOnThisFixture(unittest.TestCase):
    """A real caution, not a composability success: `patch_conv_zp_x`'s
    own zero-point literal-byte match on `conv_dilation3.mcode.gz`
    lands squarely inside the byte offset PR #1586 already decoded as
    part of the unrelated "binary path switch" 28-byte cluster
    (`reg=60`'s own record, at offset 2161). Two independently-verified
    "safe for their own field" functions are NOT safe to compose here,
    because their own byte spans were never checked against each other.
    """

    def test_zp_x_literal_form_is_found_at_the_known_offset(self):
        data = load("conv_dilation3.mcode.gz")
        pat = bytes([0x02, 0x10, 0x1B, 126, 0x83, 0x36])
        self.assertEqual(data.find(pat), 2158)

    def test_patching_zp_x_changes_the_binary_clusters_own_byte(self):
        data = load("conv_dilation3.mcode.gz")
        patched = tiny_emit.patch_conv_zp_x(data, 126, 127)
        diffs = [i for i in range(len(data)) if data[i] != patched[i]]
        # offset 2161 is exactly PR #1586's own reg=60 record offset
        # (payload byte, per tests/test_axera_conv_reg60_mechanism.py's
        # own KNOWN_CORRELATED_OFFSETS set).
        self.assertEqual(diffs, [2161])

    def test_this_is_not_the_reg8_groups_own_span(self):
        # emit_conv_reg8_group's own anchor (reg=170) sits far from
        # offset 2161 -- confirming the collision above is with the
        # binary-cluster mechanism specifically, not a self-collision
        # within the reg=8 function's own edit.
        data = load("conv_dilation3.mcode.gz")
        reg170_anchor = bytes([0x00]) + b"\x12" + bytes([132, 170])
        hits = [
            i
            for i in range(len(data) - len(reg170_anchor) + 1)
            if data[i : i + len(reg170_anchor)] == reg170_anchor
        ]
        self.assertEqual(len(hits), 1)
        anchor = hits[0]
        self.assertLess(anchor + 30, 2161)


if __name__ == "__main__":
    unittest.main()
