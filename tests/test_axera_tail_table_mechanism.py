"""Continues `tests/test_axera_reg8_emit_capability_synthesis.py` (PR
#1631)'s own flagged recurring failure mode: three independent length-
changing edits this session's own generator work built
(`emit_conv_reg8_group`'s length-changing reconfigurations,
`emit_gemm_reg8_group`'s cross-anchor-form splices, and
`bank81_field192_operand`'s own K-driven reflow) all trip the identical
`mcode.check()` error -- `"tail: no readable segment table (no header
word points at a tail table vector)"` -- but nobody had decoded what
actually triggers it.

## The mechanism, decoded directly from `scripts/axera/mcode.py`

`mcode.tail_vector()` locates an mcode stream's tail FlatBuffers vector
by scanning the fixed header region (the first ~297 bytes, or up to the
convolution-engine channel-extent marker `a1 00 40 02` for graphs that
have one) for a 4-byte little-endian word `w` at some offset `o` such
that `o + w` points at a valid-looking vector -- a FlatBuffers-style
*relative* uoffset, not an absolute one. Every `reg=8` group this
session's own emit functions touch lives well after this header and
before the tail vector -- so when an edit changes that region's own
total length, the tail vector's own TRUE position shifts by the same
delta, but the header's own stored uoffset -- untouched by any of the
emit functions, none of which write before byte ~300 -- does not. The
scan in `tail_vector()` then fails to find ANY word satisfying its own
`o + w == <a valid vector>` check, hence "no header word points at a
tail table vector."

Confirmed directly at a single concrete case first (`emit_conv_reg8_group`'s
own shrink from `conv_dilation3.mcode.gz` -- 3528 bytes -- to a 2-byte-
shorter reconfiguration): the header word lives at absolute offset
`272`, stores `2792` before the edit (`272 + 2792 = 3064`, the real
tail vector on the unedited reference), and is unchanged -- still
`2792` -- after the edit, even though the true tail vector has moved to
`3062`. Rewriting that one word to `3062 - 272 = 2790` makes
`mcode.tail_vector()`, `mcode.segments()`, `mcode.decode()`, and
`mcode.check()` all succeed cleanly on the edited stream.

## The fix: `tiny_emit.retarget_tail_vector`

`scripts/axera/tiny_emit.py`'s new `retarget_tail_vector(reference_mcode,
edited_mcode)` generalizes this: it re-locates the header word fresh
from `reference_mcode` on every call (not hardcoded to offset 272 --
that number is specific to this one shape family, confirmed identical
across both the Conv and Gemm cases checked below only because they
happen to share it, not assumed), computes the length delta directly
from `len(edited_mcode) - len(reference_mcode)`, and rewrites that one
word in a copy of `edited_mcode`.

## Verified against all 4 already-known broken cases, both directions
## each

| case | direction | delta | before | after |
| --- | --- | --- | --- | --- |
| `emit_conv_reg8_group` | P4-form -> long-form (shrink) | -2 | tail-table error | clean |
| `emit_conv_reg8_group` | long-form -> P4-form (grow) | +2 | tail-table error | clean |
| `emit_gemm_reg8_group` | short-anchor donor -> long-anchor ref | +2 | tail-table error | clean |
| `emit_gemm_reg8_group` | long-anchor donor -> short-anchor ref | -2 | tail-table error | clean |

Every one of these 4 is checked below directly against the real
`tiny_emit.emit_conv_reg8_group`/`tiny_emit.emit_gemm_reg8_group`
functions and real committed fixtures -- not simulated. After the fix,
`mcode.segments()` is also checked to tile exactly to the (correctly
relocated) tail vector, and `mcode.decode()` is checked to succeed and
recover the expected number of records -- a genuine structural fix, not
a check-passes technicality.

## What this does NOT fix, checked directly and honestly

`bank81_field192_operand`'s own K-changing patch
(`tests/test_axera_bank81_field192_patch_verify.py`, PR #1625) is
LENGTH-PRESERVING -- a single 1-byte value overwrite, `len(patched) ==
len(reference)` -- so it never triggers `tail_vector()`'s own error in
the first place (`mcode.check()` already reports it clean, confirmed
below); `retarget_tail_vector` correctly computes `delta=0` and is a
genuine no-op there. Field192's own problem (a real `K=256` build
differs from the patched `K=512`-based stream in 73% of their shared
bytes) is not a stale pointer -- `K` drives the entire stream's
tiling/scheduling, not one field's own recorded value -- and this file
does not claim otherwise.
"""

import gzip
import os
import sys
import unittest

_AXERA_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts", "axera")
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import mcode  # noqa: E402
import tiny_emit  # noqa: E402

FIX = os.path.join(_AXERA_DIR, "fixtures")

CONV_CONFIG = {
    "conv_dilation3.mcode.gz": (("P2", 130), (8, "P1", 130), (242, "P3", 130)),
    "conv_dilation3_rebuild0.mcode.gz": (
        ("P4", 130),
        (8, "P2", 130),
        (8, "P1", 130),
    ),
}


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def hard_errors(data):
    return [e for e in mcode.check(data) if not e.startswith("coverage:")]


class TestConvShrinkIsFixed(unittest.TestCase):
    """conv_dilation3 (P2/reg8/reg242, long-form slot1) reconfigured to
    rebuild0's own config (P4 slot1, short form) shrinks the group by 2
    bytes and breaks the tail table before the fix."""

    def _out(self):
        base = load("conv_dilation3.mcode.gz")
        target = CONV_CONFIG["conv_dilation3_rebuild0.mcode.gz"]
        return base, tiny_emit.emit_conv_reg8_group(base, *target)

    def test_broken_before_fix(self):
        base, out = self._out()
        self.assertEqual(len(out), len(base) - 2)
        errs = hard_errors(out)
        self.assertTrue(errs)
        self.assertIn("tail", errs[0])

    def test_clean_after_fix(self):
        base, out = self._out()
        fixed = tiny_emit.retarget_tail_vector(base, out)
        self.assertEqual(hard_errors(fixed), [])

    def test_segments_and_decode_succeed_after_fix(self):
        base, out = self._out()
        fixed = tiny_emit.retarget_tail_vector(base, out)
        header, segs = mcode.segments(fixed)
        self.assertEqual(segs[-1][0] + segs[-1][1], mcode.tail_vector(fixed))
        recs = mcode.decode(fixed, **mcode.FULL_RULE)
        self.assertGreater(len(recs), 0)


class TestConvGrowIsFixed(unittest.TestCase):
    """The reverse direction: rebuild0 (short form) reconfigured to
    conv_dilation3's own config (long form) grows the group by 2 bytes."""

    def _out(self):
        base = load("conv_dilation3_rebuild0.mcode.gz")
        target = CONV_CONFIG["conv_dilation3.mcode.gz"]
        return base, tiny_emit.emit_conv_reg8_group(base, *target)

    def test_broken_before_fix(self):
        base, out = self._out()
        self.assertEqual(len(out), len(base) + 2)
        self.assertTrue(hard_errors(out))

    def test_clean_after_fix(self):
        base, out = self._out()
        fixed = tiny_emit.retarget_tail_vector(base, out)
        self.assertEqual(hard_errors(fixed), [])


class TestGemmCrossFormSplicesAreFixed(unittest.TestCase):
    """emit_gemm_reg8_group's own cross-anchor-form (length-changing)
    splices, both directions, from tests/test_axera_gemm_reg8_emit_verify.py's
    own TestCrossFormSpliceFailsPredictably."""

    def test_short_ref_long_donor(self):
        ref = load("gemm_1x512x1000_tb0.mcode.gz")
        donor = load("gemm_1x512x1000_tb0_rebuild0.mcode.gz")
        out = tiny_emit.emit_gemm_reg8_group(ref, donor)
        self.assertNotEqual(len(out), len(ref))
        self.assertTrue(hard_errors(out))
        fixed = tiny_emit.retarget_tail_vector(ref, out)
        self.assertEqual(hard_errors(fixed), [])
        header, segs = mcode.segments(fixed)
        self.assertEqual(segs[-1][0] + segs[-1][1], mcode.tail_vector(fixed))

    def test_long_ref_short_donor(self):
        ref = load("gemm_1x512x1000_tb0_rebuild0.mcode.gz")
        donor = load("gemm_1x512x1000_tb0.mcode.gz")
        out = tiny_emit.emit_gemm_reg8_group(ref, donor)
        self.assertNotEqual(len(out), len(ref))
        self.assertTrue(hard_errors(out))
        fixed = tiny_emit.retarget_tail_vector(ref, out)
        self.assertEqual(hard_errors(fixed), [])


class TestHeaderWordIsAtTheExpectedOffsetInThisSharedShapeFamily(unittest.TestCase):
    """Not hardcoded by retarget_tail_vector itself, but confirmed here
    that this shape family's own header word lives at the same absolute
    offset (272) for both Conv and Gemm's own fixtures -- a fact about
    this specific header layout, not an assumption the fix depends on."""

    def test_conv_header_word_offset(self):
        import struct

        ref = load("conv_dilation3.mcode.gz")
        t = mcode.tail_vector(ref)
        self.assertEqual(272 + struct.unpack_from("<I", ref, 272)[0], t)

    def test_gemm_header_word_offset(self):
        import struct

        ref = load("gemm_1x512x1000_tb0.mcode.gz")
        t = mcode.tail_vector(ref)
        self.assertEqual(272 + struct.unpack_from("<I", ref, 272)[0], t)


class TestFieldOneNineTwoPatchDoesNotNeedThisFixAndIsCorrectlyANoOp(unittest.TestCase):
    """bank81_field192_operand's own K-changing patch is length-
    preserving (a 1-byte in-place overwrite) -- it never trips
    tail_vector()'s own error, mcode.check() already reports it clean,
    and applying retarget_tail_vector to it is a correct, verified
    no-op (delta=0), not a silent failure to detect an unrelated
    problem."""

    def test_field192_patch_already_passes_check(self):
        ref = load("gemm_1x512x1000_tb0.mcode.gz")
        recs = mcode.decode(ref, **mcode.FULL_RULE)
        hits = [
            r
            for r in recs
            if r["kind"] == "V" and r.get("bank") == 0x81 and r.get("field") == 192
        ]
        self.assertTrue(hits)
        out = bytearray(ref)
        predicted = tiny_emit.bank81_field192_operand(256)
        for r in hits:
            at = r["at"]
            out[at + 4 : at + 7] = predicted
        out = bytes(out)
        self.assertEqual(len(out), len(ref))
        self.assertEqual(hard_errors(out), [])

    def test_retarget_is_a_correct_no_op_here(self):
        ref = load("gemm_1x512x1000_tb0.mcode.gz")
        recs = mcode.decode(ref, **mcode.FULL_RULE)
        hits = [
            r
            for r in recs
            if r["kind"] == "V" and r.get("bank") == 0x81 and r.get("field") == 192
        ]
        out = bytearray(ref)
        predicted = tiny_emit.bank81_field192_operand(256)
        for r in hits:
            at = r["at"]
            out[at + 4 : at + 7] = predicted
        out = bytes(out)
        fixed = tiny_emit.retarget_tail_vector(ref, out)
        self.assertEqual(fixed, out)


if __name__ == "__main__":
    unittest.main()
