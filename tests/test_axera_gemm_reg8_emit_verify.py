"""Extends `scripts/axera/tiny_emit.py`'s `emit_matmul_reg8_quad`
(`tests/test_axera_matmul_reg8_emit_verify.py`, merged) -- this
project's first genuine GENERATION capability for the reg=8 "second
noise mechanism" -- to Gemm's own version of the same mechanism
(`tests/test_axera_gemm_reg8_second_noise_source.py`, PR #1577), which
this project's earlier stock-take
(`tests/test_axera_generator_progress_stocktake.py`, PR #1620) had not
yet done.

## Gemm's own version is structurally harder than MatMul's

MatMul's own group is fixed-length (a 4-slot, always-a-full-permutation
overwrite in place). Gemm's own group is NOT fixed-length: its anchor
record (`reg=78`) has a short form (1-byte payload, `p=0`) and a long
form (3-byte payload, `p=2`), and the group's own total byte span is
either 21 or 23 bytes depending on which form the anchor uses --
verified directly against all 8 already-committed real
`Gemm(1,512,1000)` fixtures below (`TestGroupSpanCorrelatesWithAnchorForm`).
The `reg=0` biconditional indicator PR #1577 decoded is not a
separately-inserted record at all: it is a different grammar
interpretation of a FIXED byte range that always exists after the pool
slots (confirmed by diffing a no-indicator fixture against a
with-indicator one byte-for-byte -- both reach the group's own end
marker at the identical absolute offset).

`scripts/axera/tiny_emit.py`'s new `emit_gemm_reg8_group` does not
attempt to synthesize an arbitrary slot assignment from scratch (too
much real risk of mis-deriving the encoding under either anchor form)
-- it splices a real, already-compiled group verbatim from a donor
build into a reference stream, locating both via `mcode.decode()` (the
one function in `tiny_emit.py` that needs it; every other function in
that file is deliberately decode-free, since Gemm's own anchor bytes
recur elsewhere in the stream and a raw literal search is genuinely
ambiguous here, unlike MatMul's own unique anchor -- verified directly
below, `TestRawByteAnchorWouldBeAmbiguous`).

## Result: works cleanly within one anchor-form class, fails predictably
## across it

All 12 ordered same-anchor-form cross-pairs (4 short-form fixtures x 3
other short-form donors each, plus the analogous 4x3 for long-form)
splice cleanly: same total length, zero `mcode.check()` hard errors,
and the spliced group's own content plus `reg=0` indicator state read
back exactly as the donor's own. Splicing ACROSS anchor forms (a
length-changing edit) reliably breaks with a specific hard error --
`"tail: no readable segment table"` -- the same "local edit, global
consequence" pattern this session's other generator-progress work
(`bank81_field192_operand`'s own K-changing reflow test) already found
for a completely different field.
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

SHORT_FORM = [
    "gemm_1x512x1000_tb0.mcode.gz",
    "gemm_1x512x1000_tb0_rebuild1.mcode.gz",
    "gemm_1x512x1000_tb0_rebuild2.mcode.gz",
    "gemm_1x512x1000_tb0_stability_r2.mcode.gz",
]
LONG_FORM = [
    "gemm_1x512x1000_tb0_rebuild0.mcode.gz",
    "gemm_1x512x1000_tb0_stability_r0.mcode.gz",
    "gemm_1x512x1000_tb0_stability_r1.mcode.gz",
    "gemm_1x512x1000_tb0_stability_r3.mcode.gz",
]
ALL_NAMES = SHORT_FORM + LONG_FORM


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def decode(data):
    return mcode.decode(data, **mcode.FULL_RULE)


def reg0_has_extra_b147(recs):
    return any(
        r["kind"] == "B" and r.get("tag") == 147 and r.get("reg") == 0 for r in recs
    )


class TestGroupSpanCorrelatesWithAnchorForm(unittest.TestCase):
    """Every SHORT_FORM fixture's group spans exactly 21 bytes; every
    LONG_FORM fixture's spans exactly 23 -- confirmed directly via
    `_gemm_reg8_group_bounds`, not assumed."""

    def test_short_form_span_is_21(self):
        for name in SHORT_FORM:
            start, end = tiny_emit._gemm_reg8_group_bounds(load(name))
            self.assertEqual(start, 887, name)
            self.assertEqual(end - start, 21, name)

    def test_long_form_span_is_23(self):
        for name in LONG_FORM:
            start, end = tiny_emit._gemm_reg8_group_bounds(load(name))
            self.assertEqual(start, 887, name)
            self.assertEqual(end - start, 23, name)


class TestRawByteAnchorWouldBeAmbiguous(unittest.TestCase):
    """Confirms, directly, why this function needs `mcode.decode()`
    rather than a raw literal-byte anchor search the way
    `emit_matmul_reg8_quad` used: both of the anchor's own closing
    tag+reg byte pairs recur elsewhere in a real Gemm stream."""

    def test_short_anchor_pattern_is_not_unique(self):
        data = load("gemm_1x512x1000_tb0.mcode.gz")
        count = 0
        i = 0
        while True:
            j = data.find(b"\x84\x4e", i)
            if j == -1:
                break
            count += 1
            i = j + 1
        self.assertGreater(count, 1)

    def test_long_anchor_pattern_is_not_unique(self):
        data = load("gemm_1x512x1000_tb0_rebuild0.mcode.gz")
        count = 0
        i = 0
        while True:
            j = data.find(b"\x82\x4e", i)
            if j == -1:
                break
            count += 1
            i = j + 1
        self.assertGreater(count, 1)


class TestNoOpSpliceIsByteIdentical(unittest.TestCase):
    def test_all_eight_fixtures(self):
        for name in ALL_NAMES:
            data = load(name)
            out = tiny_emit.emit_gemm_reg8_group(data, data)
            self.assertEqual(out, data, name)


class TestSameFormSplicesRoundTripCleanly(unittest.TestCase):
    """All 12 ordered same-anchor-form cross-pairs per form class: same
    total length, zero mcode.check() hard errors, spliced group content
    and reg=0 indicator state exactly match the donor."""

    def _check_pair(self, ref_name, donor_name):
        ref = load(ref_name)
        donor = load(donor_name)
        out = tiny_emit.emit_gemm_reg8_group(ref, donor)
        self.assertEqual(len(out), len(ref), (ref_name, donor_name))
        hard = [e for e in mcode.check(out) if not e.startswith("coverage:")]
        self.assertEqual(hard, [], (ref_name, donor_name, hard))

        ref_start, ref_end = tiny_emit._gemm_reg8_group_bounds(ref)
        d_start, d_end = tiny_emit._gemm_reg8_group_bounds(donor)
        expected_group = donor[d_start:d_end]
        o_start, o_end = tiny_emit._gemm_reg8_group_bounds(out)
        self.assertEqual(out[o_start:o_end], expected_group, (ref_name, donor_name))

        out_recs = decode(out)
        donor_recs = decode(donor)
        self.assertEqual(
            reg0_has_extra_b147(out_recs),
            reg0_has_extra_b147(donor_recs),
            (ref_name, donor_name),
        )

        # Everything outside the spliced span is untouched.
        self.assertEqual(out[:o_start], ref[:ref_start], (ref_name, donor_name))
        self.assertEqual(out[o_end:], ref[ref_end:], (ref_name, donor_name))

    def test_all_short_form_cross_pairs(self):
        for ref_name in SHORT_FORM:
            for donor_name in SHORT_FORM:
                if ref_name == donor_name:
                    continue
                self._check_pair(ref_name, donor_name)

    def test_all_long_form_cross_pairs(self):
        for ref_name in LONG_FORM:
            for donor_name in LONG_FORM:
                if ref_name == donor_name:
                    continue
                self._check_pair(ref_name, donor_name)


class TestSpliceIndicatorFollowsDonorNotReference(unittest.TestCase):
    """A no-indicator reference spliced with a with-indicator donor
    gains the indicator; the reverse loses it -- the indicator state is
    a property of the spliced bytes, never the original reference's
    own state."""

    def test_reference_without_gains_indicator_from_donor_with(self):
        ref = load("gemm_1x512x1000_tb0.mcode.gz")
        donor = load("gemm_1x512x1000_tb0_rebuild1.mcode.gz")
        self.assertFalse(reg0_has_extra_b147(decode(ref)))
        self.assertTrue(reg0_has_extra_b147(decode(donor)))
        out = tiny_emit.emit_gemm_reg8_group(ref, donor)
        self.assertTrue(reg0_has_extra_b147(decode(out)))

    def test_reference_with_loses_indicator_from_donor_without(self):
        ref = load("gemm_1x512x1000_tb0_rebuild1.mcode.gz")
        donor = load("gemm_1x512x1000_tb0.mcode.gz")
        self.assertTrue(reg0_has_extra_b147(decode(ref)))
        self.assertFalse(reg0_has_extra_b147(decode(donor)))
        out = tiny_emit.emit_gemm_reg8_group(ref, donor)
        self.assertFalse(reg0_has_extra_b147(decode(out)))


class TestCrossFormSpliceFailsPredictably(unittest.TestCase):
    """Splicing across anchor forms (a length-changing edit) reliably
    produces the same specific hard mcode.check() error -- not a silent
    success, and not an arbitrary/unpredictable failure."""

    def test_short_ref_long_donor_breaks_tail_table(self):
        ref = load("gemm_1x512x1000_tb0.mcode.gz")
        donor = load("gemm_1x512x1000_tb0_rebuild0.mcode.gz")
        out = tiny_emit.emit_gemm_reg8_group(ref, donor)
        self.assertNotEqual(len(out), len(ref))
        hard = [e for e in mcode.check(out) if not e.startswith("coverage:")]
        self.assertEqual(len(hard), 1)
        self.assertIn("tail", hard[0])
        self.assertIn("segment table", hard[0])

    def test_long_ref_short_donor_breaks_tail_table(self):
        ref = load("gemm_1x512x1000_tb0_rebuild0.mcode.gz")
        donor = load("gemm_1x512x1000_tb0.mcode.gz")
        out = tiny_emit.emit_gemm_reg8_group(ref, donor)
        self.assertNotEqual(len(out), len(ref))
        hard = [e for e in mcode.check(out) if not e.startswith("coverage:")]
        self.assertEqual(len(hard), 1)
        self.assertIn("tail", hard[0])
        self.assertIn("segment table", hard[0])


class TestBoundsHelperRaisesOnInvalidInput(unittest.TestCase):
    def test_no_anchor_raises(self):
        with self.assertRaises(ValueError):
            tiny_emit._gemm_reg8_group_bounds(b"\x00" * 300)

    def test_no_donor_group_in_truncated_stream_raises(self):
        ref = load("gemm_1x512x1000_tb0.mcode.gz")
        truncated = ref[:900]
        with self.assertRaises(ValueError):
            tiny_emit._gemm_reg8_group_bounds(truncated)


if __name__ == "__main__":
    unittest.main()
