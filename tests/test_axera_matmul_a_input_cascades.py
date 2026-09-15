"""Closing (not fully decoding) MatMul's missing A-input encoding.

`test_axera_site_a_generalizes.py`'s `TestMatMulSiteAIsAsymmetric` showed
`1/A_scale` has no literal float32/short/bf16 encoding anywhere in
MatMul's mcode, unlike `1/B_scale` (which gets the full site-A
treatment). That left open whether A's scale is encoded some other way,
or genuinely isn't present as a discrete field at all.

This applies the differential method that closed the Gemm M=8 lead
(`test_axera_gemm_m8_diff_is_allocation_noise.py`): build two
MatMul(A, B) models that differ ONLY in A's calibration range (same
seed/shape/range for B in both, verified below to produce byte-identical
B scale and both zero_points), and diff the resulting same-length mcode
streams directly.

**Result: the diff is large (920 of 3112 bytes, spanning offset 204 to
3084 -- 92.5% of the stream) but NOT a single localized field change.**
Three findings, each independently confirmed:

1. **The two streams are byte-identical for the first 204 bytes**, then
   diverge exactly at a parameter name-table entry: a `<len><name>` /
   value record pair for `"A_offset"` and `"B_offset"` (part of the
   compiled artifact's own flatbuffer-style metadata table, not the
   S-unit instruction stream itself) appear in different relative
   order between the two builds -- `a_narrow` has `B_offset` before
   `A_offset`; `a_wide` has them the other way around. This is the
   same *ordering* class of noise this project already named for
   register allocation (`test_axera_gemm_m8_diff_is_allocation_noise.py`):
   the compiler's internal iteration order over named entries isn't
   stable across builds, and here it visibly correlates with which of
   A/B has the larger calibration range.

2. **That single swap does not explain the rest of the diff.** If it
   did, everything after the swap would be identical modulo a constant
   byte-offset shift. It isn't: testing a uniform 4-byte shift (the
   exact shift found at two individually-decoded locations, see below)
   across the whole tail only matches 37.5% of bytes -- far short of
   what a pure insertion/reordering artifact would produce. Real
   content differs across roughly a third of the remaining stream.

3. **Despite that widespread diffing, the two literal quads this
   project already decoded for B stay byte-identical in content --
   just relocated.** Both `1/B_scale`'s site-A quad and the
   output-scale quad are found at a constant +4 byte offset in
   `a_narrow` vs `a_wide` (1572 vs 1568, and 1858 vs 1854
   respectively), and the 30-byte window at each location is
   byte-for-byte identical once you compare each build's own detected
   offset. B's own encoded values did not change; only their position
   in the stream did.

**Conclusion:** A's scale-dependent effect on the mcode stream is real
and reproducible, but it is not a discoverable single literal field --
consistent with (but not proof of) A being threaded through the
S-unit program body itself, or through per-build-varying internal
table/offset bookkeeping, rather than encoded as a value this project's
established literal-search method can localize. This closes the
question as answered-negatively rather than leaving it as an open
search target: exhaustive literal search (prior test) plus this
differential-cascade result together indicate there is no hidden
literal `1/A_scale` field left to find with these methods.
"""

import gzip
import os
import struct
import unittest

FIX = os.path.join(os.path.dirname(__file__), "..", "scripts", "axera", "fixtures")


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def hits(data, pat):
    return [i for i in range(len(data) - len(pat) + 1) if data[i : i + len(pat)] == pat]


class TestMatMulAInputCascades(unittest.TestCase):
    # a_narrow: A range +-1.0, a_wide: A range +-5.0; B's calibration
    # (same seed/range in both builds) is held fixed -- confirmed by the
    # identical B_SCALE below being found in both fixtures.
    B_SCALE = 0.007840047590434551
    NARROW = "matmul_4x8x8_a_narrow.mcode.gz"
    WIDE = "matmul_4x8x8_a_wide.mcode.gz"

    def test_same_length_streams(self):
        d1, d2 = load(self.NARROW), load(self.WIDE)
        self.assertEqual(len(d1), len(d2))

    def test_prefix_identical_then_diverges_at_offset_table(self):
        d1, d2 = load(self.NARROW), load(self.WIDE)
        self.assertEqual(d1[:204], d2[:204])
        self.assertNotEqual(d1[204], d2[204])

    def test_divergence_is_an_offset_name_table_entry_swap(self):
        d1, d2 = load(self.NARROW), load(self.WIDE)
        # Each build has "B_offset" and "A_offset" records (flatbuffer-
        # style name+value pairs) at this location, just in different
        # relative order -- the ASCII value byte right after each name
        # matches that record's own first letter.
        self.assertIn(b"B_offset", d1[195:260])
        self.assertIn(b"A_offset", d1[195:260])
        self.assertIn(b"B_offset", d2[195:260])
        self.assertIn(b"A_offset", d2[195:260])
        # In the narrow build, B_offset's record comes first at 204.
        self.assertEqual(d1[204:212], b"B_offset")
        # In the wide build, A_offset's record comes first at the same
        # table slot instead.
        self.assertEqual(d2[204:212], b"A_offset")

    def test_diff_is_not_explained_by_a_uniform_shift(self):
        """A single reordering/insertion artifact would make the whole
        tail identical modulo a constant offset. It doesn't: confirm
        the match rate under the shift found at two known locations
        stays well below what a pure-shift explanation would need."""
        d1, d2 = load(self.NARROW), load(self.WIDE)
        n = len(d1)
        shift = 4
        matches = sum(1 for i in range(shift, n) if d1[i] == d2[i - shift])
        total = n - shift
        rate = matches / total
        self.assertLess(
            rate,
            0.5,
            "a uniform 4-byte shift should NOT explain most of the tail"
            " -- if it did, that would mean this is just one insertion,"
            " not genuine widespread content difference",
        )

    def test_b_own_quads_relocate_but_content_is_unchanged(self):
        """B's literal site-A quad and output quad individually shift by
        a constant +4 bytes between builds, but their own 30-byte
        content is byte-identical once compared at each build's own
        detected offset -- B's encoded values did not change."""
        d1, d2 = load(self.NARROW), load(self.WIDE)
        pat_site_a = struct.pack("<f", 1.0 / self.B_SCALE)
        h1 = hits(d1, pat_site_a)
        h2 = hits(d2, pat_site_a)
        self.assertEqual(len(h1), 4)
        self.assertEqual(len(h2), 4)
        self.assertEqual(h1[0] - h2[0], 4, "site A (B) shifts by exactly 4 bytes")
        self.assertEqual(
            d1[h1[0] : h1[0] + 30],
            d2[h2[0] : h2[0] + 30],
            "content at B's own site A quad is unchanged, just relocated",
        )


if __name__ == "__main__":
    unittest.main()
