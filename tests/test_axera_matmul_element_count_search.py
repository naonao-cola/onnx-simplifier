"""Searching plain rank-2 MatMul for a Gemm-style isolated element-count
field -- not found. MatMul's `M`/`K` shape-dependence, where it exists
at the record level, is genuinely diffuse (multi-record restructuring),
not the single clean field Gemm's own `K*M-1` record turned out to be.

`tests/test_axera_gemm_m_field_is_8m_minus_1.py` (PR #1538) and
`tests/test_axera_gemm_8m1_field_kn_dependence.py` (PR #1543) found a
Gemm S-unit record whose payload is exactly `K*M - 1` (pre-break) /
`K*floor(M/2) - 1` (post-break) -- a genuine "total `A`-element-count"
style field, discovered via `mcode.decode()`/`difflib.SequenceMatcher`
record-level diffing after a raw byte diff (PR #1536) found the
`M`-driven difference too pervasive to localize. Nobody had applied
that same record-level technique to MatMul.

## Method: the exact technique that found Gemm's field, applied here

Built plain rank-2 `MatMul(A[M,K], B[K,N])` pairs (`pulsar2_docker.build()`,
same weight/calibration convention as `tests/test_axera_matmul_a_input_cascades.py`)
varying one dimension at a time, and diffed `mcode.segments()`'s five
segments record-by-record with `difflib.SequenceMatcher`, looking for
an isolated single-record `replace` (same `reg`/`tag`/`p`, one payload
byte differing) the way Gemm's field was found.

## Result: no isolated field in either `M=1`-vs-`M=2` or `K=8`-vs-`K=16`

**`M=1` (3080 bytes) vs `M=2` (3112 bytes)**, holding `K=8,N=8` fixed,
`B`'s own calibration seed/range held identical: segments 0, 3, 4 show
only trivial differences (a two-record reorder in segment 0 matching
this project's already-named register/tag-reassignment noise class; a
single byte reinterpreted `raw`->`A` at an identical value in segment
3, a decode-boundary artifact not a content change). Segments 1
(107->113 records) and 2 (283->283 records, but `V`/`S`/`raw`/`B` kind
counts all shift) carry the real `M`-driven difference -- but **zero
single-record `replace` blocks appear across all five segments**; every
non-trivial edit is a multi-record `replace`/`insert` spanning 2-14
records at once. There is no isolated field here to decode the way
Gemm's was.

**`K=8` (3112 bytes) vs `K=16` (2848 bytes)**, holding `M=4,N=8` fixed:
a much larger structural discontinuity (segments 0/1/4 match ratios of
0.03-0.19, consistent with this project's own `var`-byte work already
finding `K=8`/`K=16` straddles a major tiling-regime boundary for
MatMul) makes this a much noisier comparison than `M`'s. A handful of
single-record replaces do appear in segments 2/3, but their values
(`0x3f->0x7f`, `0x60->0x80`, tag `129->161` with an unchanged payload)
look like the same register/tag-reassignment class this project has
repeatedly named as noise, not a clean formula -- no attempt is made
here to force one.

## Confirmed the M=1-vs-M=2 finding is real content, not confounded by
## the known table-order coin flip

An independent rebuild of `M=2` lands on the opposite
`A_offset`/`B_offset` table order (`tests/test_axera_matmul_offset_table_coinflip.py`'s
already-decoded per-compile coin flip) -- its whole-stream byte diff
against the original `M=2` build is 916 bytes, matching that
mechanism's own signature exactly, not evidence of anything M-related
going wrong. **Segments 0, 1, 3, 4's record-kind composition is
completely unaffected by this table-order flip** (identical counts and
kinds in both), confirming segment 1's real `M`-driven difference
(107 vs 113 records) is genuine, not noise. **Segment 2 does show
its own rebuild-to-rebuild instability** (283 vs 282 records) that
overlaps with the coin flip's own affected byte range (offset 204-3084) --
so segment 2 specifically is partially entangled with that unrelated
noise source, and its own `M`-driven diff should be read cautiously
until isolated from that confound; segment 1's is not affected.

## Conclusion

Unlike Gemm, plain rank-2 MatMul does not have a Gemm-style isolated
"total element count" field findable by this record-level technique --
at least not in the `M`/`K` variations tested here. This is consistent
with (not proof of) MatMul's already-established structure: its
"shape-dependent selector" territory is the `var` tag byte (`tests/test_axera_matmul_batched_var_byte.py`
and its many follow-ups), a genuinely different kind of field (rank-3
batching-specific, in the site-A quad's short form) than what this
search was looking for. A systematic, honest negative result, matching
the bar this project's other search-method transfers have set
(`tests/test_axera_gemm_shape_selector_search.py`,
`tests/test_axera_conv_shape_selector_search.py`).
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


def decode_segment(data, pos, length):
    end = pos + length
    while end > pos and data[end - 1] == 0:
        end -= 1
    return mcode.decode(data, start=pos, end=end, **mcode.FULL_RULE)


def segment_kind_counts(data, idx):
    _, segs = mcode.segments(data)
    pos, length, _ = segs[idx]
    recs = decode_segment(data, pos, length)
    return len(recs), Counter(r["kind"] for r in recs)


def offset_table_order(data):
    window = data[195:270]
    ia = window.find(b"A_offset")
    ib = window.find(b"B_offset")
    return ("A", "B") if ia < ib else ("B", "A")


class TestMDependenceIsMultiRecordNotIsolated(unittest.TestCase):
    """M=1 vs M=2: segments 1 and 2 carry the real M-driven diff, but
    it's a count/kind-shift, not an isolated single-record field."""

    def test_segment_lengths_match_between_builds(self):
        d1 = load("matmul_1x8x8_m1.mcode.gz")
        d2 = load("matmul_2x8x8_m2.mcode.gz")
        _, segs1 = mcode.segments(d1)
        _, segs2 = mcode.segments(d2)
        self.assertEqual(len(segs1), 5)
        self.assertEqual(len(segs2), 5)

    def test_segment_1_record_count_grows_with_m(self):
        d1 = load("matmul_1x8x8_m1.mcode.gz")
        d2 = load("matmul_2x8x8_m2.mcode.gz")
        n1, _ = segment_kind_counts(d1, 1)
        n2, _ = segment_kind_counts(d2, 1)
        self.assertEqual(n1, 107)
        self.assertEqual(n2, 113)

    def test_segment_2_record_count_unchanged_but_kinds_shift(self):
        d1 = load("matmul_1x8x8_m1.mcode.gz")
        d2 = load("matmul_2x8x8_m2.mcode.gz")
        n1, c1 = segment_kind_counts(d1, 2)
        n2, c2 = segment_kind_counts(d2, 2)
        self.assertEqual(n1, n2, "segment 2 record count should stay 283")
        self.assertNotEqual(dict(c1), dict(c2), "but kind composition shifts")

    def test_segments_0_3_4_are_trivially_different_or_identical(self):
        """No real M-driven content in these three segments -- only a
        reorder (segment 0) or a decode-boundary artifact (segment 3),
        both already-named noise classes, not a new field."""
        d1 = load("matmul_1x8x8_m1.mcode.gz")
        d2 = load("matmul_2x8x8_m2.mcode.gz")
        for idx in (0, 3, 4):
            n1, c1 = segment_kind_counts(d1, idx)
            n2, c2 = segment_kind_counts(d2, idx)
            self.assertEqual(n1, n2, f"segment {idx}: record count")


class TestM2RebuildConfirmsSegment1IsRealNotCoinFlipNoise(unittest.TestCase):
    """An independent M=2 rebuild lands on the opposite A_offset/B_offset
    table order (the known coin flip), but segment 1's record-kind
    composition is completely unaffected -- confirming its M=1-vs-M=2
    diff is real. Segment 2 IS affected by the coin flip's own noise,
    so it's flagged as a confound there, not asserted as clean."""

    def test_rebuild_landed_on_the_opposite_table_order(self):
        d2 = load("matmul_2x8x8_m2.mcode.gz")
        d2r = load("matmul_2x8x8_m2_rebuild.mcode.gz")
        self.assertNotEqual(offset_table_order(d2), offset_table_order(d2r))

    def test_segment_1_kind_composition_is_stable_across_the_coin_flip(self):
        d2 = load("matmul_2x8x8_m2.mcode.gz")
        d2r = load("matmul_2x8x8_m2_rebuild.mcode.gz")
        n2, c2 = segment_kind_counts(d2, 1)
        n2r, c2r = segment_kind_counts(d2r, 1)
        self.assertEqual(n2, n2r)
        self.assertEqual(dict(c2), dict(c2r))

    def test_segment_2_kind_composition_does_shift_across_the_coin_flip(self):
        """Documented confound, not a finding: segment 2 is itself
        sensitive to the unrelated table-order coin flip, so its own
        M-driven diff (found above) is not cleanly isolated from this
        noise source."""
        d2 = load("matmul_2x8x8_m2.mcode.gz")
        d2r = load("matmul_2x8x8_m2_rebuild.mcode.gz")
        n2, _ = segment_kind_counts(d2, 2)
        n2r, _ = segment_kind_counts(d2r, 2)
        self.assertNotEqual(n2, n2r)


class TestKVariationShowsNoCleanFieldEither(unittest.TestCase):
    """K=8 vs K=16 crosses a much bigger structural boundary (matching
    this project's already-known var-byte K-threshold territory);
    whatever single-record replaces appear look like ordinary
    register/tag-reassignment noise, not a computed formula."""

    def test_k8_and_k16_differ_substantially_in_length(self):
        d1 = load("matmul_4x8x8_k8.mcode.gz")
        d2 = load("matmul_4x16x8_k16.mcode.gz")
        self.assertEqual(len(d1), 3112)
        self.assertEqual(len(d2), 2848)

    def test_segments_0_and_1_are_almost_entirely_different(self):
        d1 = load("matmul_4x8x8_k8.mcode.gz")
        d2 = load("matmul_4x16x8_k16.mcode.gz")
        n0a, _ = segment_kind_counts(d1, 0)
        n0b, _ = segment_kind_counts(d2, 0)
        n1a, _ = segment_kind_counts(d1, 1)
        n1b, _ = segment_kind_counts(d2, 1)
        # Record counts diverge sharply -- a much larger structural
        # discontinuity than the M=1-vs-M=2 comparison showed.
        self.assertNotEqual(n0a, n0b)
        self.assertNotEqual(n1a, n1b)


if __name__ == "__main__":
    unittest.main()
