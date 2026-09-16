"""What happens to Gemm's `K*M-1` field when it goes valueless
(Regime 2, `K in [33,64]` at `M=4`): the missing value does not move to
another record -- the whole segment reflows diffusely at every regime
boundary, the same "genuine restructuring, not a relocation" pattern
this project has already found for MatMul's and Conv's own shape
transitions.

`tests/test_axera_gemm_km1_k_regime_check.py` (PR #1545) found Gemm's
decoded `K*M-1` short-unit record (payload prefix `d0 0c`) goes
valueless at `K in [33,64]` (`M=4` fixed): `p` (declared payload
length) shrinks from 2 to 1, removing room for a trailing value byte
entirely rather than truncating one. It left open whether the missing
value moves somewhere else in the stream or is genuinely dropped.

## The whole-stream length is invariant to K across all three regimes

Fixture whole-stream lengths, `M=4` throughout: `K=32` (Regime 1,
has-value) = 2632 bytes; `K=33`/`K=64` (Regime 2, valueless) = 2632
bytes; `K=65` (Regime 3, widened 2-byte value) = 2632 bytes. Despite
the `d0 0c`-prefixed record itself changing width -- 3 bytes (Regime
1, `p=2`) -> 2 bytes (Regime 2, `p=1`) -> 4 bytes (Regime 3, `p=3`) --
the total mcode length never moves. `mcode.segments()` confirms all
five segments sit at the identical `(offset, length)` pairs in every
one of these four builds; the byte(s) freed or consumed by the field's
own width change are absorbed entirely within the same 448-byte
op-program segment (offset 344), not by a shift in segment boundaries.

## The missing value is not relocated to another record

Searching the FULL decoded stream (every segment, not just the one
carrying the field) for any other short unit at `reg=70` -- the same
register the field itself uses -- finds none in the valueless builds:
`K=32` has two `reg=70` records (the field itself, value `127`, plus
one unrelated `tag=129` record); `K=33` and `K=64` have exactly one
(the valueless field, `d0 0c`, no second `reg=70` record anywhere);
`K=65` (Regime 3) has zero (`reg=70` isn't used there at all -- Regime
3's own widened record uses `reg=26` instead, per PR #1545). No new
record with the field's own `d0 0c` payload prefix appears anywhere
else either. This search is exhaustive across the whole stream, not
limited to the field's own segment -- if the value 127 (or its
Regime-2-appropriate equivalent) had moved to a same-tag/same-reg
record somewhere else, this would have found it.

## Instead: the surrounding segment reflows diffusely at every regime boundary

Record-level (`mcode.decode()`) diffing of the 448-byte op-program
segment, the same `difflib.SequenceMatcher`-on-record-signatures
technique that resolved Conv's own kernel-orientation and Gemm's own
`K*M-1` questions:

- **Regime 1 -> Regime 2** (`K=32` vs `K=33`): match ratio **0.79**,
  record count **109 -> 110** (net +1). The first content difference
  appears at offset 377 -- well *before* the field's own offset 513 --
  confirming `K`'s value already perturbs upstream segment content
  independent of the field itself, not just at the one decoded record.
- **Regime 2 -> Regime 3** (`K=64` vs `K=65`): match ratio **0.78**,
  record count **110 -> 115** (net +5).

Both transitions show the same qualitative shape: dozens of small
`replace`/`insert` edits scattered through the segment, absolute
record offsets drifting apart after the first divergence point, and a
net record-count change that differs between the two boundaries (+1
vs. +5) -- not a clean "one field moved from slot A to slot B" story.
This matches the pattern this project has already established for
other shape-driven regime transitions (Conv's `k=8`-to-`k=9` kernel-
orientation shift, `tests/test_axera_conv_orientation_k9_regime_shift.py`;
MatMul's own diffuse `M`/`K` restructuring,
`tests/test_axera_matmul_element_count_search.py`) -- a real
compiler-level reflow of the surrounding instruction stream, not a
single relocated value.

## Conclusion

The value the field would carry under a naive extension of `K*M-1`
into Regime 2 is genuinely absent from the mcode, not moved to a
recoverable location -- confirmed by an exhaustive same-register
search across the whole stream, not just a local check. Its
disappearance coincides with (and is likely a downstream symptom of,
not the cause of) a broader, diffuse restructuring of the same segment
that begins upstream of the field's own offset. Why the compiler drops
the value specifically in this `K` range, rather than writing a
still-single-byte value the way Regime 1 does, remains undecoded --
this file narrows "did it move" to a confident no, not "why it's
gone."
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


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def decode_segment(data, pos, length):
    end = pos + length
    while end > pos and data[end - 1] == 0:
        end -= 1
    return mcode.decode(data, start=pos, end=end, **mcode.FULL_RULE)


def full_decode(data):
    out = []
    _, segs = mcode.segments(data)
    for pos, length, _ in segs:
        out += decode_segment(data, pos, length)
    return out


class TestWholeStreamLengthIsInvariantAcrossAllThreeRegimes(unittest.TestCase):
    FIXTURES = [
        "gemm_4x32x8_m4k32.mcode.gz",
        "gemm_4x33x8_m4k33_r2probe.mcode.gz",
        "gemm_4x64x8_m4k64_r2probe.mcode.gz",
        "gemm_4x65x8_m4k65_r2probe.mcode.gz",
    ]

    def test_all_four_builds_are_the_same_length(self):
        lengths = {len(load(name)) for name in self.FIXTURES}
        self.assertEqual(lengths, {2632})

    def test_all_four_builds_have_identical_segment_offsets(self):
        offsets = set()
        for name in self.FIXTURES:
            _, segs = mcode.segments(load(name))
            offsets.add(tuple((pos, length) for pos, length, _ in segs))
        self.assertEqual(
            len(offsets), 1, "expected identical segment layout in all four"
        )


class TestNoRelocatedRecordAtTheSameRegister(unittest.TestCase):
    """Exhaustive whole-stream search for another reg=70 short unit --
    if the missing value had moved to a same-register record
    elsewhere, this would find it. It doesn't."""

    def _reg70_short_units(self, name):
        return [
            r
            for r in full_decode(load(name))
            if r.get("kind") == "S" and r.get("reg") == 70
        ]

    def test_k32_has_the_field_plus_one_unrelated_reg70_record(self):
        recs = self._reg70_short_units("gemm_4x32x8_m4k32.mcode.gz")
        self.assertEqual(len(recs), 2)
        payloads = {r["payload"] for r in recs}
        self.assertIn(b"\xd0\x0c\x7f", payloads, "the field itself, value 127 = 4*32-1")

    def test_k33_has_only_the_valueless_field_no_second_reg70_record(self):
        recs = self._reg70_short_units("gemm_4x33x8_m4k33_r2probe.mcode.gz")
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["payload"], b"\xd0\x0c")

    def test_k64_has_only_the_valueless_field_no_second_reg70_record(self):
        recs = self._reg70_short_units("gemm_4x64x8_m4k64_r2probe.mcode.gz")
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["payload"], b"\xd0\x0c")

    def test_k65_uses_no_reg70_record_at_all(self):
        """Regime 3 moves the field to reg=26 (per PR #1545); reg=70
        is entirely unused in this build, not merely valueless."""
        recs = self._reg70_short_units("gemm_4x65x8_m4k65_r2probe.mcode.gz")
        self.assertEqual(recs, [])


class TestSegmentReflowsDiffuselyAtBothRegimeBoundaries(unittest.TestCase):
    """Record-level diff of the 448-byte op-program segment (offset
    344) shows dozens of scattered edits at each regime boundary, with
    a different net record-count change each time -- not a single
    relocated field."""

    SEG_POS, SEG_LEN = 344, 448

    def _records(self, name):
        return decode_segment(load(name), self.SEG_POS, self.SEG_LEN)

    def test_regime1_to_regime2_boundary_reflows(self):
        import difflib

        r32 = self._records("gemm_4x32x8_m4k32.mcode.gz")
        r33 = self._records("gemm_4x33x8_m4k33_r2probe.mcode.gz")
        self.assertEqual(len(r32), 109)
        self.assertEqual(len(r33), 110)

        sig32 = [tuple(sorted((k, v) for k, v in r.items() if k != "at")) for r in r32]
        sig33 = [tuple(sorted((k, v) for k, v in r.items() if k != "at")) for r in r33]
        ratio = difflib.SequenceMatcher(a=sig32, b=sig33).ratio()
        self.assertLess(
            ratio, 0.85, "expected genuinely scattered edits, not near-identical"
        )
        self.assertGreater(
            ratio, 0.5, "sanity: still mostly the same op-program skeleton"
        )

    def test_regime2_to_regime3_boundary_reflows(self):
        import difflib

        r64 = self._records("gemm_4x64x8_m4k64_r2probe.mcode.gz")
        r65 = self._records("gemm_4x65x8_m4k65_r2probe.mcode.gz")
        self.assertEqual(len(r64), 110)
        self.assertEqual(len(r65), 115)

        sig64 = [tuple(sorted((k, v) for k, v in r.items() if k != "at")) for r in r64]
        sig65 = [tuple(sorted((k, v) for k, v in r.items() if k != "at")) for r in r65]
        ratio = difflib.SequenceMatcher(a=sig64, b=sig65).ratio()
        self.assertLess(
            ratio, 0.85, "expected genuinely scattered edits, not near-identical"
        )
        self.assertGreater(
            ratio, 0.5, "sanity: still mostly the same op-program skeleton"
        )

    def test_first_divergence_between_k32_and_k33_is_upstream_of_the_field(self):
        """The field itself sits at offset 513; the segment's content
        already diverges at offset 377, well before it -- K perturbs
        the segment broadly, not just at the one decoded record."""
        r32 = self._records("gemm_4x32x8_m4k32.mcode.gz")
        r33 = self._records("gemm_4x33x8_m4k33_r2probe.mcode.gz")
        first_diff_at = None
        for a, b in zip(r32, r33):
            if {k: v for k, v in a.items() if k != "at"} != {
                k: v for k, v in b.items() if k != "at"
            }:
                first_diff_at = a["at"]
                break
        self.assertIsNotNone(first_diff_at)
        self.assertLess(first_diff_at, 513)


if __name__ == "__main__":
    unittest.main()
