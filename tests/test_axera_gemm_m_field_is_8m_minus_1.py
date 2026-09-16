"""Decoding Gemm's `M`-dependence at the record level: buried inside the
"pervasive, unlocalizable" diff PR #1536 found, one isolated S-unit
record encodes `8*M - 1` exactly.

`tests/test_axera_gemm_shape_selector_search.py` (PR #1536) applied
the raw-byte-diff search that found MatMul's shape-dependent `var`
byte to Gemm, and found no analog: `M=1` vs `M=2` (byte-identical
`B`/`C` weight data, same 2,568-byte mcode length) differs at 395
bytes spanning two large segments, far too pervasive to localize a
single selector byte by raw bytes alone. It explicitly flagged the
next step as unresolved: "whether an isolated Gemm selector byte
exists at all, buried inside this pervasive difference and only
findable via a `mcode.decode()`/`segments()`-level structural diff...
rather than a raw byte diff."

This does that record-level diff, using PR #1536's own `gemm_1x8x8_m1`/
`gemm_2x8x8_m2` fixtures (reused here byte-for-byte, not rebuilt) plus
one new `M=4` data point.

## Method: `difflib.SequenceMatcher` on record signatures, the
## technique that resolved Conv's own kernel-orientation question

Decoding the two segments PR #1536 found differ (offset 344/416 and
760/960) via `mcode.decode()` and comparing RECORD sequences (not raw
bytes) with `difflib.SequenceMatcher` gives a much higher match ratio
than Conv's analogous comparisons (`0.847`/`0.897` here vs. Conv
orientation work's `0.35`/`0.52`) -- Gemm's `M=1`-vs-`M=2` difference
is mostly small, localized `replace` edits on an otherwise-shared
record sequence, not the wholesale restructuring Conv shows. Both
segments show a net **+5 records** between `M=1` and `M=2`.

## The isolated field: one S-unit's payload is `8*M - 1`

Among the small `replace` blocks, one is a single record swapped for a
single record, same `reg`/`tag`/`p`, differing only in the payload's
last byte -- `mcode.decode()`'s `p=2` short unit at (segment-relative)
offset 513, `tag=130`, `reg=70`, payload `d0 0c <XX>`:

| M | payload trailing byte | value | `8*M - 1` |
| --- | --- | --- | --- |
| 1 | `0x07` | 7 | 7 |
| 2 | `0x0f` | 15 | 15 |
| 4 | `0x1f` | 31 | 31 |

Exact match at all three points tested. The `d0 0c` prefix and the
record's own `p`/`tag`/`reg` fields are identical across all three
builds -- only this one trailing byte moves, and it moves by exactly
what `8*M - 1` predicts.

**Confirmed above the noise floor.** An independent rebuild of `M=4`
reproduces the identical record at the identical offset
(`d0 0c 1f`, unchanged) -- the only 2 bytes that differ between the
`M=4` build and its rebuild are at offsets 303/311, inside this
project's already-known `~295-330` noise zone, nowhere near this
field's own offset (513).

## What this means for PR #1536's framing

PR #1536 was right that a *raw byte diff* can't localize a Gemm
selector -- the surrounding 395-byte diff is real and pervasive, most
of it presumably legitimate M-driven tiling/requantization content
this file does not decode. But a genuine, isolated, falsifiable field
DOES exist inside that diff, at the record level: this specific S-unit
is not part of the pervasive restructuring, it is one clean value that
happens to sit among a lot of noise-to-this-search other content.
*Why* `8*M - 1` (plausibly a zero-indexed row-count-minus-one style
field, or an address/stride quantity derived from `M`) is not decoded
here -- only that this exact field, at this exact record signature,
carries it. Only `M ∈ {1,2,4}` were tested; whether the formula holds
for non-power-of-two `M` or breaks down at larger values (the way
MatMul's own clean-looking rules sometimes turned out to have hidden
periodicity once tested further) is untested and left open.
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


def find_field(data):
    """Locate the p=2 short unit tag=130/reg=70 payload d0 0c <XX> in
    the 344-byte-offset op-program segment, and return its trailing
    payload byte, or None if absent."""
    _, segs = mcode.segments(data)
    for pos, length, _ in segs:
        for r in decode_segment(data, pos, length):
            if (
                r.get("kind") == "S"
                and r.get("tag") == 130
                and r.get("reg") == 70
                and r.get("p") == 2
                and r["payload"][:2] == b"\xd0\x0c"
            ):
                return r["payload"][2]
    return None


class TestGemmMFieldIs8MMinus1(unittest.TestCase):
    CASES = {
        "gemm_1x8x8_m1.mcode.gz": 1,
        "gemm_2x8x8_m2.mcode.gz": 2,
        "gemm_4x8x8_m4.mcode.gz": 4,
    }

    def test_field_matches_8m_minus_1_at_every_tested_m(self):
        for fname, m in self.CASES.items():
            data = load(fname)
            value = find_field(data)
            self.assertIsNotNone(value, f"{fname}: field not found")
            self.assertEqual(value, 8 * m - 1, f"{fname}: M={m}")

    def test_field_survives_an_independent_rebuild_of_m4(self):
        original = load("gemm_4x8x8_m4.mcode.gz")
        rebuild = load("gemm_4x8x8_m4_rebuild.mcode.gz")
        self.assertEqual(find_field(original), 31)
        self.assertEqual(find_field(rebuild), 31)

    def test_m4_rebuild_diff_is_small_and_outside_the_field(self):
        """The only bytes that move between the M=4 build and its
        rebuild are ordinary noise, not this field."""
        original = load("gemm_4x8x8_m4.mcode.gz")
        rebuild = load("gemm_4x8x8_m4_rebuild.mcode.gz")
        self.assertEqual(len(original), len(rebuild))
        diffs = [i for i in range(len(original)) if original[i] != rebuild[i]]
        self.assertLessEqual(
            len(diffs), 10, f"unexpectedly large rebuild diff: {diffs}"
        )
        for i in diffs:
            self.assertTrue(290 <= i <= 335, f"@{i}: outside the known noise zone")


if __name__ == "__main__":
    unittest.main()
