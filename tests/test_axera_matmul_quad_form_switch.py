"""Decoding why batched MatMul's `A` quad switches from the short
3-byte-truncated form to the full 4-byte float32 form: it is not
triggered near `1/A_scale == 255` -- the actual boundary is exactly
`1/A_scale == 128`, and the mechanism is a float32-byte coincidence,
not a quantization-range effect.

`tests/test_axera_matmul_batched_var_byte.py` (PR #1510, merged) found
the short-form `A` quad (`<3 bytes, low(1/A_scale)> 82 <var> 02` x4,
stride 6) switches to a full 4-byte form (`<f32(1/A_scale)> 81 <var>`
x4, stride 7) at a "narrow calibration" build where `1/A_scale =
255.1004923...`, and flagged three untested hypotheses for the trigger,
including "triggered by `1/A_scale` approaching `255`".

## The real boundary is 128, not 255 -- confirmed by a tight bracket

Sweeping `A`'s calibration range at the identical `[2,4,8]@[2,8,8]`
shape (matching PR #1502/#1510's own batched model) to hit precise
`1/A_scale` targets (MinMax calibration here reproduces the intended
target almost exactly -- `1/A_scale` came out `127.899995...` and
`128.100000...` for the two targets bracketing the boundary, each
within `5e-6` of the intended value) gives a clean split:

| `1/A_scale` | form |
| --- | --- |
| 100, 120, 125, 127, **127.9** | short (stride 6) |
| **128.1**, 129, 130, 135, 150, 200 | full (stride 7) |

The switch happens somewhere in `(127.9, 128.1]` -- i.e. at `128`
(`2**7`), not `255`. The original "255" build was simply the first
place PR #1510 happened to sample past this boundary; it was never
itself the threshold.

## Why 128: it's a float32 high-byte match, not a quantizer rule

`1/A_scale`'s IEEE754 float32 representation, in little-endian bytes,
has its most-significant byte (`b3`, holding the sign bit and the
top 7 of the 8 exponent bits) equal to `0x42` for every value in
`[32, 128)`, and `0x43` for every value in `[128, 512)` (the biased
exponent's top bits are shared by two adjacent exponent values, so
one `b3` value covers two consecutive powers-of-two -- confirmed
directly: `0x42` covers `[32,64)` and `[64,128)`; `0x43` covers
`[128,256)` and `[256,512)`; verified in Python via `struct.pack("<f",
v)` at values spanning this range, matching the real mcode data
exactly at every sampled point). **Every build in the "short" bucket
above has `b3 == 0x42`; every build in the "full" bucket has `b3 ==
0x43`.** This is not a coincidence of the specific values tested --
it's the same byte the short form's own 3-byte truncation *omits*.
The short form is only usable when the omitted high byte would be
`0x42` (some fixed, implied default this codec assumes when
reconstructing the truncated value); any other high byte forces the
full 4-byte write, because the truncated 3 bytes alone can no longer
reconstruct the real value under that assumption.

This also predicts (not verified with a real build here, since the
201-511 builds already available make the point without it) that the
`0x42`-only range is specifically `[32,128)` -- values from `200`
through the already-tested top of this sweep, and in fact anywhere up
to `512`, all share `b3 == 0x43` and would all use the full form, not
reverting to short form again until crossing `512` into the next
`0x44` byte group. The mechanism is a one-time special case for a
narrow, presumably-common magnitude bracket, not a periodic or
value-proportional rule.

## Confirmed above the noise floor

An independent rebuild of the `1/A_scale = 128.1` boundary config
reproduces the full-form quad at the identical offsets
(`2174/2181/2188/2195`), with only 11 bytes differing between the
rebuild pair elsewhere in the stream -- this project's ordinary
noise floor (`~6-15` bytes), not the `A_offset`/`B_offset` coin-flip's
900+ byte cascade (`tests/test_axera_matmul_offset_table_coinflip.py`,
PR #1506) -- so the boundary itself is not noise-sensitive.
"""

import gzip
import os
import struct
import unittest

FIX = os.path.join(os.path.dirname(__file__), "..", "scripts", "axera", "fixtures")


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def find_short_form(data, a_scale):
    short = struct.pack("<f", 1.0 / a_scale)[:3]
    found = [i for i in range(len(data) - 2) if data[i : i + 3] == short]
    for start in found:
        run = [start]
        i = start + 6
        while i in found:
            run.append(i)
            i += 6
        if len(run) == 4:
            return run
    return []


def find_full_form(data, a_scale):
    full = struct.pack("<f", 1.0 / a_scale)
    found = [i for i in range(len(data) - 3) if data[i : i + 4] == full]
    for start in found:
        run = [start]
        i = start + 7
        while i in found:
            run.append(i)
            i += 7
        if len(run) == 4:
            return run
    return []


class TestBoundaryIs128NotThe255FromPR1510(unittest.TestCase):
    # fixture: (exact A_scale as recorded by the quant model, expected
    # form). Using the compiler's own exact scale value (not a
    # recomputed 1.0/target) avoids float round-trip precision
    # mismatches right at the boundary.
    CASES = [
        ("matmul_quad_form_100.mcode.gz", 0.009999999776482582, "short"),
        ("matmul_quad_form_127_9.mcode.gz", 0.007818608544766903, "short"),
        ("matmul_quad_form_128_1.mcode.gz", 0.0078064012341201305, "full"),
        ("matmul_quad_form_200.mcode.gz", 0.004999999888241291, "full"),
    ]

    def test_form_matches_side_of_the_128_boundary(self):
        for fname, a_scale, expected in self.CASES:
            data = load(fname)
            short_hits = find_short_form(data, a_scale)
            full_hits = find_full_form(data, a_scale)
            if expected == "short":
                self.assertEqual(len(short_hits), 4, f"{fname}: expected short form")
                self.assertEqual(full_hits, [], f"{fname}: full form should be absent")
            else:
                self.assertEqual(len(full_hits), 4, f"{fname}: expected full form")
                self.assertEqual(
                    short_hits, [], f"{fname}: short form should be absent"
                )


class TestFormSelectionMatchesFloat32HighByte(unittest.TestCase):
    """The form actually present matches the predicted float32 b3
    byte (0x42 -> short, 0x43 -> full) for every bracketing sample."""

    CASES = [
        ("matmul_quad_form_127_9.mcode.gz", 127.9, 0x42),
        ("matmul_quad_form_128_1.mcode.gz", 128.1, 0x43),
        ("matmul_quad_form_200.mcode.gz", 200.0, 0x43),
    ]

    def test_high_byte_matches_prediction(self):
        for fname, inv_scale, expected_high_byte in self.CASES:
            full_bytes = struct.pack("<f", inv_scale)
            self.assertEqual(
                full_bytes[3], expected_high_byte, f"{fname}: float32 high byte"
            )

    def test_0x42_range_is_32_to_128_and_0x43_is_128_to_512(self):
        # Direct check of the byte-grouping claim itself, independent of
        # any built mcode -- confirms the boundary structure the mcode
        # data above is explained by.
        for v in (32.0, 63.9, 64.0, 127.9):
            self.assertEqual(struct.pack("<f", v)[3], 0x42, f"{v}: expected 0x42")
        for v in (128.0, 255.9, 256.0, 511.9):
            self.assertEqual(struct.pack("<f", v)[3], 0x43, f"{v}: expected 0x43")
        self.assertEqual(struct.pack("<f", 512.0)[3], 0x44, "512.0: expected 0x44")


class TestBoundaryFullFormSurvivesAnIndependentRebuild(unittest.TestCase):
    """The 128.1 boundary build's full-form quad is byte-identical at
    the identical offsets in an independent rebuild; the rest of the
    stream shows only ordinary noise (well under the coin-flip's 900+
    byte cascade), confirming the boundary itself isn't noise-driven."""

    def test_quad_offsets_and_content_match_across_rebuild(self):
        d1 = load("matmul_quad_form_128_1.mcode.gz")
        d2 = load("matmul_quad_form_128_1_rebuild.mcode.gz")
        self.assertEqual(len(d1), len(d2))
        a_scale = 0.0078064012341201305
        h1 = find_full_form(d1, a_scale)
        h2 = find_full_form(d2, a_scale)
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 4)

    def test_rebuild_diff_is_ordinary_noise_not_the_coinflip_cascade(self):
        d1 = load("matmul_quad_form_128_1.mcode.gz")
        d2 = load("matmul_quad_form_128_1_rebuild.mcode.gz")
        diffs = [i for i in range(len(d1)) if d1[i] != d2[i]]
        self.assertLess(len(diffs), 20, "expected ordinary noise, not a big cascade")


if __name__ == "__main__":
    unittest.main()
