"""Systematic search for a MatMul-`var`-byte-style shape-dependent
selector byte in Conv's mcode: not found. Conv's shape-dependence is
diffuse across hundreds of bytes in its op-program segments, not
concentrated in one or two isolated selector bytes the way batched
MatMul's `var` byte is.

This session found batched MatMul has a single tag byte ("var", near
its site-A slot) whose value is a rich but reproducible function of
`batch`/`M`/`K` (`tests/test_axera_matmul_batched_var_byte.py`,
`tests/test_axera_matmul_var_byte_thresholds.py`,
`tests/test_axera_matmul_var_byte_k_plateaus.py`, and the period-32/
mod-4 structure decoded in later files). Conv has never been
systematically searched for an analogous isolated selector, only
compared build-to-build for specific already-known questions (the
output-scale quad, site A, the dilation-3/4 trigger, the `3x1`-vs-`1x3`
kernel-orientation delta). This file runs that search.

## Method

Built a small `Conv(x, w)` (`k=3`, `dilation=1`, `pad=1`, no bias) via
`pulsar2_docker.build()` (compile-only), varying one shape dimension
at a time and diffing same-length build pairs, the technique this
project has used throughout:

1. **Channel count and `group`** (`cin`/`cout` in `{2,4,8}`, `group` in
   `{1,2}`, `insz=8` fixed): three builds -- `cin=8,cout=4`; `cin=2,
   cout=2`; `cin=4,cout=4,group=2` -- coincidentally land on the
   identical 3176-byte total length. Pairwise diffs among them are 533-669
   bytes out of 3176 (17-21%) -- far too broad to localize a single
   selector byte; every dimension perturbs the weight table's own
   per-channel calibration (`weight_scales`) and the input/output
   scale distribution together, entangling shape with quantization
   throughout the stream.

2. **Spatial size alone** (`cin=cout=4`, `k=3`, `group=1` all held
   fixed, only `insz` varies -- the cleanest possible isolation, since
   spatial size does not touch the weight tensor's element count or
   its own calibration at all): `insz=9,10,14` land on an identical
   3888-byte length. `insz=9` vs `insz=10` differ at **1308 of 3888
   bytes (34%)** -- still far too broad, and this time with no weight-
   count confound to blame.

## Segment-level characterization: diffuse, not isolated -- and it
## excludes the two segments already known to be boilerplate

`mcode.segments()` on the `insz=9`-vs-`10` pair shows the diff
concentrated in segments 0-2 (the op-program-carrying segments this
project's kernel-orientation/dilation work already established) --
384/608, 194/576, and 723/1344 bytes differing respectively -- while
**segments 3 and 4 (256 and 352 bytes) show exactly zero diffs**, the
same "boilerplate, shape-independent" status this project already
assigned to the analogous segments in the `3x1`-vs-`1x3` orientation
work (`tests/test_axera_conv_kernel_orientation.py`) and the dilation-
trigger work.

**Confirmed this is real shape-driven content, not noise, via an
independent rebuild.** A rebuild of the identical `insz=9` config
differs from the original at only **3 bytes total, all inside segment
0** -- two orders of magnitude below the 1308-byte shape-driven diff,
and segments 1-4 are *exactly* byte-identical across the rebuild
(matching segments 3/4's already-zero shape-diff, and segment 1/2's
zero *rebuild*-diff contrasting with their substantial *shape*-diff).
None of the 1308 shape-driven diff bytes fall in this project's known
`~301-325` noise zone.

## Conclusion: no isolated selector byte found -- and this explains
## structurally why the MatMul method doesn't transfer

Unlike MatMul (two pure-activation tensors, whose op-program content
this project's fixtures show is largely shape-invariant at small
sizes, so a shape-dependent effect concentrates into a few specific
framing bytes near a quad), Conv's op-program instruction content
itself changes substantially with shape -- hundreds of bytes, not a
handful -- even in the *most* isolated single-dimension case tested
(spatial size alone, no weight-tensor confound). There is no
MatMul-`var`-byte analog waiting to be found by this search method in
Conv's mcode; the shape-dependence is genuinely diffuse throughout
segments 0-2, consistent with (and reinforcing) this project's already-
established finding that Conv's per-op encoding is deeply shape/scan-
order dependent (the `3x1`-vs-`1x3` orientation work's own "35%
subsequence match" finding, and the dilation-trigger work's real,
undecoded content). Not chased further here: decoding *what* varies
throughout segments 0-2 with spatial size, at the individual-record
level, is a much larger undertaking than this search-for-a-selector-
byte task and is left open.
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


class TestChannelAndGroupChangesArePervasive(unittest.TestCase):
    """Three shapes with nothing in common but an incidentally-matching
    total length (cin=8/cout=4, cin=2/cout=2, cin=4/cout=4/group=2) all
    differ from each other across 500+ of 3176 bytes -- far too broad
    to localize a single selector byte."""

    FIXTURES = [
        "conv_8c4c_insz8.mcode.gz",
        "conv_2c2c_insz8.mcode.gz",
        "conv_4c4c_insz8_group2.mcode.gz",
    ]

    def test_all_three_share_the_same_length(self):
        lengths = {len(load(name)) for name in self.FIXTURES}
        self.assertEqual(lengths, {3176})

    def test_every_pair_differs_across_hundreds_of_bytes(self):
        datas = [load(name) for name in self.FIXTURES]
        for i in range(len(datas)):
            for j in range(i + 1, len(datas)):
                diffs = sum(
                    1 for k in range(len(datas[i])) if datas[i][k] != datas[j][k]
                )
                self.assertGreater(
                    diffs,
                    400,
                    f"{self.FIXTURES[i]} vs {self.FIXTURES[j]}: expected a broad diff",
                )


class TestSpatialSizeAloneIsAlsoDiffuse(unittest.TestCase):
    """Holding cin=cout=4, k=3, group=1 fixed and varying ONLY insz
    (the cleanest possible isolation -- no weight-tensor confound)
    still produces a broad, diffuse diff, not an isolated byte."""

    def test_insz9_vs_insz10_differs_across_over_a_thousand_bytes(self):
        a = load("conv_4c4c_insz9.mcode.gz")
        b = load("conv_4c4c_insz10.mcode.gz")
        self.assertEqual(len(a), len(b))
        diffs = [i for i in range(len(a)) if a[i] != b[i]]
        self.assertGreater(len(diffs), 1000)
        self.assertEqual(mcode.check(a), [])
        self.assertEqual(mcode.check(b), [])


class TestDiffIsConcentratedInOpProgramSegmentsNotSelectorBytes(unittest.TestCase):
    """Segment-level breakdown: segments 0-2 (op-program) carry the
    diff broadly; segments 3-4 are exactly zero-diff, matching this
    project's already-established boilerplate classification for
    those segments at this shape family."""

    def test_segments_0_through_2_carry_the_diff_segments_3_4_dont(self):
        a = load("conv_4c4c_insz9.mcode.gz")
        b = load("conv_4c4c_insz10.mcode.gz")
        diffs = [i for i in range(len(a)) if a[i] != b[i]]
        _, segs = mcode.segments(a)
        self.assertEqual(len(segs), 5)
        per_segment = []
        for pos, length, _kind in segs:
            end = pos + length
            per_segment.append(sum(1 for d in diffs if pos <= d < end))
        # segments 0, 1, 2 each carry a real share of the diff
        for i in range(3):
            self.assertGreater(per_segment[i], 0, f"segment {i}: expected diff")
        # segments 3, 4 carry none at all
        self.assertEqual(per_segment[3], 0, "segment 3: expected zero diff")
        self.assertEqual(per_segment[4], 0, "segment 4: expected zero diff")

    def test_no_diff_bytes_fall_in_the_known_noise_zone(self):
        a = load("conv_4c4c_insz9.mcode.gz")
        b = load("conv_4c4c_insz10.mcode.gz")
        diffs = [i for i in range(len(a)) if a[i] != b[i]]
        noise_zone_hits = [d for d in diffs if 301 <= d <= 325]
        self.assertEqual(noise_zone_hits, [])


class TestRebuildNoiseFloorIsTwoOrdersOfMagnitudeSmaller(unittest.TestCase):
    """An independent rebuild of the identical insz=9 config confirms
    the 1308-byte insz=9-vs-10 diff is real shape-driven content, not
    noise: the rebuild pair differs at only 3 bytes total, all in
    segment 0, with segments 1-4 exactly byte-identical."""

    def test_rebuild_diff_is_tiny_and_segments_1_through_4_are_untouched(self):
        a = load("conv_4c4c_insz9.mcode.gz")
        a_rebuild = load("conv_4c4c_insz9_rebuild.mcode.gz")
        self.assertEqual(len(a), len(a_rebuild))
        diffs = [i for i in range(len(a)) if a[i] != a_rebuild[i]]
        self.assertLessEqual(len(diffs), 6, "expected an ordinary noise-floor diff")
        _, segs = mcode.segments(a)
        for idx, (pos, length, _kind) in enumerate(segs):
            if idx == 0:
                continue
            end = pos + length
            n = sum(1 for d in diffs if pos <= d < end)
            self.assertEqual(n, 0, f"segment {idx}: expected zero rebuild diff")


if __name__ == "__main__":
    unittest.main()
