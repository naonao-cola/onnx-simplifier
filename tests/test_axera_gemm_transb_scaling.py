"""Scaling up Gemm's `transB` diff toward the README's original
"gemm_base" 85-byte block -- reproduces two already-known noise classes
at every scale tried, never a third, novel field.

`scripts/axera/README.md`'s "### `Gemm` joins the MAC engines, and a
real, substantial signal from `transB`" section (found on a real
`resnet18d`-derived FC-layer-shaped Gemm, "gemm_base", never committed
as a fixture) reported a clean 85-byte contiguous block (offset
1620-1705, with 5-byte substrings `f129ff3b81`/`f5852513c8` each
repeated 3x) as "by a wide margin the largest and cleanest real signal
isolated in this whole investigation... a strong, well-motivated,
precisely-located target for whoever attempts the next level of
decoding." `tests/test_axera_gemm_output_quad.py` (already merged)
explicitly built Gemm at a much smaller synthetic scale specifically to
make the *output-scale quad* tractable, and explicitly disclaimed
reproducing this 85-byte block. This file is the first attempt to chase
that block directly, by scaling shape up toward it.

## What was tried

A same-length `transB=0`/`transB=1` pair was built and diffed at nine
shapes beyond the three already-committed tiny fixtures (`1x8x8`,
`4x16x8`, `8x8x8`): `4x8x32`, `4x8x64`, `4x32x8`, `4x64x8`, `4x32x32`,
`4x64x64`, `4x128x8`, `4x256x8`, and finally **`1x512x1000`** -- the
README's own text says this Gemm is resnet18d's final FC layer, and
standard resnet18's FC is `in_features=512`, `out_features=1000`
(ImageNet), batch `M=1`, which is the closest reconstruction of
"gemm_base" attempted here (not claimed to be the exact original
weights/calibration -- those are lost -- only the same op shape).

**Result: at every one of these nine shapes, every byte that differs
between `transB=0` and `transB=1` (net of the already-known ~295-330
basic noise zone) belongs to one of exactly two already-decoded/
already-named mechanisms. No third, novel signal appeared at any
scale, including the real FC-layer shape.**

1. **The output-scale quad's ULP flip** (`TestGemmOutputScaleQuad`,
   `test_axera_gemm_output_quad.py`) generalizes to every larger shape
   tested where it fires at all: `4x64x8` (offsets 1675/1682/1689/1696,
   `0xe9`->`0xea`), `4x128x8` (1676/1683/1690/1697, `0x4c`->`0x4b`), and
   -- confirmed here directly, not just by analogy -- **the real
   FC-layer shape itself**: `gemm_1x512x1000_{tb0,tb1}.mcode.gz` differ
   at offsets 2457/2464/2471/2478, each `0x08`->`0x07`, and decoding the
   bytes just before the first hit shows the exact established frame,
   unchanged: `05 50 0f 08 fb 57 3e 81 76 03` -- `05 50 0f` lead-in,
   `08 fb 57 3e` = float32 `0.21091854572296143`, `81 76 03` tail,
   repeated 4x at stride 7 (`test_output_scale_quad_present_at_fc_scale`
   below). The quant model confirms this is genuinely the output scale:
   `output_scales` is `0.21091854572296143` (tb0) vs
   `0.21091853082180023` (tb1) -- the same tiny float64->float32
   rounding-boundary artifact already explained for the small shapes,
   now reproduced at ~40x the original tested scale. `4x256x8` shows no
   diff outside the basic noise zone at all -- this artifact is
   probabilistic (whether double-precision output_scale happens to
   straddle a float32 rounding boundary for that build's specific
   weight/calibration data), so its *absence* at a given shape is
   expected some of the time, not evidence against the mechanism.

2. **A 3-way rotation across two S-unit payloads and one raw (undecoded)
   byte pair**, the same general noise class this project already named
   for Gemm's own M=8 case (`test_axera_gemm_m8_diff_is_allocation_noise.py`)
   and for MatMul's parameter name-table entries
   (`test_axera_matmul_a_input_cascades.py`), but precisely characterized
   here rather than just multiset-matched. `mcode.decode()` on offsets
   891-908 finds two `[p][payload][tag][reg]` short units at fixed
   positions 891 and 897 (both `tag=0x82`, both `reg=8` -- the same
   "reg8 hosts rotating short-unit payloads" pattern the M8 closing file
   already documented), each a 3-byte payload whose middle byte is
   always `0x00`; immediately after them, 5 more bytes decode() leaves
   as unparsed `raw` (903, 905, 907 constant; 904 and 906 vary). Reducing
   each payload/raw-pair to its two non-constant bytes gives three
   "slots" -- `d[892],d[894]` (record 1's payload), `d[898],d[900]`
   (record 2's payload), `d[904],d[906]` (the trailing raw pair) -- each
   holding one of exactly three 2-byte values, `(0x93,0x40)`,
   `(0x93,0x30)`, `(0x23,0x20)`, in `tb0`. In `tb1` the same three values
   reappear at the same three slots, cyclically rotated: `tb0`'s
   record-1 value becomes `tb1`'s raw-pair value, `tb0`'s record-2 value
   becomes `tb1`'s record-1 value, and `tb0`'s raw-pair value becomes
   `tb1`'s record-2 value (`test_the_fc_scale_permutation_is_a_3way_rotation`
   below). Every other byte in the 891-908 window (`p`/tag/reg bytes,
   both payloads' constant middle byte, the raw region's own separator
   bytes) is identical between builds -- this is a clean reassignment of
   which of 3 fixed slots holds which of 3 fixed values, the same
   register/slot-allocation phenomenon as the earlier-decoded cases, not
   a value computed from `transB` itself. `4x128x8`'s six
   noise-zone-adjacent bytes at 301-325 are ordinary basic noise
   (already in the known zone), not this class.

**What this does and does not establish.** It does not decode the
original 85-byte "gemm_base" block -- that exact shape/weights were
never reproduced, and no shape tried here (even the real FC-layer
dimensions) produced an 85-byte contiguous diff with repeated 5-byte
substrings. But it substantially narrows what that block plausibly is:
every "real" (non-basic-noise) `transB` diff mechanism this project has
ever found, at any scale from `4x8x8` up to a real 512->1000 FC layer,
reduces to these same two already-decoded/already-named classes. An
85-byte block is easily explained by scaling class 2 (S-unit
register/payload permutation noise) up: `resnet18d`'s own mcode has
*thousands* of short units in its configuration blocks
(README's "The layout that explains all of it" section: 750-1,444 short
units per config block), so a resnet18d-scale Gemm plausibly has far
more of these swappable short-unit records than the 2-record pair found
here, and a compiler-internal allocator reordering more of them at once
would produce a proportionally larger contiguous-looking permutation
block -- this is offered as a plausible, evidence-consistent hypothesis
for what the original finding was, not a proof, since the original
build/weights are not reproducible from the README text alone.
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

FIX = os.path.join(_AXERA_DIR, "fixtures")


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


class TestOutputScaleQuadGeneralizesToFCScale(unittest.TestCase):
    """The already-decoded output-scale quad's ULP-flip transB signal,
    now confirmed at a real FC-layer shape (512 -> 1000)."""

    OUTPUT_SCALE_TB0 = 0.21091854572296143

    def test_output_scale_quad_present_at_fc_scale(self):
        d0 = load("gemm_1x512x1000_tb0.mcode.gz")
        pat = struct.pack("<f", self.OUTPUT_SCALE_TB0)
        found = [i for i in range(len(d0) - 3) if d0[i : i + 4] == pat]
        self.assertEqual(len(found), 4, "quad count")
        strides = {b - a for a, b in zip(found, found[1:])}
        self.assertEqual(strides, {7}, "quad stride")
        self.assertEqual(d0[found[0] - 3 : found[0]].hex(), "05500f", "lead-in")
        for i in found:
            self.assertEqual(d0[i + 4], 0x81, f"@{i}: tail byte 0")

    def test_the_only_other_diff_class_is_the_known_permutation(self):
        d0 = load("gemm_1x512x1000_tb0.mcode.gz")
        d1 = load("gemm_1x512x1000_tb1.mcode.gz")
        self.assertEqual(len(d0), len(d1))
        diffs = [i for i in range(len(d0)) if d0[i] != d1[i]]
        quad_offsets = set(
            range(i, i + 4)
            for i in [
                j
                for j in range(len(d0) - 3)
                if d0[j : j + 4] == struct.pack("<f", self.OUTPUT_SCALE_TB0)
            ]
        )
        quad_bytes = {b for rng in quad_offsets for b in rng}
        permutation_bytes = set(range(894, 907))
        for i in diffs:
            self.assertTrue(
                i in quad_bytes or i in permutation_bytes,
                f"@{i}: unexplained diff outside both known classes",
            )


class TestFCScalePermutationIsRegisterAllocationNoise(unittest.TestCase):
    """The other diff cluster at FC scale, decoded via mcode.decode():
    a 3-way rotation across two S-unit payloads and one raw (undecoded)
    byte pair -- the same general slot/register-reassignment class as
    Gemm's own M=8 case and MatMul's A-input cascade, precisely
    characterized here rather than just multiset-matched."""

    def test_the_two_short_units_are_reg8_at_fixed_positions(self):
        d0 = load("gemm_1x512x1000_tb0.mcode.gz")
        d1 = load("gemm_1x512x1000_tb1.mcode.gz")
        for d, label in ((d0, "tb0"), (d1, "tb1")):
            recs = mcode.decode(d, start=891, end=903)
            self.assertEqual(len(recs), 2, f"{label}: expected 2 short units")
            for r in recs:
                self.assertEqual(r["kind"], "S", f"{label}: expected a short unit")
                self.assertEqual(r["tag"], 0x82, f"{label}: expected tag 0x82")
                self.assertEqual(r["reg"], 8, f"{label}: expected reg8")
                self.assertEqual(r["payload"][1], 0, f"{label}: payload middle byte")

    def test_the_permutation_is_a_3way_rotation(self):
        d0 = load("gemm_1x512x1000_tb0.mcode.gz")
        d1 = load("gemm_1x512x1000_tb1.mcode.gz")

        def slots(d):
            return [(d[892], d[894]), (d[898], d[900]), (d[904], d[906])]

        s0, s1 = slots(d0), slots(d1)
        self.assertEqual(sorted(s0), sorted(s1), "same 3 values, just reassigned")
        self.assertEqual(
            s1, [s0[1], s0[2], s0[0]], "expected a specific 3-cycle: 0->2, 1->0, 2->1"
        )

        # Everything else in the window is untouched: the two records'
        # p/tag/reg bytes and payload middle byte, and the raw region's
        # separator bytes.
        for off in (891, 893, 895, 896, 897, 899, 901, 902, 903, 905, 907):
            self.assertEqual(d0[off], d1[off], f"@{off}: expected unchanged")


class TestOutputScaleQuadAtSmallerLargerScale(unittest.TestCase):
    """A second, smaller confirmation point (K=64, N=8) between the
    tiny already-committed fixtures and the full FC-layer scale."""

    def test_quad_only_diff_at_4x64x8(self):
        d0 = load("gemm_4x64x8_tb0.mcode.gz")
        d1 = load("gemm_4x64x8_tb1.mcode.gz")
        self.assertEqual(len(d0), len(d1))
        diffs = [i for i in range(len(d0)) if d0[i] != d1[i]]
        self.assertEqual(
            diffs, [1675, 1682, 1689, 1696], "expected exactly the quad's 4 bytes"
        )
        for i in diffs:
            self.assertEqual(d0[i], 0xE9)
            self.assertEqual(d1[i], 0xEA)


if __name__ == "__main__":
    unittest.main()
