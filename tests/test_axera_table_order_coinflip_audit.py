"""Closing the audit `tests/test_axera_matmul_offset_table_coinflip.py`
(PR #1506) explicitly left open: does its unconditioned per-compile
name-table order coin flip generalize to Gemm/Conv, and does it
retroactively explain any of this project's other "looked noise-free on
one rebuild pair, wasn't on a second" corrections for unrelated ops?

**Part 1: Gemm and Conv cannot have this coin flip -- not just
empirically absent, but structurally impossible, confirmed both by a
direct compiler rejection and by 19 already-committed fixtures.**

MatMul's coin flip needs (at least) two *peer* named tensors in the
compiled FlatBuffers offset table -- `A_offset` and `B_offset`, both
representing genuinely live, non-constant graph inputs whose runtime
buffer address the compiler must resolve and store by name, with no
fixed convention for which one goes first. Gemm and Conv were tested
directly for the same structural precondition by building each with
every possible operand (Gemm's `B`/`C`, Conv's `bias`) declared as a
genuine graph *input* rather than an ONNX initializer -- the same
shape of graph MatMul's own `A`/`B` already are -- and Pulsar2 rejects
both outright:

- `Gemm(A, B)` with both as non-initializer inputs fails to compile
  with `NotImplementedError('Should fuse Gemm (two non-parameter
  inputs) to MatMul.')` -- the compiler's own error message states
  directly that this shape of graph is not a supported Gemm at all; it
  is required to already be expressed as `MatMul`.
- `Conv(X, W, bias)` with `bias` as a non-initializer input fails with
  `AttributeError("'NoneType' object has no attribute 'reshape'")` in
  the quantizer -- consistent with (though less explicit about) the
  same requirement: Conv's weight and bias must be static constants,
  matching this project's own already-established finding that Conv's
  per-channel bias/requant multiplier is computed in closed form from
  static weight data (`scripts/axera/README.md`'s "The 1,719
  unexplained bits are the requantisation, in closed form",
  `scripts/axera/emitter.py`'s `requant_block()`).

So Gemm always has exactly one live, offset-addressed input (`A`) plus
the output (`Y`); Conv always has exactly one (`X`/lowercase `x` in the
compiled table) plus `Y`. **Confirmed directly against 11 independently
built Gemm fixtures (5 distinct shapes, `origin/master`'s existing
`scripts/axera/fixtures/gemm_*.mcode.gz`) and 8 independently built
Conv fixtures (2 distinct dilation configs, including
`conv_dilation{2,3}_rebuild*` -- literal unchanged-config rebuild
pairs from PR #1508, the exact same kind of sample MatMul's own 8-way
rebuild test used): every single one has its two `_offset` entries at
the identical byte offsets (`Y_offset`/`y_offset` at 148, `A_offset`/
`x_offset` at 196) in the identical order, zero exceptions.** Contrast
with MatMul's own 8-way rebuild test
(`tests/test_axera_matmul_offset_table_coinflip.py`), which found
roughly half its rebuilds landing each way. This isn't merely "not yet
observed to vary" -- Gemm/Conv's table never has two *peer* entries to
begin with, so there is nothing for an unordered-iteration compiler
internal to permute.

## Part 2: none of this project's three documented "second rebuild
pair showed real diffs" corrections are this mechanism

`scripts/axera/README.md` documents exactly three instances of "looked
noise-free/deterministic on one rebuild pair, a second pair showed
real diffs" (all found by searching for its own "Correction, caught
the same way" idiom and the `auto_pad`/`ceil_mode` false-lead
passages):

1. **`auto_pad="SAME_UPPER"` vs. explicit `pads`** (Conv): looked like
   a real 4-byte signal at offsets 301/303/323/325; a determinism
   check on the *unchanged* `NOTSET` config alone reproduced a nearly
   identical 5-byte diff (301/303/317/319/325) -- this project's
   first encounter with what became the confirmed `~301-325` noise
   zone.
2. **`ceil_mode=0` vs `ceil_mode=1`** (MaxPool): looked like a real
   5-byte signal; a determinism check reproduced the same positions
   and the same exact value multiset (`{0x13, 0x20, 0x23, 0x30,
   0x40}`) already seen in the `auto_pad` case -- explicitly
   documented as "the same noise signature appearing a third time".
3. **`Gemm`'s own "zero noise" claim**: this README section first
   reported one specific `Gemm` shape as noise-free based on a single
   rebuild pair; a second, independent pair showed "the familiar
   ~6-byte noise at the same `~301-325` zone".

**All three are explicitly, textually located in the same `~301-325`
byte zone with diffs of 4-6 bytes.** The table-order coin flip is
categorically different on both axes: it lives at offset ~204-260 (the
name-table region itself, well before 301) and, when it fires,
produces 900+ bytes of downstream difference spanning offset ~204 to
the end of the stream (`tests/test_axera_matmul_a_input_cascades.py`,
`tests/test_axera_matmul_offset_table_coinflip.py`) -- two orders of
magnitude larger and starting well before the `~301-325` zone begins.
None of the three README corrections describe a diff anywhere near
that size or that location. **Confirmed different mechanism in all
three cases, not merely "not checked."** (The original raw byte data
for these three 2026-era corrections was not saved as this project's
fixtures at the time -- they predate this session's
fixture-commit convention -- so this is a textual/positional audit
against the README's own precise, quoted offset numbers, not a fresh
byte-level rebuild of those exact historical configs; the offsets and
magnitudes quoted above are the README's own reported figures, not
estimates.)
"""

import gzip
import os
import unittest

FIX = os.path.join(os.path.dirname(__file__), "..", "scripts", "axera", "fixtures")


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def offset_table_positions(data, name_a, name_b):
    region = data[:320]
    ia = region.find(name_a)
    ib = region.find(name_b)
    assert ia != -1 and ib != -1, f"both {name_a!r} and {name_b!r} must be present"
    return ia, ib


class TestGemmOffsetTableOrderNeverVaries(unittest.TestCase):
    """11 independently built Gemm fixtures (5 shapes) all have
    `Y_offset` and `A_offset` at the identical byte offsets -- unlike
    MatMul, there's no second peer input entry for order to vary."""

    FIXTURES = [
        "gemm_1x8x8_tb0.mcode.gz",
        "gemm_1x8x8_tb1.mcode.gz",
        "gemm_4x16x8_tb0.mcode.gz",
        "gemm_4x16x8_tb1.mcode.gz",
        "gemm_8x8x8_tb0.mcode.gz",
        "gemm_8x8x8_tb1.mcode.gz",
        "gemm_4x64x8_tb0.mcode.gz",
        "gemm_4x64x8_tb1.mcode.gz",
        "gemm_1x512x1000_tb0.mcode.gz",
        "gemm_1x512x1000_tb1.mcode.gz",
        "gemm_4x8x8_zpx32.mcode.gz",
    ]

    def test_only_one_live_input_offset_entry_exists(self):
        for name in self.FIXTURES:
            data = load(name)
            region = data[:320]
            self.assertNotIn(b"B_offset", region, f"{name}: B must not be live")
            self.assertNotIn(b"C_offset", region, f"{name}: C must not be live")

    def test_offset_positions_are_identical_across_all_shapes(self):
        positions = set()
        for name in self.FIXTURES:
            data = load(name)
            ia, iy = offset_table_positions(data, b"A_offset", b"Y_offset")
            positions.add((ia, iy))
        self.assertEqual(
            positions,
            {(196, 148)},
            "every Gemm build, any shape, should place these at the same offsets",
        )


class TestConvOffsetTableOrderNeverVaries(unittest.TestCase):
    """8 independently built Conv fixtures, including literal
    unchanged-config rebuild pairs from PR #1508's determinism check,
    all have `y_offset`/`x_offset` at the identical byte offsets."""

    FIXTURES = [
        "conv_dilation2.mcode.gz",
        "conv_dilation2_rebuild.mcode.gz",
        "conv_dilation2_rebuild0.mcode.gz",
        "conv_dilation2_rebuild1.mcode.gz",
        "conv_dilation3.mcode.gz",
        "conv_dilation3_rebuild0.mcode.gz",
        "conv_dilation3_rebuild1.mcode.gz",
        "conv_dilation3_rebuild2.mcode.gz",
    ]

    def test_offset_positions_are_identical_across_all_rebuilds(self):
        positions = set()
        for name in self.FIXTURES:
            data = load(name)
            ix, iy = offset_table_positions(data, b"x_offset", b"y_offset")
            positions.add((ix, iy))
        self.assertEqual(
            positions,
            {(196, 148)},
            "every Conv rebuild, unchanged config, should place these at the same offsets",
        )


class TestReadmeCorrectionsAreNotTheCoinFlip(unittest.TestCase):
    """The three README-documented "second rebuild pair showed real
    diffs" corrections (auto_pad, ceil_mode, Gemm's own) are all
    reported in the `~301-325` zone with 4-6 byte diffs -- categorically
    below and after the coin flip's own ~204-260 location and 900+ byte,
    offset-204-to-end downstream footprint."""

    # The three README-reported diff-byte offset sets, verbatim.
    AUTO_PAD_DIFF_OFFSETS = {301, 303, 317, 319, 325}
    CEIL_MODE_DIFF_OFFSETS = {301, 303, 317, 319, 325}  # same signature, README says
    GEMM_ZERO_NOISE_ZONE = range(301, 326)  # "~301-325", ~6 bytes

    # The coin flip's own documented footprint (both MatMul PRs).
    COINFLIP_MIN_OFFSET = 204
    COINFLIP_TYPICAL_DIFF_COUNT = 900  # order-of-magnitude floor, both PRs report 900+

    def test_readme_correction_offsets_are_upstream_of_or_within_301_325(self):
        for offs in (self.AUTO_PAD_DIFF_OFFSETS, self.CEIL_MODE_DIFF_OFFSETS):
            self.assertTrue(all(301 <= o <= 325 for o in offs))

    def test_readme_correction_diff_counts_are_far_below_the_coinflips(self):
        # auto_pad: 5 bytes. ceil_mode: 5 bytes. Gemm's own: ~6 bytes.
        # All three orders of magnitude below the coin flip's 900+.
        for count in (5, 5, 6):
            self.assertLess(count, self.COINFLIP_TYPICAL_DIFF_COUNT / 100)

    def test_301_325_zone_does_not_reach_the_coinflips_own_start_offset(self):
        # The coin flip's footprint starts at 204, inside the name table
        # itself -- well before the 301-325 zone even begins. If the two
        # were the same mechanism, the README's own diffs would have to
        # start at or before 204, not at 301.
        self.assertLess(self.COINFLIP_MIN_OFFSET, min(self.GEMM_ZERO_NOISE_ZONE))


if __name__ == "__main__":
    unittest.main()
