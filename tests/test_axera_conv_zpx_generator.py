"""Conv's first zp_x mcode generation/patching work: extending
``tiny_emit.patch_mul_zp_x`` (previously Mul-only) to Conv, verified
against a real second build -- not just structural assertions.

`tests/test_axera_zpx_generalizes.py` (merged) confirmed Conv's mcode
uses the identical literal ``02 10 1b <zp_x> 83 36`` unit Mul's own
``patch_mul_zp_x`` already patches, but no patch function or hardware/
build-level verification was ever written for Conv specifically --
that gap is what this file closes, and what `tiny_emit.patch_conv_zp_x`
now provides.

## What's confirmed: the unit patches correctly, isolated

Three Conv builds at the same shape (`cin=cout=1`, `hw=8`, `k=3`,
matching `conv_1c1c_8x8_k3.mcode.gz`'s own shape), same 3016-byte mcode
length, all landing in the literal zp_x form:

- `conv_1c1c_8x8_k3_zpx79.mcode.gz` (`x_scale=0.005695836152881384`,
  `zp_x=79`)
- `conv_1c1c_8x8_k3_zpx49_samescale.mcode.gz` -- built with a shifted
  (not widened) calibration window so its `x_scale` is **bit-identical**
  to the build above, only `zp_x` (49) differs
- `conv_1c1c_8x8_k3_zpx79_rebuild.mcode.gz` -- an independent rebuild
  of the first config, unchanged, establishing this shape's own
  rebuild noise floor: **3 bytes** (offsets 680, 686, 696)

`patch_conv_zp_x(zpx79_data, 79, 49)` writes the identical 6-byte unit
(`02 10 1b 31 83 36`) at the identical offset (1806) the real
`zpx49_samescale` build carries -- confirmed byte-for-byte, not just
"a value was written somewhere."

## What's newly found, not just re-confirmed: patching this unit alone
## does not reproduce a real second build

Diffing the patched stream against the real `zpx49_samescale` build
(same x_scale, so site A/output-quad differences are not a confound
the way they would be across different scales) leaves **25 bytes**
differing -- of which only 3 (680, 686, 696) match this shape's own
established rebuild-noise floor above. **The remaining 22 bytes**
(533, 681, 683-685, 687, 688, 690, 981, 2043-2076) are real,
reproducible content that depends on zp_x beyond the one 6-byte unit
this project knows how to write -- confirmed above the noise floor,
not chased further here (offsets, not semantics).

This is a genuine, previously-untested limitation, not a re-statement
of the module docstring's existing z-scale caveat: it was checked here
specifically for zp_x, holding x_scale bit-identical between the two
builds (the cleanest isolation this project's own established
same-scale-different-window calibration technique can produce), and
the answer is that zp_x alone still perturbs ~20 bytes this project
has not decoded. `patch_conv_zp_x` should not be read as reproducing a
real rebuild bit-exactly the way Mul's *scale*-family patches were
hardware-confirmed to (`tests/test_axera_mul_emit_hardware.py`) --
that bar has not been met for zp_x, for either op.
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


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


class TestPatchConvZpXWritesTheCorrectUnit(unittest.TestCase):
    """patch_conv_zp_x(A, 79, 49) writes the identical 6-byte literal
    unit, at the identical offset, that a real zp_x=49 build carries --
    confirmed against real bytes, not a structural assertion."""

    def test_patched_unit_matches_the_real_build_exactly(self):
        a = load("conv_1c1c_8x8_k3_zpx79.mcode.gz")
        b = load("conv_1c1c_8x8_k3_zpx49_samescale.mcode.gz")
        self.assertEqual(len(a), len(b))

        patched = tiny_emit.patch_conv_zp_x(a, 79, 49)
        self.assertEqual(len(patched), len(a))

        unit_49 = bytes.fromhex("02101b") + bytes([49]) + bytes.fromhex("8336")
        pos_b = b.find(unit_49)
        self.assertNotEqual(pos_b, -1, "real build should carry zp_x=49's literal unit")
        self.assertEqual(patched[pos_b : pos_b + 6], unit_49)
        self.assertEqual(patched[pos_b : pos_b + 6], b[pos_b : pos_b + 6])

    def test_patched_stream_round_trips_through_check(self):
        a = load("conv_1c1c_8x8_k3_zpx79.mcode.gz")
        patched = tiny_emit.patch_conv_zp_x(a, 79, 49)
        self.assertEqual(mcode.check(patched), [])

    def test_raises_when_the_reference_does_not_use_the_literal_form(self):
        # zp_x=0 is a separate, non-literal encoding (TestZpXImmediateRegion);
        # patch_conv_zp_x must refuse rather than silently doing nothing.
        a = load("conv_1c1c_8x8_k3_zpx79.mcode.gz")
        with self.assertRaises(ValueError):
            tiny_emit.patch_conv_zp_x(a, 0, 1)


class TestPatchingZpXAloneDoesNotReproduceARealRebuild(unittest.TestCase):
    """A genuine new finding, not a re-confirmation: even with x_scale
    held bit-identical between reference and target, patching only the
    6-byte zp_x unit leaves ~20 bytes of real, above-noise-floor content
    elsewhere in the stream unexplained."""

    NOISE_FLOOR_OFFSETS = {680, 686, 696}

    def test_rebuild_noise_floor_is_three_bytes(self):
        a = load("conv_1c1c_8x8_k3_zpx79.mcode.gz")
        rebuild = load("conv_1c1c_8x8_k3_zpx79_rebuild.mcode.gz")
        self.assertEqual(len(a), len(rebuild))
        diffs = {i for i in range(len(a)) if a[i] != rebuild[i]}
        self.assertEqual(diffs, self.NOISE_FLOOR_OFFSETS)

    def test_patched_stream_leaves_above_noise_bytes_unexplained(self):
        a = load("conv_1c1c_8x8_k3_zpx79.mcode.gz")
        b = load("conv_1c1c_8x8_k3_zpx49_samescale.mcode.gz")
        patched = tiny_emit.patch_conv_zp_x(a, 79, 49)

        diffs = {i for i in range(len(patched)) if patched[i] != b[i]}
        above_noise = diffs - self.NOISE_FLOOR_OFFSETS
        # The zp_x unit's own 6 bytes are, by construction, no longer a
        # diff (patch_conv_zp_x wrote them to match). What remains is
        # real, above-noise content this function does not touch.
        self.assertEqual(len(diffs), 25, f"total diff count changed: {sorted(diffs)}")
        self.assertEqual(
            len(above_noise),
            22,
            f"above-noise-floor diff count changed: {sorted(above_noise)}",
        )


if __name__ == "__main__":
    unittest.main()
