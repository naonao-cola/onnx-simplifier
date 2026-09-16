"""Gemm and Conv's site A slot has no short form at all -- neither
known trigger mechanism (MatMul's byte-truncation-domain rule, Mul's
zp_x!=0 rule) applies, even combined.

`tests/test_axera_site_a_generalizes.py` (merged) confirmed Gemm and
Conv both carry site A (`<f32(1/x_scale)> a1 00 <id>` x4, stride 8) in
its *full* 4-byte form, but only checked one scale value each and
never asked whether a short form exists at all. This project has since
found TWO different, unrelated short-form mechanisms for two other
ops' analogous slots:

- Mul's site B (`tests/test_axera_mcode_reciprocal.py`'s
  `TestSiteBFormSelectorIsZpX`, `tests/test_axera_mul_siteb_not_byte_truncation.py`
  PR #1518): short form iff `x_scale/y_scale` ratio is close to 1 AND
  `zp_x != 0`.
- Batched MatMul's `A` (`tests/test_axera_matmul_quad_form_switch.py`,
  PR #1514): short form iff `1/A_scale`'s float32 high byte equals
  `0x42` (value in `[32,128)`) -- a pure byte-truncation-domain rule,
  independent of zero points entirely.

## Result: neither rule -- nor their combination -- ever produces a
short form for Gemm/Conv's site A

**Already-decisive without new builds**: the existing
`conv_1c1c_8x8_k3.mcode.gz` fixture (`test_axera_site_a_generalizes.py`'s
own `1/x_scale = 127.913`) has float32 high byte `0x42` -- squarely
inside MatMul's `[32,128)` short-form-triggering range -- yet that
already-merged test asserts (and this file re-confirms) the FULL
4-byte form is present, with the standard `a1 00 <id>` frame directly
following it. If Conv's site A followed MatMul's rule, this exact,
already-committed fixture would have to be short-form; it isn't.
`gemm_1x8x8_tb0.mcode.gz`'s own `1/A_scale = 128.209` (high byte
`0x43`, just outside the range) is full-form too, consistent either
way but not itself decisive on its own.

**New builds combine both known trigger conditions simultaneously**,
to rule out the possibility that either mechanism needs a Gemm/Conv-
specific variant not yet tried: built Conv and Gemm each with an
asymmetric input calibration range chosen to land `1/x_scale`
comfortably inside `[32,128)` (byte3 `0x42`, MatMul's trigger) *and*
produce a nonzero input zero point (Mul's trigger) at the same time:

| op | `1/x_scale` | high byte | `zp_x` | form |
| --- | --- | --- | --- | --- |
| Conv | `62.397` | `0x42` | `5` | **full** |
| Gemm | `62.279` | `0x42` | `6` | **full** |

Both still show the full 4-byte form, with the identical `a1 00 <id>`
frame this project has seen at every other Gemm/Conv site A instance.
Neither MatMul's byte-truncation rule nor Mul's zp_x rule (nor some
hybrid requiring both at once) governs Gemm/Conv's site A slot.

## Conclusion

The short-form mechanism is not universal across every op with a site
A/site-A-shaped slot -- it's present for Mul (its own op-specific
ratio+zp_x rule) and for MatMul (a different, pure byte-truncation
rule), but genuinely absent for Gemm and Conv, at least across the
value ranges tested here (both known trigger conditions, individually
and combined). Whether Gemm/Conv's site A ever switches under some
other, not-yet-tested condition remains open -- but the two mechanisms
this project has actually decoded elsewhere do not transfer.
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


class TestExistingConvFixtureAlreadyRefutesByteTruncationRule(unittest.TestCase):
    """conv_1c1c_8x8_k3's own 1/x_scale has high byte 0x42 -- MatMul's
    rule would force short form here. It doesn't happen."""

    X_SCALE = 0.007817814126610756

    def test_high_byte_is_0x42(self):
        full = struct.pack("<f", 1.0 / self.X_SCALE)
        self.assertEqual(full[3], 0x42)

    def test_full_form_is_present_anyway(self):
        data = load("conv_1c1c_8x8_k3.mcode.gz")
        full = struct.pack("<f", 1.0 / self.X_SCALE)
        found = hits(data, full)
        self.assertEqual(len(found), 4, "full-form site A hits")
        strides = {b - a for a, b in zip(found, found[1:])}
        self.assertEqual(strides, {8})
        for i in found:
            self.assertEqual(data[i + 4 : i + 6].hex(), "a100", f"@{i}: frame")


class TestCombinedTriggerConditionsStillGiveFullFormConv(unittest.TestCase):
    """Conv, 1/x_scale=62.397 (high byte 0x42, MatMul's trigger) AND
    zp_x=5 (nonzero, Mul's trigger), combined -- still full form."""

    X_SCALE = 0.016026519238948822
    ZP_X = 5

    def test_high_byte_and_zp_meet_both_known_trigger_conditions(self):
        full = struct.pack("<f", 1.0 / self.X_SCALE)
        self.assertEqual(full[3], 0x42)
        self.assertNotEqual(self.ZP_X, 0)

    def test_full_form_present_short_form_absent(self):
        data = load("conv_sitea_zpx_inrange.mcode.gz")
        full = struct.pack("<f", 1.0 / self.X_SCALE)
        found = hits(data, full)
        self.assertEqual(len(found), 4, "full-form site A hits")
        strides = {b - a for a, b in zip(found, found[1:])}
        self.assertEqual(strides, {8})
        for i in found:
            self.assertEqual(data[i + 4 : i + 6].hex(), "a100", f"@{i}: frame")


class TestCombinedTriggerConditionsStillGiveFullFormGemm(unittest.TestCase):
    """Gemm, 1/A_scale=62.279 (high byte 0x42, MatMul's trigger) AND
    zp_A=6 (nonzero, Mul's trigger), combined -- still full form."""

    A_SCALE = 0.01605680026113987
    ZP_A = 6

    def test_high_byte_and_zp_meet_both_known_trigger_conditions(self):
        full = struct.pack("<f", 1.0 / self.A_SCALE)
        self.assertEqual(full[3], 0x42)
        self.assertNotEqual(self.ZP_A, 0)

    def test_full_form_present_short_form_absent(self):
        data = load("gemm_sitea_zpx_inrange.mcode.gz")
        full = struct.pack("<f", 1.0 / self.A_SCALE)
        found = hits(data, full)
        self.assertEqual(len(found), 4, "full-form site A hits")
        strides = {b - a for a, b in zip(found, found[1:])}
        self.assertEqual(strides, {8})
        for i in found:
            self.assertEqual(data[i + 4 : i + 6].hex(), "a100", f"@{i}: frame")


if __name__ == "__main__":
    unittest.main()
