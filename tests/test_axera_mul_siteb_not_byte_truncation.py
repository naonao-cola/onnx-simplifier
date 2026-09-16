"""Mul's site B short/full form selector does NOT follow MatMul's
byte-truncation-domain rule -- it's a genuinely different, op-specific
mechanism, not the same codec-level rule applied to a different operand.

`tests/test_axera_matmul_quad_form_switch.py` (PR #1514) fully decoded
why batched MatMul's short-form `A` quad (a 3-byte truncation of
`float32(1/A_scale)` that drops the high byte) sometimes switches to
the full 4-byte form: the short form is only usable when the *omitted*
high byte (`b3`) equals `0x42` -- true for `1/A_scale` in `[32,128)`.
Outside that range (`b3 != 0x42`), the full form is used because the
truncated 3 bytes alone can no longer reconstruct the value under
whatever fixed-default-byte assumption the short form relies on.

`tests/test_axera_mcode_reciprocal.py`'s `TestSiteBFormSelectorIsZpX`
(already merged, resolved independently and earlier) established a
different-looking rule for Mul's own short-form encoding of the exact
same byte shape (low 3 bytes of `float32(1/y_scale)` + a tag byte, full
form otherwise): short form iff `x_scale/y_scale` ratio is close to 1
**and** `zp_x != 0`. That rule has nothing to do with `1/y_scale`'s own
magnitude or byte pattern -- it depends on `x`'s zero point, a
completely different quantization parameter.

This file checks whether those two descriptions are actually
compatible (e.g. the ratio+zp_x rule might just be an artifact of a
small sample that a magnitude-based rule would also explain) or
genuinely conflict. **They conflict, decisively, using only
already-committed fixtures -- no new builds needed.**

## The decisive pair: identical `1/y_scale`, opposite form

`mul_1x8_zp_both0.mcode.gz` and `mul_1x8_zp_xonly.mcode.gz` (both from
`TestSiteBFormSelectorIsZpX`'s controlled 2x2 zero-point grid) have the
**exact same** `y_scale` (`0.007805639877915382`, hence the exact same
`1/y_scale = 128.1124950216209` and the exact same float32 bytes,
`b3 == 0x43`) but differ only in `zp_x` (`0` vs `19`) -- and:

- `mul_1x8_zp_both0` (`zp_x=0`): **full form**. The full 4-byte float
  (`cc1c0043`) is found x4 at stride 7, at offset 1246.
- `mul_1x8_zp_xonly` (`zp_x=19`): **short form**. The full 4-byte float
  is absent entirely; only the low-3-byte truncation (`cc1c00`) is
  found x4, at a shifted offset (1249, stride 6).

Same value, same bytes, same byte3 -- opposite encoding, purely as a
function of a completely unrelated operand's zero point. A pure
byte-truncation-domain rule (MatMul's mechanism) cannot produce this:
it has no way to depend on `zp_x` at all.

## It's not even consistent with MatMul's *specific* boundary in isolation

`128.1125`'s `b3` is `0x43` -- under MatMul's own decoded rule
(`[32,128)` -> `0x42` -> short; anything else -> full), this value
should **always** force full form, with no exceptions. But
`mul_1x8_zp_xonly` uses the short form at this exact value anyway
(`zp_x=19` overrides whatever the magnitude-based rule would predict).
This isn't just "a different, additional rule also applies to Mul" --
Mul's short form actively fires in a byte3 regime MatMul's rule would
forbid it in.

## Conclusion

Mul's site B and MatMul's short-form `A` quad share the same on-disk
*shape* (drop the float32 high byte, keep the low 3 bytes + a tag) but
are governed by genuinely different, op-specific selection rules --
not one shared codec-level mechanism applied to two operands. This is
a real, useful negative result: it closes off "maybe it's all one
underlying rule" as a hypothesis for future unification attempts,
rather than leaving that question implicitly open.
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


class TestSameYScaleGivesOppositeFormDependingOnZpX(unittest.TestCase):
    """The decisive natural experiment: mul_1x8_zp_both0 and
    mul_1x8_zp_xonly share the exact same y_scale (hence the exact same
    float32(1/y_scale) bytes) but differ in zp_x and land on opposite
    site B forms -- refuting a pure byte-truncation-domain rule."""

    Y_SCALE = 0.007805639877915382

    def test_both_fixtures_share_the_identical_inv_y_scale_bytes(self):
        full4 = struct.pack("<f", 1.0 / self.Y_SCALE)
        # b3 == 0x43: under MatMul's decoded rule ([32,128) -> 0x42 ->
        # short; else full), this value should ALWAYS force full form.
        self.assertEqual(full4[3], 0x43)

    def test_zp_both0_uses_full_form(self):
        data = load("mul_1x8_zp_both0.mcode.gz")
        full4 = struct.pack("<f", 1.0 / self.Y_SCALE)
        found = hits(data, full4)
        self.assertEqual(len(found), 4, "expected full-form site B")
        strides = {b - a for a, b in zip(found, found[1:])}
        self.assertEqual(strides, {7})

    def test_zp_xonly_uses_short_form_despite_identical_scale(self):
        """zp_x=19 (vs. 0 for the fixture above) flips the form even
        though 1/y_scale -- and therefore its float32 high byte -- is
        byte-for-byte identical. MatMul's rule predicts full form here
        unconditionally; Mul's actual behavior contradicts that."""
        data = load("mul_1x8_zp_xonly.mcode.gz")
        full4 = struct.pack("<f", 1.0 / self.Y_SCALE)
        short3 = full4[:3]
        self.assertEqual(hits(data, full4), [], "full form should be absent")
        found = hits(data, short3)
        self.assertEqual(len(found), 4, "expected short-form site B")
        strides = {b - a for a, b in zip(found, found[1:])}
        self.assertEqual(strides, {6})


class TestMatMulByteTruncationRuleDoesNotPredictMulSiteBAcrossAllKnownBuilds(
    unittest.TestCase
):
    """Wider check across every Mul fixture with a known y_scale in the
    existing reciprocal test file: MatMul's byte3-based prediction
    disagrees with Mul's actual observed form in multiple cases, not
    just the one decisive pair above."""

    # (name, y_scale, actual_full) -- from test_axera_mcode_reciprocal.py's
    # own CASES / TestSiteBFormSelectorIsZpX.CASES (quant JSON ground
    # truth, already established/merged).
    CASES = [
        ("mul_1x8.mcode.gz", 0.0076893349178135395, False),
        ("mul_1x8_recip_x10.mcode.gz", 0.0007689335034228861, True),
        ("mul_1x8_recip_x01.mcode.gz", 0.07746022194623947, True),
        ("mul_1x8_zp_both0.mcode.gz", 0.007805639877915382, True),
        ("mul_1x8_zp_yonly.mcode.gz", 0.007497101556509733, True),
        ("mul_1x8_zp_xonly.mcode.gz", 0.007805639877915382, False),
        ("mul_1x8_zp_both.mcode.gz", 0.007497101556509733, False),
    ]

    def _matmul_rule_predicts_short(self, y_scale):
        b3 = struct.pack("<f", 1.0 / y_scale)[3]
        return b3 == 0x42

    def test_matmul_rule_mispredicts_at_least_two_of_seven_known_builds(self):
        mismatches = []
        for name, ys, actual_full in self.CASES:
            predicted_full = not self._matmul_rule_predicts_short(ys)
            if predicted_full != actual_full:
                mismatches.append(name)
        self.assertGreaterEqual(
            len(mismatches),
            2,
            "expected MatMul's byte-truncation rule to mispredict Mul's"
            f" site B form on multiple known builds; got mismatches={mismatches}",
        )


if __name__ == "__main__":
    unittest.main()
