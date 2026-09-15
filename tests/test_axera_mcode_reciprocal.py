"""Input-side reciprocal slots in single-Mul mcode streams.

Discovery (factorial sweep over Mul[1,8] input ranges, 2026-09-15): the NPU
quantizes float inputs with an integer multiplier derived from the
*reciprocal* of each input's MinMax scale, stored as float32 in the stream:

- Site A (~offset 1122, 4 copies at stride 8): float32(1 / x_scale).
- Site B (~offset 1246, 4 copies at stride 7): float32(1 / y_scale), except
  when the S-unit program takes its short form (base build), where only the
  low 3 bytes of the float appear followed by an 0x82 tag byte and the unit
  shrinks to stride 6.  The S-unit program shape adapts to operand magnitude.

Verified values (quant JSON scales -> stream bytes):

- base (x=0.00779978, y=0.00768933): site A = 128.2088 = 1/x (x4, stride 8);
  site B carries de0c02 = low 3 bytes of float32(1/y) = float32(130.05).
- x10y01 (x=0.07799776, y=0.00076893, z identical to base): site A = 12.8209
  = 1/x (x4, stride 8); site B = 1300.50 = 1/y (x4, stride 7).
- x01y10 (x=0.00077252, y=0.07746022): site A = 1294.47 = 1/x (x4, stride 8);
  site B = 12.91 = 1/y (x4, stride 7).

Open (correlated but undecoded): the 3-byte groups preceding each 42a100
marker (4 copies): base 7c7cee, x10y01 7b7cee (byte0 - 1 for 10x x with z
fixed), mul_sep (x=0.78) 7a7cee (-2 for 100x), x01y10 9b53c3 (all bytes
change when z also moves).  They track the input scales but are not a pure
function of any single scale; pinned here as observations.

Update (2026-09-16): SOLVED -- those "3-byte groups" were a misaligned
read. The real structure is a third reciprocal-family slot:

- Site C (~offset 1421, 4 copies at stride 8, framed `0f <f32> a1 00 <id>`
  like site A): float32(z_scale / (x_scale * y_scale)), the integer
  requant multiplier, bit-exact in all five scratch builds (base, x10y01,
  x01y10, sep, w2 -- including w2's apparent "8 markers", which were just
  its site-A slots matching the old `42 a1 00` motif by float-byte
  coincidence when 1/x fell in [32, 64)). The old motif conflated two
  different sites sharing one frame: site A carries the input reciprocal
  (high byte varies: 43/42/41/44/3f), site C the requant (high byte 42
  in every build so far). `TestRequantMultiplierSlot` pins site C.
"""

import gzip
import os
import struct
import unittest

FIX = os.path.join(os.path.dirname(__file__), "..", "scripts", "axera", "fixtures")

CASES = {
    # fixture: (x_scale, y_scale, siteB_full_float, group_hex)
    "mul_1x8.mcode.gz": (0.007799775805324316, 0.0076893349178135395, False, "7c7cee"),
    "mul_1x8_recip_x10.mcode.gz": (
        0.07799775898456573,
        0.0007689335034228861,
        True,
        "7b7cee",
    ),
    "mul_1x8_recip_x01.mcode.gz": (
        0.0007725197938270867,
        0.07746022194623947,
        True,
        "9b53c3",
    ),
}

# Output scales for the same fixtures (quant JSON, pinned alongside the
# inputs so site C is computable without a rebuild).
_Z_SCALES = {
    "mul_1x8.mcode.gz": 0.007151617668569088,
    "mul_1x8_recip_x10.mcode.gz": 0.007151617202907801,
    "mul_1x8_recip_x01.mcode.gz": 0.005844127852469683,
}

# Unary op (Neg): same site-A mechanism for its single input.  All 13 builds
# of the Neg input-range sweep carry float32(1/x) x4 at stride 8 (offsets
# 1067/1068 family); pinned here for the neg_1x8 fixture (x=0.00768933).
NEG_CASES = {
    "neg_1x8.mcode.gz": 0.0076893349178135395,
}


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def hits(data, pat):
    return [i for i in range(len(data) - len(pat) + 1) if data[i : i + len(pat)] == pat]


class TestInputReciprocalSlots(unittest.TestCase):
    def test_site_a_carries_inverse_x_scale(self):
        for name, (xs, _ys, _full, _grp) in CASES.items():
            data = load(name)
            pat = struct.pack("<f", 1.0 / xs)
            found = hits(data, pat)
            self.assertEqual(len(found), 4, f"{name}: site A hits for 1/x={1.0 / xs}")
            strides = {b - a for a, b in zip(found, found[1:])}
            self.assertEqual(strides, {8}, f"{name}: site A stride")
        for name, xs in NEG_CASES.items():
            data = load(name)
            pat = struct.pack("<f", 1.0 / xs)
            found = hits(data, pat)
            self.assertEqual(len(found), 4, f"{name}: unary site A hits")
            strides = {b - a for a, b in zip(found, found[1:])}
            self.assertEqual(strides, {8}, f"{name}: unary site A stride")

    def test_site_b_carries_inverse_y_scale(self):
        for name, (_xs, ys, full, _grp) in CASES.items():
            data = load(name)
            pat = struct.pack("<f", 1.0 / ys)
            found = hits(data, pat)
            if full:
                self.assertEqual(
                    len(found), 4, f"{name}: site B hits for 1/y={1.0 / ys}"
                )
                strides = {b - a for a, b in zip(found, found[1:])}
                self.assertEqual(strides, {7}, f"{name}: site B stride")
            else:
                # Short S-unit form: low 3 bytes of the float, 0x82 tag.
                self.assertGreaterEqual(
                    len(hits(data, pat[:3])), 3, f"{name}: site B short form"
                )

    def test_group_values_pinned(self):
        marker = bytes.fromhex("42a100")
        for name, (_xs, _ys, _full, grp) in CASES.items():
            data = load(name)
            idx = hits(data, marker)
            self.assertEqual(len(idx), 4, f"{name}: marker count")
            for i in idx:
                self.assertEqual(data[i - 3 : i].hex(), grp, f"{name}: group")


class TestRequantMultiplierSlot(unittest.TestCase):
    """Site C: float32(z_scale / (x_scale * y_scale)) x4 at stride 8.

    The integer requant multiplier, bit-exact -- the value the NPU's
    fixed-point core multiplies the (x_q - zpx)(y_q - zpy) product by
    before shifting down to the output scale. Same `0f <f32> a1 00 <id>`
    framing as site A (verified context bytes), different payload.
    """

    def test_site_c_carries_output_over_input_product_scale(self):
        for name, (xs, ys, _full, _grp) in CASES.items():
            data = load(name)
            req = struct.pack("<f", _Z_SCALES[name] / (xs * ys))
            found = hits(data, req)
            self.assertEqual(len(found), 4, f"{name}: site C hits")
            strides = {b - a for a, b in zip(found, found[1:])}
            self.assertEqual(strides, {8}, f"{name}: site C stride")


if __name__ == "__main__":
    unittest.main()


# NOTE: this class lives after the __main__ guard on purpose -- the
# stacked site-C diff edits the docstring and appends its own class
# above, and keeping this addition textually disjoint avoids a merge
# collision between the two PRs.
class TestOutputScaleQuads(unittest.TestCase):
    """Output scale quads: `03 <float32(z_scale)> 81 82` x4 at stride 7.

    The output-side counterpart to the input slot families: four copies
    of the exact float32 output scale, each framed by a 0x03 lead byte
    and constant 0x81/0x82 tag bytes (verified on all three fixtures).
    Two things this settles:

    - The `81` tag is NOT the output zero point: x01's z zero point is
      110 (0x6e) yet its quads still read `... 81 82`. Raw zp immediates
      (128/127/126/129/110 census over the program region) and zp*scale
      combined floats were also ruled out across five builds -- zero
      points live elsewhere (still open).
    - Emitter caution: base and x10 carry the same real-valued z scale
      to ~1e-9 yet their quad floats differ by 1 ULP (1e vs 1d) -- the
      two builds' float64 scales straddle a float32 boundary. Patch by
      exact float32 word, never by recomputing from float64.
    """

    _Z = {
        "mul_1x8.mcode.gz": 0.007151617668569088,
        "mul_1x8_recip_x10.mcode.gz": 0.007151617202907801,
        "mul_1x8_recip_x01.mcode.gz": 0.005844127852469683,
    }

    def test_quads_carry_exact_float32_output_scale(self):
        for name, zs in self._Z.items():
            data = load(name)
            pat = struct.pack("<f", zs)
            found = hits(data, pat)
            self.assertEqual(len(found), 4, f"{name}: quad count")
            strides = {b - a for a, b in zip(found, found[1:])}
            self.assertEqual(strides, {7}, f"{name}: quad stride")
            for i in found:
                self.assertEqual(data[i - 1], 0x03, f"{name}@{i}: lead")
                self.assertEqual(data[i + 4 : i + 6].hex(), "8182", f"{name}@{i}: tags")
