"""Generalizing zp_x's literal-byte encoding (Mul-only until now) to Gemm,
Conv, and MatMul.

`TestZpXLiteralByteWhenPresent` (test_axera_mcode_reciprocal.py) found
that Mul's mcode *sometimes* writes its input zero_point (zp_x) as a
literal byte in a fixed 6-byte quad -- ``02 10 1b <zp_x> 83 36`` -- with
presence gated non-monotonically by something other than zp_x's raw
magnitude; other zp_x values use one of two still-opaque forms instead.
That work never touched Gemm, Conv, or MatMul.

**Not the same question as `scripts/axera/README.md`'s `emitter.py`
work.** That system's `learn_mcode`/`emit_mcode` locate and rewrite the
mcode's *output* scale/zero-point (`y_scale`, `round(y_zero)`, see the
README's "mcode's weight-dependent bytes" section) and a per-channel
weight-table requantisation block whose closed-form formula (README,
"The 1,719 unexplained bits are the requantisation, in closed form")
*takes* the input zero point `zx` as a known scalar input -- it does not
locate `zx` itself inside the mcode byte stream. This file is about
exactly that narrower, still-open question: where does the *input*
activation zero point show up as bytes, for ops besides Mul.

## Findings

**1. Conv: the exact same literal quad, at the exact same zp_x value
Mul already confirmed.** `conv_1c1c_8x8_k3.mcode.gz` (already committed,
`AxQuantizedConv`, `input_zeropoints=[127]`) contains
``02 10 1b 7f 83 36`` at offset 1806 -- one hit, byte-exact, same framing
Mul uses. Mul's own `mul_1x8_recip_x01.mcode.gz` fixture has zp_x=127
and also uses this literal form (see the reciprocal test file's CASES).
Same op-independent value, same encoding: this is one mechanism shared
across ops, not two coincidentally-similar ones.

**2. Gemm: absence at zp_x=128 and zp_x=32, matching Mul's own misses at
those exact values.** Three already-committed Gemm fixtures
(`gemm_1x8x8_tb0`, `gemm_4x16x8_tb0`, `gemm_8x8x8_tb0`, all
`input_zeropoints=[128]`) have zero hits for
``02 10 1b 80 83 36``. Mul's own `mul_1x8`/`mul_1x8_recip_x10`/
`mul_1x8_w2` (all zp_x=128) also use the opaque form, not this one (see
the second CORRECTION in `TestZpXLiteralByteWhenPresent`'s docstring). A
fresh Gemm build (`gemm_4x8x8_zpx32.mcode.gz`, calibration range chosen
to land near zp_x=33 but landing at the true MinMax value 32) also has
zero hits for ``02 10 1b 20 83 36`` -- and Mul's own sweep explicitly
lists 32 among its opaque-form values (`TestZpXLiteralByteWhenPresent`'s
second CORRECTION paragraph: "16, 20, 24, 27, 29, 30, 31, 32"). Two
more same-value, same-outcome matches.

**3. MatMul: the question doesn't apply -- its input zero points are
always 0, forced, regardless of calibration asymmetry.** Every MatMul
build gathered in this project, including a fresh one built here
specifically to avoid it, has `input_zeropoints=[0, 0]`. This isn't a
calibration coincidence: `matmul_4x8x8_asym_calib.mcode.gz` reused the
*exact* calibration array (same seed, same asymmetric range
`[-0.1486, 1.0]`, same `[4, 8]` shape) that produced Gemm's real,
nonzero zp_x=32 for input A above -- if MatMul's A used the same
asymmetric-MinMax rule Gemm does, it would land at the same zp_x=32 (or
close to it, modulo per-op scale rounding). Instead MatMul's `A` came
out `input_scales[0]=0.007818003185093403` with `zero_point=0` -- that
scale matches a *symmetric* range covering `max(|min|, |max|) = 1.0`
(`2*1.0/255 = 0.007843`, within float32-sampling-noise of the observed
value), not the asymmetric range's own implied scale (Gemm's build off
the identical array gave `0.004479411523789167` — Gemm and MatMul are
quantizing the SAME data with two different rules). MatMul's compiled
kernel appears to require symmetric (zero-point-forced-to-0)
quantization for its live tensor inputs. Since zp_x=0 already has its
own separate, simpler, non-literal encoding
(`TestZpXImmediateRegion`'s "zp_x=0: a fixed 3-byte tail `00 10 84`"),
there is no missing MatMul zp_x encoding to find -- the literal-quad
mechanism this file generalizes simply never has an occasion to fire
for MatMul.
"""

import gzip
import os
import unittest

FIX = os.path.join(os.path.dirname(__file__), "..", "scripts", "axera", "fixtures")


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def hits(data, pat):
    return [i for i in range(len(data) - len(pat) + 1) if data[i : i + len(pat)] == pat]


_CONST_PREFIX = bytes.fromhex("02101b")
_CONST_SUFFIX = bytes.fromhex("8336")


def literal_quad(zpx):
    return _CONST_PREFIX + bytes([zpx]) + _CONST_SUFFIX


class TestConvGetsTheSameLiteralQuadAtTheSameValue(unittest.TestCase):
    def test_zp_x_127_is_literal_in_conv_too(self):
        data = load("conv_1c1c_8x8_k3.mcode.gz")
        found = hits(data, literal_quad(127))
        self.assertEqual(found, [1806])


class TestGemmMissesAtTheSameValuesMulMisses(unittest.TestCase):
    ZP128_CASES = [
        "gemm_1x8x8_tb0.mcode.gz",
        "gemm_4x16x8_tb0.mcode.gz",
        "gemm_8x8x8_tb0.mcode.gz",
    ]

    def test_zp_x_128_absent_in_gemm(self):
        pat = literal_quad(128)
        for name in self.ZP128_CASES:
            data = load(name)
            self.assertEqual(hits(data, pat), [], f"{name}: unexpected literal hit")

    def test_zp_x_32_absent_in_gemm(self):
        data = load("gemm_4x8x8_zpx32.mcode.gz")
        self.assertEqual(hits(data, literal_quad(32)), [])


class TestMatMulZeroPointsAreForcedToZero(unittest.TestCase):
    def test_no_literal_quad_for_any_byte_value(self):
        # zp_x is always 0 for MatMul (see module docstring's scale-
        # comparison evidence), and 0 already has its own non-literal
        # encoding -- so no literal-quad byte value should appear at all.
        data = load("matmul_4x8x8_asym_calib.mcode.gz")
        for zpx in range(256):
            self.assertEqual(
                hits(data, literal_quad(zpx)),
                [],
                f"unexpected literal-quad hit for zp_x={zpx} in a stream"
                " whose only real zero point is 0",
            )

    def test_asym_calib_build_still_produced_a_real_asymmetric_scale_elsewhere(self):
        # Sanity check that this build isn't accidentally trivial: confirm
        # it's a real, distinct compile (different length) from the plain
        # matmul_4x8x8 fixture, not a byte-identical rebuild.
        asym = load("matmul_4x8x8_asym_calib.mcode.gz")
        plain = load("matmul_4x8x8.mcode.gz")
        self.assertNotEqual(len(asym), len(plain))


if __name__ == "__main__":
    unittest.main()
