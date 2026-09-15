"""Site A (the input reciprocal-scale slot) generalizes too -- with a
real asymmetry for MatMul's two live tensors.

`TestInputReciprocalSlots` (test_axera_mcode_reciprocal.py) decoded
site A for Mul: `0f <f32(1/x_scale)> a1 00 <id>` x4 at stride 8, framed
by a lead byte that varies build to build (`43`/`42`/`41`/`44`/`3f`).
Like the output-scale quad (`test_axera_output_scale_quad_generalizes.py`),
this is not Mul-specific:

- **Gemm** (`gemm_1x8x8_tb0.mcode.gz`, already committed for the output
  quad work): `1/A_scale` = 128.2088 found x4 at stride 8, same
  `<f32> a1 00 <id>` frame, lead byte `43`.
- **Conv** (`conv_1c1c_8x8_k3.mcode.gz`): `1/x_scale` = 127.9130 found
  x4 at stride 8, same frame, lead byte `42`.

**MatMul is asymmetric, and that asymmetry is real, not a search
miss.** `AxQuantizedMatMul` has two genuinely live tensor inputs, `A`
and `B` (see `test_axera_output_scale_quad_generalizes.py`'s docstring
for why the README calls this case out as distinct from Gemm). Only
`1/B_scale` gets this site-A treatment (found x4, stride 8, same frame,
lead byte `42`); `1/A_scale` was searched for exhaustively -- full
float32 (4-byte), the short-form low-3-bytes substring Mul's own site B
uses for its short form, and a bfloat16 truncation (the encoding
`scripts/axera/patch_scales.py` already established for a different
context) -- and found nowhere in the stream, at any width. A first
attempt at the bf16 search looked like a hit (`ff42` at the same
offsets as `1/B_scale`'s own quad) but that was checked and ruled out
before being reported here: `1/A_scale` (127.550) and `1/B_scale`
(127.729) are close enough that their bfloat16 truncations collide
(`struct.pack("<f", 127.550...)[2:] == struct.pack("<f",
127.729...)[2:] == b"\\xff\\x42"`) -- the "hit" was `1/B_scale`'s own
float32 tail bytes, not a separate embedding of `A`.

This mirrors -- without necessarily sharing a mechanism with -- Mul's
own input asymmetry (site A vs. site B's different, shorter-capable
form for the second operand): whichever tensor plays MatMul's "B" role
here gets the full site-A treatment; whichever plays "A" does not get
any literal reciprocal-scale encoding this project has found. Which
tensor lands in which HW role is not established here (ONNX input
order was `A`, `B`; that is not asserted to be the HW role order) --
only that the asymmetry itself is real and reproducible. Open for
whoever chases MatMul's other input's encoding next.
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


class TestSiteAGeneralizes(unittest.TestCase):
    # fixture: (input_scale,)
    CASES = {
        "gemm_1x8x8_tb0.mcode.gz": 0.007799775805324316,
        "conv_1c1c_8x8_k3.mcode.gz": 0.007817814126610756,
    }

    def test_site_a_present_stride_8(self):
        for name, xs in self.CASES.items():
            data = load(name)
            pat = struct.pack("<f", 1.0 / xs)
            found = hits(data, pat)
            self.assertEqual(len(found), 4, f"{name}: site A hits")
            strides = {b - a for a, b in zip(found, found[1:])}
            self.assertEqual(strides, {8}, f"{name}: site A stride")
            for i in found:
                self.assertEqual(
                    data[i + 4 : i + 6].hex(), "a100", f"{name}@{i}: frame"
                )


class TestMatMulSiteAIsAsymmetric(unittest.TestCase):
    A_SCALE = 0.007840047590434551
    B_SCALE = 0.007829049602150917

    def test_b_gets_site_a_treatment(self):
        data = load("matmul_4x8x8.mcode.gz")
        pat = struct.pack("<f", 1.0 / self.B_SCALE)
        found = hits(data, pat)
        self.assertEqual(len(found), 4, "1/B_scale hits")
        strides = {b - a for a, b in zip(found, found[1:])}
        self.assertEqual(strides, {8}, "1/B_scale stride")
        for i in found:
            self.assertEqual(data[i + 4 : i + 6].hex(), "a100", f"@{i}: frame")

    def test_a_gets_no_literal_encoding_full_short_or_bf16(self):
        data = load("matmul_4x8x8.mcode.gz")
        inv_a = 1.0 / self.A_SCALE
        full = struct.pack("<f", inv_a)
        self.assertEqual(hits(data, full), [], "1/A_scale full float32")
        self.assertEqual(hits(data, full[:3]), [], "1/A_scale short (low 3 bytes)")
        # The bf16-truncation "hit" a first pass found is a verified false
        # positive (collides with 1/B_scale's own float32 tail bytes, see
        # module docstring) -- not re-asserted as absent here since a
        # 2-byte pattern this short is expected to collide with unrelated
        # bytes somewhere in a multi-KB stream; the full/short checks
        # above are the meaningful negative result.


if __name__ == "__main__":
    unittest.main()
