"""The output-scale quad is a universal S-unit mechanism, not op-specific.

`TestOutputScaleQuads` (test_axera_mcode_reciprocal.py) decoded this
family for Mul: `03 <f32(z_scale)> 81 <tag2>` x4 at stride 7.
`TestGemmOutputScaleQuad` (test_axera_gemm_output_quad.py) found the
same family for Gemm, framed `05 50 0f <f32(output_scale)> 81 <tag2>
03` instead -- the `05 50 0f` lead-in matching, by name, the general
mcode corpus's own long-standing, never-attributed "unanchored
companion-shaped writes" (`scripts/axera/README.md`'s "A five-byte form
that programs its pair twice").

This file checks two more, structurally unrelated ops for the same
family, and finds it in both -- confirming this is a general AX650
mechanism present on every quantized op probed so far (Mul, Gemm, Conv,
MatMul), not something specific to the MAC-engine ops or to elementwise
math:

- **Conv** (`AxQuantizedConv`, a single 1x1-channel 3x3 conv, previously
  undecoded despite substantial prior investigation in this README --
  `scripts/axera/README.md`'s "A first real crack at
  `AxQuantizedConv`'s command encoding" section located a *different*,
  still-undecoded 3-byte periodic field near offset 2561 and a
  *different*, already-decoded bfloat16 `y_scale` family near offset
  1849 -- neither overlaps this quad, found here for the first time,
  at offset ~2053 in this specific small build): same `05 50 0f <f32>
  81 <tag2> 03` frame, tag2 = `0x8e`.
- **MatMul** (`AxQuantizedMatMul`, two genuinely live tensors -- no
  Gemm-style constant weight parameter at all, and symmetric S8
  quantization with zp=0 on both inputs rather than Gemm's asymmetric
  U8 -- the README calls this case out as distinct from Gemm precisely
  because of that): same frame again, tag2 = `0x70`.

The `tag2` byte (`0x6e`/`0x72` for the two Gemm shapes, `0x8e` for
Conv, `0x70` for MatMul) is confirmed build-specific, not a shared
constant -- consistent with the same lesson already learned for Mul's
own quad (`TestOutputScaleQuads`) and Gemm's (`TestGemmOutputScaleQuad`):
don't over-assert it. The fourth copy's tail byte also gets a high bit
set (`0x03` -> `0x83`) in every op checked here, the same as the other
two families.

Practical upshot: any future op-coverage probe that needs an output
scale's mcode offset can now search for this exact frame first, before
assuming a new op needs its own from-scratch byte hunt -- three of four
ops checked across this project's whole mcode investigation land on the
identical mechanism.
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


class TestOutputScaleQuadGeneralizes(unittest.TestCase):
    # fixture: (output_scale,)
    CASES = {
        "conv_1c1c_8x8_k3.mcode.gz": 0.006336795166134834,
        "matmul_4x8x8.mcode.gz": 0.019181855022907257,
    }

    def test_quad_present_with_correct_frame(self):
        for name, zs in self.CASES.items():
            data = load(name)
            pat = struct.pack("<f", zs)
            found = hits(data, pat)
            self.assertEqual(len(found), 4, f"{name}: quad count")
            strides = {b - a for a, b in zip(found, found[1:])}
            self.assertEqual(strides, {7}, f"{name}: quad stride")
            self.assertEqual(
                data[found[0] - 3 : found[0]].hex(), "05500f", f"{name}: lead-in"
            )
            for i in found:
                self.assertEqual(data[i + 4], 0x81, f"{name}@{i}: tail byte 0")
            # Fourth copy's tail gets a high bit set, same as Gemm's own
            # quad -- 0x03 -> 0x83 on the last copy only.
            self.assertEqual(data[found[-1] + 6], 0x83, f"{name}: last-copy high bit")
            self.assertTrue(
                all(data[i + 6] == 0x03 for i in found[:-1]),
                f"{name}: earlier copies should read 0x03, not 0x83",
            )


if __name__ == "__main__":
    unittest.main()
