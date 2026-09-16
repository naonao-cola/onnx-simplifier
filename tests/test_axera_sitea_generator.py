"""First generalized-op mcode *generation* (patching), not decoding.

Every mcode PR this project has produced up to now was pure decode
work: characterizing what a byte carries, never writing a function that
computes and writes new bytes for a value nobody built with Pulsar2 yet.
`scripts/axera/tiny_emit.py` already has `patch_mul_scales`/
`patch_mul_output_quad`/`patch_mul_zp_x` for Mul specifically, hardware-
confirmed bit-exact. This file adds and verifies the same idea for
Gemm/Conv/MatMul's own site-A slot, using the byte patterns
`tests/test_axera_site_a_generalizes.py` and
`tests/test_axera_matmul_quad_form_switch.py` already decoded (never
before turned into a patch function).

## What works: `patch_site_a` (Gemm/Conv, and MatMul's `B` operand)

Site A's full-form quad (`<f32(1/x_scale)> a1 00 <id>` x4, stride 8) is
the same shape Mul's own `patch_mul_scales` already patches for Mul's
`x`; `patch_site_a` is a standalone entry point for ops that have no
output-side y-operand to patch alongside it. Verified below against
real, freshly-built Gemm and Conv model pairs (two different input
calibration ranges, same weights, same mcode length): patching one
build's site A to the other's real scale reproduces that scale's exact
quad bytes, byte for byte, at the real target build's own offsets.
Residual diffs elsewhere in the stream after patching (Gemm: 23 bytes;
Conv: 26 bytes) are expected and not a bug -- changing the input scale
also legitimately moves the output-scale quad (`z_scale` depends on the
whole graph's calibrated range) and, for Conv, the weight-table's
closed-form requant constants (`M_c = x_scale * w_scale_c / y_scale`,
`scripts/axera/README.md`'s own formula) -- neither of those families
is `patch_site_a`'s job, mirroring `patch_mul_scales`'s own documented
scope (full-stream equality after patching a subset of fields was never
the goal for that function either).

## What works, with a real limit: `patch_matmul_a_scale`

MatMul's rank-3 batched `A` quad has two encodings
(`tests/test_axera_matmul_quad_form_switch.py`): short-form when
`1/A_scale`'s float32 high byte is `0x42` (value in `[32,128)`),
full-form otherwise. `patch_matmul_a_scale` patches within one form
(verified below, both directions) but refuses a form-crossing edit.

That refusal is not a guess. A controlled pair -- same `B` calibration
seed, identical 3336-byte mcode length, identical `A_offset`/`B_offset`
table order (ruling out both of this project's two known confounds:
the noise floor and `tests/test_axera_matmul_offset_table_coinflip.py`'s
per-compile coin flip) -- differing *only* in whether `A`'s scale
crosses the `0x42`/`0x43` boundary, shows **201 bytes differ, spanning
offset 785 to 2896** -- not the quad's own ~28-byte footprint. One
concrete anchor inside that span: a little-endian u32 field at offset
2896 reads 1060 in the short-form build and 1064 in the full-form one
-- exactly the difference between the short form's `4x6=24`-byte
footprint and the full form's `4x7=28`-byte footprint. The extra 4
bytes the full form needs are drawn from elsewhere in the stream (that
count field, plus a real content change at offset 785-802), not local
padding to the quad itself -- so even though the *total* stream length
happens to stay equal across this specific pair, the edit is not a
local swap, and `patch_matmul_a_scale` does not attempt one.

## What remains open

Neither function touches the S-unit program bodies, MatMul's own
`var`-byte selector (`tests/test_axera_matmul_batched_var_byte.py` and
its many follow-ups -- no closed formula exists for most of its range,
so there is nothing to compute a target value from), Gemm's `K*M-1`
field (already shown, by `tests/test_axera_gemm_km1_regime2_missing_value.py`,
to sit inside its own diffuse reflow once a shape crosses a regime
boundary -- the same class of problem this file's `patch_matmul_a_scale`
refusal demonstrates directly), or Conv's dilation/orientation fields.
Cross-form MatMul `A` edits, and any shape-driven (not calibration-
driven) field this session decoded, remain generator-blocked for the
same underlying reason: characterizing *what* varies is not the same
bar as safely *writing* it without reflowing bytes nobody has mapped.
"""

import gzip
import os
import struct
import sys
import unittest

_AXERA_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts", "axera")
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import tiny_emit  # noqa: E402

FIX = os.path.join(_AXERA_DIR, "fixtures")


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def hits(data, pat):
    return [i for i in range(len(data) - len(pat) + 1) if data[i : i + len(pat)] == pat]


class TestPatchSiteAGemm(unittest.TestCase):
    OLD_SCALE = 0.0076232957653701305
    NEW_SCALE = 0.019058240577578545

    def test_patched_quad_matches_real_target_build_exactly(self):
        ref = load("gemm_sitea_h0.mcode.gz")
        target = load("gemm_sitea_h1.mcode.gz")
        self.assertEqual(len(ref), len(target))
        patched = tiny_emit.patch_site_a(ref, self.OLD_SCALE, self.NEW_SCALE)
        pat = struct.pack("<f", 1.0 / self.NEW_SCALE)
        found = hits(patched, pat)
        self.assertEqual(len(found), 4)
        for off in found:
            self.assertEqual(patched[off : off + 4], target[off : off + 4])

    def test_residual_diff_after_patching_is_small_and_elsewhere(self):
        """Patching site A alone does not reproduce the target build
        byte-for-byte -- the output-scale quad also legitimately moves
        with the input range. This checks the residual is small (not a
        sign the patch itself is wrong), not that it's zero."""
        ref = load("gemm_sitea_h0.mcode.gz")
        target = load("gemm_sitea_h1.mcode.gz")
        patched = tiny_emit.patch_site_a(ref, self.OLD_SCALE, self.NEW_SCALE)
        diffs = [i for i in range(len(patched)) if patched[i] != target[i]]
        self.assertLess(len(diffs), 40)


class TestPatchSiteAConv(unittest.TestCase):
    OLD_SCALE = 0.007832584902644157
    NEW_SCALE = 0.019581463187932968

    def test_patched_quad_matches_real_target_build_exactly(self):
        ref = load("conv_sitea_h0.mcode.gz")
        target = load("conv_sitea_h1.mcode.gz")
        self.assertEqual(len(ref), len(target))
        patched = tiny_emit.patch_site_a(ref, self.OLD_SCALE, self.NEW_SCALE)
        pat = struct.pack("<f", 1.0 / self.NEW_SCALE)
        found = hits(patched, pat)
        self.assertEqual(len(found), 4)
        for off in found:
            self.assertEqual(patched[off : off + 4], target[off : off + 4])

    def test_residual_diff_after_patching_is_small_and_elsewhere(self):
        ref = load("conv_sitea_h0.mcode.gz")
        target = load("conv_sitea_h1.mcode.gz")
        patched = tiny_emit.patch_site_a(ref, self.OLD_SCALE, self.NEW_SCALE)
        diffs = [i for i in range(len(patched)) if patched[i] != target[i]]
        self.assertLess(len(diffs), 40)


class TestMatMulAQuadForm(unittest.TestCase):
    def test_short_form_below_128(self):
        self.assertEqual(tiny_emit.matmul_a_quad_form(1.0 / 100.0), "short")
        self.assertEqual(tiny_emit.matmul_a_quad_form(1.0 / 127.9), "short")

    def test_full_form_at_and_above_128(self):
        self.assertEqual(tiny_emit.matmul_a_quad_form(1.0 / 128.1), "full")
        self.assertEqual(tiny_emit.matmul_a_quad_form(1.0 / 200.0), "full")


class TestPatchMatMulAScaleShortForm(unittest.TestCase):
    OLD_SCALE = 0.009999999776482582  # 1/scale ~= 100.0
    NEW_SCALE = 0.009994043037295341  # 1/scale ~= 100.06, still short-form

    def test_patched_quad_matches_real_target_build_exactly(self):
        ref = load("matmul_a_scale_short_r0.mcode.gz")
        target = load("matmul_a_scale_short_r1.mcode.gz")
        self.assertEqual(len(ref), len(target))
        patched = tiny_emit.patch_matmul_a_scale(ref, self.OLD_SCALE, self.NEW_SCALE)
        pat = struct.pack("<f", 1.0 / self.NEW_SCALE)[:3]
        found = hits(patched, pat)
        self.assertEqual(len(found), 4)
        for off in found:
            # only the 3-byte truncated value is this function's job;
            # the tag/var bytes are preserved from the reference as-is.
            self.assertEqual(patched[off : off + 3], target[off : off + 3])


class TestPatchMatMulAScaleFullForm(unittest.TestCase):
    OLD_SCALE = 0.004999999888241291  # 1/scale == 200.0
    NEW_SCALE = 0.004997021518647671  # 1/scale ~= 200.12, still full-form

    def test_patched_quad_matches_real_target_build_exactly(self):
        ref = load("matmul_a_scale_full_r0.mcode.gz")
        target = load("matmul_a_scale_full_r1.mcode.gz")
        self.assertEqual(len(ref), len(target))
        patched = tiny_emit.patch_matmul_a_scale(ref, self.OLD_SCALE, self.NEW_SCALE)
        pat = struct.pack("<f", 1.0 / self.NEW_SCALE)
        found = hits(patched, pat)
        self.assertEqual(len(found), 4)
        for off in found:
            self.assertEqual(patched[off : off + 4], target[off : off + 4])


class TestPatchMatMulAScaleRefusesFormCrossing(unittest.TestCase):
    """The refusal itself, plus the controlled-pair evidence backing it:
    same B seed, same length, same A_offset/B_offset table order,
    differing only in which side of the 0x42/0x43 boundary A's scale
    lands on -- 201 bytes differ, offset 785 to 2896, not a local swap."""

    SHORT_SCALE = 0.009999999776482582
    FULL_SCALE = 0.004999999888241291

    def test_short_to_full_is_refused(self):
        ref = load("matmul_a_scale_short_r0.mcode.gz")
        with self.assertRaises(ValueError):
            tiny_emit.patch_matmul_a_scale(ref, self.SHORT_SCALE, self.FULL_SCALE)

    def test_full_to_short_is_refused(self):
        ref = load("matmul_a_scale_full_r0.mcode.gz")
        with self.assertRaises(ValueError):
            tiny_emit.patch_matmul_a_scale(ref, self.FULL_SCALE, self.SHORT_SCALE)

    def test_controlled_pair_shows_a_diffuse_201_byte_reflow(self):
        """fixedseed_short/full: identical B calibration (seed 3),
        identical A_offset/B_offset table order (both ("A","B")) --
        ruling out the two known confounds -- yet 201 bytes differ,
        spanning offset 785-2896, far beyond the quad's own footprint."""
        short = load("matmul_a_scale_reflow_short.mcode.gz")
        full = load("matmul_a_scale_reflow_full.mcode.gz")
        self.assertEqual(len(short), len(full))

        def table_order(data):
            window = data[195:270]
            return (
                ("A", "B")
                if window.find(b"A_offset") < window.find(b"B_offset")
                else ("B", "A")
            )

        self.assertEqual(table_order(short), table_order(full))
        diffs = [i for i in range(len(short)) if short[i] != full[i]]
        self.assertGreater(len(diffs), 100, "expected a wide reflow, not a local swap")
        self.assertEqual(min(diffs), 785)
        self.assertEqual(max(diffs), 2896)
        # The 4-byte length-field anchor: short form's 24-byte footprint
        # vs. full form's 28-byte footprint, drawn from a count field far
        # from the quad itself.
        self.assertEqual(short[2896:2900], bytes.fromhex("24040000"))
        self.assertEqual(full[2896:2900], bytes.fromhex("28040000"))


if __name__ == "__main__":
    unittest.main()
