"""``patch_output_quad`` -- the first generalized-op mcode *generation*
work in this project (previously only Mul had a patch function,
``patch_mul_output_quad``).

`tests/test_axera_output_scale_quad_generalizes.py` and
`tests/test_axera_gemm_output_quad.py` (both merged) established that
the output-scale quad decoded for Mul also appears, byte-identically
framed, on Gemm/Conv/MatMul -- but with a `05 50 0f` lead-in before the
first copy that Mul's own bare frame doesn't have. Nobody had written a
patch function for the generalized frame, or verified that patching it
actually reproduces a real second build -- decoding *where* a field is
is not the same as being able to *write* it correctly.

``scripts/axera.tiny_emit.patch_output_quad`` fills that gap: it
detects either frame (the `05500f` lead-in, or Mul's bare `03`) and
patches the 4-byte float32 value at all four stride-7 copies, the same
defensive style as ``patch_mul_output_quad`` (checks the `0x81` tail
byte on every copy, never depends on the unexplained `tag2` byte).

## Verification: no fresh build needed -- two already-committed Gemm
## pairs already differ *only* in output_scale (plus known noise)

`tests/test_axera_gemm_output_quad.py`'s own
`test_transb_pair_differs_only_at_this_quad_plus_known_noise` already
proved, for two shapes, that a `transB=0`/`transB=1` pair is
byte-identical except for the output-scale quad's own bytes and the
already-known `~295-330` noise zone. That is exactly the condition
needed to test a patch function honestly: patch the `transB=0` build's
quad to the `transB=1` build's *real* output_scale, and check the
result against `transB=1`'s own real mcode.

**Both pairs reproduce the real target build byte-for-byte outside the
noise zone** -- `gemm_1x8x8_tb0` patched to `tb1`'s scale, and
`gemm_4x16x8_tb0` patched to `tb1`'s scale, each differ from the real
`tb1` fixture at exactly the same handful of bytes
`test_transb_pair_differs_only_at_this_quad_plus_known_noise` already
attributed entirely to ordinary non-determinism, and nowhere else.

## What this does and does not establish

This is compile-only evidence (no physical AXCL device access used or
needed) for Gemm specifically, using data this project already had --
not a fresh device run, and not (yet) independently confirmed for Conv
or MatMul, which share the identical frame per
`test_axera_output_scale_quad_generalizes.py` but have no existing
same-shape, scale-only-differing fixture pair to patch against without
a fresh build (unlike Gemm's `transB` flip, no free, weight-preserving
lever that nudges only the output scale was readily available for Conv
in this pass -- a real, honest scope limit, not a decode gap). The
function's own framing/tail-byte checks are written the same
defensively for all three generalized-frame ops, so there is no reason
to expect Conv/MatMul to behave differently, but that is an inference
from the shared frame, not this file's own independent proof for those
two ops.
"""

import gzip
import os
import sys
import unittest

_AXERA_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts", "axera")
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import tiny_emit  # noqa: E402

FIX = os.path.join(_AXERA_DIR, "fixtures")

NOISE_ZONE = range(295, 330)


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


class TestPatchOutputQuadReproducesARealSecondBuild(unittest.TestCase):
    """Patching a Gemm reference's output quad to a second real build's
    own output_scale reproduces that build byte-for-byte outside this
    project's already-known noise zone -- the same strength of evidence
    ("confirmed via rebuild"/"matches the real compiled output") this
    project has used throughout for decode work, now applied to a
    generator."""

    # (reference fixture, its real scale, target fixture, its real scale)
    PAIRS = [
        (
            "gemm_1x8x8_tb0.mcode.gz",
            0.018938595429062843,
            "gemm_1x8x8_tb1.mcode.gz",
            0.018938593566417694,
        ),
        (
            "gemm_4x16x8_tb0.mcode.gz",
            0.029019242152571678,
            "gemm_4x16x8_tb1.mcode.gz",
            0.029019244015216827,
        ),
    ]

    def test_patched_stream_matches_real_target_outside_noise_zone(self):
        for ref_name, old_z, tgt_name, new_z in self.PAIRS:
            ref = load(ref_name)
            tgt = load(tgt_name)
            patched = tiny_emit.patch_output_quad(ref, old_z, new_z)
            self.assertEqual(len(patched), len(tgt), ref_name)
            diffs = [i for i in range(len(patched)) if patched[i] != tgt[i]]
            outside_noise = [i for i in diffs if i not in NOISE_ZONE]
            self.assertEqual(
                outside_noise,
                [],
                f"{ref_name} patched to {tgt_name}'s scale: unexplained diff"
                f" outside the noise zone at {outside_noise}",
            )

    def test_patching_round_trips_back_to_the_reference(self):
        """Patching there and back reproduces the original reference
        exactly -- the function is a pure value rewrite, not lossy."""
        for ref_name, old_z, _tgt_name, new_z in self.PAIRS:
            ref = load(ref_name)
            there = tiny_emit.patch_output_quad(ref, old_z, new_z)
            back = tiny_emit.patch_output_quad(there, new_z, old_z)
            self.assertEqual(back, ref, ref_name)


class TestPatchOutputQuadDetectsBothFrames(unittest.TestCase):
    """The generalized frame (05500f lead-in, Gemm/Conv/MatMul) and
    Mul's own bare-03 frame are both recognized by the same function."""

    def test_generalized_frame_gemm(self):
        ref = load("gemm_1x8x8_tb0.mcode.gz")
        out = tiny_emit.patch_output_quad(
            ref, 0.018938595429062843, 0.018938595429062843
        )
        self.assertEqual(out, ref, "patching to the identical value is a no-op")

    def test_generalized_frame_conv(self):
        # conv_1c1c_8x8_k3.mcode.gz's own output_scale, per
        # test_axera_output_scale_quad_generalizes.py's CASES.
        ref = load("conv_1c1c_8x8_k3.mcode.gz")
        out = tiny_emit.patch_output_quad(
            ref, 0.006336795166134834, 0.006336795166134834
        )
        self.assertEqual(out, ref)

    def test_generalized_frame_matmul(self):
        # matmul_4x8x8.mcode.gz's own output_scale, per the same file.
        ref = load("matmul_4x8x8.mcode.gz")
        out = tiny_emit.patch_output_quad(
            ref, 0.019181855022907257, 0.019181855022907257
        )
        self.assertEqual(out, ref)

    def test_bare_frame_mul_still_works(self):
        # mul_1x8.mcode.gz's own z_scale, per
        # test_axera_mcode_reciprocal.py's own CALIB table (line 92) --
        # its output quad is Mul's bare-03 frame, not the generalized
        # 05500f one Gemm/Conv/MatMul carry.
        ref = load("mul_1x8.mcode.gz")
        out = tiny_emit.patch_output_quad(
            ref, 0.007151617668569088, 0.007151617668569088
        )
        self.assertEqual(out, ref)


if __name__ == "__main__":
    unittest.main()
