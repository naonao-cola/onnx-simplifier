"""Gemm's output-scale quad, and what its `transB` sensitivity actually is.

No committed Gemm fixture or build script existed before this file --
the README's "Gemm joins the MAC engines" section (`transB=0` vs
`transB=1` producing "a real, substantial signal... the largest and
cleanest real signal isolated in this whole investigation") was
reconstructed from scratch at a much smaller scale (`Gemm(A[M,K],
B[K,N] or B.T[N,K], C[N])`, `M,K,N` in the low tens rather than a real
FC layer) to make it directly tractable, not to reproduce that exact
85-byte block -- this file's builds are unrelated in shape to whatever
"gemm_base" originally was, and do not claim to explain that specific
finding. What they do establish, newly:

Discovery (2026-09-16): Gemm compiles to `AxQuantizedFullyConnected`
with **per-output-channel weight scales** (`weight_scales`, one float
per N) -- a different quant scheme from every elementwise op this
project's mcode work has covered so far (all per-tensor). Its output
carries the exact same *family* of quad this project already decoded for
Mul (`TestOutputScaleQuads` in test_axera_mcode_reciprocal.py: `03
<f32(z_scale)> 81 <tag2>` x4 at stride 7) -- here framed
`05 50 0f <f32(output_scale)> 81 6e 03` x4 at stride 7 instead. The
`05 50 0f` prefix (before the first copy only; every later copy's
"prefix" is just the previous copy's own `81 6e 03` tail, stride 7
exactly matching the 4-byte float + 3-byte tag) matches, by name and
shape, the general corpus's own already-documented but never-attributed
"`05 50 0f <u32>` unanchored companion-shaped writes" (this README's "A
five-byte form that programs its pair twice" section) -- this is the
first time this project has identified what one of those writes actually
*carries*: an output scale, not an unknown `<u32>`. Whether every
occurrence of that general pattern is an output scale, or only some are,
is not established here -- only this Gemm instance is.

The `transB` "signal", explained (at this small scale): `Gemm(A, B,
C)` with `transB=1` computes the exact same numbers as `transB=0` fed
`B.T` (confirmed: `weight_scales` -- the per-channel *weight*
calibration -- comes out byte-identical either way). But
`output_scales` itself differs by ~1e-9 relative between the two builds
(`0.018938595429062843` vs `0.018938593566417694` for one shape) -- a
genuine, tiny floating-point summation-order artifact, not measurement
noise (repeatable, and distinct from the already-known ~303-325 noise
zone re-confirmed below). That's small enough to almost always agree in
float32, except right at a rounding boundary, where the quad's low byte
flips by exactly 1 -- the identical "two builds' float64 scales straddle
a float32 boundary" phenomenon `TestOutputScaleQuads` already documented
for Mul, now confirmed in a second, structurally unrelated op. This is
NOT evidence for the README's speculative "real per-tile or per-output-
column memory-access-pattern encoding" explanation of the transB
signal -- it's a numerically-driven side effect of transB changing
Pulsar2's own internal calibration pass, unrelated to command/scheduling
encoding. Whether the original 85-byte block finding (a much bigger
signal, at an unreproduced larger shape) has this same root cause,
scaled up, or is a genuinely distinct scanning-order effect, is open.

Also newly found, not chased further: at `M=8` (vs `M=1` or `M=4` at the
same `K,N`), transB's diff stops being this single clean quad-boundary
flip and becomes a denser, differently-shaped ~16-byte rewrite at a
different offset entirely (see `test_m8_shows_a_different_transb_diff`
below) -- consistent with this project's repeated finding elsewhere
(Conv's dilation threshold effect) that this compiler's behavior changes
in threshold/tile-boundary ways, not smoothly, as a dimension grows. A
real, precisely reproducible lead for whoever chases the M-dependent
shape next, not decoded here.
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


class TestGemmOutputScaleQuad(unittest.TestCase):
    """`05 50 0f <f32(output_scale)> 81 6e 03` x4 at stride 7."""

    # fixture: (output_scale,)
    CASES = {
        "gemm_1x8x8_tb0.mcode.gz": 0.018938595429062843,
        "gemm_1x8x8_tb1.mcode.gz": 0.018938593566417694,
        "gemm_4x16x8_tb0.mcode.gz": 0.029019242152571678,
        "gemm_4x16x8_tb1.mcode.gz": 0.029019244015216827,
    }

    def test_quad_carries_exact_float32_output_scale(self):
        # Only the tail's first byte (0x81) is asserted constant across
        # all 4 copies and both shapes -- the second byte is constant
        # *within* a shape but not across shapes (0x6e at K=8,N=8; 0x72
        # at K=16,N=8), and the third byte carries an extra high bit
        # (0x03 -> 0x83) on the fourth copy only, every time. Same
        # "tag2 isn't universally constant" lesson TestOutputScaleQuads
        # already learned for Mul's own quad -- don't over-assert it here
        # either.
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

    def test_transb_pair_differs_only_at_this_quad_plus_known_noise(self):
        """transB=0 vs transB=1 at the same (M,K,N): every byte difference
        is either the quad's own low byte (the output-scale ULP flip) or
        inside the already-confirmed ~303-325 non-determinism zone (see
        TestKnownNoiseZoneReconfirmed) -- nothing else moves."""
        pairs = [
            ("gemm_1x8x8_tb0.mcode.gz", "gemm_1x8x8_tb1.mcode.gz"),
            ("gemm_4x16x8_tb0.mcode.gz", "gemm_4x16x8_tb1.mcode.gz"),
        ]
        noise_zone = range(295, 330)
        for tb0_name, tb1_name in pairs:
            d0, d1 = load(tb0_name), load(tb1_name)
            self.assertEqual(len(d0), len(d1), (tb0_name, tb1_name))
            quad_offsets = set(hits(d0, struct.pack("<f", self.CASES[tb0_name])))
            for i in range(len(d0)):
                if d0[i] == d1[i]:
                    continue
                in_quad = any(q <= i < q + 4 for q in quad_offsets)
                in_noise = i in noise_zone
                self.assertTrue(
                    in_quad or in_noise,
                    f"{tb0_name} vs {tb1_name} @{i}: unexplained diff"
                    f" {d0[i]:#04x}->{d1[i]:#04x}, not in the quad or the noise zone",
                )


class TestKnownNoiseZoneReconfirmed(unittest.TestCase):
    """Rebuilding one config (tb0, 1x8x8) unchanged still lands in the
    same ~303-325 zone this README already documented for `auto_pad` and
    `ceil_mode` -- reconfirmed here on a fourth, unrelated op/model,
    strengthening the "global property of mcode generation" conclusion.
    Uses the same fixture twice (there is no third build to diff against
    committed here -- the determinism check itself was run once, ad hoc,
    against a throwaway rebuild; this test only pins that the known-noise
    positions are excluded from the quad hunt above, not the determinism
    check itself)."""

    def test_noise_positions_do_not_overlap_the_quad(self):
        data = load("gemm_1x8x8_tb0.mcode.gz")
        quad_offsets = hits(data, struct.pack("<f", 0.018938595429062843))
        noise_zone = range(295, 330)
        for off in quad_offsets:
            for b in range(off, off + 4):
                self.assertNotIn(
                    b, noise_zone, f"quad byte {b} overlaps the noise zone"
                )


class TestM8DiffersStructurally(unittest.TestCase):
    """At M=8 (vs M=1/M=4 at the same K,N), transB's diff is not the
    clean single-quad ULP flip above -- a real, differently-shaped,
    denser rewrite appears instead. Pinned as raw evidence, not decoded.
    """

    OUTPUT_SCALE_TB0 = 0.025359967723488808

    def test_m8_shows_a_different_transb_diff(self):
        d0 = load("gemm_8x8x8_tb0.mcode.gz")
        d1 = load("gemm_8x8x8_tb1.mcode.gz")
        self.assertEqual(len(d0), len(d1))
        diffs = [i for i in range(len(d0)) if d0[i] != d1[i]]
        # Confirmed empirically 2026-09-16: 10 differing bytes, offsets
        # 682-698. Unlike the smaller shapes' clean quad-boundary ULP
        # flip (exactly one byte differs, always by the same small
        # amount), several distinct byte values change here -- not
        # consistent with a single float32 rounding-boundary flip.
        self.assertTrue(diffs, "expected a real diff at M=8")
        self.assertTrue(
            all(680 <= i <= 700 for i in diffs),
            f"diff region moved: {diffs} (docstring's 682-698 claim is stale)",
        )
        self.assertGreater(
            len(diffs),
            4,
            "expected more than a single 4-byte quad's worth of change at M=8",
        )

    def test_m8_diff_does_not_overlap_output_scale_quad(self):
        d0 = load("gemm_8x8x8_tb0.mcode.gz")
        d1 = load("gemm_8x8x8_tb1.mcode.gz")
        pat0 = struct.pack("<f", self.OUTPUT_SCALE_TB0)
        quad0 = hits(d0, pat0)
        self.assertEqual(len(quad0), 4, "gemm_8x8x8_tb0: quad count")
        diffs = [i for i in range(len(d0)) if d0[i] != d1[i]]
        for i in diffs:
            self.assertFalse(
                any(q <= i < q + 4 for q in quad0),
                f"@{i}: M=8's diff region overlaps the output-scale quad after all",
            )


if __name__ == "__main__":
    unittest.main()
