"""The first mcode decode work on a training/backward-graph stream in
this project: the output-scale quad and site A (both established for
forward-inference ops -- Mul, Gemm, Conv, MatMul) generalize cleanly to
`Reshape`+`Gather`, a real backward-pass slice from this project's own
training-graph op-coverage battery.

`scripts/axera/README.md`'s "Backward-graph ops" section (and
`tests/test_axera_mcode_validator.py`, which carries the already-
committed `reshape_gather_bwd.mcode.gz` fixture) establish this graph's
*numerical* correctness on real hardware (`Reshape+Gather` max|diff|
0.005 against ORT) and that its mcode parses cleanly under the codec's
general structural grammar -- but nothing about what any specific field
in that stream *means*. Every other mcode-decode PR this project has
produced (dozens, across Mul/Gemm/Conv/MatMul) never touched a
training-graph or backward-slice stream at all.

## Method: reproduce the graph from scratch with a known ground truth

`Reshape`(`X`[8] -> [2,4]) into `Gather`(axis=1, indices=[0,2]) ->
`Y`[2,2], built via `pulsar2_docker.build()` (compile-only, no physical
AXCL device access needed) with MinMax calibration over `X ~
U(-1,1)`. The compiled `quant_axmodel.onnx` shows exactly one real
scale in the whole graph -- `AxQuantizeLinear` on `X`
(`output_scales=[0.0076232957653701305]`, `zp=131`) and
`AxDequantizeLinear` on `Y` reading the **identical** scale/zero-point
back out. `AxReshape`/`AxGather` carry no scale attributes of their
own: neither op requantizes, consistent with the README's own framing
("no gradient rule, and none needed" for this op family) -- there is
only one real quantization parameter for this entire graph, input and
output share it exactly.

## Both already-decoded quads are present, at the ground-truth value

- **Output-scale quad** (`05 50 0f <f32(z_scale)> 81 <tag2> 03` x4,
  stride 7, confirmed generalized across Mul/Gemm/Conv/MatMul in
  `tests/test_axera_output_scale_quad_generalizes.py`): found at offset
  1374, `float32 = 0.0076232957653701305` -- an exact match to the
  quant model's real scale, tag2 = `0xc6`.
- **Site A** (`<f32(1/x_scale)> a1 00 <id>` x4, stride 8, confirmed
  generalized in `tests/test_axera_site_a_generalizes.py`): found at
  offset 1067, `float32 = 131.16793...` = `1/0.0076232957653701305`
  exactly.

**Confirmed above the noise floor.** An independent rebuild of the
identical config reproduces both quads byte-for-byte at the identical
offsets; the only 3 bytes that differ between the two builds are at
offsets 311-325, inside this project's already-known `~295-330` noise
zone.

## The pre-existing `reshape_gather_bwd.mcode.gz` fixture shows the
## identical mechanism -- confirmed with NO external ground truth needed

`scripts/axera/fixtures/reshape_gather_bwd.mcode.gz` (already committed
via `tests/test_axera_mcode_validator.py`, provenance/exact build
config unknown) shows the same two quad frames at nearly the same
offsets as the fresh build above: site A at the **identical** offset
1067 (`float32 = 60.75748825073242`), and the output-scale quad at
offset 1367 (`float32 = 0.016458876430988312`). These two numbers are
not independently meaningful -- but `struct.pack("<f", 1.0 /
60.75748825073242)` unpacks back to **exactly**
`0.016458876430988312`, the output quad's own value, bit for bit. That
is precisely the same `x_scale == z_scale` relationship the fresh,
ground-truth-verified build above establishes for this op pair -- a
closed, falsifiable numerical identity within the pre-existing fixture
itself, needing no rebuild or external quant-model access to confirm.
This is strong, self-contained evidence that both decoded quads
generalize correctly to this graph's *original* mystery-provenance
fixture too, not just to a freshly-built stand-in.

## What this does not establish

This confirms the two already-known scale-carrying quads generalize to
a backward-pass op pair; it does not touch anything specific to
*training* semantics (gradients, Adam moments, loss scaling) -- Reshape
and Gather are pure data-movement ops, chosen here because they are the
smallest, cleanest backward-graph fixture already in the repo. The much
larger `toy_training_step.mcode.gz` (a full forward+loss+backward+Adam
step, 31032 bytes) and the small `adam_update_fp32.mcode.gz`/
`loss_head_kd.mcode.gz` fixtures remain completely undecoded -- this
file is a first foothold, not a survey of training-graph mcode.
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


class TestFreshBuildConfirmsBothQuadsAtGroundTruthValue(unittest.TestCase):
    """A from-scratch Reshape(X[8]->[2,4])+Gather(axis=1) build, MinMax
    calibration, with the real quant-model scale known: both the
    output-scale quad and site A carry that exact value."""

    X_SCALE = 0.0076232957653701305

    def test_output_scale_quad_matches_the_real_scale(self):
        data = load("reshape_gather_fresh.mcode.gz")
        pat = struct.pack("<f", self.X_SCALE)
        found = hits(data, pat)
        self.assertEqual(len(found), 4, "output-scale quad hits")
        strides = {b - a for a, b in zip(found, found[1:])}
        self.assertEqual(strides, {7}, "output-scale quad stride")
        self.assertEqual(data[found[0] - 3 : found[0]].hex(), "05500f", "lead-in")
        for i in found[:-1]:
            self.assertEqual(data[i + 4 : i + 7].hex()[:2], "81", f"@{i}: tail byte 0")
            self.assertEqual(data[i + 6], 0x03, f"@{i}: earlier-copy tail")
        self.assertEqual(data[found[-1] + 6], 0x83, "last-copy high bit")

    def test_site_a_matches_the_real_reciprocal_scale(self):
        data = load("reshape_gather_fresh.mcode.gz")
        pat = struct.pack("<f", 1.0 / self.X_SCALE)
        found = hits(data, pat)
        self.assertEqual(len(found), 4, "site A hits")
        strides = {b - a for a, b in zip(found, found[1:])}
        self.assertEqual(strides, {8}, "site A stride")
        for i in found:
            self.assertEqual(data[i + 4 : i + 6].hex(), "a100", f"@{i}: frame")

    def test_both_quads_survive_an_independent_rebuild(self):
        orig = load("reshape_gather_fresh.mcode.gz")
        rebuild = load("reshape_gather_bwd_rebuild.mcode.gz")
        self.assertEqual(len(orig), len(rebuild))
        diffs = [i for i in range(len(orig)) if orig[i] != rebuild[i]]
        self.assertTrue(diffs, "sanity: the two builds should not be byte-identical")
        self.assertGreaterEqual(min(diffs), 295)
        self.assertLessEqual(max(diffs), 335)
        # The two quads themselves (offsets 1067-1099 and 1370-1401) are
        # untouched by the rebuild's own noise.
        self.assertEqual(orig[1067:1099], rebuild[1067:1099], "site A region")
        self.assertEqual(orig[1370:1401], rebuild[1370:1401], "output-quad region")


class TestPreExistingFixtureIsSelfConsistentWithNoExternalGroundTruth(
    unittest.TestCase
):
    """The already-committed reshape_gather_bwd.mcode.gz fixture (of
    unknown build provenance) shows the same x_scale == z_scale
    relationship the fresh build proves directly -- confirmed via a
    closed numerical identity inside the fixture itself, needing no
    external quant-model access."""

    def test_site_a_and_output_quad_are_reciprocal_partners(self):
        data = load("reshape_gather_bwd.mcode.gz")

        site_a_val = 60.75748825073242
        pat_a = struct.pack("<f", site_a_val)
        found_a = hits(data, pat_a)
        self.assertEqual(len(found_a), 4, "site A hits")
        self.assertEqual({b - a for a, b in zip(found_a, found_a[1:])}, {8})
        for i in found_a:
            self.assertEqual(data[i + 4 : i + 6].hex(), "a100")

        quad_val = 0.016458876430988312
        pat_z = struct.pack("<f", quad_val)
        found_z = hits(data, pat_z)
        self.assertEqual(len(found_z), 4, "output-scale quad hits")
        self.assertEqual({b - a for a, b in zip(found_z, found_z[1:])}, {7})
        self.assertEqual(data[found_z[0] - 3 : found_z[0]].hex(), "05500f")

        # The closed identity: float32(1/site_A) == the output quad's own
        # value, bit for bit -- proving x_scale == z_scale for this op
        # pair without needing this fixture's original build config.
        self.assertEqual(struct.pack("<f", 1.0 / site_a_val), pat_z)


if __name__ == "__main__":
    unittest.main()
