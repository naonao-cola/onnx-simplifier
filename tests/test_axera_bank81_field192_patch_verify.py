"""Runs the exact patch-and-verify test `tests/test_axera_generator_progress_stocktake.py`
(PR #1620) identified as the single most valuable next step toward this
project's original macro-goal (a from-scratch mcode generator): take a
real reference build, patch `tiny_emit.bank81_field192_operand`'s own
predicted value for a DIFFERENT K into it, and check whether the result
is closer to or farther from a genuine second build at that target K --
the same reflow-or-no-reflow bar `patch_matmul_a_scale`'s own
form-crossing check and `patch_conv_zp_x`'s own zp_x check were held to.

## Method

Two already-committed fixtures, both `Gemm(A[1,K],B[K,1000],C[1000])`,
`transB=0`, `M=1`, `N=1000`: `gemm_1x512x1000_tb0.mcode.gz` (K=512) and
`gemm_1x256x1000.mcode.gz` (K=256). Each carries bank `0x81` field=192
twice (`mcode.decode()`, not a raw byte search, locates both V-records
precisely). The patch: take one fixture's raw bytes, overwrite both
field=192 operands in place with `bank81_field192_operand(target_k)`'s
own predicted bytes, leave every other byte untouched. Compare the
result against the REAL fixture at `target_k`, byte-for-byte.

## Result: unambiguously NOT patchable in place -- worse than any prior
## reflow finding in this project

**Forward (K=512 source -> K=256 target):** the two raw stream LENGTHS
already differ (4368 vs 3792 bytes) -- no in-place byte patch can ever
close a 576-byte length gap. Over the shared 3792-byte prefix, **2776
bytes differ (73%)**, with the first mismatch at byte 36 -- nowhere near
either copy of the patched field itself (bytes 533/1172).

**Reverse (K=256 source -> K=512 target):** the identical picture --
same 2776/3792 diff count over the shared prefix, same first-mismatch
offset (36). Symmetric, not an artifact of picking one direction.

**The patched stream still decodes/checks cleanly** (`mcode.check()`
reports zero errors either direction) -- confirmed here as a weak,
uninformative signal on its own, consistent with this project's grammar
validating structural well-formedness, not semantic correctness against
a real build.

This is a clean, unambiguous negative -- more decisive than
`patch_matmul_a_scale`'s own 201-byte/3336-byte, single-contiguous-region
finding for a form-crossing MatMul `A`-scale edit. Here the diff is not
confined to one region at all; it starts 36 bytes into the stream and
covers nearly three-quarters of the shared prefix, fully consistent with
`K` being Gemm's own real contraction dimension -- changing it
restructures tiling/scheduling throughout the compiled output, not just
this one field's own recorded value. `tiny_emit.bank81_field192_operand`'s
own docstring has been updated directly with this finding (2026-09-18),
matching `patch_conv_zp_x`'s own precedent of recording a verified
negative in the function's own docstring rather than only in a test file.

## What this establishes for the macro-goal

`bank81_field192_operand` remains exactly what it was before this file:
a correct, verified PREDICTOR for one field's own value given K, useful
for a generator that is choosing to target a specific K deliberately
(e.g. reading off what field=192 *should* say once a full generator
exists) -- but not a working PATCH function, and this file confirms that
distinction was not merely a hedge in the original docstring: it is a
directly measured, symmetric, ~73%-of-stream-scale reflow. No new patch
function is added here for exactly that reason -- adding one would
misrepresent a verified-unsafe operation as safe.
"""

import gzip
import os
import sys
import unittest

_AXERA_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts", "axera")
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import mcode  # noqa: E402
import tiny_emit  # noqa: E402

FIX = os.path.join(_AXERA_DIR, "fixtures")

SRC_512 = "gemm_1x512x1000_tb0.mcode.gz"
SRC_256 = "gemm_1x256x1000.mcode.gz"


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def decode(name):
    return mcode.decode(load(name), **mcode.FULL_RULE)


def field192_hits(name):
    recs = decode(name)
    return [
        r
        for r in recs
        if r["kind"] == "V" and r.get("bank") == 0x81 and r.get("field") == 192
    ]


def patch_field192(data, hits, new_operand):
    out = bytearray(data)
    for r in hits:
        at = r["at"]
        assert bytes(out[at + 4 : at + 7]) == r["operand"], (
            "operand location assumption broke",
            r,
        )
        out[at + 4 : at + 7] = new_operand
    return bytes(out)


class TestFieldLocationAssumptionHolds(unittest.TestCase):
    """Both fixtures carry field=192 exactly twice, matching the
    already-established K=512/K=256 large-N pattern this test builds on
    (tests/test_axera_bank81_cross_op_check.py)."""

    def test_two_hits_each(self):
        self.assertEqual(len(field192_hits(SRC_512)), 2)
        self.assertEqual(len(field192_hits(SRC_256)), 2)

    def test_predicted_values_match_the_real_fixtures_own_content(self):
        for name, k in ((SRC_512, 512), (SRC_256, 256)):
            hits = field192_hits(name)
            predicted = tiny_emit.bank81_field192_operand(k)
            for r in hits:
                self.assertEqual(r["operand"], predicted, name)


class TestForwardPatchDoesNotReproduceARealBuild(unittest.TestCase):
    """K=512 source, patched to K=256's own predicted field=192 value,
    diffed against the REAL K=256 fixture."""

    def _patched(self):
        src = load(SRC_512)
        hits = field192_hits(SRC_512)
        predicted = tiny_emit.bank81_field192_operand(256)
        return patch_field192(src, hits, predicted)

    def test_raw_lengths_already_differ(self):
        patched = self._patched()
        real = load(SRC_256)
        self.assertNotEqual(len(patched), len(real))
        self.assertEqual(len(patched), 4368)
        self.assertEqual(len(real), 3792)

    def test_diff_over_shared_prefix_is_large_and_starts_far_from_the_field(self):
        patched = self._patched()
        real = load(SRC_256)
        n = min(len(patched), len(real))
        diffs = [i for i in range(n) if patched[i] != real[i]]
        self.assertEqual(len(diffs), 2776)
        self.assertEqual(diffs[0], 36)
        # The patched field's own two copies sit at byte offsets 533 and
        # 1172 -- the diff starting at byte 36 is nowhere near either.
        self.assertLess(diffs[0], 533)

    def test_patched_stream_still_decodes_and_checks_cleanly(self):
        patched = self._patched()
        self.assertEqual(mcode.check(patched), [])
        recs = mcode.decode(patched, **mcode.FULL_RULE)
        self.assertGreater(len(recs), 0)


class TestReversePatchShowsTheIdenticalPicture(unittest.TestCase):
    """K=256 source, patched to K=512's own predicted field=192 value,
    diffed against the REAL K=512 fixture -- confirms the forward
    result is not an artifact of direction."""

    def _patched(self):
        src = load(SRC_256)
        hits = field192_hits(SRC_256)
        predicted = tiny_emit.bank81_field192_operand(512)
        return patch_field192(src, hits, predicted)

    def test_raw_lengths_already_differ(self):
        patched = self._patched()
        real = load(SRC_512)
        self.assertEqual(len(patched), 3792)
        self.assertEqual(len(real), 4368)

    def test_diff_count_and_offset_match_the_forward_direction(self):
        patched = self._patched()
        real = load(SRC_512)
        n = min(len(patched), len(real))
        diffs = [i for i in range(n) if patched[i] != real[i]]
        self.assertEqual(len(diffs), 2776)
        self.assertEqual(diffs[0], 36)

    def test_patched_stream_still_decodes_and_checks_cleanly(self):
        patched = self._patched()
        self.assertEqual(mcode.check(patched), [])


if __name__ == "__main__":
    unittest.main()
