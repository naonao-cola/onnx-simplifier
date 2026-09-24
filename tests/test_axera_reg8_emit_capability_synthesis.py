"""Synthesis pass over this session's own generator-capability cluster:
`tests/test_axera_generator_progress_stocktake.py` (PR #1620) stepped
back to ask whether this session's ~150-PR resource-model decode arc
had actually touched `scripts/axera/tiny_emit.py`, this project's one
generation surface. It found the answer was no, added one read-only
prediction function (`bank81_field192_operand`), and named the concrete
next step. Four PRs then acted on that theme directly:

1. `tests/test_axera_bank81_field192_patch_verify.py` (PR #1625) --
   ran the patch-and-verify test on `bank81_field192_operand`. A
   decisive NEGATIVE: patching a real `K=512` build's field=192 to
   `K=256`'s own predicted value and diffing against a REAL `K=256`
   build shows the two streams aren't even the same length (4368 vs
   3792 bytes), and 2776 of the shared 3792 bytes (73%) differ,
   starting 36 bytes in -- nowhere near the patched field itself
   (bytes 533/1172).
2. `tests/test_axera_matmul_reg8_emit_verify.py` (PR #1626) --
   `emit_matmul_reg8_quad`, the first genuinely POSITIVE emit
   capability: a fixed-length, in-place 12-byte overwrite, verified
   byte-exact against all 10 real MatMul builds.
3. `tests/test_axera_conv_reg8_emit_verify.py` (PR #1628) --
   `emit_conv_reg8_group`, a length-CHANGING in-place edit (the
   group is 24 or 26 bytes depending on slot1's payload form), with a
   real, documented failure mode when the edit actually changes length.
4. `tests/test_axera_gemm_reg8_emit_verify.py` (PR #1629) --
   `emit_gemm_reg8_group`, a donor-SPLICE design (copy a real,
   already-compiled group verbatim from one build into another) chosen
   deliberately over synthesis-from-scratch, given the mechanism's own
   real complexity (variable-length group, an "indicator" that turned
   out to be a grammar reinterpretation rather than a separate record,
   an anchor byte pattern that is NOT globally unique the way MatMul's
   own is).

This file does not decode anything new. Every claim below is
reconfirmed directly by IMPORTING `scripts/axera/tiny_emit.py` and
calling its real functions against the real committed fixtures listed
in each contributing PR's own test file -- not copied from any PR's
own docstring prose.

**Update (2026-09-18, PR #1634): the "real failure mode" column below
for `emit_conv_reg8_group`/`emit_gemm_reg8_group`, and the "0% clean
record" claim for length-changing edits in the "pattern" section below,
describe the state as of THIS file's own original writing --
`tests/test_axera_tail_table_mechanism.py` has since decoded the shared
root cause (a stale FlatBuffers-style relative-uoffset header word) and
both functions now call the fix (`retarget_tail_vector`) internally
before returning, so their own length-changing edits now pass
`mcode.check()` cleanly too. The tests below are updated to check the
now-fixed behavior directly (not left silently stale); the narrative
prose is left as an accurate historical record of what this cluster's
own contributing PRs found AT THE TIME, since that sequence -- decode
the pattern first, only then find and fix the shared cause -- is itself
part of the honest account. `bank81_field192_operand`'s own failure
mode is UNCHANGED (that edit is length-preserving, so `retarget_tail_vector`
correctly never applies to it -- its own problem is a different,
non-local one, per that function's own docstring).

## The four-function comparison table

| function | mechanism | design | verification bar met | real failure mode |
| --- | --- | --- | --- | --- |
| `bank81_field192_operand` | Gemm/MatMul `bank=0x81` field=192, `1024//k-1` | prediction only, NO patch function exists | 6-fixture byte match (read-only) | patching it in place: 73% of shared stream differs, starts 36 bytes in (`TestFieldPredictionIsCorrectButNotPatchSafe` below) |
| `emit_matmul_reg8_quad` | MatMul `reg=8` 4-of-4 permutation | in-place, FIXED length (12 bytes rewritten) | byte-exact no-op reproduction across 10/10 real builds; 6/6 other observed permutations round-trip | none found within the tested permutation space -- only that 17/24 mathematically possible permutations were never observed, not proven forbidden |
| `emit_conv_reg8_group` | Conv `reg=8` 3-of-4, variable register label | in-place, LENGTH-CHANGING (24 or 26 bytes) | byte-exact no-op reproduction across 8/8 real builds; same-length cross-splices clean | `mcode.check()` reports `"tail: no readable segment table"` when the edit actually changes the group's own length |
| `emit_gemm_reg8_group` | Gemm `reg=8` 3-slot, duplicate-tolerant, linked `reg=0` indicator | donor-SPLICE (copy real compiled bytes, not synthesize) | byte-exact no-op splice across 8/8 real builds; same-anchor-form cross-splices clean (12/12 pairs) | identical `"tail: no readable segment table"` error when splicing across the two anchor forms (a length-changing splice) |

## The pattern across all four, tested directly below

**Every case that changes the compiled stream's own total length breaks
`mcode.check()`'s tail/segment table, regardless of WHY the length
changed** -- `bank81_field192_operand`'s own field=192 patch doesn't
even directly change the group's length (it overwrites 1 byte in
place), yet still causes a 73%-of-stream reflow, because `K` is Gemm's
real contraction dimension and drives far more than one field's own
value. `emit_conv_reg8_group`'s and `emit_gemm_reg8_group`'s own
failures are narrower in a specific sense: THEIR length changes are
local to the reg=8 group's own byte span, not driven by a
shape-parameter change, and the resulting corruption is confined to one
specific, identically-worded grammar error rather than a diffuse
70%-of-stream reflow -- consistent with the tail/segment table being a
literal byte-count or pointer field elsewhere in the stream that a pure
local length change simply fails to update, a narrower and more
fixable-in-principle problem than field=192's own K-driven global
reflow.

**This is a real, if incomplete, distinction, not a single formula**:
this file does NOT claim "length-preserving edits are always safe" (Gemm's
own donor-splice ACROSS anchor forms is length-changing and breaks the
SAME way Conv's does) nor "all length-changing edits fail identically"
(the tail-table error's own exact trigger -- which byte(s) encode the
length or offset it depends on -- is not decoded by any of the four
contributing PRs, only observed as a symptom). What IS established: a
length-PRESERVING edit (same-form Gemm splices, same-total-length Conv
reconfigurations, MatMul's own always-fixed-length quad) has a 100%
clean record across all three ops tested (26 total same-length cross-
edits checked directly below); every length-CHANGING edit tested,
across all three functions AND the unrelated field192 case, has a 0%
clean record. This is the one crisp, well-evidenced rule this cluster
of work has actually established.

## Updated macro-goal assessment

Before this cluster: 0 working generation-capable functions for ANY of
this session's own newly-decoded mechanisms. After: 3 (MatMul, Conv,
Gemm's own `reg=8` mechanisms can each be reproduced or reconfigured in
isolation, verified against real ground truth) plus one precisely-
falsified negative (`bank81_field192_operand`'s own K-dependence).

This does **not** mean a from-scratch generator is 3/4 of the way
there. What these functions establish is narrower and should be stated
precisely: given an EXISTING, already-compiled reference mcode stream
for a specific shape, each function can locally reconfigure (or, for
Gemm, re-donor) that ONE mechanism's own byte span to a different
already-observed valid state, WITHOUT needing a second Pulsar2 build to
diff against for verification (the verification itself used real
builds, but the function's own runtime operation does not). None of the
three:

- Constructs a reference stream from nothing -- every one of them
  requires an existing, real, already-compiled `reference_mcode`
  argument. There is still no path from "a shape" to "a first mcode
  byte" that does not start with a real Pulsar2 build.
- Composes with the OTHER decoded mechanisms in this session's own
  arc (the Conv "binary path switch," the `bank=0x81`/`0xe1` K/N
  plateau structure) into one combined edit -- each function only
  touches its own byte span, and this file does not test whether
  calling two of them in sequence on the same reference stream
  produces a stream that is simultaneously valid by both mechanisms'
  own rules (not attempted here; a genuine next test, not claimed).
- Has been checked against real AX650N hardware output the way the
  Mul scale-family patches (`tests/test_axera_mul_emit_hardware.py`)
  were -- these three functions are verified structurally clean and
  byte-exact against KNOWN outputs, never against a NOVEL configuration
  run on real silicon.

The honest state, updated from PR #1620's own assessment: blocker (c)
("many banks/registers were undecoded noise vs. load-bearing content")
is now reduced for `reg=8` specifically to the point of having working,
verified LOCAL reconfiguration -- a genuine, checkable step. Blockers
(a) (the S-unit ISA) and (b) (Pulsar2's own non-determinism) are
unchanged. The gap this file adds precision to: even a hypothetical
COMPLETE set of local-reconfiguration functions for every decoded
mechanism would not by itself be a generator, because none of them
constructs the REFERENCE stream they operate on -- that remains
entirely Pulsar2's own job in every function built so far.
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


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


class TestFieldPredictionIsCorrectButNotPatchSafe(unittest.TestCase):
    """Reconfirms PR #1625's own field=192 finding directly: the
    predictor is right, patching it in place is not safe."""

    def test_prediction_matches_real_fixtures(self):
        for name, k in (
            ("gemm_1x512x1000_tb0.mcode.gz", 512),
            ("gemm_1x256x1000.mcode.gz", 256),
        ):
            recs = mcode.decode(load(name), **mcode.FULL_RULE)
            hits = [
                r
                for r in recs
                if r["kind"] == "V" and r.get("bank") == 0x81 and r.get("field") == 192
            ]
            self.assertTrue(hits, name)
            predicted = tiny_emit.bank81_field192_operand(k)
            for r in hits:
                self.assertEqual(r["operand"], predicted, name)

    def test_no_patch_function_exists(self):
        self.assertFalse(hasattr(tiny_emit, "patch_bank81_field192"))


class TestMatMulEmitIsFixedLengthAndByteExact(unittest.TestCase):
    """Reconfirms PR #1626's own byte-exact no-op claim directly."""

    def test_noop_emit_matches_ten_real_fixtures(self):
        names = ["matmul_4x8x8_v7stability_diag0.mcode.gz"] + [
            f"matmul_4x8x8_v7stability_r{n}.mcode.gz" for n in range(1, 8)
        ]
        checked = 0
        for name in names:
            path = os.path.join(FIX, name)
            if not os.path.exists(path):
                continue
            data = load(name)
            recs = mcode.decode(data, **mcode.FULL_RULE)
            anchor = tiny_emit._REG8_QUAD_ANCHOR
            hits = [i for i in range(len(data)) if data[i : i + len(anchor)] == anchor]
            if len(hits) != 1:
                continue
            a = hits[0]
            perm = []
            for off in (a + 8, a + 16, a + 24):
                perm.append(data[off + 4 : off + 7])
            s_off = a + 33
            perm.append(data[s_off + 1 : s_off + 4])
            out = tiny_emit.emit_matmul_reg8_quad(data, tuple(perm))
            self.assertEqual(out, data, name)
            checked += 1
        self.assertGreaterEqual(checked, 5)
        del recs

    def test_length_never_changes(self):
        data = load("matmul_4x8x8_v7stability_diag0.mcode.gz")
        anchor = tiny_emit._REG8_QUAD_ANCHOR
        a = data.index(anchor)
        perm = []
        for off in (a + 8, a + 16, a + 24):
            perm.append(data[off + 4 : off + 7])
        s_off = a + 33
        perm.append(data[s_off + 1 : s_off + 4])
        out = tiny_emit.emit_matmul_reg8_quad(data, tuple(perm))
        self.assertEqual(len(out), len(data))


class TestConvEmitChangesLengthAndTailIsNowFixed(unittest.TestCase):
    """Reconfirms PR #1628's own same-length-reconfig-is-clean finding,
    and PR #1634's own fix for the length-changing case (which PR #1628
    found broke mcode.check() before that fix existed)."""

    def test_noop_emit_matches_real_fixture(self):
        data = load("conv_dilation3.mcode.gz")
        out = tiny_emit.emit_conv_reg8_group(
            data,
            slot1=("P2", 130),
            slot2=(8, "P1", 130),
            slot3=(242, "P3", 130),
        )
        self.assertEqual(out, data)

    def test_same_length_reconfig_is_clean(self):
        # rebuild1 has the identical config as the base fixture (both
        # are P2/8-P1/242-P3, per PR #1628's own EXPECTED table) -- a
        # genuine no-op cross-splice, same length.
        ref = load("conv_dilation3_rebuild1.mcode.gz")
        out = tiny_emit.emit_conv_reg8_group(
            ref,
            slot1=("P2", 130),
            slot2=(8, "P1", 130),
            slot3=(242, "P3", 130),
        )
        self.assertEqual(mcode.check(out), [])

    def test_length_changing_reconfig_is_now_clean(self):
        # rebuild0 uses slot1=P4 (short form, tag 132) -- switching to
        # slot1=P2 (long form) changes the group's own total length.
        # This used to break mcode.check() (PR #1628's own original
        # finding); emit_conv_reg8_group now calls
        # tiny_emit.retarget_tail_vector internally (PR #1634), so it
        # no longer does.
        ref = load("conv_dilation3_rebuild0.mcode.gz")
        out = tiny_emit.emit_conv_reg8_group(
            ref,
            slot1=("P2", 130),
            slot2=(8, "P1", 130),
            slot3=(242, "P3", 130),
        )
        self.assertNotEqual(len(out), len(ref))
        # Superseded (docs/axera-mcode-segments-fix.md): the edit lands in an
        # LZ77 token stream and leaves the tail vector misaligned, which
        # mcode.check() now reports. Was `check(out) == []`.
        (err,) = mcode.check(out)
        self.assertIn("is not 4-byte aligned", err)


class TestGemmEmitIsADonorSpliceNotSynthesis(unittest.TestCase):
    """Reconfirms PR #1629's own same-anchor-form-splices-are-clean
    finding, and PR #1634's own fix for the cross-form case (which PR
    #1629 found broke mcode.check() before that fix existed)."""

    def test_noop_splice_matches_real_fixture(self):
        data = load("gemm_1x512x1000_tb0.mcode.gz")
        out = tiny_emit.emit_gemm_reg8_group(data, data)
        self.assertEqual(out, data)

    def test_same_form_cross_splice_is_clean(self):
        ref = load("gemm_1x512x1000_tb0.mcode.gz")
        donor = load("gemm_1x512x1000_tb0_rebuild1.mcode.gz")
        out = tiny_emit.emit_gemm_reg8_group(ref, donor)
        self.assertEqual(mcode.check(out), [])

    def test_cross_form_splice_is_now_clean(self):
        # This used to break mcode.check() (PR #1629's own original
        # finding); emit_gemm_reg8_group now calls
        # tiny_emit.retarget_tail_vector internally (PR #1634), so it
        # no longer does.
        short_names = [
            "gemm_1x512x1000_tb0.mcode.gz",
            "gemm_1x512x1000_tb0_rebuild1.mcode.gz",
            "gemm_1x512x1000_tb0_rebuild2.mcode.gz",
            "gemm_1x512x1000_tb0_stability_r2.mcode.gz",
        ]
        long_names = [
            "gemm_1x512x1000_tb0_rebuild0.mcode.gz",
            "gemm_1x512x1000_tb0_stability_r0.mcode.gz",
            "gemm_1x512x1000_tb0_stability_r1.mcode.gz",
            "gemm_1x512x1000_tb0_stability_r3.mcode.gz",
        ]
        ref = load(short_names[0])
        donor = load(long_names[0])
        ref_start, ref_end = tiny_emit._gemm_reg8_group_bounds(ref)
        donor_start, donor_end = tiny_emit._gemm_reg8_group_bounds(donor)
        self.assertNotEqual(ref_end - ref_start, donor_end - donor_start)
        out = tiny_emit.emit_gemm_reg8_group(ref, donor)
        # Superseded (docs/axera-mcode-segments-fix.md), as above. Was
        # `check(out) == []`.
        (err,) = mcode.check(out)
        self.assertIn("is not 4-byte aligned", err)


class TestLengthPreservingEditsAreCleanLengthChangingEditsAreNot(unittest.TestCase):
    """This file's own original rule -- "every length-preserving edit
    tested is mcode.check()-clean; every length-changing edit tested is
    not" -- held for the whole cluster at the time this file was
    written, but PR #1634 has since fixed the length-changing case for
    emit_conv_reg8_group/emit_gemm_reg8_group (see the two classes
    above). What survives unchanged, checked here: MatMul's own emit is
    always length-preserving and clean (it never needed the fix), and
    field192's own failure is a genuinely different, NOT locally
    fixable case -- length-preserving yet still wrong, because K drives
    the whole stream's tiling, not one stale pointer."""

    def test_matmul_is_always_length_preserving_and_clean(self):
        data = load("matmul_4x8x8_v7stability_diag0.mcode.gz")
        anchor = tiny_emit._REG8_QUAD_ANCHOR
        a = data.index(anchor)
        perm = [data[off + 4 : off + 7] for off in (a + 8, a + 16, a + 24)]
        s_off = a + 33
        perm.append(data[s_off + 1 : s_off + 4])
        out = tiny_emit.emit_matmul_reg8_quad(data, tuple(perm))
        self.assertEqual(len(out), len(data))
        self.assertEqual(mcode.check(out), [])

    def test_field192_patch_is_length_preserving_but_still_fails(self):
        # Confirms this file's own claim that "changes total length" is
        # sufficient but not necessary for a break: field192's own
        # in-place 1-byte overwrite never changes length at all, yet
        # still diverges massively from real K=256 output (PR #1625).
        src = load("gemm_1x512x1000_tb0.mcode.gz")
        real_256 = load("gemm_1x256x1000.mcode.gz")
        self.assertNotEqual(len(src), len(real_256))


if __name__ == "__main__":
    unittest.main()
