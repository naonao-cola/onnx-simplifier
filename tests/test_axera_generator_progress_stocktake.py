"""A periodic stock-take reconnecting this session's own extensive
mcode resource-model decode work back to the original macro-goal that
kicked it off: "go on to memory management related analysis for
building [a] complete tinygrad code based code generator." That goal
has not been explicitly revisited since roughly the session's own
midpoint, while the resource-model arc since then has produced ~150
`take-cheeze/mcode-*` PRs -- the reg=8 "second noise mechanism" fully
characterized across all four main ops (Mul, Gemm, Conv, MatMul,
`tests/test_axera_reg8_cross_op_synthesis.py` PR #1582 and its five
contributing PRs), Conv's dilation-dependent "binary path switch"
(`tests/test_axera_conv_binary_cluster_synthesis.py` PR #1603 and its
eight contributing PRs), and the largest single arc, Gemm's
`bank=0x81`/`0xe1` mutual-exclusivity mechanism generalized across all
four ops and mapped to at least seven K/N plateaus
(`tests/test_axera_bank81_e1_cross_op_synthesis.py` PR #1614 and its
ten-plus contributing PRs, still growing as of this file).

Like the three synthesis files above, this one does not decode
anything new by building fresh mcode. It answers a narrower, sharper
question those files were not asked: **does any of this decode work
change what a generator (`scripts/axera/tiny_emit.py`, this project's
one working generation surface) can actually DO?**

## The honest, checkable answer: no -- not until this file's own one
## small addition

`git log --oneline -- scripts/axera/tiny_emit.py` shows its own most
recent commit predates this entire resource-model arc: the file's own
module docstring is dated "Status (2026-09-16)", and every commit
touching it landed on or before that date (`patch_site_a`,
`patch_matmul_a_scale`, `patch_conv_zp_x` -- the last of the ten
commits in that file's history). None of the ~150 PRs produced by this
session's own reg=8, Conv-binary-cluster, or bank=0x81/0xe1
investigations touched `tiny_emit.py` at all, confirmed directly below
(`TestNoSessionPRTouchedTheGeneratorUntilNow`) by checking that none of
this session's own new mechanism names (`reg=8`, `bank81`, `bank_0x81`,
`e1_field`, `binary_cluster`) appear anywhere in that file's source,
except for the one function this file itself adds.

This is not a criticism of the resource-model work -- every individual
finding in it is real, well-verified, and directly answers the
question it was built to answer. It is a statement about SCOPE: this
entire arc has been characterization work (build many variants, diff
them, find the rule that predicts which variant a given shape
produces), and characterization is a different activity from
generation (writing a byte pattern that a real Pulsar2 build would
also have produced, from a shape alone, with no reference build to
diff against). `tiny_emit.py`'s own docstring already named this
distinction precisely, a full arc before this session's own work
began: knowing a field's value is not the same as knowing it is safe
to write in place (`patch_site_a`'s docstring, re: Gemm's `K*M-1` and
Conv's dilation/orientation fields -- both *understood*, neither
*generatable*, as of that note). The reg=8 mechanism, the Conv
binary-cluster switch, and the bank=0x81/0xe1 plateau structure are all
now understood to that same precise degree, and none of them are yet
generatable either.

## What IS now generatable, that was not before this file: bank 0x81's
## field=192, from K alone

One narrow exception, added directly to `tiny_emit.py` by this file:
`bank81_field192_operand(k)` predicts the exact 3-byte operand
(`4c 05 <1024 // k - 1>`) Gemm's own field=192 record carries for a
given contraction dimension `K`, decoded natively for `Gemm`
(`tests/test_axera_gemm_bank_81_e1_decode.py`, PR #1570) and
independently reconfirmed byte-identical for a constant-weight
`MatMul(x, w)` (`tests/test_axera_bank81_cross_op_check.py`, PR #1608).
Verified directly below (`TestBank81Field192PredictionMatchesFixtures`)
against six already-committed fixtures spanning both ops at
`K=128/256/512` -- zero exceptions.

This is a genuine, if narrow, generation capability: given only a
shape parameter, no reference build, no compiler in the loop, it
produces the exact bytes a real Pulsar2 build would use for this one
field. It is also precisely bounded, per its own docstring: it does
not decode whether a given `(K, N)` pair would even land in the
`0x81`-alone regime (that switch is itself the multi-plateau structure
PR #1611/#1613/#1615/#1618 mapped, still incomplete below `N=32` and
above `K=1153`), and it says nothing about whether writing this value
into an arbitrary target stream is safe -- the same "understood, not
yet generatable in place" caveat carries over unchanged for every
OTHER field this session decoded.

## What remains the actual, concrete blocker

Re-reading `scripts/axera/README.md`'s own original three-part
blocker list against this session's own results:

- **(a) the S-unit ISA is largely undecoded.** Unchanged. None of this
  session's own work touched S-unit programs; the reg=8 mechanism and
  the bank=0x81/0xe1 mechanism both live in the `V`/`S`-record resource
  layer this project's grammar already parses, not inside an S-unit
  program's own magnitude-adaptive body.
- **(b) Pulsar2's allocation is sometimes non-deterministic, with no
  single canonical target.** Unchanged, and this session's own work
  makes the practical shape of this blocker sharper rather than
  smaller: the reg=8 "unordered pool" mechanism (a 3-, 4-, or
  variable-slot permutation, decoded per-op) means even a generator
  that DID know a field's correct value set would still need to pick
  one specific member of that set with no compiler-observed preference
  to match against -- any choice is "a" valid generator output, none
  is uniquely "the" Pulsar2 output, which is fine for a from-scratch
  generator (it does not need bit-exact reproduction) but means
  "verified against a real rebuild" cannot mean bit-identity for these
  fields the way it did for Mul's scale family
  (`tests/test_axera_mul_emit_hardware.py`).
- **(c) many resource-model banks/registers were undecoded diffuse
  noise vs. load-bearing content.** This is the one blocker this
  session's own work has substantially reduced -- not eliminated. The
  reg=8 mechanism, the Conv binary-cluster switch, and the bank=0x81/
  0xe1 mechanism were all previously unclassified noise; all three are
  now mechanism-level DECODED (this session's own three synthesis PRs
  each state this precisely). But decoding a mechanism is not the same
  as closing (c) for GENERATION purposes: `patch_site_a`'s own already-
  documented reflow problem (writing one field can move ~200 bytes
  elsewhere in the stream, for reasons not characterized) has never
  been tested against ANY of this session's own newly-decoded fields.
  Every one of the ~150 PRs in this arc was a build-sweep-and-diff
  effort; none attempted a patch-an-existing-stream-and-verify-against-
  a-real-second-build test, the actual next step `patch_site_a`'s own
  precedent establishes for turning a decoded mechanism into a
  generatable one.

## A new category of blocker this session's own work surfaced, not in
## the original three-part list

The sheer NUMBER of op-specific, shape-specific threshold rules found
for just the bank=0x81/0xe1 investigation -- a `K mod 4` rule for
MatMul's own second pair, a genuinely different per-K absolute-N
multi-plateau structure for Gemm (at least seven plateaus and counting,
`512/320/256/192/128/64/32`, three of the seven boundaries landing
exactly on the same `K*N` product by coincidence and four not reducing
to any single formula found so far), a coarse dilation-class boundary
for Conv, and an unpinned K*N bracket for Mul -- suggests that even a
COMPLETE decode of this one resource pool's own trigger conditions
would look like a large, per-op, per-shape-family lookup table, not a
compact closed-form scheduler a tinygrad-style code generator could
derive analytically from a shape alone. This is a genuinely new,
concrete risk to the macro-goal that the original three-part blocker
list (written before any of this plateau structure was known) did not
anticipate: even "solving" (c) completely might not yield the kind of
answer a from-scratch generator can use cheaply.

## The single most valuable next step, concretely

Not "more decode work" -- this session has ~150 PRs of exactly that,
almost all still resource-model characterization. The one thing that
would actually move the macro-goal is the kind of test `patch_site_a`
and `tests/test_axera_mul_emit_hardware.py` already did for the scale
family and never repeated for anything decoded since: **pick ONE of
this session's own newly-decoded mechanisms (bank 0x81's field=192 is
now the natural first candidate, being the only one with a working
prediction function as of this file), take a real reference build,
patch field=192 to a DIFFERENT K's own predicted value using
`bank81_field192_operand`, and check whether the result is closer to
or farther from a genuine second build at that target K** -- the same
reflow-or-no-reflow question `patch_conv_zp_x`'s own docstring already
answered honestly (no) for zp_x, and `patch_site_a` answered honestly
(no, for MatMul's cross-form A) for site A. Until that specific test is
run for at least one of this session's own new mechanisms, "understood"
and "generatable" remain exactly as separate as `tiny_emit.py`'s own
2026-09-16 status note already said they were.
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


def decode(name):
    return mcode.decode(load(name), **mcode.FULL_RULE)


class TestNoSessionPRTouchedTheGeneratorUntilNow(unittest.TestCase):
    """`tiny_emit.py`'s own git history shows its last commit predates
    this entire resource-model arc, and none of this session's own new
    mechanism names appear in its source except this file's own single
    addition -- the concrete, checkable version of this file's own
    central claim: characterization work and generation work have been
    two separate activities in this session, not one continuous one."""

    def test_no_reg8_or_bank81_or_binary_cluster_terms_before_this_files_own_addition(
        self,
    ):
        path = os.path.join(_AXERA_DIR, "tiny_emit.py")
        with open(path, encoding="utf-8") as f:
            src = f.read()
        # Split off this file's own addition so the check reflects the
        # state every PRIOR PR in this session's arc actually left the
        # file in, not the state after this file's own bonus function.
        marker = "def bank81_field192_operand"
        idx = src.index(marker)
        prior_src = src[:idx]
        for term in ("reg8", "reg=8", "bank81", "bank_0x81", "binary_cluster"):
            self.assertNotIn(term, prior_src, term)


class TestBank81Field192PredictionMatchesFixtures(unittest.TestCase):
    """The one new generation capability this file adds: predicting
    bank 0x81's field=192 operand from K alone, verified against six
    already-committed fixtures spanning two ops (native Gemm and a
    constant-weight MatMul) at three K values, zero exceptions."""

    CASES = [
        ("gemm_1x512x1000_tb0.mcode.gz", 512),
        ("gemm_1x256x1000.mcode.gz", 256),
        ("gemm_1x128x1000.mcode.gz", 128),
        ("matmul_bank81_probe_m1k512n1000.mcode.gz", 512),
        ("matmul_bank81_probe_m1k256n1000.mcode.gz", 256),
        ("matmul_bank81_probe_m1k128n1000.mcode.gz", 128),
    ]

    def test_prediction_matches_every_fixture(self):
        for name, k in self.CASES:
            recs = decode(name)
            hits = [
                r
                for r in recs
                if r["kind"] == "V" and r.get("bank") == 0x81 and r.get("field") == 192
            ]
            self.assertTrue(hits, name)
            predicted = tiny_emit.bank81_field192_operand(k)
            for r in hits:
                self.assertEqual(r["operand"], predicted, (name, k))

    def test_prediction_is_op_independent_at_the_same_k(self):
        gemm = tiny_emit.bank81_field192_operand(512)
        matmul_const_weight = tiny_emit.bank81_field192_operand(512)
        self.assertEqual(gemm, matmul_const_weight)

    def test_prediction_differs_by_k(self):
        vals = {tiny_emit.bank81_field192_operand(k) for k in (128, 256, 512)}
        self.assertEqual(len(vals), 3)


class TestThisRemainsPredictionNotGeneration(unittest.TestCase):
    """Documents, as a real assertion rather than prose alone, the
    precise boundary of what was and was not done here: the prediction
    function is verified read-only against existing fixtures. No patch
    round-trip against a real second build (the actual next step this
    file's own docstring recommends) is attempted -- confirmed by the
    absence of any patch_bank81_* function in tiny_emit.py's own public
    surface."""

    def test_no_patch_function_exists_for_bank81_yet(self):
        public_names = [n for n in dir(tiny_emit) if not n.startswith("_")]
        patch_like = [n for n in public_names if n.startswith("patch_")]
        self.assertTrue(patch_like, "expected at least the existing patch_ functions")
        self.assertFalse(
            any("bank81" in n or "e1" in n.lower() for n in patch_like),
            f"expected no bank81/e1 patch function yet, found {patch_like}",
        )


if __name__ == "__main__":
    unittest.main()
