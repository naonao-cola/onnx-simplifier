"""Continuing `tests/test_axera_sparse_resource_clustering.py` (PR #1565)'s
own flagged gap -- pinning the exact size threshold where Gemm's sparse
bank `0x81` first appears -- and finding the premise itself needs a real
correction along the way: it is not a raw-mcode-length threshold at all.

PR #1565 found `gemm_1x512x1000_tb0.mcode.gz` (4,368 bytes, a real
resnet18-FC-layer-sized shape) carries sparse bank `0x81`, while three
smaller Gemm shapes (2,568-3,016 bytes) don't, and concluded size (not
op label) predicts sparse-resource usage. That comparison only varied
total shape/length as one bundled quantity -- this file isolates `N`
specifically (holding `M=1, K=512` fixed, matching the FC shape's own
`K`) and finds the raw-length story does not survive a proper bisection.

## Bisection 1: `N=20` and `N=40` are BOTH 2,632 bytes -- but only one
## has bank `0x81`

Sweeping `N` from 8 up to 1000 at fixed `M=1, K=512` (`pulsar2_docker.build()`,
same convention as `tests/test_axera_gemm_transb_scaling.py`'s own FC-shape
build) gives, among others:

| N | len | bank `0x81` | bank `0xe1` |
| --- | --- | --- | --- |
| 20 | 2632 | no | yes |
| 40 | 2632 | **yes** | no |
| 60 | 2632 | yes | no |
| 80 | 2928 | **no** | yes |

`N=20` and `N=40` are the *identical* 2,632-byte length PR #1565's own
`gemm_8x8x8_tb0` non-carrier fixture also has -- yet one has bank `0x81`
and the other doesn't. `N=80` is *larger* than `N=40`/`60` (2,928 vs
2,632 bytes) yet does NOT have `0x81`. Raw mcode length does not predict
this bank's presence; something else does.

## It's not a new resource "unlocking" -- bank `0x81` and bank `0xe1`
## are mutually exclusive substitutes at this shape family

Every build checked has *exactly one* of the two: `0x81` xor `0xe1`,
never both, never neither. This reframes the question again: PR #1565's
"second resource tier" framing (implying additional resources on top of
the core set) is not quite right for this specific pair -- it looks more
like two alternate identities for the same functional slot, with the
compiler picking one or the other depending on something about the
build.

## Confirmed real and deterministic, not allocator noise -- 3 rebuilds
## each way, zero exceptions

Given this project's repeated finding that superficially input-dependent
bank/register choices can turn out to be unconditioned per-compile coin
flips (`tests/test_axera_matmul_offset_table_coinflip.py`,
`tests/test_axera_universal_bank_reg_stability.py`'s own `reg=8` finding),
the obvious first check is whether `N=20` vs `N=40`'s bank choice is
itself noise. It is not: 3 independent rebuilds of `N=20` (all `0xe1`,
never `0x81`) and 3 of `N=40` (all `0x81`, never `0xe1`) show zero
exceptions -- this is real, `N`-driven, deterministic content.

## The exact boundary: `N=32` uses `0xe1`, `N=33` uses `0x81` -- and 32
## is suspicious

Bisecting the `N=20..40` gap (`N=25,28,30,32` all `0xe1`; `N=33,34,35`
all `0x81`) pins the transition to exactly between `N=32` and `N=33`,
confirmed with an independent rebuild on each side (`N=32` rebuild:
`0xe1` again; `N=33` rebuild: `0x81` again -- see the fixtures below).
`N=32 = 2**5` is not an arbitrary-looking number -- consistent with (not
proof of) a hardware tile/vector width of 32 for this dimension, the
same kind of power-of-two boundary this project has repeatedly found
elsewhere (MatMul's byte-truncation switch at `1/A_scale=128=2**7`,
`tests/test_axera_matmul_quad_form_switch.py`; the many `K`-regime
`0x81`-adjacent 32-multiple plateaus in
`tests/test_axera_gemm_km1_k_regime_check.py`).

## What remains open

This pins the boundary at exactly `M=1, K=512`, sweeping `N` alone --
whether the threshold is `N` in absolute terms, or `N` relative to `K`
(a ratio, or `N*K` total), or something about `M=1` specifically, is
untested: no second `K` value was swept here to check whether the
`N=32/33` boundary shifts. What `0x81`/`0xe1`'s own fields *compute*, and
why the compiler picks one over the other at exactly this size, are also
not decoded here -- this file only pins precisely *where* the switch
happens and confirms it is real, not *why* 32 specifically, or what the
two forms encode.
"""

import gzip
import os
import sys
import unittest

_AXERA_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts", "axera")
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import mcode  # noqa: E402

FIX = os.path.join(_AXERA_DIR, "fixtures")


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def banks_of(data):
    recs = mcode.decode(data, **mcode.FULL_RULE)
    return set(r["bank"] for r in recs if r["kind"] == "V")


class TestBoundaryIsWellFormedAndSameLength(unittest.TestCase):
    def test_all_four_fixtures_check_clean(self):
        for name in (
            "gemm_1x512x32.mcode.gz",
            "gemm_1x512x32_rebuild.mcode.gz",
            "gemm_1x512x33.mcode.gz",
            "gemm_1x512x33_rebuild.mcode.gz",
        ):
            self.assertEqual(mcode.check(load(name)), [], name)

    def test_n32_and_n33_are_the_identical_length(self):
        """The boundary is not a length effect: N=32 and N=33 compile to
        the exact same total mcode length, only the bank choice differs."""
        self.assertEqual(
            len(load("gemm_1x512x32.mcode.gz")), len(load("gemm_1x512x33.mcode.gz"))
        )


class TestN32UsesBankE1NotBank81(unittest.TestCase):
    def test_original_and_rebuild_both_use_0xe1(self):
        for name in ("gemm_1x512x32.mcode.gz", "gemm_1x512x32_rebuild.mcode.gz"):
            banks = banks_of(load(name))
            self.assertIn(0xE1, banks, name)
            self.assertNotIn(0x81, banks, name)


class TestN33UsesBank81NotBankE1(unittest.TestCase):
    def test_original_and_rebuild_both_use_0x81(self):
        for name in ("gemm_1x512x33.mcode.gz", "gemm_1x512x33_rebuild.mcode.gz"):
            banks = banks_of(load(name))
            self.assertIn(0x81, banks, name)
            self.assertNotIn(0xE1, banks, name)


class TestChoiceIsMutuallyExclusiveNeverBoth(unittest.TestCase):
    def test_exactly_one_of_the_two_banks_present_at_each_n(self):
        for name in (
            "gemm_1x512x32.mcode.gz",
            "gemm_1x512x32_rebuild.mcode.gz",
            "gemm_1x512x33.mcode.gz",
            "gemm_1x512x33_rebuild.mcode.gz",
        ):
            banks = banks_of(load(name))
            has81 = 0x81 in banks
            hase1 = 0xE1 in banks
            self.assertNotEqual(
                has81, hase1, f"{name}: expected exactly one of the two"
            )


if __name__ == "__main__":
    unittest.main()
