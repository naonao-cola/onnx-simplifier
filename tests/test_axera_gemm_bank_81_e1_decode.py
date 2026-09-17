"""Continuing `tests/test_axera_gemm_sparse_bank_n_boundary.py` (PR #1568)'s
own explicitly flagged gap: that file pinned *where* Gemm's mutually
exclusive sparse banks `0x81` (large `N`) and `0xe1` (small `N`) switch
at fixed `M=1, K=512` (exactly between `N=32` and `N=33`), but left
"what `0x81`/`0xe1`'s own fields *compute*" fully open. This file
decodes that content directly.

## Bank `0x81`, field=192: the third operand byte is a clean function
## of `K` alone -- `1024 // K - 1`

At every large-`N` shape checked, bank `0x81`'s field=192 record has a
3-byte operand whose first two bytes are constant (`4c 05`) and whose
third byte varies only with `K`:

| `K` | operand | third byte | `1024 // K - 1` |
| --- | --- | --- | --- |
| 128 | `4c 05 07` | 7 | 7 |
| 256 | `4c 05 03` | 3 | 3 |
| 512 | `4c 05 01` | 1 | 1 |

Confirmed at `K=512` across three very different `N` values (`N=33,
100, 1000` -- and 3 independent `N=1000` rebuilds) all giving the
*identical* operand `4c 05 01` -- ruling out `N` as a contributor and
isolating this byte as a pure function of `K`. Also confirmed with a
dedicated independent rebuild of `m1k512n100` under a different RNG
seed (seed=2, vs. the presumed seed=1 used for the rest of this
project's Gemm probes) -- same operand, same field, same offset
pattern: this is real compiled content, not per-build allocator noise.

`1024 // K` has a natural reading: `1024` is 8x this project's own
already-established Gemm `K`-regime widening boundary (`K>=65` rounds
up to the next multiple of 32 in the *separate*, already-decoded `K*M-1`
field -- `tests/test_axera_gemm_km1_k_regime_check.py`), consistent
with (not proof of) a fixed total-tile-budget divided by `K`'s own
tile width. Not decoded further here -- this file only establishes the
formula, not its hardware rationale.

## Bank `0xe1`, field=112: a constant operand, independent of `N`, `M`,
## and `K` (within this project's own already-tested small-`N` set)

Every small-`N` (`N<=32`) Gemm fixture in this corpus that carries bank
`0xe1` has the exact same field=112 record: operand `35 81 1a`, at
every `M` in `{1, 4, 8, 16, 32}` tested and every `K` in
`{32, 33, 64, 512}` tested. This is a fixed control-word for this
functional slot's "small" form, not an encoding of any of this op's
own shape parameters -- consistent with `0x81`'s field=192 slot being
the *only* shape-dependent content in this pair, with `0xe1`'s
field=112 slot instead a plain constant marker for "use the small-N
form."

## What remains genuinely open (not forced into a false pattern)

Two secondary phenomena are visible in this same fixture set but not
cleanly explained by anything checked here, and are recorded honestly
as open rather than claimed as decoded:

- At large `N` (`N>=100` in this dataset), two *additional* `0xe1`
  records appear alongside the primary `0x81` bank -- field=32
  (`33 03 00`) and field=48 (`35 03 00 a1`) -- constant-valued and
  present at every large-`N` shape checked (`K` in `{128, 256, 512}`,
  `N` in `{100, 1000}`).
- A further `0x81` field=144 record (`08 03 00 a1`) appears in *some*
  but not all of those same large-`N` shapes -- present at
  `(K=128, N=1000)` and `(K=512, N=1000)`, absent at `(K=256, N=1000)`
  and `(K=512, N=100)`. No simple function of `K`, `N`, or `N/K`
  checked here explains this split (`N/K` is 7.8, 3.9, 1.95, and 0.195
  respectively for those four cases -- present at the two extremes of
  that range and absent at the two in between, not a threshold).

This file does not attempt to resolve either point; both remain fully
open for future work, same as the still-untested "does the `N=32/33`
boundary itself shift with `K`" question `tests/test_axera_gemm_sparse_bank_n_boundary.py`
already flagged.

## Corpus footnote: this pair isn't universal even among small `N`

A handful of pre-existing fixtures with `N=8` fixed (`gemm_8x8x8_m8`,
`gemm_1x8x8_m1`, `gemm_4x96x8_m4k96`) carry *neither* `0x81` nor `0xe1`
at all -- they use a different sparse bank (`0x84`) instead, while
other `N=8` fixtures at different `K` (`gemm_4x33x8_m4k33`,
`gemm_4x64x8_m4k64`) *do* carry `0xe1`. This means the `0x81`/`0xe1`
pair's own *presence* (as opposed to which of the two, decoded above)
depends on something beyond `N` alone that this file does not
investigate -- `tests/test_axera_gemm_sparse_bank_n_boundary.py`'s own
boundary work holds at its own fixed `M=1, K=512` slice; it is not
being claimed here to generalize to arbitrary `K`.
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


def records_of(data, bank, field):
    recs = mcode.decode(data, **mcode.FULL_RULE)
    return [
        r
        for r in recs
        if r["kind"] == "V" and r["bank"] == bank and r["field"] == field
    ]


class TestNewFixturesCheckClean(unittest.TestCase):
    def test_all_new_fixtures_check_clean(self):
        for name in (
            "gemm_1x128x1000.mcode.gz",
            "gemm_1x256x1000.mcode.gz",
            "gemm_1x512x100.mcode.gz",
            "gemm_1x512x100_rebuild.mcode.gz",
        ):
            self.assertEqual(mcode.check(load(name)), [], name)


class TestBank81Field192EncodesK(unittest.TestCase):
    """Third operand byte of bank 0x81's field=192 record ==
    1024 // K - 1, independent of N."""

    def _third_byte(self, name):
        # Some fixtures carry this same (bank, field) record more than
        # once (e.g. once per matching S-unit) -- always identical in
        # every case checked, so assert that and read the common value.
        recs = records_of(load(name), 0x81, 192)
        self.assertGreaterEqual(len(recs), 1, name)
        operands = {r["operand"] for r in recs}
        self.assertEqual(len(operands), 1, f"{name}: expected identical duplicates")
        operand = next(iter(operands))
        self.assertEqual(operand[:2], b"\x4c\x05", name)
        return operand[2]

    def test_k128_gives_7(self):
        self.assertEqual(self._third_byte("gemm_1x128x1000.mcode.gz"), 1024 // 128 - 1)

    def test_k256_gives_3(self):
        self.assertEqual(self._third_byte("gemm_1x256x1000.mcode.gz"), 1024 // 256 - 1)

    def test_k512_gives_1_regardless_of_n(self):
        # K=512 held fixed, N varies 33 -> 100 -> 1000 (+ 3 independent
        # N=1000 rebuilds): the byte never changes, isolating it as a
        # pure function of K, not N.
        for name in (
            "gemm_1x512x33.mcode.gz",
            "gemm_1x512x33_rebuild.mcode.gz",
            "gemm_1x512x100.mcode.gz",
            "gemm_1x512x100_rebuild.mcode.gz",
            "gemm_1x512x1000_tb0.mcode.gz",
            "gemm_1x512x1000_tb0_rebuild0.mcode.gz",
            "gemm_1x512x1000_tb0_rebuild1.mcode.gz",
            "gemm_1x512x1000_tb0_rebuild2.mcode.gz",
        ):
            self.assertEqual(self._third_byte(name), 1024 // 512 - 1, name)

    def test_independent_rebuild_with_different_seed_matches(self):
        """gemm_1x512x100_rebuild.mcode.gz was built with RNG seed=2,
        deliberately different from this project's usual seed=1
        convention -- confirms the byte is real compiled content, not
        a coincidence of one specific random calibration/weight draw."""
        self.assertEqual(
            self._third_byte("gemm_1x512x100.mcode.gz"),
            self._third_byte("gemm_1x512x100_rebuild.mcode.gz"),
        )


class TestBankE1Field112IsConstant(unittest.TestCase):
    """Bank 0xe1's field=112 record is the fixed operand 35 81 1a at
    every small-N (N<=32) shape checked, regardless of M or K."""

    EXPECTED = b"\x35\x81\x1a"

    def _operand(self, name):
        recs = records_of(load(name), 0xE1, 112)
        self.assertGreaterEqual(len(recs), 1, name)
        operands = {r["operand"] for r in recs}
        self.assertEqual(len(operands), 1, f"{name}: expected identical duplicates")
        return next(iter(operands))

    def test_constant_across_m_and_k(self):
        for name in (
            "gemm_1x512x32.mcode.gz",
            "gemm_1x512x32_rebuild.mcode.gz",
            "gemm_16x8x8_m16.mcode.gz",
            "gemm_32x8x8_m32.mcode.gz",
            "gemm_4x32x8_k32.mcode.gz",
            "gemm_4x32x8_m4k32.mcode.gz",
            "gemm_4x33x8_m4k33.mcode.gz",
            "gemm_4x64x8_m4k64.mcode.gz",
            "gemm_8x32x8_m8k32.mcode.gz",
            "gemm_8x33x8_m8k33.mcode.gz",
            "gemm_8x64x8_m8k64.mcode.gz",
            "gemm_8x64x8_m8k64_rebuild.mcode.gz",
        ):
            self.assertEqual(self._operand(name), self.EXPECTED, name)


class TestLargeNSecondaryRecords(unittest.TestCase):
    """The two extra 0xe1 records (field=32, field=48) that show up
    alongside bank 0x81 at large N -- constant-valued, present at every
    large-N shape checked here."""

    def test_field32_and_field48_present_and_constant_at_large_n(self):
        for name in (
            "gemm_1x128x1000.mcode.gz",
            "gemm_1x256x1000.mcode.gz",
            "gemm_1x512x100.mcode.gz",
            "gemm_1x512x100_rebuild.mcode.gz",
            "gemm_1x512x1000_tb0.mcode.gz",
        ):
            f32 = records_of(load(name), 0xE1, 32)
            f48 = records_of(load(name), 0xE1, 48)
            self.assertGreaterEqual(len(f32), 1, name)
            self.assertGreaterEqual(len(f48), 1, name)
            for r in f32:
                self.assertEqual(r["operand"], b"\x33\x03\x00", name)
            for r in f48:
                self.assertEqual(r["operand"], b"\x35\x03\x00\xa1", name)

    def test_field144_presence_is_inconsistent_not_a_clean_function(self):
        """Honest negative result: field=144's presence does not follow
        N, K, or N/K alone across these four shapes."""
        present = {
            "gemm_1x128x1000.mcode.gz": True,  # N/K = 7.8
            "gemm_1x256x1000.mcode.gz": False,  # N/K = 3.9
            "gemm_1x512x100.mcode.gz": False,  # N/K = 0.195
            "gemm_1x512x1000_tb0.mcode.gz": True,  # N/K = 1.95
        }
        for name, expected in present.items():
            recs = records_of(load(name), 0x81, 144)
            self.assertEqual(bool(recs), expected, name)
            for r in recs:
                self.assertEqual(r["operand"], b"\x08\x03\x00\xa1", name)


class TestNeitherBankIsNotUniversalAmongSmallN(unittest.TestCase):
    """Some small-N fixtures use neither 0x81 nor 0xe1 at all (bank
    0x84 instead) -- the pair's own presence depends on more than N,
    outside this file's scope to resolve."""

    def test_some_n8_fixtures_have_neither(self):
        for name in (
            "gemm_8x8x8_m8.mcode.gz",
            "gemm_1x8x8_m1.mcode.gz",
            "gemm_4x96x8_m4k96.mcode.gz",
        ):
            data = load(name)
            self.assertEqual(records_of(data, 0x81, 192), [], name)
            self.assertEqual(records_of(data, 0xE1, 112), [], name)

    def test_other_n8_fixtures_do_have_0xe1(self):
        for name in ("gemm_4x33x8_m4k33.mcode.gz", "gemm_4x64x8_m4k64.mcode.gz"):
            self.assertNotEqual(records_of(load(name), 0xE1, 112), [], name)


if __name__ == "__main__":
    unittest.main()
