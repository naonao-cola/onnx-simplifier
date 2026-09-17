"""Continues `tests/test_axera_matmul_binary_cluster_search.py` (PR
#1592)'s own explicitly flagged methodological limit: that file applied
`tests/test_axera_conv_reg60_mechanism.py` (PR #1586)'s brute-force
byte-offset cluster-discovery method to Gemm's existing
`gemm_1x512x1000_tb0*` fixtures and got an **inconclusive** result, not
a clean negative -- that shape's decoded record COUNT varies diffusely
(1129-1144 across 8 samples of the same raw byte length), which defeats
a fixed-byte-offset scan regardless of whether a real cluster exists.
PR #1592 suggested "a future investigation with a cleaner grouping...
might still find something."

This file gives the method a fair shot at Gemm by using a much smaller
shape, `Gemm(A[1,16], B[16,16], C[16])` (`transA=0, transB=0`, `B`/`C`
compile-time-constant initializers, matching this project's established
convention that Pulsar2 forces Gemm's weight/bias to constants). 8
independent rebuilds, one config, one calibration seed throughout (no
provenance confound), built via a standalone script that replicates
`pulsar2_docker.py`'s own `docker run pulsar2 build` invocation and
`scripts/axera/calib_search.py`'s `calibration_format: Numpy` config
directly (this worktree lacks a compiled `onnxsim_cpp2py_export`
extension, so `pulsar2_docker.py` itself cannot be imported -- the same
bootstrap workaround several sibling PRs this session already used).

## Record count is finally constant -- the scan is valid here

All 8 samples are the same 2,568-byte raw length AND decode to the same
679 records -- unlike the original `(K=512,N=1000)` shape, this smaller
config does not exhibit diffuse positional reflow. That makes a
fixed-byte-offset scan meaningful, resolving PR #1592's own
methodological blocker.

`mcode.check()` reports one informational, non-blocking coverage
warning (93.9%, four small unexplained byte ranges) identically on all
8 samples -- a pre-existing gap in the grammar's own explanation of
this small fixture, not a defect introduced here, and not something
this file's own read of the reg=8 group depends on (see e.g. PR
#1575's/#1580's own "coverage caveat" precedent for treating this as
informational).

## Result: a clean, confident negative -- no cluster outside the
## already-known reg=8 pool group

Computing the union of every pairwise byte-diff across all 8 samples
(28 pairs) finds differences **only** at 8 offsets --
`{301, 303, 309, 311, 317, 319, 323, 325}` -- and every one of these
falls inside a single, already-known 4-slot group spanning offsets
`297`-`328`: three `verb=162,bank=0,field=0` `V`-kind records (the same
anchor pattern `tests/test_axera_matmul_reg8_noise_source.py`, PR
#1583, and `tests/test_axera_mul_reg8_noise_source.py`, PR #1591, both
used) followed by the one `S`-kind, `reg=8`, `tag=130` record PR #1577
originally decoded for Gemm. **Zero offsets outside that ~30-byte
window differ in any pairwise comparison, out of the full 2,568-byte
fixture.**

Two independent rebuilds (`_r0`/`_r1`, invoked separately via
`docker run`) landed byte-for-byte identical -- the same direct,
positive evidence for "nothing else varies" that PR #1592's own MatMul
check used (its `_r1`/`_r2` pair).

A targeted grouped scan (splitting the 8 samples by which of the pool's
4 candidate classes the `reg=8` slot itself drew -- 5 samples land on
class `C`, 2 on class `B`, 1 singleton on class `A` excluded from this
particular two-group comparison) finds correlated offsets only at
`{323, 325}` -- both inside the same known window. An arbitrary
build-order sanity-check split (first 4 samples vs. last 4, a grouping
with no reason to correlate with anything) finds **zero** correlated
offsets anywhere, confirming this scan's own specificity -- it is not
spuriously finding structure in an unrelated split.

## A small, honestly-flagged aside (not this file's main question)

Unlike PR #1577's own `(K=512,N=1000)` decode (a duplicate-tolerant
3-of-3 assignment, leading bytes only `0x23`/`0x93`), this smaller
shape's group is always a duplicate-free, complete 4-of-4 permutation
of the pool -- matching MatMul's and Mul's own "always a full
permutation" character (PR #1583/#1591) rather than Gemm's own larger-
shape behavior -- and its leading bytes mix `0x23` (Gemm's own
established byte) with `0x13` (previously seen only as Mul's leading
byte, PR #1582's cross-op synthesis). This is a real, observed
difference from Gemm's own larger-shape decode, but it is orthogonal to
this file's actual question (the binary-cluster search) and is not
investigated further here -- flagged for whoever picks up the pool's
own shape-dependence thread next (`tests/test_axera_gemm_reg8_pool_shape_dependence.py`,
PR #1585).

## What this establishes

Conv's PR #1586 binary-cluster phenomenon still does **not** generalize
to Gemm, and this time with a confident, methodologically clean
negative rather than PR #1592's own inconclusive one: at a Gemm shape
small enough to give the byte-offset scan valid footing, nothing beyond
the already-decoded reg=8 pool mechanism varies anywhere in the
fixture. Combined with PR #1592's own clean MatMul negative, the
emerging picture across three of four ops (Gemm, MatMul -- Mul has not
been checked this way) is that Conv's large all-or-nothing binary
switch looks increasingly Conv- or dilation-specific, not a general
codec-level mechanism -- though this remains a per-shape, per-op
finding, not a proof that no Gemm shape anywhere exhibits one.
"""

import gzip
import itertools
import os
import sys
import unittest

_AXERA_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts", "axera")
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import mcode  # noqa: E402

FIX = os.path.join(_AXERA_DIR, "fixtures")

ALL_NAMES = [f"gemm_1x16x16_binary_cluster_r{i}.mcode.gz" for i in range(8)]

# The already-known 4-slot reg=8 pool group's own byte window (three
# verb=162,bank=0,field=0 V-records + one S-kind reg=8,tag=130 record).
KNOWN_REG8_GROUP_WINDOW = range(297, 329)

KNOWN_CORRELATED_OFFSETS = frozenset([301, 303, 309, 311, 317, 319, 323, 325])


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def decode(name):
    return mcode.decode(load(name), **mcode.FULL_RULE)


def within_group_diff_offsets(names):
    """Union, across every pairwise comparison within `names`, of byte
    offsets that differ. Mirrors
    tests/test_axera_matmul_binary_cluster_search.py's own helper of the
    same name."""
    datas = {n: load(n) for n in names}
    length = len(datas[names[0]])
    for n in names:
        assert len(datas[n]) == length, n
    found = set()
    for a, b in itertools.combinations(names, 2):
        for i in range(length):
            if datas[a][i] != datas[b][i]:
                found.add(i)
    return found


class TestAllEightSamplesDecodeToTheSameRecordCount(unittest.TestCase):
    """The property PR #1592's own larger-shape Gemm attempt lacked:
    constant record count across every rebuild, making a fixed-byte-
    offset scan valid here."""

    def test_2568_bytes_and_679_records_every_time(self):
        for n in ALL_NAMES:
            data = load(n)
            self.assertEqual(len(data), 2568, n)
            self.assertEqual(len(decode(n)), 679, n)

    def test_coverage_warning_is_the_only_check_error_and_is_identical_everywhere(
        self,
    ):
        for n in ALL_NAMES:
            errs = mcode.check(load(n))
            self.assertEqual(len(errs), 1, (n, errs))
            self.assertTrue(errs[0].startswith("coverage:"), (n, errs))


class TestTwoIndependentRebuildsAreByteIdentical(unittest.TestCase):
    """r0 and r1 (two separately-invoked docker builds of the identical
    config) landed on exactly the same bytes -- direct evidence that
    nothing else varies, the same kind of incidental confirmation PR
    #1592's own MatMul r1/r2 pair provided."""

    def test_zero_diff(self):
        self.assertEqual(
            load("gemm_1x16x16_binary_cluster_r0.mcode.gz"),
            load("gemm_1x16x16_binary_cluster_r1.mcode.gz"),
        )


class TestNoClusterOutsideTheKnownReg8Window(unittest.TestCase):
    """The core finding: across all 8 samples, every pairwise byte-diff
    falls inside the already-decoded reg=8 pool group's own window
    (297-328). A clean negative -- unlike Conv's own PR #1586 positive
    finding, and unlike PR #1592's own inconclusive Gemm attempt."""

    def test_all_pairwise_diffs_confined_to_known_window(self):
        offsets = within_group_diff_offsets(ALL_NAMES)
        self.assertEqual(offsets, KNOWN_CORRELATED_OFFSETS)
        for o in offsets:
            self.assertIn(o, KNOWN_REG8_GROUP_WINDOW, o)

    def test_grouped_scan_by_reg8_class_finds_nothing_new(self):
        """Splitting by which pool class the reg=8 slot itself drew (5
        samples class C, 2 class B, excluding the 1 class-A singleton)
        -- still confined to the known window."""
        datas = {n: load(n) for n in ALL_NAMES}
        length = len(datas[ALL_NAMES[0]])
        group_c = [ALL_NAMES[i] for i in (0, 1, 4, 5, 6)]
        group_b = [ALL_NAMES[i] for i in (2, 7)]

        found = []
        for pos in range(length):
            vals_c = {datas[n][pos] for n in group_c}
            vals_b = {datas[n][pos] for n in group_b}
            if len(vals_c) == 1 and len(vals_b) == 1 and vals_c != vals_b:
                found.append(pos)
        self.assertTrue(found, "expected at least the known reg8-window diffs")
        for o in found:
            self.assertIn(o, KNOWN_REG8_GROUP_WINDOW, o)

    def test_arbitrary_build_order_split_finds_nothing_a_sanity_baseline(self):
        """First 4 samples vs. last 4 -- a grouping with no reason to
        correlate with anything real. Finding zero confirms the scan
        method itself isn't spuriously discovering structure in an
        unrelated split."""
        datas = {n: load(n) for n in ALL_NAMES}
        length = len(datas[ALL_NAMES[0]])
        found = []
        for pos in range(length):
            va = {datas[n][pos] for n in ALL_NAMES[:4]}
            vb = {datas[n][pos] for n in ALL_NAMES[4:]}
            if len(va) == 1 and len(vb) == 1 and va != vb:
                found.append(pos)
        self.assertEqual(found, [])


if __name__ == "__main__":
    unittest.main()
