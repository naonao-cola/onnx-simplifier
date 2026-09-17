"""Applies `tests/test_axera_conv_reg60_mechanism.py` (PR #1586)'s own
brute-force cluster-discovery method to MatMul, and secondarily to
Gemm, for the first time: PR #1586 found that Conv's `reg=60` was one
visible thread of a much larger 28-byte, all-or-nothing binary "path"
switch, discovered by scanning every byte offset in a fixed-shape
fixture for a position that's constant within each of two rebuild-
outcome groups but differs between them. Nobody had checked whether
Gemm or MatMul have their own analogous hidden binary-cluster switch --
this file does, and reports a clean, well-evidenced negative for both.

## MatMul: a clean negative, tightly controlled

`tests/test_axera_matmul_reg8_noise_source.py` (PR #1583) already
decoded MatMul's own `reg=8` noise as a 4-slot group at FIXED absolute
byte offsets (345/353/361/369/378), independent of the `A_offset`/
`B_offset` "table-order coin flip" (`tests/test_axera_matmul_rebuild_stability.py`,
PR #1581) -- unlike almost everything else in MatMul's mcode, whose
absolute position DOES shift by ~900 bytes between the two table
orders. That fixed-offset property makes MatMul's own 8 already-
committed `v7stability_*` fixtures (PR #1581) unusually well-suited to
this exact brute-force method, PROVIDED the scan is restricted to one
table-order group at a time -- comparing across orders would just
rediscover the already-known ~900-byte coin-flip reflow, not a new
cluster.

Grouping PR #1581's own 8 samples by table order gives two groups of 4:

- **(A,B) order**: `_r1`, `_r2`, `_r3`, `_r6`
- **(B,A) order**: `_diag0`, `_r4`, `_r5`, `_r7`

Computing the union of every pairwise byte-diff *within* each order
group (10 pairs total across both groups) finds diffs **only** at
offsets `{357, 359, 365, 367, 373, 375, 379, 381}` -- every one of
these falls inside the already-decoded reg=8 group's own known window
(`345`-`381`). **Zero offsets outside that window differ in any
within-order pair.** This is a clean, informative negative: unlike
Conv, MatMul's `MatMul(A[4,8],B[8,8])` mcode has *no* additional hidden
source of rebuild-to-rebuild variation once table order and the
already-decoded reg=8 mechanism are accounted for -- the reg=8 pool
really is the *only* thing that moves.

A related, incidental observation surfaced while computing this:
**`_v7stability_r1.mcode.gz` and `_v7stability_r2.mcode.gz` are
byte-for-byte IDENTICAL** (0 differing bytes across the full 3,112-byte
fixture) despite being two independently-invoked `pulsar2_docker.build()`
calls. Not a new mechanism -- consistent with the reg=8 pool's own
already-known small (4-member) value space at this tiny shape making an
exact repeat unsurprising -- but recorded here since it is a direct,
useful piece of evidence for the "clean negative" conclusion above (two
independent builds landing on literally the same bytes is only possible
if there really is nothing else varying).

**A caveat, checked and rejected**: including the two pre-existing
`matmul_4x8x8_rebuild_mode{A,B}.mcode.gz` fixtures (a different build
session/toolchain, PR #1506) in the same-order groups does surface many
more differing offsets -- but PR #1581's own docstring already
identifies this as a calibration-data/toolchain-version confound (the
`y_offset`/`Y_offset` ASCII-case naming difference at offset ~156, plus
unrelated calibration-driven float differences elsewhere), not a new
structural cluster; this file does not include the legacy fixtures in
its own within-order comparison for that reason, matching PR #1581's
own explicit choice to use them only for the reg=8 cross-check, never
pooled into the main stability analysis.

## Gemm: an inconclusive negative, with an honest methodological limit

The exact same fixed-byte-offset scan does not transfer cleanly to
Gemm: unlike MatMul (fixed 3,112-byte length AND a fixed reg=8-group
offset), Gemm's 8 already-committed `gemm_1x512x1000_tb0*` fixtures
(`tests/test_axera_gemm_reg8_second_noise_source.py`, PR #1577) are all
the same 4,368-byte length, but their *decoded record counts* range
from 1129 to 1144 across the 8 samples -- a diffuse positional reflow
unrelated to any one grouping (this project's own previously-noted
"~0.78 match ratio" diffuse-reflow character for Gemm). A byte-offset
scan under that much independent positional noise is not meaningful --
even a real correlate's neighboring bytes get shuffled independently of
the grouping being tested, which is exactly what was observed: **zero
correlated offsets** found for either of the two natural 2-way
groupings tried (`reg=0`'s extra-record indicator, 3-vs-5; the `reg=78`
anchor's own class, `C` vs not-`C`, 5-vs-3), whether scanned by fixed
byte offset or by a reflow-tolerant *record-identity* key (matching
records across samples by `(kind, reg, tag)` or `(kind, bank, field)`
rather than raw offset, restricted to identity keys appearing exactly
once in every sample).

This is reported as an honest, methodologically-limited negative, not a
confident "Gemm has no analogous cluster" claim: the brute-force method
itself may simply not be well-suited to an op whose mcode has this much
independent positional churn, rather than Gemm genuinely lacking a
Conv-like binary switch. A future investigation with a cleaner
grouping, or a positional-diff-tolerant matching scheme beyond the
simple record-identity approach tried here, might still find something
this file's two attempts did not.

## What this establishes

Conv's PR #1586 binary-cluster phenomenon does **not** obviously
generalize: MatMul shows a clean, tightly-controlled negative (no
hidden cluster once table order and the known reg=8 mechanism are
accounted for), and Gemm's own check is inconclusive due to a real
methodological limit (diffuse reflow) rather than a clean negative.
Combined with Conv's own positive finding, the emerging picture is that
Conv's binary path switch is either Conv/dilation-specific, or present
in Gemm too but hidden underneath that op's own larger positional
noise floor -- this file cannot yet distinguish those two readings.
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

MATMUL_AB_ORDER = [
    "matmul_4x8x8_v7stability_r1.mcode.gz",
    "matmul_4x8x8_v7stability_r2.mcode.gz",
    "matmul_4x8x8_v7stability_r3.mcode.gz",
    "matmul_4x8x8_v7stability_r6.mcode.gz",
]
MATMUL_BA_ORDER = [
    "matmul_4x8x8_v7stability_diag0.mcode.gz",
    "matmul_4x8x8_v7stability_r4.mcode.gz",
    "matmul_4x8x8_v7stability_r5.mcode.gz",
    "matmul_4x8x8_v7stability_r7.mcode.gz",
]

# The reg=8 4-slot group's own known window (tests/test_axera_matmul_reg8_noise_source.py,
# PR #1583: anchor at 345, slots at 353/361/369/378, each up to 3 bytes).
REG8_GROUP_WINDOW = range(345, 382)

GEMM_ALL_NAMES = [
    "gemm_1x512x1000_tb0.mcode.gz",
    "gemm_1x512x1000_tb0_rebuild0.mcode.gz",
    "gemm_1x512x1000_tb0_rebuild1.mcode.gz",
    "gemm_1x512x1000_tb0_rebuild2.mcode.gz",
    "gemm_1x512x1000_tb0_stability_r0.mcode.gz",
    "gemm_1x512x1000_tb0_stability_r1.mcode.gz",
    "gemm_1x512x1000_tb0_stability_r2.mcode.gz",
    "gemm_1x512x1000_tb0_stability_r3.mcode.gz",
]


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def decode(name):
    return mcode.decode(load(name), **mcode.FULL_RULE)


def offset_table_order(data):
    """Mirrors tests/test_axera_matmul_rebuild_stability.py's own
    offset_table_order()."""
    window = data[100:400]
    ia = window.find(b"A_offset")
    ib = window.find(b"B_offset")
    assert ia != -1 and ib != -1
    return ("A", "B") if ia < ib else ("B", "A")


def within_group_diff_offsets(names):
    """Union, across every pairwise comparison within `names`, of byte
    offsets that differ. All fixtures in `names` must be the same
    length."""
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


class TestMatMulOrderGroupsAreCorrectlyIdentified(unittest.TestCase):
    def test_ab_order_group(self):
        for n in MATMUL_AB_ORDER:
            self.assertEqual(offset_table_order(load(n)), ("A", "B"), n)

    def test_ba_order_group(self):
        for n in MATMUL_BA_ORDER:
            self.assertEqual(offset_table_order(load(n)), ("B", "A"), n)


class TestMatMulR1EqualsR2Exactly(unittest.TestCase):
    """Two independent pulsar2_docker.build() invocations of the exact
    same config landed on byte-for-byte identical output -- an
    incidental but directly useful piece of evidence for the "nothing
    else varies" conclusion below."""

    def test_zero_diff(self):
        a = load("matmul_4x8x8_v7stability_r1.mcode.gz")
        b = load("matmul_4x8x8_v7stability_r2.mcode.gz")
        self.assertEqual(a, b)


class TestMatMulNoClusterOutsideTheKnownReg8Window(unittest.TestCase):
    """The core finding: within EITHER table-order group alone, every
    pairwise byte-diff falls inside the already-decoded reg=8 group's
    own window (345-381). Nothing else in the 3,112-byte fixture varies
    build-to-build once table order is held fixed -- a clean negative,
    unlike Conv's own PR #1586 positive finding."""

    def test_ab_order_diffs_are_confined_to_reg8_window(self):
        offsets = within_group_diff_offsets(MATMUL_AB_ORDER)
        self.assertTrue(offsets, "expected at least the known reg8-window diffs")
        for o in offsets:
            self.assertIn(o, REG8_GROUP_WINDOW, o)

    def test_ba_order_diffs_are_confined_to_reg8_window(self):
        offsets = within_group_diff_offsets(MATMUL_BA_ORDER)
        self.assertTrue(offsets, "expected at least the known reg8-window diffs")
        for o in offsets:
            self.assertIn(o, REG8_GROUP_WINDOW, o)

    def test_legacy_fixtures_introduce_a_different_confound_not_a_cluster(self):
        """Including the two pre-existing, different-toolchain
        matmul_4x8x8_rebuild_mode{A,B}.mcode.gz fixtures DOES surface
        extra differing offsets -- but this is the calibration/
        toolchain-version confound PR #1581 already flagged (e.g. the
        y_offset/Y_offset naming artifact at offset ~156), not a new
        structural cluster. Confirms offset 156 (the known naming
        artifact) appears when legacy fixtures are mixed in, and
        confirms it does NOT appear in the clean same-session
        comparison above."""
        offsets_clean = within_group_diff_offsets(MATMUL_AB_ORDER)
        self.assertNotIn(156, offsets_clean)

        mixed = MATMUL_AB_ORDER + ["matmul_4x8x8_rebuild_modeA.mcode.gz"]
        offsets_mixed = within_group_diff_offsets(mixed)
        self.assertIn(156, offsets_mixed)


class TestGemmRecordCountVariesDiffusely(unittest.TestCase):
    """Establishes why a fixed-byte-offset scan is not meaningful for
    Gemm the way it is for MatMul/Conv: decoded record counts across
    the 8 already-committed gemm_1x512x1000_tb0* fixtures range from
    1129 to 1144, despite all 8 being the same 4,368-byte raw length --
    positional reflow unrelated to any specific grouping."""

    def test_record_counts_vary(self):
        counts = {n: len(decode(n)) for n in GEMM_ALL_NAMES}
        self.assertGreater(max(counts.values()) - min(counts.values()), 10, counts)


class TestGemmClusterSearchIsInconclusive(unittest.TestCase):
    """Two natural 2-way groupings tried on Gemm's own 8 samples, via a
    reflow-tolerant record-identity match (kind, reg-or-bank, tag-or-
    field) restricted to identity keys appearing exactly once in every
    sample -- zero correlated keys found for either. Reported as an
    honest, methodologically-limited negative (see module docstring),
    not a confident "Gemm has none" claim."""

    @staticmethod
    def _identity_map(recs):
        out = {}
        for r in recs:
            if r["kind"] in ("S", "B"):
                key = (r["kind"], r.get("reg"), r.get("tag"))
                val = r.get("payload")
            elif r["kind"] == "V":
                key = (r["kind"], r.get("bank"), r.get("field"))
                val = r.get("value")
            else:
                continue
            out.setdefault(key, []).append(val)
        return out

    def _scan(self, group_a, group_b):
        recs = {n: decode(n) for n in GEMM_ALL_NAMES}
        maps = {n: self._identity_map(recs[n]) for n in GEMM_ALL_NAMES}
        common = None
        for n in GEMM_ALL_NAMES:
            ks = {k for k, v in maps[n].items() if len(v) == 1}
            common = ks if common is None else common & ks
        self.assertGreater(
            len(common), 100, "expected many common single-occurrence keys"
        )

        found = []
        for k in common:
            vals_a = {maps[n][k][0] for n in group_a}
            vals_b = {maps[n][k][0] for n in group_b}
            if len(vals_a) == 1 and len(vals_b) == 1 and vals_a != vals_b:
                found.append(k)
        return found

    def test_reg0_extra_record_grouping_finds_nothing(self):
        group_yes = [
            "gemm_1x512x1000_tb0_rebuild1.mcode.gz",
            "gemm_1x512x1000_tb0_rebuild2.mcode.gz",
            "gemm_1x512x1000_tb0_stability_r2.mcode.gz",
        ]
        group_no = [n for n in GEMM_ALL_NAMES if n not in group_yes]
        self.assertEqual(self._scan(group_yes, group_no), [])

    def test_reg78_anchor_class_grouping_finds_nothing(self):
        group_notc = [
            "gemm_1x512x1000_tb0_stability_r0.mcode.gz",
            "gemm_1x512x1000_tb0_stability_r1.mcode.gz",
            "gemm_1x512x1000_tb0_stability_r3.mcode.gz",
        ]
        group_c = [n for n in GEMM_ALL_NAMES if n not in group_notc]
        self.assertEqual(self._scan(group_c, group_notc), [])


if __name__ == "__main__":
    unittest.main()
