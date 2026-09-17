"""Continues `tests/test_axera_generator_progress_stocktake.py` (PR
#1620)'s own generator-progress theme with a second, structurally
different mechanism: `reg=8`'s "second noise mechanism"
(`tests/test_axera_reg8_cross_op_synthesis.py` PR #1582 and its
contributing PRs) is a MULTI-BYTE, MULTI-SLOT unordered assignment, not
a single scalar formula like bank `0x81` field=192's `1024//k-1`
(PR #1620's own `bank81_field192_operand`, tested for patchability by a
concurrent sibling PR -- not duplicated here).

## Why MatMul's own version is the right first test case

`tests/test_axera_matmul_reg8_noise_source.py` (PR #1583) found
MatMul's own `reg=8` mechanism is the simplest of the three op-specific
versions decoded this session: a fixed 4-slot group at FIXED absolute
byte offsets (independent of the `A_offset`/`B_offset` table-order coin
flip), always a complete, duplicate-free permutation of the shared
`{0x10,0x20,0x30,0x40}`-trailing-byte pool, with no biconditional
indicator needed (unlike Gemm's duplicate-tolerant 3-slot version or
Conv's always-omits-one, variable-register-label 3-of-4 version). This
also makes it a genuinely good test case for the actual bar a
from-scratch generator needs to clear: reg=8's own assignment is
non-deterministic across real Pulsar2 builds (any one of several valid
permutations is "correct" -- Pulsar2 itself never commits to one
canonical choice), so "emit SOMETHING a real build could ALSO have
produced" is the right success criterion here, not bit-identical
reproduction the way `patch_site_a`'s own reflow tests require.

## What was actually observed in real Pulsar2 output

Decoding all 10 already-committed MatMul rebuild-stability fixtures
(`tests/test_axera_matmul_rebuild_stability.py` PR #1581's own 8
`v7stability_*` samples, plus 2 pre-existing cross-toolchain fixtures
from `tests/test_axera_matmul_offset_table_coinflip.py` PR #1506) for
the reg=8 quad's own exact slot assignment finds **7 distinct
permutations out of the 24 mathematically possible ones**:

    ('P1','P3','P4','P2'), ('P1','P4','P2','P3'), ('P2','P1','P4','P3'),
    ('P2','P4','P3','P1'), ('P3','P2','P1','P4'), ('P3','P2','P4','P1'),
    ('P4','P2','P3','P1')

(`P1`-`P4` are the pool's own 4 trailing-byte-distinguished candidates,
`33 00 20`/`23 00 40`/`23 00 30`/`23 00 10`, per PR #1583's own
naming.) This is a real, if small, characterization of Pulsar2's actual
constraint space that a naive "any of the 24 permutations is equally
valid" assumption would not capture on its own -- though 10 samples is
far too few to conclude the OTHER 17 are actually forbidden by the real
compiler, only that they were not seen in this specific corpus.

## `tiny_emit.emit_matmul_reg8_quad`: emits a syntactically-valid quad,
## verified against every real permutation this project has on record

`scripts/axera/tiny_emit.py`'s own new `emit_matmul_reg8_quad(reference,
permutation)` locates the group's stable anchor record (byte pattern
`a2 00 00 00 12 00 00 00`, searched for rather than hardcoded to a
fixed offset) and rewrites only the 4 candidate-identity fields (12
bytes total), leaving every framing byte around them untouched -- this
project's usual "patch only what varies" discipline.

Verified below, directly against the real fixtures, three separate
claims:

1. **No-op emit is byte-identical.** Re-emitting each of the 10 real
   fixtures' own already-observed permutation reproduces that fixture's
   exact bytes, with zero difference anywhere in the stream --
   confirms the function's own byte offsets and framing are exactly
   right, not just "close."
2. **Emitting a DIFFERENT observed permutation round-trips correctly.**
   Taking one real fixture as a base and emitting each of the OTHER 6
   observed permutations produces a stream that re-decodes to exactly
   that target permutation, with every byte outside the 39-byte group
   (offsets 345-383) unchanged, and `mcode.check()` clean.
3. **An UNobserved permutation is still syntactically well-formed.**
   One of the 17 mathematically-possible-but-never-seen permutations
   was emitted and re-decoded successfully with zero `mcode.check()`
   errors -- this project's own grammar does not itself forbid it, even
   though no real Pulsar2 build in this corpus has ever produced it.
   This is an honest, informative result either way: emission is not
   artificially restricted to the observed set, but this also does not
   prove an unobserved permutation is something a real Pulsar2 build
   would ever actually choose.

## What this does and does not establish

This is a genuine, verified step beyond PR #1620's own "characterized
but not generatable" finding for reg=8: MatMul's own version of the
mechanism can now be EMITTED, not just decoded, and the emitted output
is confirmed syntactically valid by this project's own grammar for
both observed and unobserved permutations. It does **not** establish:
end-to-end shape-to-mcode generation (this only rewrites one already-
compiled reference's own reg=8 quad, the same scope every other
`patch_*`/`emit_*` function in this file has); that an unobserved
permutation is something a real Pulsar2 build would ever produce (only
that this project's grammar doesn't reject it); or anything about
device-level correctness (unlike the scale-family patches,
`tests/test_axera_mul_emit_hardware.py`, this has not been checked
against real AX650N hardware -- the pool's own semantic meaning, e.g.
whether it represents interchangeable scratch/tile-buffer addresses
where any valid permutation is functionally equivalent, remains open
per PR #1582's own repeated caveat).
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
import tiny_emit  # noqa: E402

FIX = os.path.join(_AXERA_DIR, "fixtures")

V7_BATCH = [
    f"matmul_4x8x8_v7stability_{n}.mcode.gz"
    for n in ("diag0", *[f"r{i}" for i in range(1, 8)])
]
LEGACY_BATCH = [
    "matmul_4x8x8_rebuild_modeA.mcode.gz",
    "matmul_4x8x8_rebuild_modeB.mcode.gz",
]
ALL_NAMES = V7_BATCH + LEGACY_BATCH

CANDS = {b"\x33\x00\x20", b"\x23\x00\x40", b"\x23\x00\x30", b"\x23\x00\x10"}
CLASS_NAME = {
    b"\x33\x00\x20": "P1",
    b"\x23\x00\x40": "P2",
    b"\x23\x00\x30": "P3",
    b"\x23\x00\x10": "P4",
}
NAME_BYTES = {v: k for k, v in CLASS_NAME.items()}

ANCHOR_AT = 345
SLOT_OFFSETS = (353, 361, 369, 378)

EXPECTED_OBSERVED_PERMS = {
    ("P1", "P3", "P4", "P2"),
    ("P1", "P4", "P2", "P3"),
    ("P2", "P1", "P4", "P3"),
    ("P2", "P4", "P3", "P1"),
    ("P3", "P2", "P1", "P4"),
    ("P3", "P2", "P4", "P1"),
    ("P4", "P2", "P3", "P1"),
}


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def decode(name_or_bytes):
    data = load(name_or_bytes) if isinstance(name_or_bytes, str) else name_or_bytes
    return mcode.decode(data, **mcode.FULL_RULE)


def find_group(recs):
    slots = {}
    for r in recs:
        at = r.get("at")
        if at not in SLOT_OFFSETS[:3]:
            continue
        if (
            r["kind"] == "V"
            and r.get("verb") == 162
            and r.get("field") == 0
            and r.get("bank") == 0
            and r.get("operand", b"")[:3] in CANDS
        ):
            slots[at] = r["operand"][:3]
    for r in recs:
        if (
            r["kind"] == "S"
            and r.get("reg") == 8
            and r.get("tag") == 130
            and r.get("payload") in CANDS
            and r.get("at") == SLOT_OFFSETS[3]
        ):
            slots[SLOT_OFFSETS[3]] = r["payload"]
    return slots


def perm_of(name_or_bytes):
    slots = find_group(decode(name_or_bytes))
    return tuple(CLASS_NAME[slots[off]] for off in SLOT_OFFSETS)


def bytes_perm(name_perm):
    return tuple(NAME_BYTES[n] for n in name_perm)


class TestSevenDistinctPermutationsObservedAcrossTenRealBuilds(unittest.TestCase):
    """Recomputes PR #1583's own permutation table directly from the
    fixtures, not trusted from any docstring."""

    def test_exactly_seven_distinct_permutations(self):
        observed = {perm_of(name) for name in ALL_NAMES}
        self.assertEqual(observed, EXPECTED_OBSERVED_PERMS)

    def test_seven_of_twentyfour_possible(self):
        all24 = set(itertools.permutations(("P1", "P2", "P3", "P4")))
        self.assertEqual(len(all24), 24)
        self.assertTrue(EXPECTED_OBSERVED_PERMS <= all24)
        self.assertEqual(len(EXPECTED_OBSERVED_PERMS), 7)


class TestNoOpEmitIsByteIdentical(unittest.TestCase):
    """Re-emitting each fixture's own already-observed permutation
    reproduces that fixture's exact bytes -- confirms the function's
    own offsets/framing are exactly right, not just close."""

    def test_all_ten_fixtures_round_trip_byte_identical(self):
        for name in ALL_NAMES:
            data = load(name)
            perm = bytes_perm(perm_of(name))
            out = tiny_emit.emit_matmul_reg8_quad(data, perm)
            self.assertEqual(out, data, name)


class TestEmittingADifferentObservedPermutationRoundTrips(unittest.TestCase):
    """Taking one real fixture as base, emit each of the OTHER 6
    observed permutations and confirm: (a) re-decoding recovers exactly
    that target permutation, (b) every byte outside the 39-byte group
    is untouched, (c) mcode.check() is clean."""

    def test_all_six_other_observed_perms(self):
        base_name = "matmul_4x8x8_v7stability_diag0.mcode.gz"
        base_data = load(base_name)
        base_perm = perm_of(base_name)
        others = EXPECTED_OBSERVED_PERMS - {base_perm}
        self.assertEqual(len(others), 6, others)
        for target in others:
            patched = tiny_emit.emit_matmul_reg8_quad(base_data, bytes_perm(target))
            got = perm_of(patched)
            self.assertEqual(got, target, target)
            self.assertEqual(patched[:ANCHOR_AT], base_data[:ANCHOR_AT], target)
            self.assertEqual(patched[384:], base_data[384:], target)
            errs = mcode.check(patched)
            hard = [e for e in errs if not e.startswith("coverage:")]
            self.assertEqual(hard, [], (target, hard))


class TestUnobservedPermutationIsStillSyntacticallyValid(unittest.TestCase):
    """One of the 17 mathematically-possible-but-never-seen permutations
    is emitted and re-decoded successfully with zero mcode.check()
    errors -- this project's own grammar does not itself forbid it,
    even though no real build in this corpus has produced it. An
    honest, informative result either way (see module docstring)."""

    def test_one_unobserved_permutation_emits_and_decodes_cleanly(self):
        all24 = set(itertools.permutations(("P1", "P2", "P3", "P4")))
        unobserved = sorted(all24 - EXPECTED_OBSERVED_PERMS)
        self.assertEqual(len(unobserved), 17)
        target = unobserved[0]
        base_data = load("matmul_4x8x8_v7stability_diag0.mcode.gz")
        patched = tiny_emit.emit_matmul_reg8_quad(base_data, bytes_perm(target))
        got = perm_of(patched)
        self.assertEqual(got, target)
        errs = mcode.check(patched)
        hard = [e for e in errs if not e.startswith("coverage:")]
        self.assertEqual(hard, [])


class TestInvalidPermutationIsRejected(unittest.TestCase):
    """A duplicate or an omission (not a genuine 4-member permutation)
    is refused with ValueError rather than silently miswriting the
    quad -- matching this project's usual defensive posture
    (_strided_run and friends)."""

    def test_duplicate_is_rejected(self):
        base_data = load("matmul_4x8x8_v7stability_diag0.mcode.gz")
        bad = (
            b"\x33\x00\x20",
            b"\x33\x00\x20",
            b"\x23\x00\x30",
            b"\x23\x00\x10",
        )
        with self.assertRaises(ValueError):
            tiny_emit.emit_matmul_reg8_quad(base_data, bad)

    def test_wrong_length_is_rejected(self):
        base_data = load("matmul_4x8x8_v7stability_diag0.mcode.gz")
        with self.assertRaises(ValueError):
            tiny_emit.emit_matmul_reg8_quad(
                base_data, (b"\x33\x00\x20", b"\x23\x00\x40", b"\x23\x00\x30")
            )

    def test_anchor_not_found_is_rejected(self):
        with self.assertRaises(ValueError):
            tiny_emit.emit_matmul_reg8_quad(
                b"\x00" * 500,
                (
                    b"\x33\x00\x20",
                    b"\x23\x00\x40",
                    b"\x23\x00\x30",
                    b"\x23\x00\x10",
                ),
            )


if __name__ == "__main__":
    unittest.main()
