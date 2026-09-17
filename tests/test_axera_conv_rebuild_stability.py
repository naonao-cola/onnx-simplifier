"""Extends the census-based rebuild-stability check
(`tests/test_axera_universal_bank_reg_stability.py` PR #1566,
`tests/test_axera_core_bank_stability_extend.py` PR #1567 for Mul;
`tests/test_axera_gemm_rebuild_stability.py` PR #1576 for Gemm) to
**Conv**, completing the sweep across this project's three main ops for
the first time.

Conv, like Gemm, structurally cannot exhibit the "table-order coin
flip" mechanism (`scripts/axera/README.md`, `tests/test_axera_matmul_offset_table_coinflip.py`)
found for Mul's `reg=8` -- Pulsar2 forces Conv's weight to a
compile-time constant, leaving only one live, non-constant tensor
(`x`). This file confirms that precondition doesn't even have a
footprint here at all: unlike Gemm (whose fixtures carry literal
`Y_offset`/`A_offset` ASCII strings), **none of this file's 8 Conv
fixtures contain any `*_offset`-named table string whatsoever** -- there
is no name table to reorder in the first place, a stronger absence than
Gemm's "table exists but its order never varies."

## Method

Reused `Conv(k=3, dilation=3, pad=3, cin=4, cout=4, insz=16)` --
the exact shape `tests/test_axera_conv_dilation_b_run_is_reliable.py`
(PR #1507) already built 4 independent samples of (the committed
`conv_dilation3.mcode.gz` plus `_rebuild{0,1,2}.mcode.gz`, all 3,528
bytes, built via the `pulsar2:6.0-lite` Docker image). This file adds
**4 more fresh, independent rebuilds** (`_v7stability_r{0,1,2,3}.mcode.gz`,
also 3,528 bytes each) for 8 total samples.

**A methodological wrinkle, surfaced and handled explicitly**: this
worktree's Docker daemon only has `pulsar2:7.0-lite` loaded, not the
`pulsar2:6.0-lite` the original 4 fixtures were built with (the
project's `DEFAULT_IMAGE`) -- no `6.0-lite` image tarball was available
to load here. Before treating a 7.0-lite build as a valid "identical
config" rebuild, this file confirmed compatibility first: a single
diagnostic 7.0-lite build was checked against `conv_dilation3.mcode.gz`
and found (a) identical total length (3,528 bytes), (b) only 20 bytes
differing -- within this project's established ~6-15-to-a-few-dozen-byte
ordinary rebuild-noise floor, nowhere near a structural
re-serialization, and (c) an exact match on
`test_axera_conv_dilation_b_run_is_reliable.py`'s own "extra B-run"
marker (`[(603, 4), (617, 3), (1211, 4), (1225, 3)]`), the one
previously-decoded field known to be dilation-sensitive. All three
checks passed, so the two toolchain versions are treated as
interchangeable for this specific shape/config -- flagged here for
honesty rather than silently mixed in as if they were the same image.

Like PR #1576, this file compares content **within each independently-
built batch first**, before pooling, to avoid conflating genuine
allocator noise with a batch-provenance artifact (here, the image
version difference rather than an unrecoverable calibration seed --
this file's calibration data uses the identical `RandomState(0)` seed
`tests/test_axera_mcode_structure.py`'s own `_build_and_get_mcode_bytes`
helper does for every sample, both old and new, so calibration
provenance is *not* a confound here the way it was for PR #1576's Gemm
check).

## Result

**All 9 near-universal banks (`0x00`-`0x04`, `0x0e`, `0x0f`, `0x1c`,
`0x1e`) are byte-for-byte content-stable** across all 8 samples, both
within each batch and pooled -- the same clean result Mul and Gemm both
got. Three ops, three confirmations.

**Registers split differently than Gemm did** -- only two categories
appeared here, not three:

1. **34 of 36 registers are stable everywhere** (within each batch and
   pooled): `2, 7, 9, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 32, 36,
   38, 40, 52, 62, 66, 72, 74, 76, 78, 96, 100, 101, 106, 107, 120,
   126, 128, 138`, plus `0`. Notably, **`reg=0` and `reg=78` -- both
   found genuinely noisy for Gemm (PR #1576) -- are fully stable for
   Conv.** The specific set of registers affected by whatever this
   second noise mechanism is is NOT a fixed, op-independent set; it
   differs by op.
2. **Two registers (`8`, `60`) are genuinely noisy** -- unstable within
   at least one independently-built batch, which is sufficient to
   confirm real variation regardless of what the *other* batch happens
   to show (a batch of only 4 samples drawn from a small discrete value
   space can land on the same value by chance; that does not establish
   stability, only failing to unstably diverge does).
   - **`reg=8`** is unstable within BOTH the old batch and the fresh
     batch independently. Its variation is not a simple 2-value flip:
     the exact number and payload values of its `tag=130`
     small-integer records change from build to build (e.g. one extra
     `(S, 130, b'#\\x00\\x10')` record appears in some samples and not
     others, at varying positions). **This is the third op (after Mul
     and Gemm) where `reg=8` shows rebuild-to-rebuild instability, and
     the third op where the *known* table-order coin-flip mechanism
     cannot be the explanation** -- Conv has no offset-name table at
     all to reorder (see above), an even stronger case than Gemm's
     "table exists, order confirmed fixed." Left honestly undecoded,
     same as Gemm's own residual.
   - **`reg=60`** is a new noise register not flagged for either Mul or
     Gemm's own near-universal-register reports. Its variation looks
     almost structured -- a single-byte, off-by-one 2-value flip
     (`0x7e` vs `0x7f`, i.e. `126` vs `127`) -- but the split does
     *not* align with the old/fresh (image-version) batch boundary:
     `conv_dilation3.mcode.gz` itself (part of the "old" batch) reads
     `0x7e`, matching all 4 fresh builds, while its own batch-mates
     `_rebuild{0,1,2}.mcode.gz` all read `0x7f`. This rules out
     "toolchain version" as the driver and leaves the true cause
     unidentified -- reported as an open, off-by-one-valued residual
     rather than forced into either a version-driven or a
     batch-driven story.

No register fell into a genuine "stable within each batch alone but
differs between batches" (batch-confounded) category here, unlike
Gemm's 4-register finding -- this file's calibration-provenance
control (identical seed, not just identical shape) likely explains why
that specific ambiguity doesn't recur.

## What this establishes for the resource-model picture

The rebuild-stability triad (Mul, Gemm, Conv) is now complete at the
census-sweep level. All three ops share the same 9-bank / ~36-register
near-universal structure, and all three show it is genuinely
content-stable for the overwhelming majority of that structure. But
`reg=8` is now confirmed unstable in *all three*, and in *two of three*
(Gemm, Conv) this cannot be the table-order coin flip that explains
Mul's own case -- meaning whatever the real mechanism is, it is general
across this codec, not specific to any one op's table-ordering quirk,
and remains this project's clearest open thread in the noise-vs-load-
bearing resource-model picture.
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

OLD_BATCH = [
    "conv_dilation3.mcode.gz",
    "conv_dilation3_rebuild0.mcode.gz",
    "conv_dilation3_rebuild1.mcode.gz",
    "conv_dilation3_rebuild2.mcode.gz",
]
FRESH_BATCH = [
    "conv_dilation3_v7stability_r0.mcode.gz",
    "conv_dilation3_v7stability_r1.mcode.gz",
    "conv_dilation3_v7stability_r2.mcode.gz",
    "conv_dilation3_v7stability_r3.mcode.gz",
]
ALL_NAMES = OLD_BATCH + FRESH_BATCH

CORE_BANKS = (0x00, 0x01, 0x02, 0x03, 0x04, 0x0E, 0x0F, 0x1C, 0x1E)
CORE_REGS = (
    0,
    2,
    7,
    8,
    9,
    10,
    12,
    14,
    16,
    18,
    20,
    22,
    24,
    26,
    28,
    32,
    36,
    38,
    40,
    52,
    60,
    62,
    66,
    72,
    74,
    76,
    78,
    96,
    100,
    101,
    106,
    107,
    120,
    126,
    128,
    138,
)

GENUINE_NOISE_REGS = (8, 60)
STABLE_EVERYWHERE_REGS = tuple(sorted(set(CORE_REGS) - set(GENUINE_NOISE_REGS)))

EXPECTED_D3_B_RUN = [(603, 4), (617, 3), (1211, 4), (1225, 3)]


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def bank_records(recs, bank):
    return sorted(
        (r["field"], r.get("value"))
        for r in recs
        if r["kind"] == "V" and r["bank"] == bank
    )


def reg_records(recs, reg):
    return [
        (r["kind"], r.get("tag"), r.get("payload"))
        for r in recs
        if r["kind"] in ("S", "B") and r.get("reg") == reg
    ]


def decode_all(names):
    return [mcode.decode(load(n), **mcode.FULL_RULE) for n in names]


class TestAllEightSamplesAreTheSameLength(unittest.TestCase):
    def test_3528_bytes_every_time(self):
        for n in ALL_NAMES:
            self.assertEqual(len(load(n)), 3528, n)


class TestNoOffsetTableExistsAtAll(unittest.TestCase):
    """Unlike Gemm (PR #1576) and MatMul, Conv's mcode carries no
    `*_offset`-named ASCII string at all in any of the 8 samples -- a
    stronger absence than "table exists but order never varies": there
    is no table here to reorder in the first place."""

    def test_no_offset_named_strings_in_any_sample(self):
        import re

        for n in ALL_NAMES:
            d = load(n)
            self.assertEqual(
                re.findall(rb"[A-Za-z_][A-Za-z0-9_]{2,20}_offset", d), [], n
            )


class TestFreshBatchImageVersionIsCompatible(unittest.TestCase):
    """The fresh batch was built with `pulsar2:7.0-lite` (this
    worktree's only loaded Docker image), not the `pulsar2:6.0-lite`
    the original 4 fixtures used. Before trusting a cross-version
    comparison, confirm the two are compatible for this exact
    shape/config: same length, ordinary noise-floor-scale byte diff
    (not a structural re-serialization), and an exact match on the one
    previously-decoded dilation-sensitive field."""

    def test_same_length_as_old_batch(self):
        self.assertEqual(
            len(load("conv_dilation3.mcode.gz")),
            len(load("conv_dilation3_v7stability_r0.mcode.gz")),
        )

    def test_byte_diff_is_ordinary_noise_scale_not_structural(self):
        a = load("conv_dilation3.mcode.gz")
        b = load("conv_dilation3_v7stability_r0.mcode.gz")
        diffs = sum(1 for i in range(len(a)) if a[i] != b[i])
        self.assertGreater(diffs, 0)
        self.assertLess(
            diffs,
            100,
            "should be ordinary noise scale, not a version-driven restructuring",
        )

    def test_extra_b_run_marker_still_matches(self):
        sys.path.insert(0, os.path.dirname(__file__))
        import importlib

        m = importlib.import_module("test_axera_conv_dilation_b_run_is_reliable")
        for n in FRESH_BATCH:
            self.assertEqual(m.early_b_runs(load(n)), EXPECTED_D3_B_RUN, n)


class TestCoreBanksAreContentStable(unittest.TestCase):
    """All 9 near-universal banks are byte-identical across all 8
    independent Conv(dilation=3) rebuilds, matching Mul's and Gemm's
    own results."""

    def test_nine_banks_stable_across_all_eight(self):
        all_recs = decode_all(ALL_NAMES)
        for bank in CORE_BANKS:
            sets = [bank_records(recs, bank) for recs in all_recs]
            self.assertTrue(
                all(s == sets[0] for s in sets),
                f"bank {bank:#04x} should be content-stable across rebuilds",
            )
            self.assertGreater(len(sets[0]), 0, f"bank {bank:#04x} should be present")


class TestGenuineAllocatorNoiseRegisters(unittest.TestCase):
    """reg=8 and reg=60 vary within at least one independently-built
    4-sample batch on its own -- sufficient to confirm real variation,
    since a batch that happens to agree by chance does not establish
    stability, only a batch that disagrees establishes instability."""

    def test_reg8_unstable_within_old_batch_alone(self):
        recs = decode_all(OLD_BATCH)
        sets = [reg_records(r, 8) for r in recs]
        self.assertFalse(all(s == sets[0] for s in sets))

    def test_reg8_unstable_within_fresh_batch_alone(self):
        recs = decode_all(FRESH_BATCH)
        sets = [reg_records(r, 8) for r in recs]
        self.assertFalse(all(s == sets[0] for s in sets))

    def test_reg60_unstable_within_old_batch_alone(self):
        recs = decode_all(OLD_BATCH)
        sets = [reg_records(r, 60) for r in recs]
        self.assertFalse(all(s == sets[0] for s in sets))

    def test_reg60_is_a_clean_off_by_one_two_value_flip(self):
        """0x7e vs 0x7f in the reg=60, tag=131 records -- structured-
        looking, but the split does not align with the old/fresh
        (image-version) batch boundary (see module docstring): the
        original `conv_dilation3.mcode.gz` itself reads 0x7e, matching
        all 4 fresh builds, while its own batch-mates read 0x7f."""
        all_recs = decode_all(ALL_NAMES)
        values = set()
        for recs in all_recs:
            for kind, tag, payload in reg_records(recs, 60):
                if tag == 131 and payload is not None:
                    values.add(payload)
        self.assertEqual(values, {b"\x03~", b"\x03\x7f"})

    def test_reg8_confirmed_noisy_in_all_three_ops_now(self):
        """Mul (PR #1566, table-order coin flip), Gemm (PR #1576,
        unstable but coin-flip ruled out), and now Conv (this file,
        unstable and no offset table even exists) -- reg=8 is the one
        register this project has seen unstable across all three main
        ops, and in two of the three the known mechanism cannot
        explain it."""
        all_recs = decode_all(ALL_NAMES)
        sets = [reg_records(recs, 8) for recs in all_recs]
        self.assertFalse(all(s == sets[0] for s in sets))


class TestRegZeroAndSeventyEightDifferFromGemm(unittest.TestCase):
    """PR #1576 found reg=0 and reg=78 genuinely noisy for Gemm. Here,
    for Conv, both are fully stable -- the second noise mechanism's
    affected-register set is not a fixed, op-independent set."""

    def test_reg0_stable_here(self):
        all_recs = decode_all(ALL_NAMES)
        sets = [reg_records(recs, 0) for recs in all_recs]
        self.assertTrue(all(s == sets[0] for s in sets))

    def test_reg78_stable_here(self):
        all_recs = decode_all(ALL_NAMES)
        sets = [reg_records(recs, 78) for recs in all_recs]
        self.assertTrue(all(s == sets[0] for s in sets))


class TestRemainingRegistersAreStableEverywhere(unittest.TestCase):
    def test_stable_in_every_sample(self):
        self.assertEqual(len(STABLE_EVERYWHERE_REGS) + len(GENUINE_NOISE_REGS), 36)
        all_recs = decode_all(ALL_NAMES)
        for reg in STABLE_EVERYWHERE_REGS:
            sets = [reg_records(recs, reg) for recs in all_recs]
            self.assertTrue(
                all(s == sets[0] for s in sets),
                f"reg={reg} should be content-stable across all 8 samples",
            )


if __name__ == "__main__":
    unittest.main()
