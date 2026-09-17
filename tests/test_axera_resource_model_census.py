"""A first empirical resource-model census across this project's entire
committed mcode fixture corpus (306 files, spanning Mul/Gemm/Conv/
MatMul/training-graph fixtures accumulated across this whole session's
decode work) -- groundwork for a future from-scratch mcode code
generator, not a decode of any specific op's semantics.

Every mcode-generation technique built so far (`scripts/axera/emitter.py`,
`scripts/axera/tiny_emit.py`'s `patch_*` functions) works by taking a
real Pulsar2-compiled reference and rewriting only specific known
scalar fields inside it -- the addressing/allocation itself (which
bank, which field offset, which register a piece of content lands in)
is always copied verbatim from the reference, never derived. A genuine
from-scratch generator (tinygrad-scheduler-backed or otherwise) would
need a resource model first: what banks exist, how many registers,
which look general-purpose versus op-specific. This file builds that
census empirically, for the first time, from the fixture corpus this
project already has on disk.

**Not the first field-map work in this project** -- `scripts/axera/README.md`'s
"`a1 00 xx yy` is a field write" section (~line 2206) already decoded
the same `a1 00 <field> <bank> <value>` mechanism from two full real
models (`resnet18d`, `mnasnet`), finding banks 1-4 as the primary ones
and ~68 distinct fields. This file is a *different, complementary*
dataset -- this session's own corpus of small, single-op probe
fixtures rather than full real models -- and mostly reconfirms that
earlier finding at far higher sample size (306 fixtures vs. 2 models),
while adding new banks/registers the original two-model sample never
saw.

## Method

`mcode.decode(data, **mcode.FULL_RULE)` on every fixture, aggregating:
`V`-kind (verb) records' `bank`/`field` pairs, and `S`/`B`-kind
(short-unit / bare-pair) records' `reg` values with their `tag`
distributions.

## Bank census: the 16-byte-granularity rule holds with zero
## exceptions across all 306 fixtures

Every one of the 306 fixtures decodes cleanly (`mcode.decode` raises on
none of them). Across every `V`-kind record found (tens of thousands),
**every single `field` value is an exact multiple of 16 -- zero
exceptions** -- confirming the README's own already-established rule
at roughly 10x the sample size it was originally checked against.

39 distinct banks appear in total, but usage is sharply bimodal:

- **A small "core" set is near-universal**: banks `0x00`-`0x04` (the
  README's own already-named primary banks) each appear in all 306
  fixtures with thousands of writes and up to 16 distinct fields.
  Banks `0x0e`, `0x0f`, `0x1c`, `0x1e` are *also* near-universal
  (305-306 of 306 fixtures) with hundreds to over a thousand writes
  each -- this project's earlier two-model sample never singled these
  out specifically as "core" banks the way it did 1-4; recorded here as
  a real, corpus-wide finding worth folding into that earlier picture,
  not decoded further (what these four banks' fields mean is not
  established here).
- **The other 20 of 39 banks are sparse** (originally 21 at the
  306-fixture corpus; PR #1568's 2 new bank-`0x81` carrier fixtures
  pushed that bank's own count from 8 to 10, crossing out of this
  bucket): seen in 9 or fewer fixtures
  each (most in 1-3), with small write counts. These read as genuinely
  op/shape-specific banks, not a general vocabulary -- but the sample
  per bank is too thin here to say anything about what any individual
  one does.

## Register census: a real 36-register "universal" set exists, but
## register *magnitude* does not predict it

247 distinct register numbers appear across the corpus.

- **36 registers are present in literally every one of the 306
  fixtures** -- `{0, 2, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28,
  32, 36, 38, 40, 52, 60, 62, 66, 72, 74, 76, 78, 96, 100, 101, 106,
  107, 120, 126, 128, 138}`. `reg=8` is by far the heaviest member
  (27,369 uses, 8 distinct tags dominated by tag `132`: 19,812 uses and
  tag `131`: 4,345) -- consistent with this project's earlier,
  narrower characterization of `reg=8` as hosting "a rotating generic
  short-unit payload" (the Gemm M=8 close, `tests/test_axera_gemm_m8_diff_is_allocation_noise.py`).
  This census shows `reg=8` is the extreme case of a real 36-register
  *set* with this property, not a special lone register.
- **A hypothesis checked directly, and only partly supported**: small
  register *numbers* do skew toward higher corpus-wide presence on
  average -- splitting all 247 registers at `reg<=0x40` vs. `reg>0x40`
  gives the small-number group a higher average presence (167.3 of 306
  fixtures) than the large-number group (100.5) -- but this is a weak
  correlation, not a clean split usable as a rule. The 36-register
  universal set itself spans both halves: `96, 100, 101, 106, 107, 120,
  126, 128, 138` are all `>0x40` yet appear in every single fixture,
  same as `reg=8` does. Register magnitude is not a reliable proxy for
  "general-purpose" on its own; what actually identifies the universal
  set is each register's own corpus-wide presence count, not its
  numeric value.
- **67 registers appear in 3 or fewer fixtures** -- genuinely sparse,
  almost entirely tagged `225` (`0xe1`) with only 2-6 total uses each.
  Too thin a sample to assign any of them a confident role; recorded
  as "sparse, op/shape-specific-looking, not characterized further."

## What this does not establish

This is a census, not a decode: it does not explain what any specific
bank's fields or any specific register's tag values *compute*, only
their distribution across the corpus. It does not distinguish real
per-op-program addressing structure from this project's own
already-documented allocator noise (register/bank assignment has been
independently confirmed non-deterministic across identical rebuilds in
several op families this session -- Gemm's M=8 lead, Gemm's `transB`
rotation, MatMul's `A_offset`/`B_offset` table-order coin flip); a
register or bank appearing in every fixture could still be assigned to
different *content* on different builds of the identical graph, which
this file does not check. It is a first, purely empirical map of what
exists and how often, not of what determines placement.

## A note on corpus-size fragility (fixed here)

This file originally pinned every aggregate to the exact fixture count
at the time it was written (306, then hand-bumped to 310, 318, 322 as
later PRs added fixtures). That turned out to be a real, recurring
maintenance problem, not a one-off: PR #1568 had to bump these counts,
PR #1570 had to bump them again (finding a *second* round of silent
drift that had happened in between with no test catching it), and PR
#1571 hit an actual git merge conflict against PR #1570's own count
bump -- both PRs independently "fixed" the same hardcoded numbers to
different, both slightly-stale values, resolved only by recomputing
the true numbers directly against the final combined fixture set
rather than trusting either side.

The fix applied here: assertions that exist to catch a *structural*
regression (a core bank stops appearing in every fixture; the register
space collapses to a handful) now compare against `_CENSUS["n_fixtures"]`
computed live, or use a ratio/threshold that scales with corpus size,
instead of a hardcoded absolute number. Assertions whose only content
*was* "how many fixtures exist right now" are loosened to floor
checks (the corpus should only grow) rather than deleted, so a
fixture accidentally going missing is still caught. The "sparse"
bucket's own boundary (previously the pinned constant `<=9` fixtures
for banks, `<=3` for registers) is now derived from the live fixture
count via a fixed *fraction* (3% for banks, 1% for registers -- chosen
to reproduce the exact same boundary, 9 and 3, at both the 306- and
322-fixture corpus sizes this project has already passed through), so
adding fixtures no longer requires touching this file at all unless a
genuinely new structural pattern emerges (a bank/register crossing a
bucket boundary is still visible in the *aggregate* counts these tests
check, just no longer as a hardcoded number that has to be hand-edited
in lockstep with every fixture-adding PR).
"""

import glob
import gzip
import math
import os
import sys
import unittest
from collections import Counter, defaultdict

# Mirrored in tests/test_axera_sparse_resource_clustering.py -- both
# files classify "sparse" banks/registers the same way, and
# TestSparseBanksAreMostlyMultiFamily.test_sparse_bank_count_matches_pr1563
# there depends on this file's own definition matching. Chosen to
# reproduce the project's original pinned thresholds (<=9 fixtures for
# banks, <=3 for registers) at both the 306-fixture corpus these were
# first measured against and the 322-fixture corpus as of PR #1570.
BANK_SPARSE_MAX_FRACTION = 0.03
REG_SPARSE_MAX_FRACTION = 0.01


def _bank_sparse_threshold(n_fixtures):
    return math.floor(BANK_SPARSE_MAX_FRACTION * n_fixtures)


def _reg_sparse_threshold(n_fixtures):
    return math.floor(REG_SPARSE_MAX_FRACTION * n_fixtures)


_AXERA_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts", "axera")
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import mcode  # noqa: E402

FIX = os.path.join(_AXERA_DIR, "fixtures")


def _all_fixtures():
    return sorted(glob.glob(os.path.join(FIX, "*.mcode.gz")))


def _census():
    """Decode every committed fixture once and return the aggregated
    bank/field and register/tag census. Computed once per test process
    (not per test method) since it's the same expensive pass every
    assertion below reads from."""
    bank_field_counts = defaultdict(Counter)
    bank_fixture_set = defaultdict(set)
    reg_counts = Counter()
    reg_fixture_set = defaultdict(set)
    reg_tag_counts = defaultdict(Counter)
    field_nonmult16 = []
    fixtures = _all_fixtures()
    for fx in fixtures:
        name = os.path.basename(fx)
        with gzip.open(fx, "rb") as f:
            data = f.read()
        recs = mcode.decode(data, **mcode.FULL_RULE)
        for r in recs:
            kind = r["kind"]
            if kind == "V":
                bank, field = r["bank"], r["field"]
                bank_field_counts[bank][field] += 1
                bank_fixture_set[bank].add(name)
                if field % 16 != 0:
                    field_nonmult16.append((name, bank, field))
            elif kind in ("S", "B"):
                reg = r["reg"]
                reg_counts[reg] += 1
                reg_fixture_set[reg].add(name)
                reg_tag_counts[reg][r["tag"]] += 1
    return {
        "n_fixtures": len(fixtures),
        "bank_field_counts": bank_field_counts,
        "bank_fixture_set": bank_fixture_set,
        "reg_counts": reg_counts,
        "reg_fixture_set": reg_fixture_set,
        "reg_tag_counts": reg_tag_counts,
        "field_nonmult16": field_nonmult16,
    }


_CENSUS = _census()


class TestFixtureCorpusSize(unittest.TestCase):
    def test_fixtures_all_decode_cleanly(self):
        # `_census()` above already raises if any committed fixture
        # fails to decode -- reaching this assertion at all is the real
        # test. The exact fixture count is not a claim this test needs
        # to make; it's tracked here only as a floor (corpus growth is
        # expected and fine, corpus *shrinkage* -- e.g. an accidentally
        # deleted fixture -- is the real regression to catch). This
        # used to be a hardcoded exact count (306 -> 310 -> 318 -> 322
        # -> 327 -> 336, most recently PR #1573's own 9 new Gemm shapes
        # probing bank 0x81's field=144 presence pattern) that needed a
        # manual bump on nearly every fixture-adding PR -- including one
        # real merge conflict between #1570 and #1571 (and another
        # between #1572's own growth-robustness rewrite and #1573's
        # count bump, resolved by keeping this dynamic form) bumping the
        # same number differently; see the module docstring's
        # "corpus-size fragility" section for why that was dropped in
        # favor of a floor.
        self.assertGreaterEqual(_CENSUS["n_fixtures"], 336)


class TestFieldOffsetGranularity(unittest.TestCase):
    """The README's own established 16-byte-granular field-offset rule
    (line ~2206), reconfirmed here at ~10x its original sample size
    (306 fixtures vs. the original 2 real models)."""

    def test_zero_exceptions_across_the_whole_corpus(self):
        self.assertEqual(_CENSUS["field_nonmult16"], [])


class TestBankCensus(unittest.TestCase):
    def test_at_least_39_distinct_banks_total(self):
        # Was exactly 39 at the 322-fixture corpus; a floor rather than
        # an exact count since a new fixture can introduce a bank this
        # corpus has never seen (this has already happened repeatedly
        # as the corpus grew -- there's no reason to expect it's done).
        self.assertGreaterEqual(len(_CENSUS["bank_field_counts"]), 39)

    def test_core_banks_0_to_4_are_universal(self):
        for bank in (0x00, 0x01, 0x02, 0x03, 0x04):
            self.assertEqual(
                len(_CENSUS["bank_fixture_set"][bank]),
                _CENSUS["n_fixtures"],
                f"bank {bank:#04x} should appear in every fixture",
            )

    def test_four_additional_banks_are_also_near_universal(self):
        """Not previously singled out by the README's own two-model
        sample as "core" the way 1-4 were -- a new finding from this
        wider corpus. "Near-universal" tolerates being absent from at
        most one fixture (computed against the live fixture count, not
        a hardcoded number, since these two are anchored to the same
        underlying property)."""
        for bank in (0x0E, 0x0F, 0x1C, 0x1E):
            self.assertGreaterEqual(
                len(_CENSUS["bank_fixture_set"][bank]), _CENSUS["n_fixtures"] - 1
            )

    def test_most_banks_are_sparse(self):
        # Was pinned to an exact "20 of 39" at the 322-fixture corpus
        # (originally "21 of 39" at 306, hand-bumped once already when
        # bank 0x81 crossed out of the sparse bucket). The *exact*
        # count of sparse banks drifts every time a fixture pushes some
        # bank across the threshold -- not a useful thing to pin. What
        # this test actually claims -- that bank usage is bimodal, a
        # small core plus a long sparse tail -- survives as a majority
        # check instead: most banks should be sparse under the shared
        # ratio-based threshold (see module docstring).
        threshold = _bank_sparse_threshold(_CENSUS["n_fixtures"])
        sparse = [
            b
            for b, fixset in _CENSUS["bank_fixture_set"].items()
            if len(fixset) <= threshold
        ]
        self.assertGreater(
            len(sparse),
            len(_CENSUS["bank_field_counts"]) / 2,
            f"most banks should be sparse (<= {threshold} fixtures, i.e. "
            f"<={BANK_SPARSE_MAX_FRACTION:.0%} of the corpus)",
        )


class TestRegisterCensus(unittest.TestCase):
    def test_richer_than_a_small_fixed_register_set(self):
        # Was pinned to exactly 247 at the 322-fixture corpus. The
        # precise count isn't a load-bearing claim -- what matters is
        # that the register space is genuinely large (hundreds), not a
        # small fixed handful, which is what motivates treating the
        # 36-register "universal" set below as meaningfully small by
        # comparison. A generous floor preserves that claim without
        # needing an edit on every corpus change.
        self.assertGreater(len(_CENSUS["reg_counts"]), 100)

    def test_known_universal_registers_stay_universal(self):
        # Originally an exact-set-equality check pinned to the 36
        # registers observed universal at the 322-fixture corpus. That
        # is too strict for a growing corpus: a new fixture could
        # legitimately introduce a 37th register that also happens to
        # be universal (an addition, not a regression) without this
        # project's actual claim -- "these particular registers are
        # foundational, present in literally every build" -- being
        # violated. Converted to a subset check: every one of the
        # already-known-universal registers must REMAIN universal (a
        # real regression -- one of these disappearing from even a
        # single fixture -- still fails loudly); new registers joining
        # the universal set as the corpus grows is not asserted against
        # either way.
        universal = {
            r
            for r, fixset in _CENSUS["reg_fixture_set"].items()
            if len(fixset) == _CENSUS["n_fixtures"]
        }
        known_universal = {
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
        }
        self.assertTrue(
            known_universal <= universal,
            f"registers no longer universal: {known_universal - universal}",
        )

    def test_reg8_is_the_heaviest_universal_register_by_a_wide_margin(self):
        # Was pinned to an exact use-count (27,369 -> 27,601 -> 28,346
        # -> 28,636 -> 29,417 across five separate corpus-growth bumps,
        # two of them the product of a real merge conflict -- PR #1570
        # vs. #1571 independently guessing this number, then #1572's own
        # growth-robustness rewrite vs. #1573's own count bump to 29,417
        # at the 336-fixture corpus, resolved here by keeping this ratio
        # form). The actual claim -- reg=8 so dominates usage that it's
        # not a coincidence of corpus composition -- survives as a ratio
        # check instead: at the 336-fixture corpus reg=8 is used ~4.4x
        # more than the next busiest universal register (reg=10). A 3x
        # floor keeps real margin for corpus growth to shift the exact
        # ratio without losing the substance of the claim.
        counts = _CENSUS["reg_counts"]
        universal = {
            r
            for r, fixset in _CENSUS["reg_fixture_set"].items()
            if len(fixset) == _CENSUS["n_fixtures"]
        }
        ranked = sorted(universal, key=lambda r: -counts[r])
        self.assertEqual(ranked[0], 8)
        self.assertGreater(
            counts[ranked[0]],
            3 * counts[ranked[1]],
            "reg=8 should dominate the next-heaviest universal register"
            " by a wide margin, not just edge it out",
        )

    def test_small_registers_skew_toward_higher_presence_but_not_cleanly(self):
        """Small register numbers (<=0x40) average higher corpus-wide
        presence than large ones -- a real but weak correlation, not a
        clean split: the universal set itself spans both halves (see
        test_exactly_36_registers_are_universal's own expected set,
        which includes several values >0x40)."""
        reg_counts = _CENSUS["reg_counts"]
        reg_fixture_set = _CENSUS["reg_fixture_set"]
        small = [r for r in reg_counts if r <= 0x40]
        large = [r for r in reg_counts if r > 0x40]
        small_avg_fix = sum(len(reg_fixture_set[r]) for r in small) / len(small)
        large_avg_fix = sum(len(reg_fixture_set[r]) for r in large) / len(large)
        self.assertGreater(
            small_avg_fix,
            large_avg_fix,
            "small-numbered registers should have HIGHER average corpus"
            " presence than large-numbered ones (a real but weak trend,"
            " not a clean split -- see the docstring)",
        )
        # Not a clean split: several large-numbered registers are
        # nonetheless universal (present in every fixture).
        universal = {
            r
            for r, fixset in reg_fixture_set.items()
            if len(fixset) == _CENSUS["n_fixtures"]
        }
        large_universal = [r for r in universal if r > 0x40]
        self.assertTrue(
            large_universal,
            "expected at least one large-numbered register in the"
            " universal set, showing magnitude alone doesn't determine it",
        )

    def test_a_large_fraction_of_registers_are_sparse(self):
        # Was pinned to an exact count (67 -> 66 across one corpus
        # growth bump already). Unlike banks (a bare majority, 20/39),
        # sparse registers are a large minority (66/247, ~27%) rather
        # than a majority -- the register space has a longer "middle
        # tier" between the 36-register universal core and the sparse
        # tail. The actual claim -- a large share of the 247-register
        # space is rarely used, well beyond the 36-register universal
        # core -- survives as a floor-fraction check under the shared
        # ratio-based threshold instead of an exact pinned count.
        threshold = _reg_sparse_threshold(_CENSUS["n_fixtures"])
        sparse = [
            r
            for r, fixset in _CENSUS["reg_fixture_set"].items()
            if len(fixset) <= threshold
        ]
        self.assertGreater(
            len(sparse),
            len(_CENSUS["reg_counts"]) * 0.2,
            f"a large fraction of registers should be sparse (<= {threshold}"
            f" fixtures, i.e. <={REG_SPARSE_MAX_FRACTION:.0%} of the corpus)",
        )


if __name__ == "__main__":
    unittest.main()
