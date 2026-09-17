"""Closing `tests/test_axera_gemm_sparse_bank_n_boundary.py` (PR #1568)'s
own flagged gap: it pinned Gemm's `0x81`/`0xe1` bank-switch boundary to
exactly `N=32` (uses `0xe1`) vs. `N=33` (uses `0x81`) at fixed `M=1,
K=512`, but explicitly left open "whether the threshold is `N` in
absolute terms, or `N` relative to `K` (a ratio, or `N*K` total), or
something about `M=1` specifically... no second `K` value was swept
here." This file sweeps a second `K` value, `K=256` (half of PR #1568's
`K=512`), across the same `N` region.

## The boundary is unchanged at `K=256`: still exactly `N=32` / `N=33`

Sweeping `N` from 8 up to 128 at `M=1, K=256`, then bisecting the gap
(`N=36,40,44` all carry `0x81`) and pinning it (`N=33,34,35` all `0x81`)
lands the switch at the *identical* `N=32`/`N=33` boundary PR #1568
found at `K=512` -- a 2x change in `K` does not move the threshold at
all. This directly supports the **absolute-N** hypothesis over the two
`K`-relative alternatives PR #1568 left open (a ratio like `N/K`, or a
product like `N*K`, would each predict a different `N` threshold when
`K` is halved; neither happens).

Confirmed with independent rebuilds on both sides of the boundary
(`N=32` rebuild, `N=33` rebuild -- different RNG-seeded calibration and
weight draws each time, not repeated identical builds), matching PR
#1568's own rigor standard.

## A refinement PR #1568 did not anticipate: below the threshold, `K=256`
## has *neither* bank, not `0xe1`

PR #1568 characterized `0x81`/`0xe1` as "mutually exclusive substitutes
at this shape family": every build it checked had exactly one of the
two, never both, never neither. That framing was built entirely on
`K=512` data. At `K=256`, `N=16` and the pre-threshold boundary point
`N=32` both carry **neither** bank -- not `0xe1` the way `K=512`'s own
`N=20`/`N=32` pre-threshold builds did. So the "substitute" framing does
not universalize across `K`: the post-threshold identity is consistently
`0x81` at both `K` values (supporting the absolute-N reading), but the
pre-threshold slot is `K`-dependent -- filled by `0xe1` at `K=512`,
empty entirely at `K=256`. What decides whether the pre-threshold slot
is `0xe1`-filled or bank-absent (some other, unswept `K` value, or a
different boundary along `K` itself) is not tested here.

## What remains open

Only two `K` values (256, 512) were compared, at `M=1` only; whether the
`N=32` threshold itself is truly `K`-independent at more extreme `K`
(e.g. `K=1024`, or very small `K`), and what determines the pre-threshold
`0xe1`-vs-absent split observed here, are both untested. What the
`0x81`/`0xe1` fields actually *compute* remains undecoded, same caveat
as PR #1568.
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


class TestAllFixturesDecodeCleanly(unittest.TestCase):
    def test_all_five_fixtures_decode_without_error(self):
        """Unlike PR #1568's own K=512 boundary fixtures (which
        mcode.check() reports as fully clean), all five K=256 fixtures
        here -- including both boundary sides -- carry the identical
        small "93.x% coverage" warning (four fixed unexplained-byte
        ranges: (321,322),(428,431),(433,434),~(456,458)). Same warning
        on every fixture regardless of N or bank choice means it is a
        K=256-shape-family quirk in mcode.py's own decode rules, not
        something specific to the 0x81/0xe1 boundary -- so this only
        checks that decode() itself doesn't raise, rather than requiring
        an empty check() list."""
        for name in (
            "gemm_1x256x16.mcode.gz",
            "gemm_1x256x32.mcode.gz",
            "gemm_1x256x32_rebuild.mcode.gz",
            "gemm_1x256x33.mcode.gz",
            "gemm_1x256x33_rebuild.mcode.gz",
        ):
            data = load(name)
            recs = mcode.decode(data, **mcode.FULL_RULE)
            self.assertGreater(len(recs), 0, name)


class TestN32AtK256HasNeitherBank(unittest.TestCase):
    """Unlike K=512's own N=32 (which carries 0xe1), K=256's N=32 carries
    neither bank -- the "mutually exclusive substitute" framing from PR
    #1568 does not universalize across K below the threshold."""

    def test_original_and_rebuild_have_neither_bank(self):
        for name in ("gemm_1x256x32.mcode.gz", "gemm_1x256x32_rebuild.mcode.gz"):
            banks = banks_of(load(name))
            self.assertNotIn(0x81, banks, name)
            self.assertNotIn(0xE1, banks, name)

    def test_n16_also_has_neither_bank(self):
        banks = banks_of(load("gemm_1x256x16.mcode.gz"))
        self.assertNotIn(0x81, banks)
        self.assertNotIn(0xE1, banks)


class TestN33AtK256UsesBank81(unittest.TestCase):
    """Matches K=512's own N=33 identity exactly (0x81, not 0xe1) --
    the post-threshold identity is K-independent."""

    def test_original_and_rebuild_both_use_0x81(self):
        for name in ("gemm_1x256x33.mcode.gz", "gemm_1x256x33_rebuild.mcode.gz"):
            banks = banks_of(load(name))
            self.assertIn(0x81, banks, name)
            self.assertNotIn(0xE1, banks, name)


class TestBoundaryMatchesK512ExactlySupportingAbsoluteN(unittest.TestCase):
    def test_n32_lacks_bank81_n33_has_it_same_as_k512(self):
        n32 = banks_of(load("gemm_1x256x32.mcode.gz"))
        n33 = banks_of(load("gemm_1x256x33.mcode.gz"))
        self.assertNotIn(0x81, n32)
        self.assertIn(0x81, n33)


if __name__ == "__main__":
    unittest.main()
