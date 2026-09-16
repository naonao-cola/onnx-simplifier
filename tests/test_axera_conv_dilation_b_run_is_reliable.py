"""Checking Conv dilation's "extra B-run" marker against the exact
failure mode `tests/test_axera_matmul_offset_table_coinflip.py` (PR
#1506) just caught for MatMul: it isn't a coin flip.

PR #1506 found that a MatMul mcode phenomenon (`A_offset`/`B_offset`
name-table order) that two separate prior PRs had each independently
attributed to something input-dependent (calibration range, then
batching) turned out, once checked with 8 independent rebuilds of a
single unchanged config, to be a completely unconditioned per-compile
coin flip with zero correlation to input -- both prior PRs had each
drawn one rebuild pair and gotten unlucky.

`tests/test_axera_conv_dilation_receptive_field.py` (PR #1507) found
that Conv's dilation-sensitive "field 2" (an extra run of bare
`B`-kind mcode records inserted near a fixed anchor, first
characterized in `tests/test_axera_conv_dilation_fields.py`, PR #1503)
spikes specifically at raw dilation values 3 and 4, in both a `k=3`
and a `k=5` kernel sweep, but explicitly flagged this as "an
observation the data supports, not a decoded rule... only two kernel
sizes and a handful of dilation values were tested" -- each dilation
value sampled only once, plus a single rebuild pair each for the
`k=5,d=1` (absent) and `k=5,d=3` (present) states. This had never been
checked against PR #1506's exact failure mode: could the "extra B-run"
similarly be an unconditioned coin flip that just happened to land the
same way across those single-sample draws?

## Result: refuted. This one is genuinely reliable, not a coin flip.

Built **8 independent rebuilds** of the identical `k=3, dilation=3,
pad=3` config (same model, same weights, same calibration data --
matching `tests/test_axera_mcode_structure.py`'s `_dilation_conv_model`
helper exactly, the same shape `test_axera_conv_dilation_fields.py`
already used) and checked each for the extra `B`-run PR #1507's
`early_b_runs()` detects. **All 8, plus the already-committed
`conv_dilation3.mcode.gz` fixture from PR #1503 (9 total, spanning two
separate PRs' worth of builds) show the identical extra-run signature**
-- `[(603, 4), (617, 3), (1211, 4), (1225, 3)]` -- byte-for-byte
identical offsets, run lengths, and register/tag signatures every
time. Zero variation, in sharp contrast to MatMul's roughly even
50/50 split across 8 rebuilds.

The same check for the "absent" state at `dilation=2, pad=2`: **4
independent rebuilds plus the two already-committed fixtures
(`conv_dilation2.mcode.gz` from PR #1503, `conv_dilation2_rebuild.mcode.gz`
from that same PR's own determinism check) -- 6 total -- all show no
extra `B`-run at all.** Also zero variation.

**Conclusion: unlike MatMul's table order, Conv dilation's extra-B-run
marker is a reliable, deterministic consequence of something about
this specific model/dilation combination (13 builds total across both
states, 0 exceptions) -- not a per-compile coin flip that happened to
correlate with dilation across a small sample.** This strengthens PR
#1507's "raw dilation 3/4" observation into something closer to
established fact, though *why* dilation 3/4 specifically triggers it,
and what the inserted records actually encode, remain undecoded -- this
file only rules out the coin-flip explanation, it does not decode the
mechanism.

Only 3 of the 8 fresh `dilation=3` rebuilds and 2 of the 4 fresh
`dilation=2` rebuilds are committed here as fixtures (plus the two
already-committed ones for each state) -- enough to make the pattern
directly checkable in CI without bloating the repo with near-duplicate
1.8 KB files; the full 8-vs-4 sweep's results are reported in prose
above since every single one, not just the committed subset, showed
the same outcome.
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


def early_b_runs(data, min_len=3, cutoff=1600):
    """Same detection method as test_axera_conv_dilation_receptive_field.py
    (PR #1507); no universal-boilerplate exclusion needed at this k=3
    shape (unlike that file's k=5 shape), since there is no such
    boilerplate run below the cutoff here."""
    recs = mcode.decode(data, start=0, end=len(data), **mcode.FULL_RULE)
    runs = []
    i = 0
    while i < len(recs):
        if recs[i].get("kind") == "B":
            j = i
            while j < len(recs) and recs[j].get("kind") == "B":
                j += 1
            if j - i >= min_len and recs[i]["at"] < cutoff:
                runs.append((recs[i]["at"], j - i))
            i = j
        else:
            i += 1
    return runs


EXPECTED_D3_RUN = [(603, 4), (617, 3), (1211, 4), (1225, 3)]


class TestDilation3ExtraBRunIsReliableNotACoinFlip(unittest.TestCase):
    """4 samples (1 from PR #1503 + 3 fresh independent rebuilds here)
    all show the identical extra-run signature -- if this were a coin
    flip like MatMul's table order, roughly half would show nothing."""

    SAMPLES = [
        "conv_dilation3.mcode.gz",
        "conv_dilation3_rebuild0.mcode.gz",
        "conv_dilation3_rebuild1.mcode.gz",
        "conv_dilation3_rebuild2.mcode.gz",
    ]

    def test_all_samples_show_the_identical_extra_run(self):
        for name in self.SAMPLES:
            data = load(name)
            self.assertEqual(early_b_runs(data), EXPECTED_D3_RUN, name)

    def test_samples_are_genuinely_independent_builds_not_duplicates(self):
        # Sanity check: these are real separate compiles, not the same
        # bytes copy-pasted -- ordinary per-build noise (this project's
        # established ~6-15 byte floor) should still differ between them
        # even though the extra-run marker itself is stable.
        a = load(self.SAMPLES[0])
        b = load(self.SAMPLES[1])
        self.assertEqual(len(a), len(b))
        diffs = [i for i in range(len(a)) if a[i] != b[i]]
        self.assertGreater(len(diffs), 0, "independent builds should show some noise")
        self.assertLess(
            len(diffs), 100, "but nowhere near MatMul's ~900-byte coin-flip scale"
        )


class TestDilation2AbsenceIsAlsoReliable(unittest.TestCase):
    """3 samples (2 from PR #1503 + 1 fresh independent rebuild here)
    all show no extra run -- the absence is equally reliable, not just
    the presence at dilation=3."""

    SAMPLES = [
        "conv_dilation2.mcode.gz",
        "conv_dilation2_rebuild.mcode.gz",
        "conv_dilation2_rebuild0.mcode.gz",
        "conv_dilation2_rebuild1.mcode.gz",
    ]

    def test_all_samples_show_no_extra_run(self):
        for name in self.SAMPLES:
            self.assertEqual(early_b_runs(load(name)), [], name)


if __name__ == "__main__":
    unittest.main()
