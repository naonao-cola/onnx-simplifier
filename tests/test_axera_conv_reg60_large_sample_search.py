"""Continues `tests/test_axera_conv_reg60_mechanism.py` (PR #1586)'s own
repeatedly-flagged open thread: Conv's `reg=60` "binary path switch" (a
28-byte all-or-nothing state flip across independent rebuilds of
`Conv(k=3, dilation=3, pad=3, cin=4, cout=4, insz=16)`, `RandomState(0)`
weights and calibration) had its root trigger checked against several
hypotheses (toolchain image version, build batch, the `reg=8` pool's own
slot content) with all of them ruled out -- but using only 8 total
samples (a 5-vs-3 split). This file gives the search real statistical
power: **24 new, independent rebuilds** (`pulsar2:7.0-lite`, the only
image loaded in this worktree's Docker daemon), for 32 total samples
combined with the 8 already committed.

## The fuller split: 29-vs-3, not close to 5-vs-3's own ~62.5%/37.5%

Every one of the 24 new samples reads `reg=60 = 0x7e` (`payload[-1]`).
Combined with the original 8 (5x `0x7e`, 3x `0x7f`), the corpus-wide
split is now **29 `0x7e` / 3 `0x7f`** out of 32 -- ~90.6%/9.4%, a much
more lopsided ratio than the original 8-sample snapshot suggested. This
alone is a real, useful update: the original 5-vs-3 split looked like it
could plausibly be a near-50/50 coin flip sampled unluckily; at N=32 it
clearly is not -- `0x7e` is the overwhelmingly dominant outcome, `0x7f`
the rare one, at least for this specific `RandomState(0)`-seeded shape.

## A genuinely new, statistically strong correlate: toolchain image
## version tracks the OBSERVED RATE, even though it does not
## deterministically fix the value

Splitting all 32 samples by which `pulsar2` Docker image built them
(this project's own already-documented convention,
`tests/test_axera_conv_rebuild_stability.py`'s own `OLD_BATCH` =
`pulsar2:6.0-lite`, `FRESH_BATCH` = `pulsar2:7.0-lite`) gives:

| image | n | `0x7e` | `0x7f` |
| --- | --- | --- | --- |
| `pulsar2:6.0-lite` (`OLD_BATCH`: `conv_dilation3`, `_rebuild{0,1,2}`) | 4 | 1 | 3 |
| `pulsar2:7.0-lite` (`FRESH_BATCH`'s own 4 + this file's 24 new) | 28 | **28** | **0** |

**Every single one of the 28 `pulsar2:7.0-lite` samples reads `0x7e`,
zero exceptions** -- while the 4-sample `pulsar2:6.0-lite` batch alone
already contains 3 of the corpus's only 3 `0x7f` instances. PR #1586's
own "not the old/fresh batch boundary" finding is still correct as
stated (it used exactly one within-`OLD_BATCH` counter-example --
`conv_dilation3` itself reads `0x7e` despite being `6.0-lite`, same as
its `0x7f`-reading `_rebuild{0,1,2}` siblings -- to prove image version
does not *deterministically fix* the outcome). But that check only had
4 `pulsar2:7.0-lite` samples to compare against, and all 4 coincidentally
read `0x7e` -- not enough power to notice that `7.0-lite`'s own true
rate of `0x7f` might be dramatically lower than `6.0-lite`'s. This file's
24 additional `7.0-lite` samples make that visible for the first time:
**image version very plausibly shifts the underlying probability of this
near-tie substantially, even though it does not act as a hard switch.**

This is consistent with (not proof of) PR #1586's own original reading --
a genuine floating-point near-tie in Pulsar2's own per-channel
weight-statistics reduction order, with thread-scheduling non-determinism
deciding the outcome only when the computation is close enough to a tie
for order to matter. A compiler version change between `6.0` and `7.0`
plausibly altered the reduction order/scheduling enough to move this
specific shape's own computation further from that tie (without proving
it moved the *calibration data's* own tie-closeness the way
`tests/test_axera_conv_binary_cluster_calibration_dependence.py`
(PR #1598) showed a DIFFERENT calibration seed can).

**Honest limits, stated plainly:**

- This worktree has no `pulsar2:6.0-lite` image tarball loaded (only
  `7.0-lite`), so no additional `6.0-lite` samples could be built here to
  either confirm or refute that its own true `0x7f` rate is genuinely
  ~75% (the `n=4` estimate) rather than a small-sample artifact of its
  own -- `6.0-lite`'s own rate remains known only from the original 4
  samples, unchanged by this file.
- Zero occurrences of `0x7f` in 28 `7.0-lite` samples is strong evidence
  the true rate is much lower than `6.0-lite`'s own ~75%, but does not
  prove it is exactly zero -- a large-enough further sample could still
  find one. This file reports a strong correlation with the observed
  RATE, not a decoded, deterministic root cause.
- This is a genuine strengthening of the "toolchain-adjacent" hypothesis
  PR #1586/#1598 area already touched on and moved past too quickly with
  too little data, not a full reversal of any prior finding -- no earlier
  PR claimed image version fully explains the switch, and this file does
  not either.

## The 28-offset cluster is unchanged at N=32

Re-running PR #1586's exact brute-force scan (a position counts only if
constant *within* each of the two `reg=60`-defined groups but differs
between them) across all 32 samples finds **the identical 28 offsets**
PR #1586 originally found at N=8 -- zero more, zero fewer, recomputed
from scratch here, not trusted from any prior file's own hardcoded list.
The cluster's own membership is robust to 4x more data; only the
statistical picture of what predicts landing in each group has
sharpened.

## Build-order/index is not informative here

All 24 of this file's own new samples were built in a fixed, known
sequence (`r0` through `r23`) and every single one reads `0x7e` --
zero internal variance, so no odd/even or early/late correlate can even
be checked within this batch alone (a real observation, not a gap: if
build order/index mattered on its own, at least SOME variance would be
expected across 24 sequential builds of the identical config, and none
appeared).
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

OLD_6LITE = [
    "conv_dilation3.mcode.gz",
    "conv_dilation3_rebuild0.mcode.gz",
    "conv_dilation3_rebuild1.mcode.gz",
    "conv_dilation3_rebuild2.mcode.gz",
]
FRESH_7LITE_ORIG = [
    "conv_dilation3_v7stability_r0.mcode.gz",
    "conv_dilation3_v7stability_r1.mcode.gz",
    "conv_dilation3_v7stability_r2.mcode.gz",
    "conv_dilation3_v7stability_r3.mcode.gz",
]
NEW_7LITE = [f"conv_dilation3_largesample_r{i}.mcode.gz" for i in range(24)]
ALL_7LITE = FRESH_7LITE_ORIG + NEW_7LITE
ALL_NAMES = OLD_6LITE + ALL_7LITE

GROUP_A_0X7E = [n for n in OLD_6LITE if n == "conv_dilation3.mcode.gz"] + ALL_7LITE
GROUP_B_0X7F = [
    "conv_dilation3_rebuild0.mcode.gz",
    "conv_dilation3_rebuild1.mcode.gz",
    "conv_dilation3_rebuild2.mcode.gz",
]

KNOWN_CORRELATED_OFFSETS = frozenset(
    [
        687,
        1295,
        2161,
        2190,
        2191,
        2192,
        2198,
        2199,
        2200,
        2206,
        2207,
        2208,
        2214,
        2215,
        2216,
        2550,
        2561,
        2562,
        2563,
        2568,
        2569,
        2570,
        2575,
        2576,
        2577,
        2582,
        2583,
        2584,
    ]
)


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def decode(name):
    return mcode.decode(load(name), **mcode.FULL_RULE)


def reg60_value(recs):
    hits = [
        r
        for r in recs
        if r["kind"] == "S"
        and r.get("reg") == 60
        and r.get("tag") == 131
        and r.get("payload")
    ]
    assert len(hits) >= 1, hits
    return hits[0]["payload"][-1]


class TestAllNewFixturesDecodeCleanly(unittest.TestCase):
    def test_3528_bytes_zero_check_errors(self):
        for n in NEW_7LITE:
            data = load(n)
            self.assertEqual(len(data), 3528, n)
            self.assertEqual(mcode.check(data), [], n)


class TestFullerSplitIsTwentyNineVsThree(unittest.TestCase):
    """At N=32 (up from N=8), the split is 29 `0x7e` / 3 `0x7f` -- far
    from a near-50/50 coin flip, `0x7e` is the overwhelmingly dominant
    outcome for this shape/seed."""

    def test_all_new_samples_are_0x7e(self):
        for n in NEW_7LITE:
            self.assertEqual(reg60_value(decode(n)), 0x7E, n)

    def test_corpus_wide_split_is_29_vs_3(self):
        vals = [reg60_value(decode(n)) for n in ALL_NAMES]
        self.assertEqual(sum(1 for v in vals if v == 0x7E), 29)
        self.assertEqual(sum(1 for v in vals if v == 0x7F), 3)
        self.assertEqual(len(vals), 32)


class TestToolchainVersionCorrelatesWithObservedRate(unittest.TestCase):
    """The core new finding: every one of the 28 `pulsar2:7.0-lite`
    samples reads `0x7e` (zero exceptions), while the 4-sample
    `pulsar2:6.0-lite` batch alone contains all 3 of the corpus's own
    `0x7f` instances. This does not prove image version deterministically
    fixes the outcome (PR #1586's own single within-6.0-lite
    counter-example, `conv_dilation3` itself, already rules that out --
    reconfirmed below) -- but it is a real, statistically strong
    correlation with the RATE, invisible at the original n=4-per-image
    sample size."""

    def test_all_seven_lite_samples_are_0x7e(self):
        for n in ALL_7LITE:
            self.assertEqual(reg60_value(decode(n)), 0x7E, n)
        self.assertEqual(len(ALL_7LITE), 28)

    def test_six_lite_batch_contains_all_three_0x7f_instances(self):
        vals = {n: reg60_value(decode(n)) for n in OLD_6LITE}
        f_count = sum(1 for v in vals.values() if v == 0x7F)
        self.assertEqual(f_count, 3)
        self.assertEqual(len(OLD_6LITE), 4)

    def test_toolchain_does_not_deterministically_fix_the_value(self):
        """Reconfirms PR #1586's own finding: conv_dilation3.mcode.gz
        (6.0-lite) reads 0x7e despite sharing its toolchain with three
        0x7f-reading siblings -- image version alone is not a hard
        switch, only a strong rate correlate."""
        self.assertEqual(reg60_value(decode("conv_dilation3.mcode.gz")), 0x7E)
        for n in (
            "conv_dilation3_rebuild0.mcode.gz",
            "conv_dilation3_rebuild1.mcode.gz",
            "conv_dilation3_rebuild2.mcode.gz",
        ):
            self.assertEqual(reg60_value(decode(n)), 0x7F, n)


class TestClusterMembershipUnchangedAtThirtyTwoSamples(unittest.TestCase):
    """Recomputes PR #1586's own 28-offset cluster from scratch against
    the fuller 29-vs-3 grouping -- confirms the identical offset set,
    not just trusted from the smaller-sample discovery."""

    def test_exactly_28_offsets_matching_the_known_set(self):
        datas = {n: load(n) for n in ALL_NAMES}
        length = len(datas[ALL_NAMES[0]])
        for n in ALL_NAMES:
            self.assertEqual(len(datas[n]), length, n)

        found = []
        for i in range(length):
            vals_a = {datas[n][i] for n in GROUP_A_0X7E}
            vals_b = {datas[n][i] for n in GROUP_B_0X7F}
            if len(vals_a) == 1 and len(vals_b) == 1 and vals_a != vals_b:
                found.append(i)

        self.assertEqual(frozenset(found), KNOWN_CORRELATED_OFFSETS)
        self.assertEqual(len(found), 28)


class TestNewSamplesShowZeroInternalVariance(unittest.TestCase):
    """All 24 new samples, built in a fixed sequential order, agree with
    each other -- build order/index cannot be tested as a correlate
    within this batch alone since there is no variance to correlate
    against, itself a real (if limited) observation."""

    def test_all_new_samples_agree_with_each_other(self):
        vals = {reg60_value(decode(n)) for n in NEW_7LITE}
        self.assertEqual(vals, {0x7E})


if __name__ == "__main__":
    unittest.main()
