"""Continues `tests/test_axera_conv_binary_cluster_calibration_dependence.py`
(PR #1598)'s own single-seed finding for Conv's 28-byte binary "path"
switch (`Conv(k=3, dilation=3, pad=3, cin=4, cout=4, insz=16)`,
`tests/test_axera_conv_reg60_mechanism.py` PR #1586): that file tested
exactly ONE alternative calibration seed (`RandomState(42)`) against the
original's `RandomState(0)` (which produces a non-deterministic 5-vs-3
split between two states, `reg=60` = `0x7e`/`0x7f`) and found `RandomState(42)`
is instead FULLY deterministic at a third state (`0x73`, 8/8 samples).
That single data point could not distinguish "near-ties are rare" from
"near-ties are common, seed=42 just wasn't one" -- this file surveys
four MORE calibration seeds to find out.

## Method

Four new calibration seeds -- `RandomState(1)`, `RandomState(7)`,
`RandomState(100)`, `RandomState(999)` -- at the identical
`Conv(k=3, dilation=3, pad=3, cin=4, cout=4, insz=16)` shape and model
weights (`RandomState(0)`, unchanged, isolating calibration-only
dependence the same way PR #1598 did). **3 independent rebuilds per
seed** (not 8 -- sufficient to detect a split at all, and to build real
confidence in a "fully stable" reading, per this project's own
established "don't trust fewer than ~3 independent rebuilds" floor;
none of the four seeds showed ANY internal disagreement in 3/3 samples,
so none needed the extra rebuilds PR #1586's own protocol calls for
when a split IS detected). Built via a standalone script that
replicates `tests/test_axera_mcode_structure.py`'s own
`_dilation_conv_model`/`_build_and_get_mcode_bytes` helpers directly
(this worktree lacks a compiled `onnxsim_cpp2py_export` extension by
default; a pre-built `.abi3.so` + `version.py` were copied in from a
sibling checkout to make `scripts/axera/pulsar2_docker.py` importable,
the same workaround several sibling PRs this session used -- neither
file is part of this PR's own diff, confirmed via `git status`), using
the `pulsar2:7.0-lite` image (the only one loaded in this worktree's
Docker daemon; PR #1579/#1597/#1598 already established this is a
valid substitute for `6.0-lite` at this exact shape family).

## Result: near-ties are rare -- 5 of 6 seeds tested are fully
## deterministic, only the ORIGINAL seed=0 shows a real split

| seed | rebuilds | `reg=60` | deterministic? |
| --- | --- | --- | --- |
| 0 (original, PR #1586) | 8 | `0x7e` (5), `0x7f` (3) | **NO -- the only split found** |
| 1 (this file) | 3 | `0x71` | yes, 3/3 |
| 7 (this file) | 3 | `0x7c` | yes, 3/3 |
| 42 (PR #1598) | 8 | `0x73` | yes, 8/8 |
| 100 (this file) | 3 | `0x74` | yes, 3/3 |
| 999 (this file) | 3 | `0x85` | yes, 3/3 |

**Every one of the 5 non-original seeds is perfectly stable across all
of its own independent rebuilds, and every seed (including the
original's own two split states) resolves to a DIFFERENT `reg=60`
value -- 7 distinct values across 7 states, zero collisions.** This is
strong, direct evidence that `RandomState(0)`'s own calibration data is
genuinely unusual (sitting close enough to a numerical tie for
compiler-internal thread-scheduling non-determinism to flip the
outcome) rather than typical -- if near-ties were common, several of
these 5 new seeds would plausibly have shown their own splits too, and
none did.

## The 28-offset cluster generalizes cleanly to a 7-way split

Pooling all 7 states (`RandomState(0)`'s own two split groups, PR
#1598's `RandomState(42)`, and this file's 4 new seeds) and re-running
PR #1586's exact brute-force scan (a position counts only if constant
*within* each of the 7 groups but not identical across all 7) finds
**the identical 28 offsets already established, zero more, zero
fewer.** The cluster itself -- which records move together -- is
completely stable across 7 independently-calibrated states; only the
resolved values differ. `reg=54`/`reg=232`'s own already-known
leading-byte-constant, trailing-byte-varies structure holds for every
new seed too, and the two computed float32 quantities (`verb=161,
bank=15` and `reg=224`) take a genuinely distinct value at every new
seed -- none of the 4 new seeds' float pairs match each other, PR
#1586's, or PR #1598's own values, consistent with each seed driving a
real, different computed per-channel statistic (not a small fixed
menu):

| seed | `bank15` float | `reg224` float |
| --- | --- | --- |
| 0, group A | `33.811733` | `0.0145932` |
| 0, group B | `127.514183` | `0.0097953` |
| 42 | `35.577232` | `0.0167611` |
| 1 | `35.855095` | `0.0169835` |
| 7 | `35.040691` | `0.0151205` |
| 100 | `36.078632` | `0.0158318` |
| 999 | `37.855480` | `0.0183505` |

(One coincidental collision, noted for honesty rather than hidden:
`reg=232`'s own trailing byte happens to be `0x7c` for BOTH seed=0's
group A and seed=999 -- despite those two states having completely
different `reg=60` values, `0x7e` vs `0x85`. `reg=60`/`reg=54` (which
mirrors it exactly) never collide across any of the 7 states; `reg=232`
is a separate, independently-computed quantity that merely happens to
round to the same byte for two different underlying values here -- not
evidence the two states are otherwise related.)

## A secondary, unrelated observation: the `reg=8` pool group's own
## byte range shifts more than PR #1598 saw

PR #1598 found seed=42's internal diffs (across its own 8 samples)
confined to offsets 855-897 plus one downstream reflow byte at 3232.
This file's four new seeds each show a WIDER internal-diff range
(as low as offset 853, as high as 882, still always including 3232) --
still entirely inside the already-known `reg=8` pool-group region
(nowhere near the 2161-2584 binary-cluster range this file's own
finding is about) and consistent with that group's own several
already-decoded degrees of freedom (PR #1580's variable slot count,
register-label mapping, and payload-length forms) reflowing slightly
differently at different seeds -- not a new mechanism, and not chased
further here.

## What this establishes (and does not)

This firms up PR #1598's own "not purely data-independent" finding
into "near-ties are the exception, not the rule": across 6 calibration
seeds tested at this exact shape (0, 1, 7, 42, 100, 999), only the
original `RandomState(0)` produces genuine rebuild-to-rebuild
non-determinism. This does NOT prove `RandomState(0)` is the ONLY seed
that would ever show a split (a systematic search across many more
seeds could still find another one), nor does it explain WHY that
particular seed's calibration data sits near a tie -- both remain
open. What's now established with real breadth (not just one
data point) is that determinism is the norm and `RandomState(0)`'s own
behavior is the outlier requiring explanation, not the default this
whole mcode codec exhibits.
"""

import gzip
import os
import struct
import sys
import unittest

_AXERA_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts", "axera")
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import mcode  # noqa: E402

FIX = os.path.join(_AXERA_DIR, "fixtures")

ORIG_GROUP_A = [
    "conv_dilation3.mcode.gz",
    "conv_dilation3_v7stability_r0.mcode.gz",
    "conv_dilation3_v7stability_r1.mcode.gz",
    "conv_dilation3_v7stability_r2.mcode.gz",
    "conv_dilation3_v7stability_r3.mcode.gz",
]
ORIG_GROUP_B = [
    "conv_dilation3_rebuild0.mcode.gz",
    "conv_dilation3_rebuild1.mcode.gz",
    "conv_dilation3_rebuild2.mcode.gz",
]
SEED42_NAMES = [f"conv_dilation3_calibseed42_r{i}.mcode.gz" for i in range(8)]

NEW_SEEDS = (1, 7, 100, 999)
NEW_SEED_NAMES = {
    seed: [f"conv_dilation3_calibseed{seed}_r{i}.mcode.gz" for i in range(3)]
    for seed in NEW_SEEDS
}
ALL_NEW_NAMES = [n for names in NEW_SEED_NAMES.values() for n in names]

EXPECTED_REG60 = {1: 0x71, 7: 0x7C, 100: 0x74, 999: 0x85}

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


class TestAllTwelveNewSamplesDecodeCleanly(unittest.TestCase):
    def test_3528_bytes_zero_check_errors(self):
        for n in ALL_NEW_NAMES:
            data = load(n)
            self.assertEqual(len(data), 3528, n)
            self.assertEqual(mcode.check(data), [], n)


class TestEachNewSeedIsFullyDeterministic(unittest.TestCase):
    """The core finding: all 4 new seeds are perfectly stable across
    their own 3 independent rebuilds -- unlike seed=0's own 5-vs-3
    split, none of these show any internal disagreement."""

    def test_each_seed_matches_expected_value_in_every_rebuild(self):
        for seed, names in NEW_SEED_NAMES.items():
            for n in names:
                self.assertEqual(reg60_value(decode(n)), EXPECTED_REG60[seed], n)

    def test_each_seed_is_internally_unanimous(self):
        for seed, names in NEW_SEED_NAMES.items():
            vals = {reg60_value(decode(n)) for n in names}
            self.assertEqual(len(vals), 1, (seed, vals))


class TestSevenStatesAreAllDistinct(unittest.TestCase):
    """Every one of the 7 states found across all seeds tested so far
    (seed=0's two split states, seed=42, and this file's 4 new seeds)
    has a UNIQUE reg=60 value -- zero collisions."""

    def test_no_two_states_share_a_reg60_value(self):
        values = {
            "orig_A": reg60_value(decode(ORIG_GROUP_A[0])),
            "orig_B": reg60_value(decode(ORIG_GROUP_B[0])),
            "seed42": reg60_value(decode(SEED42_NAMES[0])),
        }
        for seed in NEW_SEEDS:
            values[f"seed{seed}"] = EXPECTED_REG60[seed]
        self.assertEqual(len(set(values.values())), len(values), values)


class TestSevenWayClusterDiscoveryMatchesKnownOffsets(unittest.TestCase):
    """Recomputes the correlated-offset set from scratch (not trusted
    from the module docstring's own copy) against a 7-way grouping: a
    position counts only if it's constant *within* each of the 7
    groups but not identical across all 7. Confirms the exact same 28
    offsets PR #1586 originally found for the 2-way split."""

    def test_exactly_28_offsets_matching_the_known_set(self):
        groups = {
            "orig_A": ORIG_GROUP_A,
            "orig_B": ORIG_GROUP_B,
            "seed42": SEED42_NAMES,
        }
        for seed in NEW_SEEDS:
            groups[f"seed{seed}"] = NEW_SEED_NAMES[seed]

        datas = {n: load(n) for names in groups.values() for n in names}
        length = len(next(iter(datas.values())))
        for n, d in datas.items():
            self.assertEqual(len(d), length, n)

        found = []
        for i in range(length):
            per_group_vals = {}
            consistent = True
            for label, names in groups.items():
                vals = {datas[n][i] for n in names}
                if len(vals) != 1:
                    consistent = False
                    break
                per_group_vals[label] = next(iter(vals))
            if consistent and len(set(per_group_vals.values())) > 1:
                found.append(i)

        self.assertEqual(frozenset(found), KNOWN_CORRELATED_OFFSETS)
        self.assertEqual(len(found), 28)


class TestNewSeedsFloatValuesAreAllDistinct(unittest.TestCase):
    """The two computed float32 quantities (bank=15 verb=161 operand,
    reg=224 payload) take a genuinely distinct value at each of the 4
    new seeds -- none collide with each other or with seed=0/seed=42's
    own values, consistent with each seed driving a real, different
    per-channel computed statistic rather than a small fixed menu."""

    def _bank15_float(self, recs):
        hits = [
            r
            for r in recs
            if r["kind"] == "V"
            and r.get("verb") == 161
            and r.get("bank") == 15
            and r.get("field") in (96, 112, 128)
            and r.get("operand") not in (None, b"$\x83N")
        ]
        self.assertEqual(len(hits), 3, hits)
        ops = {h["operand"] for h in hits}
        self.assertEqual(len(ops), 1, ops)
        op = next(iter(ops))
        return struct.unpack("<f", op)[0]

    def _reg224_float(self, recs):
        hits = [
            r
            for r in recs
            if r.get("reg") == 224 and r.get("tag") == 129 and r.get("payload")
        ]
        self.assertEqual(len(hits), 4, hits)
        payloads = {h["payload"][-4:] for h in hits}
        self.assertEqual(len(payloads), 1, payloads)
        p = next(iter(payloads))
        return struct.unpack("<f", p)[0]

    def test_bank15_floats_are_pairwise_distinct(self):
        vals = []
        for seed in NEW_SEEDS:
            recs = decode(NEW_SEED_NAMES[seed][0])
            vals.append(self._bank15_float(recs))
        # Also include seed=0's two states and seed=42's own value.
        vals.append(self._bank15_float(decode(ORIG_GROUP_A[0])))
        vals.append(self._bank15_float(decode(ORIG_GROUP_B[0])))
        vals.append(self._bank15_float(decode(SEED42_NAMES[0])))
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                self.assertNotAlmostEqual(vals[i], vals[j], places=3, msg=(i, j, vals))

    def test_reg224_floats_are_pairwise_distinct(self):
        vals = []
        for seed in NEW_SEEDS:
            recs = decode(NEW_SEED_NAMES[seed][0])
            vals.append(self._reg224_float(recs))
        vals.append(self._reg224_float(decode(ORIG_GROUP_A[0])))
        vals.append(self._reg224_float(decode(ORIG_GROUP_B[0])))
        vals.append(self._reg224_float(decode(SEED42_NAMES[0])))
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                self.assertNotAlmostEqual(vals[i], vals[j], places=4, msg=(i, j, vals))


class TestReg232CanCoincidentallyCollideAcrossDifferentStates(unittest.TestCase):
    """Honesty check: reg=232's own trailing byte happens to be 0x7c
    for BOTH seed=0's group A and seed=999, despite those two states
    having completely different reg=60 values (0x7e vs 0x85) -- a
    coincidental collision on a separate, independently-computed
    quantity, not evidence the two states are otherwise related.
    reg=60/reg=54 themselves never collide (TestSevenStatesAreAllDistinct)."""

    def test_reg232_collision_is_real_but_reg60_still_differs(self):
        recs_a = decode(ORIG_GROUP_A[0])
        recs_999 = decode(NEW_SEED_NAMES[999][0])
        reg232_a = [r for r in recs_a if r.get("reg") == 232 and r.get("tag") == 132][
            0
        ]["payload"][-1]
        reg232_999 = [
            r for r in recs_999 if r.get("reg") == 232 and r.get("tag") == 132
        ][0]["payload"][-1]
        self.assertEqual(reg232_a, reg232_999)
        self.assertNotEqual(reg60_value(recs_a), reg60_value(recs_999))


if __name__ == "__main__":
    unittest.main()
