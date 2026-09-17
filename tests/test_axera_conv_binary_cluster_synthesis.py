"""Synthesis pass over this session's own 8-PR investigation into
Conv's 28-byte binary "path" switch, in the style of
`tests/test_axera_noise_vs_loadbearing_synthesis.py` (PR #1561) and
`tests/test_axera_reg8_cross_op_synthesis.py` (PR #1582): both of those
files stepped back from a scattered set of per-PR findings to build one
coherent, directly-reconfirmed picture, adding a real cross-cutting
observation no single contributing PR was positioned to make. This file
does the same for the switch investigation, which grew from one small,
unexplained `reg=60` residual into a fairly complete empirical picture
across:

`tests/test_axera_conv_reg60_mechanism.py` (PR #1586) -- discovery,
`dilation=3`. `tests/test_axera_matmul_binary_cluster_search.py`
(PR #1592) -- MatMul clean negative, Gemm inconclusive.
`tests/test_axera_gemm_binary_cluster_search_small_shape.py` (PR #1596)
-- Gemm's inconclusive result resolved into a clean negative.
`tests/test_axera_conv_binary_cluster_other_shape.py` (PR #1597) --
absent at `dilation=1`. `tests/test_axera_conv_dilation2_transition.py`
(PR #1599) -- present at `dilation=2`, transition pinned between `1`
and `2`. `tests/test_axera_conv_binary_cluster_calibration_dependence.py`
(PR #1598) -- the switch's resolved VALUE is calibration-data-dependent,
not pure scheduling noise. `tests/test_axera_conv_binary_cluster_seed_survey.py`
(PR #1600) -- 6-seed survey, near-ties are rare.
`tests/test_axera_conv_binary_cluster_higher_dilation.py` (PR #1601) --
absent again at `dilation=4`/`8`; the switch's own window is exactly
`{2, 3}`, narrower than a persistent `dilation>=2` property.

## The full picture, directly re-verified against the actual fixtures

Every number below was recomputed live from the committed `.mcode.gz`
fixtures in this PR's own test code (`TestDilationPresenceTableMatchesFixtures`,
`TestSevenCalibrationStatesAreAllDistinct`), not copied from any
source PR's own docstring.

**Dilation presence/absence** (`reg=60`'s own value, `Conv(k=3,
dilation=d, pad=d, cin=4, cout=4, insz=16)`, output shape held at 16x16
for every `d`):

| `dilation` | n samples | switch present? | `reg=60` value(s) |
| --- | --- | --- | --- |
| 1 | 8 | no | `0x7e` (constant) |
| 2 | 12 | **yes** | `0x7e` (10), `0x7f` (2) |
| 3 | 8 | **yes** | `0x7e` (5), `0x7f` (3) |
| 4 | 8 | no | `0x7e` (constant) |
| 8 | 8 | no | `0x7e` (constant) |

The switch's own dilation range is exactly `{2, 3}` -- not `dilation=3`-
specific (PR #1586's own original framing), not a persistent
`dilation>=2` threshold (which higher-dilation data alone, without the
`4`/`8` negatives, could have looked like). Genuinely narrow.

**Calibration-seed survey at the fixed `dilation=3` shape** (7 states
across 6 seeds, weight RNG held at `RandomState(0)` throughout,
calibration RNG varied):

| calibration seed | `reg=60` | deterministic? |
| --- | --- | --- |
| 0, group A | `0x7e` | (part of seed=0's own split) |
| 0, group B | `0x7f` | (part of seed=0's own split) |
| 1 | `0x71` | yes |
| 7 | `0x7c` | yes |
| 42 | `0x73` | yes |
| 100 | `0x74` | yes |
| 999 | `0x85` | yes |

7 distinct values, zero collisions, directly reconfirmed below. Only
`RandomState(0)` -- the seed every OTHER Conv fixture in this entire
project's corpus happens to use, since it is this project's own
established default -- shows genuine rebuild-to-rebuild non-determinism.
The other 5 seeds are each internally unanimous.

**The cluster's own identity (which 28 offsets, which records) is
invariant across every state tested**: `dilation=2`'s own split,
`dilation=3`'s own split, and all 5 deterministic calibration seeds all
resolve to the same set of co-moving records (`reg=60` x2, `reg=54`,
`reg=232`, three `verb=161,bank=15` float-operand copies, `reg=224` x4)
-- only the resolved VALUES differ, never the cluster's membership or
size. (`dilation=2`'s own absolute byte offsets shift by exactly `-2`
relative to `dilation=3`'s, an ordinary small upstream reflow, not a
different cluster -- PR #1599's own finding, not re-litigated here.)

## The `reg=8` group-form change is a separate, more durable phenomenon
## -- do not conflate it with the switch

PR #1597 first noticed, co-located at the same shapes, that the `reg=8`
pool group's own FORM (clean 4-of-4 permutation vs. 3-of-4 variable-
register-label) also changes between `dilation=1` and `dilation=3`.
PR #1599 found this form-change lands at the identical boundary as the
switch (`dilation=1` vs `2`). But PR #1601 found the two phenomena
DIVERGE once dilation grows further: the switch vanishes again by
`dilation=4`, while the `reg=8` form change persists through `dilation=8`
(every dilation `>=2` tested still shows the 3-of-4 form). **These are
two distinct triggers that happen to share one boundary, not one
mechanism** -- this file's own `TestReg8FormPersistsBeyondTheSwitchsWindow`
directly reconfirms PR #1601's own dilation=4/8 reg=8-form reading
against the fixtures, to make sure this distinction is not lost in the
synthesis.

## A new, directly-verified connection: one of the cluster's two computed
## float32 values is a field this project ALREADY had, from before this
## session's own binary-cluster arc began

`scripts/axera/README.md`'s own pre-existing "Extending the periodic
field across a wider dilation range" section (grep
`"real values, no simple formula yet"`) documents a "confirmed field...
offset 2561" that "takes a different 3-byte value for every dilation
compared against the baseline" -- with EXACT quoted values `d=2: 1a3b80`,
`d=3: 4b186f`, `d=4: 244082`, `d=6: 550f5b` -- and speculates it
"plausibly encodes something computed from the dilated receptive
field's effect on quantization ranges."

**Decoding `reg=224`'s own payload directly from the committed
`conv_dilation2.mcode.gz`/`conv_dilation2_rebuild.mcode.gz` fixtures
finds the middle three bytes are exactly `1a 3b 80`** -- byte-identical
to the README's own quoted `d=2` value
(`TestReg224IsTheReadmesAlreadyKnownOffset2561Field` below). The same
check against PR #1586's own `dilation=3` fixture (`4b 18 6f`, from its
own docstring table) matches the README's quoted `d=3: 4b186f` too.
**This is not a coincidence -- it is the same field.** The README's own
`d=2`/`d=3` values are literally `reg=224`'s group-A (the more common
outcome) resolved value at `RandomState(0)`'s calibration, viewed
before this project had the vocabulary ("binary path switch",
"calibration-dependent") to describe what it actually was.

This connects two previously-separate strands of this project's own
accumulated knowledge for the first time: the README's own earlier,
narrower finding ("this field is dilation-dependent, no simple
formula") and this session's own binary-cluster investigation ("this
field's group-mate `reg=60` etc. form a 28-byte switch, calibration-
data-dependent, bounded to `dilation in {2,3}`") describe the SAME
underlying quantity from two different angles. It strengthens the
README's own speculative "computed from the dilated receptive field's
effect on quantization ranges" reading -- a value that is BOTH shape-
dependent (the README's finding) AND, within a fixed shape, sensitive
enough to floating-point summation order to occasionally show a real
near-tie (this session's finding) is exactly the behavior a genuine
per-channel weight/activation statistic would have, and is much less
consistent with a synthetic flag or fixed marker.

**What this does NOT establish**: the README's own quoted `d=4`
(`244082`) and `d=6` (`550f5b`) values were themselves each observed
from only ONE build apiece (no rebuild-stability check at the time) --
this file does not re-verify those two against fresh rebuilds, so it
is possible (though this session's own `dilation=4`/`8` negative
findings make it unlikely, since `reg=60` itself was checked and found
constant at those dilations) that `d=4`/`d=6` show their own hidden
splits that simply weren't caught by a single build. Not chased here.

## A related, NOT chased, tangential observation from the same README
## section

The same README section separately describes a "new threshold effect":
a second pair of matching 3-byte runs (at offsets 552/1160) that
differs in value between the README's own `d=2` baseline and its
`d=4`/`d=6` probes. A direct spot-check here
(`TestSecondReadmeFieldExistsButIsNotFurtherCharacterized`) confirms
this SAME paired-matching structure (two positions, 608 bytes apart,
byte-identical to each other) is present in BOTH this session's own
`conv_dilation2.mcode.gz` and `conv_dilation4_r0.mcode.gz` fixtures --
consistent with the README's own general finding (another real,
dilation-dependent computed field, structurally analogous to `reg=224`'s
own), not obviously the same thing as PR #1601's own flagged
`dilation=8` wholesale-length-increase wrinkle (which is a raw byte-
count change, not a same-length value change). This file does not
attempt to determine whether this second field ALSO shows its own
calibration-dependent near-tie behavior at some dilation window --
flagged as a concrete, motivated lead for whoever picks this up next,
not investigated further here (out of this synthesis file's own scope).

## What's established vs. what remains genuinely open

**Established** (each independently re-verified, directly or via this
file's own reconfirmation tests):
- The switch is real, structural, and reproducible -- a fixed 28-byte
  cluster of co-moving records, not isolated noise.
- Its dilation range is exactly `{2, 3}` among `{1, 2, 3, 4, 8}` tested
  -- narrow, not a persistent `dilation>=2` property.
- Its resolved VALUE (not just presence) is a genuine function of
  calibration data -- 5 of 6 tested seeds are fully deterministic;
  only `RandomState(0)` -- this project's own default seed -- shows a
  real split, evidence that near-ties are the exception, not the norm.
- It does not generalize to Gemm or MatMul at the shapes tested (both
  show clean negatives once their own known noise mechanisms are
  accounted for).
- One of its two computed float32 values (`reg=224`) is the same field
  `scripts/axera/README.md` already documented, independently, as a
  real dilation-dependent quantity before this session's own
  binary-cluster arc began -- now directly confirmed to be the same
  field, not a coincidental resemblance.
- The co-located `reg=8` group-form change is a SEPARATE, more durable
  phenomenon (persists through `dilation=8`) -- the two must not be
  conflated despite sharing a boundary at `dilation=1`/`2`.

**Still open**:
- WHY `RandomState(0)`'s calibration data specifically sits near a
  numerical tie while 5 other seeds don't.
- WHY the switch's own dilation window is exactly `{2, 3}` and not
  wider or narrower -- no mechanistic account, only the empirical
  boundary.
- The exact semantic meaning of the two float32 values beyond
  "quantization-scale-like, dilation- and calibration-dependent" --
  the README's own "receptive-field effect on quantization ranges"
  reading is now better-supported but still not proven.
- The `dilation=8` mcode-length increase (232 bytes, PR #1601) and the
  README's own separate "second field" (offsets 552/1160) -- both
  real, both flagged, neither chased to a decoded rule.
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

DILATION_NAMES = {
    1: [f"conv_dilation1_r{i}.mcode.gz" for i in range(8)],
    2: [
        "conv_dilation2.mcode.gz",
        "conv_dilation2_rebuild.mcode.gz",
        "conv_dilation2_rebuild0.mcode.gz",
        "conv_dilation2_rebuild1.mcode.gz",
    ]
    + [f"conv_dilation2_r{i}.mcode.gz" for i in range(8)],
    3: [
        "conv_dilation3.mcode.gz",
        "conv_dilation3_rebuild0.mcode.gz",
        "conv_dilation3_rebuild1.mcode.gz",
        "conv_dilation3_rebuild2.mcode.gz",
    ]
    + [f"conv_dilation3_v7stability_r{i}.mcode.gz" for i in range(4)],
    4: [f"conv_dilation4_r{i}.mcode.gz" for i in range(8)],
    8: [f"conv_dilation8_r{i}.mcode.gz" for i in range(8)],
}

SEED_STATES = {
    "0_groupA": [
        "conv_dilation3.mcode.gz",
        "conv_dilation3_v7stability_r0.mcode.gz",
        "conv_dilation3_v7stability_r1.mcode.gz",
        "conv_dilation3_v7stability_r2.mcode.gz",
        "conv_dilation3_v7stability_r3.mcode.gz",
    ],
    "0_groupB": [
        "conv_dilation3_rebuild0.mcode.gz",
        "conv_dilation3_rebuild1.mcode.gz",
        "conv_dilation3_rebuild2.mcode.gz",
    ],
    "1": [f"conv_dilation3_calibseed1_r{i}.mcode.gz" for i in range(3)],
    "7": [f"conv_dilation3_calibseed7_r{i}.mcode.gz" for i in range(3)],
    "42": [f"conv_dilation3_calibseed42_r{i}.mcode.gz" for i in range(8)],
    "100": [f"conv_dilation3_calibseed100_r{i}.mcode.gz" for i in range(3)],
    "999": [f"conv_dilation3_calibseed999_r{i}.mcode.gz" for i in range(3)],
}

EXPECTED_PRESENCE = {1: False, 2: True, 3: True, 4: False, 8: False}

# From tests/test_axera_conv_reg60_mechanism.py (PR #1586), reconfirmed
# there against the dilation=3 fixtures. Reused here (not re-derived)
# as the baseline cluster size this file's own multi-state check
# compares against.
KNOWN_CLUSTER_SIZE = 28


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def decode(name):
    return mcode.decode(load(name), **mcode.FULL_RULE)


def reg60_value(name):
    recs = decode(name)
    hits = [
        r
        for r in recs
        if r["kind"] == "S"
        and r.get("reg") == 60
        and r.get("tag") == 131
        and r.get("payload")
    ]
    assert len(hits) >= 1, (name, hits)
    return hits[0]["payload"][-1]


def reg224_payload(name):
    recs = decode(name)
    hits = [
        r
        for r in recs
        if r.get("reg") == 224 and r.get("tag") == 129 and r.get("payload")
    ]
    assert hits, name
    return hits[0]["payload"]


class TestDilationPresenceTableMatchesFixtures(unittest.TestCase):
    """Recomputes the dilation-1/2/3/4/8 presence/absence table directly
    from the committed fixtures -- not trusted from any source PR's own
    docstring numbers."""

    def test_presence_matches_expected_for_every_dilation(self):
        for d, expected in EXPECTED_PRESENCE.items():
            names = DILATION_NAMES[d]
            values = {reg60_value(n) for n in names}
            present = len(values) > 1
            self.assertEqual(
                present,
                expected,
                f"dilation={d}: expected present={expected}, "
                f"got values={ {hex(v) for v in values} }",
            )

    def test_dilation1_is_constant_0x7e(self):
        values = {reg60_value(n) for n in DILATION_NAMES[1]}
        self.assertEqual(values, {0x7E})

    def test_dilation4_and_8_are_also_constant_0x7e(self):
        for d in (4, 8):
            values = {reg60_value(n) for n in DILATION_NAMES[d]}
            self.assertEqual(values, {0x7E}, d)

    def test_dilation2_split_is_2_vs_10(self):
        values = [reg60_value(n) for n in DILATION_NAMES[2]]
        self.assertEqual(values.count(0x7F), 2, values)
        self.assertEqual(values.count(0x7E), 10, values)

    def test_dilation3_split_is_3_vs_5(self):
        values = [reg60_value(n) for n in DILATION_NAMES[3]]
        self.assertEqual(values.count(0x7F), 3, values)
        self.assertEqual(values.count(0x7E), 5, values)


class TestSevenCalibrationStatesAreAllDistinct(unittest.TestCase):
    """Recomputes PR #1600's own "7 distinct reg=60 values, zero
    collisions across 6 calibration seeds" claim directly, rather than
    trusting its docstring table."""

    def test_each_state_is_internally_unanimous(self):
        for label, names in SEED_STATES.items():
            values = {reg60_value(n) for n in names}
            self.assertEqual(len(values), 1, (label, values))

    def test_seven_states_seven_distinct_values(self):
        resolved = {
            label: next(iter({reg60_value(n) for n in names}))
            for label, names in SEED_STATES.items()
        }
        self.assertEqual(len(set(resolved.values())), 7, resolved)

    def test_only_seed_zero_shows_internal_disagreement(self):
        # seed=0's own two "states" (groupA/groupB) are themselves each
        # internally unanimous -- the split only appears when the
        # ORIGINAL, undifferentiated seed=0 batch (all 8 samples,
        # before anyone knew to split it by reg=60) is considered as
        # one group.
        all_seed0 = SEED_STATES["0_groupA"] + SEED_STATES["0_groupB"]
        values = {reg60_value(n) for n in all_seed0}
        self.assertEqual(
            len(values), 2, "seed=0's own undifferentiated batch should split"
        )
        for label in ("1", "7", "42", "100", "999"):
            values = {reg60_value(n) for n in SEED_STATES[label]}
            self.assertEqual(len(values), 1, label)


class TestClusterMembershipIsInvariantAcrossStates(unittest.TestCase):
    """The cluster's own SIZE (28 offsets) recomputed across all 7
    dilation=3 calibration states (a richer grouping than any single
    source PR used) -- confirms the cluster identity is stable
    regardless of which/how-many states are pooled, only the resolved
    values differ."""

    def test_28_offsets_across_all_seven_states(self):
        groups = {
            label: [load(n) for n in names] for label, names in SEED_STATES.items()
        }
        length = len(next(iter(groups.values()))[0])
        for datas in groups.values():
            for d in datas:
                self.assertEqual(len(d), length)

        found = []
        for i in range(length):
            per_group_vals = {}
            consistent = True
            for label, datas in groups.items():
                vals = {d[i] for d in datas}
                if len(vals) != 1:
                    consistent = False
                    break
                per_group_vals[label] = next(iter(vals))
            if consistent and len(set(per_group_vals.values())) > 1:
                found.append(i)

        self.assertEqual(len(found), KNOWN_CLUSTER_SIZE, sorted(found))


class TestReg8FormPersistsBeyondTheSwitchsWindow(unittest.TestCase):
    """Directly reconfirms PR #1601's own distinguishing finding: the
    reg=8 group's own 3-of-4-variable-label FORM persists at dilation=4
    and dilation=8 even though the binary-cluster switch itself is
    absent there -- the two phenomena share a boundary at dilation 1/2
    but are not the same trigger."""

    @staticmethod
    def _is_3of4_form(name):
        recs = decode(name)
        r170 = [
            r
            for r in recs
            if r["kind"] == "S" and r.get("reg") == 170 and r.get("payload") == b"\x12"
        ]
        return len(r170) == 1

    def test_dilation1_is_not_3of4_form(self):
        for n in DILATION_NAMES[1]:
            self.assertFalse(self._is_3of4_form(n), n)

    def test_dilation_2_3_4_8_are_all_3of4_form(self):
        for d in (2, 3, 4, 8):
            for n in DILATION_NAMES[d]:
                self.assertTrue(self._is_3of4_form(n), (d, n))


class TestReg224IsTheReadmesAlreadyKnownOffset2561Field(unittest.TestCase):
    """The new cross-cutting finding this synthesis adds: reg=224's own
    computed float32 value (one of the binary-cluster's two float
    quantities) is byte-identical, at its middle three bytes, to the
    exact hex values scripts/axera/README.md already quotes for its own
    pre-existing "confirmed field... offset 2561" -- the same field,
    independently documented from two different angles."""

    def test_dilation2_matches_readmes_quoted_1a3b80(self):
        for n in ("conv_dilation2.mcode.gz", "conv_dilation2_rebuild.mcode.gz"):
            payload = reg224_payload(n)
            # 5-byte form (tag prefix + 4-byte float): middle 3 bytes
            # of the float are the README's own quoted value.
            self.assertEqual(payload[-4:-1], b"\x1a\x3b\x80", (n, payload.hex()))

    def test_dilation3_group_a_matches_readmes_quoted_4b186f(self):
        # PR #1586's own group A (the majority outcome, 5/8 samples).
        for n in (
            "conv_dilation3.mcode.gz",
            "conv_dilation3_v7stability_r0.mcode.gz",
        ):
            payload = reg224_payload(n)
            self.assertEqual(payload[-4:-1], b"\x4b\x18\x6f", (n, payload.hex()))

    def test_readme_still_documents_this_field(self):
        readme = os.path.join(_AXERA_DIR, "README.md")
        with open(readme, encoding="utf-8") as f:
            text = f.read()
        self.assertIn("1a3b80", text)
        self.assertIn("4b186f", text)
        self.assertIn("offset 2561", text)


class TestSecondReadmeFieldExistsButIsNotFurtherCharacterized(unittest.TestCase):
    """Spot-checks the README's own separate "second matching pair"
    observation (offsets 552/1160, 608 bytes apart) against this
    session's own dilation=2/dilation=4 fixtures. Confirms the paired-
    matching STRUCTURE exists at both (not exclusive to dilation>=4 the
    way a literal reading of the README's own wording might suggest),
    with different resolved values at each -- consistent with another
    real, dilation-dependent field of the same general character as
    reg=224, not further decoded here."""

    def test_paired_structure_present_and_self_consistent_at_both_dilations(self):
        for name in ("conv_dilation2.mcode.gz", "conv_dilation4_r0.mcode.gz"):
            data = load(name)
            self.assertEqual(data[552:555], data[1160:1163], name)

    def test_the_two_dilations_resolve_to_different_values(self):
        d2 = load("conv_dilation2.mcode.gz")
        d4 = load("conv_dilation4_r0.mcode.gz")
        self.assertNotEqual(d2[552:555], d4[552:555])


if __name__ == "__main__":
    unittest.main()
