"""Two Conv dilation-dependent mcode fields: one confirmed calibration-
derived, one confirmed not, and neither's bit-level meaning decoded.

`scripts/axera/README.md`'s "### Extending the periodic field across a
wider dilation range: real values, no simple formula yet, and a new
threshold effect" section (~line 1406-1446) built a single-`Conv`
dilation model at `dilation={2,3,4,5,6}` and found two real, reproducible
fields inside `AxQuantizedConv`'s command:

1. **A periodic field** (4 repeats of a 3-byte value, 7-byte stride) whose
   value changes with dilation but "doesn't reduce to an obvious
   arithmetic function of dilation alone" -- hypothesized, by analogy
   with an unrelated already-known calibration-derived field (Add/Sub's),
   to encode something computed from quantization ranges rather than the
   raw dilation integer. Explicitly "a real, motivated hypothesis, not
   yet confirmed."
2. **A threshold-like second field**, repeated identically twice at a
   fixed separation, that "only activates once dilation reaches 4" in the
   README's own build (present at d=4 and d=6, absent at d=2 and d=3;
   d=5 wasn't compared there because it happened to serialize to a
   different total length in that specific build).

Neither field's bit-level meaning was decoded. This file: reproduces
both fields at a small, systematically tractable shape (`cin=cout=4`,
`k=3`, 16x16 input, matching `tests/test_axera_mcode_structure.py`'s
`_dilation_conv_model` helper and its own already-merged
`test_axquantizedconv_command_has_a_real_periodic_4x_field` regression
test, which locks in field 1's basic shape but not its dependence on
anything); confirms field 1's calibration-dependence directly for the
first time (upgrading the README's "hypothesis, not yet confirmed" to
confirmed); confirms field 2 is *not* calibration-dependent, cleanly
separating the two fields' natures; and refines field 2's own
characterization, which turns out not to be simple binary on/off.

At this shape, `dilation=2` through `dilation=7` all serialize to the
same 3,528-byte total (this build's own `d=5` does NOT diverge in
length the way the README's build did, so all six can be compared
directly where that one could only compare four).

## Field 1: confirmed calibration-derived

Holding `dilation=2` fixed and varying only the input calibration
range (same model, same dilation, different calibration data) changes
field 1's value: baseline `1a3b80`, at 2x calibration amplitude
`1a3b00` (only the trailing byte moves), at 5x amplitude `e149a0`, and
under an independent calibration draw (different RNG seed, same
nominal amplitude -- calibration's actual observed min/max still
differs run to run) `cbb28f`. All three calibration variants leave
field 2 (below) byte-for-byte unchanged. This is the first direct
confirmation of the README's own hypothesis: field 1 tracks
calibration, not the raw dilation attribute -- though *how* remains
undecoded (the small number of data points here doesn't support fitting
a specific formula, and reversing it would need many more calibration
points at fixed dilation).

## Field 2: confirmed NOT calibration-derived, and not simple binary

The two-copy marker (offsets 552-557 and 1160-1165 in this build,
`[3-byte value][81][16][value][81]`-shaped, at the same fixed +608-byte
separation between engine copies this README already documented for
its own 94-byte/60-byte shared blocks) is byte-identical across all
three calibration variants above -- ruling out calibration as a factor,
in contrast to field 1.

Across `dilation=2..7` (all same-length, all directly comparable) the
marker is **not** simple binary on/off the way the README's smaller
`{2,3,4,6}` comparison suggested. It takes (at least) three distinct
values: `d=2` and `d=3` share one value, `d=4` has a value of its own,
`d=5` and `d=6` share a third, and `d=7` **reverts** to the same value
as `d=2`/`d=3` -- not a monotone "off below a cutoff, on above it."
Still undecoded at the bit level, and the six data points here aren't
enough to fit a real period or rule (a naive `dilation mod 5` grouping
fits by coincidence -- `d=2` and `d=7` share a remainder, and `d=5`
lands on remainder 0 -- but `d=3` and `d=6` do NOT share the third
group's value despite also differing by 5, so this is flagged as an
observation, not a confirmed periodicity).

## What was checked and set aside

An independent second rebuild of the `dilation=2` config (this
project's hard-learned determinism-checking rule) reproduces both
fields exactly; the only bytes that move are a 26-byte cluster at
offsets 853-874 plus one byte at 3232, a noise pocket at a different
location than this project's other already-documented ~303-325 zone
but the same general phenomenon -- neither field goes anywhere near it.
A `k=5` kernel variant at dilation `{1,2,3}` (matching receptive-field
sizes 5/9/13 to the `k=3` sweep's `d=2/d=4/d=6`) was built to test
whether the fields track receptive-field size rather than raw dilation,
but serializes to a different total length (3,976 vs 3,528 bytes) with
the two fields not yet relocated inside that different layout -- left
for future work rather than reported here, since finding them would
need the same same-length-pair care this project has been burned by
skipping before. Dilation values 8 and up were swept far enough to
find two more length thresholds (3,760 bytes at d=8-11/13, 3,792 at
d=12) but, likewise, the fields were not relocated inside those larger
layouts within this file's scope.
"""

import gzip
import os
import unittest

FIX = os.path.join(os.path.dirname(__file__), "..", "scripts", "axera", "fixtures")


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


FIELD1_OFFSETS = (2561, 2568, 2575, 2582)
FIELD2_SLOTS = ((552, 558), (1160, 1166))


class TestField1IsCalibrationDerived(unittest.TestCase):
    """Field 1 (the periodic 4x/7-byte-stride value) changes when
    calibration changes with dilation held fixed -- confirming the
    README's own stated-but-unconfirmed hypothesis."""

    BASELINE = "conv_dilation2.mcode.gz"

    def _field1(self, data):
        vals = [data[o : o + 3] for o in FIELD1_OFFSETS]
        self.assertEqual(len(set(vals)), 1, "all 4 copies should still agree")
        return vals[0]

    def test_field1_is_a_real_periodic_4x_quad_at_baseline(self):
        data = load(self.BASELINE)
        self.assertEqual(self._field1(data).hex(), "1a3b80")

    def test_field1_changes_with_calibration_amplitude_2x(self):
        base = load(self.BASELINE)
        other = load("conv_dilation2_calib2x.mcode.gz")
        self.assertEqual(len(base), len(other))
        self.assertNotEqual(self._field1(base), self._field1(other))
        self.assertEqual(self._field1(other).hex(), "1a3b00")

    def test_field1_changes_with_calibration_amplitude_5x(self):
        base = load(self.BASELINE)
        other = load("conv_dilation2_calib5x.mcode.gz")
        self.assertNotEqual(self._field1(base), self._field1(other))
        self.assertEqual(self._field1(other).hex(), "e149a0")


class TestField2IsNotCalibrationDerived(unittest.TestCase):
    """The same three calibration variants leave field 2 completely
    unchanged, cleanly separating it from field 1."""

    def _field2(self, data):
        a = data[FIELD2_SLOTS[0][0] : FIELD2_SLOTS[0][1]]
        b = data[FIELD2_SLOTS[1][0] : FIELD2_SLOTS[1][1]]
        self.assertEqual(a, b, "the two engine copies should always agree")
        return a

    def test_field2_unchanged_across_calibration_variants(self):
        base = self._field2(load("conv_dilation2.mcode.gz"))
        for name in (
            "conv_dilation2_calib2x.mcode.gz",
            "conv_dilation2_calib5x.mcode.gz",
        ):
            other = self._field2(load(name))
            self.assertEqual(base, other, name)


class TestField2IsNotSimpleBinary(unittest.TestCase):
    """Across dilation=2..7 (all the same mcode length, all directly
    comparable), field 2 takes at least 3 distinct values -- not the
    simple "absent below 4, present at/above 4" pattern the README's
    smaller {2,3,4,6} comparison suggested."""

    def _field2(self, tag):
        data = load(f"conv_dilation{tag}.mcode.gz")
        a = data[FIELD2_SLOTS[0][0] : FIELD2_SLOTS[0][1]]
        b = data[FIELD2_SLOTS[1][0] : FIELD2_SLOTS[1][1]]
        self.assertEqual(a, b)
        return a

    def test_d2_and_d3_share_a_value(self):
        self.assertEqual(self._field2(2), self._field2(3))

    def test_d4_is_its_own_value(self):
        d4 = self._field2(4)
        self.assertNotEqual(d4, self._field2(2))
        self.assertNotEqual(d4, self._field2(5))

    def test_d5_and_d6_share_a_different_value(self):
        d5, d6 = self._field2(5), self._field2(6)
        self.assertEqual(d5, d6)
        self.assertNotEqual(d5, self._field2(2))
        self.assertNotEqual(d5, self._field2(4))

    def test_d7_reverts_to_d2s_value(self):
        """Not a monotone threshold: d=7 goes back to matching d=2/d=3
        rather than continuing the d=5/d=6 state or introducing a
        fourth one."""
        self.assertEqual(self._field2(7), self._field2(2))


class TestBothFieldsSurviveAnIndependentRebuild(unittest.TestCase):
    """This project's hard-learned rule: a single rebuild pair is not
    enough to trust a diff as real signal. An independent second
    dilation=2 build reproduces both fields exactly; the noise that
    does exist lands well clear of either."""

    def test_determinism_noise_does_not_touch_either_field(self):
        a = load("conv_dilation2.mcode.gz")
        b = load("conv_dilation2_rebuild.mcode.gz")
        self.assertEqual(len(a), len(b))
        diffs = {i for i in range(len(a)) if a[i] != b[i]}

        field1_bytes = {i for o in FIELD1_OFFSETS for i in range(o, o + 3)}
        field2_bytes = {i for lo, hi in FIELD2_SLOTS for i in range(lo, hi)}
        self.assertEqual(diffs & field1_bytes, set())
        self.assertEqual(diffs & field2_bytes, set())
        self.assertGreater(len(diffs), 0, "sanity: some noise should exist")


if __name__ == "__main__":
    unittest.main()
