"""Extending Conv dilation's insz/trigger grid past PR #1519's own
edges: confirms `dilation=4`'s below-threshold symmetry, and finds a
THIRD, previously-unseen trigger signature past `insz=16` -- the
pattern stays irregular, with no periodicity or formula, all the way
out to `insz=20`.

`tests/test_axera_conv_dilation_insz_threshold.py` (PR #1519, merged)
pinned the extra-`B`-run trigger's first appearance at `insz=13` (for
`cin=cout=4`, `k=3`, `pad=dilation`) and mapped a non-monotone pattern
through `insz=16`:

| | insz=12 | insz=13 | insz=14 | insz=15 | insz=16 |
| --- | --- | --- | --- | --- | --- |
| `d=3` | no | 4+3-style | no | 8-style | 4+3-style |
| `d=4` | no | 4+3-style | no | no | 8-style |

It explicitly left two things untested: `d=4` at `insz=9,10,11` (only
`8`/`12` were checked below threshold for `d=4`), and `insz=17+` to see
whether the pattern continues, stabilizes, or was a narrow window.

## Part 1: d=4's below-threshold symmetry, confirmed

`insz=9,10,11` at `dilation=4` all show **no** early trigger run (only
the already-named universal-boilerplate 5-record run every build in
this shape family carries) -- matching `d=3`'s own already-confirmed
absence at these same `insz` values. The "below insz=13, nothing
triggers" picture is now confirmed symmetric for both dilations across
`insz=9..12`.

## Part 2: insz=17..20 -- a THIRD trigger signature appears, and the
pattern never settles into anything periodic or monotone

|         | insz=17 | insz=18 | insz=19 | insz=20 |
| ---     | ---     | ---     | ---     | ---     |
| `d=3`   | **NEW 3-rec** | no | no | **NEW 3-rec** (different content) |
| `d=4`   | **4+3-style AND NEW 3-rec, together** | no | no | no |

The full grid, `insz=9` through `20`, at both dilations, never repeats
a clean cycle: absent for four straight values (`9-12`), present at
`13`, absent at `14`, dilation-dependent at `15`, present-for-both at
`16`, a new signature at `17` (dilation-dependent in a different way --
`d=3` gets only the new signature, `d=4` gets the new signature *in
addition to* the already-known `4+3-style`), absent again at `18-19`,
and the new signature recurs at `20` for `d=3` only, with *different*
register content than its own `insz=17` occurrence. No `insz-dilation`
formula, no fixed period, and no simple "on above/below X" rule fits
any of this.

**The new signature** is a standalone 3-record run, tag sequence
`(225,129,132)` -- the same tag shape as the *second half* of the
already-known `4+3-style` pattern's own 3-record tail, but appearing
here on its own, without a preceding 4-record run, and at a different
offset (~800/1411 rather than ~552-620/1160-1230). Its `reg` values
differ between occurrences: `insz=17,d=3/d=4` share `reg=93`; `insz=20,d=3`
uses `reg=67` instead -- not byte-identical the way `insz=13`'s trigger
was identical across `d=3` and `d=4`.

**Confirmed above the noise floor.** Independent rebuilds of both
`d=3,insz=17` (new signature present) and `d=3,insz=18` (no trigger)
reproduce their respective outcomes -- `insz=17`'s rebuild reproduces
the identical trigger signature at the identical offsets and identical
stream length (3,688 both times), with only 3 bytes differing
elsewhere in the stream (offset 858/864/870, ordinary noise-floor
magnitude, nowhere near the trigger's own ~800/1411 offsets);
`insz=18`'s rebuild confirms the *absence* holds even though (a
related, not-decoded-further observation) the rebuild's own
whole-stream length shifts to 3,688 from the original's 3,720 -- the
same length-class-boundary instability PR #1519 first noted for
`d=3,insz=13`, now seen a second time at a different `insz`.

## What remains open

No formula fits the full `insz=9..20` grid. The new 3-record signature
at `17`/`20` is real and reproducible but its trigger condition and
content-dependence are undecoded. This grid could be extended further
(`insz=21+`) to look for eventual periodicity, but that is not
attempted here -- reported as an honest, precise negative result on
"does a pattern exist," not a decode of one.
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


def b_runs(data, min_len=3, cutoff=1700):
    recs = mcode.decode(data, start=0, end=len(data), **mcode.FULL_RULE)
    runs = []
    i = 0
    while i < len(recs):
        if recs[i].get("kind") == "B":
            j = i
            while j < len(recs) and recs[j].get("kind") == "B":
                j += 1
            if j - i >= min_len and recs[i]["at"] < cutoff:
                sig = tuple((r["reg"], r["tag"]) for r in recs[i:j])
                runs.append((recs[i]["at"], j - i, sig))
            i = j
        else:
            i += 1
    return runs


_UNIVERSAL_BOILERPLATE_SIG = (7, 225), (24, 129), (9, 225), (24, 129), (8, 132)


def early_trigger_runs(data):
    return [r for r in b_runs(data) if r[2] != _UNIVERSAL_BOILERPLATE_SIG]


class TestD4BelowThresholdSymmetry(unittest.TestCase):
    """d=4 at insz=9,10,11 all show no trigger, matching d=3's own
    already-confirmed absence at these same values (PR #1519)."""

    def test_insz9_d4_has_no_trigger(self):
        self.assertEqual(early_trigger_runs(load("conv_dilation4_insz9.mcode.gz")), [])

    def test_insz10_d4_has_no_trigger(self):
        self.assertEqual(early_trigger_runs(load("conv_dilation4_insz10.mcode.gz")), [])

    def test_insz11_d4_has_no_trigger(self):
        self.assertEqual(early_trigger_runs(load("conv_dilation4_insz11.mcode.gz")), [])


class TestNewThirdSignatureAtInsz17(unittest.TestCase):
    """insz=17 introduces a run shape never seen at insz<=16: a
    standalone 3-record run with no preceding 4-record run, present at
    d=3 and (in addition to the already-known 4+3-style pair) at d=4."""

    NEW_SIG = ((93, 225), (54, 129), (8, 132))
    KNOWN_4PLUS3_FIRST = ((47, 225), (22, 129), (49, 225), (22, 129))
    KNOWN_4PLUS3_SECOND = ((53, 225), (22, 129), (8, 132))

    def test_d3_insz17_has_only_the_new_signature(self):
        runs = early_trigger_runs(load("conv_dilation3_insz17.mcode.gz"))
        sigs = [sig for _, _, sig in runs]
        self.assertEqual(sigs.count(self.NEW_SIG), 2, "mirrored engine copies")
        # d3,insz17 does NOT also carry the 4+3-style pair.
        self.assertNotIn(self.KNOWN_4PLUS3_FIRST, sigs)

    def test_d4_insz17_has_both_signatures_together(self):
        runs = early_trigger_runs(load("conv_dilation4_insz17.mcode.gz"))
        sigs = [sig for _, _, sig in runs]
        self.assertIn(self.NEW_SIG, sigs)
        self.assertIn(self.KNOWN_4PLUS3_FIRST, sigs)
        self.assertIn(self.KNOWN_4PLUS3_SECOND, sigs)

    def test_insz17_d3_survives_independent_rebuild(self):
        """The trigger signature itself is stable across an independent
        rebuild. The rebuild is NOT whole-stream byte-identical (3
        bytes differ, at offset 858/864/870 -- ordinary noise-floor
        magnitude, not the trigger's own ~800/1411 offsets), so this
        checks the record-level signature rather than raw bytes."""
        original = load("conv_dilation3_insz17.mcode.gz")
        rebuild = load("conv_dilation3_insz17_rebuild.mcode.gz")
        self.assertEqual(len(original), len(rebuild), "no length-class shift here")
        diffs = [i for i in range(len(original)) if original[i] != rebuild[i]]
        self.assertEqual(len(diffs), 3, "ordinary noise-floor magnitude")
        trigger_offsets = {800, 801, 802, 1411, 1412, 1413}
        self.assertFalse(
            trigger_offsets & set(diffs), "noise must not touch the trigger"
        )
        rebuild_sigs = [sig for _, _, sig in early_trigger_runs(rebuild)]
        self.assertEqual(rebuild_sigs.count(self.NEW_SIG), 2)


class TestInsz18And19RevertToNoTrigger(unittest.TestCase):
    def test_d3_insz18_has_no_trigger(self):
        self.assertEqual(early_trigger_runs(load("conv_dilation3_insz18.mcode.gz")), [])

    def test_d3_insz19_has_no_trigger(self):
        self.assertEqual(early_trigger_runs(load("conv_dilation3_insz19.mcode.gz")), [])

    def test_d4_insz18_has_no_trigger(self):
        self.assertEqual(early_trigger_runs(load("conv_dilation4_insz18.mcode.gz")), [])

    def test_d4_insz19_has_no_trigger(self):
        self.assertEqual(early_trigger_runs(load("conv_dilation4_insz19.mcode.gz")), [])

    def test_d4_insz20_has_no_trigger(self):
        self.assertEqual(early_trigger_runs(load("conv_dilation4_insz20.mcode.gz")), [])

    def test_insz18_d3_absence_survives_rebuild_despite_a_length_class_shift(self):
        """Same class of length instability PR #1519 first found at
        d=3,insz=13: the rebuild's whole-stream length differs (3720 ->
        3688), but the trigger's absence is unaffected."""
        original = load("conv_dilation3_insz18.mcode.gz")
        rebuild = load("conv_dilation3_insz18_rebuild.mcode.gz")
        self.assertNotEqual(len(original), len(rebuild))
        self.assertEqual(early_trigger_runs(rebuild), [])


class TestInsz20D3RecurrenceHasDifferentContent(unittest.TestCase):
    """d=3,insz=20 shows the same NEW 3-record signature shape as
    insz=17, but with different register content -- not a byte-
    identical recurrence the way insz=13's trigger was across d=3/d=4."""

    INSZ20_SIG = ((67, 225), (54, 129), (8, 132))
    INSZ17_SIG = ((93, 225), (54, 129), (8, 132))

    def test_insz20_d3_has_the_new_signature_shape_with_different_regs(self):
        runs = early_trigger_runs(load("conv_dilation3_insz20.mcode.gz"))
        sigs = [sig for _, _, sig in runs]
        self.assertEqual(sigs.count(self.INSZ20_SIG), 2)
        self.assertNotEqual(self.INSZ20_SIG, self.INSZ17_SIG)
        # But the tag shape (225, 129, 132) -- ignoring reg -- matches.
        self.assertEqual(
            tuple(t for _, t in self.INSZ20_SIG), tuple(t for _, t in self.INSZ17_SIG)
        )


if __name__ == "__main__":
    unittest.main()
