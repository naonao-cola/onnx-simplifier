"""Pinning the insz threshold where Conv dilation's extra-B-run trigger
starts appearing -- and finding the picture above that threshold is
richer (dilation-dependent, non-monotone) than a simple on/off switch.

`tests/test_axera_conv_dilation_shape_dependence.py` (PR #1513, merged)
found the extra-`B`-run trigger (real and reliable at raw dilation
`3`/`4` for `cin=cout=4`, `k=3`, `insz=16`, per PR #1508's 13-rebuild
check) does not fire at all at `insz=8`, confirming the trigger is
shape-dependent -- but left "denser sampling across more `insz` values
to find where between 8 and 16 the trigger starts appearing" as
explicitly untried.

## The threshold: insz=13, confirmed for both dilation 3 and 4

Sweeping `insz=9..15` at `dilation=3` (`pad=dilation`, same convention
as every prior file in this thread), plus targeted `dilation=4` builds
at `insz=12..15` to cross-check: **no build at `insz<=12` shows the
early trigger run, at either dilation (`d=3`: `insz=9,10,11,12,14`
checked and absent; `d=4`: `insz=8,12` checked and absent). The first
`insz` where it appears, at both dilations, is `insz=13`** -- and at
that exact `insz`, `dilation=3` and `dilation=4` produce
**byte-identical** trigger content (`reg` sequence `21,23,27`, tag
sequence `225,129,225,129` then `225,129,132` -- the same "4-record run
+ 3-record run ending on `reg=8`/tag=`132`" shape this project already
named for `d=3`'s own `insz=16` trigger, `test_axera_conv_dilation_trigger_extent.py`).
This answers the PR #1513 question precisely: the threshold is `insz=13`,
not merely "somewhere between 8 and 16".

## But the picture above insz=13 is not "on above threshold" -- it's
non-monotone and dilation-dependent, refuting the simplest formulas

|         | insz=12 | insz=13 | insz=14 | insz=15 | insz=16 |
| ---     | ---     | ---     | ---     | ---     | ---     |
| `d=3`   | no      | **4+3-style** | no | **8-style**   | **4+3-style** |
| `d=4`   | no      | **4+3-style** | no | no            | **8-style**   |

Three things this rules out, tested directly rather than assumed:

1. **Not "present at or above insz=13"**: `insz=14` reverts to absent at
   *both* dilations, immediately after `insz=13` triggered.
2. **Not a function of `insz - dilation` alone**: `d=3,insz=14` and
   `d=4,insz=14` share `insz=14` but have `insz-dilation` of `11` and
   `10` respectively, yet both give "no" -- while `d=3,insz=13`
   (`insz-dilation=10`, same value as `d=4,insz=14`) gives "yes". If
   `insz-dilation` were the driver, `d=3,insz=13` and `d=4,insz=14`
   (both `insz-dilation=10`) would agree; they don't.
3. **`insz=15` is where dilation-dependence becomes unambiguous**:
   `d=3,insz=15` produces the **exact same 8-record trigger content**
   (`reg` sequence `27,29,31,33`) as the already-known `d=4,insz=16`
   trigger -- a real, byte-identical coincidence across two different
   `(dilation, insz)` pairs -- while `d=4,insz=15` shows no trigger at
   all. Same `insz`, different `dilation`, different outcome.

**Confirmed above the noise floor, with a bonus finding.** An
independent rebuild of `d=3,insz=13` reproduces the trigger's own
content byte-for-byte (`reg` sequence `21,23,27` again, at the *same*
absolute offset for the first engine copy) -- but the **whole stream
length itself shifts by 32 bytes** between the two builds (3,688 vs
3,720), with the second engine copy's mirrored offset shifting by the
same 32 bytes. This is the same phenomenon PR #1509 found for `d=12` at
`insz=16` ("a threshold this project's earlier work already noted
existed but didn't explore") -- `insz=13,d=3` sits right at one of
these length-class boundaries, so ordinary build-to-build noise can tip
the *whole-stream* length by a fixed chunk even though it does not
touch the trigger's own content. This is reported as a related,
noteworthy observation, not decoded further here.

## What remains open

The `insz=13` threshold itself, and why it produces identical content
regardless of dilation, is pinned but not explained. The richer
dilation-dependence emerging at `insz=15`/`16` (same trigger family,
different specific `reg` sequences depending on dilation) and the
`insz=14` "gap" are real, confirmed phenomena with no formula fitted
here -- a `d=4` sweep at `insz=9..11` (untested, only `12` and `8` were
checked at `d=4` below the threshold) would be the next natural
extension, along with an `insz=17+` sweep to see whether the pattern
found here (13: yes, 14: no, 15: dilation-dependent, 16: both yes)
continues or was itself a narrow window.
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


def _early_trigger_runs(data):
    """Runs below cutoff, excluding the single universal-boilerplate
    5-record run this shape always carries at this offset range."""
    return [r for r in b_runs(data) if r[2] != _UNIVERSAL_BOILERPLATE_SIG]


class TestNoTriggerAtOrBelowInsz12(unittest.TestCase):
    def test_d3_insz12_has_no_trigger(self):
        self.assertEqual(
            _early_trigger_runs(load("conv_dilation3_insz12.mcode.gz")), []
        )

    def test_d4_insz12_has_no_trigger(self):
        self.assertEqual(
            _early_trigger_runs(load("conv_dilation4_insz12.mcode.gz")), []
        )


class TestInsz13IsTheFirstAppearanceForBothDilations(unittest.TestCase):
    FIRST_COPY_SIG = ((21, 225), (22, 129), (23, 225), (22, 129))
    SECOND_COPY_SIG = ((27, 225), (22, 129), (8, 132))

    def _first_run_pair(self, name):
        """Both mirrored engine copies produce this 4-record + 3-record
        run pair, so 4 runs total (2 copies x 2 runs each)."""
        runs = _early_trigger_runs(load(name))
        self.assertEqual(
            len(runs), 4, f"{name}: expected 2 mirrored 4+3-record run pairs"
        )
        return [sig for _, _, sig in runs]

    def test_d3_insz13_triggers_with_this_exact_content(self):
        sigs = self._first_run_pair("conv_dilation3_insz13.mcode.gz")
        self.assertIn(self.FIRST_COPY_SIG, sigs)
        self.assertIn(self.SECOND_COPY_SIG, sigs)

    def test_d4_insz13_triggers_with_byte_identical_content(self):
        """Same insz, different dilation -- same trigger content exactly."""
        sigs = self._first_run_pair("conv_dilation4_insz13.mcode.gz")
        self.assertIn(self.FIRST_COPY_SIG, sigs)
        self.assertIn(self.SECOND_COPY_SIG, sigs)

    def test_insz13_survives_independent_rebuild_content_wise(self):
        """The trigger's own content is stable across a rebuild, even
        though (see below) the whole-stream length is not."""
        sigs = self._first_run_pair("conv_dilation3_insz13_rebuild.mcode.gz")
        self.assertIn(self.FIRST_COPY_SIG, sigs)
        self.assertIn(self.SECOND_COPY_SIG, sigs)

    def test_insz13_sits_at_an_unstable_length_class_boundary(self):
        """Unlike every other build in this file, d=3,insz=13's rebuild
        has a different total stream length -- a +32-byte shift, the
        same class of length-class-boundary noise PR #1509 found for
        d=12 at insz=16. The trigger content itself is unaffected (see
        above); only the whole-stream length is."""
        original = load("conv_dilation3_insz13.mcode.gz")
        rebuild = load("conv_dilation3_insz13_rebuild.mcode.gz")
        self.assertNotEqual(len(original), len(rebuild))
        self.assertEqual(len(rebuild) - len(original), 32)


class TestInsz14RevertsToNoTriggerAtBothDilations(unittest.TestCase):
    """Immediately after insz=13 triggers, insz=14 goes back to
    nothing -- refutes "present at or above the threshold"."""

    def test_d3_insz14_has_no_trigger(self):
        self.assertEqual(
            _early_trigger_runs(load("conv_dilation3_insz14.mcode.gz")), []
        )

    def test_d4_insz14_has_no_trigger(self):
        self.assertEqual(
            _early_trigger_runs(load("conv_dilation4_insz14.mcode.gz")), []
        )


class TestInsz15DivergesByDilation(unittest.TestCase):
    """d=3 at insz=15 produces the exact same 8-record trigger content
    as the already-known d=4,insz=16 case; d=4 at insz=15 shows no
    trigger at all. Same insz, different dilation, different outcome --
    the clearest single data point ruling out any insz-only or
    insz-minus-dilation-only formula."""

    EIGHT_RECORD_SIG = (
        (27, 225),
        (22, 129),
        (29, 225),
        (82, 129),
        (31, 225),
        (74, 129),
        (33, 225),
        (44, 129),
    )

    def test_d3_insz15_matches_d4_insz16s_own_trigger_content(self):
        runs = _early_trigger_runs(load("conv_dilation3_insz15.mcode.gz"))
        sigs = [sig for _, _, sig in runs]
        self.assertIn(self.EIGHT_RECORD_SIG, sigs)

        # Cross-check directly against the already-committed d=4,insz=16
        # fixture rather than hardcoding its content twice.
        d4_insz16_runs = _early_trigger_runs(load("conv_dilation4.mcode.gz"))
        d4_insz16_sigs = [sig for _, _, sig in d4_insz16_runs]
        self.assertIn(self.EIGHT_RECORD_SIG, d4_insz16_sigs)

    def test_d4_insz15_has_no_trigger(self):
        self.assertEqual(
            _early_trigger_runs(load("conv_dilation4_insz15.mcode.gz")), []
        )


if __name__ == "__main__":
    unittest.main()
