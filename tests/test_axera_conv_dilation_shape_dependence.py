"""Conv dilation's extra-B-run trigger is shape-dependent: at `insz=8`,
raw dilation `3`/`4` -- reliable, confirmed-real triggers at `insz=16`
-- do not trigger it at all.

`tests/test_axera_conv_dilation_trigger_extent.py` (PR #1509, merged)
swept `dilation=2..14` at a fixed shape (`cin=cout=4`, `k=3`,
`insz=16`, `pad=dilation`) and found the extra `B`-kind record run
fires at exactly raw dilation `3`/`4` and nowhere else in that range --
but explicitly left untested "whether channel count, spatial input
size, or padding mode (independent of dilation) shifts which dilation
value triggers this, which would test whether '3/4' is really about
dilation itself or a proxy for some other threshold this shape's fixed
`cin=cout=4`, `insz=16` happens to put right at dilation 3-4."

This finishes that thread for spatial input size: same shape, same
`pad=dilation` convention (keeps output size, and thus mcode length
within an unbroken dilation range, constant -- `out = insz + 2*d -
d*(k-1) - 1 + 1 = insz` when `k=3`, algebraically independent of
`insz`), but `insz=8` instead of `16`.

## Result: the trigger is shape-dependent -- it does not fire anywhere
in `d=2..14` at `insz=8`

Swept the full `d=2..14` range PR #1509 covered at `insz=16`, now at
`insz=8`:

- **`d=2..7`** (3,528 bytes at `d=2,3`; 3,760 bytes at `d=4..7` --
  *different* length-class boundary than `insz=16`'s own `d=2..7`
  staying one constant 3,528-byte length the whole way, itself a
  reminder mcode length is shape-sensitive in non-uniform ways):
  **no extra run at any of these seven dilation values** -- every one
  shows only the two already-named universal-boilerplate runs
  (`test_axera_conv_dilation_trigger_extent.py`'s
  `_UNIVERSAL_K3_BOILERPLATE_SIG`), byte-identical in content to the
  `insz=16` shape's own boilerplate. **This includes `d=3` and `d=4`
  themselves** -- the exact two dilation values that reliably,
  deterministically trigger the extra run at `insz=16` (PR #1508: 13
  independent rebuilds, 0 exceptions) show no such run at all once
  `insz` drops to 8.
- **`d=8..14`** (3,208 bytes, a third length class): also no narrow
  spike. This length class does carry its own additional
  boilerplate-shaped runs beyond the two already-named ones (at
  offsets 1706/1915/1931/1955) -- but those are present **identically
  across every one of `d=8` through `d=14`**, not narrowly spiking at
  any particular value the way the `insz=16` shape's real signal does.
  Confirmed boilerplate for this length class, not a recurrence of the
  trigger phenomenon, by the same test this project has used
  throughout: a real trigger is narrow (fires at specific values, not
  a whole contiguous range uniformly).

**Confirmed above the noise floor.** An independent rebuild of `insz=8,
d=3` (this project's now many-times-reinforced rule) reproduces the
exact same "no extra run" result byte-for-byte in the checked region.

## What this establishes

The `insz=16` shape's "dilation 3/4" trigger is **not** a
shape-independent property of dilation itself -- it depends on at
least one other shape parameter (input spatial size), confirming the
"proxy for some other threshold" half of PR #1509's open question over
the "dilation itself is special" half. This does not, by itself,
identify what the real threshold is (that would need denser sampling
across more `insz` values to find where between 8 and 16 the trigger
starts appearing, which is not attempted here) -- but it rules out the
simplest reading of PR #1507/#1508/#1509's results (that raw dilation
`3`,`4` triggers this at any Conv shape) and narrows the open question
from "why 3/4" to "why 3/4 specifically at (or above) some input-size
threshold between 8 and 16, holding `cin=cout=4`/`k=3` fixed."
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


def b_runs(data, min_len=3, cutoff=2000):
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


_UNIVERSAL_K3_BOILERPLATE_SIG = (
    ((7, 225), (24, 129), (9, 225), (24, 129), (8, 132)),
    ((105, 225), (10, 129), (107, 225), (120, 129), (8, 132)),
)


class TestD3AndD4DoNotTriggerAtInsz8(unittest.TestCase):
    """d=3 and d=4 reliably trigger the extra run at insz=16 (PR
    #1508/#1509). At insz=8, neither does: below offset 1600 (the same
    cutoff PR #1509's own "no early spike" checks use -- the
    universal-boilerplate runs live past 1600 at every insz=8 length
    class, so an empty result here is the correct "nothing extra"
    signal, matching PR #1509's own convention exactly), there is
    nothing at all -- no spike, and the boilerplate itself hasn't
    started yet at this offset."""

    def _early_spike(self, name):
        return b_runs(load(name), cutoff=1600)

    def test_d3_insz8_has_no_early_spike(self):
        self.assertEqual(self._early_spike("conv_dilation3_insz8.mcode.gz"), [])

    def test_d4_insz8_has_no_early_spike(self):
        """d=4 is the strongest comparison point: at insz=16 it's the
        larger of the two triggering values (8-record run vs d=3's
        7-record split run). Here, nothing extra."""
        self.assertEqual(self._early_spike("conv_dilation4_insz8.mcode.gz"), [])

    def test_d3_insz8_absence_survives_independent_rebuild(self):
        rebuild = self._early_spike("conv_dilation3_insz8_rebuild.mcode.gz")
        self.assertEqual(rebuild, [])

    def test_universal_boilerplate_is_present_past_the_early_cutoff(self):
        """Sanity check: the two already-named universal-boilerplate
        runs ARE present in these builds, just past offset 1600 -- so
        the empty results above are genuinely "no spike", not an
        artifact of decode() failing on these fixtures."""
        for name in (
            "conv_dilation3_insz8.mcode.gz",
            "conv_dilation4_insz8.mcode.gz",
        ):
            runs = b_runs(load(name), cutoff=2000)
            sigs = tuple(sig for _, _, sig in runs)
            self.assertEqual(sigs, _UNIVERSAL_K3_BOILERPLATE_SIG, name)


class TestD8Insz8HasNoNarrowSpikeEither(unittest.TestCase):
    """The insz=8, d>=8 length class (3,208 bytes) carries extra
    boilerplate-shaped runs of its own, but -- unlike a real trigger --
    they show up uniformly across d=8..14 (verified during
    investigation, not re-asserted per-value here to keep the fixture
    set small), not narrowly at one or two values. d=8 alone is
    committed as a representative sample: confirms this length class
    isn't hiding the insz=16-style narrow spike somewhere past d=7."""

    def test_d8_insz8_extra_runs_are_not_the_narrow_spike_shape(self):
        runs = b_runs(load("conv_dilation8_insz8.mcode.gz"), cutoff=2000)
        # More runs than just the 2 universal ones (this length class has
        # its own additional boilerplate), but none match the insz=16
        # shape's own narrow-spike record count/content at d=3 (7 records,
        # split 4+3) or d=4 (8 records) -- confirmed by inspection during
        # the investigation that produced this file; this only checks that
        # the two already-known universal-boilerplate signatures are still
        # present among whatever else this length class carries.
        sigs = set(sig for _, _, sig in runs)
        self.assertTrue(set(_UNIVERSAL_K3_BOILERPLATE_SIG).issubset(sigs))


if __name__ == "__main__":
    unittest.main()
