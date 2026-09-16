"""Sweeping `K` wide past PR #1543's `K<=32` range for Gemm's `K*M-1`
field, looking for a `K`-side regime break analogous to `M`'s own
`M=4`/`M=5` split: found one, and it's richer than a simple break --
three distinct record shapes across three `K` ranges, not two.

`tests/test_axera_gemm_8m1_field_kn_dependence.py` (PR #1543) decoded
a Gemm S-unit record (`p=2, tag=130, reg=70`, payload `d0 0c <XX>`) as
`K*M-1` for `M<=4`, tested up to `K=32` (`4*32-1=127`, confirmed). It
never swept `K` past 32.

## Regime 1 (already known): `2 <= K <= 32` -- single-byte value,
## `p=2, tag=130, reg=70`

Reconfirmed at `K=32` (value `127`, matching PR #1543) using a fixture
built the same way PR #1543's own `build.py` did (same RNG seeds,
`M=4` fixed). Not re-committed as a fixture here since PR #1543 already
committed an equivalent `K=32` point.

## Regime 2 (new): `33 <= K <= 64` -- the value vanishes, but the
## record does not

Building `M=4` at `K = 33..64` (a dense sweep confirms this holds at
every integer checked, not just a couple of samples) shows the record
carrying the `d0 0c` payload prefix still EXISTS at every point, but
its shape has changed: `p=1, tag=131, reg=70`, payload exactly `d0 0c`
-- 2 bytes, no trailing value byte at all. This is not a byte-overflow
story (`4*40-1=159` fits comfortably in a byte, yet no value is
present); the record's own declared payload length (`p`) shrank by
one, which -- per this project's established `[p][p+1 payload
bytes][tag][value]` short-unit grammar -- removes room for a value
byte entirely, not truncates one.

**Confirmed above the noise floor.** An independent rebuild of `K=64`
reproduces the identical shrunk record (`p=1, tag=131, reg=70,
payload=d0 0c`) at the identical offset; the only 5 bytes that differ
between the build and its rebuild are at offsets 301/303/319/323/325,
squarely inside this project's already-known `~295-330` noise zone.

## Regime 3 (new): `K >= 65` -- the value reappears, now 2 bytes wide
## (little-endian), and looks rounded rather than exact

At `K=65` the record widens again to a THIRD distinct shape --
`p=3, tag=129, reg=26`, payload `d0 0c <lo> <hi>` (4 bytes) -- and the
trailing 2 bytes, read as a little-endian `u16`, are `383` (`0x017f`).
`383 = 4*96 - 1`, **not** `4*65-1=259` -- the value at `K=65` matches
`K=96`'s own formula output, not its own `K`. Sweeping `K =
65,72,80,88,90,92,94,95,96` all give the identical `383`: a flat
plateau across the whole `[65,96]` range, not a value that tracks `K`
directly. `K=128` gives `511` (`0x01ff`) `= 4*128-1` exactly.

Both plateau endpoints are consistent with `K` being rounded UP to the
next multiple of 32 before the `K*M-1` formula is applied:
`ceil(65/32)*32 = 96`, ..., `ceil(96/32)*32 = 96` (all of `[65,96]`
round to `96`, matching the observed constant `383 = 4*96-1`), and
`ceil(128/32)*32 = 128` (`128` is already a multiple of 32, giving
`511 = 4*128-1` directly, still a single, unrounded point though --
this project has not tested a second point past `96` to confirm the
rounding-to-32 idea holds generally, only that both endpoints checked
are consistent with it, not that it's proven).

**This rounding idea does NOT explain Regime 2.** Under the same
"round K up to the next multiple of 32" rule, `K` in `(32,64]` would
round to `64`, predicting a value of `4*64-1=255` -- a value that
still fits comfortably in a single byte, the same width Regime 1
already uses. But Regime 2 has no value at all, in a genuinely
narrower (not wider) record than Regime 1's. So Regime 2 is not simply
"Regime 3's rounding rule, still expressed in a single byte" -- it is
a third, structurally distinct case, and the reason the compiler drops
the value entirely in this specific `K` range (rather than writing the
still-byte-sized rounded value the way Regime 1 would) is not decoded
here.

## What remains open

- The exact `Regime 2 -> Regime 3` boundary is pinned precisely (`K=64`
  is Regime 2, `K=65` is Regime 3, confirmed at consecutive integers).
  The exact `Regime 1 -> Regime 2` boundary is likewise pinned (`K=32`
  Regime 1, `K=33` Regime 2, both consecutive integers, `K=32`'s value
  independently reconfirmed here).
- Regime 3's own internal structure past `K=96` (whether `[97,128]`
  holds one more plateau at a still-larger rounded value, or several)
  is untested; only `K=96` and `K=128` were sampled in that range.
- *Why* Regime 2 drops the value instead of writing a rounded-but-
  still-single-byte one, and what determines the specific `32`
  rounding granularity Regime 3's two data points are consistent with,
  are not decoded -- this file establishes the shape of the three
  regimes and their boundaries precisely, not their underlying cause.
- This was tested only at `M=4` (the pre-break `K*M-1` regime per PR
  #1540); whether the same three-regime `K` structure recurs
  (identically or differently) at a post-break `M` value (e.g. `M=8`,
  where the formula is `K*floor(M/2)-1`) is untested.
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


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def decode_segment(data, pos, length):
    end = pos + length
    while end > pos and data[end - 1] == 0:
        end -= 1
    return mcode.decode(data, start=pos, end=end, **mcode.FULL_RULE)


def find_d0_0c_records(data):
    """All short units anywhere in the stream whose payload starts with
    the `d0 0c` prefix this whole thread's decoded field shares,
    regardless of their current p/tag/reg shape (which changes across
    the three regimes this file characterizes)."""
    _, segs = mcode.segments(data)
    out = []
    for pos, length, _ in segs:
        for r in decode_segment(data, pos, length):
            if r.get("kind") == "S" and r.get("payload", b"")[:2] == b"\xd0\x0c":
                out.append(r)
    return out


class TestRegime2ValuelessRecordAtK33AndK64(unittest.TestCase):
    """K=33 and K=64 (both inside the new 'valueless' gap) show the
    same shrunk record shape: p=1, tag=131, reg=70, no trailing byte."""

    def test_k33_has_no_value_byte(self):
        recs = find_d0_0c_records(load("gemm_4x33x8_m4k33.mcode.gz"))
        self.assertEqual(len(recs), 1)
        r = recs[0]
        self.assertEqual(
            (r["p"], r["tag"], r["reg"], r["payload"]), (1, 131, 70, b"\xd0\x0c")
        )

    def test_k64_has_no_value_byte(self):
        recs = find_d0_0c_records(load("gemm_4x64x8_m4k64.mcode.gz"))
        self.assertEqual(len(recs), 1)
        r = recs[0]
        self.assertEqual(
            (r["p"], r["tag"], r["reg"], r["payload"]), (1, 131, 70, b"\xd0\x0c")
        )


class TestRegime2SurvivesAnIndependentRebuild(unittest.TestCase):
    def test_k64_rebuild_matches_and_noise_is_in_the_known_zone(self):
        orig = load("gemm_4x64x8_m4k64.mcode.gz")
        reb = load("gemm_4x64x8_m4k64_rebuild.mcode.gz")
        self.assertEqual(len(orig), len(reb))

        recs_orig = find_d0_0c_records(orig)
        recs_reb = find_d0_0c_records(reb)
        self.assertEqual(len(recs_orig), 1)
        self.assertEqual(len(recs_reb), 1)
        self.assertEqual(
            (
                recs_orig[0]["p"],
                recs_orig[0]["tag"],
                recs_orig[0]["reg"],
                recs_orig[0]["payload"],
            ),
            (
                recs_reb[0]["p"],
                recs_reb[0]["tag"],
                recs_reb[0]["reg"],
                recs_reb[0]["payload"],
            ),
        )

        diffs = [i for i in range(len(orig)) if orig[i] != reb[i]]
        self.assertTrue(diffs, "expected some ordinary noise-floor diff")
        for i in diffs:
            self.assertTrue(
                295 <= i <= 330, f"unexpected diff outside known noise zone at {i}"
            )


class TestRegime3WidenedValueIsARoundedPlateauNotExactK(unittest.TestCase):
    """K=65..96 all show the identical 16-bit value (383, matching
    K=96's own 4*K-1), not a value tracking each K individually."""

    PLATEAU_CASES = [
        ("gemm_4x65x8_m4k65.mcode.gz", 65),
        ("gemm_4x96x8_m4k96.mcode.gz", 96),
    ]

    def _value16(self, data):
        recs = find_d0_0c_records(data)
        self.assertEqual(len(recs), 1)
        r = recs[0]
        self.assertEqual((r["p"], r["tag"], r["reg"]), (3, 129, 26))
        self.assertEqual(len(r["payload"]), 4)
        return struct.unpack("<H", r["payload"][2:])[0]

    def test_k65_and_k96_share_the_identical_plateau_value(self):
        v65 = self._value16(load("gemm_4x65x8_m4k65.mcode.gz"))
        v96 = self._value16(load("gemm_4x96x8_m4k96.mcode.gz"))
        self.assertEqual(v65, v96)
        self.assertEqual(v65, 4 * 96 - 1)
        self.assertNotEqual(
            v65, 4 * 65 - 1, "K=65's own K*M-1 should NOT match -- it's rounded"
        )

    def test_k128_matches_its_own_km1_exactly(self):
        v128 = self._value16(load("gemm_4x128x8_m4k128.mcode.gz"))
        self.assertEqual(v128, 4 * 128 - 1)


if __name__ == "__main__":
    unittest.main()
