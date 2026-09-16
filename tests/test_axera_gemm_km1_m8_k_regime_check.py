"""Checking whether Gemm's `K*M-1` field's three-regime `K`-side
structure (found at `M=4` by PR #1545) recurs at `M=8`: it does,
byte-for-byte identically -- same regime boundaries, same record
shapes, and (because `M=8`'s post-break multiplier happens to equal
`M=4`'s own multiplier) even the same numeric values.

`tests/test_axera_gemm_km1_k_regime_check.py` (PR #1545) found Gemm's
`K*M-1` field, at `M=4` (the pre-break regime), has THREE distinct
`K`-side record shapes: `K in [2,32]` (single-byte value, exact
formula), `K in [33,64]` (value vanishes, record shrinks to a
valueless 2-byte form), `K>=65` (widens to a little-endian 2-byte
value, a plateau consistent with `K` rounded up to the next multiple
of 32). It explicitly left open whether this recurs at a post-break
`M` value, e.g. `M=8` (formula `K*floor(M/2)-1 = K*4-1`, the same
multiplier as `M=4`'s own `K*4-1`).

## Result: identical structure at every point tested

Building `M=8` at `K = 32, 33, 64, 65, 96, 128` (mirroring PR #1545's
own `M=4` sweep exactly) and locating the `d0 0c`-prefixed record:

| K   | M=4 (PR #1545)                  | M=8 (this file)                 |
| --- | -------------------------------- | -------------------------------- |
| 32  | p=2,tag=130,reg=70, value=127    | p=2,tag=130,reg=70, value=127    |
| 33  | p=1,tag=131,reg=70, no value     | p=1,tag=131,reg=70, no value     |
| 64  | p=1,tag=131,reg=70, no value     | p=1,tag=131,reg=70, no value     |
| 65  | p=3,tag=129,reg=26, value=383    | p=3,tag=129,reg=26, value=383    |
| 96  | p=3,tag=129,reg=26, value=383    | p=3,tag=129,reg=26, value=383    |
| 128 | p=3,tag=129,reg=26, value=511    | p=3,tag=129,reg=26, value=511    |

Every regime boundary (`K=32/33`, `K=64/65`), every record shape
(`p`/`tag`/`reg`), and every numeric value is identical between `M=4`
and `M=8` at the corresponding `K`. The values match because `M=8`'s
post-break multiplier (`floor(8/2)=4`) equals `M=4`'s own pre-break
multiplier -- both give `4*K-1` (or the same `K=96`-rounded plateau
value `383`, or `4*128-1=511`) -- but the record *shapes* being
identical too (not just the values) is the more informative result:
it shows the `p`/`tag`/`reg` identity of this record, and the `K`
boundaries where its shape changes, are not `M=4`-specific artifacts.
They are a property of `K` alone, with `M` only entering through the
value that gets written once the shape is otherwise determined.

**Confirmed above the noise floor.** An independent rebuild of
`M=8,K=64` reproduces the identical valueless record (`p=1, tag=131,
reg=70, payload=d0 0c`) exactly. Both `M=8,K=64` and its rebuild pass
`mcode.check()` cleanly (`[]`), as do all six new `M=8` builds.

## What remains open

The same open questions PR #1545 already listed for `M=4` apply here
too, now doubly confirmed rather than resolved: why Regime 2 drops the
value instead of writing a still-byte-sized rounded one, and what
determines the `32` rounding granularity Regime 3 is consistent with.
This file adds a second `M` value's worth of evidence that the `K`-side
structure is `M`-independent in shape (only its value differs, per the
already-known `K*floor(M/2)-1` formula) -- it does not decode the
underlying mechanism.
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
    the three K-side regimes)."""
    _, segs = mcode.segments(data)
    out = []
    for pos, length, _ in segs:
        for r in decode_segment(data, pos, length):
            if r.get("kind") == "S" and r.get("payload", b"")[:2] == b"\xd0\x0c":
                out.append(r)
    return out


class TestM8Regime1MatchesM4Exactly(unittest.TestCase):
    def test_k32_is_127_same_shape_as_m4(self):
        data = load("gemm_8x32x8_m8k32.mcode.gz")
        self.assertEqual(mcode.check(data), [])
        recs = find_d0_0c_records(data)
        self.assertEqual(len(recs), 2)
        for r in recs:
            self.assertEqual((r["p"], r["tag"], r["reg"]), (2, 130, 70))
            self.assertEqual(r["payload"][2], 127)


class TestM8Regime2MatchesM4Exactly(unittest.TestCase):
    def test_k33_has_no_value_byte(self):
        recs = find_d0_0c_records(load("gemm_8x33x8_m8k33.mcode.gz"))
        self.assertEqual(len(recs), 2)
        for r in recs:
            self.assertEqual(
                (r["p"], r["tag"], r["reg"], r["payload"]), (1, 131, 70, b"\xd0\x0c")
            )

    def test_k64_has_no_value_byte(self):
        recs = find_d0_0c_records(load("gemm_8x64x8_m8k64.mcode.gz"))
        self.assertEqual(len(recs), 2)
        for r in recs:
            self.assertEqual(
                (r["p"], r["tag"], r["reg"], r["payload"]), (1, 131, 70, b"\xd0\x0c")
            )

    def test_k64_survives_an_independent_rebuild(self):
        orig = load("gemm_8x64x8_m8k64.mcode.gz")
        reb = load("gemm_8x64x8_m8k64_rebuild.mcode.gz")
        self.assertEqual(mcode.check(orig), [])
        self.assertEqual(mcode.check(reb), [])
        recs_orig = find_d0_0c_records(orig)
        recs_reb = find_d0_0c_records(reb)
        self.assertEqual(len(recs_orig), 2)
        self.assertEqual(len(recs_reb), 2)
        for r in recs_orig + recs_reb:
            self.assertEqual(
                (r["p"], r["tag"], r["reg"], r["payload"]), (1, 131, 70, b"\xd0\x0c")
            )


class TestM8Regime3MatchesM4Exactly(unittest.TestCase):
    def _value16(self, data):
        recs = find_d0_0c_records(data)
        self.assertEqual(len(recs), 2)
        r = recs[0]
        self.assertEqual((r["p"], r["tag"], r["reg"]), (3, 129, 26))
        self.assertEqual(len(r["payload"]), 4)
        for other in recs[1:]:
            self.assertEqual(other["payload"], r["payload"])
        return struct.unpack("<H", r["payload"][2:])[0]

    def test_k65_and_k96_share_the_identical_plateau_value(self):
        v65 = self._value16(load("gemm_8x65x8_m8k65.mcode.gz"))
        v96 = self._value16(load("gemm_8x96x8_m8k96.mcode.gz"))
        self.assertEqual(v65, v96)
        self.assertEqual(v65, 383)  # matches M=4's own K=65/96 plateau exactly

    def test_k128_matches_its_own_formula_exactly(self):
        v128 = self._value16(load("gemm_8x128x8_m8k128.mcode.gz"))
        self.assertEqual(v128, 511)  # K*floor(M/2)-1 = 128*4-1 = 511


if __name__ == "__main__":
    unittest.main()
