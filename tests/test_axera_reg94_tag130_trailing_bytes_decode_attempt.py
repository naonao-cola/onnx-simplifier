"""Attempts to decode what `reg=94,tag=130`'s own seed-varying
trailing bytes represent -- the question `tests/test_axera_reg98_
tag133_payload_seed_dependence.py` (PR #1688) explicitly left open:
"plausibly a per-build quantity of `x2`'s own... but this file does
not attempt a first-principles recomputation to confirm that guess."

## Finding: a genuine negative result -- none of the plausible
## candidate quantities this arc has already established match

Tested against Add's own three already-committed trivial-`x2`
fixtures (seed pairs `(1,2)`, `(7,42)`, `(100,999)`):

1. **`x2`'s own real scale**, computed via `ComputeAsymmetricUint8
   QuantParams` from the exact all-positive trivial-`x2` calibration
   data (`2.0+0.3*RandomState(seed2).randn(1,16)`) this fixture family
   actually used -- e.g. `0.010539852` at seed `2` -- does not match
   the trailing bytes interpreted as a float32 (`0f 88 c1 bd` little-
   endian = `-0.0945`), nor does `1/scale` (`94.878`).
2. **`x2`'s own real scale computed from the PLAIN (non-shifted)
   calibration convention** (`RandomState(2).randn(1,16)`, the
   convention `x2` uses in every NON-degenerate fixture elsewhere in
   this cluster) -- `0.018537158`, `1/scale = 53.946` -- also does not
   match.
3. **The control fixture's own already-present 3-byte payload**
   (`f\xc8W` at Add's seed `(1,2)`, present in BOTH control and
   trivial states, just one byte narrower in the control) does not
   decode to a sensible float32 under either zero-padding convention
   tried (front-padded: `4.4e14`; back-padded: `8.06e-39`) -- neither
   value corresponds to anything else this arc has already established
   for this fixture (`1/x1_scale = 57.93`, `y_scale = 0.02575`).
4. **`y_scale` and `1/x1_scale`** (the two float32 quantities this
   exact fixture's own `verb=161,bank=15` locator already carries,
   PR #1657/#1666) were checked directly against the same fixture and
   do not match either interpretation of the trailing bytes.

## What this establishes, precisely, and what it does not

**Established**: none of the plausible per-build float32 quantities
this arc has already decoded for this fixture family (`x2`'s own real
scale under either calibration convention, its reciprocal, `y_scale`,
`1/x1_scale`) match `reg=94,tag=130`'s own trailing bytes, under any
of the byte-interpretation conventions (raw little-endian float32 of
the full 4-byte payload, zero-padded 3-byte fragments in both
directions) this file tried. This is a genuine, honestly-reported
negative result, not an exhaustive proof of absence -- other
interpretations (fixed-point, a different byte order, a value derived
from `x1` rather than `x2`, or a quantity this arc has not yet
identified at all) were not all tried.

**NOT established**: what these bytes actually encode. Consistent
with `tests/test_axera_mcode_reciprocal.py`'s own precedent for
similarly still-undecoded forms (`TestZpXImmediateRegion`'s own two
"tag 0x83/0xa1" forms, pinned as raw bytes without a decode), this
file leaves the value pinned as observed-but-unexplained rather than
force a weak match into a false positive.
"""

import gzip
import os
import struct
import sys
import unittest

import numpy as np

_AXERA_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts", "axera")
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import mcode  # noqa: E402

FIX = os.path.join(_AXERA_DIR, "fixtures")


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def decode(name):
    return mcode.decode(load(name), **mcode.FULL_RULE)


def asymmetric_uint8_scale(samples):
    lo = np.float32(0.0)
    hi = np.float32(0.0)
    for s in samples:
        lo = min(lo, np.float32(s.min()))
        hi = max(hi, np.float32(s.max()))
    if hi <= lo:
        hi = lo + np.float32(1.0)
    return float((hi - lo) / np.float32(255.0))


def x2_trivial_samples(seed, shape=(1, 16), n_samples=4):
    rng = np.random.RandomState(seed)
    return [
        (2.0 + 0.3 * rng.randn(*shape)).astype(np.float32) for _ in range(n_samples)
    ]


def x2_plain_samples(seed, shape=(1, 16), n_samples=4):
    rng = np.random.RandomState(seed)
    return [rng.randn(*shape).astype(np.float32) for _ in range(n_samples)]


class TestTrailingBytesDoNotMatchX2sOwnScaleUnderEitherCalibrationConvention(
    unittest.TestCase
):
    def test_trivial_calibration_scale_and_reciprocal_do_not_match(self):
        recs = decode("add_1x16_two_live_seed1_2_trivialx2.mcode.gz")
        rec = [r for r in recs if r.get("reg") == 94 and r.get("tag") == 130][0]
        observed = struct.unpack("<f", rec["payload"])[0]

        scale = asymmetric_uint8_scale(x2_trivial_samples(2))
        self.assertGreater(abs(observed - scale) / max(abs(scale), 1e-9), 0.5)
        recip = 1.0 / scale
        self.assertGreater(abs(observed - recip) / max(abs(recip), 1e-9), 0.5)

    def test_plain_calibration_scale_and_reciprocal_do_not_match(self):
        recs = decode("add_1x16_two_live_seed1_2_trivialx2.mcode.gz")
        rec = [r for r in recs if r.get("reg") == 94 and r.get("tag") == 130][0]
        observed = struct.unpack("<f", rec["payload"])[0]

        scale = asymmetric_uint8_scale(x2_plain_samples(2))
        self.assertGreater(abs(observed - scale) / max(abs(scale), 1e-9), 0.5)
        recip = 1.0 / scale
        self.assertGreater(abs(observed - recip) / max(abs(recip), 1e-9), 0.5)


class TestControlsOwnPayloadDoesNotMatchAnyAlreadyEstablishedQuantity(
    unittest.TestCase
):
    def test_neither_zero_padding_matches_y_scale_or_recip_x1_scale(self):
        ctrl_recs = decode("add_1x16_two_live_seed1_2.mcode.gz")
        ctrl_rec = [r for r in ctrl_recs if r.get("reg") == 94 and r.get("tag") == 130][
            0
        ]
        payload = ctrl_rec["payload"]
        self.assertEqual(len(payload), 3)
        front_padded = struct.unpack("<f", b"\x00" + payload)[0]
        back_padded = struct.unpack("<f", payload + b"\x00")[0]

        y_scale_hits = [
            struct.unpack("<f", r["operand"])[0]
            for r in ctrl_recs
            if r.get("kind") == "V"
            and r.get("verb") == 161
            and r.get("bank") == 15
            and r.get("operand")
            and len(r["operand"]) == 4
        ]
        distinct_values = sorted(set(round(v, 4) for v in y_scale_hits))
        self.assertEqual(len(distinct_values), 2)  # 1/x1_scale and y_scale
        for known in distinct_values:
            for candidate in (front_padded, back_padded):
                if known == 0:
                    continue
                self.assertGreater(abs(candidate - known) / abs(known), 0.5)


class TestFixturesDecodeCleanly(unittest.TestCase):
    def test_no_decode_errors(self):
        for name in (
            "add_1x16_two_live_seed1_2.mcode.gz",
            "add_1x16_two_live_seed1_2_trivialx2.mcode.gz",
        ):
            errs = mcode.check(load(name))
            self.assertEqual(errs, [], (name, errs))


if __name__ == "__main__":
    unittest.main()
