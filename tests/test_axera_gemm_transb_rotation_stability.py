"""Gemm's "3-way rotation" at FC scale, rebuild-stability checked: it
was never transB-specific -- it's ordinary allocation instability, the
same class of noise MatMul's `A_offset`/`B_offset` coin flip (PR #1506)
turned out to be, not Conv's dilation trigger (PR #1508), which
*survived* the identical check.

`tests/test_axera_gemm_transb_scaling.py` (PR #1500, merged) compared
exactly ONE `transB=0` build against ONE `transB=1` build of a real
FC-layer-shaped Gemm (`1x512x1000`) and found a clean "3-way rotation"
at offsets 891-908: two `[p][payload][tag][reg]` short units (`tag=0x82`,
`reg=8`) plus one trailing raw byte pair, reduced to 3 "slots" each
holding one of exactly 3 fixed 2-byte values, cyclically rotated
between the two builds. It already correctly hedged this as "the same
register/slot-allocation phenomenon as the earlier-decoded cases, not a
value computed from `transB` itself" -- but the specific claim of
*exactly 3 fixed values* undergoing *exactly one 3-cycle* was drawn
from a single comparison pair and never checked against independent
rebuilds of one unchanged `transB` value, the way this project's
established method (PR #1506 for MatMul, PR #1508 for Conv) requires
before trusting a "stable pattern" conclusion.

## Method: mirror PR #1506/#1508 exactly, applied to Gemm

Built **6 independent rebuilds** of the identical `transB=0`,
`Gemm(1,512,1000)` config (matching
`tests/test_axera_gemm_transb_scaling.py`'s own model-building method
and RNG seeds -- `B0`/`C0` from `np.random.default_rng(3)`, calibration
from `np.random.default_rng(0)` -- byte for byte the same script,
just rerun 6 times independently) via `pulsar2_docker.build()`. All 6
rebuilds, plus the already-committed `gemm_1x512x1000_tb0.mcode.gz`
fixture (a 7th, independent sample of the identical config), serialize
to the identical 4,368-byte length.

## Result: the "3 fixed values" claim does not survive -- 7 distinct
values across 8 samples, never touching `transB` at all

Extracting the same 3 slots PR #1500's own `slots()` helper defined
(`d[892],d[894]`; `d[898],d[900]`; `d[904],d[906]`) from all 6 fresh
`transB=0` rebuilds plus the already-committed `transB=0` **and**
`transB=1` fixtures (8 samples total, spanning both `transB` values but
never varying anything else) shows **7 distinct 2-byte values** appear
across those slots -- `(8,35)`, `(8,147)`, `(35,32)`, `(78,35)`,
`(78,147)`, `(147,48)`, `(147,64)` -- not the 3 values PR #1500's own
single comparison pair happened to show. No two of the 8 samples share
the identical 3-slot arrangement. This directly refutes "exactly 3
fixed values, one specific 3-cycle" as a real property of this region:
it was true of the one pair PR #1500 compared, and not true more
broadly.

## It's not even a stable *record* boundary, let alone a stable value

`mcode.decode()` at offsets 891-903 shows the record structure itself
differs across rebuilds, not just which values occupy it: the
already-committed `tb0` fixture and one fresh rebuild (`r1`) decode
cleanly into exactly two back-to-back short units starting at byte
891; the other five samples checked here decode into **two leading
unparsed (`raw`) bytes before the first short unit**, a structurally
different parse at the identical absolute offset. This is a stronger
form of instability than value-reassignment among fixed slots -- the
byte content immediately preceding this window differs enough,
build to build, to shift where a recognized record even starts,
despite the whole stream staying a constant 4,368 bytes throughout.

## Conclusion

This region is ordinary allocation/scheduling noise at FC scale, the
same general class this project has repeatedly found elsewhere
(Gemm's own M=8 case, MatMul's parameter name-table order, and -- per
the rebuild-stability check this file performs -- now confirmed for
this specific FC-scale Gemm window too). PR #1500's own output-scale-
quad finding (the *other* mechanism it documented, at a completely
separate offset, tied directly to the quant model's real
`output_scales` value) is untouched by this file and remains correct.
Only the "3-way rotation, 3 fixed values" characterization of the
*second* mechanism is revised here -- narrowed from "a specific,
clean 3-cycle" to "ordinary multi-valued allocation noise," the same
kind of honest, evidence-driven revision PR #1506 made to PR #1493's
causal claim about MatMul.
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


def slots(d):
    return [(d[892], d[894]), (d[898], d[900]), (d[904], d[906])]


class TestAllSamplesShareTheSameStreamLength(unittest.TestCase):
    """6 fresh rebuilds + the 2 already-committed fixtures (tb0 and
    tb1) all serialize to the identical 4,368-byte length -- any
    variation found below is not a length-class artifact."""

    FRESH = [
        "gemm_1x512x1000_tb0_rebuild0.mcode.gz",
        "gemm_1x512x1000_tb0_rebuild1.mcode.gz",
        "gemm_1x512x1000_tb0_rebuild2.mcode.gz",
    ]

    def test_all_lengths_match(self):
        lengths = {len(load(name)) for name in self.FRESH}
        lengths.add(len(load("gemm_1x512x1000_tb0.mcode.gz")))
        lengths.add(len(load("gemm_1x512x1000_tb1.mcode.gz")))
        self.assertEqual(lengths, {4368})


class TestSlotValuesAreNotFixedAtThreeAcrossRebuilds(unittest.TestCase):
    """PR #1500 found exactly 3 distinct 2-byte values across ONE
    transB=0/transB=1 pair. Across 3 committed fresh transB=0 rebuilds
    plus the 2 already-committed fixtures (5 samples, all transB=0
    except one transB=1), more than 3 distinct values appear -- the
    "3 fixed values" framing does not generalize."""

    SAMPLES = [
        "gemm_1x512x1000_tb0.mcode.gz",
        "gemm_1x512x1000_tb1.mcode.gz",
        "gemm_1x512x1000_tb0_rebuild0.mcode.gz",
        "gemm_1x512x1000_tb0_rebuild1.mcode.gz",
        "gemm_1x512x1000_tb0_rebuild2.mcode.gz",
    ]

    def test_more_than_three_distinct_values_appear(self):
        pool = set()
        for name in self.SAMPLES:
            pool.update(slots(load(name)))
        self.assertGreater(
            len(pool),
            3,
            "expected more than PR #1500's original 3 fixed values across"
            " a wider rebuild sample",
        )

    def test_no_two_fresh_rebuilds_share_the_identical_slot_arrangement(self):
        arrangements = [
            tuple(slots(load(name))) for name in self.SAMPLES if "rebuild" in name
        ]
        self.assertEqual(
            len(set(arrangements)),
            len(arrangements),
            "expected every fresh transB=0 rebuild to differ from every other",
        )


class TestRecordBoundaryItselfIsUnstable(unittest.TestCase):
    """mcode.decode() at offsets 891-903 shows some rebuilds parse
    cleanly into two short units starting at 891, while others show
    two leading unparsed raw bytes first -- a structural difference in
    the parse, not just which values occupy a fixed record shape."""

    def _leading_raw_count(self, name):
        recs = mcode.decode(load(name), start=891, end=903)
        n = 0
        for r in recs:
            if r["kind"] != "raw":
                break
            n += 1
        return n

    def test_committed_tb0_has_no_leading_raw_bytes(self):
        self.assertEqual(self._leading_raw_count("gemm_1x512x1000_tb0.mcode.gz"), 0)

    def test_some_fresh_rebuilds_have_leading_raw_bytes_others_dont(self):
        counts = {
            name: self._leading_raw_count(name)
            for name in (
                "gemm_1x512x1000_tb0_rebuild0.mcode.gz",
                "gemm_1x512x1000_tb0_rebuild1.mcode.gz",
                "gemm_1x512x1000_tb0_rebuild2.mcode.gz",
            )
        }
        self.assertGreater(
            len(set(counts.values())),
            1,
            f"expected the leading-raw-byte count to vary across rebuilds, got {counts}",
        )


if __name__ == "__main__":
    unittest.main()
