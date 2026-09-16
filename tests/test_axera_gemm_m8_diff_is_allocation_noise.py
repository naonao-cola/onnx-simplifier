"""The Gemm M=8 transB diff, closed: register-allocation reordering, not
a new field.

`test_axera_gemm_output_quad.py`'s `TestM8DiffersStructurally` located
this (offsets 682-698, denser than the clean single-quad ULP flip seen
at M=1/M=4) but didn't decode it, flagging it as a real, reproducible,
still-open lead. Running it through `mcode.decode()` (see
`scripts/axera/mcode.py`) instead of reading raw bytes resolves it: it
is a run of six short S-units -- the same `[p][payload][tag][reg]` short-
unit grammar this project's earlier zero-point work (see
`tests/test_axera_mcode_reciprocal.py`'s zp_x sections) already
established for an unrelated op -- and both builds carry the *identical
set* of records (same count, same total byte length, same payload
values `0x10`/`0x20`/`0x30`/`0x40`-ish), just assigned to different
register numbers:

```
tb0: reg102(0x12) reg104(0x22) reg106(23 00 20) reg8(30) reg8(23 00 10) reg8(23 00 40)
tb1: reg102(0x12) reg104(0x22) reg106(23 00 40) reg108(23 00 20)         reg8(30) reg8(23 00 10)
```

`reg106`'s payload and one of `reg8`'s three payloads simply trade
places (`23 00 20` and `23 00 40` swap which register holds them, and a
third `reg8` entry becomes a new `reg108` entry instead) -- the same
"ordering bytes"/register-allocation class of noise this project has
already named in `scripts/axera/README.md` (the Conv weight-dependence
section's "six single bytes... the ordering bytes noted earlier"), not
a new numeric field encoding something about `transB` or `M=8`
specifically. This closes the M=8 lead as understood-but-uninteresting
rather than leaving it open: the *location* is now decoded (a small
per-shape setup table of short S-units), even though *why* the
allocator orders it differently for `transB=0` vs `transB=1` at this
particular M is not -- consistent with this project's broader,
repeated finding that register/order assignment in this compiler is
allocator output, not a rule waiting to be found.
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


class TestM8DiffIsAllocationNoise(unittest.TestCase):
    def test_same_record_count_and_length_different_register_assignment(self):
        d0 = load("gemm_8x8x8_tb0.mcode.gz")
        d1 = load("gemm_8x8x8_tb1.mcode.gz")
        self.assertEqual(len(d0), len(d1))

        recs0 = mcode.decode(d0, start=671, end=701)
        recs1 = mcode.decode(d1, start=671, end=701)
        self.assertEqual(len(recs0), 6, "tb0: expected 6 short units in this window")
        self.assertEqual(len(recs1), 6, "tb1: expected 6 short units in this window")

        payloads0 = sorted(r["payload"] for r in recs0)
        payloads1 = sorted(r["payload"] for r in recs1)
        self.assertEqual(
            payloads0,
            payloads1,
            "the same set of payload values should appear in both builds,"
            " just on different registers",
        )

        regs0 = [r["reg"] for r in recs0]
        regs1 = [r["reg"] for r in recs1]
        self.assertNotEqual(
            regs0, regs1, "expected the register assignment itself to differ"
        )


if __name__ == "__main__":
    unittest.main()
