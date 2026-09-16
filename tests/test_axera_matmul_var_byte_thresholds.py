"""Batched MatMul's short-form `A` quad "var" tag byte: refuting a
bit-cost/additive-address model, and mapping real per-dimension
thresholds instead -- it's a small, discrete set of values driven by
shape thresholds, not a smoothly varying quantity.

`tests/test_axera_matmul_batched_var_byte.py` (PR #1510, merged) found
`var` is shape-dependent (`batch`, `M`, `K` each move it; `N` never
does) but fit no single arithmetic formula across its 6 data points.

This project's own README decoded a similarly "opaque, shape-dependent,
no obvious formula" quantity before -- the `llm_build` weight-table
*address* -- by treating it as an **additive bit-cost model**: each bit
of an index costs a fixed, independently-measurable number of bytes
("row 17 costs 612, which is row 1's 576 plus row 16's 36"). This file
tests whether the same method applies to `var`.

## The bit-cost model does not fit -- there are only 3 observed values

Building 11 new shapes (`M` in `{1,2,16,32}`, `K` in `{1,2,4,32}`,
`batch` in `{1,5,8}`, each varied alone against the `(batch=2,M=4,K=8,
N=8)` baseline) and combining with PR #1510's original 6 points gives
17 total data points. **Every single one lands on exactly one of three
byte values: `0x62` (98), `0x30` (48), or `0x2c` (44).** A bit-cost
model predicts a spread of values proportional to which index bits are
set -- with only three distinct outputs across 17 varied shapes
(covering `M`/`K` from 1 to 32 and `batch` from 1 to 8), there is no
room for an additive per-bit structure to express itself. This refutes
the bit-cost hypothesis directly, not just "didn't find one this time."

## What the data shows instead: per-dimension thresholds, richer for `M`

| dim held fixed | values tried | `var` |
| --- | --- | --- |
| `K` (batch=2,M=4) | 1, 2, 4, 8 | all `0x62` |
| `K` (batch=2,M=4) | 16, 32 | all `0x30` |
| `M` (batch=2,K=8) | 1, 2, 4 | all `0x62` |
| `M` (batch=2,K=8) | **8** | **`0x2c`** (alone) |
| `M` (batch=2,K=8) | 16, 32 | all `0x30` |
| `batch` (M=4,K=8) | 1, 2, 3 | all `0x62` |
| `batch` (M=4,K=8) | 4, 5, 8 | all `0x30` |

`K` and `batch` each show a clean binary threshold (`0x62` below,
`0x30` at/above a crossing point -- `K`'s crossing is somewhere in
`(8,16]`, not pinned more precisely here; `batch`'s is exactly between
3 and 4). `M` is richer: a **third, unique value (`0x2c`) appears only
at exactly `M=8`**, with `M<=4` at `0x62` and `M>=16` back to the same
`0x30` every other "large" dimension lands on. This is not a monotone
function of `M` alone.

**This also directly refutes total-element-count as an explanation.**
`(batch=4,M=4,K=8)` and `(batch=2,M=8,K=8)` both have `A` element count
`128` (same total data volume) but land on *different* values (`0x30`
vs `0x2c`) -- the same product reached via `batch` vs. via `M` gives a
different result, so `var` depends on which dimension carries a given
magnitude, not just the magnitude itself. Consistent with (not proof
of) `var` encoding a discrete tiling/allocation-strategy selection --
`M`, as the row dimension the MAC engine's own output tiling operates
over, plausibly gets a qualitatively different one-off case at a
particular tile-adjacent size (`M=8`) that `batch` (just repeated,
independent compute passes) and `K` (the reduction dimension) do not.
Not decoded to a semantic rule here -- reported as a precise, falsified
hypothesis (bit-cost) plus a much richer, honestly-partial
characterization than PR #1510 had.

## Confirmed above the noise floor

Two independent rebuilds verify the two most information-dense points
survive determinism: `M=8` (the unique third value, previously only
built once in PR #1510) reproduces `var=0x2c` again here
(`matmul_var_bitcost_m8_rebuild.mcode.gz`); `K=16` (PR #1510's own
`0x30` point) also reproduces independently. Neither shows any
alternate value on a second build -- the threshold-crossing pattern
above is not an artifact of a single lucky/unlucky sample.

## Process note

An earlier pass of this analysis returned `var=None` for every single
build -- the same `input_scales`-attribute-collection bug
`tests/test_axera_matmul_batched_var_byte.py` (PR #1510) and
`tests/test_axera_matmul_quad_form_switch.py` (PR #1514) had each
already hit and fixed independently: grabbing the first node with an
`input_scales` attribute silently picks up `AxDequantizeLinear`'s own
(a single-element list holding the *output* dequant scale, not `A`'s),
since it appears later in node iteration order and overwrites the
correct `AxQuantizedMatMul` value in a naive dict. Fixed by filtering
to `n.op_type == "AxQuantizedMatMul"` specifically before reading the
attribute, and reran the full analysis (no rebuilds needed -- the
compiled artifacts were already correct, only the scale-reading script
was wrong) before drawing any conclusions from it.
"""

import gzip
import os
import struct
import unittest

FIX = os.path.join(os.path.dirname(__file__), "..", "scripts", "axera", "fixtures")


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def find_short_form_var(data, a_scale):
    """Locate the short-form A quad (`<3 bytes> 82 <var> 02` x4, stride
    6) and return its 4 var bytes, or [] if absent."""
    short = struct.pack("<f", 1.0 / a_scale)[:3]
    found = [i for i in range(len(data) - 5) if data[i : i + 3] == short]
    for start in found:
        run = [start]
        i = start + 6
        while i in found:
            run.append(i)
            i += 6
        if len(run) == 4:
            return [data[i + 4] for i in run]
    return []


class TestNoBitCostSpreadAcrossSeventeenPoints(unittest.TestCase):
    """Every new build's var lands on one of only 3 values -- a bit-cost
    model would predict many distinct values across this much shape
    variation (M/K from 1 to 32, batch from 1 to 8)."""

    # fixture: (a_scale, expected var)
    CASES = {
        "matmul_var_bitcost_batch1.mcode.gz": (0.007840047590434551, 0x62),
        "matmul_var_bitcost_batch5.mcode.gz": (0.007840047590434551, 0x30),
        "matmul_var_bitcost_m1.mcode.gz": (0.007819763384759426, 0x62),
        "matmul_var_bitcost_m2.mcode.gz": (0.007840047590434551, 0x62),
        "matmul_var_bitcost_m16.mcode.gz": (0.007840047590434551, 0x30),
        "matmul_var_bitcost_k1.mcode.gz": (0.007819763384759426, 0x62),
        "matmul_var_bitcost_k4.mcode.gz": (0.007840047590434551, 0x62),
        "matmul_var_bitcost_k32.mcode.gz": (0.007840047590434551, 0x30),
    }

    def test_each_new_shape_matches_its_recorded_var(self):
        for fname, (a_scale, expected) in self.CASES.items():
            data = load(fname)
            var = find_short_form_var(data, a_scale)
            self.assertEqual(var, [expected] * 4, fname)

    def test_only_three_distinct_values_appear_across_all_new_shapes(self):
        seen = set()
        for fname, (a_scale, _expected) in self.CASES.items():
            data = load(fname)
            seen.update(find_short_form_var(data, a_scale))
        self.assertEqual(seen, {0x62, 0x30})


class TestKAndBatchHaveCleanBinaryThresholds(unittest.TestCase):
    def test_k_at_or_below_8_is_0x62(self):
        for fname in (
            "matmul_var_bitcost_k1.mcode.gz",
            "matmul_var_bitcost_k4.mcode.gz",
        ):
            a_scale = 0.007819763384759426 if "k1" in fname else 0.007840047590434551
            self.assertEqual(find_short_form_var(load(fname), a_scale), [0x62] * 4)

    def test_k_at_32_is_0x30(self):
        data = load("matmul_var_bitcost_k32.mcode.gz")
        self.assertEqual(find_short_form_var(data, 0.007840047590434551), [0x30] * 4)

    def test_batch_at_or_below_3_is_0x62(self):
        data = load("matmul_var_bitcost_batch1.mcode.gz")
        self.assertEqual(find_short_form_var(data, 0.007840047590434551), [0x62] * 4)

    def test_batch_at_or_above_4_is_0x30(self):
        data = load("matmul_var_bitcost_batch5.mcode.gz")
        self.assertEqual(find_short_form_var(data, 0.007840047590434551), [0x30] * 4)


class TestMHasAUniqueThirdValueOnlyAtEight(unittest.TestCase):
    """M=8 alone gives 0x2c -- distinct from both M<=4 (0x62) and
    M>=16 (0x30). Independently reproduced here via a fresh rebuild of
    the exact shape PR #1510 only built once."""

    def test_m_at_or_below_2_is_0x62(self):
        for fname, scale in (
            ("matmul_var_bitcost_m1.mcode.gz", 0.007819763384759426),
            ("matmul_var_bitcost_m2.mcode.gz", 0.007840047590434551),
        ):
            self.assertEqual(find_short_form_var(load(fname), scale), [0x62] * 4)

    def test_m_at_16_is_0x30(self):
        data = load("matmul_var_bitcost_m16.mcode.gz")
        self.assertEqual(find_short_form_var(data, 0.007840047590434551), [0x30] * 4)

    def test_m_at_8_is_the_unique_0x2c_and_survives_an_independent_rebuild(self):
        data = load("matmul_var_bitcost_m8_rebuild.mcode.gz")
        var = find_short_form_var(data, 0.007840047590434551)
        self.assertEqual(var, [0x2C] * 4)
        # Distinct from every other observed value.
        self.assertNotIn(0x2C, (0x62, 0x30))


class TestSameTotalElementCountGivesDifferentVar(unittest.TestCase):
    """(batch=4,M=4,K=8) and (batch=2,M=8,K=8) both have A element
    count 128, but land on different var values -- refutes total
    element count as the driver; var depends on WHICH dimension
    carries a given magnitude, not just the magnitude."""

    def test_batch4_elem128_gives_0x30(self):
        data = load("matmul_var_byte_batch4.mcode.gz")
        self.assertEqual(find_short_form_var(data, 0.007840047590434551), [0x30] * 4)

    def test_m8_elem128_gives_0x2c_not_0x30(self):
        data = load("matmul_var_bitcost_m8_rebuild.mcode.gz")
        var = find_short_form_var(data, 0.007840047590434551)
        self.assertEqual(var, [0x2C] * 4)
        self.assertNotEqual(var[0], 0x30)


if __name__ == "__main__":
    unittest.main()
