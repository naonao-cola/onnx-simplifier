"""Extending PR #1563's resource-model census: do the 21 sparse banks
and 67 sparse registers cluster by OP FAMILY, or by something else?

`tests/test_axera_resource_model_census.py` (PR #1563, merged) found
39 distinct banks (9 near-universal, 21 sparse -- present in <=9 of
306 fixtures each) and 247 distinct registers (36 universal, 67
sparse -- present in <=3 fixtures each), reporting the sparse items as
"genuinely op/shape-specific... but the sample per bank is too thin
here to say anything about what any individual one does." This file
tests the natural next hypothesis directly: does a sparse bank/register
belong to one op family (Gemm-only, Conv-only, etc.), the way a
per-op-type instruction vocabulary would predict?

## Result: NOT op-family clustering -- it's fixture SIZE/complexity

Classifying all 306 fixtures by op family from their filename prefix
(folding shape-suffixed variants like `conv_dilation3_insz13` into
`conv`, `gemm_1x512x1000_tb0` into `gemm`, etc.) gives a corpus
dominated by four routine probe families (matmul 127, conv 86, gemm
60, mul 18) plus eleven essentially one-off "real model" fixtures
(`piper_vocoder`, `resnet18_int8`, `toy_training_step`,
`w2v2fe_training_step`, `attn_qkv_softmax`, `loss_head_kd`,
`layernorm_last_axis`, `adam_update_fp32`, `dwconv_g32`, `neg_1x8`,
plus the `reshape_gather_bwd` family at 5 fixtures).

**18 of the 21 sparse banks are MULTI-family** -- they appear across
several of those one-off "real model" fixtures simultaneously (e.g.
bank `0x1f` appears in `piper_vocoder`, `resnet18_int8`,
`toy_training_step`, AND `w2v2fe_training_step` -- four completely
different op families/graphs). That is the opposite of a per-op-type
vocabulary. Only 3 sparse banks are confined to a single family, and
each of those has just 1 fixture backing it (`bank=0x60` only in
`toy_training_step`, `bank=0x82` only in `w2v2fe_training_step`,
`bank=0x85` only in `dwconv_g32`) -- too thin to call a real
per-op-family pattern rather than "this op family only has one
fixture in the corpus at all."

**The real predictor is fixture size.** Every fixture that carries at
least one sparse bank or sparse register averages **18,700-18,850
raw mcode bytes**; every fixture that carries none averages **~3,280-3,293
bytes** -- roughly a 5.7x gap, and the corpus median is 3,496 bytes,
close to the non-carrier average. Only 24 of 306 fixtures (7.8%)
carry any sparse bank/register at all, and they are overwhelmingly
the largest, most complex streams in the corpus: full real-model
builds (`piper_vocoder` 35,410 compressed bytes, `w2v2fe_training_step`
88,330, `resnet18_int8` 20,130) and the handful of large synthetic
probes deliberately built at bigger shapes within this session's own
Gemm/Conv work.

**The single cleanest piece of evidence: the SAME op family, at
different sizes, shows the pattern purely as a function of scale.**
Within Gemm alone -- `gemm_1x8x8_tb0` (2,568 bytes), `gemm_4x16x8_tb0`
(2,632), `gemm_8x8x8_tb0` (3,016) -- none carry sparse bank `0x81`.
`gemm_1x512x1000_tb0` (4,368 bytes, a real resnet18-FC-layer-sized
shape built in this session's own transB-scaling work,
`tests/test_axera_gemm_transb_scaling.py`) does. Same op, same
codebase, same build script family -- the only thing that changed is
scale. This rules out "Gemm has a special bank" and confirms "large
enough Gemm shapes start using banks small ones don't."

## What this means, honestly bounded

This reframes PR #1563's own open question: the 21 sparse banks and
67 sparse registers are not evidence of a per-op instruction
vocabulary waiting to be mapped op-by-op. They look like a SECOND
resource tier that only activates once a graph is large/complex
enough to need it -- consistent with (not proof of) something like
additional OCM tiles, additional pipeline stages, or additional
scratch registers that only get allocated past some real-model-scale
threshold. This is a real, useful reframing for a future generator:
the "core vocabulary" (9 banks, 36 registers) is what a small
single-op kernel needs; the sparse tier is what large/real graphs
additionally draw on, and its own internal structure (which bank
unlocks at what scale, in what order) is untested here -- this file
establishes the *what* (size, not op-type, predicts sparse-resource
usage) and leaves the *at what threshold, and why* fully open.

Sample-size caveat, stated plainly: only 24 of 306 fixtures carry any
sparse bank/register, and most of the "families" in that set have
exactly one fixture in the whole corpus. This is not enough to fit a
real size-threshold curve, only to establish that size (not op label)
is the right axis to look along next.
"""

import glob
import gzip
import os
import sys
import unittest
from collections import defaultdict

_AXERA_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts", "axera")
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import mcode  # noqa: E402

FIX = os.path.join(_AXERA_DIR, "fixtures")


def _op_family(name):
    base = name.replace(".mcode.gz", "")
    prefix = base.split("_")[0]
    for fam in ("conv", "gemm", "matmul", "mul"):
        if prefix.startswith(fam):
            return fam
    return prefix


def _all_fixtures():
    return sorted(glob.glob(os.path.join(FIX, "*.mcode.gz")))


def _analysis():
    fixtures = _all_fixtures()
    bank_fixtures = defaultdict(set)
    bank_families = defaultdict(set)
    reg_fixtures = defaultdict(set)
    reg_families = defaultdict(set)
    raw_len = {}
    for fx in fixtures:
        name = os.path.basename(fx)
        fam = _op_family(name)
        with gzip.open(fx, "rb") as f:
            data = f.read()
        raw_len[name] = len(data)
        recs = mcode.decode(data, **mcode.FULL_RULE)
        for r in recs:
            if r["kind"] == "V":
                bank_fixtures[r["bank"]].add(name)
                bank_families[r["bank"]].add(fam)
            elif r["kind"] in ("S", "B"):
                reg_fixtures[r["reg"]].add(name)
                reg_families[r["reg"]].add(fam)
    return {
        "n_fixtures": len(fixtures),
        "bank_fixtures": bank_fixtures,
        "bank_families": bank_families,
        "reg_fixtures": reg_fixtures,
        "reg_families": reg_families,
        "raw_len": raw_len,
    }


_A = _analysis()


class TestSparseBanksAreMostlyMultiFamily(unittest.TestCase):
    """18 of 21 sparse banks (<=9 fixtures) span multiple op families --
    the opposite of a per-op vocabulary."""

    def test_sparse_bank_count_matches_pr1563(self):
        sparse = [b for b, fs in _A["bank_fixtures"].items() if len(fs) <= 9]
        self.assertEqual(len(sparse), 21)

    def test_18_of_21_sparse_banks_are_multi_family(self):
        sparse = [b for b, fs in _A["bank_fixtures"].items() if len(fs) <= 9]
        multi = [b for b in sparse if len(_A["bank_families"][b]) > 1]
        self.assertEqual(len(multi), 18)

    def test_bank_0x1f_spans_four_unrelated_families(self):
        self.assertEqual(
            _A["bank_families"][0x1F],
            {"piper", "resnet18", "toy", "w2v2fe"},
        )


class TestSparseResourceUsagePredictedByFixtureSize(unittest.TestCase):
    """Fixtures carrying >=1 sparse bank average ~5.7x the raw mcode
    length of fixtures carrying none -- size, not op label, is the
    real predictor."""

    def test_carrier_fixtures_are_much_larger_on_average(self):
        sparse_banks = [b for b, fs in _A["bank_fixtures"].items() if len(fs) <= 9]
        carriers = set()
        for b in sparse_banks:
            carriers |= _A["bank_fixtures"][b]
        noncarriers = set(_A["raw_len"]) - carriers
        avg_carrier = sum(_A["raw_len"][n] for n in carriers) / len(carriers)
        avg_non = sum(_A["raw_len"][n] for n in noncarriers) / len(noncarriers)
        self.assertGreater(avg_carrier, 5 * avg_non)
        self.assertEqual(len(carriers), 24)

    def test_same_gemm_family_shows_the_pattern_purely_by_scale(self):
        """Within Gemm alone: small shapes never carry bank 0x81; the
        one large (real-resnet18-FC-sized) shape does. Rules out an
        op-specific explanation directly."""
        small_gemm = [
            "gemm_1x8x8_tb0.mcode.gz",
            "gemm_4x16x8_tb0.mcode.gz",
            "gemm_8x8x8_tb0.mcode.gz",
        ]
        for name in small_gemm:
            self.assertNotIn(
                name, _A["bank_fixtures"][0x81], f"{name}: should NOT carry bank 0x81"
            )
        self.assertIn(
            "gemm_1x512x1000_tb0.mcode.gz",
            _A["bank_fixtures"][0x81],
            "the large Gemm shape SHOULD carry bank 0x81",
        )


class TestSparseRegistersShowTheSamePattern(unittest.TestCase):
    def test_sparse_register_count_matches_pr1563(self):
        sparse = [r for r, fs in _A["reg_fixtures"].items() if len(fs) <= 3]
        self.assertEqual(len(sparse), 67)

    def test_carrier_fixtures_are_much_larger_on_average(self):
        sparse_regs = [r for r, fs in _A["reg_fixtures"].items() if len(fs) <= 3]
        carriers = set()
        for r in sparse_regs:
            carriers |= _A["reg_fixtures"][r]
        noncarriers = set(_A["raw_len"]) - carriers
        avg_carrier = sum(_A["raw_len"][n] for n in carriers) / len(carriers)
        avg_non = sum(_A["raw_len"][n] for n in noncarriers) / len(noncarriers)
        self.assertGreater(avg_carrier, 5 * avg_non)


if __name__ == "__main__":
    unittest.main()
