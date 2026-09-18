"""Answers a question `tests/test_axera_add_quant_fields_two_live_
inputs.py` (PR #1657) explicitly left open: PR #1657 found `Add(x1,
x2)` treats its two live inputs asymmetrically -- the "first" input
gets the same `zp_x` literal-quad (`02 10 1b <zp> 83 36`) and
`verb=161,bank=15` scale locators every other op in this arc's single
live input uses, while the "second" input only gets a zero point, at
its own distinct `reg=94,tag=132` locator, with no findable scale at
all. PR #1657's own fixtures happened to have "declared first in the
graph's input list" and "passed as the Add node's first operand"
coincide for the same tensor, so it could not tell which of those two
orderings actually determines the treatment.

## Fixtures

Four fresh `Add(a, b)` builds, all using the exact same two calibration
seeds PR #1657's own first seed pair used (`RandomState(1)` for the
tensor logically called `x1` below, `RandomState(2)` for `x2` -- `zp1
= 133`, `zp2 = 131`, matching PR #1657's own already-established
values for that seed pair, confirmed independently here), varying only
which of `x1`/`x2` is declared first in the graph's own input list vs.
which is passed as the `Add` node's own first operand:

| variant | graph inputs declared | node operands |
| --- | --- | --- |
| A | `[x1, x2]` | `Add(x1, x2)` |
| B | `[x2, x1]` | `Add(x1, x2)` |
| C | `[x1, x2]` | `Add(x2, x1)` |
| D | `[x2, x1]` | `Add(x2, x1)` |

A and D have both orderings agree (mirror images of each other); B and
C each flip ONE of the two orderings independently, disentangling them.

## Finding: NODE OPERAND ORDER determines the treatment, not graph
## input declaration order -- confirmed unambiguously across all 4
## variants

| variant | literal-quad locator carries | `reg=94,tag=132` carries |
| --- | --- | --- |
| A (`in=[x1,x2]`, `op=(x1,x2)`) | `zp1`=133 | `zp2`=131 |
| B (`in=[x2,x1]`, `op=(x1,x2)`) | `zp1`=133 | `zp2`=131 |
| C (`in=[x1,x2]`, `op=(x2,x1)`) | `zp2`=131 | `zp1`=133 |
| D (`in=[x2,x1]`, `op=(x2,x1)`) | `zp2`=131 | `zp1`=133 |

Flipping the GRAPH INPUT LIST order alone (A -> B) changes nothing --
the literal-quad locator still carries `x1`'s own value in both. Flipping
the NODE OPERAND order alone (A -> C) flips which tensor's value the
literal-quad locator carries. A and B (same operand order `(x1,x2)`,
different declared-input order) agree with each other exactly; C and D
(same operand order `(x2,x1)`, different declared-input order) likewise
agree with each other; A and C (same declared-input order, different
operand order) disagree. The pattern is unambiguous: whichever tensor is
the `Add` node's own FIRST operand gets the literal-quad `zp_x`/
`verb=161,bank=15`-scale treatment; whichever is the SECOND operand gets
the `reg=94,tag=132`-only treatment. Graph input declaration order has
no effect on either locator's own content, confirmed by two independent
A/B and C/D pairs, not merely one comparison.

Every one of the 4 fixtures decodes with `mcode.check()` reporting zero
errors, and the literal-quad/`reg=94` hit counts are both exactly 1 per
fixture (checked directly, not assumed) -- no fixture shows both
locators carrying the same tensor's value, or either locator missing.

One caveat found while writing this file: graph-input declaration
order is not entirely inert -- the literal quad's own BYTE OFFSET shifts
by 4 bytes between A (offset 1092) and B (offset 1096), even though
both carry the same tensor's (`x1`'s) value. Declaring `x2` first in
the graph's own input list does perturb the stream's early layout
slightly; it just never changes WHICH tensor's value either locator
carries, which is the property this file actually tests.

## What this establishes, precisely, and what it does not

**Established**: the `Add`-input asymmetry PR #1657 found (one input
gets a full `zp_x`+scale locator, the other gets zero-point-only) is
governed by the `Add` ONNX node's own operand order, not by the order
tensors are declared in the graph's input list -- confirmed by two
independent order-flip comparisons (A vs. B for graph-input order
alone, A vs. C for operand order alone), each isolating one axis while
holding the other fixed.

**NOT established**: whether this generalizes to `Sub`/`Mul`/`Div`
with two live operands (PR #1657's own open item, not re-tested here);
whether it holds at shapes other than `(1,16)`; whether Pulsar2's own
internal graph representation genuinely preserves ONNX node operand
order verbatim, or whether some other equivalent property (e.g. tensor
name lexical order, which happens to coincide with operand order in
every one of this file's own fixtures since `x1 < x2` lexically in all
4 variants) is the true underlying driver -- not distinguished here,
since this file never tested a case where operand order and tensor-name
lexical order disagree.
"""

import gzip
import math
import os
import sys
import unittest

import numpy as np

_AXERA_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts", "axera")
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import mcode  # noqa: E402

FIX = os.path.join(_AXERA_DIR, "fixtures")

SEED1, SEED2 = 1, 2

VARIANTS = {
    "A_in12_op12": "x1",
    "B_in21_op12": "x1",
    "C_in12_op21": "x2",
    "D_in21_op21": "x2",
}


def load(variant):
    path = os.path.join(FIX, f"add_operand_order_{variant}.mcode.gz")
    with gzip.open(path, "rb") as f:
        return f.read()


def decode(variant):
    return mcode.decode(load(variant), **mcode.FULL_RULE)


def calib_samples(seed, shape=(1, 16), n_samples=4):
    """`RandomState(seed).randn(1, 16)` x4 -- the exact calibration
    data this file's own fixture-build script used, matching PR
    #1657's own first seed pair (1, 2) exactly so `zp1`/`zp2` are
    directly comparable to its own already-established values."""
    rng = np.random.RandomState(seed)
    return [rng.randn(*shape).astype(np.float32) for _ in range(n_samples)]


def asymmetric_uint8_quant_params(samples):
    """`ComputeAsymmetricUint8QuantParams` (`onnxsim/passes/static_
    quantize_matmul.h`), the plain float32-throughout formula this
    whole arc has used since PR #1636/#1640."""
    lo = np.float32(0.0)
    hi = np.float32(0.0)
    for s in samples:
        lo = min(lo, np.float32(s.min()))
        hi = max(hi, np.float32(s.max()))
    if hi <= lo:
        hi = lo + np.float32(1.0)
    scale = (hi - lo) / np.float32(255.0)
    zp_f = float(np.float32(-lo / scale))
    zp = math.floor(zp_f + 0.5) if zp_f >= 0 else math.ceil(zp_f - 0.5)
    return max(0, min(255, zp)), zp_f, float(scale)


ZP1, _, _ = asymmetric_uint8_quant_params(calib_samples(SEED1))
ZP2, _, _ = asymmetric_uint8_quant_params(calib_samples(SEED2))

ZP_BY_TENSOR = {"x1": ZP1, "x2": ZP2}


def literal_quad_hits(data, zp):
    quad = bytes.fromhex("02101b") + bytes([zp]) + bytes.fromhex("8336")
    return [i for i in range(len(data) - 5) if data[i : i + 6] == quad]


def reg94_hits(recs, zp):
    return [
        r
        for r in recs
        if r.get("kind") == "S"
        and r.get("reg") == 94
        and r.get("tag") == 132
        and r["payload"][-1] == zp
    ]


class TestFixturesDecodeCleanly(unittest.TestCase):
    def test_no_hard_decode_errors_on_any_variant(self):
        for variant in VARIANTS:
            errs = mcode.check(load(variant))
            self.assertEqual(errs, [], variant)


class TestZp1AndZp2MatchPR1657sOwnAlreadyEstablishedValues(unittest.TestCase):
    def test_zp1_is_133_and_zp2_is_131(self):
        # Cited directly from PR #1657's own seed pair (1, 2) results --
        # recomputed independently here, not copied, to confirm this
        # file's own calibration convention is identical.
        self.assertEqual(ZP1, 133)
        self.assertEqual(ZP2, 131)


class TestNodeOperandOrderDeterminesTheLocator(unittest.TestCase):
    """The core finding: whichever tensor is the Add node's own FIRST
    operand gets the literal-quad locator; the other gets reg=94 --
    regardless of graph input declaration order."""

    def test_first_operand_tensor_carries_the_literal_quad_exactly_once(self):
        for variant, first_operand in VARIANTS.items():
            data = load(variant)
            zp_first = ZP_BY_TENSOR[first_operand]
            hits = literal_quad_hits(data, zp_first)
            self.assertEqual(len(hits), 1, (variant, first_operand, zp_first))

    def test_second_operand_tensor_carries_reg94_exactly_once(self):
        for variant, first_operand in VARIANTS.items():
            second_operand = "x2" if first_operand == "x1" else "x1"
            recs = decode(variant)
            zp_second = ZP_BY_TENSOR[second_operand]
            hits = reg94_hits(recs, zp_second)
            self.assertEqual(len(hits), 1, (variant, second_operand, zp_second))

    def test_first_operand_tensor_does_not_also_appear_at_reg94(self):
        """Confirms the two locators are mutually exclusive per tensor,
        not that reg=94 just always fires regardless of which tensor's
        value is queried."""
        for variant, first_operand in VARIANTS.items():
            recs = decode(variant)
            zp_first = ZP_BY_TENSOR[first_operand]
            hits = reg94_hits(recs, zp_first)
            self.assertEqual(hits, [], (variant, first_operand))

    def test_second_operand_tensor_does_not_also_appear_in_the_literal_quad(self):
        for variant, first_operand in VARIANTS.items():
            second_operand = "x2" if first_operand == "x1" else "x1"
            data = load(variant)
            zp_second = ZP_BY_TENSOR[second_operand]
            hits = literal_quad_hits(data, zp_second)
            self.assertEqual(hits, [], (variant, second_operand))


class TestGraphInputDeclarationOrderHasNoEffect(unittest.TestCase):
    """A vs. B (and C vs. D) share the same node-operand order but
    differ in graph-input-declaration order -- both locators' own
    content should be identical within each pair."""

    def test_variant_a_and_b_agree_on_which_tensor_each_locator_carries(self):
        # Content agrees (each locator still carries the same tensor's
        # own value); the literal quad's own BYTE OFFSET is not claimed
        # to agree -- it shifts by 4 bytes between A and B, since
        # declaring x2 first in the graph's own input list does perturb
        # the stream's early byte layout, just not which physical
        # tensor either locator's own content refers to.
        data_a, data_b = load("A_in12_op12"), load("B_in21_op12")
        self.assertEqual(len(literal_quad_hits(data_a, ZP1)), 1)
        self.assertEqual(len(literal_quad_hits(data_b, ZP1)), 1)
        recs_a, recs_b = decode("A_in12_op12"), decode("B_in21_op12")
        self.assertEqual(
            [r["payload"][-1] for r in reg94_hits(recs_a, ZP2)],
            [r["payload"][-1] for r in reg94_hits(recs_b, ZP2)],
        )

    def test_variant_c_and_d_agree_on_which_tensor_each_locator_carries(self):
        data_c, data_d = load("C_in12_op21"), load("D_in21_op21")
        self.assertEqual(len(literal_quad_hits(data_c, ZP2)), 1)
        self.assertEqual(len(literal_quad_hits(data_d, ZP2)), 1)
        recs_c, recs_d = decode("C_in12_op21"), decode("D_in21_op21")
        self.assertEqual(
            [r["payload"][-1] for r in reg94_hits(recs_c, ZP1)],
            [r["payload"][-1] for r in reg94_hits(recs_d, ZP1)],
        )


if __name__ == "__main__":
    unittest.main()
