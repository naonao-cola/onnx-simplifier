"""Formal check for RewriteScatterElementsToScatterND (opt-in; onnxsim's own
``onnxsim/passes/rewrite_scatterelements_to_scatternd.h``): rewrites
``ScatterElements(data, indices, updates, axis=0)`` into
``ScatterND(data, rows, updates)`` with ``rows`` of shape ``[N, 1]``, when
``indices`` provably does not vary along any axis but 0.

Both ops are "for every position of ``updates``, write that value to one
position of the output"; they agree iff they send every ``updates``
position to the same output position. For update position ``(i0, i1)``
(``i1`` standing for all trailing coordinates at once):

  ScatterElements, axis 0:  (norm(indices(i0, i1)), i1)
  ScatterND, [N,1] rows:    (norm(rows(i0)), i1)   -- whole trailing slice

where ``norm`` is the shared negative-index convention
(``v < 0 -> v + D0``, ``D0`` being ``data``'s axis-0 size). With the
invariance hypothesis ``indices(i0, i1) == indices(i0, 0)`` and
``rows(i0) == indices(i0, 0)`` (what the pass builds), the two target maps
are equal, so both ops perform the same set of writes. Duplicate targets are
forbidden by both specs under ``reduction="none"``, and a duplicate row in
the ScatterND form is exactly a duplicate element in the ScatterElements
form, so both are defined on the same inputs. The negative control shows
that without invariance Z3 finds a counterexample. The shape side condition
(``updates.shape[1:] == data.shape[1:]``, so ScatterND's whole-slice writes
cover exactly the positions ScatterElements writes) is enforced by the
pass's predicate and exercised in ``tests/test_scatterelements_to_scatternd.py``.
"""

import numpy as np
import onnx
from _formal_verify_common import producer, prove, simplify_isolated_extra, z3
from onnx import parser

PASS = "rewrite_scatterelements_to_scatternd"


def _norm(v, d0):
    return z3.If(v < 0, v + d0, v)


def test_rewrite_scatterelements_to_scatternd_is_sound():
    indices = z3.Function("indices", z3.IntSort(), z3.IntSort(), z3.IntSort())
    rows = z3.Function("rows", z3.IntSort(), z3.IntSort())
    d0 = z3.Int("d0")
    i0, i1, i1p = z3.Ints("i0 i1 i1p")

    invariance = z3.ForAll([i0, i1, i1p], indices(i0, i1) == indices(i0, i1p))
    rows_def = z3.ForAll([i0], rows(i0) == indices(i0, 0))
    hypotheses = z3.And(invariance, rows_def)

    se_target = (_norm(indices(i0, i1), d0), i1)
    nd_target = (_norm(rows(i0), d0), i1)
    prove(
        z3.Implies(
            hypotheses,
            z3.And(se_target[0] == nd_target[0], se_target[1] == nd_target[1]),
        )
    )


def test_rewrite_scatterelements_to_scatternd_negative_control():
    indices = z3.Function("indices", z3.IntSort(), z3.IntSort(), z3.IntSort())
    rows = z3.Function("rows", z3.IntSort(), z3.IntSort())
    d0 = z3.Int("d0")
    i0, i1 = z3.Ints("i0 i1")
    rows_def = z3.ForAll([i0], rows(i0) == indices(i0, 0))
    claim = z3.ForAll([i0, i1], _norm(indices(i0, i1), d0) == _norm(rows(i0), d0))
    solver = z3.Solver()
    solver.add(rows_def)
    solver.add(z3.Not(claim))
    assert solver.check() == z3.sat


def test_pass_fires_on_invariant_constant_indices():
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 13]>
        g (float[4,3] data, float[2,3] updates) => (float[4,3] Y)
        {
            Y = ScatterElements<axis = 0>(data, indices, updates)
        }
        """
    )
    model.graph.initializer.append(
        onnx.numpy_helper.from_array(
            np.array([[3, 3, 3], [-4, -4, -4]], np.int64), "indices"
        )
    )
    sim_model, ops = simplify_isolated_extra(model, PASS)
    assert ops["ScatterElements"] == 0
    node = producer(sim_model, "Y")
    assert node.op_type == "ScatterND"
    rows = next(i for i in sim_model.graph.initializer if i.name == node.input[1])
    assert onnx.numpy_helper.to_array(rows).tolist() == [[3], [-4]]


def test_pass_declines_on_varying_constant_indices():
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: ["": 13]>
        g (float[4,3] data, float[2,3] updates) => (float[4,3] Y)
        {
            Y = ScatterElements<axis = 0>(data, indices, updates)
        }
        """
    )
    model.graph.initializer.append(
        onnx.numpy_helper.from_array(
            np.array([[3, 0, 3], [1, 1, 1]], np.int64), "indices"
        )
    )
    sim_model, ops = simplify_isolated_extra(model, PASS)
    assert ops["ScatterElements"] == 1
    assert producer(sim_model, "Y").op_type == "ScatterElements"
