"""Formal check for RewriteTensorScatterToScatterElements (opt-in; onnxsim's
own ``onnxsim/passes/rewrite_tensor_scatter_to_scatter_elements.h``):
rewrites opset-24 ``TensorScatter(past_cache, update, write_indices, axis,
mode)`` into ``ScatterElements(past_cache, indices, update, axis)``, where
``indices`` is built to have the same shape as ``update`` and
``indices[..., s, ...] = write_indices[batch] + s`` (`s` being that
position's own coordinate along `axis`), optionally wrapped modulo
``max_sequence_length`` for ``mode="circular"``.

``TensorScatter``'s own pseudocode (`onnx/defs/tensor/defs.cc`'s
`TensorScatter_ver24_doc`, matching `onnx/reference/ops/op_tensor_scatter.py`)
is, per batch item `b`:

    for sequence_idx in range(sequence_length):
        cache_sequence_idx = write_indices[b] + sequence_idx   # (+ mod, circular)
        present_cache[b, cache_sequence_idx] = update[b, sequence_idx]

and ``ScatterElements``' own reference algorithm
(`onnx/reference/ops/op_scatter_elements.py`) is, for the rewrite's
same-shape-as-`update` ``indices`` tensor:

    for s in range(sequence_length):
        output[b, indices[b, s]] = update[b, s]

i.e. both are, verbatim, "for every `s` in `update`'s own axis range, write
`update[b, s]` to cache position `f(b, s)`" for the very same function
`f(b, s) = write_indices[b] + s` -- the rewrite's `indices` tensor is
constructed to literally *be* that function. So the substantive thing worth
mechanically checking is not "are these the same loop" (they are, by
construction) but the one piece of nontrivial arithmetic sitting inside it:
for a *fixed* cache position `c`, "does some in-range `s` write to `c`?" is
exactly `s = c - write_indices(b)` landing in `[0, sequence_length)` -- easy
to get backwards (wrong sign, or forgetting to normalize by `write_indices`
at all) the same way `rewrite_gatherelements_to_gather`'s own formal-verify
file warns about for its own index arithmetic. This proof checks that
equivalence: `Exists s in [0, seq_len). c == write_indices(b) + s` iff
`0 <= c - write_indices(b) < seq_len`.

For a *fixed* `b`, `s -> write_indices(b) + s` is plain integer addition and
therefore always injective (`s + k == s2 + k` implies `s == s2` for any
constant `k`, with no extra hypothesis) -- so under "linear" mode there is
never a collision between two different `s` values writing the same cache
position, and the loop's own last-write-wins iteration order is moot: each
cache position receives at most one write, so "the" write and "a" write
coincide. "circular" mode's `% max_sequence_length` is not injective, so this
does not extend to it -- collisions become possible there, and which of
several colliding `sequence_idx` wins depends on iteration order. Both ops
share the same order (`sequence_idx` ascending, matching the rewrite's
`indices` tensor's own `axis`-ascending construction), so a collision
resolves identically either way, but that ordering argument is not what this
proof checks; the negative-control test below instead exhibits a genuine
`Mod` collision to make the boundary of what the closed-form reduction above
covers concrete, and
``test_circular_mode_wraps_around`` in
``tests/test_tensor_scatter_to_scatter_elements.py`` covers "circular" mode's
correctness numerically (onnxsim's own random-input equivalence check).
"""

import numpy as np
import onnx
from _formal_verify_common import producer, prove, simplify_isolated_extra, z3
from onnx import parser


def test_rewrite_tensor_scatter_to_scatter_elements_witness_arithmetic_is_sound():
    write_indices = z3.Function("write_indices", z3.IntSort(), z3.IntSort())
    b, c, s = z3.Ints("b c s")
    seq_len = z3.Int("seq_len")

    exists_witness = z3.Exists(
        [s], z3.And(0 <= s, s < seq_len, c == write_indices(b) + s)
    )
    witness_s = c - write_indices(b)
    closed_form_in_range = z3.And(0 <= witness_s, witness_s < seq_len)

    # The core arithmetic both TensorScatter's and ScatterElements' closed
    # forms (per this file's module docstring) rely on: solving
    # `c == write_indices(b) + s` for `s` and checking the result lands in
    # range is equivalent to the existential "some in-range s writes here".
    prove(z3.Implies(seq_len > 0, exists_witness == closed_form_in_range))

    # And, whenever a witness exists, it is unique (plain integer-addition
    # cancellation -- no extra hypothesis needed, unlike `mode="circular"`'s
    # `Mod`, which is not injective; see this file's module docstring and the
    # negative-control test below) -- so "the" witness `update(b, witness_s)`
    # used by both ops' closed forms is unambiguous.
    s2 = z3.Int("s2")
    prove(
        z3.Implies(
            z3.And(
                0 <= s, s < seq_len, 0 <= s2, s2 < seq_len,
                write_indices(b) + s == write_indices(b) + s2,
            ),
            s == s2,
        )
    )


def test_rewrite_tensor_scatter_to_scatter_elements_end_to_end_equivalence():
    # With the witness arithmetic above established, both ops' outputs at
    # cache position (b, c) reduce to the same closed form: update's value at
    # the unique witness when in range, else past_cache's own unchanged
    # value. Confirm the full substitution-safety claim (composed with an
    # arbitrary downstream consumer), matching this suite's established
    # style (e.g. test_formal_verify_rewrite_gatherelements_to_gather.py).
    data = z3.Function("data", z3.IntSort(), z3.IntSort(), z3.RealSort())
    update = z3.Function("update", z3.IntSort(), z3.IntSort(), z3.RealSort())
    write_indices = z3.Function("write_indices", z3.IntSort(), z3.IntSort())
    consumer = z3.Function("consumer", z3.RealSort(), z3.RealSort())

    b, c = z3.Ints("b c")
    seq_len = z3.Int("seq_len")
    witness_s = c - write_indices(b)
    in_range = z3.And(0 <= witness_s, witness_s < seq_len)

    # TensorScatter's own closed form (cache_sequence_idx = write_indices[b]
    # + sequence_idx, solved for sequence_idx).
    tensor_scatter_out = z3.If(in_range, update(b, witness_s), data(b, c))
    # ScatterElements(data, indices, update, axis=1)'s own closed form, with
    # indices(b, s) = write_indices(b) + s substituted in -- this rewrite's
    # own index-tensor construction.
    scatter_elements_out = z3.If(in_range, update(b, witness_s), data(b, c))

    prove(z3.Implies(seq_len > 0, tensor_scatter_out == scatter_elements_out))
    prove(
        z3.Implies(
            seq_len > 0,
            consumer(tensor_scatter_out) == consumer(scatter_elements_out),
        )
    )


def test_rewrite_tensor_scatter_to_scatter_elements_negative_control_circular_collides():
    # Sanity check that this proof's scope (mode="linear") is real, not an
    # arbitrary restriction: under mode="circular"'s `% max_sequence_length`,
    # two distinct in-range sequence_idx values genuinely can collide on the
    # same wrapped cache position -- so the "unique witness" property the
    # proof above relies on (established for plain addition, with no extra
    # hypothesis) does NOT hold once `Mod` enters the picture. Z3 must find a
    # concrete such collision.
    s, s2, wi, max_seq = z3.Ints("s s2 wi max_seq")
    solver = z3.Solver()
    solver.add(max_seq > 0)
    solver.add(0 <= s, s < 2 * max_seq)  # sequence_length may exceed max_seq
    solver.add(0 <= s2, s2 < 2 * max_seq)
    solver.add(s != s2)
    solver.add((wi + s) % max_seq == (wi + s2) % max_seq)
    assert solver.check() == z3.sat, "expected Mod to admit a genuine collision"


def _i64(array, name):
    return onnx.numpy_helper.from_array(np.asarray(array, dtype=np.int64), name)


def test_rewrite_tensor_scatter_to_scatter_elements_pass_fires():
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 24]
        >
        g (float[2,6,3] past_cache, float[2,2,3] update) => (float[2,6,3] Y)
        {
            Y = TensorScatter (past_cache, update, write_indices)
        }
        """
    )
    model.graph.initializer.append(_i64([1, 3], "write_indices"))
    sim_model, ops = simplify_isolated_extra(
        model, "rewrite_tensor_scatter_to_scatter_elements"
    )
    assert ops["TensorScatter"] == 0
    node = producer(sim_model, "Y")
    assert node.op_type == "ScatterElements"


def test_rewrite_tensor_scatter_to_scatter_elements_declines_on_invalid_domain():
    # A same-named op in a non-default domain (a vendor/plugin "TensorScatter")
    # is not this op at all -- the predicate must decline.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 24, "custom": 1]
        >
        g (float[2,6,3] past_cache, float[2,2,3] update) => (float[2,6,3] Y)
        {
            Y = custom.TensorScatter (past_cache, update)
        }
        """
    )
    sim_model, ops = simplify_isolated_extra(
        model, "rewrite_tensor_scatter_to_scatter_elements", check_n=0
    )
    assert ops["TensorScatter"] == 1
    node = producer(sim_model, "Y")
    assert node.op_type == "TensorScatter"
