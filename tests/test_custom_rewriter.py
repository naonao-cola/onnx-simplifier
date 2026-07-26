"""Tests for the ``custom_rewriter`` hook of ``onnxsim.simplify``.

``custom_rewriter`` lets a Python callable rewrite the model from inside
onnxsim's simplification fixed point, interleaved with the built-in optimizer,
shape inference and constant folding. These tests exercise the hook with a
plain-Python rewriter (no extra dependency) and, when available, with an
``onnxscript.rewriter`` rule set -- the intended real-world use case.
"""

import collections

import numpy as np
import onnx
import pytest

import onnxsim


def _model(nodes, inputs, outputs, initializer, opset=13):
    graph = onnx.helper.make_graph(nodes, "g", inputs, outputs, initializer)
    return onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", opset)], ir_version=10
    )


def _vi(name, shape):
    return onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _matmul_add_model():
    """x @ W + B, i.e. a MatMul feeding an Add -- the classic Gemm fusion."""
    nodes = [
        onnx.helper.make_node("MatMul", ["x", "W"], ["mm"], name="mm"),
        onnx.helper.make_node("Add", ["mm", "B"], ["y"], name="add"),
    ]
    inits = [
        _f32(np.random.randn(4, 5), "W"),
        _f32(np.random.randn(5), "B"),
    ]
    return _model(
        nodes,
        [_vi("x", [3, 4])],
        [_vi("y", [3, 5])],
        inits,
    )


def _op_types(model):
    return collections.Counter(n.op_type for n in model.graph.node)


def test_custom_rewriter_runs_and_rewrites():
    """A plain-Python rewriter that renames every Relu to a Sigmoid is applied.

    Sigmoid is chosen because, like Relu, it is a single-input/single-output
    elementwise op valid in opset 13, so the rewritten model stays schema-valid
    (unlike, say, Gelu, which is only registered from opset 20).
    """
    model = _model(
        [onnx.helper.make_node("Relu", ["x"], ["y"], name="r")],
        [_vi("x", [2, 2])],
        [_vi("y", [2, 2])],
        [],
    )

    def rewrite(m: onnx.ModelProto) -> onnx.ModelProto:
        for node in m.graph.node:
            if node.op_type == "Relu":
                node.op_type = "Sigmoid"
        return m

    sim_model, _ = onnxsim.simplify(model, check_n=0, custom_rewriter=rewrite)
    assert _op_types(sim_model)["Sigmoid"] == 1
    assert _op_types(sim_model)["Relu"] == 0


def test_custom_rewriter_in_place_none_return():
    """A rewriter that mutates in place and returns ``None`` is supported."""
    model = _model(
        [onnx.helper.make_node("Relu", ["x"], ["y"], name="r")],
        [_vi("x", [2, 2])],
        [_vi("y", [2, 2])],
        [],
    )

    def rewrite(m: onnx.ModelProto) -> None:
        for node in m.graph.node:
            if node.op_type == "Relu":
                node.op_type = "Sigmoid"
        # no return

    sim_model, _ = onnxsim.simplify(model, check_n=0, custom_rewriter=rewrite)
    assert _op_types(sim_model)["Sigmoid"] == 1


def test_custom_rewriter_none_is_noop():
    """Passing no rewriter leaves behaviour identical to a plain simplify."""
    model = _matmul_add_model()
    a, _ = onnxsim.simplify(model, check_n=3)
    b, _ = onnxsim.simplify(model, check_n=3, custom_rewriter=None)
    assert _op_types(a) == _op_types(b)


def test_custom_rewriter_called():
    """The hook is actually invoked at least once."""
    model = _matmul_add_model()
    calls = {"n": 0}

    def rewrite(m: onnx.ModelProto) -> onnx.ModelProto:
        calls["n"] += 1
        return m

    onnxsim.simplify(model, check_n=0, custom_rewriter=rewrite)
    assert calls["n"] >= 1


def test_custom_rewriter_false_signals_no_change():
    """Returning ``False`` reports "nothing rewritten" and leaves the model as-is.

    A rewriter that changes the model on its first round and then reports no
    further change with ``False`` must still keep that change: the ``False``
    sentinel only skips copying an *unchanged* model back, it never discards a
    rewrite. It also must not loop forever -- ``False`` lets the fixed point
    converge.
    """
    model = _model(
        [onnx.helper.make_node("Relu", ["x"], ["y"], name="r")],
        [_vi("x", [2, 2])],
        [_vi("y", [2, 2])],
        [],
    )
    calls = {"n": 0}

    def rewrite(m: onnx.ModelProto):
        calls["n"] += 1
        relus = [n for n in m.graph.node if n.op_type == "Relu"]
        if not relus:
            # Already rewritten: report no change so onnxsim skips the copy.
            return False
        for node in relus:
            node.op_type = "Sigmoid"
        return m

    sim_model, _ = onnxsim.simplify(model, check_n=0, custom_rewriter=rewrite)
    counts = _op_types(sim_model)
    # The rewrite is kept: returning ``False`` on a later round never discards
    # the earlier Relu -> Sigmoid change.
    assert counts["Sigmoid"] == 1 and counts["Relu"] == 0, counts
    # The rewriter ran. (Once the optimizer settles on the rewritten model the
    # fixed point can converge without another rewrite call, so the exact count
    # is not asserted -- the no-change ``False`` path itself is covered by
    # test_custom_rewriter_false_from_the_start_is_noop and
    # test_custom_rewriter_onnxscript_count_skips_copy.)
    assert calls["n"] >= 1


def test_custom_rewriter_false_from_the_start_is_noop():
    """A rewriter that only ever returns ``False`` never changes the model."""
    model = _matmul_add_model()
    baseline, _ = onnxsim.simplify(model, check_n=0)

    def rewrite(m: onnx.ModelProto) -> bool:
        return False

    sim_model, _ = onnxsim.simplify(model, check_n=0, custom_rewriter=rewrite)
    assert _op_types(sim_model) == _op_types(baseline)


def test_custom_rewriter_with_onnxscript():
    """Real-world case: register an onnxscript.rewriter rule set as the pass."""
    rewriter = pytest.importorskip("onnxscript.rewriter")
    from onnxscript.rewriter import pattern

    def _pat(op, x, w, b):
        return op.Add(op.MatMul(x, w), b)

    def _rep(op, x, w, b):
        return op.Gemm(x, w, b)

    rules = pattern.RewriteRuleSet([pattern.RewriteRule(_pat, _rep)])

    model = _matmul_add_model()

    def rewrite(m: onnx.ModelProto) -> onnx.ModelProto:
        return rewriter.rewrite(m, pattern_rewrite_rules=rules)

    sim_model, check_ok = onnxsim.simplify(model, check_n=3, custom_rewriter=rewrite)
    counts = _op_types(sim_model)
    assert counts["Gemm"] == 1, counts
    assert counts["MatMul"] == 0 and counts["Add"] == 0, counts
    assert check_ok


def test_custom_rewriter_onnxscript_passresult_skips_copy():
    """The documented onnxscript pattern: run the rules through onnx-ir's
    ``PassManager`` and return ``False`` when the resulting ``PassResult`` reports
    the model was not modified, so onnxsim skips the copy -- while still applying
    the fusion when a rule fires."""
    pytest.importorskip("onnxscript.rewriter")
    from onnxscript import ir
    from onnxscript.rewriter import RewritePass, pattern

    def _pat(op, x, w, b):
        return op.Add(op.MatMul(x, w), b)

    def _rep(op, x, w, b):
        return op.Gemm(x, w, b)

    rules = pattern.RewriteRuleSet([pattern.RewriteRule(_pat, _rep)])
    rewrite_pass = ir.passes.PassManager([RewritePass(rules)])

    model = _matmul_add_model()
    modified_seen = []

    def rewrite(m: onnx.ModelProto):
        model_ir = ir.serde.deserialize_model(m)
        result = rewrite_pass(model_ir)
        modified_seen.append(result.modified)
        if not result.modified:
            return False  # no rule fired: skip the copy
        return ir.serde.serialize_model(result.model)

    sim_model, check_ok = onnxsim.simplify(model, check_n=3, custom_rewriter=rewrite)
    counts = _op_types(sim_model)
    assert counts["Gemm"] == 1, counts
    assert counts["MatMul"] == 0 and counts["Add"] == 0, counts
    assert check_ok
    # The rewriter ran, and the final round reported the model was not modified
    # and returned ``False`` -- exercising the no-copy path that converges the
    # fixed point. (The fusion itself may come from this rule or from the
    # built-in optimizer, whichever fires first; either way the last round
    # rewrites nothing.)
    assert modified_seen and not modified_seen[-1]
