"""Compile an onnxscript ``RewriteRule``'s ``pattern``/``rewrite`` methods to
``onnx.FunctionProto`` by reusing onnxscript's ``@script`` converter, then drive
the result through onnxsim's ``function_rewrite_rules``.

The pattern/rewrite of an onnxscript rewrite rule are Python functions of the
form ``fn(op, *inputs)`` (or bound methods ``fn(self, op, *inputs)``), where
``op`` is the *pattern builder*. onnxscript's ``@script`` decorator already
compiles op-authoring functions to FunctionProtos, but it resolves ``op`` from
the *module globals* and AST-parses real source, so you can't decorate a rule
method directly. ``_compile_fn_to_function_proto`` below reuses the same
converter machinery (``onnxscript._internal.main.script_check``): it strips
``self``/``op`` from the signature, binds ``op`` to an authoring opset, and runs
the checker on the resulting AST.

**Caveats.** This uses onnxscript *internal* APIs (guarded -- the module skips if
they move) and only handles the *structural* subset: plain op-call graphs. A
method that computes attributes or constants from the match -- e.g.
``MatMulAddToGemm.rewrite``'s ``if self.trans_a: attrs["transA"]=1;
op.Gemm(..., **attrs)`` or ``ReshapeReshape.rewrite``'s
``op.initializer(ir.Tensor(self._new_shape, ...))`` -- is *not* compilable this
way and raises. In practice the ``pattern`` side of a rule is usually structural
(so it compiles), while the ``rewrite`` side often is not, so you pair a
compiled pattern with a simple replacement function. Some rules do have a
structural rewrite too -- e.g. ``FuseSuccessiveRelu`` (``Relu(Relu(x)) ->
Relu(x)``), whose rewrite is a plain ``op.Relu(x)`` -- and compile end to end
from their own methods. This matches onnxsim's own matcher limits.
"""

import ast
import collections
import inspect
import textwrap

import numpy as np
import onnx
import pytest
from onnx import helper

import onnxsim

# Internal onnxscript machinery this helper stands on. Skip the whole module if
# onnxscript (or the private entry point) is unavailable.
_oxs_main = pytest.importorskip("onnxscript._internal.main")
_oxs_values = pytest.importorskip("onnxscript.values")
_oxs_opsets = pytest.importorskip("onnxscript.onnx_opset")
common = pytest.importorskip("onnxscript.rewriter.rules.common")
_rewriter = pytest.importorskip("onnxscript.rewriter")


class _StripAnnotations(ast.NodeTransformer):
    """Drop parameter/return annotations. Rewrite-rule methods annotate their
    tensor parameters ``x: ir.Value`` (onnxscript's rewriter typing), which the
    ``@script`` converter rejects; the annotations carry no meaning for a
    structural op graph, so remove them."""

    def visit_arg(self, node):
        node.annotation = None
        return node

    def visit_FunctionDef(self, node):
        node.returns = None
        self.generic_visit(node)
        return node


def _compile_fn_to_function_proto(fn, *, opset_version=18, name=None):
    """Compile ``fn(op, *inputs)`` / ``fn(self, op, *inputs)`` to a FunctionProto.

    ``op`` is bound to the ``opset_version`` authoring opset; the remaining
    parameters become the function inputs (a Python-typed attribute parameter,
    e.g. ``alpha: float``, becomes a ``ref_attr_name`` = onnxsim's ``@name``
    wildcard). Raises whatever onnxscript raises when ``fn`` falls outside the
    op-authoring subset.
    """
    op_builder = getattr(_oxs_opsets, f"opset{opset_version}")
    src = textwrap.dedent(inspect.getsource(fn))
    fdef = ast.parse(src).body[0]
    fdef.decorator_list = []
    fdef.args.args = [a for a in fdef.args.args if a.arg not in ("self", "op")]
    _StripAnnotations().visit(fdef)
    ast.fix_missing_locations(fdef)
    if name is not None:
        fdef.name = name
    env = {**getattr(inspect.getmodule(fn), "__dict__", {}), "op": op_builder}
    ir = _oxs_main.script_check(
        fdef, _oxs_values.Opset("this", 1), env, src, default_opset=op_builder
    )
    return ir.to_function_proto()


# Self-test at import: if onnxscript's internal compile path has drifted, skip
# the whole module rather than fail with a confusing internal error.
def _selftest_identity(op, x):
    return op.Identity(x)


try:
    _probe = _compile_fn_to_function_proto(_selftest_identity, name="probe")
    assert _probe.node[0].op_type == "Identity"
except Exception as exc:  # pragma: no cover - environment/version guard
    pytest.skip(
        f"onnxscript @script internals unavailable/changed: {exc!r}",
        allow_module_level=True,
    )


def _op_types(model):
    return collections.Counter(n.op_type for n in model.graph.node)


def _apply_onnxscript_rule(model, rule):
    rules = _rewriter.pattern.RewriteRuleSet([rule])
    return _rewriter.rewrite(model, pattern_rewrite_rules=rules)


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _vi(name, shape):
    return helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)


# --------------------------------------------------------------------------
# Compile a real rule's `pattern` method + a simple replacement, and run it.
# --------------------------------------------------------------------------
def _gemm_replacement(op, input_a, input_b, input_c):
    return op.Gemm(input_a, input_b, input_c)


def test_compiled_matmul_add_pattern_from_real_rule():
    from onnxscript.rewriter.rules.common._matmul_add_to_gemm import MatMulAddToGemm

    pattern_fp = _compile_fn_to_function_proto(MatMulAddToGemm.pattern, name="pat")
    replacement_fp = _compile_fn_to_function_proto(_gemm_replacement, name="rep")
    # The pattern really came from the onnxscript rule.
    assert [n.op_type for n in pattern_fp.node] == ["MatMul", "Add"]

    def build():
        nodes = [
            helper.make_node("MatMul", ["x", "W"], ["mm"]),
            helper.make_node("Add", ["mm", "B"], ["y"]),
        ]
        inits = [_f32(np.random.randn(4, 5), "W"), _f32(np.random.randn(5), "B")]
        graph = helper.make_graph(
            nodes, "g", [_vi("x", [3, 4])], [_vi("y", [3, 5])], inits
        )
        return helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10
        )

    ours, check_ok = onnxsim.simplify(
        build(),
        check_n=2,
        skipped_optimizers=["fuse_matmul_add_bias_into_gemm"],
        function_rewrite_rules=[(pattern_fp, replacement_fp)],
    )
    counts = _op_types(ours)
    assert counts["Gemm"] == 1 and counts["MatMul"] == 0 and counts["Add"] == 0, counts
    assert check_ok

    theirs = _apply_onnxscript_rule(build(), common.matmul_add_to_gemm_rule)
    tcounts = _op_types(theirs)
    assert tcounts["Gemm"] == 1 and tcounts["MatMul"] == 0 and tcounts["Add"] == 0


def _reshape_replacement(op, x, shape):
    return op.Reshape(x, shape)


def test_compiled_reshape_reshape_pattern_from_real_rule():
    from onnxscript.rewriter.rules.common._basic_rules import ReshapeReshape

    pattern_fp = _compile_fn_to_function_proto(ReshapeReshape.pattern, name="pat")
    replacement_fp = _compile_fn_to_function_proto(_reshape_replacement, name="rep")
    assert [n.op_type for n in pattern_fp.node] == ["Reshape", "Reshape"]

    def build():
        nodes = [
            helper.make_node("Reshape", ["x", "s1"], ["t"]),
            helper.make_node("Reshape", ["t", "s2"], ["y"]),
        ]
        inits = [
            onnx.numpy_helper.from_array(np.array([2, 12], dtype=np.int64), "s1"),
            onnx.numpy_helper.from_array(np.array([4, 6], dtype=np.int64), "s2"),
        ]
        graph = helper.make_graph(
            nodes, "g", [_vi("x", [3, 8])], [_vi("y", [4, 6])], inits
        )
        return helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10
        )

    ours, check_ok = onnxsim.simplify(
        build(), check_n=2, function_rewrite_rules=[(pattern_fp, replacement_fp)]
    )
    assert _op_types(ours)["Reshape"] == 1, _op_types(ours)
    assert check_ok

    theirs = _apply_onnxscript_rule(build(), common.reshape_reshape_rule)
    assert _op_types(theirs)["Reshape"] == 1, _op_types(theirs)


def test_compiled_successive_relu_both_from_real_rule():
    """Some rules have *both* a structural pattern and a structural rewrite --
    ``FuseSuccessiveRelu`` (``Relu(Relu(x)) -> Relu(x)``), whose rewrite is a
    plain ``op.Relu(x)`` with no match-derived attributes -- so the whole rule
    compiles straight from its own methods."""
    from onnxscript.rewriter.rules.common._fuse_relus_clips import FuseSuccessiveRelu

    pattern_fp = _compile_fn_to_function_proto(FuseSuccessiveRelu.pattern, name="pat")
    replacement_fp = _compile_fn_to_function_proto(
        FuseSuccessiveRelu.rewrite, name="rep"
    )
    assert [n.op_type for n in pattern_fp.node] == ["Relu", "Relu"]
    assert [n.op_type for n in replacement_fp.node] == ["Relu"]

    def build():
        nodes = [
            helper.make_node("Relu", ["x"], ["t"]),
            helper.make_node("Relu", ["t"], ["y"]),
        ]
        graph = helper.make_graph(
            nodes, "g", [_vi("x", [2, 2])], [_vi("y", [2, 2])], []
        )
        return helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10
        )

    ours, check_ok = onnxsim.simplify(
        build(), check_n=2, function_rewrite_rules=[(pattern_fp, replacement_fp)]
    )
    assert _op_types(ours)["Relu"] == 1, _op_types(ours)
    assert check_ok

    theirs = _apply_onnxscript_rule(build(), common.successive_relu_rule)
    assert _op_types(theirs)["Relu"] == 1, _op_types(theirs)


def test_nonstructural_rewrite_method_is_rejected():
    """A rewrite that derives attributes from the match can't be compiled this
    way -- the boundary is explicit (and matches onnxsim's matcher)."""
    from onnxscript.rewriter.rules.common._matmul_add_to_gemm import MatMulAddToGemm

    with pytest.raises(Exception):
        _compile_fn_to_function_proto(MatMulAddToGemm.rewrite, name="rep")
