"""Use onnxscript's built-in *common* rewrite rules as onnxsim
``function_rewrite_rules``.

onnxscript ships a library of ready-made rewrite rules in
``onnxscript.rewriter.rules.common`` (e.g. ``matmul_add_to_gemm_rule``,
``reshape_reshape_rule``). Those rules are authored in onnxscript's pattern DSL,
but the simple, structural ones are equivalently expressible as a
``(pattern, replacement)`` pair of ``onnx.FunctionProto`` -- exactly what
onnxsim's ``function_rewrite_rules`` consumes, with no onnxscript dependency at
run time.

Each test here hand-authors the FunctionProto equivalent of a named common rule
and asserts two things:
  * onnxsim applies it (the fusion happens), and
  * it matches what the *actual* onnxscript common rule does on the same model
    (parity), so the FunctionProto really is that rule.

Only rules within onnxsim's first-version matcher (main-graph, structural,
attribute equality/``@``-wildcards, <=2-operand commutativity) are covered;
rules that need condition predicates or attribute arithmetic (e.g.
``cast_cast_rule``, ``transpose_transpose_rule``) are out of scope.
"""

import collections

import numpy as np
import onnx
import pytest
from onnx import parser

import onnxsim

# The common rules live under this module; skip the whole file if the installed
# onnxscript predates it.
common = pytest.importorskip("onnxscript.rewriter.rules.common")
_rewriter = pytest.importorskip("onnxscript.rewriter")


def _op_types(model):
    return collections.Counter(n.op_type for n in model.graph.node)


def _apply_onnxscript_rule(model, rule):
    """Run a single onnxscript common rule over ``model``."""
    rules = _rewriter.pattern.RewriteRuleSet([rule])
    return _rewriter.rewrite(model, pattern_rewrite_rules=rules)


def _model(body, initializer=(), opset=18, ir_version=10):
    model = parser.parse_model(
        f"""
        <
          ir_version: {ir_version},
          opset_import: ["": {opset}]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


# --------------------------------------------------------------------------
# matmul_add_to_gemm_rule:  Add(MatMul(a, b), c)  ->  Gemm(a, b, c)
# --------------------------------------------------------------------------
_MATMUL_ADD_PATTERN = parser.parse_function(
    """
<domain: "onnxscript.common", opset_import: ["" : 18]>
matmul_add (a, b, c) => (y)
{
    mm = MatMul(a, b)
    y = Add(mm, c)
}
"""
)
_MATMUL_ADD_REPLACEMENT = parser.parse_function(
    """
<domain: "onnxscript.common", opset_import: ["" : 18]>
gemm (a, b, c) => (y)
{
    y = Gemm(a, b, c)
}
"""
)


def _matmul_add_model():
    return _model(
        """
        g (float[3,4] x) => (float[3,5] y)
        {
            mm = MatMul(x, W)
            y = Add(mm, B)
        }
        """,
        initializer=[_f32(np.random.randn(4, 5), "W"), _f32(np.random.randn(5), "B")],
    )


def test_matmul_add_to_gemm_common_rule():
    # onnxsim applies the FunctionProto form. The built-in
    # ``fuse_matmul_add_bias_into_gemm`` pass is skipped so the rule is provably
    # what produces the Gemm.
    ours, check_ok = onnxsim.simplify(
        _matmul_add_model(),
        check_n=2,
        skipped_optimizers=["fuse_matmul_add_bias_into_gemm"],
        function_rewrite_rules=[(_MATMUL_ADD_PATTERN, _MATMUL_ADD_REPLACEMENT)],
    )
    counts = _op_types(ours)
    assert counts["Gemm"] == 1 and counts["MatMul"] == 0 and counts["Add"] == 0, counts
    assert check_ok

    # Parity: the actual onnxscript common rule fuses the same way.
    theirs = _apply_onnxscript_rule(_matmul_add_model(), common.matmul_add_to_gemm_rule)
    tcounts = _op_types(theirs)
    assert tcounts["Gemm"] == 1 and tcounts["MatMul"] == 0 and tcounts["Add"] == 0


# --------------------------------------------------------------------------
# reshape_reshape_rule:  Reshape(Reshape(x, s1), s2)  ->  Reshape(x, s2)
# (Valid whenever s2 carries no zero dims -- which onnxscript's rule guards and
# this test's shapes satisfy.)
# --------------------------------------------------------------------------
_RESHAPE_RESHAPE_PATTERN = parser.parse_function(
    """
<domain: "onnxscript.common", opset_import: ["" : 18]>
reshape_reshape (x, s1, s2) => (y)
{
    t = Reshape(x, s1)
    y = Reshape(t, s2)
}
"""
)
_RESHAPE_RESHAPE_REPLACEMENT = parser.parse_function(
    """
<domain: "onnxscript.common", opset_import: ["" : 18]>
reshape (x, s2) => (y)
{
    y = Reshape(x, s2)
}
"""
)


def _reshape_reshape_model():
    return _model(
        """
        g (float[3,8] x) => (float[4,6] y)
        <int64[2] s1 = {2, 12}, int64[2] s2 = {4, 6}>
        {
            t = Reshape(x, s1)
            y = Reshape(t, s2)
        }
        """
    )


def test_reshape_reshape_common_rule():
    ours, check_ok = onnxsim.simplify(
        _reshape_reshape_model(),
        check_n=2,
        function_rewrite_rules=[
            (_RESHAPE_RESHAPE_PATTERN, _RESHAPE_RESHAPE_REPLACEMENT)
        ],
    )
    assert _op_types(ours)["Reshape"] == 1, _op_types(ours)
    assert check_ok
    # The surviving Reshape consumes the original input and the outer shape.
    reshape = next(n for n in ours.graph.node if n.op_type == "Reshape")
    assert list(reshape.input) == ["x", "s2"], list(reshape.input)

    # Parity: the actual onnxscript common rule collapses the pair the same way.
    theirs = _apply_onnxscript_rule(
        _reshape_reshape_model(), common.reshape_reshape_rule
    )
    assert _op_types(theirs)["Reshape"] == 1, _op_types(theirs)
