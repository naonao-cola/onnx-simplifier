"""Author onnxsim ``function_rewrite_rules`` with onnxscript's ``@script``.

An alternative to writing the ``(pattern, replacement)`` FunctionProtos by hand
(``onnx.parser.parse_function``): write each as an ``onnxscript.script`` function
and call ``.to_function_proto()``. That is the same authoring surface onnxscript
uses for op functions, so:

  * nested op calls, value inputs and concrete attributes all just work;
  * a Python-typed *attribute parameter* (``alpha: float``, as opposed to a
    tensor-typed input) compiles to a ``ref_attr_name`` in the body -- exactly
    onnxsim's ``@name`` attribute wildcard, bound on match and substituted into
    the replacement.

``@script`` compiles an *executable* ONNX function, i.e. the same structural
subset onnxsim's matcher handles; it can't express matcher-only constructs
(OrValue, ANY_VALUE, condition predicates) or attributes computed from the match
at rewrite time. These tests cover the structural rules and, where the rule
mirrors an onnxscript *common* rule, check parity against it.
"""

import collections

import numpy as np
import onnx
import pytest
from onnx import parser

import onnxsim

# ``@script`` AST-parses its source at import, so onnxscript must be importable
# for this module to even be collected.
pytest.importorskip("onnxscript")
common = pytest.importorskip("onnxscript.rewriter.rules.common")
_rewriter = pytest.importorskip("onnxscript.rewriter")

from onnxscript import opset18 as op  # noqa: E402
from onnxscript import script  # noqa: E402


# --------------------------------------------------------------------------
# Rules authored as @script functions. `.to_function_proto()` yields the
# onnx.FunctionProto that onnxsim consumes directly.
# --------------------------------------------------------------------------
@script()
def matmul_add_pattern(a, b, c):
    return op.Add(op.MatMul(a, b), c)


@script()
def gemm_replacement(a, b, c):
    return op.Gemm(a, b, c)


@script()
def reshape_reshape_pattern(x, s1, s2):
    return op.Reshape(op.Reshape(x, s1), s2)


@script()
def reshape_replacement(x, s2):
    return op.Reshape(x, s2)


# `alpha: float` is an *attribute* parameter (not a tensor input), so it becomes
# a ref-attribute -> onnxsim's `@name` wildcard.
@script()
def relu_leaky_pattern(x, alpha: float):
    return op.LeakyRelu(op.Relu(x), alpha=alpha)


@script()
def leaky_replacement(x, alpha: float):
    return op.LeakyRelu(x, alpha=alpha)


_MATMUL_ADD_RULE = (
    matmul_add_pattern.to_function_proto(),
    gemm_replacement.to_function_proto(),
)
_RESHAPE_RULE = (
    reshape_reshape_pattern.to_function_proto(),
    reshape_replacement.to_function_proto(),
)
_RELU_LEAKY_RULE = (
    relu_leaky_pattern.to_function_proto(),
    leaky_replacement.to_function_proto(),
)


def _op_types(model):
    return collections.Counter(n.op_type for n in model.graph.node)


def _apply_onnxscript_rule(model, rule):
    rules = _rewriter.pattern.RewriteRuleSet([rule])
    return _rewriter.rewrite(model, pattern_rewrite_rules=rules)


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _model(body, initializer=(), opset=18, ir_version=10):
    # `body` is the ONNX text form graph declaration (and, optionally, its own
    # inline literal initializers); `initializer` holds extra TensorProtos --
    # e.g. random weights -- attached after parsing rather than spelled out as
    # text literals.
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


def test_script_to_function_proto_shape():
    """Sanity: ``@script`` produced the expected FunctionProto shape, including
    the attribute parameter surfacing as a ref-attribute."""
    pat = _RELU_LEAKY_RULE[0]
    assert list(pat.attribute) == ["alpha"]
    leaky = next(n for n in pat.node if n.op_type == "LeakyRelu")
    alpha_attr = next(a for a in leaky.attribute if a.name == "alpha")
    assert alpha_attr.ref_attr_name == "alpha"


def test_script_matmul_add_to_gemm():
    def build():
        return _model(
            """
            g (float[3,4] x) => (float[3,5] y)
            {
              mm = MatMul(x, W)
              y = Add(mm, B)
            }
            """,
            initializer=[
                _f32(np.random.randn(4, 5), "W"),
                _f32(np.random.randn(5), "B"),
            ],
        )

    # onnxsim applies the @script-authored rule (built-in Gemm fusion skipped so
    # the rule is provably responsible).
    ours, check_ok = onnxsim.simplify(
        build(),
        check_n=2,
        skipped_optimizers=["fuse_matmul_add_bias_into_gemm"],
        function_rewrite_rules=[_MATMUL_ADD_RULE],
    )
    counts = _op_types(ours)
    assert counts["Gemm"] == 1 and counts["MatMul"] == 0 and counts["Add"] == 0, counts
    assert check_ok

    # Parity with the onnxscript common rule.
    theirs = _apply_onnxscript_rule(build(), common.matmul_add_to_gemm_rule)
    tcounts = _op_types(theirs)
    assert tcounts["Gemm"] == 1 and tcounts["MatMul"] == 0 and tcounts["Add"] == 0


def test_script_reshape_reshape():
    def build():
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

    ours, check_ok = onnxsim.simplify(
        build(), check_n=2, function_rewrite_rules=[_RESHAPE_RULE]
    )
    assert _op_types(ours)["Reshape"] == 1, _op_types(ours)
    assert check_ok
    reshape = next(n for n in ours.graph.node if n.op_type == "Reshape")
    assert list(reshape.input) == ["x", "s2"], list(reshape.input)

    theirs = _apply_onnxscript_rule(build(), common.reshape_reshape_rule)
    assert _op_types(theirs)["Reshape"] == 1, _op_types(theirs)


def test_script_attribute_wildcard():
    """The ``alpha: float`` attribute parameter round-trips: onnxsim binds the
    matched ``LeakyRelu``'s ``alpha`` and substitutes it into the replacement,
    dropping the ``Relu``. (Structural demo -- not semantics preserving, so no
    correctness check.)"""
    model = _model(
        """
        g (float[2,2] x) => (float[2,2] y)
        {
          t = Relu(x)
          y = LeakyRelu<alpha = 0.2>(t)
        }
        """
    )
    ours, _ = onnxsim.simplify(
        model, check_n=0, function_rewrite_rules=[_RELU_LEAKY_RULE]
    )
    counts = _op_types(ours)
    assert counts["Relu"] == 0 and counts["LeakyRelu"] == 1, counts
    leaky = next(n for n in ours.graph.node if n.op_type == "LeakyRelu")
    alpha = next(a.f for a in leaky.attribute if a.name == "alpha")
    assert abs(alpha - 0.2) < 1e-6, alpha
