"""Renesas RZ/V DRP-AI TVM legalization rewrites, checked offline.

`scripts/renesas/legalize.py` holds rewrites for the three ONNX ops absent
from TVM v0.8's ONNX-import convert map
(`drp_ai_tvm_ops.DRP_AI_TVM_IMPORTABLE_OPS`) that have an *exact* primitive
decomposition per their own ONNX spec text: `HardSwish`, `Mish`,
`LayerNormalization`. Each test checks the two properties that matter: the
rewrite fires where it should (and nowhere else), and it does not change
what the graph computes -- plus, since (unlike Voyager SDK's
attribute-level constraints) this is a binary op_type-membership question,
that `drp_ai_tvm_simulator.would_import_succeed()` flips from `False` to
`True` for a graph using each op.

Numeric equivalence is checked with `onnx.reference.ReferenceEvaluator`,
same as `tests/test_axelera_legalize.py`; nothing here needs onnxsim built,
TVM installed, or a real DRP-AI TVM/HyCo compiler.
"""

import importlib.util
import os
import sys

import numpy as np
import onnx
import pytest
from onnx import parser
from onnx.reference import ReferenceEvaluator

_RENESAS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "renesas"
)
if _RENESAS_DIR not in sys.path:
    sys.path.insert(0, _RENESAS_DIR)

# scripts/axera/legalize.py, scripts/axelera/legalize.py and
# scripts/renesas/legalize.py are three different, same-named modules -- a
# plain `import legalize` here would share one `sys.modules["legalize"]`
# entry with the other two test files' own `import legalize` (whichever
# collects first "wins", silently handing the others the wrong module).
# Load this one under a private key instead, same fix
# test_axelera_legalize.py uses.
_spec = importlib.util.spec_from_file_location(
    "renesas_legalize", os.path.join(_RENESAS_DIR, "legalize.py")
)
legalize = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = legalize
_spec.loader.exec_module(legalize)

import drp_ai_tvm_simulator as sim  # noqa: E402


def _model(body, initializer=(), opset=18, ir_version=10):
    model = parser.parse_model(f"""
        <
          ir_version: {ir_version},
          opset_import: ["": {opset}]
        >
        {body}
        """)
    model.graph.initializer.extend(initializer)
    onnx.checker.check_model(model)
    return model


def _assert_same_output(before, after, feeds):
    onnx.checker.check_model(after)
    before_out = ReferenceEvaluator(before).run(None, feeds)
    after_out = ReferenceEvaluator(after).run(None, feeds)
    for b, a in zip(before_out, after_out):
        assert np.allclose(b, a, atol=1e-5), np.abs(np.asarray(b) - np.asarray(a)).max()


# --- hardswish_to_primitives ---


def test_hardswish_rewrites_to_mul_hardsigmoid_and_computes_the_same_thing():
    before = _model("""
        g (float[2,4] x) => (float[2,4] y)
        { y = HardSwish(x) }
        """)
    after = _model("""
        g (float[2,4] x) => (float[2,4] y)
        { y = HardSwish(x) }
        """)
    assert sim.would_import_succeed(before) is False

    assert legalize.hardswish_to_primitives(after) == 1
    assert [n.op_type for n in after.graph.node] == ["HardSigmoid", "Mul"]
    assert sim.would_import_succeed(after) is True

    x = np.random.RandomState(0).randn(2, 4).astype(np.float32) * 5
    _assert_same_output(before, after, {"x": x})


def test_hardswish_leaves_other_ops_alone():
    model = _model("""
        g (float[2,4] x) => (float[2,4] y)
        { y = Relu(x) }
        """)
    assert legalize.hardswish_to_primitives(model) == 0
    assert [n.op_type for n in model.graph.node] == ["Relu"]


# --- mish_to_primitives ---


def test_mish_rewrites_to_mul_tanh_softplus_and_computes_the_same_thing():
    before = _model("""
        g (float[2,4] x) => (float[2,4] y)
        { y = Mish(x) }
        """)
    after = _model("""
        g (float[2,4] x) => (float[2,4] y)
        { y = Mish(x) }
        """)
    assert sim.would_import_succeed(before) is False

    assert legalize.mish_to_primitives(after) == 1
    assert [n.op_type for n in after.graph.node] == ["Softplus", "Tanh", "Mul"]
    assert sim.would_import_succeed(after) is True

    x = np.random.RandomState(1).randn(2, 4).astype(np.float32) * 5
    _assert_same_output(before, after, {"x": x})


# --- layer_normalization_to_primitives ---


def _ln_model(with_bias=True, axis=None):
    # Scale is a required input per the LayerNormalization spec (min 2
    # inputs: X, Scale); only B (bias) is optional.
    inputs = "float[2,3,4] x, float[4] scale"
    args = ["x", "scale"]
    rng = np.random.RandomState(2)
    if with_bias:
        inputs += ", float[4] bias"
        args.append("bias")
    attr = f"<axis = {axis}>" if axis is not None else ""
    body = f"""
        g ({inputs}) => (float[2,3,4] y)
        {{ y = LayerNormalization{attr}({", ".join(args)}) }}
        """
    return _model(body), rng


def _ln_feeds(rng, with_bias):
    feeds = {
        "x": rng.randn(2, 3, 4).astype(np.float32),
        "scale": rng.randn(4).astype(np.float32),
    }
    if with_bias:
        feeds["bias"] = rng.randn(4).astype(np.float32)
    return feeds


@pytest.mark.parametrize("with_bias", [True, False])
def test_layer_normalization_rewrites_and_computes_the_same_thing(with_bias):
    before, rng = _ln_model(with_bias)
    after, _ = _ln_model(with_bias)
    assert sim.would_import_succeed(before) is False

    assert legalize.layer_normalization_to_primitives(after) == 1
    assert "LayerNormalization" not in [n.op_type for n in after.graph.node]
    assert sim.would_import_succeed(after) is True

    feeds = _ln_feeds(rng, with_bias)
    _assert_same_output(before, after, feeds)


def test_layer_normalization_respects_explicit_axis():
    before, rng = _ln_model(with_bias=False, axis=1)
    after, _ = _ln_model(with_bias=False, axis=1)
    assert legalize.layer_normalization_to_primitives(after) == 1
    reduce_mean = next(n for n in after.graph.node if n.op_type == "ReduceMean")
    # opset 18's ReduceMean takes `axes` as its second input (an
    # initializer here), not an attribute -- see legalize.py's
    # _make_reduce_mean().
    axes_init = next(
        i for i in after.graph.initializer if i.name == reduce_mean.input[1]
    )
    axes = onnx.numpy_helper.to_array(axes_init)
    assert list(axes) == [1, 2]  # rank 3, axis=1 -> normalize over dims [1, 2]

    feeds = _ln_feeds(rng, with_bias=False)
    _assert_same_output(before, after, feeds)


def test_layer_normalization_pre_opset18_uses_reducemean_axes_attribute():
    # ReduceMean's `axes` was an attribute before opset 18 (see
    # legalize.py's _make_reduce_mean() docstring) -- LayerNormalization
    # itself only exists from opset 17 onward, so 17 is the lowest opset
    # this branch is reachable at.
    before = _model(
        """
        g (float[2,3,4] x, float[4] scale) => (float[2,3,4] y)
        { y = LayerNormalization(x, scale) }
        """,
        opset=17,
    )
    after = _model(
        """
        g (float[2,3,4] x, float[4] scale) => (float[2,3,4] y)
        { y = LayerNormalization(x, scale) }
        """,
        opset=17,
    )
    assert legalize.layer_normalization_to_primitives(after) == 1
    reduce_mean = next(n for n in after.graph.node if n.op_type == "ReduceMean")
    assert len(reduce_mean.input) == 1
    axes = next(a for a in reduce_mean.attribute if a.name == "axes").ints
    assert list(axes) == [2]  # rank 3, default axis=-1 -> normalize over dim [2]

    rng = np.random.RandomState(3)
    feeds = {
        "x": rng.randn(2, 3, 4).astype(np.float32),
        "scale": rng.randn(4).astype(np.float32),
    }
    _assert_same_output(before, after, feeds)


def test_layer_normalization_skipped_when_mean_output_consumed():
    model = _model("""
        g (float[2,3,4] x, float[4] scale) => (float[2,3,4] y, float[2,3,1] mean)
        { y, mean = LayerNormalization(x, scale) }
        """)
    assert legalize.layer_normalization_to_primitives(model) == 0
    assert model.graph.node[0].op_type == "LayerNormalization"


def test_layer_normalization_skipped_when_rank_unknown():
    # No value_info at all for `x` beyond a placeholder parse, and no
    # initializer either -- shape_inference can't resolve it, so the rule
    # must fail closed rather than guess a rank.
    model = _model("""
        g (float[2,3,4] x, float[4] scale) => (float[2,3,4] y)
        { y = LayerNormalization(x, scale) }
        """)
    model.graph.input[0].type.tensor_type.ClearField("shape")
    assert legalize.layer_normalization_to_primitives(model) == 0


# --- legalize() / as_custom_rewriter() orchestration ---


def test_legalize_applies_all_three_rules_by_default():
    model = _model("""
        g (float[2,4] x) => (float[2,4] y)
        {
          h = HardSwish(x)
          y = Mish(h)
        }
        """)
    applied = legalize.legalize(model)
    assert applied == {
        "hardswish_to_primitives": 1,
        "mish_to_primitives": 1,
        "layer_normalization_to_primitives": 0,
    }
    assert sim.would_import_succeed(model) is True


def test_as_custom_rewriter_reports_none_when_it_changed_something_else_false():
    model = _model("""
        g (float[2,4] x) => (float[2,4] y)
        { y = HardSwish(x) }
        """)
    rewriter = legalize.as_custom_rewriter()
    assert rewriter(model) is None
    assert sim.would_import_succeed(model) is True

    unchanged = _model("""
        g (float[2,4] x) => (float[2,4] y)
        { y = Relu(x) }
        """)
    assert legalize.as_custom_rewriter()(unchanged) is False
