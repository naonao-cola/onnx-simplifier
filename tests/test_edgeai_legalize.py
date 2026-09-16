"""TIDL-preferred fused-op legalization rewrites, checked offline.

`scripts/edgeai/legalize.py` holds rewrites that fuse a decomposed op
export into the single fused op edgeai-tidl-tools' own documentation
recommends (see that module's docstring for why this, unlike
`scripts/axera/legalize.py`, has no real-compiler motivation behind it).
The tests here check the two properties that matter: the rewrite fires on
the pattern it targets and leaves everything else alone, and it does not
change what the graph computes.

Needs no vendor package or device -- correctness is checked against onnx's
own reference evaluator, not a real TIDL run.
"""

import copy
import importlib.util
import os
import sys

import numpy as np
import onnx
from onnx import numpy_helper, parser
from onnx.reference import ReferenceEvaluator

_EDGEAI_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "edgeai"
)
_AXERA_DIR = os.path.join(os.path.dirname(_EDGEAI_DIR), "axera")
for _dir in (_EDGEAI_DIR, _AXERA_DIR):
    if _dir not in sys.path:
        sys.path.insert(0, _dir)

# scripts/axera/legalize.py and scripts/edgeai/legalize.py are two
# different, same-named modules -- a plain `import legalize` here would
# share one `sys.modules["legalize"]` entry with whichever of the two test
# files collects first, silently handing this one the wrong module. Load
# this one under a private key instead, so the two never collide regardless
# of collection order -- see tests/test_axera_legalize.py, which already
# does this on its own side of the same collision.
_spec = importlib.util.spec_from_file_location(
    "edgeai_legalize", os.path.join(_EDGEAI_DIR, "legalize.py")
)
legalize = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = legalize
_spec.loader.exec_module(legalize)

import tidl_backend as tidl  # noqa: E402
from _local_import import fresh  # noqa: E402

models = fresh("models", _EDGEAI_DIR)


def _decomposed_layernorm(affine=False, opset=17):
    """The hand-written LayerNorm export `fuse_decomposed_layernorm` targets."""
    if not affine:
        model = parser.parse_model(
            f"""
            <
              ir_version: 8,
              opset_import: ["": {opset}]
            >
            decomposed_layernorm (float[1,4] x) => (float[1,4] y)
            <float two = {{2.0}}, float eps = {{1e-05}}>
            {{
              mean = ReduceMean<axes = [-1], keepdims = 1>(x)
              centered = Sub(x, mean)
              sq = Pow(centered, two)
              var = ReduceMean<axes = [-1], keepdims = 1>(sq)
              var_eps = Add(var, eps)
              std = Sqrt(var_eps)
              y = Div(centered, std)
            }}
            """
        )
        onnx.checker.check_model(model)
        return model

    model = parser.parse_model(
        f"""
        <
          ir_version: 8,
          opset_import: ["": {opset}]
        >
        decomposed_layernorm_affine (float[1,4] x) => (float[1,4] y)
        <float two = {{2.0}}, float eps = {{1e-05}}>
        {{
          mean = ReduceMean<axes = [-1], keepdims = 1>(x)
          centered = Sub(x, mean)
          sq = Pow(centered, two)
          var = ReduceMean<axes = [-1], keepdims = 1>(sq)
          var_eps = Add(var, eps)
          std = Sqrt(var_eps)
          normed = Div(centered, std)
          scaled = Mul(normed, gamma)
          y = Add(scaled, beta)
        }}
        """
    )
    # Kept as numpy-built initializers per this repo's CLAUDE.md guidance --
    # the parser encodes tensor literals as `float_data`, byte-different
    # from a `numpy_helper.from_array` tensor, which does not matter here
    # but matches this project's established convention regardless.
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(
                np.array([1.1, 0.9, 1.2, 0.8], np.float32), "gamma"
            ),
            numpy_helper.from_array(
                np.array([0.1, -0.1, 0.2, -0.2], np.float32), "beta"
            ),
        ]
    )
    onnx.checker.check_model(model)
    return model


def _erf_gelu():
    """The `torch.onnx.export`-style exact/erf-based GELU export."""
    model = parser.parse_model(
        """
        <
          ir_version: 8,
          opset_import: ["": 17]
        >
        erf_gelu (float[1,4] x) => (float[1,4] y)
        <float sqrt2 = {1.4142135}, float one = {1.0}, float half = {0.5}>
        {
          t0 = Div(x, sqrt2)
          t1 = Erf(t0)
          t2 = Add(t1, one)
          t3 = Mul(x, t2)
          y = Mul(t3, half)
        }
        """
    )
    onnx.checker.check_model(model)
    return model


def _assert_same_output(before, after, input_name, x):
    ref = ReferenceEvaluator(before).run(None, {input_name: x})
    got = ReferenceEvaluator(after).run(None, {input_name: x})
    for r, g in zip(ref, got):
        np.testing.assert_allclose(r, g, rtol=1e-4, atol=1e-5)


def test_fuse_decomposed_layernorm_matches_reference_output():
    model = _decomposed_layernorm()
    before = copy.deepcopy(model)

    assert legalize.fuse_decomposed_layernorm(model) == 1
    onnx.checker.check_model(model)
    assert [n.op_type for n in model.graph.node] == ["LayerNormalization"]

    x = np.random.RandomState(0).randn(1, 4).astype(np.float32)
    _assert_same_output(before, model, "x", x)


def test_fuse_decomposed_layernorm_folds_trailing_affine():
    """A trailing `Mul(scale)`/`Add(bias)` pair folds into `LayerNormalization`'s
    own scale/bias inputs rather than being left dangling after the rest of
    the chain is replaced."""
    model = _decomposed_layernorm(affine=True)
    before = copy.deepcopy(model)

    assert legalize.fuse_decomposed_layernorm(model) == 1
    onnx.checker.check_model(model)
    assert [n.op_type for n in model.graph.node] == ["LayerNormalization"]
    ln = model.graph.node[0]
    assert ln.input[1:] == ["gamma", "beta"]

    x = np.random.RandomState(1).randn(1, 4).astype(np.float32)
    _assert_same_output(before, model, "x", x)


def test_fuse_decomposed_layernorm_clears_the_normalization_risk():
    """Closes the loop with `tidl_ops.has_decomposed_normalization`: flagged
    before the rewrite, clear after."""
    model = _decomposed_layernorm()
    assert tidl.normalization_risks(model)

    legalize.fuse_decomposed_layernorm(model)
    assert tidl.normalization_risks(model) == []
    assert tidl.coverage(model) == "full"


def test_fuse_decomposed_layernorm_leaves_other_reduce_mean_pairs_alone():
    """Two `ReduceMean`s and a `Sqrt` used for something else entirely (not
    this exact wiring) must not be mistaken for the pattern."""
    model = parser.parse_model(
        """
        <
          ir_version: 8,
          opset_import: ["": 17]
        >
        not_layernorm (float[1,4] x, float[1,4] w) => (float[1,4] y)
        {
          a = ReduceMean<axes = [-1], keepdims = 1>(x)
          b = ReduceMean<axes = [-1], keepdims = 1>(w)
          s = Sqrt(b)
          y = Add(a, s)
        }
        """
    )
    onnx.checker.check_model(model)
    assert legalize.fuse_decomposed_layernorm(model) == 0
    assert [n.op_type for n in model.graph.node] == [
        "ReduceMean",
        "ReduceMean",
        "Sqrt",
        "Add",
    ]


def test_fuse_erf_gelu_matches_reference_output():
    model = _erf_gelu()
    before = copy.deepcopy(model)

    assert legalize.fuse_erf_gelu(model) == 1
    onnx.checker.check_model(model)
    assert [n.op_type for n in model.graph.node] == ["Gelu"]
    assert any(o.version >= 20 for o in model.opset_import if o.domain == "")

    x = np.random.RandomState(2).randn(1, 4).astype(np.float32)
    _assert_same_output(before, model, "x", x)


def test_fuse_erf_gelu_ignores_the_wrong_constant():
    """A `Div` by something other than `sqrt(2)` is not GELU -- must not fuse."""
    model = parser.parse_model(
        """
        <
          ir_version: 8,
          opset_import: ["": 17]
        >
        not_gelu (float[1,4] x) => (float[1,4] y)
        <float two = {2.0}, float one = {1.0}, float half = {0.5}>
        {
          t0 = Div(x, two)
          t1 = Erf(t0)
          t2 = Add(t1, one)
          t3 = Mul(x, t2)
          y = Mul(t3, half)
        }
        """
    )
    onnx.checker.check_model(model)
    assert legalize.fuse_erf_gelu(model) == 0


def test_legalize_dispatches_selected_rules_only():
    model = _decomposed_layernorm()
    counts = legalize.legalize(model, rules=["fuse_erf_gelu"])
    assert counts == {"fuse_erf_gelu": 0}
    # fuse_decomposed_layernorm was not asked for, so the pattern survives.
    assert tidl.normalization_risks(model)


def test_legalize_all_rules_is_a_no_op_on_already_fused_fixtures():
    """The suite's own MobileNet/ViT fixtures are already fused; legalizing
    them must be a true no-op, not a spurious rewrite."""
    for name in ("mobilenet_block", "vision_transformer_block", "conv_bn_relu"):
        model = models.build(name)
        counts = legalize.legalize(model)
        assert counts == {"fuse_decomposed_layernorm": 0, "fuse_erf_gelu": 0}, name
