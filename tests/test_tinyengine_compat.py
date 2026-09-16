"""TinyEngine static coverage compatibility test.

Verifies that onnxsim's output doesn't gain a new op type that TinyEngine's
real code generator (`code_generator/TfliteConvertor.py`'s `_handleOperator`
dispatch) has no case for -- see `scripts/tinyengine/README.md` for why this
is a static, ONNX-op-to-TFLite-op-mapping heuristic rather than a real
TFLite conversion + codegen run: this repository has no ONNX->TFLite
converter, and TinyEngine itself has no PyPI package.

Like `tests/test_edgeai_tidl_compat.py`, this needs no vendor package or
device, so this module is not skip-guarded and always runs.
"""

import os
import sys

import pytest

_TINYENGINE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts",
    "tinyengine",
)
_AXERA_DIR = os.path.join(os.path.dirname(_TINYENGINE_DIR), "axera")
for _dir in (_TINYENGINE_DIR, _AXERA_DIR):
    if _dir not in sys.path:
        sys.path.insert(0, _dir)

# fresh(), not a bare `import models`/`from worker import check` -- see
# _local_import.py's docstring and scripts/edgeai's own test for why a plain
# import here can silently resolve to a different vendor's same-named module
# in the full test suite.
import tinyengine_backend as tinyengine  # noqa: E402
from _local_import import fresh  # noqa: E402

models = fresh("models", _TINYENGINE_DIR)
check = fresh("worker", _TINYENGINE_DIR).check


@pytest.mark.parametrize("name", models.names())
def test_tinyengine_compat_suite(name):
    """Each suite model: simplification must not introduce a new TinyEngine
    blocker. Unlike TIDL, a new blocker here means the *entire* compile
    would fail (see `tinyengine_backend.py`'s docstring), not "part of the
    graph falls back to a slower path".
    """
    result = check(name, None)
    assert result["status"] == "ok", result


def test_shared_activation_and_reduction_fixtures_stay_partial():
    """Several shared-suite models use ops with no TinyEngine dispatch case
    (`Tanh`, unfused `Relu`, `Sigmoid`) -- confirms this heuristic actually
    flags them rather than reporting false full coverage, and that
    simplification doesn't change that verdict either way for these.
    """
    for name in ("matmul_bias_tanh", "redundant_transpose", "sigmoid_mul_swish"):
        model = models.build(name)
        assert tinyengine.coverage(model) == "partial", name


def test_conv_activation_fusion_depends_on_conv_being_sole_producer():
    """`Relu`/`Clip` are only treated as TinyEngine-supported when they are
    the sole, immediate consumer of a `Conv` output -- approximating
    TFLite's `Conv2D`/`DepthwiseConv2D` `FusedActivationFunction` fusion
    (see `tinyengine_ops.is_fusable_activation`'s docstring). A `Relu` fed
    by anything else (here, an `Add`) has no standalone dispatch case.
    Built via `onnx.parser` per this repo's CLAUDE.md guidance for new
    test models; the 1x1 kernel keeps the tensor literals small.
    """
    import onnx
    from onnx import parser

    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 18]
        >
        unfused_relu (float[1,3,8,8] x) => (float[1,4,8,8] y)
        <
          float[4,3,1,1] w = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0},
          float[1,4,1,1] k = {0.0, 0.0, 0.0, 0.0}
        >
        {
          c = Conv(x, w)
          a = Add(c, k)
          y = Relu(a)
        }
        """
    )
    onnx.checker.check_model(model)

    assert tinyengine.coverage(model) == "partial"
    assert {op.op_type for op in tinyengine.blockers(model)} == {"Relu"}


def test_resize_only_supports_nearest_mode():
    """Only `Resize(mode="nearest")` maps onto `RESIZE_NEAREST_NEIGHBOR`;
    `linear`/`cubic` have no TinyEngine equivalent in this dispatch.

    Uses `onnx.helper` rather than `onnx.parser` (see this repo's CLAUDE.md):
    `Resize`'s optional `roi`/`scales` inputs are most naturally expressed
    as the empty-string positional placeholders `make_node` supports, which
    the text format has no equivalent for.
    """
    import onnx
    from onnx import TensorProto, helper

    def resize_model(mode: str) -> onnx.ModelProto:
        nodes = [
            helper.make_node(
                "Resize", ["x", "", "", "sizes"], ["y"], mode=mode, name="resize"
            ),
        ]
        sizes = helper.make_tensor("sizes", onnx.TensorProto.INT64, [4], [1, 3, 16, 16])
        model = helper.make_model(
            helper.make_graph(
                nodes,
                f"resize_{mode}",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 16, 16])],
                [sizes],
            ),
            opset_imports=[helper.make_opsetid("", 18)],
            ir_version=10,
        )
        onnx.checker.check_model(model)
        return model

    assert tinyengine.coverage(resize_model("nearest")) == "full"
    linear = resize_model("linear")
    assert tinyengine.coverage(linear) == "partial"
    assert {op.op_type for op in tinyengine.blockers(linear)} == {"Resize"}


def test_standalone_mul_is_blocked_but_se_window_mul_is_not():
    """The sharp real constraint: `Mul` is only ever supported as part of
    the exact `Add -> Mul -> Mul` squeeze-and-excite window; anywhere else
    (`standalone_mul_leaf`), it has no dispatch case at all.
    """
    standalone = models.build("standalone_mul_leaf")
    assert tinyengine.coverage(standalone) == "partial"
    assert {op.op_type for op in tinyengine.blockers(standalone)} == {"Mul"}

    se_gated = models.se_gate_block()
    assert tinyengine.coverage(se_gated) == "full"
    assert tinyengine.blockers(se_gated) == []


def test_prelu_has_real_opcode_but_no_dispatch_case():
    """`PRelu` has a valid `BuiltinOperator` enum value in TinyEngine's own
    vendored TFLite schema bindings, but no case in `_handleOperator`'s
    dispatch -- a schema-only check would wrongly call this supported.
    """
    model = models.build("prelu_leaf")
    assert tinyengine.coverage(model) == "partial"
    assert {op.op_type for op in tinyengine.blockers(model)} == {"PRelu"}


def test_reshape_is_skip_op_but_flagged_with_a_caveat():
    """`Reshape` is in TinyEngine's own `SKIP_OPs` (silently dropped, not
    rejected) -- but `constant.py`'s own `# TODO: Handle RESHAPE during
    codegen` comment means this heuristic surfaces it as a caveat rather
    than reporting silently-clean coverage.
    """
    model = models.build("foldable_shape_reshape")
    risks = tinyengine.skip_op_risks(model)
    assert risks and "RESHAPE" in risks[0]


def test_dynamic_batch_leaf_flagged_as_static_shape_risk():
    """A symbolic batch dimension must be flagged: a source-code generator
    needs every shape fixed ahead of time, the same hard requirement
    `scripts/edgeai/tidl_ops.has_dynamic_shape` documents for TIDL, for a
    different underlying reason (fixed-size generated C buffers here,
    a fixed-shape accelerator partitioner there).
    """
    model = models.build("tinyengine_dynamic_batch_leaf")
    assert tinyengine.coverage(model) == "partial"
    risks = tinyengine.dynamic_shape_risks(model)
    assert any("fixed-size" in r for r in risks)


def test_mobilenet_dw_block_bn_fold_clears_the_activation_blocker():
    """Before `onnxsim.simplify()`, each Clip (Relu6) sits after a BN
    Mul/Add pair rather than directly after its Conv, so it is not eligible
    for TinyEngine's Conv-activation fusion. Folding the BN into the Conv
    (onnxsim's own optimization) is what makes the block TinyEngine-eligible
    -- see `scripts/tinyengine/models.py`'s `mobilenet_dw_block` docstring.
    """
    from onnxsim import simplify

    model = models.build("mobilenet_dw_block")
    assert tinyengine.coverage(model) == "partial"
    assert tinyengine.blockers(model) != []

    simp, check_ok = simplify(model)
    assert check_ok
    assert len(simp.graph.node) < len(model.graph.node)
    assert tinyengine.new_blocking_op_types(model, simp) == set()
    assert tinyengine.coverage(simp) == "full"


def test_se_gate_block_regresses_after_onnxsim_affine_fold():
    """A real, worth-documenting finding, not a bug in this heuristic:
    `se_gate_block` has full TinyEngine coverage *before*
    `onnxsim.simplify()` (its trailing `Mul` matches the `Add -> Mul ->
    Mul` squeeze-and-excite window). onnxsim's own affine-fold collapses
    the block's `Add(bias)`/`Mul(scale)` pair directly into the preceding
    `Conv`'s weight and bias -- the same fold `test_mobilenet_dw_block_
    bn_fold_clears_the_activation_blocker` above benefits from -- which
    here *removes* the exact node-type signature the SE-gate fusion
    depends on, leaving a standalone trailing `Mul` with nothing to match.
    `new_blocking_op_types` correctly flags this as a regression. This is
    why `se_gate_block` is deliberately excluded from
    `models.names()`'s default suite -- see that module's docstring.
    """
    from onnxsim import simplify

    model = models.se_gate_block()
    assert tinyengine.coverage(model) == "full"

    simp, check_ok = simplify(model)
    assert check_ok
    assert tinyengine.new_blocking_op_types(model, simp) == {"Mul"}
    assert tinyengine.coverage(simp) == "partial"
