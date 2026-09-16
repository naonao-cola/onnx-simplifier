"""TI edgeai / TIDL static coverage compatibility test.

Verifies that onnxsim's output doesn't gain a new op type, or a newly
dynamic input shape, that TIDL's own published documentation says its
accelerator partitioner can't schedule -- see
``scripts/edgeai/README.md`` for why this is a static heuristic rather than a
real TIDL compiler/device run: TI's `edgeai <https://github.com/
TexasInstruments/edgeai>`_ SDK's TIDL-enabled ``onnxruntime`` build has no
plain-pip package, unlike the QNN/OpenVINO/MIGraphX compat tests.

Like ``tests/test_pulsar2_compat.py``, this needs no vendor package or
device, so this module is not skip-guarded and always runs.
"""

import os
import sys

import pytest

# The TIDL harness lives under scripts/edgeai; reuse it rather than duplicate.
_EDGEAI_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "edgeai"
)
_AXERA_DIR = os.path.join(os.path.dirname(_EDGEAI_DIR), "axera")
for _dir in (_EDGEAI_DIR, _AXERA_DIR):
    if _dir not in sys.path:
        sys.path.insert(0, _dir)

# fresh(), not a bare `import models`/`from worker import check`: every
# scripts/<vendor>/ directory has its own models.py (and most have their own
# worker.py too), all imported by the same bare name -- see
# _local_import.py's docstring for why a plain import here can silently
# resolve to a *different* vendor's module in the full test suite.
import tidl_backend as tidl  # noqa: E402
from _local_import import fresh  # noqa: E402

models = fresh("models", _EDGEAI_DIR)
check = fresh("worker", _EDGEAI_DIR).check


@pytest.mark.parametrize("name", models.names())
def test_edgeai_tidl_compat_suite(name):
    """Each suite model: simplification must not introduce a new TIDL blocker."""
    result = check(name, None)
    assert result["status"] == "ok", result


def test_no_blockers_in_clean_synthetic_models():
    """None of the shared synthetic models should trip the blocker heuristic."""
    for name in models.names():
        if name == "edgeai_dynamic_batch_leaf":
            continue
        model = models.build(name)
        assert tidl.coverage(model) == "full", (name, tidl.blockers(model))


def test_new_blocking_op_types_detects_introduced_control_flow():
    """Sanity-check the diffing itself: a newly-added `If` node should be flagged."""
    import onnx
    from onnx import TensorProto, helper

    orig = models.conv_bn_relu()

    then_graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"])],
        "then",
        [],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
    )
    else_graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"])],
        "else",
        [],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
    )
    cond = helper.make_tensor("cond", TensorProto.BOOL, [], [True])
    if_node = helper.make_node(
        "If", ["cond"], ["y"], then_branch=then_graph, else_branch=else_graph
    )
    graph = helper.make_graph(
        [if_node],
        "with_if",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
        [cond],
    )
    simp_with_if = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10
    )
    onnx.checker.check_model(simp_with_if)

    assert tidl.new_blocking_op_types(orig, simp_with_if) == {"If"}


def test_non_max_suppression_flagged_as_host_only():
    """NonMaxSuppression is documented as host post-processing, not on-accelerator."""
    import onnx
    from onnx import TensorProto, helper

    node = helper.make_node("NonMaxSuppression", ["boxes", "scores"], ["selected"])
    graph = helper.make_graph(
        [node],
        "nms_leaf",
        [
            helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, 6, 4]),
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 1, 6]),
        ],
        [helper.make_tensor_value_info("selected", TensorProto.INT64, [None, 3])],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10
    )
    onnx.checker.check_model(model)

    assert tidl.coverage(model) == "partial"
    assert {op.op_type for op in tidl.blockers(model)} == {"NonMaxSuppression"}


def test_dynamic_batch_leaf_flagged_as_static_shape_risk():
    """A symbolic batch dimension must be flagged: TIDL needs fully static shapes."""
    model = models.edgeai_dynamic_batch_leaf()
    assert tidl.coverage(model) == "partial"
    risks = tidl.dynamic_shape_risks(model)
    assert any("static" in r for r in risks)


def test_overwrite_input_shapes_clears_the_dynamic_shape_risk():
    """Freezing the batch dim via `overwrite_input_shapes` is the documented fix.

    onnxsim itself doesn't decide to fix a dynamic shape on its own -- a
    caller preparing a model for TIDL is expected to pass
    `overwrite_input_shapes` (onnxsim's own mechanism for this) to make the
    graph TIDL-eligible before simplifying it further.
    """
    from onnxsim import simplify

    model = models.edgeai_dynamic_batch_leaf()
    assert tidl.dynamic_shape_risks(model)

    simp, _ = simplify(model, overwrite_input_shapes={"x": [1, 3, 8, 8]})
    assert tidl.dynamic_shape_risks(simp) == []
    assert tidl.coverage(simp) == "full"


def test_mobilenet_block_bn_fold_keeps_full_coverage():
    """MobileNetV2's inverted-residual block (edgeai-tidl-tools' own quickstart
    example model) must simplify cleanly and stay TIDL-blocker-free.

    onnxsim folds the expand/depthwise/project convs' BN Mul/Add pairs into
    the preceding Conv, so the simplified graph has fewer nodes -- that fold
    must not introduce anything TIDL's accelerator can't schedule.
    """
    from onnxsim import simplify

    model = models.mobilenet_block()
    assert tidl.coverage(model) == "full"

    simp, check_ok = simplify(model)
    assert check_ok
    assert len(simp.graph.node) < len(model.graph.node)
    assert tidl.new_blocking_op_types(model, simp) == set()
    assert tidl.coverage(simp) == "full"


def test_vision_transformer_block_stays_full_coverage():
    """A pre-LN ViT encoder block, built from the doc-preferred fused
    `LayerNormalization`/`Gelu` ops, must not trip any TIDL heuristic and
    must simplify without introducing a new blocker.
    """
    from onnxsim import simplify

    model = models.vision_transformer_block()
    assert tidl.coverage(model) == "full"
    assert tidl.normalization_risks(model) == []

    simp, check_ok = simplify(model)
    assert check_ok
    assert tidl.new_blocking_op_types(model, simp) == set()
    assert tidl.coverage(simp) == "full"


def test_decomposed_layer_norm_flagged_as_normalization_risk():
    """LayerNorm spelled out by hand (mean/sub/pow/mean/add/sqrt/div) must be
    flagged: edgeai-tidl-tools' transformer-support notes prefer the fused
    `LayerNormalization` op over this decomposed form -- see
    `tidl_ops.has_decomposed_normalization`'s docstring. Built via
    `onnx.parser` per this repo's CLAUDE.md guidance for new test models.
    """
    import onnx
    from onnx import parser

    model = parser.parse_model(
        """
        <
          ir_version: 8,
          opset_import: ["": 17]
        >
        decomposed_layernorm (float[1,4] x) => (float[1,4] y)
        <float two = {2.0}, float eps = {1e-05}>
        {
          mean = ReduceMean<axes = [-1], keepdims = 1>(x)
          centered = Sub(x, mean)
          sq = Pow(centered, two)
          var = ReduceMean<axes = [-1], keepdims = 1>(sq)
          var_eps = Add(var, eps)
          std = Sqrt(var_eps)
          y = Div(centered, std)
        }
        """
    )
    onnx.checker.check_model(model)

    assert tidl.coverage(model) == "partial"
    assert tidl.normalization_risks(model)
