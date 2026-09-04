"""Axera Pulsar2 (AXCL NPU) static coverage compatibility test.

Verifies that onnxsim's output doesn't gain a new op type this repo's
Pulsar2-NPU-blocker heuristic flags (control flow, sequence/optional types,
string ops, data-dependent-shape ops, or a non-standard ONNX `domain`) that
wasn't already present before simplification.

Unlike the QNN/OpenVINO/MIGraphX compat tests, this needs no vendor package
or device -- there is no real Pulsar2 compiler to invoke, only a static
heuristic (see ``scripts/axera/README.md`` for why) -- so this module is not
skip-guarded and always runs.
"""

import os
import sys

import pytest

# The Pulsar2 harness lives under scripts/axera; reuse it rather than duplicate.
_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

# fresh(), not a bare `import models`/`from worker import check`: every
# scripts/<vendor>/ directory has its own models.py (and most have their own
# worker.py too), all imported by the same bare name -- see
# _local_import.py's docstring for why a plain import here can silently
# resolve to a *different* vendor's module in the full test suite.
import pulsar2_backend as pulsar2  # noqa: E402
from _local_import import fresh  # noqa: E402

models = fresh("models", _AXERA_DIR)
check = fresh("worker", _AXERA_DIR).check


@pytest.mark.parametrize("name", models.names())
def test_pulsar2_compat_suite(name):
    """Each suite model: simplification must not introduce a new NPU blocker.

    ``pulsar2_unsafe_for_simplify`` (the ``axera_npu_compiled_leaf`` fixture)
    is expected, not a failure: it means the worker correctly declined to run
    onnxsim on a model that already has a compiled NPU subgraph node -- see
    ``test_onnxsim_corrupts_a_compiled_npu_subgraph`` below for what happens
    if that guard is bypassed.
    """
    result = check(name, None)
    assert result["status"] in ("ok", "pulsar2_unsafe_for_simplify"), result


def test_no_blockers_in_clean_synthetic_models():
    """None of the shared synthetic models should trip the blocker heuristic."""
    for name in models.names():
        model = models.build(name)
        assert pulsar2.coverage(model) == "full", (
            name,
            pulsar2.blockers(model),
        )


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

    assert pulsar2.new_blocking_op_types(orig, simp_with_if) == {"If"}


def test_axera_leaf_fixture_is_flagged_unsafe():
    """The synthetic fixture should trip the pre-flight guard, like a real .axmodel."""
    model = models.axera_npu_compiled_leaf()
    assert pulsar2.unsafe_for_simplify(model)
    assert (
        check("axera_npu_compiled_leaf", None)["status"]
        == "pulsar2_unsafe_for_simplify"
    )


def test_onnxsim_corrupts_a_compiled_npu_subgraph():
    """Documents the real, reproduced bug this harness exists to guard against.

    Confirmed on a real AX650N with a real ``AXERA-TECH/YOLOv8`` `.axmodel`:
    plain ``onnxsim.simplify()`` drops the NPU weight/command initializer a
    compiled subgraph node references only via a JSON attribute (not a
    declared node input), and the result fails to even load on-device. This
    synthetic fixture reproduces the same shape without needing hardware.

    If this test starts failing because ``stripped_npu_data`` comes back
    empty, onnxsim's bug has been fixed upstream -- update
    ``pulsar2_ops.py``'s docstring and ``worker.py``'s pre-flight guard
    (``pulsar2_ops.has_out_of_band_npu_data``) accordingly instead of just
    deleting this test.
    """
    from onnxsim import simplify

    model = models.axera_npu_compiled_leaf()
    assert not pulsar2.stripped_npu_data(model)

    simp, _ = simplify(model)
    assert pulsar2.stripped_npu_data(simp) == {"npu_params"}


def test_llm_layer_leaf_fixture_multi_node_detection():
    """The same corruption bug, confirmed on a real per-layer LLM `.axmodel`.

    A real `Qwen/Qwen3-0.6B` build (see `pulsar2_docker.llm_build()`'s and
    `pulsar2_ops.py`'s docstrings) compiles each transformer layer to a
    graph with *two* `neu mode` nodes (decode + prefill), not one --
    `axera_llm_layer_leaf` reproduces that shape. Checks the detector
    handles multiple NPU nodes sharing one initializer correctly, both
    before (nothing missing) and after (everything referenced gets
    stripped) simplification.
    """
    from onnxsim import simplify

    model = models.axera_llm_layer_leaf()
    assert pulsar2.unsafe_for_simplify(model)
    assert not pulsar2.stripped_npu_data(model)
    assert (
        check("axera_llm_layer_leaf", None)["status"] == "pulsar2_unsafe_for_simplify"
    )

    simp, _ = simplify(model)
    assert pulsar2.stripped_npu_data(simp) == {
        "npu_params",
        "subgraph_npu_b1_neu",
        "subgraph_npu_1_b1_neu",
    }


def test_ax650_build_risks_matches_real_conversions():
    """Sanity-checks the confirmed AX650 op list against two real conversions.

    Both predictions were verified against the real ``pulsar2:6.0-lite``
    toolchain: ``resnet18d_Opset18`` built cleanly (single NPU subgraph,
    identical compiler-reported cost and bit-identical on-device output
    before/after onnxsim simplification); a `googlenet-6`-shaped graph using
    `LRN` at opset 9 hard-failed the real build at the frontend parse stage
    (``KeyError('dont support LRN opr ...')``), not a graceful CPU fallback.
    This test only exercises the static predictor, not the real toolchain --
    see ``pulsar2_ops.py``'s docstring for the full record.
    """
    import onnx
    from onnx import TensorProto, helper

    clean = models.conv_bn_relu()
    assert pulsar2.ax650_build_risks(clean) == []

    lrn_node = helper.make_node("LRN", ["x"], ["y"], size=5)
    graph = helper.make_graph(
        [lrn_node],
        "old_opset_with_lrn",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 8, 8])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 8, 8])],
    )
    old_model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 9)], ir_version=4
    )
    onnx.checker.check_model(old_model)

    risks = pulsar2.ax650_build_risks(old_model)
    assert any("LRN" in r for r in risks)
    assert any("opset 9" in r for r in risks)
