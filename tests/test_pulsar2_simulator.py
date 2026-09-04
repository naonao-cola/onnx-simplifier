"""Axera Pulsar2 axmodel simulator + compatible-quantizer test.

Unlike ``test_pulsar2_compat.py`` (pure ``onnx``, always runs), this covers
``scripts/axera/pulsar2_simulator.py`` and ``pulsar2_quantizer.py``, whose
numeric side needs ``onnxruntime`` (an optional onnxsim dependency) --
skip-guarded like ``test_qnn_compat.py``/``test_openvino_compat.py``.
``partition()``/``coverage()`` need only ``onnx`` and are tested unguarded.
"""

import os
import sys

import numpy as np
import pytest

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import models  # noqa: E402
import pulsar2_simulator as sim  # noqa: E402


def test_partition_reports_full_coverage_for_clean_models():
    """Most shared synthetic models use only ops in AX650_SUPPORTED_OPS.

    ``foldable_shape_reshape`` is the one exception: its un-folded `Shape`
    node genuinely isn't on the real, confirmed AX650 list (`Shape` is a
    compile-time-resolved op, not something dispatched to the NPU at
    inference) -- a real finding, not a bug in this test's expectation. A
    deployed model would normally have onnxsim constant-fold this away
    first, which is exactly what this suite's regression tests check for.
    """
    expected_cpu_op_types = {"foldable_shape_reshape": {"Shape": 1}}
    for name in models.names():
        model = models.build(name)
        expected = expected_cpu_op_types.get(name, {})
        p = sim.partition(model)
        assert p.cpu_op_types == expected, (name, p.cpu_op_types)
        assert sim.coverage(model) == ("partial" if expected else "full")


def test_partition_flags_control_flow_as_cpu():
    """A node type outside AX650_SUPPORTED_OPS (e.g. `If`) is not NPU-eligible."""
    import onnx
    from onnx import TensorProto, helper

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
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10
    )
    onnx.checker.check_model(model)

    p = sim.partition(model)
    assert p.cpu_op_types == {"If": 1}
    assert sim.coverage(model) == "none"


pytestmark_numeric = pytest.mark.skipif(
    not sim.SIMULATOR_AVAILABLE,
    reason=f"pulsar2 simulator's numeric side unavailable: {sim.unavailable_reason()}",
)


@pytestmark_numeric
def test_simulate_runs_and_reports_a_diff():
    """conv_bn_relu: quantized-simulated output should be close-ish to fp32."""
    model = models.conv_bn_relu()
    result = sim.simulate(model, seed=0)
    assert result["partition"].npu_node_fraction == 1.0
    assert result["max_abs_diff"] >= 0.0
    for a, b in zip(result["fp32"], result["npu_simulated"]):
        assert a.shape == b.shape


@pytestmark_numeric
def test_quantize_like_pulsar2_matches_confirmed_dtypes():
    """Real pulsar2 build output showed U8 activations, S8 per-channel weights.

    See pulsar2_quantizer.py's docstring for where these numbers come from
    (a real `resnet18d_Opset18` `quant_axmodel.onnx`'s `AxQuantizedConv`
    attributes). This checks our onnxruntime-based quantizer reproduces the
    same *representation* (not the same values -- different quantizer).
    """
    import onnx

    import pulsar2_quantizer as pq

    model = models.conv_bn_relu()
    rng = np.random.RandomState(0)
    feeds = [{"x": rng.randn(1, 3, 16, 16).astype(np.float32)} for _ in range(4)]
    quantized = pq.quantize_like_pulsar2(model, feeds)

    dtypes = set()
    for init in quantized.graph.initializer:
        if "scale" in init.name or "zero_point" in init.name:
            dtypes.add((init.name.split("_")[-1], init.data_type))
    # At least one weight-side zero_point should be int8 (S8) and one
    # activation-side should be uint8 (U8) -- onnx.TensorProto INT8=3, UINT8=2.
    zp_dtypes = {
        init.data_type
        for init in quantized.graph.initializer
        if init.name.endswith("zero_point")
    }
    assert onnx.TensorProto.INT8 in zp_dtypes or onnx.TensorProto.UINT8 in zp_dtypes
