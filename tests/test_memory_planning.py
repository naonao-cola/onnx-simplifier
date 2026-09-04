"""Tests for ``onnxsim.plan_activation_memory`` / ``print_memory_plan``.

Every model is built via ``onnx.parser.parse_model`` (the ONNX text format).
The chain-reuse expectations mirror ``onnxsim/memory_planning_test.cpp``'s
``TestChainReuse``, worked out by hand there: with four equally-sized tensors
in a linear chain, only immediate neighbors are simultaneously live, so a
greedy best-fit allocator (processed largest-first, ties broken by name) can
pack the whole chain into half the naive (no-reuse) size.
"""

import numpy as np
from onnx import numpy_helper, parser

import onnxsim
from onnxsim import MemoryPlan, plan_activation_memory, print_memory_plan


def _model(body, initializer=(), opset=23, ir_version=10):
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


def _weight(shape, name):
    return numpy_helper.from_array(np.zeros(shape, np.float32), name)


def test_chain_reuse():
    # x -> a -> b -> y, each [25] float32 = 100 bytes; only immediate
    # neighbors overlap in liveness, so half the tensors can share offsets
    # with a non-adjacent one -- see memory_planning_test.cpp's TestChainReuse
    # for the interval math.
    body = """
    g (float[25] x) => (float[25] y)
    {
      a = Relu(x)
      b = Relu(a)
      y = Relu(b)
    }
    """
    plan = plan_activation_memory(_model(body))

    assert plan.unplanned == []
    assert plan.naive_bytes == 400
    assert plan.arena_bytes == 200
    assert plan.compression_ratio == 0.5

    off = plan.tensor_offsets
    assert off["a"][0] == off["y"][0]  # a and y share a freed slot
    assert off["b"][0] == off["x"][0]  # b and x share a freed slot
    assert off["a"][0] != off["b"][0]  # but adjacent (overlapping) pairs differ
    assert off["x"][1] == off["a"][1] == off["b"][1] == off["y"][1] == 100


def test_weights_excluded_and_no_reuse_when_overlapping():
    # A single Conv reads x while producing y, so under the allocator's
    # conservative same-node-boundary rule they can never share space: the
    # arena is exactly xb + yb, and the weight never appears in the plan at
    # all (weights stay resident, outside the activation arena).
    body = """
    g (float[1,3,8,8] x) => (float[1,4,8,8] y)
    {
      y = Conv<kernel_shape=[3,3], pads=[1,1,1,1]>(x, w)
    }
    """
    plan = plan_activation_memory(_model(body, [_weight([4, 3, 3, 3], "w")]))

    xb = 1 * 3 * 8 * 8 * 4
    yb = 1 * 4 * 8 * 8 * 4
    assert plan.unplanned == []
    assert "w" not in plan.tensor_offsets
    assert plan.naive_bytes == xb + yb
    assert plan.arena_bytes == xb + yb


def test_dynamic_shape_is_unplanned():
    body = """
    g (float[batch,8] x) => (float[batch,8] y)
    {
      y = Relu(x)
    }
    """
    plan = plan_activation_memory(_model(body))

    assert plan.tensor_offsets == {}
    assert plan.naive_bytes == 0
    assert plan.arena_bytes == 0
    assert plan.compression_ratio == 0.0  # defined as 0, not a division by zero
    assert set(plan.unplanned) == {"x", "y"}


def test_returned_from_top_level_package():
    # plan_activation_memory / print_memory_plan / MemoryPlan are part of the
    # public onnxsim surface, not just onnxsim.memory_planning.
    assert onnxsim.plan_activation_memory is plan_activation_memory
    assert onnxsim.print_memory_plan is print_memory_plan
    assert onnxsim.MemoryPlan is MemoryPlan


def test_print_memory_plan_does_not_raise(capsys):
    body = """
    g (float[25] x) => (float[25] y)
    {
      a = Relu(x)
      y = Relu(a)
    }
    """
    plan = plan_activation_memory(_model(body))
    print_memory_plan(plan)
    captured = capsys.readouterr()
    assert "Arena" in captured.out


def test_print_memory_plan_reports_unplanned(capsys):
    body = """
    g (float[batch,8] x) => (float[batch,8] y)
    {
      y = Relu(x)
    }
    """
    plan = plan_activation_memory(_model(body))
    print_memory_plan(plan)
    captured = capsys.readouterr()
    assert "could not be planned" in captured.out


def test_print_memory_plan_caps_output(capsys):
    plan = MemoryPlan(
        arena_bytes=100,
        naive_bytes=500,
        tensor_offsets={f"t{i}": (i * 100, 100) for i in range(5)},
        unplanned=[],
    )
    print_memory_plan(plan, limit=2)
    captured = capsys.readouterr()
    assert "... and 3 more" in captured.out
