"""Tests for ``onnxsim.plan_activation_memory`` / ``print_memory_plan``.

Every model is built via ``onnx.parser.parse_model`` (the ONNX text format).
The chain-reuse expectations mirror ``onnxsim/memory_planning_test.cpp``'s
corresponding C++ tests, worked out by hand there.
"""

import numpy as np
from onnx import numpy_helper, parser

import onnxsim
from onnxsim import (
    MemoryPlan,
    annotate_memory_plan,
    plan_activation_memory,
    print_memory_plan,
)
from onnxsim.model_info import METADATA_PREFIX


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


def _metadata(proto):
    return {entry.key: entry.value for entry in proto.metadata_props}


def test_chain_reuse():
    # x -> a -> b -> y, each [25] float32 = 100 bytes. Relu is in-place-safe
    # (see IsInPlaceSafeOp), and a/b are each the sole input of the next
    # Relu, so a, b and y union into one group; x stays separate (a graph
    # input is never aliased away) -- see memory_planning_test.cpp's
    # TestChainReuse for the derivation. Net effect: 2x compression as
    # before, now via one shared slot for {a, b, y} plus x's own slot.
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
    assert off["a"] == off["b"] == off["y"]  # literally one group now
    assert off["x"][0] != off["a"][0]  # still can't share with x
    assert off["x"][1] == off["a"][1] == off["b"][1] == off["y"][1] == 100


def test_in_place_aliasing_collapses_whole_chain():
    # A pure elementwise chain with no graph-input/output boundary in the
    # middle: x -> Relu -> a -> Sigmoid -> b -> Tanh -> c -> Neg -> d ->
    # Identity -> out. Every internal tensor is the sole consumer of the
    # previous one, so a/b/c/d/out all union into a single group; only x (a
    # graph input) stays separate. The arena is 2 slots regardless of chain
    # length, while naive_bytes keeps growing -- see
    # memory_planning_test.cpp's TestInPlaceAliasingCollapsesWholeChain.
    body = """
    g (float[25] x) => (float[25] out)
    {
      a = Relu(x)
      b = Sigmoid(a)
      c = Tanh(b)
      d = Neg(c)
      out = Identity(d)
    }
    """
    plan = plan_activation_memory(_model(body))

    assert plan.unplanned == []
    assert plan.naive_bytes == 600  # 6 tensors x 100 bytes
    assert plan.arena_bytes == 200  # x's slot + one shared slot

    off = plan.tensor_offsets
    assert off["x"][0] != off["a"][0]
    assert off["a"] == off["b"] == off["c"] == off["d"] == off["out"]


def test_in_place_aliasing_blocked_by_multiple_consumers():
    # `a` feeds two separate in-place-eligible ops (Neg -> y1, Sigmoid ->
    # y2). Without the "sole consumer" guard, both would try to alias `a`
    # away, incorrectly merging y1 and y2 -- which are both graph outputs,
    # both live simultaneously -- into the same slot.
    body = """
    g (float[4] x) => (float[4] y1, float[4] y2)
    {
      a = Relu(x)
      y1 = Neg(a)
      y2 = Sigmoid(a)
    }
    """
    plan = plan_activation_memory(_model(body))

    assert plan.unplanned == []
    assert plan.tensor_offsets["y1"][0] != plan.tensor_offsets["y2"][0]


def test_binary_in_place_aliasing_donates_operand():
    # x -> Relu -> a, then y = Add(a, w) with w a weight. `a` is the sole
    # consumer feeding Add, so it donates into y (see IsInPlaceSafeBinaryOp);
    # w is a weight and never planned at all. See memory_planning_test.cpp's
    # TestBinaryInPlaceAliasingDonatesOperand for the byte-offset derivation:
    # arena is 2 slots (x's own, plus one shared {a, y} slot) rather than 3.
    body = """
    g (float[25] x) => (float[25] y)
    {
      a = Relu(x)
      y = Add(a, w)
    }
    """
    plan = plan_activation_memory(_model(body, [_weight([25], "w")]))

    assert plan.unplanned == []
    assert "w" not in plan.tensor_offsets
    assert plan.naive_bytes == 300  # x + a + y, 100 bytes each
    assert plan.arena_bytes == 200

    off = plan.tensor_offsets
    assert off["a"] == off["y"]
    assert off["x"][0] != off["a"][0]


def test_binary_in_place_aliasing_other_operand_still_tracked():
    # Both operands of Add are otherwise alias-eligible (each the sole
    # consumer of its own Relu), but at most one is donated: input[0] (`a`)
    # is tried first and succeeds, so `b` stays an ordinary, independently
    # tracked tensor. See memory_planning_test.cpp's
    # TestBinaryInPlaceAliasingOtherOperandStillTracked for the full
    # byte-offset derivation.
    body = """
    g (float[25] x1, float[25] x2) => (float[25] y)
    {
      a = Relu(x1)
      b = Relu(x2)
      y = Add(a, b)
    }
    """
    plan = plan_activation_memory(_model(body))

    assert plan.unplanned == []
    assert plan.naive_bytes == 500  # x1, x2, a, b, y, 100 bytes each
    assert plan.arena_bytes == 300

    off = plan.tensor_offsets
    assert off["a"] == off["y"]  # a donated into y's group
    assert off["b"] != off["a"]  # b is its own, separate group
    assert off["b"] != off["y"]


def test_binary_in_place_aliasing_skips_broadcast_operand():
    # `scale` ([1], 4 bytes) broadcasts up to the output's shape ([25], 100
    # bytes), so its byte size never matches the output's -- it must never be
    # aliased no matter how eligible it otherwise looks. `a` ([25], 100
    # bytes) is the exact-shape operand and is still eligible, exercising the
    # "input[0] doesn't qualify -> fall back to input[1]" order since `scale`
    # is input[0] here. See memory_planning_test.cpp's
    # TestBinaryInPlaceAliasingSkipsBroadcastOperand for the derivation.
    body = """
    g (float[1] s0, float[25] x) => (float[25] y)
    {
      scale = Relu(s0)
      a = Relu(x)
      y = Add(scale, a)
    }
    """
    plan = plan_activation_memory(_model(body))

    assert plan.unplanned == []
    assert plan.naive_bytes == 308  # 4 + 100 + 4 + 100 + 100
    assert plan.arena_bytes == 204

    off = plan.tensor_offsets
    assert off["a"] == off["y"]  # a still donated (input[1])
    assert off["scale"] != off["y"]  # broadcast operand never aliased
    assert off["scale"][1] == 4
    assert off["y"][1] == 100


def test_binary_in_place_aliasing_blocked_by_multiple_consumers():
    # `a` is consumed by Neg(a) -> y1 and Add(a, a) -> y2, the latter using
    # the same tensor as both operands. The consumer-count guard counts
    # (node, input-slot) pairs, so Add(a, a) alone contributes two consumer
    # events -- well past the "consumed exactly once" bar -- so neither the
    # unary nor the binary aliasing pass ever touches `a`, and y1/y2 (both
    # simultaneously live graph outputs) are never incorrectly merged.
    body = """
    g (float[25] x) => (float[25] y1, float[25] y2)
    {
      a = Relu(x)
      y1 = Neg(a)
      y2 = Add(a, a)
    }
    """
    plan = plan_activation_memory(_model(body))

    assert plan.unplanned == []
    assert plan.naive_bytes == 400
    assert plan.arena_bytes == 300  # no donation possible anywhere

    off = plan.tensor_offsets
    assert off["a"] != off["y1"]
    assert off["a"] != off["y2"]
    assert off["y1"] != off["y2"]


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


# --------------------------------------------------------------------------- #
# annotate_memory_plan
# --------------------------------------------------------------------------- #
def test_annotate_memory_plan_model_level_metadata():
    body = """
    g (float[25] x) => (float[25] y)
    {
      a = Relu(x)
      b = Relu(a)
      y = Relu(b)
    }
    """
    model = _model(body)
    annotated = annotate_memory_plan(model)

    meta = _metadata(annotated)
    assert meta[METADATA_PREFIX + "memory_plan_arena_bytes"] == "200"
    assert meta[METADATA_PREFIX + "memory_plan_naive_bytes"] == "400"
    assert meta[METADATA_PREFIX + "memory_plan_compression_ratio"] == "0.5000"
    assert meta[METADATA_PREFIX + "memory_plan_unplanned_count"] == "0"
    assert METADATA_PREFIX + "memory_plan_unplanned" not in meta

    # The original model is untouched.
    assert len(model.metadata_props) == 0


def test_annotate_memory_plan_value_level_metadata():
    body = """
    g (float[25] x) => (float[25] y)
    {
      a = Relu(x)
      b = Relu(a)
      y = Relu(b)
    }
    """
    model = _model(body)
    plan = plan_activation_memory(model)
    annotated = annotate_memory_plan(model)

    value_infos = {
        vi.name: vi
        for vi in list(annotated.graph.input)
        + list(annotated.graph.output)
        + list(annotated.graph.value_info)
    }
    for name, (offset, size) in plan.tensor_offsets.items():
        meta = _metadata(value_infos[name])
        assert meta[METADATA_PREFIX + "mem_offset"] == str(offset)
        assert meta[METADATA_PREFIX + "mem_size"] == str(size)


def test_annotate_memory_plan_unplanned_tensors_left_unannotated():
    body = """
    g (float[batch,8] x) => (float[batch,8] y)
    {
      y = Relu(x)
    }
    """
    model = _model(body)
    annotated = annotate_memory_plan(model)

    meta = _metadata(annotated)
    assert meta[METADATA_PREFIX + "memory_plan_arena_bytes"] == "0"
    assert meta[METADATA_PREFIX + "memory_plan_unplanned_count"] == "2"
    assert set(meta[METADATA_PREFIX + "memory_plan_unplanned"].split(", ")) == {
        "x",
        "y",
    }

    x_info = annotated.graph.input[0]
    y_info = annotated.graph.output[0]
    assert METADATA_PREFIX + "mem_offset" not in _metadata(x_info)
    assert METADATA_PREFIX + "mem_offset" not in _metadata(y_info)


def test_annotate_memory_plan_custom_prefix():
    body = """
    g (float[1,3,8,8] x) => (float[1,4,8,8] y)
    {
      y = Conv<kernel_shape=[3,3], pads=[1,1,1,1]>(x, w)
    }
    """
    model = _model(body, [_weight([4, 3, 3, 3], "w")])
    annotated = annotate_memory_plan(model, prefix="custom.")

    meta = _metadata(annotated)
    assert "custom.memory_plan_arena_bytes" in meta
    assert METADATA_PREFIX + "memory_plan_arena_bytes" not in meta
