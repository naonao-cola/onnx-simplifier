"""Tests for MACs / FLOPs counting in ``onnxsim.model_info``.

Every model is built directly with ``onnx.helper`` (no torch dependency). MAC
counts are asserted against hand-computed values. The symbolic (sympy) path is
exercised when sympy is installed and skipped otherwise, mirroring the optional
``onnxsim[symbolic]`` extra.
"""

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from onnxsim import model_info
from onnxsim.model_info import (
    METADATA_PREFIX,
    ModelInfo,
    annotate_metadata,
    human_readable_density,
    human_readable_num,
    human_readable_size,
)


def _model(nodes, inputs, outputs, initializers=None, opset=23):
    graph = helper.make_graph(nodes, "g", inputs, outputs, initializers or [])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    return model


def _vi(name, shape, dtype=TensorProto.FLOAT):
    return helper.make_tensor_value_info(name, dtype, shape)


def _macs(nodes, inputs, outputs, initializers=None, opset=23):
    return ModelInfo(_model(nodes, inputs, outputs, initializers, opset)).macs


def _weight(name, shape):
    return helper.make_tensor(
        name, TensorProto.FLOAT, shape, np.zeros(shape, np.float32).flatten()
    )


# --------------------------------------------------------------------------- #
# Core compute operators
# --------------------------------------------------------------------------- #
def test_conv_macs():
    # output 1*4*8*8, cin/group 3, kernel 3*3 -> 256 * 3 * 9
    x = _vi("x", [1, 3, 8, 8])
    w = _weight("w", [4, 3, 3, 3])
    y = _vi("y", [1, 4, 8, 8])
    node = helper.make_node(
        "Conv", ["x", "w"], ["y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]
    )
    assert _macs([node], [x], [y], [w]) == 1 * 4 * 8 * 8 * 3 * (3 * 3)


def test_conv_grouped_macs():
    # depthwise: groups=4, weight [4, 1, 3, 3]; cin/group = 1
    x = _vi("x", [1, 4, 8, 8])
    w = _weight("w", [4, 1, 3, 3])
    y = _vi("y", [1, 4, 8, 8])
    node = helper.make_node(
        "Conv", ["x", "w"], ["y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1], group=4
    )
    assert _macs([node], [x], [y], [w]) == 1 * 4 * 8 * 8 * 1 * (3 * 3)


def test_conv_transpose_macs():
    # input 1*3*8*8, out_channels/group 4, kernel 3*3
    x = _vi("x", [1, 3, 8, 8])
    w = _weight("w", [3, 4, 3, 3])  # [in, out/group, kH, kW]
    y = _vi("y", [1, 4, 10, 10])
    node = helper.make_node("ConvTranspose", ["x", "w"], ["y"], kernel_shape=[3, 3])
    assert _macs([node], [x], [y], [w]) == 1 * 3 * 8 * 8 * 4 * (3 * 3)


def test_gemm_macs():
    a = _vi("a", [5, 7])
    b = _vi("b", [7, 3])
    y = _vi("y", [5, 3])
    node = helper.make_node("Gemm", ["a", "b"], ["y"])
    assert _macs([node], [a, b], [y]) == 5 * 3 * 7


def test_gemm_transposed_macs():
    # transB=1: b is [N, K] = [3, 7]; M=5, N=3, K=7
    a = _vi("a", [5, 7])
    b = _vi("b", [3, 7])
    y = _vi("y", [5, 3])
    node = helper.make_node("Gemm", ["a", "b"], ["y"], transB=1)
    assert _macs([node], [a, b], [y]) == 5 * 3 * 7


def test_matmul_batched_macs():
    # A [2, 5, 7], B [7, 3] -> Y [2, 5, 3]; K=7
    a = _vi("a", [2, 5, 7])
    b = _weight("b", [7, 3])
    y = _vi("y", [2, 5, 3])
    node = helper.make_node("MatMul", ["a", "b"], ["y"])
    assert _macs([node], [a], [y], [b]) == 2 * 5 * 3 * 7


# --------------------------------------------------------------------------- #
# Attention (ai.onnx opset 23+)
# --------------------------------------------------------------------------- #
def _attention_4d(hq, hkv):
    b, sq, skv, d, dv = 2, 16, 16, 64, 64
    q = _vi("q", [b, hq, sq, d])
    k = _vi("k", [b, hkv, skv, d])
    v = _vi("v", [b, hkv, skv, dv])
    y = _vi("y", [b, hq, sq, dv])
    node = helper.make_node("Attention", ["q", "k", "v"], ["y"])
    expected = b * hq * sq * skv * d + b * hq * sq * skv * dv
    return _macs([node], [q, k, v], [y]), expected


def test_attention_4d_mha():
    got, expected = _attention_4d(hq=8, hkv=8)
    assert got == expected


def test_attention_4d_gqa_uses_query_heads():
    # kv heads < q heads, but all q heads are evaluated.
    got, expected = _attention_4d(hq=8, hkv=2)
    assert got == expected


def test_attention_3d_uses_head_attrs():
    b, sq, skv, hidden, heads = 2, 16, 16, 512, 8
    q = _vi("q", [b, sq, hidden])
    k = _vi("k", [b, skv, hidden])
    v = _vi("v", [b, skv, hidden])
    y = _vi("y", [b, sq, hidden])
    node = helper.make_node(
        "Attention", ["q", "k", "v"], ["y"], q_num_heads=heads, kv_num_heads=heads
    )
    d = hidden // heads
    expected = b * heads * sq * skv * d + b * heads * sq * skv * d
    assert _macs([node], [q, k, v], [y]) == expected


def test_attention_3d_without_head_attrs_is_zero():
    # Head split is unknowable without q_num_heads / kv_num_heads.
    b, sq, hidden = 2, 16, 512
    q = _vi("q", [b, sq, hidden])
    k = _vi("k", [b, sq, hidden])
    v = _vi("v", [b, sq, hidden])
    y = _vi("y", [b, sq, hidden])
    node = helper.make_node("Attention", ["q", "k", "v"], ["y"])
    assert _macs([node], [q, k, v], [y]) == 0


# --------------------------------------------------------------------------- #
# Quantized twins reuse the float formulas at the right operand indices
# --------------------------------------------------------------------------- #
def test_matmul_integer_macs():
    a = _vi("a", [4, 8], TensorProto.UINT8)
    b = _vi("b", [8, 16], TensorProto.UINT8)
    y = _vi("y", [4, 16], TensorProto.INT32)
    node = helper.make_node("MatMulInteger", ["a", "b"], ["y"])
    assert _macs([node], [a, b], [y], opset=10) == 4 * 16 * 8


def test_qlinearconv_weight_at_input3():
    # QLinearConv packs weight at input[3]: x, x_s, x_z, w, w_s, w_z, y_s, y_z
    x = _vi("x", [1, 3, 8, 8], TensorProto.UINT8)
    w = _vi("w", [4, 3, 3, 3], TensorProto.UINT8)
    y = _vi("y", [1, 4, 8, 8], TensorProto.UINT8)
    scalars = [
        _vi("x_s", []),
        _vi("x_z", [], TensorProto.UINT8),
        _vi("w_s", [4]),
        _vi("w_z", [4], TensorProto.UINT8),
        _vi("y_s", []),
        _vi("y_z", [], TensorProto.UINT8),
    ]
    node = helper.make_node(
        "QLinearConv",
        ["x", "x_s", "x_z", "w", "w_s", "w_z", "y_s", "y_z"],
        ["y"],
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
    )
    assert _macs([node], [x] + scalars + [w], [y], opset=10) == 1 * 4 * 8 * 8 * 3 * 9


# --------------------------------------------------------------------------- #
# Unknown / dynamic shapes
# --------------------------------------------------------------------------- #
def test_unnamed_dynamic_dim_counts_per_sample():
    # A batch dim with neither a value nor a name: ONNX shape inference assigns
    # it a generated symbol (e.g. "unk__0"), so the count is linear in that axis
    # and collapses to the per-sample MACs when the axis is set to 1.
    x = _vi("x", [None, 3, 8, 8])
    w = _weight("w", [4, 3, 3, 3])
    y = _vi("y", [None, 4, 8, 8])
    node = helper.make_node(
        "Conv", ["x", "w"], ["y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]
    )
    macs = _macs([node], [x], [y], [w])
    assert model_info._representative_number(macs) == 1 * 4 * 8 * 8 * 3 * 9


def test_uninferrable_node_contributes_zero():
    # When a required operand has no shape at all (empty shape map), the counter
    # returns 0 rather than guessing.
    node = helper.make_node("Gemm", ["a", "b"], ["y"])
    assert model_info._gemm_macs(node, {}) == 0


def test_flops_is_twice_macs():
    a = _vi("a", [5, 7])
    b = _vi("b", [7, 3])
    y = _vi("y", [5, 3])
    info = ModelInfo(_model([helper.make_node("Gemm", ["a", "b"], ["y"])], [a, b], [y]))
    assert info.flops == 2 * info.macs


# --------------------------------------------------------------------------- #
# Symbolic (sympy) path
# --------------------------------------------------------------------------- #
def test_dynamic_dim_symbolic():
    sympy = pytest.importorskip("sympy")
    x = _vi("x", ["batch", 3, 8, 8])
    w = _weight("w", [4, 3, 3, 3])
    y = _vi("y", ["batch", 4, 8, 8])
    node = helper.make_node(
        "Conv", ["x", "w"], ["y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]
    )
    macs = ModelInfo(_model([node], [x], [y], [w])).macs
    batch = sympy.Symbol("batch", positive=True, integer=True)
    assert sympy.simplify(macs - 1 * 4 * 8 * 8 * 3 * 9 * batch) == 0


def test_symbolic_dims_unify_across_tensors():
    # A shared dim_param name ("seq") must collapse to one symbol so the two
    # attention matmuls combine into a single seq**2 term.
    sympy = pytest.importorskip("sympy")
    b, hq, d = 2, 8, 64
    q = _vi("q", [b, hq, "seq", d])
    k = _vi("k", [b, hq, "seq", d])
    v = _vi("v", [b, hq, "seq", d])
    y = _vi("y", [b, hq, "seq", d])
    node = helper.make_node("Attention", ["q", "k", "v"], ["y"])
    macs = ModelInfo(_model([node], [q, k, v], [y])).macs
    seq = sympy.Symbol("seq", positive=True, integer=True)
    assert sympy.simplify(macs - 2 * b * hq * d * seq**2) == 0


def test_symbolic_human_readable_num():
    sympy = pytest.importorskip("sympy")
    batch = sympy.Symbol("batch", positive=True, integer=True)
    assert human_readable_num(9472 * batch) == "9472*batch"


def test_print_simplifying_info_symbolic_does_not_raise():
    pytest.importorskip("sympy")
    x = _vi("x", ["batch", 3, 8, 8])
    w = _weight("w", [4, 3, 3, 3])
    y = _vi("y", ["batch", 4, 8, 8])
    node = helper.make_node(
        "Conv", ["x", "w"], ["y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]
    )
    model = _model([node], [x], [y], [w])
    model_info.print_simplifying_info(model, model)  # must not raise on symbolic "<"


def test_dynamic_dim_without_sympy_assumes_one(monkeypatch):
    # With sympy unavailable, dynamic dims are assumed 1 (per-sample MACs).
    monkeypatch.setattr(model_info, "sympy", None)
    x = _vi("x", ["batch", 3, 8, 8])
    w = _weight("w", [4, 3, 3, 3])
    y = _vi("y", ["batch", 4, 8, 8])
    node = helper.make_node(
        "Conv", ["x", "w"], ["y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]
    )
    assert ModelInfo(_model([node], [x], [y], [w])).macs == 1 * 4 * 8 * 8 * 3 * 9


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def test_human_readable_num_units():
    assert human_readable_num(0) == "0.0"
    assert human_readable_num(9472) == "9.5K"
    assert human_readable_num(3_000_000) == "3.0M"


def test_human_readable_size_units():
    assert human_readable_size(512) == "512.0B"
    assert human_readable_size(1024) == "1.0KiB"


def test_human_readable_density_units():
    assert human_readable_density(0) == "0.00 FLOP/Byte"
    assert human_readable_density(2.5) == "2.50 FLOP/Byte"


def test_human_readable_size_symbolic():
    sympy = pytest.importorskip("sympy")
    batch = sympy.Symbol("batch", positive=True, integer=True)
    assert human_readable_size(512 * batch) == "512*batch"


# --------------------------------------------------------------------------- #
# Memory metrics: access traffic, peak footprint, compute density
# --------------------------------------------------------------------------- #
def _info(nodes, inputs, outputs, initializers=None, opset=23):
    return ModelInfo(_model(nodes, inputs, outputs, initializers, opset))


def test_memory_access_gemm():
    # float32 tensors: reads a (5*7) + b (7*3), writes y (5*3), 4 bytes each.
    a = _vi("a", [5, 7])
    b = _vi("b", [7, 3])
    y = _vi("y", [5, 3])
    node = helper.make_node("Gemm", ["a", "b"], ["y"])
    expected = (5 * 7 + 7 * 3 + 5 * 3) * 4
    assert _info([node], [a, b], [y]).mem_access == expected


def test_memory_access_counts_weights():
    # A weight fed as an initializer is read from memory, so its bytes count.
    a = _vi("a", [4, 8])
    b = _weight("b", [8, 16])
    y = _vi("y", [4, 16])
    node = helper.make_node("MatMul", ["a", "b"], ["y"])
    expected = (4 * 8 + 8 * 16 + 4 * 16) * 4
    assert _info([node], [a], [y], [b]).mem_access == expected


def test_memory_access_respects_dtype_size():
    # float16 halves the bytes of the float32 case.
    a = _vi("a", [5, 7], TensorProto.FLOAT16)
    b = _vi("b", [7, 3], TensorProto.FLOAT16)
    y = _vi("y", [5, 3], TensorProto.FLOAT16)
    node = helper.make_node("Gemm", ["a", "b"], ["y"])
    expected = (5 * 7 + 7 * 3 + 5 * 3) * 2
    assert _info([node], [a, b], [y]).mem_access == expected


def test_memory_access_unknown_shape_contributes_zero():
    # An intermediate with no inferred shape simply drops out of the total; the
    # known operands are still counted. No shapes are supplied (empty maps), so
    # the node's traffic is entirely unknown and totals zero.
    node = helper.make_node("Gemm", ["a", "b"], ["y"])
    assert model_info._node_memory_access(node, {}, {}) == 0


def test_compute_density_is_flops_over_bytes():
    a = _vi("a", [5, 7])
    b = _vi("b", [7, 3])
    y = _vi("y", [5, 3])
    info = _info([helper.make_node("Gemm", ["a", "b"], ["y"])], [a, b], [y])
    assert info.compute_density == info.flops / info.mem_access


def test_compute_density_zero_when_no_traffic():
    # A shapeless model has no measurable traffic, so density is 0 (not a crash).
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, None)
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, None)
    info = _info([helper.make_node("Relu", ["x"], ["y"])], [x], [y])
    assert info.mem_access == 0
    assert info.compute_density == 0


def test_memory_footprint_single_node():
    # Gemm: inputs a, b live through the node and output y is produced -> peak is
    # every tensor resident at once.
    a = _vi("a", [5, 7])
    b = _vi("b", [7, 3])
    y = _vi("y", [5, 3])
    info = _info([helper.make_node("Gemm", ["a", "b"], ["y"])], [a, b], [y])
    assert info.memory_footprint == (5 * 7 + 7 * 3 + 5 * 3) * 4


def test_memory_footprint_reuses_freed_activations():
    # x -> Conv -> h -> Relu -> y. x is dead after Conv, so it is not resident
    # during Relu; the peak is the larger of the two per-node working sets, not
    # the sum of every tensor.
    x = _vi("x", [1, 3, 8, 8])
    w = _weight("w", [4, 3, 3, 3])
    y = _vi("y", [1, 4, 8, 8])
    nodes = [
        helper.make_node(
            "Conv", ["x", "w"], ["h"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]
        ),
        helper.make_node("Relu", ["h"], ["y"]),
    ]
    info = _info(nodes, [x], [y], [w])
    b = 4  # float32
    weight_bytes = 4 * 3 * 3 * 3 * b
    x_bytes = 1 * 3 * 8 * 8 * b
    h_bytes = 1 * 4 * 8 * 8 * b
    y_bytes = 1 * 4 * 8 * 8 * b
    conv_peak = weight_bytes + x_bytes + h_bytes  # x still live, h produced
    relu_peak = weight_bytes + h_bytes + y_bytes  # x freed, y produced
    assert info.memory_footprint == max(conv_peak, relu_peak)
    # ... and strictly less than holding every tensor at once.
    assert info.memory_footprint < weight_bytes + x_bytes + h_bytes + y_bytes


def test_memory_metrics_symbolic_dynamic_batch():
    sympy = pytest.importorskip("sympy")
    a = _vi("a", ["batch", 7])
    b = _weight("b", [7, 3])
    y = _vi("y", ["batch", 3])
    info = _info([helper.make_node("MatMul", ["a", "b"], ["y"])], [a], [y], [b])
    batch = sympy.Symbol("batch", positive=True, integer=True)
    expected = (7 * batch + 3 * batch) * 4 + (
        7 * 3
    ) * 4  # a + y scale with batch; b fixed
    assert sympy.simplify(info.mem_access - expected) == 0


def test_memory_metrics_reported_in_summary(capsys):
    a = _vi("a", [5, 7])
    b = _vi("b", [7, 3])
    y = _vi("y", [5, 3])
    model = _model([helper.make_node("Gemm", ["a", "b"], ["y"])], [a, b], [y])
    model_info.print_simplifying_info(model, model)
    out = capsys.readouterr().out
    assert "Memory Access" in out
    assert "Memory Footprint" in out
    assert "Compute Density" in out


# --------------------------------------------------------------------------- #
# Storing metrics into metadata_props
# --------------------------------------------------------------------------- #
def _meta(proto):
    return {e.key: e.value for e in proto.metadata_props}


def _gemm_model():
    a = _vi("a", [5, 7])
    b = _vi("b", [7, 3])
    y = _vi("y", [5, 3])
    return _model(
        [helper.make_node("Gemm", ["a", "b"], ["y"], name="gemm0")], [a, b], [y]
    )


def test_annotate_metadata_model_level():
    model = _gemm_model()
    info = ModelInfo(model)
    out = annotate_metadata(model)
    meta = _meta(out)
    p = METADATA_PREFIX
    assert meta[p + "macs"] == str(info.macs)
    assert meta[p + "flops"] == str(info.flops)
    assert meta[p + "mem_access"] == str(info.mem_access)
    assert meta[p + "memory_footprint"] == str(info.memory_footprint)
    assert meta[p + "model_size"] == str(info.model_size)
    assert p + "compute_density" in meta


def test_annotate_metadata_node_level():
    out = annotate_metadata(_gemm_model())
    (node,) = out.graph.node
    meta = _meta(node)
    p = METADATA_PREFIX
    assert meta[p + "macs"] == str(5 * 3 * 7)
    assert meta[p + "flops"] == str(2 * 5 * 3 * 7)
    assert meta[p + "mem_access"] == str((5 * 7 + 7 * 3 + 5 * 3) * 4)


def test_annotate_metadata_value_level():
    out = annotate_metadata(_gemm_model())
    by_name = {
        vi.name: _meta(vi) for vi in list(out.graph.input) + list(out.graph.output)
    }
    p = METADATA_PREFIX
    assert by_name["a"][p + "bytes"] == str(5 * 7 * 4)
    assert by_name["b"][p + "bytes"] == str(7 * 3 * 4)
    assert by_name["y"][p + "bytes"] == str(5 * 3 * 4)


def test_annotate_metadata_annotates_initializers():
    a = _vi("a", [4, 8])
    b = _weight("b", [8, 16])
    y = _vi("y", [4, 16])
    model = _model([helper.make_node("MatMul", ["a", "b"], ["y"])], [a], [y], [b])
    out = annotate_metadata(model)
    (init,) = out.graph.initializer
    assert _meta(init)[METADATA_PREFIX + "bytes"] == str(8 * 16 * 4)


def test_annotate_metadata_does_not_mutate_input():
    model = _gemm_model()
    annotate_metadata(model)
    assert len(model.metadata_props) == 0
    assert all(len(n.metadata_props) == 0 for n in model.graph.node)
    assert all(len(vi.metadata_props) == 0 for vi in model.graph.input)


def test_annotate_metadata_output_passes_checker():
    onnx.checker.check_model(annotate_metadata(_gemm_model()))


def test_annotate_metadata_custom_prefix():
    out = annotate_metadata(_gemm_model(), prefix="myprefix.")
    assert "myprefix.macs" in _meta(out)
    assert "myprefix.macs" in _meta(out.graph.node[0])


def test_annotate_metadata_symbolic_stores_formula():
    pytest.importorskip("sympy")
    a = _vi("a", ["batch", 7])
    b = _weight("b", [7, 3])
    y = _vi("y", ["batch", 3])
    model = _model(
        [helper.make_node("MatMul", ["a", "b"], ["y"], name="mm")], [a], [y], [b]
    )
    out = annotate_metadata(model)
    assert _meta(out.graph.node[0])[METADATA_PREFIX + "macs"] == "21*batch"
    a_vi = next(vi for vi in out.graph.input if vi.name == "a")
    assert _meta(a_vi)[METADATA_PREFIX + "bytes"] == "28*batch"


def test_annotate_metadata_recurses_subgraphs():
    # An If whose branches each hold a Gemm: the node inside a branch must be
    # annotated, and the model total must include that branch's MACs.
    cond = helper.make_tensor_value_info("cond", TensorProto.BOOL, [])
    y = _vi("y", [5, 3])

    def branch(name):
        a = _weight("a_" + name, [5, 7])
        b = _weight("b_" + name, [7, 3])
        node = helper.make_node(
            "Gemm", ["a_" + name, "b_" + name], ["y"], name="gemm_" + name
        )
        return helper.make_graph([node], name, [], [_vi("y", [5, 3])], [a, b])

    if_node = helper.make_node(
        "If", ["cond"], ["y"], then_branch=branch("t"), else_branch=branch("e")
    )
    model = _model([if_node], [cond], [y])
    out = annotate_metadata(model)
    # Locate the Gemm inside the then-branch and check it carries metrics.
    then_g = next(a.g for a in out.graph.node[0].attribute if a.name == "then_branch")
    gemm = next(n for n in then_g.node if n.op_type == "Gemm")
    assert _meta(gemm)[METADATA_PREFIX + "macs"] == str(5 * 3 * 7)
    # Model total counts both branches' Gemms.
    assert _meta(out)[METADATA_PREFIX + "macs"] == str(2 * 5 * 3 * 7)


# --------------------------------------------------------------------------- #
# Warnings replace silent failures
# --------------------------------------------------------------------------- #
def test_warns_when_shape_inference_fails(monkeypatch):
    def boom(_model):
        raise RuntimeError("boom")

    monkeypatch.setattr(model_info.shape_inference, "infer_shapes", boom)
    x = _vi("x", [1, 4])
    y = _vi("y", [1, 4])
    node = helper.make_node("Relu", ["x"], ["y"])
    with pytest.warns(UserWarning, match="Shape inference failed"):
        ModelInfo(_model([node], [x], [y]))


# There is no "warn when a MAC counter raises" test: the counting is delegated
# to the C++ implementation, whose counters are bounds-checked and return 0
# rather than throwing, so a malformed node degrades to 0 without a per-counter
# Python warning (the graceful-degradation behaviour is preserved structurally;
# see the "macs == 0 when shapes are unknown" cases above).


# --------------------------------------------------------------------------- #
# Model-local function operators are counted via their bodies
# --------------------------------------------------------------------------- #
def _linear_function():
    # A local function "MyLinear(x, w) -> MatMul(x, w)".
    return helper.make_function(
        "custom",
        "MyLinear",
        ["x", "w"],
        ["y"],
        [helper.make_node("MatMul", ["x", "w"], ["y"])],
        [helper.make_opsetid("", 18)],
    )


def _function_model(nodes, inputs, outputs, initializers, functions):
    model = helper.make_model(
        helper.make_graph(nodes, "g", inputs, outputs, initializers),
        opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("custom", 1)],
        functions=functions,
    )
    onnx.checker.check_model(model)
    return model


def test_function_body_counted_and_op_kept():
    # MyLinear used twice: op counts show the function op, MACs sum the bodies.
    x = _vi("X", [4, 8])
    y = _vi("Y", [4, 32])
    inits = [_weight("W1", [8, 16]), _weight("W2", [16, 32])]
    nodes = [
        helper.make_node("MyLinear", ["X", "W1"], ["H"], domain="custom"),
        helper.make_node("MyLinear", ["H", "W2"], ["Y"], domain="custom"),
    ]
    info = ModelInfo(_function_model(nodes, [x], [y], inits, [_linear_function()]))
    assert info.op_nums["MyLinear"] == 2  # op count keeps the function op
    assert info.macs == 4 * 16 * 8 + 4 * 32 * 16  # MACs come from the bodies


def test_nested_function_body_counted():
    # Block(x, w) -> MyLinear(x, w) -> Relu; nested functions are inlined too.
    inner = _linear_function()
    outer = helper.make_function(
        "custom",
        "Block",
        ["x", "w"],
        ["y"],
        [
            helper.make_node("MyLinear", ["x", "w"], ["t"], domain="custom"),
            helper.make_node("Relu", ["t"], ["y"]),
        ],
        [helper.make_opsetid("", 18), helper.make_opsetid("custom", 1)],
    )
    x = _vi("X", [4, 8])
    y = _vi("Y", [4, 16])
    nodes = [helper.make_node("Block", ["X", "W1"], ["Y"], domain="custom")]
    info = ModelInfo(
        _function_model(nodes, [x], [y], [_weight("W1", [8, 16])], [inner, outer])
    )
    assert info.op_nums["Block"] == 1
    assert info.macs == 4 * 16 * 8


def test_warns_when_inlining_fails(monkeypatch):
    import onnx.inliner

    def boom(_model):
        raise RuntimeError("inliner boom")

    monkeypatch.setattr(onnx.inliner, "inline_local_functions", boom)
    x = _vi("X", [4, 8])
    y = _vi("Y", [4, 16])
    nodes = [helper.make_node("MyLinear", ["X", "W1"], ["Y"], domain="custom")]
    model = _function_model(
        nodes, [x], [y], [_weight("W1", [8, 16])], [_linear_function()]
    )
    with pytest.warns(UserWarning, match="Failed to expand function bodies"):
        info = ModelInfo(model)
    assert info.macs == 0  # body compute uncounted, but no crash


def test_schema_function_fallback_counts_attention(monkeypatch):
    # Drop Attention's bespoke counter so the generic schema-function fallback
    # must expand its context-dependent body and count the two internal MatMuls.
    monkeypatch.delitem(model_info._MAC_COUNTERS, "Attention")
    b, h, sq, d = 2, 8, 16, 64
    q = _vi("Q", [b, h, sq, d])
    k = _vi("K", [b, h, sq, d])
    v = _vi("V", [b, h, sq, d])
    y = _vi("Y", [b, h, sq, d])
    node = helper.make_node("Attention", ["Q", "K", "V"], ["Y"])
    model = helper.make_model(
        helper.make_graph([node], "g", [q, k, v], [y]),
        opset_imports=[helper.make_opsetid("", 24)],
    )
    info = ModelInfo(model)
    assert info.op_nums["Attention"] == 1  # op count still shows the op
    assert info.macs == b * h * sq * sq * d * 2  # exact, via the expanded body


# --------------------------------------------------------------------------- #
# Graph diff: node/value level diff between original and simplified graphs
# --------------------------------------------------------------------------- #
def test_diff_graphs_unchanged_model_is_empty():
    x = _vi("x", [1, 4])
    y = _vi("y", [1, 4])
    model = _model([helper.make_node("Relu", ["x"], ["y"])], [x], [y])
    diff = model_info.diff_graphs(model, model)
    assert diff.removed_nodes == []
    assert diff.added_nodes == []
    assert diff.changed_nodes == []
    assert diff.removed_values == []
    assert diff.added_values == []


def test_diff_graphs_detects_removed_and_added_nodes():
    # x -> Identity -> Relu -> y  becomes  x -> Relu -> y (Identity eliminated).
    x = _vi("x", [1, 4])
    y = _vi("y", [1, 4])
    ori = _model(
        [
            helper.make_node("Identity", ["x"], ["mid"], name="id0"),
            helper.make_node("Relu", ["mid"], ["y"], name="relu0"),
        ],
        [x],
        [y],
    )
    opt = _model([helper.make_node("Relu", ["x"], ["y"], name="relu0")], [x], [y])

    diff = model_info.diff_graphs(ori, opt)
    assert [n.name for n in diff.removed_nodes] == ["id0"]
    assert diff.added_nodes == []
    # relu0 kept its output name "y" but its inputs changed (mid -> x).
    assert len(diff.changed_nodes) == 1
    before, after = diff.changed_nodes[0]
    assert before.inputs == ("mid",)
    assert after.inputs == ("x",)
    assert diff.removed_values == ["mid"]
    assert diff.added_values == []


def test_diff_graphs_added_node_from_constant_folding():
    # A Shape node folded away and replaced by a Constant under a new name.
    x = _vi("x", [1, 4])
    y = _vi("y", [1])
    ori = _model([helper.make_node("Shape", ["x"], ["y"], name="shape0")], [x], [y])
    opt = _model(
        [
            helper.make_node(
                "Constant",
                [],
                ["y_folded"],
                name="const0",
                value=helper.make_tensor("v", TensorProto.INT64, [1], [4]),
            )
        ],
        [x],
        [_vi("y_folded", [1])],
    )
    diff = model_info.diff_graphs(ori, opt)
    assert [n.name for n in diff.removed_nodes] == ["shape0"]
    assert [n.name for n in diff.added_nodes] == ["const0"]
    assert diff.changed_nodes == []
    assert diff.removed_values == ["y"]
    assert diff.added_values == ["y_folded"]


def test_diff_graphs_detects_op_type_change():
    # Same output name "y", but the producing op_type changed.
    x = _vi("x", [1, 4])
    y = _vi("y", [1, 4])
    ori = _model([helper.make_node("Relu", ["x"], ["y"], name="n")], [x], [y])
    opt = _model([helper.make_node("Sigmoid", ["x"], ["y"], name="n")], [x], [y])
    diff = model_info.diff_graphs(ori, opt)
    assert diff.removed_nodes == []
    assert diff.added_nodes == []
    (before, after) = diff.changed_nodes[0]
    assert before.op_type == "Relu"
    assert after.op_type == "Sigmoid"


def test_diff_graphs_removed_and_added_initializers():
    x = _vi("x", [4, 8])
    y = _vi("y", [4, 16])
    b_ori = _weight("b", [8, 16])
    ori = _model([helper.make_node("MatMul", ["x", "b"], ["y"])], [x], [y], [b_ori])
    b_opt = _weight("b_folded", [8, 16])
    opt = _model(
        [helper.make_node("MatMul", ["x", "b_folded"], ["y"])], [x], [y], [b_opt]
    )
    diff = model_info.diff_graphs(ori, opt)
    assert diff.removed_values == ["b"]
    assert diff.added_values == ["b_folded"]


def test_print_graph_diff_does_not_raise(capsys):
    x = _vi("x", [1, 4])
    y = _vi("y", [1, 4])
    ori = _model(
        [
            helper.make_node("Identity", ["x"], ["mid"], name="id0"),
            helper.make_node("Relu", ["mid"], ["y"], name="relu0"),
        ],
        [x],
        [y],
    )
    opt = _model([helper.make_node("Relu", ["x"], ["y"], name="relu0")], [x], [y])
    model_info.print_graph_diff(ori, opt)
    out = capsys.readouterr().out
    assert "Nodes removed" in out
    assert "id0" in out
    assert "Values removed" in out
    assert "mid" in out


def test_print_graph_diff_caps_output(capsys):
    x = _vi("x", [1, 4])
    ori_nodes = [
        helper.make_node("Identity", ["x"], [f"mid{i}"], name=f"id{i}")
        for i in range(5)
    ]
    ori = _model(ori_nodes, [x], [_vi("mid0", [1, 4])])
    opt = _model([], [x], [_vi("x", [1, 4])])
    model_info.print_graph_diff(ori, opt, limit=2)
    out = capsys.readouterr().out
    assert "... and" in out


def test_schema_function_fallback_layernorm_is_zero():
    # A context-dependent function with no matmuls must expand cleanly to 0 MACs.
    x = _vi("X", [4, 16])
    node = helper.make_node("LayerNormalization", ["X", "scale", "bias"], ["Y"])
    model = helper.make_model(
        helper.make_graph(
            [node],
            "g",
            [x],
            [_vi("Y", [4, 16])],
            [_weight("scale", [16]), _weight("bias", [16])],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    assert ModelInfo(model).macs == 0
