"""Tests for ``onnxsim.quantize_fp8`` (the ``quantize_fp8`` C++ pass).

Like ``test_quantize_bf16.py``, models are built with multiple chained ops
(not one MatMul/Conv in isolation), since quantize_fp8 is a whole-graph
transform rather than a per-node pattern match, and use opset 19 (the first
opset where ONNX's Cast op supports float8 types).

Also like quantize_bf16's tests, these do NOT execute a float8 MatMul/Relu
through onnxruntime's CPUExecutionProvider -- float8 compute kernel coverage
there is narrower even than bfloat16's. Instead:
- Numeric correctness of the conversion itself is checked directly, by
  decoding the produced float8 initializer back to float32 (via
  ``ml_dtypes.float8_e4m3fn``/``ml_dtypes.float8_e5m2``, the numpy extension
  dtypes ``onnx.numpy_helper`` uses for these TensorProto types) and
  comparing against the value ``ml_dtypes`` itself produces for in-range
  inputs (this validates the pass's round-to-nearest-ties-to-even bit
  manipulation is correct, independent of this pass's own code).
- A dedicated test checks the deliberate difference from ``ml_dtypes``'s own
  plain ``.astype()`` for out-of-range magnitudes: this pass *saturates*
  (clamps to the format's max finite value) rather than producing a NaN
  (E4M3FN has no infinity encoding) or an infinity (E5M2) -- see
  ``docs/fp8-quantization.md``.
- One test does execute a real quantized graph through
  ``onnxruntime.InferenceSession``, using ``Identity`` as the compute op, to
  prove the boundary Cast wiring this pass builds actually loads and runs
  under a real engine.
"""

import collections

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper
import onnx.shape_inference
import pytest

import onnxsim

# A bare ``import onnxruntime`` would fail collection (not skip the test) on
# platforms onnxruntime doesn't ship wheels for (e.g. s390x).
ort = pytest.importorskip("onnxruntime")
ml_dtypes = pytest.importorskip("ml_dtypes")

_NP_DTYPE = {
    "e4m3": ml_dtypes.float8_e4m3fn,
    "e5m2": ml_dtypes.float8_e5m2,
}
_ONNX_DTYPE = {
    "e4m3": onnx.TensorProto.FLOAT8E4M3FN,
    "e5m2": onnx.TensorProto.FLOAT8E5M2,
}
_MAX_VALUE = {
    "e4m3": 448.0,
    "e5m2": 57344.0,
}


def _model(nodes, inputs, outputs, initializer, opset=19):
    graph = onnx.helper.make_graph(nodes, "g", inputs, outputs, initializer)
    return onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", opset)], ir_version=10
    )


def _vi(name, shape, elem_type=onnx.TensorProto.FLOAT):
    return onnx.helper.make_tensor_value_info(name, elem_type, shape)


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _op_counts(model):
    return collections.Counter(n.op_type for n in model.graph.node)


def _initializer_by_name(model, name):
    for init in model.graph.initializer:
        if init.name == name:
            return init
    raise KeyError(name)


def _node_input_initializer(model, op_type, input_index):
    # quantize_fp8 replaces a converted weight with a *new* initializer
    # (leaving the old float32 one orphaned in the model until a follow-up
    # simplify() call prunes it -- documented behavior, matching every other
    # onnxsim quantize_* pass), so the initializer actually feeding a node
    # must be looked up by that node's *current* input name, not by
    # iterating/indexing graph.initializer blindly.
    node = next(n for n in model.graph.node if n.op_type == op_type)
    return _initializer_by_name(model, node.input[input_index])


def _to_fp8_rounded_float32(array, fmt):
    # The reference conversion this pass's C++ FloatToFloat8Bits should
    # match for in-range values: round each float32 value to its nearest
    # float8 and back, via ml_dtypes's own (independent) implementation.
    return array.astype(_NP_DTYPE[fmt]).astype(np.float32)


def _two_matmul_model():
    rng = np.random.default_rng(0)
    k, n1, n2 = 16, 12, 8
    w1 = _f32(rng.standard_normal((k, n1)) * 0.5, "W1")
    w2 = _f32(rng.standard_normal((n1, n2)) * 0.5, "W2")
    nodes = [
        onnx.helper.make_node("MatMul", ["X", "W1"], ["H"]),
        onnx.helper.make_node("Relu", ["H"], ["Hr"]),
        onnx.helper.make_node("MatMul", ["Hr", "W2"], ["Y"]),
    ]
    model = _model(nodes, [_vi("X", [4, k])], [_vi("Y", [4, n2])], [w1, w2])
    return model, rng, k, n2


@pytest.mark.parametrize("fmt", ["e4m3", "e5m2"])
def test_quantize_fp8_keep_io_types(fmt):
    model, rng, k, n2 = _two_matmul_model()

    quant = onnxsim.quantize_fp8(model, format=fmt)
    onnx.checker.check_model(quant)

    ops = _op_counts(quant)
    assert ops["Cast"] == 2  # one boundary cast for X in, one for Y out
    assert ops["MatMul"] == 2
    assert ops["Relu"] == 1

    # The model's own declared I/O stays float32.
    assert quant.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    assert quant.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    # But the weight actually feeding the (first) MatMul is now the target
    # float8 format (the old float32 initializer is left orphaned in the
    # model -- see _node_input_initializer's comment).
    w1_init = _node_input_initializer(quant, "MatMul", 1)
    assert w1_init.data_type == _ONNX_DTYPE[fmt]

    w1_orig = onnx.numpy_helper.to_array(
        next(i for i in model.graph.initializer if i.name == "W1")
    )
    w1_fp8 = onnx.numpy_helper.to_array(w1_init).astype(np.float32)
    np.testing.assert_array_equal(w1_fp8, _to_fp8_rounded_float32(w1_orig, fmt))


@pytest.mark.parametrize("fmt", ["e4m3", "e5m2"])
def test_quantize_fp8_no_keep_io_types(fmt):
    model, rng, k, n2 = _two_matmul_model()

    quant = onnxsim.quantize_fp8(model, format=fmt, keep_io_types=False)
    onnx.checker.check_model(quant)

    ops = _op_counts(quant)
    # No boundary casts needed -- the graph's own I/O is redeclared float8.
    assert ops["Cast"] == 0
    assert quant.graph.input[0].type.tensor_type.elem_type == _ONNX_DTYPE[fmt]
    assert quant.graph.output[0].type.tensor_type.elem_type == _ONNX_DTYPE[fmt]


def test_quantize_fp8_default_format_is_e4m3():
    model, rng, k, n2 = _two_matmul_model()
    quant = onnxsim.quantize_fp8(model)
    w1_init = _node_input_initializer(quant, "MatMul", 1)
    assert w1_init.data_type == onnx.TensorProto.FLOAT8E4M3FN


def test_quantize_fp8_rejects_unknown_format():
    model, rng, k, n2 = _two_matmul_model()
    with pytest.raises(Exception):
        onnxsim.quantize_fp8(model, format="not-a-format")


@pytest.mark.parametrize("fmt", ["e4m3", "e5m2"])
def test_quantize_fp8_converts_constant_node(fmt):
    # A Constant node's embedded value is a float32 "inline initializer" --
    # FetchConstantTensor covers it the same way as a true graph
    # initializer, so it should be converted too.
    rng = np.random.default_rng(1)
    k, n = 8, 4
    w = rng.standard_normal((k, n)).astype(np.float32)
    const_node = onnx.helper.make_node(
        "Constant",
        [],
        ["W"],
        value=onnx.numpy_helper.from_array(w, "W"),
    )
    matmul_node = onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])
    model = _model(
        [const_node, matmul_node], [_vi("X", [3, k])], [_vi("Y", [3, n])], []
    )

    quant = onnxsim.quantize_fp8(model, format=fmt)
    onnx.checker.check_model(quant)

    w_init = _node_input_initializer(quant, "MatMul", 1)
    assert w_init.data_type == _ONNX_DTYPE[fmt]
    w_fp8 = onnx.numpy_helper.to_array(w_init).astype(np.float32)
    np.testing.assert_array_equal(w_fp8, _to_fp8_rounded_float32(w, fmt))


def test_quantize_fp8_e4m3_rounds_ties_to_even():
    # float8's mantissa is only 2-3 bits wide, so exact ties are common
    # enough on real data that round-to-nearest-ties-to-even (not the
    # ties-away-from-zero quantize_fp16/quantize_bf16 use) matters -- verify
    # against exact halfway values in E4M3FN's 3-bit-mantissa grid near 1.0
    # (ULP 0.125): 1.0625 is exactly halfway between 1.0 (mantissa 000,
    # even) and 1.125 (mantissa 001, odd) and must round down to 1.0;
    # 1.1875 is exactly halfway between 1.125 (odd) and 1.25 (mantissa 010,
    # even) and must round up to 1.25.
    w = np.array([[1.0625, 1.1875, -1.0625]], dtype=np.float32)
    weight = _f32(w, "W")
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    model = _model(nodes, [_vi("X", [2, 1])], [_vi("Y", [2, 3])], [weight])

    quant = onnxsim.quantize_fp8(model, format="e4m3")
    w_init = _node_input_initializer(quant, "MatMul", 1)
    w_fp8 = onnx.numpy_helper.to_array(w_init).astype(np.float32)
    np.testing.assert_array_equal(w_fp8, [[1.0, 1.25, -1.0]])
    np.testing.assert_array_equal(w_fp8, _to_fp8_rounded_float32(w, "e4m3"))


@pytest.mark.parametrize("fmt", ["e4m3", "e5m2"])
def test_quantize_fp8_saturates_out_of_range_weight(fmt):
    # Both float8 formats here are converted with saturation: a magnitude
    # beyond the format's max finite value (including +-Inf itself) clamps
    # to it rather than becoming a NaN (E4M3FN has no infinity encoding at
    # all) or an infinity (E5M2) -- a deliberate difference from
    # ml_dtypes's own plain .astype(), which does the latter (see this
    # file's module docstring and docs/fp8-quantization.md).
    max_value = _MAX_VALUE[fmt]
    w = np.array([[1.0e10, -1.0e10, 3.0, float("inf"), float("nan")]], dtype=np.float32)
    weight = _f32(w, "W")
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    model = _model(nodes, [_vi("X", [2, 1])], [_vi("Y", [2, 5])], [weight])

    quant = onnxsim.quantize_fp8(model, format=fmt)
    onnx.checker.check_model(quant)

    w_init = _node_input_initializer(quant, "MatMul", 1)
    w_fp8 = onnx.numpy_helper.to_array(w_init).astype(np.float32)
    assert w_fp8[0, 0] == max_value
    assert w_fp8[0, 1] == -max_value
    assert w_fp8[0, 2] == 3.0
    assert w_fp8[0, 3] == max_value  # +Inf saturates, doesn't become NaN/Inf
    assert np.isnan(w_fp8[0, 4])  # NaN stays NaN


def test_quantize_fp8_skips_optional_input_default_initializer():
    # An initializer whose name is also a graph input (the ONNX "optional
    # input with a default value" convention) is left alone entirely -- see
    # quantize_fp8.h's doc comment.
    w = _f32(np.random.randn(4, 2).astype(np.float32), "W")
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    model = _model(
        nodes,
        [_vi("X", [3, 4]), _vi("W", [4, 2])],
        [_vi("Y", [3, 2])],
        [w],
    )

    quant = onnxsim.quantize_fp8(model)
    onnx.checker.check_model(quant)
    w_init = quant.graph.initializer[0]
    assert w_init.data_type == onnx.TensorProto.FLOAT  # untouched


def test_quantize_fp8_clears_stale_value_info_on_already_shape_inferred_model():
    # Same regression as test_quantize_fp16.py's identically-named test: a
    # model that already went through shape inference has its interior
    # activations' value_info pre-populated float32, and quantize_fp8 (like
    # quantize_fp16/quantize_bf16) doesn't re-run shape inference itself, so
    # it must not leave that now-wrong float32 declaration in place -- a real
    # bug found via a real torchvision model. No onnxruntime.InferenceSession
    # run here (float8 compute kernel coverage on CPUExecutionProvider is
    # narrow, see this file's module docstring) -- the value_info correctness
    # itself is the check.
    model, rng, k, n2 = _two_matmul_model()
    model = onnx.shape_inference.infer_shapes(model)
    h_before = next(vi for vi in model.graph.value_info if vi.name == "H")
    assert h_before.type.tensor_type.elem_type == onnx.TensorProto.FLOAT

    quant = onnxsim.quantize_fp8(model)
    onnx.checker.check_model(quant)

    h_after = next((vi for vi in quant.graph.value_info if vi.name == "H"), None)
    if h_after is not None:
        assert h_after.type.tensor_type.elem_type != onnx.TensorProto.FLOAT


def test_quantize_fp8_boundary_casts_execute():
    # Proves the boundary Cast wiring this pass builds is not just
    # checker-valid but actually loads and runs under a real engine.
    # Identity is used as the compute op in the middle since float8 compute
    # kernel coverage on CPUExecutionProvider is narrow -- see this file's
    # module docstring.
    nodes = [onnx.helper.make_node("Identity", ["X"], ["Y"])]
    model = _model(nodes, [_vi("X", [2, 3])], [_vi("Y", [2, 3])], [])

    quant = onnxsim.quantize_fp8(model)
    onnx.checker.check_model(quant)
    ops = _op_counts(quant)
    assert ops["Cast"] == 2
    assert ops["Identity"] == 1

    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    (out,) = _run(quant, {"X": x})
    np.testing.assert_array_equal(out, x)
