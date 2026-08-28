"""Tests for ``onnxsim.quantize_fp16`` (the ``quantize_fp16`` C++ pass).

Unlike onnxsim's other quantization tests, these models are built with
multiple chained ops (not one MatMul/Conv in isolation), since quantize_fp16
is a whole-graph transform rather than a per-node pattern match. Each model
is built via the ONNX text format, quantized, and then actually run
through ONNX Runtime -- both before and after quantization -- so the
quantized graph must load and execute under a real inference engine.
"""

import collections

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper
import onnx.shape_inference
import pytest
from onnx import parser

import onnxsim

# A bare ``import onnxruntime`` would fail collection (not skip the test) on
# platforms onnxruntime doesn't ship wheels for (e.g. s390x).
ort = pytest.importorskip("onnxruntime")


def _model(body, initializer=(), opset=13, ir_version=10):
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
    # quantize_fp16 replaces a converted weight with a *new* initializer
    # (leaving the old float32 one orphaned in the model until a follow-up
    # simplify() call prunes it -- documented behavior, matching every other
    # onnxsim quantize_* pass), so the initializer actually feeding a node
    # must be looked up by that node's *current* input name, not by
    # iterating/indexing graph.initializer blindly.
    node = next(n for n in model.graph.node if n.op_type == op_type)
    return _initializer_by_name(model, node.input[input_index])


def _assert_close(float_outputs, quant_outputs, tol=0.05):
    # float16 has ~3 decimal digits of precision; a couple of chained ops
    # accumulate more rounding than a single INT8 QuantizeLinear, so this
    # uses a looser tolerance than the INT8 quantization tests' 0.1, tuned
    # down since float16 is far more precise than INT8 to begin with.
    for f, q in zip(float_outputs, quant_outputs):
        f = np.asarray(f, dtype=np.float64).ravel()
        q = np.asarray(q, dtype=np.float64).ravel()
        assert np.all(np.isfinite(q))
        rel_l2 = np.linalg.norm(f - q) / max(np.linalg.norm(f), 1e-6)
        assert rel_l2 < tol, f"relative L2 error too large: {rel_l2:.4f}"


def _two_matmul_model():
    rng = np.random.default_rng(0)
    k, n1, n2 = 16, 12, 8
    w1 = _f32(rng.standard_normal((k, n1)) * 0.5, "W1")
    w2 = _f32(rng.standard_normal((n1, n2)) * 0.5, "W2")
    model = _model(
        f"""
        g (float[4,{k}] X) => (float[4,{n2}] Y)
        {{
          H = MatMul(X, W1)
          Hr = Relu(H)
          Y = MatMul(Hr, W2)
        }}
        """,
        initializer=[w1, w2],
    )
    return model, rng, k, n2


def test_quantize_fp16_keep_io_types():
    model, rng, k, n2 = _two_matmul_model()

    quant = onnxsim.quantize_fp16(model)
    onnx.checker.check_model(quant)

    ops = _op_counts(quant)
    assert ops["Cast"] == 2  # one boundary cast for X in, one for Y out
    assert ops["MatMul"] == 2
    assert ops["Relu"] == 1

    # The model's own declared I/O stays float32.
    assert quant.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    assert quant.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    # But the weight actually feeding the (first) MatMul is now float16 (the
    # old float32 initializer is left orphaned in the model -- see
    # _node_input_initializer's comment).
    assert (
        _node_input_initializer(quant, "MatMul", 1).data_type
        == onnx.TensorProto.FLOAT16
    )

    x = rng.standard_normal((4, k)).astype(np.float32)
    _assert_close(_run(model, {"X": x}), _run(quant, {"X": x}))


def test_quantize_fp16_no_keep_io_types():
    model, rng, k, n2 = _two_matmul_model()

    quant = onnxsim.quantize_fp16(model, keep_io_types=False)
    onnx.checker.check_model(quant)

    ops = _op_counts(quant)
    # No boundary casts needed -- the graph's own I/O is redeclared float16.
    assert ops["Cast"] == 0
    assert quant.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT16
    assert quant.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT16

    x = rng.standard_normal((4, k)).astype(np.float32)
    x16 = x.astype(np.float16)
    float_out = _run(model, {"X": x})
    quant_out = _run(quant, {"X": x16})
    _assert_close(float_out, quant_out)


def test_quantize_fp16_converts_constant_node():
    # A Constant node's embedded value is a float32 "inline initializer" --
    # FetchConstantTensor covers it the same way as a true graph
    # initializer, so it should be converted too.
    #
    # The Constant's value is a randomly-generated array, which can't be
    # cleanly spelled out as a text literal, and replacing it with a graph
    # initializer would defeat the point of this test (it specifically
    # exercises the Constant-node code path) -- so this one node is still
    # built via onnx.helper/numpy_helper and spliced into the parsed graph.
    rng = np.random.default_rng(1)
    k, n = 8, 4
    w = rng.standard_normal((k, n)).astype(np.float32)
    const_node = onnx.helper.make_node(
        "Constant",
        [],
        ["W"],
        value=onnx.numpy_helper.from_array(w, "W"),
    )
    model = _model(
        f"""
        g (float[3,{k}] X) => (float[3,{n}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """
    )
    model.graph.node.insert(0, const_node)

    quant = onnxsim.quantize_fp16(model)
    onnx.checker.check_model(quant)

    x = rng.standard_normal((3, k)).astype(np.float32)
    _assert_close(_run(model, {"X": x}), _run(quant, {"X": x}))


def test_quantize_fp16_clamps_out_of_range_weight():
    # float16's largest finite magnitude is 65504; a weight far beyond that
    # must be clamped to it, not rounded to a float16 infinity that would
    # propagate NaN/Inf through downstream compute.
    model = _model(
        """
        g (float[2,1] X) => (float[2,3] Y)
        {
          Y = MatMul(X, W)
        }
        """,
        initializer=[
            onnx.numpy_helper.from_array(
                np.array([[1.0e10, -1.0e10, 3.0]], dtype=np.float32), "W"
            )
        ],
    )

    quant = onnxsim.quantize_fp16(model)
    onnx.checker.check_model(quant)

    w_init = _node_input_initializer(quant, "MatMul", 1)
    w_arr = onnx.numpy_helper.to_array(w_init).astype(np.float32)
    assert np.all(np.isfinite(w_arr))
    assert np.max(np.abs(w_arr)) <= 65504.0

    x = np.array([[1.0], [-1.0]], dtype=np.float32)
    out = _run(quant, {"X": x})
    assert np.all(np.isfinite(out[0]))


def test_quantize_fp16_skips_optional_input_default_initializer():
    # An initializer whose name is also a graph input (the ONNX "optional
    # input with a default value" convention) is left alone entirely -- see
    # quantize_fp16.h's doc comment.
    w = _f32(np.random.randn(4, 2).astype(np.float32), "W")
    model = _model(
        """
        g (float[3,4] X, float[4,2] W) => (float[3,2] Y)
        {
          Y = MatMul(X, W)
        }
        """,
        initializer=[w],
    )

    quant = onnxsim.quantize_fp16(model)
    onnx.checker.check_model(quant)
    w_init = quant.graph.initializer[0]
    assert w_init.data_type == onnx.TensorProto.FLOAT  # untouched


def test_quantize_fp16_clears_stale_value_info_on_already_shape_inferred_model():
    # A model that already went through shape inference (e.g. onnxsim's own
    # simplify(), or just onnx.shape_inference.infer_shapes() directly, as
    # here) has its interior activations' value_info pre-populated float32.
    # quantize_fp16 doesn't re-run shape inference itself (see the file's own
    # doc comment), so it must not leave that now-wrong float32 declaration
    # in place for a tensor the graph actually produces as float16 -- ONNX
    # Runtime's own load-time type-checking rejects a model with a *wrong*
    # value_info outright (unlike a merely absent one, which every conformant
    # consumer infers fresh). This was a real bug: found by running a real
    # torchvision model (already simplify()'d) through quantize_fp16 and
    # onnxruntime.InferenceSession.
    model, rng, k, n2 = _two_matmul_model()
    model = onnx.shape_inference.infer_shapes(model)
    # Sanity: shape inference actually populated a float32 value_info for the
    # interior activation "H" (or this test would not exercise the bug).
    h_before = next(vi for vi in model.graph.value_info if vi.name == "H")
    assert h_before.type.tensor_type.elem_type == onnx.TensorProto.FLOAT

    quant = onnxsim.quantize_fp16(model)
    onnx.checker.check_model(quant)

    # No value_info entry for "H" may declare it float32 anymore -- either
    # it's absent (cleared, the expected outcome) or correctly float16.
    h_after = next((vi for vi in quant.graph.value_info if vi.name == "H"), None)
    if h_after is not None:
        assert h_after.type.tensor_type.elem_type != onnx.TensorProto.FLOAT

    # The real regression check: ONNX Runtime must actually load and run the
    # quantized model, not just pass the (more lenient) checker.
    x = rng.standard_normal((4, k)).astype(np.float32)
    _assert_close(_run(model, {"X": x}), _run(quant, {"X": x}))
