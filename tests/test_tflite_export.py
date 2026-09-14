"""Tests for converting a (simplified) ONNX model to TensorFlow Lite.

onnxsim can hand its cleaned-up ``ModelProto`` to a hand-written ONNX-to-TensorFlow
translator and produce a TFLite model via ``tf.lite.TFLiteConverter``
(``onnxsim.export_tflite`` / the ``--emit-tflite`` CLI flag, implemented in
``onnxsim/tflite_export.py``). There is no maintained "convert this ONNX model" entry
point to lean on (``onnx-tf``/``onnx-tensorflow`` has been unmaintained for years), so
this translator -- not TensorFlow -- is what maps ONNX ops onto TF ops.

TensorFlow is heavy and not part of onnxsim's test requirements, so -- exactly like
``tests/test_coreml_export.py`` and ``tests/test_mlir_export.py`` -- the whole module
is skipped when it is not installed.

Unlike Core ML (whose MIL constant-folds an all-initializer graph so a converted
model's numeric behavior can be checked without Apple's runtime), TFLite conversion
always needs ``tf.lite.Interpreter`` to actually run the produced flatbuffer, and that
runtime exists on every platform TensorFlow supports (Linux/macOS/Windows) -- so these
tests run the interpreter directly and compare against onnxruntime.
"""

import numpy as np
import onnx
import pytest
from onnx import numpy_helper, parser

pytest.importorskip("tensorflow", reason="tensorflow is not installed")

import onnxruntime as ort  # noqa: E402  (imported after the tensorflow availability check)
import tensorflow as tf  # noqa: E402

import onnxsim  # noqa: E402
from onnxsim import tflite_export  # noqa: E402


def _model(
    body: str, initializer=(), opset: int = 17, ir_version: int = 8
) -> onnx.ModelProto:
    model = parser.parse_model(
        f'<ir_version: {ir_version}, opset_import: ["" : {opset}]> {body}'
    )
    model.graph.initializer.extend(initializer)
    return model


def _relu_model() -> onnx.ModelProto:
    model = _model(
        """
        relu (float[2,3] x) => (float[2,3] y)
        {
            y = Relu (x)
        }
        """
    )
    onnx.checker.check_model(model)
    return model


def _foldable_model() -> onnx.ModelProto:
    """Add(input, const_a + const_b) -- the inner Add folds to one constant."""
    a = numpy_helper.from_array(np.array([1, 2, 3], np.float32), name="a")
    b = numpy_helper.from_array(np.array([4, 5, 6], np.float32), name="b")
    model = _model(
        """
        foldadd (float[3] x) => (float[3] y)
        {
            ab = Add (a, b)
            y = Add (x, ab)
        }
        """,
        initializer=[a, b],
    )
    onnx.checker.check_model(model)
    return model


def _cnn_model() -> onnx.ModelProto:
    """Conv -> BatchNorm -> Relu -> GlobalAveragePool -> Flatten -> Gemm -> Softmax."""
    rng = np.random.RandomState(0)
    w = numpy_helper.from_array(rng.randn(4, 3, 3, 3).astype(np.float32), name="w")
    b = numpy_helper.from_array(np.zeros(4, np.float32), name="b")
    scale = numpy_helper.from_array(
        (0.5 + rng.rand(4)).astype(np.float32), name="scale"
    )
    bn_bias = numpy_helper.from_array(rng.randn(4).astype(np.float32), name="bn_bias")
    mean = numpy_helper.from_array(rng.randn(4).astype(np.float32) * 0.1, name="mean")
    var = numpy_helper.from_array((0.5 + rng.rand(4)).astype(np.float32), name="var")
    gw = numpy_helper.from_array(rng.randn(4, 4).astype(np.float32), name="gw")
    gb = numpy_helper.from_array(rng.randn(4).astype(np.float32), name="gb")
    model = _model(
        """
        cnn (float[1,3,8,8] x) => (float[1,4] y)
        {
            conv_out = Conv <kernel_shape=[3,3], pads=[1,1,1,1]> (x, w, b)
            bn_out = BatchNormalization (conv_out, scale, bn_bias, mean, var)
            relu_out = Relu (bn_out)
            gap_out = GlobalAveragePool (relu_out)
            flat_out = Flatten <axis=1> (gap_out)
            gemm_out = Gemm <transB=1> (flat_out, gw, gb)
            y = Softmax <axis=-1> (gemm_out)
        }
        """,
        initializer=[w, b, scale, bn_bias, mean, var, gw, gb],
    )
    onnx.checker.check_model(model)
    return model


def _run_tflite(tflite_model: bytes, inputs: dict):
    interp = tf.lite.Interpreter(model_content=tflite_model)
    interp.allocate_tensors()
    in_details = {d["name"]: d for d in interp.get_input_details()}
    # TFLite input tensor names get a suffix from tf.function tracing (e.g.
    # "x:0"), so match by position instead of by exact name when there's a
    # single input -- the common case in these tests.
    if len(in_details) == 1 and len(inputs) == 1:
        (detail,) = in_details.values()
        (value,) = inputs.values()
        interp.set_tensor(detail["index"], value)
    else:
        for name, value in inputs.items():
            interp.set_tensor(in_details[name]["index"], value)
    interp.invoke()
    return [interp.get_tensor(d["index"]) for d in interp.get_output_details()]


def _assert_matches_onnxruntime(model: onnx.ModelProto, inputs: dict, **export_kwargs):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    expected = sess.run(None, inputs)
    tflite_model = tflite_export.export_tflite(model, **export_kwargs)
    actual = _run_tflite(tflite_model, inputs)
    for e, a in zip(expected, actual):
        np.testing.assert_allclose(e, a, rtol=1e-4, atol=1e-4)
    return actual


# ---------------------------------------------------------------------------
# Basic conversion
# ---------------------------------------------------------------------------


def test_has_tensorflow_true_here():
    assert tflite_export.has_tensorflow() is True


def test_export_returns_tflite_bytes():
    tflite_model = onnxsim.export_tflite(_relu_model())
    assert isinstance(tflite_model, bytes)
    assert len(tflite_model) > 0


def test_export_writes_file(tmp_path):
    out = tmp_path / "relu.tflite"
    onnxsim.export_tflite(_relu_model(), str(out))
    assert out.is_file()
    assert out.read_bytes() == onnxsim.export_tflite(_relu_model())


def test_export_of_simplified_model():
    model = _foldable_model()
    simplified, ok = onnxsim.simplify(model)
    assert ok
    # The redundant const+const Add is folded away by onnxsim.
    assert [n.op_type for n in simplified.graph.node].count("Add") == 1
    x = np.random.RandomState(0).randn(3).astype(np.float32)
    _assert_matches_onnxruntime(simplified, {"x": x})


def test_convert_to_tflite_matches_export_tflite():
    model = _relu_model()
    a = tflite_export.convert_to_tflite(model)
    b = onnxsim.export_tflite(model)
    assert a == b


# ---------------------------------------------------------------------------
# A small CNN pipeline, exercising conv/norm/pool/gemm/softmax together
# ---------------------------------------------------------------------------


def test_cnn_pipeline_matches_onnxruntime():
    model = _cnn_model()
    x = np.random.RandomState(1).randn(1, 3, 8, 8).astype(np.float32)
    _assert_matches_onnxruntime(model, {"x": x})


def test_dynamic_input_raises():
    model = _model(
        """
        dyn (float[N,3] x) => (float[N,3] y)
        {
            y = Relu (x)
        }
        """
    )
    with pytest.raises(RuntimeError, match="dynamic dimension"):
        onnxsim.export_tflite(model)


def test_unsupported_op_raises_naming_the_op():
    model = _model(
        """
        unsup (float[2,3] x) => (float[2,3] y)
        {
            y = Selu (x)
        }
        """
    )
    with pytest.raises(RuntimeError, match="Selu"):
        onnxsim.export_tflite(model)


# ---------------------------------------------------------------------------
# Grouped/depthwise conv and pooling with padding
# ---------------------------------------------------------------------------


def test_depthwise_conv_and_pool_with_padding_match_onnxruntime():
    # AveragePool's default count_include_pad=0 excludes the padded zeros from the
    # average -- TF/TFLite's own avg_pool2d has no such option and always divides
    # by the full window area, so tflite_export computes a per-position divisor
    # correction. This model has non-trivial padding on both AveragePool and
    # MaxPool (whose padded elements must not affect the max either) together with
    # a depthwise (group == in_channels) Conv, to exercise all of that at once.
    rng = np.random.RandomState(2)
    dw_w = numpy_helper.from_array(rng.randn(4, 1, 3, 3).astype(np.float32), name="dwW")
    pw_w = numpy_helper.from_array(
        (0.1 * rng.randn(8, 8, 1, 1)).astype(np.float32), name="pwW"
    )
    pw_b = numpy_helper.from_array(rng.randn(8).astype(np.float32), name="pwB")
    model = _model(
        """
        dw (float[1,4,9,9] x) => (float[1,8,3,3] y)
        {
            dw = Conv <kernel_shape=[3,3], strides=[1,1], pads=[1,1,1,1], group=4> (x, dwW)
            lr = LeakyRelu <alpha=0.1> (dw)
            ap = AveragePool <kernel_shape=[3,3], strides=[3,3], pads=[1,1,1,1], count_include_pad=0> (lr)
            mp = MaxPool <kernel_shape=[3,3], strides=[3,3], pads=[1,1,1,1]> (lr)
            cc = Concat <axis=1> (ap, mp)
            y = Conv <kernel_shape=[1,1]> (cc, pwW, pwB)
        }
        """,
        initializer=[dw_w, pw_w, pw_b],
    )
    onnx.checker.check_model(model)
    x = rng.randn(1, 4, 9, 9).astype(np.float32)
    _assert_matches_onnxruntime(model, {"x": x})


def test_grouped_conv_non_depthwise_matches_onnxruntime():
    # group > 1 but not the group-per-channel depthwise case: exercises the
    # split-convolve-concat fallback path.
    rng = np.random.RandomState(3)
    w = numpy_helper.from_array(rng.randn(8, 2, 3, 3).astype(np.float32), name="w")
    b = numpy_helper.from_array(rng.randn(8).astype(np.float32), name="b")
    model = _model(
        """
        grouped (float[1,4,6,6] x) => (float[1,8,6,6] y)
        {
            y = Conv <kernel_shape=[3,3], pads=[1,1,1,1], group=2> (x, w, b)
        }
        """,
        initializer=[w, b],
    )
    onnx.checker.check_model(model)
    x = rng.randn(1, 4, 6, 6).astype(np.float32)
    _assert_matches_onnxruntime(model, {"x": x})


# ---------------------------------------------------------------------------
# Reshape/shape-manipulation chain
# ---------------------------------------------------------------------------


def test_reshape_squeeze_transpose_slice_gather_chain_matches_onnxruntime():
    model = _model(
        """
        shapes (float[2,3,4] x) => (float[1,2,1] y)
        <int64[3] shp = {2,12,1}, int64[1] sq_axes = {2}, int64[1] usq_axes = {2},
         int64[2] starts = {0,1}, int64[2] ends = {2,3}, int64[2] axes2 = {0,1},
         int64[2] steps = {1,1}, int64[1] gidx = {1}, int64[4] padsv = {0,0,1,1}>
        {
            r = Reshape (x, shp)
            sq = Squeeze (r, sq_axes)
            tp = Transpose <perm=[1,0]> (sq)
            sm = Softmax <axis=-1> (tp)
            pd = Pad <mode="constant"> (sm, padsv)
            sl = Slice (pd, starts, ends, axes2, steps)
            g = Gather <axis=0> (sl, gidx)
            y = Unsqueeze (g, usq_axes)
        }
        """
    )
    onnx.checker.check_model(model)
    x = np.random.RandomState(4).randn(2, 3, 4).astype(np.float32)
    _assert_matches_onnxruntime(model, {"x": x})


def test_slice_negative_step_reverses_full_axis():
    # Regression test: tf.strided_slice wraps a negative `end` the same way numpy
    # indexing does (silently turning ONNX's "-1 meaning off the start" sentinel
    # back into "the last element", producing an empty slice) unless `end_mask` is
    # set for that axis -- see the comment in tflite_export._op_slice.
    x = numpy_helper.from_array(np.arange(5, dtype=np.float32), name="x")
    starts = numpy_helper.from_array(np.array([4], np.int64), name="starts")
    ends = numpy_helper.from_array(np.array([-100], np.int64), name="ends")
    axes = numpy_helper.from_array(np.array([0], np.int64), name="axes")
    steps = numpy_helper.from_array(np.array([-1], np.int64), name="steps")
    model = _model(
        "slicerev () => (float[5] out) { out = Slice (x, starts, ends, axes, steps) }",
        initializer=[x, starts, ends, axes, steps],
    )
    onnx.checker.check_model(model)
    _assert_matches_onnxruntime(model, {})


def test_slice_end_sentinel_survives_int64_downcast():
    # ONNX graphs routinely use INT64_MAX as a Slice `ends` sentinel meaning "to
    # the end of this axis". This translator downcasts int64 *tensors* to int32
    # for TFLite, but Slice's bounds are read from the original (pre-downcast)
    # numpy constant tracked alongside each traced tensor, so the sentinel's exact
    # value survives -- a plain `.astype(int32)` on it would instead wrap
    # INT64_MAX around to -1 and silently drop the last element.
    x = numpy_helper.from_array(np.arange(8, dtype=np.float32), name="x")
    starts = numpy_helper.from_array(np.array([3], np.int64), name="starts")
    ends = numpy_helper.from_array(
        np.array([9223372036854775807], np.int64), name="ends"
    )
    model = _model(
        "slicesentinel () => (float[5] out) { out = Slice (x, starts, ends) }",
        initializer=[x, starts, ends],
    )
    onnx.checker.check_model(model)
    _assert_matches_onnxruntime(model, {})


def test_uneven_split_matches_onnxruntime():
    x = np.random.RandomState(5).randn(2, 6).astype(np.float32)
    splitv = numpy_helper.from_array(np.array([2, 4], np.int64), name="splitv")
    model = _model(
        "split (float[2,6] x) => (float[2,2] a, float[2,4] b) "
        "{ a, b = Split <axis=1> (x, splitv) }",
        initializer=[splitv],
    )
    onnx.checker.check_model(model)
    _assert_matches_onnxruntime(model, {"x": x})


# ---------------------------------------------------------------------------
# backend= dispatch (the "onnx2tf" backend itself is covered by
# test_onnx2tf_export.py, skipped when onnx2tf isn't installed)
# ---------------------------------------------------------------------------


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown backend"):
        onnxsim.export_tflite(_relu_model(), backend="bogus")


def test_builtin_backend_rejects_backend_specific_kwargs():
    with pytest.raises(TypeError, match="unexpected keyword arguments"):
        onnxsim.export_tflite(
            _relu_model(), keep_ncw_or_nchw_or_ncdhw_input_names=["x"]
        )


# ---------------------------------------------------------------------------
# PRelu / Dropout (translator additions; PRelu lowers to Edge TPU-mappable
# Relu/Minimum/Mul/Add, Dropout in inference mode is an identity)
# ---------------------------------------------------------------------------


def test_prelu_nchw_slope_matches_onnxruntime():
    slope = numpy_helper.from_array(
        np.array([0.1, 0.5, -0.25], np.float32), name="slope"
    )
    model = _model(
        "pr (float[2,3] x) => (float[2,3] y) { y = PRelu (x, slope) }",
        initializer=[slope],
    )
    onnx.checker.check_model(model)
    x = np.random.RandomState(6).randn(2, 3).astype(np.float32)
    _assert_matches_onnxruntime(model, {"x": x})


def test_prelu_image_slope_matches_onnxruntime():
    # Channel-shaped slope on an NCHW input: the translator must broadcast it
    # explicitly (TF's implicit trailing-alignment would misplace it).
    rng = np.random.RandomState(7)
    slope = numpy_helper.from_array(
        (0.1 * rng.randn(3, 1, 1)).astype(np.float32), name="slope"
    )
    model = _model(
        "prim (float[1,3,4,4] x) => (float[1,3,4,4] y) { y = PRelu (x, slope) }",
        initializer=[slope],
    )
    onnx.checker.check_model(model)
    x = rng.randn(1, 3, 4, 4).astype(np.float32)
    _assert_matches_onnxruntime(model, {"x": x})


def test_dropout_inference_is_identity():
    model = _model("do (float[2,3] x) => (float[2,3] y) { y = Dropout (x) }")
    onnx.checker.check_model(model)
    x = np.random.RandomState(8).randn(2, 3).astype(np.float32)
    _assert_matches_onnxruntime(model, {"x": x})


def test_dropout_training_mode_raises():
    training = numpy_helper.from_array(np.array(True), name="training")
    model = _model(
        "dot (float[2,3] x) => (float[2,3] y) { y = Dropout (x, ratio, training) }",
        initializer=[
            numpy_helper.from_array(np.array(0.5, np.float32), name="ratio"),
            training,
        ],
    )
    with pytest.raises(RuntimeError, match="training_mode"):
        onnxsim.export_tflite(model)


# ---------------------------------------------------------------------------
# Full-integer quantization (Edge TPU prerequisite)
# ---------------------------------------------------------------------------


def test_int8_quantize_produces_quantized_io():
    model = _relu_model()
    tflite_model = onnxsim.export_tflite(
        model,
        int8_quantize=True,
        inference_io_dtype="uint8",
        num_calibration_samples=5,
    )
    assert isinstance(tflite_model, bytes) and len(tflite_model) > 0
    interp = tf.lite.Interpreter(model_content=tflite_model)
    interp.allocate_tensors()
    in_dtype = interp.get_input_details()[0]["dtype"]
    out_dtype = interp.get_output_details()[0]["dtype"]
    assert in_dtype == np.uint8
    assert out_dtype == np.uint8


def test_int8_quantize_rejects_conflicts():
    with pytest.raises(ValueError, match="mutually exclusive"):
        onnxsim.export_tflite(
            _relu_model(), optimizations=["DEFAULT"], int8_quantize=True
        )
    with pytest.raises(ValueError, match="requires int8_quantize=True"):
        onnxsim.export_tflite(_relu_model(), inference_io_dtype="uint8")


def test_invalid_io_layout_raises_without_tensorflow():
    with pytest.raises(ValueError, match="io_layout"):
        onnxsim.export_tflite(_relu_model(), io_layout="nchc")


def test_onnx2tf_backend_rejects_nhwc_layout():
    with pytest.raises(TypeError, match="io_layout"):
        onnxsim.export_tflite(_relu_model(), backend="onnx2tf", io_layout="nhwc")


# ---------------------------------------------------------------------------
# io_layout="nhwc" (channel-last 4-D tensors end to end; no transposes)
# ---------------------------------------------------------------------------


def _tflite_op_counts(tflite_model: bytes):
    litert = pytest.importorskip("ai_edge_litert", reason="LiteRT is not installed")
    from ai_edge_litert.tools import flatbuffer_utils as fbu

    del litert
    model = fbu.convert_bytearray_to_object(bytearray(tflite_model))
    names = {v: k for k, v in vars(fbu.BuiltinOperator).items() if isinstance(v, int)}
    counts = {}
    for subgraph in model.subgraphs:
        for op in subgraph.operators:
            code = model.operatorCodes[op.opcodeIndex]
            builtin = int(fbu.get_builtin_code_from_operator_code(code))
            name = names.get(builtin, builtin)
            counts[name] = counts.get(name, 0) + 1
    return counts


def _assert_nhwc_matches_onnxruntime(model, inputs_nchw, **export_kwargs):
    """Like ``_assert_matches_onnxruntime`` but converts with ``io_layout="nhwc"``.

    Feeds are given in ONNX (NCHW) order and transposed for the channel-last
    model; 4-D outputs are transposed back before comparing against
    onnxruntime, so the assertion runs in NCHW space either way.
    """
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    expected = sess.run(None, inputs_nchw)
    feeds_nhwc = {
        name: (np.transpose(v, (0, 2, 3, 1)).copy() if v.ndim == 4 else v)
        for name, v in inputs_nchw.items()
    }
    tflite_model = onnxsim.export_tflite(model, io_layout="nhwc", **export_kwargs)
    actual = _run_tflite(tflite_model, feeds_nhwc)
    assert len(expected) == len(actual)
    for e, a in zip(expected, actual):
        if a.ndim == 4:
            a = np.transpose(a, (0, 3, 1, 2))
        np.testing.assert_allclose(e, a, rtol=1e-4, atol=1e-4)
    return tflite_model


def _conv_weights(rng, name, shape):
    return numpy_helper.from_array(rng.randn(*shape).astype(np.float32), name=name)


def test_nhwc_conv_chain_has_no_transposes():
    rng = np.random.RandomState(10)
    w1 = _conv_weights(rng, "w1", (4, 3, 3, 3))
    b1 = numpy_helper.from_array(np.zeros(4, np.float32), name="b1")
    w2 = _conv_weights(rng, "w2", (4, 4, 3, 3))
    b2 = numpy_helper.from_array(np.zeros(4, np.float32), name="b2")
    model = _model(
        """
        chain (float[1,3,8,8] x) => (float[1,4,8,8] y)
        {
            c1 = Conv <kernel_shape=[3,3], pads=[1,1,1,1]> (x, w1, b1)
            r1 = Relu (c1)
            y = Conv <kernel_shape=[3,3], pads=[1,1,1,1]> (r1, w2, b2)
        }
        """,
        initializer=[w1, b1, w2, b2],
    )
    onnx.checker.check_model(model)
    x = rng.randn(1, 3, 8, 8).astype(np.float32)
    blob = _assert_nhwc_matches_onnxruntime(model, {"x": x})
    assert _tflite_op_counts(blob).get("TRANSPOSE", 0) == 0


def test_nhwc_concat_topology_has_no_transposes():
    # The NCHW translator needs 5 transposes here (concat blocks TF's
    # transpose push-through); NHWC-native concatenation needs none.
    rng = np.random.RandomState(11)
    wa = _conv_weights(rng, "wa", (4, 4, 3, 3))
    ba = numpy_helper.from_array(np.zeros(4, np.float32), name="ba")
    wb = _conv_weights(rng, "wb", (4, 4, 3, 3))
    bb = numpy_helper.from_array(np.zeros(4, np.float32), name="bb")
    wc = _conv_weights(rng, "wc", (4, 8, 1, 1))
    bc = numpy_helper.from_array(np.zeros(4, np.float32), name="bc")
    model = _model(
        """
        conc (float[1,4,8,8] x) => (float[1,4,8,8] y)
        {
            a = Conv <kernel_shape=[3,3], pads=[1,1,1,1]> (x, wa, ba)
            b = Conv <kernel_shape=[3,3], pads=[1,1,1,1]> (x, wb, bb)
            c = Concat <axis=1> (a, b)
            y = Conv <kernel_shape=[1,1]> (c, wc, bc)
        }
        """,
        initializer=[wa, ba, wb, bb, wc, bc],
    )
    onnx.checker.check_model(model)
    x = rng.randn(1, 4, 8, 8).astype(np.float32)
    blob = _assert_nhwc_matches_onnxruntime(model, {"x": x})
    assert _tflite_op_counts(blob).get("TRANSPOSE", 0) == 0


def test_nhwc_batchnorm_pool_and_depthwise():
    rng = np.random.RandomState(12)
    dw_w = numpy_helper.from_array(rng.randn(4, 1, 3, 3).astype(np.float32), name="dwW")
    scale = numpy_helper.from_array(np.ones(4, np.float32), name="scale")
    bn_bias = numpy_helper.from_array(np.zeros(4, np.float32), name="bn_bias")
    mean = numpy_helper.from_array(np.zeros(4, np.float32), name="mean")
    var = numpy_helper.from_array(np.ones(4, np.float32), name="var")
    model = _model(
        """
        bnp (float[1,4,9,9] x) => (float[1,4,3,3] y)
        {
            dw = Conv <kernel_shape=[3,3], pads=[1,1,1,1], group=4> (x, dwW)
            bn = BatchNormalization (dw, scale, bn_bias, mean, var)
            y = AveragePool <kernel_shape=[3,3], strides=[3,3], pads=[1,1,1,1]> (bn)
        }
        """,
        initializer=[dw_w, scale, bn_bias, mean, var],
    )
    onnx.checker.check_model(model)
    x = rng.randn(1, 4, 9, 9).astype(np.float32)
    _assert_nhwc_matches_onnxruntime(model, {"x": x})


def test_nhwc_softmax_and_reduce_axes():
    model = _model(
        """
        axes (float[1,4,3,3] x) => (float[1,4,1,1] y)
        {
            s = Softmax <axis=1> (x)
            y = ReduceMean <axes=[2,3], keepdims=1> (s)
        }
        """
    )
    onnx.checker.check_model(model)
    x = np.random.RandomState(13).randn(1, 4, 3, 3).astype(np.float32)
    _assert_nhwc_matches_onnxruntime(model, {"x": x})


def test_nhwc_transpose_split_slice_gather():
    # Fully asymmetric dims (C != H != W) so a misplaced axis can't hide.
    model = _model(
        """
        mods (float[1,3,5,7] x) => (float[1,2,1,1] y)
        <int64[1] gidx = {1},
         int64[3] starts = {0,1,0}, int64[3] ends = {2,4,2}, int64[3] axes3 = {0,1,2},
         int64[2] splitv = {1,2}>
        {
            tp = Transpose <perm=[0,2,3,1]> (x)
            a, b = Split <axis=3> (tp, splitv)
            sl = Slice (a, starts, ends, axes3)
            g = Gather <axis=1> (sl, gidx)
            y = Transpose (g)
        }
        """
    )
    onnx.checker.check_model(model)
    x = np.random.RandomState(14).randn(1, 3, 5, 7).astype(np.float32)
    _assert_nhwc_matches_onnxruntime(model, {"x": x})


def test_nhwc_reshape_flatten_gemm_head():
    # Flatten crosses the NHWC backbone into the 2-D head through an NCHW
    # island; the model output is 2-D either way.
    rng = np.random.RandomState(15)
    w = _conv_weights(rng, "w", (4, 2, 3, 3))
    b = numpy_helper.from_array(np.zeros(4, np.float32), name="b")
    gw = numpy_helper.from_array(rng.randn(3, 64).astype(np.float32), name="gw")
    gb = numpy_helper.from_array(rng.randn(3).astype(np.float32), name="gb")
    model = _model(
        """
        head (float[1,2,4,4] x) => (float[1,3] y)
        {
            c = Conv <kernel_shape=[3,3], pads=[1,1,1,1]> (x, w, b)
            f = Flatten <axis=1> (c)
            y = Gemm <transB=1> (f, gw, gb)
        }
        """,
        initializer=[w, b, gw, gb],
    )
    onnx.checker.check_model(model)
    x = rng.randn(1, 2, 4, 4).astype(np.float32)
    _assert_nhwc_matches_onnxruntime(model, {"x": x})


def test_nhwc_reshape_4d_to_4d_and_matmul_island():
    # 4-D MatMul contracts the last two physical axes, so it runs in an NCHW
    # island. Numerics run with the XNNPACK delegate disabled: TF 2.21 /
    # ai-edge-litert 2.2.0 mis-executes transpose->BATCH_MATMUL under default
    # buffer planning (verified: reference kernels match onnxruntime to 1e-7,
    # perms/options decode correctly, preserve_all_tensors also matches -- the
    # model bytes are right, the delegate reuses the transpose output buffer
    # while BATCH_MATMUL still reads it).
    pytest.importorskip("ai_edge_litert", reason="LiteRT is not installed")
    rng = np.random.RandomState(16)
    w = _conv_weights(rng, "w", (2, 2, 1, 1))
    b = numpy_helper.from_array(np.zeros(2, np.float32), name="b")
    model = _model(
        """
        rm (float[1,2,2,2] x) => (float[1,2,2,2] y)
        <int64[4] shp = {1,2,2,2}>
        {
            c = Conv <kernel_shape=[1,1]> (x, w, b)
            r = Reshape (c, shp)
            y = MatMul (r, r)
        }
        """,
        initializer=[w, b],
    )
    onnx.checker.check_model(model)
    x = rng.randn(1, 2, 2, 2).astype(np.float32)
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (expected,) = sess.run(None, {"x": x})
    blob = onnxsim.export_tflite(model, io_layout="nhwc")
    from ai_edge_litert import interpreter as lit_interpreter

    interp = lit_interpreter.Interpreter(
        model_content=blob,
        experimental_op_resolver_type=(
            lit_interpreter.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        ),
    )
    interp.allocate_tensors()
    (detail,) = interp.get_input_details()
    interp.set_tensor(detail["index"], np.transpose(x, (0, 2, 3, 1)).copy())
    interp.invoke()
    (out_detail,) = interp.get_output_details()
    actual = np.transpose(interp.get_tensor(out_detail["index"]), (0, 3, 1, 2))
    np.testing.assert_allclose(expected, actual, rtol=1e-4, atol=1e-4)


def test_nhwc_prelu_tile_pad_squeeze_unsqueeze():
    rng = np.random.RandomState(17)
    slope = numpy_helper.from_array(
        (0.1 * rng.randn(3, 1, 1)).astype(np.float32), name="slope"
    )
    model = _model(
        """
        misc (float[1,3,4,4] x) => (float[1,6,3,4] y)
        <int64[4] reps = {1,2,1,2}, int64[8] padsv = {0,0,1,0,0,0,1,0},
         int64[1] sqax = {0}, int64[1] usqax = {0}>
        {
            p = PRelu (x, slope)
            t = Tile (p, reps)
            pd = Pad <mode="constant"> (t, padsv)
            us = Unsqueeze (pd, usqax)
            sq = Squeeze (us, sqax)
            y = AveragePool <kernel_shape=[2,2], strides=[2,2]> (sq)
        }
        """,
        initializer=[slope],
    )
    onnx.checker.check_model(model)
    x = rng.randn(1, 3, 4, 4).astype(np.float32)
    _assert_nhwc_matches_onnxruntime(model, {"x": x})


def test_nhwc_shape_to_reshape_chain():
    # Shape speaks ONNX-logical (NCHW) dims in nhwc mode too, so a Shape-fed
    # Reshape target needs no reordering inside the Reshape island -- including
    # through a Gather/Concat chain, and for the direct Shape->Reshape form.
    model = _model(
        """
        shch (float[1,2,2,4] x) => (float[1,4,2,2] y, float[1,2,2,4] z)
        <int64[1] gidx = {0}, int64[1] four = {4}, int64[2] twos = {2,2}>
        {
            sh = Shape (x)
            g0 = Gather <axis=0> (sh, gidx)
            tgt = Concat <axis=0> (g0, four, twos)
            y = Reshape (x, tgt)
            z = Reshape (x, sh)
        }
        """
    )
    onnx.checker.check_model(model)
    x = np.random.RandomState(18).randn(1, 2, 2, 4).astype(np.float32)
    _assert_nhwc_matches_onnxruntime(model, {"x": x})


def test_nhwc_mixed_rank_broadcast_uses_island():
    # A [C]-shaped vector broadcasts over W in NCHW but over C in NHWC, so the
    # mixed-rank Add runs in an NCHW island and keeps NCHW semantics.
    model = _model(
        """
        bc (float[1,2,2,3] x, float[3] v) => (float[1,2,2,3] y)
        {
            y = Add (x, v)
        }
        """
    )
    onnx.checker.check_model(model)
    rng = np.random.RandomState(19)
    x = rng.randn(1, 2, 2, 3).astype(np.float32)
    v = rng.randn(3).astype(np.float32)
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (expected,) = sess.run(None, {"x": x, "v": v})
    tflite_model = onnxsim.export_tflite(model, io_layout="nhwc")
    interp = tf.lite.Interpreter(model_content=tflite_model)
    interp.allocate_tensors()
    details = {d["name"]: d for d in interp.get_input_details()}
    assert len(details) == 2
    # NCHW input x is still fed NHWC (transposed); the vector is unchanged.
    (dx,) = [d for n, d in details.items() if list(d["shape"]) == [1, 2, 3, 2]]
    (dv,) = [d for n, d in details.items() if list(d["shape"]) == [3]]
    interp.set_tensor(dx["index"], np.transpose(x, (0, 2, 3, 1)).copy())
    interp.set_tensor(dv["index"], v)
    interp.invoke()
    (actual,) = [interp.get_tensor(d["index"]) for d in interp.get_output_details()]
    np.testing.assert_allclose(
        expected, np.transpose(actual, (0, 3, 1, 2)), rtol=1e-4, atol=1e-4
    )


def test_nhwc_public_io_order():
    model = _model(
        """
        io (float[1,3,4,5] x) => (float[1,3,4,5] y)
        {
            y = Relu (x)
        }
        """
    )
    tflite_model = onnxsim.export_tflite(model, io_layout="nhwc")
    interp = tf.lite.Interpreter(model_content=tflite_model)
    interp.allocate_tensors()
    assert list(interp.get_input_details()[0]["shape"]) == [1, 4, 5, 3]
    assert list(interp.get_output_details()[0]["shape"]) == [1, 4, 5, 3]
    nchw_model = onnxsim.export_tflite(model)
    interp = tf.lite.Interpreter(model_content=nchw_model)
    interp.allocate_tensors()
    assert list(interp.get_input_details()[0]["shape"]) == [1, 3, 4, 5]
