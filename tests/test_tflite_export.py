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
