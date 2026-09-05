"""Tests for the ``backend="onnx2tf"`` TFLite export path
(``onnxsim/onnx2tf_export.py``).

onnx2tf (https://github.com/PINTO0309/onnx2tf) is a separate, actively maintained
project that converts ONNX to TensorFlow/TFLite with far broader op coverage than
onnxsim's own builtin translator (``tflite_export.py``) -- at the cost of a much
heavier dependency chain and changing the model's public input/output tensor layout
to channel-last by default. onnxsim wires it up as an alternate backend rather than
replacing the builtin translator; see ``onnx2tf_export.py``'s module docstring for
the full trade-off.

onnx2tf is heavy and not part of onnxsim's test requirements, so -- like
``tests/test_coreml_export.py`` and ``tests/test_tflite_export.py`` -- the whole
module is skipped when it is not installed.
"""

import numpy as np
import onnx
import pytest
from onnx import numpy_helper, parser

pytest.importorskip("onnx2tf", reason="onnx2tf is not installed")
pytest.importorskip("tensorflow", reason="tensorflow is not installed")

import onnxruntime as ort  # noqa: E402  (imported after the availability checks)
import tensorflow as tf  # noqa: E402

import onnxsim  # noqa: E402
from onnxsim import onnx2tf_export, tflite_export  # noqa: E402


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


def _run_tflite(tflite_model: bytes, x: np.ndarray) -> np.ndarray:
    interp = tf.lite.Interpreter(model_content=tflite_model)
    interp.allocate_tensors()
    (inp,) = interp.get_input_details()
    (out,) = interp.get_output_details()
    # onnx2tf converts every tensor of rank >= 3 to a channel-last layout by
    # default, so a 4-D NCHW ONNX input becomes an NHWC TFLite input -- transpose
    # to match whenever the produced model's input shape says it did.
    x_in = np.transpose(x, (0, 2, 3, 1)) if list(inp["shape"]) != list(x.shape) else x
    interp.set_tensor(inp["index"], x_in)
    interp.invoke()
    return interp.get_tensor(out["index"])


def test_has_onnx2tf_true_here():
    assert onnx2tf_export.has_onnx2tf() is True


def test_export_returns_tflite_bytes():
    tflite_model = onnx2tf_export.export_tflite_via_onnx2tf(_relu_model())
    assert isinstance(tflite_model, bytes)
    assert len(tflite_model) > 0


def test_export_writes_file(tmp_path):
    out = tmp_path / "relu.tflite"
    onnx2tf_export.export_tflite_via_onnx2tf(_relu_model(), str(out))
    assert out.is_file()


def test_no_network_dependency_and_no_stray_files(tmp_path, monkeypatch):
    # Regression test: onnx2tf.convert() unconditionally tries to download a
    # sample-image .npy from GitHub releases for internal per-op verification
    # whenever a graph input looks image-shaped (see onnx2tf_export.py's module
    # docstring) -- this must never reach the network. Running with cwd pointed
    # at an empty tmp_path also checks that onnx2tf's own housekeeping file
    # writes land in the isolated scratch directory, not the caller's cwd.
    monkeypatch.chdir(tmp_path)
    onnx2tf_export.export_tflite_via_onnx2tf(_cnn_model())
    assert list(tmp_path.iterdir()) == []


def test_tflite_export_backend_dispatch_matches_direct_call():
    model = _relu_model()
    a = tflite_export.convert_to_tflite(model, backend="onnx2tf")
    b = onnx2tf_export.convert_to_tflite_via_onnx2tf(model)
    assert a == b


def test_cnn_pipeline_matches_onnxruntime():
    model = _cnn_model()
    x = np.random.RandomState(1).randn(1, 3, 8, 8).astype(np.float32)
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    expected = sess.run(None, {"x": x})[0]
    tflite_model = onnxsim.export_tflite(model, backend="onnx2tf")
    actual = _run_tflite(tflite_model, x)
    np.testing.assert_allclose(expected, actual, rtol=1e-3, atol=1e-3)


def test_unsupported_op_raises_runtime_error():
    # onnx2tf covers ~200 ops (essentially every standard ONNX op), so there's no
    # stable real op name to pick that's guaranteed to stay unsupported across
    # versions -- use a made-up op_type instead, which onnx2tf's own op dispatch
    # rejects the same way (it looks up a same-named module under onnx2tf.ops).
    # This model fails onnx's own checker (unknown op), so build it without going
    # through _model()'s onnx.checker.check_model call.
    model = parser.parse_model(
        '<ir_version: 8, opset_import: ["" : 17]> '
        "unsup (float[2,3] x) => (float[2,3] y) "
        "{ y = TotallyNotARealOp (x) }"
    )
    with pytest.raises(RuntimeError, match="onnx2tf conversion failed"):
        onnx2tf_export.export_tflite_via_onnx2tf(model)
