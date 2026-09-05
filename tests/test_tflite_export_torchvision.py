"""Real-world model coverage for TFLite export: small torchvision classifiers.

torchvision is already a test-time dependency elsewhere in this suite (e.g.
tests/test_python_api.py's torchvision detection-model tests), so this file adds
it no new cost. Feeding an actual deployed CNN through
``onnxsim.simplify() -> onnxsim.export_tflite()`` exercises real op combinations a
synthetic per-op test can't easily reproduce: residual ``Add``, depthwise ``Conv``
(``group == in_channels``), ``Clip``-as-ReLU6, ``GlobalAveragePool``, ``Gemm``, and
the chains of now-redundant ``Identity``/``Constant`` nodes a real
``torch.onnx.export`` tends to emit that onnxsim's own simplification then cleans up.

TensorFlow is onnxsim's own optional dependency for TFLite export (see
tflite_export.py's module docstring), so the whole module is skipped when it isn't
installed, same as tests/test_tflite_export.py.
"""

import numpy as np
import onnx
import pytest
import torch
import torchvision

pytest.importorskip("tensorflow", reason="tensorflow is not installed")

import onnxruntime as ort  # noqa: E402  (imported after the tensorflow availability check)
import tensorflow as tf  # noqa: E402

from onnxsim import tflite_export  # noqa: E402
from onnxsim.test_utils import export_simplify_and_check_by_python_api  # noqa: E402


def _run_tflite(tflite_model: bytes, x: np.ndarray) -> np.ndarray:
    interp = tf.lite.Interpreter(model_content=tflite_model)
    interp.allocate_tensors()
    (inp,) = interp.get_input_details()
    (out,) = interp.get_output_details()
    # The builtin backend keeps ONNX's NCHW layout; the onnx2tf backend converts
    # to channel-last by default -- transpose to match whichever this model is.
    x_in = np.transpose(x, (0, 2, 3, 1)) if list(inp["shape"]) != list(x.shape) else x
    interp.set_tensor(inp["index"], x_in)
    interp.invoke()
    return interp.get_tensor(out["index"])


def _assert_matches_onnxruntime(
    model: onnx.ModelProto, x: np.ndarray, **export_kwargs
) -> np.ndarray:
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (input_name,) = [i.name for i in sess.get_inputs()]
    expected = sess.run(None, {input_name: x})[0]
    tflite_model = tflite_export.export_tflite(model, **export_kwargs)
    actual = _run_tflite(tflite_model, x)
    np.testing.assert_allclose(expected, actual, rtol=1e-3, atol=1e-3)
    # A close-enough numeric match should also agree on the predicted class, the
    # thing that actually matters for a classifier deployed to a device.
    assert expected.argmax() == actual.argmax()
    return actual


def _random_image_batch() -> np.ndarray:
    return np.random.default_rng(0).standard_normal((1, 3, 224, 224)).astype(np.float32)


def test_resnet18_matches_onnxruntime():
    model = torchvision.models.resnet18(weights=None)
    sim_model = export_simplify_and_check_by_python_api(
        model, torch.randn(1, 3, 224, 224)
    )
    _assert_matches_onnxruntime(sim_model, _random_image_batch())


def test_mobilenet_v2_matches_onnxruntime():
    # MobileNetV2's inverted-residual blocks exercise depthwise Conv, Clip-as-
    # ReLU6, and residual Add together -- none of which resnet18's plain
    # Conv/BN/ReLU stack alone would cover.
    model = torchvision.models.mobilenet_v2(weights=None)
    sim_model = export_simplify_and_check_by_python_api(
        model, torch.randn(1, 3, 224, 224)
    )
    _assert_matches_onnxruntime(sim_model, _random_image_batch())


def test_resnet18_matches_onnxruntime_onnx2tf_backend():
    pytest.importorskip("onnx2tf", reason="onnx2tf is not installed")
    model = torchvision.models.resnet18(weights=None)
    sim_model = export_simplify_and_check_by_python_api(
        model, torch.randn(1, 3, 224, 224)
    )
    _assert_matches_onnxruntime(sim_model, _random_image_batch(), backend="onnx2tf")


def test_mobilenet_v2_matches_onnxruntime_onnx2tf_backend():
    # Regression test for a real onnx2tf bug this project's own recommendation to
    # "simplify first" happens to work around: onnx2tf 1.29.24's Clip op handler
    # raises KeyError('tf_node') on the *raw* torch.onnx.export of MobileNetV2 (35
    # Clip-as-ReLU6 nodes, each preceded by a now-redundant Identity in the raw
    # graph) -- confirmed by feeding onnx2tf_export.convert_to_tflite_via_onnx2tf
    # the unsimplified export directly. onnxsim.simplify() removes those Identity
    # nodes (209 nodes -> 102, dropping all 39 Identitys) as part of ordinary
    # constant folding/cleanup, and onnx2tf converts the resulting graph without
    # error -- this test runs the pipeline everyone actually uses
    # (export -> simplify -> export_tflite) and checks it stays that way.
    pytest.importorskip("onnx2tf", reason="onnx2tf is not installed")
    model = torchvision.models.mobilenet_v2(weights=None)
    sim_model = export_simplify_and_check_by_python_api(
        model, torch.randn(1, 3, 224, 224)
    )
    _assert_matches_onnxruntime(sim_model, _random_image_batch(), backend="onnx2tf")
