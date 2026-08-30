#!/usr/bin/env python3
"""Convert a small ONNX model to Core ML and predict through the real runtime.

``tests/test_coreml_export.py`` (run on Linux in the ``coreml-integration``
workflow) only checks the *converted* model's declared shapes/dtypes and MIL-level
constant-folded values -- conversion itself needs no macOS-specific functionality.
What it can't check is whether Apple's actual Core ML runtime accepts the model and
computes the same thing, since that only exists on macOS. This script is the
``test-predict-macos`` job's entry point for that: convert with
``skip_model_load=False`` (so the model is compiled and loaded, not just
serialized), run ``.predict()`` on it, and compare against onnxruntime on the same
random input.

Exits non-zero (with the mismatch/exception printed) if conversion, compilation, or
prediction fails, or if Core ML's output disagrees with onnxruntime's beyond a
loose numeric tolerance (Core ML models routinely run at fp16 internally, so this
isn't a bit-exact comparison).
"""

import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper

import onnxsim


def _cnn_model() -> onnx.ModelProto:
    """Conv -> BatchNorm -> Relu -> AveragePool -> Flatten -> Gemm -> Softmax."""
    rng = np.random.RandomState(0)
    w = numpy_helper.from_array(rng.randn(4, 3, 3, 3).astype(np.float32), name="w")
    b = numpy_helper.from_array(np.zeros(4, np.float32), name="b")
    scale = numpy_helper.from_array(np.ones(4, np.float32), name="scale")
    bn_bias = numpy_helper.from_array(np.zeros(4, np.float32), name="bn_bias")
    mean = numpy_helper.from_array(np.zeros(4, np.float32), name="mean")
    var = numpy_helper.from_array(np.ones(4, np.float32), name="var")
    # AveragePool(k=2, s=2) on Conv's 8x8 output halves both spatial dims to 4x4,
    # so Flatten(axis=1) yields 4 (channels) * 4 * 4 = 64 features.
    gw = numpy_helper.from_array(rng.randn(4, 4 * 4 * 4).astype(np.float32), name="gw")
    gb = numpy_helper.from_array(np.zeros(4, np.float32), name="gb")

    x = helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 3, 8, 8])
    y = helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 4])
    nodes = [
        helper.make_node(
            "Conv",
            ["x", "w", "b"],
            ["conv_out"],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        ),
        helper.make_node(
            "BatchNormalization",
            ["conv_out", "scale", "bn_bias", "mean", "var"],
            ["bn_out"],
        ),
        helper.make_node("Relu", ["bn_out"], ["relu_out"]),
        helper.make_node(
            "AveragePool",
            ["relu_out"],
            ["pool_out"],
            kernel_shape=[2, 2],
            strides=[2, 2],
        ),
        helper.make_node("Flatten", ["pool_out"], ["flat_out"], axis=1),
        helper.make_node("Gemm", ["flat_out", "gw", "gb"], ["gemm_out"], transB=1),
        helper.make_node("Softmax", ["gemm_out"], ["y"], axis=-1),
    ]
    graph = helper.make_graph(
        nodes, "cnn", [x], [y], initializer=[w, b, scale, bn_bias, mean, var, gw, gb]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def main() -> int:
    model = _cnn_model()
    rng = np.random.RandomState(1)
    x = rng.randn(1, 3, 8, 8).astype(np.float32)

    print("Simplifying...", flush=True)
    simplified, ok = onnxsim.simplify(model)
    if not ok:
        print("onnxsim.simplify() reported failure", file=sys.stderr)
        return 1

    print("Computing expected output with onnxruntime...", flush=True)
    sess = ort.InferenceSession(
        simplified.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (expected,) = sess.run(None, {"x": x})

    print("Converting to Core ML and loading for prediction...", flush=True)
    mlmodel = onnxsim.export_coreml(simplified, skip_model_load=False)

    print("Predicting through the Core ML runtime...", flush=True)
    prediction = mlmodel.predict({"x": x})
    (out_name,) = prediction.keys()
    actual = np.asarray(prediction[out_name])

    diff = np.abs(actual.reshape(expected.shape) - expected).max()
    print(f"max abs diff vs onnxruntime: {diff:.2e}", flush=True)
    if not np.allclose(actual.reshape(expected.shape), expected, atol=1e-2, rtol=1e-2):
        print("MISMATCH between Core ML and onnxruntime output", file=sys.stderr)
        return 1

    print("OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
