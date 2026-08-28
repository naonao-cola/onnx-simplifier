"""Tests for ``onnxsim.quantize_qoperator_softmax`` (the
``qoperator_quantize_softmax`` C++ pass) -- the reduction-axis analogue of
``test_qoperator_quantize_activation.py``'s ``QLinearSigmoid``/
``QLinearLeakyRelu`` coverage, using ONNX Runtime's "com.microsoft" contrib
op ``QLinearSoftmax`` instead.
"""

import collections

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim

# A bare ``import onnxruntime`` would fail collection (not skip the test) on
# platforms onnxruntime doesn't ship wheels for (e.g. s390x).
ort = pytest.importorskip("onnxruntime")


def _model(body, opset=13):
    return parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": {opset}]
        >
        {body}
        """
    )


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _op_counts(model):
    return collections.Counter(n.op_type for n in model.graph.node)


def _assert_close(float_outputs, quant_outputs, rel_l2_tol=0.1):
    for f, q in zip(float_outputs, quant_outputs):
        f = np.asarray(f, dtype=np.float64).ravel()
        q = np.asarray(q, dtype=np.float64).ravel()
        assert np.all(np.isfinite(q))
        rel_l2 = np.linalg.norm(f - q) / max(np.linalg.norm(f), 1e-6)
        assert rel_l2 < rel_l2_tol, f"relative L2 error too large: {rel_l2:.4f}"


def test_quantize_softmax_default_axis():
    rng = np.random.default_rng(0)
    model = _model(
        """
        g (float[4,8] X) => (float[4,8] Y)
        {
          Y = Softmax(X)
        }
        """
    )

    quant = onnxsim.quantize_qoperator_softmax(
        model, num_calibration_samples=16, seed=0
    )
    onnx.checker.check_model(quant)
    ops = _op_counts(quant)
    assert ops["Softmax"] == 0
    assert ops["QLinearSoftmax"] == 1
    assert ops["QuantizeLinear"] == 1
    assert ops["DequantizeLinear"] == 1
    domains = {o.domain for o in quant.opset_import}
    assert "com.microsoft" in domains

    qlop = next(n for n in quant.graph.node if n.op_type == "QLinearSoftmax")
    axis = next(a.i for a in qlop.attribute if a.name == "axis")
    opset_attr = next(a.i for a in qlop.attribute if a.name == "opset")
    assert axis == -1
    assert opset_attr == 13

    x = rng.standard_normal((4, 8)).astype(np.float32)
    _assert_close(_run(model, {"X": x}), _run(quant, {"X": x}))


def test_quantize_softmax_explicit_axis():
    rng = np.random.default_rng(1)
    model = _model(
        """
        g (float[2,4,8] X) => (float[2,4,8] Y)
        {
          Y = Softmax<axis = 1>(X)
        }
        """
    )

    quant = onnxsim.quantize_qoperator_softmax(
        model, num_calibration_samples=16, seed=1
    )
    onnx.checker.check_model(quant)
    qlop = next(n for n in quant.graph.node if n.op_type == "QLinearSoftmax")
    axis = next(a.i for a in qlop.attribute if a.name == "axis")
    assert axis == 1

    x = rng.standard_normal((2, 4, 8)).astype(np.float32)
    _assert_close(_run(model, {"X": x}), _run(quant, {"X": x}))


def test_quantize_softmax_pre_opset13_semantics():
    # Pre-opset-13 Softmax flattens the tensor into a 2-D matrix at `axis`
    # and reduces the trailing dimension -- entirely different semantics
    # from opset-13+'s in-place per-axis reduction. The rewrite must thread
    # the model's own opset through as QLinearSoftmax's `opset` attribute so
    # ONNX Runtime's kernel replicates the *correct* one, not silently
    # assume the newer semantics.
    rng = np.random.default_rng(2)
    model = _model(
        """
        g (float[2,3,4] X) => (float[2,3,4] Y)
        {
          Y = Softmax<axis = 1>(X)
        }
        """,
        opset=11,
    )

    quant = onnxsim.quantize_qoperator_softmax(
        model, num_calibration_samples=16, seed=2
    )
    onnx.checker.check_model(quant)
    qlop = next(n for n in quant.graph.node if n.op_type == "QLinearSoftmax")
    opset_attr = next(a.i for a in qlop.attribute if a.name == "opset")
    assert opset_attr == 11

    x = rng.standard_normal((2, 3, 4)).astype(np.float32)
    _assert_close(_run(model, {"X": x}), _run(quant, {"X": x}))


def test_quantize_multiple_independent_nodes():
    rng = np.random.default_rng(3)
    model = _model(
        """
        g (float[4,8] A, float[4,8] B) => (float[8,8] C)
        {
          T1 = Softmax<axis = -1>(A)
          T2 = Sigmoid(B)
          C = Concat<axis = 0>(T1, T2)
        }
        """
    )

    quant = onnxsim.quantize_qoperator_softmax(
        model, num_calibration_samples=16, seed=3
    )
    onnx.checker.check_model(quant)
    ops = _op_counts(quant)
    assert ops["QLinearSoftmax"] == 1
    assert ops["Sigmoid"] == 1  # untouched by this pass

    a = rng.standard_normal((4, 8)).astype(np.float32)
    b = rng.standard_normal((4, 8)).astype(np.float32)
    feeds = {"A": a, "B": b}
    _assert_close(_run(model, feeds), _run(quant, feeds))


def test_quantize_skips_non_float():
    model = _model(
        """
        g (float16[4] X) => (float16[4] Y)
        {
          Y = Softmax(X)
        }
        """
    )
    quant = onnxsim.quantize_qoperator_softmax(model)
    assert _op_counts(quant)["Softmax"] == 1
    assert _op_counts(quant)["QLinearSoftmax"] == 0


def test_list_qoperator_softmax_quantizable_tensors():
    model = _model(
        """
        g (float[4,8] X) => (float[4,8] Y)
        {
          Y = Softmax(X)
        }
        """
    )
    import onnxsim.onnxsim_cpp2py_export as C

    names = C.list_qoperator_softmax_quantizable_tensors(model.SerializeToString())
    assert set(names) == {"X", "Y"}


def test_list_qoperator_softmax_quantizable_tensors_no_opset_import():
    # A model with no resolvable default-domain opset import has nothing
    # quantizable -- there is no safe "opset" attribute value to guess.
    model = parser.parse_model(
        """
        <ir_version: 10, opset_import: []>
        g (float[4,8] X) => (float[4,8] Y)
        {
          Y = Softmax(X)
        }
        """
    )
    import onnxsim.onnxsim_cpp2py_export as C

    names = C.list_qoperator_softmax_quantizable_tensors(model.SerializeToString())
    assert names == []
