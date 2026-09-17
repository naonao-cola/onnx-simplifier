"""Tests for ``onnxsim.apply_ptq4vit_quantization_cpp`` -- the C++-backed
port of ``onnxsim.apply_ptq4vit_quantization`` (PTQ4ViT's twin uniform
quantization, see ``onnxsim/ptq4vit_entry.h``). Like ``test_daq_cpp.py``,
this checks a LOOSE numeric tolerance for the actual split/scale values
rather than bit-for-bit equality: unlike every weight-quantizing port in
this codebase (a pure RTN computation with no accumulated reduction),
this port's own split search scores each of 97 candidate thresholds by a
whole-dataset mean-squared-reconstruction-error reduction -- numpy's own
``np.mean``/``**2`` uses pairwise (tree) summation for a large array,
while this port's own C++ kernel sums sequentially, so the two can round
differently in the last few bits and -- on a close enough candidate --
tip the argmin to a neighboring grid point, not just a last-ulp
difference in the chosen split's own value. Structural equivalence (same
matched tensors, same inserted node shape) is still checked exactly.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

from onnxsim.onnx_simplifier import apply_ptq4vit_quantization_cpp
from onnxsim.ptq4vit import apply_ptq4vit_quantization

ort = pytest.importorskip("onnxruntime")


def _model(body, initializer=(), opset=13, ir_version=8):
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


def _softmax_model(opset=13):
    # 128 columns so a single calibration batch (16 * 128 = 2048
    # elements) comfortably clears the search's own `size >= 2 * n_levels`
    # (512 at the default n_levels=256) floor -- see
    # onnxsim.ptq4vit._search_twin_split.
    return _model(
        """
        g (float[16,128] X) => (float[16,128] Y)
        {
          Probs = Softmax<axis=-1>(X)
          Y = Identity(Probs)
        }
        """,
        opset=opset,
    )


def _gelu_model():
    return _model(
        """
        g (float[256] X) => (float[256] Y)
        {
          G = Gelu(X)
          Y = Identity(G)
        }
        """,
        opset=20,
        ir_version=9,
    )


def _gelu_decomposed_model():
    return _model(
        """
        g (float[256] X) => (float[256] Y)
        {
          Sqrt2 = Constant<value = float[1] {1.4142135}>()
          Half = Constant<value = float[1] {0.5}>()
          One = Constant<value = float[1] {1.0}>()
          Scaled = Div(X, Sqrt2)
          Erfed = Erf(Scaled)
          Shifted = Add(Erfed, One)
          Weighted = Mul(X, Shifted)
          Gelu = Mul(Weighted, Half)
          Y = Identity(Gelu)
        }
        """
    )


def _softmax_calibration(seed=0, batches=8):
    rng = np.random.default_rng(seed)
    return [
        {"X": rng.standard_normal((16, 128)).astype(np.float32)} for _ in range(batches)
    ]


def _gelu_calibration(seed=0, batches=8, n=256):
    rng = np.random.default_rng(seed)
    return [
        {"X": rng.standard_normal(n).astype(np.float32) * 3.0} for _ in range(batches)
    ]


def _scalar_const(model, name):
    for t in model.graph.initializer:
        if t.name == name:
            return float(onnx.numpy_helper.to_array(t))
    raise KeyError(name)


def _structural_op_counts(model):
    return sorted(n.op_type for n in model.graph.node)


def test_ptq4vit_cpp_wraps_softmax_output_same_shape_as_python():
    model = _softmax_model()
    calib = _softmax_calibration()
    py = apply_ptq4vit_quantization(model, calibration_data=calib)
    cpp = apply_ptq4vit_quantization_cpp(model, calibration_data=calib)
    onnx.checker.check_model(cpp)

    assert _structural_op_counts(py) == _structural_op_counts(cpp)
    assert "Where" in _structural_op_counts(cpp)
    assert "Less" in _structural_op_counts(cpp)

    # Find each side's own "..._split" constant and compare with a loose
    # tolerance -- see this file's own module docstring.
    py_split = next(
        onnx.numpy_helper.to_array(t)
        for t in py.graph.initializer
        if t.name.endswith("_split")
    )
    cpp_split = next(
        onnx.numpy_helper.to_array(t)
        for t in cpp.graph.initializer
        if t.name.endswith("_split")
    )
    assert py_split == pytest.approx(cpp_split, abs=0.05)


def test_ptq4vit_cpp_wraps_standalone_gelu_output():
    model = _gelu_model()
    calib = _gelu_calibration()
    py = apply_ptq4vit_quantization(model, calibration_data=calib)
    cpp = apply_ptq4vit_quantization_cpp(model, calibration_data=calib)
    onnx.checker.check_model(cpp)
    assert _structural_op_counts(py) == _structural_op_counts(cpp)

    identity = next(n for n in cpp.graph.node if n.op_type == "Identity")
    where_node = next(n for n in cpp.graph.node if n.op_type == "Where")
    assert identity.input[0] == where_node.output[0]


def test_ptq4vit_cpp_wraps_decomposed_erf_gelu_output():
    model = _gelu_decomposed_model()
    calib = _gelu_calibration()
    py = apply_ptq4vit_quantization(model, calibration_data=calib)
    cpp = apply_ptq4vit_quantization_cpp(model, calibration_data=calib)
    onnx.checker.check_model(cpp)
    assert _structural_op_counts(py) == _structural_op_counts(cpp)
    assert any(n.op_type == "Erf" for n in cpp.graph.node)


def test_ptq4vit_cpp_reconstruction_close_to_python_via_onnxruntime():
    model = _softmax_model()
    calib = _softmax_calibration()
    cpp = apply_ptq4vit_quantization_cpp(model, calibration_data=calib)
    onnx.checker.check_model(cpp)

    def _run(m, feeds):
        sess = ort.InferenceSession(
            m.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        return sess.run(None, feeds)

    rng = np.random.default_rng(7)
    x = rng.standard_normal((16, 128)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(cpp, {"X": x})
    assert np.all(np.isfinite(q_y))
    # Values stay in [0, 1] (Softmax's own range) and close to the float
    # reference -- twin quantization at 256 levels per side is a fine
    # round trip.
    assert np.all(q_y >= -1e-3) and np.all(q_y <= 1 + 1e-3)
    assert np.abs(float_y - q_y).max() < 0.05


def test_ptq4vit_cpp_noop_without_softmax_or_gelu():
    model = _model(
        """
        g (float[4] X) => (float[4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    cpp = apply_ptq4vit_quantization_cpp(model, calibration_data=_softmax_calibration())
    assert cpp.SerializeToString() == model.SerializeToString()


def test_ptq4vit_cpp_declines_below_opset11():
    model = _model(
        """
        g (float[16,128] X) => (float[16,128] Y)
        {
          Y = Softmax<axis=-1>(X)
        }
        """,
        opset=10,
        ir_version=6,
    )
    cpp = apply_ptq4vit_quantization_cpp(model, calibration_data=_softmax_calibration())
    assert cpp.SerializeToString() == model.SerializeToString()


def test_ptq4vit_cpp_graph_output_target_left_unwrapped():
    # Softmax's own output IS the graph output here -- rewiring it would
    # need renaming a ValueInfoProto, which this pass (like the Python
    # reference) deliberately never does.
    model = _model(
        """
        g (float[16,128] X) => (float[16,128] Probs)
        {
          Probs = Softmax<axis=-1>(X)
        }
        """,
        opset=13,
    )
    cpp = apply_ptq4vit_quantization_cpp(model, calibration_data=_softmax_calibration())
    assert cpp.SerializeToString() == model.SerializeToString()


def test_ptq4vit_cpp_missing_calibration_input_raises():
    model = _softmax_model()
    with pytest.raises((RuntimeError, ValueError)):
        apply_ptq4vit_quantization_cpp(
            model, calibration_data=[{"Wrong": np.zeros((16, 128), dtype=np.float32)}]
        )
