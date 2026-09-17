"""Tests for ``onnxsim.apply_mixed_precision_quantization_cpp`` -- the
C++-backed port of ``onnxsim.apply_mixed_precision_quantization`` (see
``onnxsim/mixed_precision_entry.h``). Like ``test_llm_int8_cpp.py``, this
runs the model over real calibration data through a real
``onnxruntime``-backed executor -- never a fake/mock executor.

Both sides run the same closed-form block-wise INT4/INT8 RTN quantizer
(no RNG, no gradient descent), so the actual quantized VALUES for any one
layer are always bit-exact between the two implementations regardless of
summation order. The only place floating-point summation order can
matter at all is the per-layer *sensitivity score* (an accumulated
Hessian-diagonal or full-Hessian reduction) that decides WHICH layers get
promoted to INT8 -- so test fixtures below plant a clearly, deliberately
more-sensitive layer (a large-magnitude outlier row, the same technique
``tests/test_mixed_precision.py``'s own ``_two_layer_model`` already
uses) so the ranking is never close enough to flip on a last-bit
difference, and the exact-parity assertion below is a real regression
check rather than a source of flakiness.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

from onnxsim.mixed_precision import apply_mixed_precision_quantization
from onnxsim.onnx_simplifier import apply_mixed_precision_quantization_cpp

ort = pytest.importorskip("onnxruntime")


def _f32(array, name):
    return onnx.numpy_helper.from_array(np.asarray(array, dtype=np.float32), name)


def _model(body, initializer=(), opset=21, ir_version=10):
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


def _two_layer_model(K=32, H=16, N=8, seed=0, opset=21):
    rng = np.random.default_rng(seed)
    w1 = (rng.standard_normal((K, H)) * 0.5).astype(np.float32)
    # w2's first row gets a large-magnitude outlier, making it far more
    # sensitive to INT4 quantization than w1 -- see this file's own
    # module docstring for why this matters for exact-parity testing.
    w2 = (rng.standard_normal((H, N)) * 0.05).astype(np.float32)
    w2[0, :] = 20.0
    return _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          H1 = MatMul(X, W1)
          Y = MatMul(H1, W2)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2")],
        opset=opset,
    )


def _calibration(K=32, num_samples=16, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((num_samples, K)).astype(np.float32)
    return [{"X": x}]


def _assert_exact_parity(model, calibration_data, **kwargs):
    py = apply_mixed_precision_quantization(model, calibration_data, **kwargs)
    cpp = apply_mixed_precision_quantization_cpp(model, calibration_data, **kwargs)
    py_nodes = sorted(
        (n.op_type, tuple(n.input), tuple(n.output), n.name) for n in py.graph.node
    )
    cpp_nodes = sorted(
        (n.op_type, tuple(n.input), tuple(n.output), n.name) for n in cpp.graph.node
    )
    assert py_nodes == cpp_nodes
    py_inits = sorted(py.graph.initializer, key=lambda t: t.name)
    cpp_inits = sorted(cpp.graph.initializer, key=lambda t: t.name)
    assert [t.name for t in py_inits] == [t.name for t in cpp_inits]
    for a, b in zip(py_inits, cpp_inits):
        assert a.data_type == b.data_type, a.name
        ta, tb = onnx.numpy_helper.to_array(a), onnx.numpy_helper.to_array(b)
        assert ta.shape == tb.shape, a.name
        assert np.array_equal(ta, tb), a.name
    return py, cpp


def test_mixed_precision_cpp_matches_python_exactly_hessian_diag():
    py, cpp = _assert_exact_parity(
        _two_layer_model(),
        _calibration(),
        block_size=8,
        high_bits_fraction=0.5,
        sensitivity_metric="hessian_diag",
    )
    # w2 (the outlier layer) should win the INT8 tier on both sides.
    assert any("W2_mixedprec_int8" in t.name for t in cpp.graph.initializer)
    assert any("W1_mixedprec_int4" in t.name for t in cpp.graph.initializer)


def test_mixed_precision_cpp_matches_python_exactly_full_hessian():
    _assert_exact_parity(
        _two_layer_model(seed=2),
        _calibration(seed=5),
        block_size=8,
        high_bits_fraction=0.5,
        sensitivity_metric="full_hessian",
    )


def test_mixed_precision_cpp_zero_fraction_is_all_int4():
    _, cpp = _assert_exact_parity(
        _two_layer_model(), _calibration(), block_size=8, high_bits_fraction=0.0
    )
    assert not any("int8" in t.name for t in cpp.graph.initializer)
    assert any("W1_mixedprec_int4" in t.name for t in cpp.graph.initializer)
    assert any("W2_mixedprec_int4" in t.name for t in cpp.graph.initializer)


def test_mixed_precision_cpp_one_fraction_is_all_int8():
    _, cpp = _assert_exact_parity(
        _two_layer_model(), _calibration(), block_size=8, high_bits_fraction=1.0
    )
    assert not any("int4" in t.name for t in cpp.graph.initializer)
    assert any("W1_mixedprec_int8" in t.name for t in cpp.graph.initializer)
    assert any("W2_mixedprec_int8" in t.name for t in cpp.graph.initializer)


def test_mixed_precision_cpp_output_close_to_float_via_onnxruntime():
    model = _two_layer_model()
    cpp = apply_mixed_precision_quantization_cpp(
        model, _calibration(), block_size=8, high_bits_fraction=0.5
    )
    onnx.checker.check_model(cpp)

    def _run(m, feeds):
        sess = ort.InferenceSession(
            m.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        return sess.run(None, feeds)

    rng = np.random.default_rng(9)
    x = rng.standard_normal((8, 32)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(cpp, {"X": x})
    assert np.all(np.isfinite(q_y))
    rel_l2 = np.linalg.norm(float_y - q_y) / max(np.linalg.norm(float_y), 1e-6)
    assert rel_l2 < 0.3


def test_mixed_precision_cpp_invalid_sensitivity_metric_raises():
    with pytest.raises(ValueError):
        apply_mixed_precision_quantization_cpp(
            _two_layer_model(), _calibration(), sensitivity_metric="bogus"
        )


def test_mixed_precision_cpp_below_min_opset_returns_unchanged():
    model = _two_layer_model(opset=20)
    cpp = apply_mixed_precision_quantization_cpp(model, _calibration())
    assert [n.op_type for n in cpp.graph.node] == [n.op_type for n in model.graph.node]
    assert len(cpp.graph.initializer) == len(model.graph.initializer)


def test_mixed_precision_cpp_no_candidates_returns_copy():
    model = _model(
        """
        g (float[batch,4] X) => (float[batch,4] Y)
        {
          Y = Relu(X)
        }
        """,
        opset=21,
    )
    cpp = apply_mixed_precision_quantization_cpp(
        model, [{"X": np.zeros((2, 4), dtype=np.float32)}]
    )
    assert [n.op_type for n in cpp.graph.node] == ["Relu"]


def test_mixed_precision_cpp_missing_calibration_input_raises():
    model = _two_layer_model()
    with pytest.raises((RuntimeError, ValueError)):
        apply_mixed_precision_quantization_cpp(
            model, [{"Wrong": np.zeros((1, 32), dtype=np.float32)}]
        )
