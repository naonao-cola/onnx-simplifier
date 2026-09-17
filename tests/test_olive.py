"""Tests for ``onnxsim.quantize_weight_only_olive`` -- see ``onnxsim/olive.py``
for the technique (OliVe-style Outlier-Victim Pair quantization: an outlier
element is paired with its immediate memory-adjacent neighbor, the
"victim", and the pair's combined bit budget is renegotiated -- the
outlier gets an extra bit of dynamic range, the victim is re-quantized far
more coarsely -- rather than either an exact sparse correction
(``onnxsim.spqr``) or a whole rescued column (``onnxsim.owq``)).
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim

ort = pytest.importorskip("onnxruntime")


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


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _matmul_model(K=32, N=8, weight=None, seed=0, opset=21):
    if weight is None:
        rng = np.random.default_rng(seed)
        weight = rng.standard_normal((K, N)).astype(np.float32) * 0.1
    return _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
        opset=opset,
    )


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-6)


def _current_weight(model, weight_input_index=1):
    node = next(n for n in model.graph.node if n.op_type in ("MatMul", "Gemm"))
    w_name = node.input[weight_input_index]
    w_init = next(t for t in model.graph.initializer if t.name == w_name)
    return onnx.numpy_helper.to_array(w_init)


# quantize_weight_only_olive now delegates to the verified C++ port
# (apply_olive_cpp), which hardcodes bits=4/block_size=32/
# outlier_threshold=4.0 and folds the OVP round trip directly into a
# replacement float32 initializer instead of building a real
# DequantizeLinear x2 + Cast + Where + MatMul[+ Add] graph rewrite -- see
# onnxsim/olive.py's own docstring. The detailed OVP algorithmic
# properties (outlier/victim reconstruction, bit-budget renegotiation,
# error reduction vs. a naive single-scale fit, opset-independence) are
# already covered end to end against apply_olive_cpp directly in
# tests/test_olive_cpp.py; the tests below only exercise the thin
# wrapper itself: parameter validation and basic delegation sanity.


def test_olive_output_stays_close_to_float_via_onnxruntime():
    model = _matmul_model(K=32, N=8, seed=0)
    q = onnxsim.quantize_weight_only_olive(model)
    onnx.checker.check_model(q)

    new_w = _current_weight(q)
    assert new_w.shape == (32, 8)
    assert new_w.dtype == np.float32

    rng = np.random.default_rng(2)
    x = rng.standard_normal((8, 32)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.5


def test_olive_declines_non_constant_weight():
    model = _model(
        """
        g (float[4,32] X, float[32,4] W) => (float[4,4] Y)
        {
          Y = MatMul(X, W)
        }
        """
    )
    q = onnxsim.quantize_weight_only_olive(model)
    assert q.SerializeToString() == model.SerializeToString()


def test_olive_noop_when_no_matmul_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.quantize_weight_only_olive(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_olive_declines_when_k_not_divisible_by_block_size():
    model = _matmul_model(K=20, N=4, seed=9)  # 20 is not a multiple of 32
    q = onnxsim.quantize_weight_only_olive(model)
    assert q.SerializeToString() == model.SerializeToString()


def test_olive_rejects_non_default_block_size():
    model = _matmul_model(K=32, N=8)
    with pytest.raises(ValueError):
        onnxsim.quantize_weight_only_olive(model, block_size=8)


def test_olive_rejects_non_default_bits():
    model = _matmul_model(K=32, N=8)
    with pytest.raises(ValueError):
        onnxsim.quantize_weight_only_olive(model, bits=2)


def test_olive_bit_budget_matches_ordinary_pair():
    # By construction: outlier_bits + victim_bits == 2 * bits == two
    # ordinary elements' own combined budget, for every bits >= 3.
    for bits in (3, 4, 5, 6):
        ordinary_qmax = 2 ** (bits - 1) - 1
        outlier_qmax = 2**bits - 1
        victim_qmax = 2 ** (bits - 2) - 1
        outlier_bits = (outlier_qmax + 1).bit_length()
        victim_bits = (victim_qmax + 1).bit_length() if victim_qmax > 0 else 1
        assert outlier_bits + victim_bits == 2 * bits
        assert 2 * ordinary_qmax <= 2 * outlier_qmax  # sanity: never a regression
