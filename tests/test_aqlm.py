"""Tests for ``onnxsim.quantize_weight_only_aqlm`` (AQLM, see
``onnxsim/aqlm.py``) -- greedy residual k-means fits ``M`` codebooks
shared across every group in a layer, each group represented as the sum
of one lookup per codebook.

``quantize_weight_only_aqlm`` now delegates to ``onnxsim.apply_aqlm_cpp``
(see ``onnxsim/aqlm.py``'s own "Delegates to" note): the returned graph
folds the additive-codebook reconstruction directly into a replacement
float32 initializer (no ``Gather``/``Add`` chain, no separate
codebook/codes initializers to inspect), and the delegated C++ port
hardcodes ``group_dim=8``/``num_codebooks=2``/``codebook_size=256``/
``num_iterations=10`` -- a non-default value now raises ``ValueError``
rather than being honored. See ``tests/test_aqlm_cpp.py`` for the full
C++ port test suite this one now exercises indirectly.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim

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


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _matmul_model(K=64, N=16, weight=None, seed=0):
    if weight is None:
        rng = np.random.default_rng(seed)
        weight = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    return _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f32(weight, "W")],
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
    # apply_aqlm_cpp (which quantize_weight_only_aqlm now delegates to)
    # folds the additive-codebook reconstruction directly into a
    # replacement float32 initializer, rewiring the matched node's own
    # weight input -- see onnxsim/aqlm.py's own "Delegates to" note.
    node = next(n for n in model.graph.node if n.op_type in ("MatMul", "Gemm"))
    w_name = node.input[weight_input_index]
    w_init = next(t for t in model.graph.initializer if t.name == w_name)
    return onnx.numpy_helper.to_array(w_init)


def test_aqlm_output_stays_close_to_float_via_onnxruntime():
    model = _matmul_model(K=64, N=16, seed=0)
    q = onnxsim.quantize_weight_only_aqlm(model)
    onnx.checker.check_model(q)

    # No new graph nodes -- folded straight into a replacement initializer.
    assert [n.op_type for n in q.graph.node] == [n.op_type for n in model.graph.node]

    rng = np.random.default_rng(1)
    x = rng.standard_normal((8, 64)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.3


def test_aqlm_replaces_weight_with_same_shape_float():
    # num_groups = N * (K // group_dim) must exceed codebook_size (256) --
    # otherwise every group can claim its own exact centroid and the
    # reconstruction is trivially exact, the same degenerate-sizing trap
    # tests/test_aqlm_cpp.py's own docstring already documents.
    K, N = 8 * 40, 10  # num_groups = 400 > codebook_size (256)
    rng = np.random.default_rng(2)
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.3
    model = _matmul_model(K=K, N=N, weight=weight)
    q = onnxsim.quantize_weight_only_aqlm(model)
    new_w = _current_weight(q)
    assert new_w.shape == (K, N)
    assert new_w.dtype == np.float32
    assert not np.array_equal(new_w, weight)


def test_aqlm_non_default_params_raise():
    # apply_aqlm_cpp (which this function now delegates to) hardcodes
    # group_dim=8/num_codebooks=2/codebook_size=256/num_iterations=10 and
    # cannot honor other values -- including varying num_codebooks/
    # codebook_size, which the old pure-Python implementation's own
    # "more codebooks never increases error"/"codes stay in codebook
    # range" tests used to exercise; that property is no longer testable
    # through this entry point, only through the C++ port's own fixed
    # configuration (see tests/test_aqlm_cpp.py).
    model = _matmul_model(K=32, N=8, seed=8)
    with pytest.raises(ValueError):
        onnxsim.quantize_weight_only_aqlm(model, group_dim=4)
    with pytest.raises(ValueError):
        onnxsim.quantize_weight_only_aqlm(model, num_codebooks=3)
    with pytest.raises(ValueError):
        onnxsim.quantize_weight_only_aqlm(model, codebook_size=16)
    with pytest.raises(ValueError):
        onnxsim.quantize_weight_only_aqlm(model, num_iterations=5)


def test_aqlm_gemm_transb():
    rng = np.random.default_rng(7)
    K, N = 64, 12
    weight = rng.standard_normal((N, K)).astype(np.float32) * 0.5
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm<transB = 1>(X, W)
        }}
        """,
        initializer=[_f32(weight, "W")],
    )
    q = onnxsim.quantize_weight_only_aqlm(model)
    onnx.checker.check_model(q)

    x = rng.standard_normal((4, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert _rel_l2(float_y, q_y) < 0.3


def test_aqlm_skips_non_group_divisible_k():
    model = _matmul_model(K=20, N=4, seed=9)  # 20 is not a multiple of 8
    q = onnxsim.quantize_weight_only_aqlm(model, group_dim=8)
    assert q.SerializeToString() == model.SerializeToString()


def test_aqlm_skips_non_constant_weight():
    model = _model(
        """
        g (float[4,32] X, float[32,4] W) => (float[4,4] Y)
        {
          Y = MatMul(X, W)
        }
        """
    )
    q = onnxsim.quantize_weight_only_aqlm(model)
    assert q.SerializeToString() == model.SerializeToString()


def test_aqlm_noop_when_no_matmul_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.quantize_weight_only_aqlm(model)
    assert result.SerializeToString() == model.SerializeToString()
