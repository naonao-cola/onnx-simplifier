"""Tests for ``onnxsim.quantize_weight_only_int4_hqq`` (HQQ, see
``onnxsim/hqq.py``) -- calibration-free, asymmetric block-wise INT4
quantization that fits each block's zero-point via IRLS to minimize a
robust (Lp, p<2) reconstruction loss instead of the ordinary least-squares
fit a naive min/max range implicitly targets.

``quantize_weight_only_int4_hqq`` now delegates to
``onnxsim.apply_hqq_cpp`` (see ``onnxsim/hqq.py``'s own "Delegates to"
note): the returned graph folds the IRLS-refined round trip directly into
a replacement float32 initializer rather than building a real
``DequantizeLinear(Wq, Ws, Wz, ...)`` node with packed UINT4 codes, so
tests here read that replacement initializer directly instead of decoding
a ``DequantizeLinear`` node -- see ``tests/test_hqq_cpp.py`` for the full
C++ port test suite this one now exercises indirectly.
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
        [_f32(weight, "W")],
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
    # apply_hqq_cpp (which quantize_weight_only_int4_hqq now delegates to)
    # folds the IRLS-refined round trip directly into a replacement
    # float32 initializer, rewiring the matched node's own weight input --
    # see onnxsim/hqq.py's own "Delegates to" note.
    node = next(n for n in model.graph.node if n.op_type in ("MatMul", "Gemm"))
    w_name = node.input[weight_input_index]
    w_init = next(t for t in model.graph.initializer if t.name == w_name)
    return onnx.numpy_helper.to_array(w_init).astype(np.float64)


def test_hqq_quantizes_matmul_to_replacement_float_weight():
    model = _matmul_model(K=64, N=16, seed=0)
    hqq_model = onnxsim.quantize_weight_only_int4_hqq(model)
    onnx.checker.check_model(hqq_model)

    # No new graph nodes -- weight-only, folded straight into a new
    # initializer (see onnxsim/hqq.py's own "Delegates to" note).
    assert [n.op_type for n in hqq_model.graph.node] == [
        n.op_type for n in model.graph.node
    ]
    new_w = _current_weight(hqq_model)
    orig_w = onnx.numpy_helper.to_array(model.graph.initializer[0]).astype(np.float64)
    assert new_w.shape == orig_w.shape
    assert not np.array_equal(new_w, orig_w)


def test_hqq_output_stays_close_to_float_via_onnxruntime():
    model = _matmul_model(K=64, N=16, seed=1)
    hqq_model = onnxsim.quantize_weight_only_int4_hqq(model)
    onnx.checker.check_model(hqq_model)

    rng = np.random.default_rng(2)
    x = rng.standard_normal((8, 64)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (hqq_y,) = _run(hqq_model, {"X": x})
    assert np.all(np.isfinite(hqq_y))
    assert _rel_l2(float_y, hqq_y) < 0.25


def test_hqq_beats_naive_minmax_on_outlier_heavy_weights():
    # HQQ's own motivating scenario: a block whose naive min/max range is
    # dominated by a couple of outlier elements, forcing a scale so wide
    # that the bulk of "normal" elements lose most of their precision. A
    # robust (Lp<2) fit should recover a tighter, more accurate zero-point
    # for the bulk at the cost of clipping the outliers -- net lower
    # reconstruction error on the block as a whole.
    rng = np.random.default_rng(3)
    K, N = 32, 8
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.1
    # Inject a couple of large outliers into one block (all of K=32 here,
    # i.e. one block at the default block_size=32).
    weight[0, :] += 5.0
    weight[1, :] += 5.0

    model = _matmul_model(K=K, N=N, weight=weight)
    hqq_model = onnxsim.quantize_weight_only_int4_hqq(model)
    w_hqq = _current_weight(hqq_model)  # [K, N]

    # Naive min/max affine quantization (p=2, i.e. plain least squares --
    # what a naive min/max range effectively targets) for comparison:
    # asymmetric codes derived directly from min/max with no IRLS refinement.
    w = weight.astype(np.float64)
    mn, mx = w.min(axis=0, keepdims=True), w.max(axis=0, keepdims=True)
    scale_naive = np.maximum((mx - mn) / 15.0, 1e-12)
    zero_naive = np.clip(np.round(-mn / scale_naive), 0, 15)
    codes_naive = np.clip(np.round(w / scale_naive + zero_naive), 0, 15)
    w_naive = (codes_naive - zero_naive) * scale_naive

    # Restrict the comparison to the *non-outlier* bulk (rows 2:), where
    # HQQ's robust fit should show its benefit most clearly.
    err_hqq = np.linalg.norm(w_hqq[2:] - w[2:])
    err_naive = np.linalg.norm(w_naive[2:] - w[2:])
    assert err_hqq < err_naive


def test_hqq_gemm_transb():
    rng = np.random.default_rng(4)
    K, N = 96, 12
    weight = rng.standard_normal((N, K)).astype(np.float32) * 0.5
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm<transB = 1>(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )
    hqq_model = onnxsim.quantize_weight_only_int4_hqq(model)
    onnx.checker.check_model(hqq_model)

    x = rng.standard_normal((4, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (hqq_y,) = _run(hqq_model, {"X": x})
    assert _rel_l2(float_y, hqq_y) < 0.25


def test_hqq_reconstructed_block_has_at_most_16_distinct_values():
    # HQQ's own codes are unsigned 4-bit ([0, 15]), so each (output-channel,
    # block) group can reconstruct at most 16 distinct values -- still
    # true of the folded float32 replacement weight, even though the codes
    # themselves are no longer visible in the graph.
    rng = np.random.default_rng(5)
    weight = rng.standard_normal((32, 8)).astype(np.float32) * 3
    model = _matmul_model(K=32, N=8, weight=weight)
    hqq_model = onnxsim.quantize_weight_only_int4_hqq(model)

    w_hqq = _current_weight(hqq_model)  # [K, N], one block per column here
    for col in range(w_hqq.shape[1]):
        assert len(np.unique(np.round(w_hqq[:, col], 6))) <= 16


def test_hqq_skips_non_block_divisible_k():
    # K=48 is not a multiple of the default block_size=32.
    model = _matmul_model(K=48, N=8, seed=6)
    hqq_model = onnxsim.quantize_weight_only_int4_hqq(model)
    assert hqq_model.SerializeToString() == model.SerializeToString()


def test_hqq_skips_non_constant_weight():
    model = _model(
        """
        g (float[4,64] X, float[64,4] W) => (float[4,4] Y)
        {
          Y = MatMul(X, W)
        }
        """
    )
    hqq_model = onnxsim.quantize_weight_only_int4_hqq(model)
    op_types = [n.op_type for n in hqq_model.graph.node]
    assert op_types.count("MatMul") == 1
    assert "DequantizeLinear" not in op_types
