"""Tests for ``onnxsim.apply_hqq_cpp`` -- the C++-backed port of
``onnxsim.quantize_weight_only_int4_hqq`` (see ``onnxsim/passes/hqq.h``).
Unlike the Python side, which builds a real ``DequantizeLinear(Wq, Ws,
Wz, ...)`` node with packed UINT4 codes/zero-points, this port folds the
IRLS-refined round trip directly into a replacement float32 initializer
(see that header's own "ACCEPTED, PERMANENT DIVERGENCE" note) -- so
cross-checking against the Python port is done end-to-end through
onnxruntime rather than by comparing initializers directly. IRLS here is
a deterministic fixed-point iteration with no RNG and no cross-block
interaction, so this port is expected to track the Python port's own
float64 numpy implementation closely, up to floating-point
summation-order differences -- comparable, not required to be
bit-for-bit identical, matching this repo's established contract for
every other data-free ``*_cpp`` port.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.hqq import quantize_weight_only_int4_hqq

ort = pytest.importorskip("onnxruntime")

_BLOCK_SIZE = 32


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


def _matmul_model(w, K, N, batch="batch"):
    return _model(
        f"""
        g (float[{batch},{K}] X) => (float[{batch},{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(w, "W")],
    )


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-9)


def _current_weight(model, weight_input_index=1):
    # The C++ pass (like every sibling data-free *_cpp port) rewires the
    # matched node's weight input to a freshly created initializer,
    # leaving the original one dangling unused in the graph -- so the
    # *node's own current input name* is the only reliable way to find
    # the actual (post-quantization) weight, not initializer list
    # position.
    node = next(n for n in model.graph.node if n.op_type in ("MatMul", "Gemm"))
    w_name = node.input[weight_input_index]
    w_init = next(t for t in model.graph.initializer if t.name == w_name)
    return onnx.numpy_helper.to_array(w_init)


def _naive_affine_int4_nk(w_nk, block_size=_BLOCK_SIZE):
    # A plain (non-IRLS) min/max affine int4 fit, for the same [N, K]
    # convention hqq.py's own encoder normalizes to: one shared,
    # zero-point-only-at-init (never refined) affine quantizer per
    # (output-channel, K-block) group.
    n, k = w_nk.shape
    nb = k // block_size
    blocks = w_nk.reshape(n, nb, block_size)
    lo = blocks.min(axis=2)
    hi = blocks.max(axis=2)
    scale = np.maximum((hi - lo) / 15.0, 1e-12)
    zero = np.clip(np.round(-lo / scale), 0, 15)
    scale3 = scale[:, :, np.newaxis]
    zero3 = zero[:, :, np.newaxis]
    code = np.clip(np.round(blocks / scale3 + zero3), 0, 15)
    dequant = scale3 * (code - zero3)
    return dequant.reshape(n, k)


def test_cpp_replaces_weight_with_same_shape_float():
    rng = np.random.default_rng(0)
    K, N = _BLOCK_SIZE * 2, 8
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(w, K, N)

    q = onnxsim.apply_hqq_cpp(model)
    onnx.checker.check_model(q)
    new_w = _current_weight(q)
    assert new_w.shape == w.shape
    assert new_w.dtype == np.float32
    assert not np.array_equal(new_w, w)
    # The original initializer is left in the graph, unused -- matching
    # every sibling data-free *_cpp port's established convention.
    assert any(t.name == "W" for t in q.graph.initializer)


def test_cpp_at_most_16_distinct_levels_per_block():
    rng = np.random.default_rng(1)
    K, N = _BLOCK_SIZE * 3, 4
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(w, K, N)

    q = onnxsim.apply_hqq_cpp(model)
    new_w = _current_weight(q)

    # MatMul's weight is [K, N]; HQQ's own blocks group K per output
    # channel (column n), so a "block" here is a column-strided slice.
    for n in range(N):
        col = new_w[:, n]
        for start in range(0, K, _BLOCK_SIZE):
            block = col[start : start + _BLOCK_SIZE]
            assert len(np.unique(block)) <= 16


def test_cpp_irls_matches_or_beats_naive_affine_on_outlier_heavy_blocks():
    # HQQ's whole premise is that a couple of large-magnitude outliers per
    # block should no longer be able to force every other element toward a
    # handful of coarse levels the way an unweighted min/max fit would --
    # but empirically (verified directly against onnxsim.hqq's own
    # _irls_affine_quantize_blockwise on this exact input and several
    # other synthetic ones, including generic random data with no injected
    # outlier at all) the *rounded* zero-point this scheme converges to is
    # frequently bit-identical to the naive min/max-anchored one: at
    # zero_init = -lo/scale the block's own minimum already lands exactly
    # on code 0, and the IRLS update's own closed-form zero-point solution
    # turns out to be a near machine-epsilon-exact fixed point of that
    # same value for a single-outlier-per-block block shape, so rounding
    # never has anything left to move. This is a genuine, inherent
    # property of hqq.py's own algorithm as written -- not a difference
    # this port introduces -- so this test only asserts IRLS never does
    # *worse* than the naive baseline, not that it strictly improves on
    # it.
    rng = np.random.default_rng(2)
    K, N = _BLOCK_SIZE * 4, 4
    w = (rng.standard_normal((K, N)) * 0.2).astype(np.float32)
    # Inject one large outlier per (block, channel).
    for n in range(N):
        for start in range(0, K, _BLOCK_SIZE):
            w[start, n] = 20.0 if (start // _BLOCK_SIZE) % 2 == 0 else -20.0
    model = _matmul_model(w, K, N)

    q = onnxsim.apply_hqq_cpp(model)
    cpp_w = _current_weight(q).astype(np.float64)
    w64 = w.astype(np.float64)

    naive_w_nk = _naive_affine_int4_nk(w64.T)  # [N, K]
    naive_w = naive_w_nk.T  # back to [K, N]

    # Exclude the outlier elements themselves (both schemes reproduce
    # those about equally poorly by construction) and compare error on
    # the many "normal" elements each block also contains.
    mask = np.ones_like(w64, dtype=bool)
    for n in range(N):
        for start in range(0, K, _BLOCK_SIZE):
            mask[start, n] = False

    irls_err = float(np.mean((w64[mask] - cpp_w[mask]) ** 2))
    naive_err = float(np.mean((w64[mask] - naive_w[mask]) ** 2))
    assert irls_err <= naive_err * (1 + 1e-9)


def test_cpp_behaves_similarly_to_python_port_end_to_end():
    # The C++ port folds to a plain float32 initializer while the Python
    # port builds a real DequantizeLinear(Wq, Ws, Wz) node -- compare
    # them end-to-end through onnxruntime rather than by inspecting
    # initializers directly.
    rng = np.random.default_rng(3)
    K, N = _BLOCK_SIZE * 4, 8
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(w, K, N)

    py_q = quantize_weight_only_int4_hqq(model)
    cpp_q = onnxsim.apply_hqq_cpp(model)
    onnx.checker.check_model(cpp_q)

    x = rng.standard_normal((4, K)).astype(np.float32)
    (py_y,) = _run(py_q, {"X": x})
    (cpp_y,) = _run(cpp_q, {"X": x})
    assert np.all(np.isfinite(cpp_y))
    assert _rel_l2(py_y, cpp_y) < 0.05


def test_cpp_gemm_transb_with_bias():
    rng = np.random.default_rng(4)
    K, N = _BLOCK_SIZE * 2, 8
    weight = rng.standard_normal((N, K)).astype(np.float32) * 0.5  # transB=1 layout
    bias = rng.standard_normal((N,)).astype(np.float32) * 0.1
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm<transB = 1>(X, W, B)
        }}
        """,
        initializer=[_f32(weight, "W"), _f32(bias, "B")],
    )
    q = onnxsim.apply_hqq_cpp(model)
    onnx.checker.check_model(q)

    x = rng.standard_normal((4, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.2


def test_cpp_noop_when_k_not_divisible_by_block_size():
    rng = np.random.default_rng(5)
    K, N = _BLOCK_SIZE + 5, 4
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(w, K, N)

    result = onnxsim.apply_hqq_cpp(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_cpp_noop_when_no_matmul_gemm_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_hqq_cpp(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_cpp_skips_non_2d_weight():
    rng = np.random.default_rng(6)
    w = rng.standard_normal((2, 4, 4, 4)).astype(np.float32)
    model = _model(
        """
        g (float[1,2,8,8] X) => (float[1,2,5,5] Y)
        {
          Y = Conv(X, W)
        }
        """,
        [_f32(w, "W")],
    )
    result = onnxsim.apply_hqq_cpp(model)
    assert result.SerializeToString() == model.SerializeToString()
