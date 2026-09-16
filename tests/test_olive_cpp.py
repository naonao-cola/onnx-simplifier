"""Tests for ``onnxsim.apply_olive_cpp`` -- the C++-backed port of
``onnxsim.quantize_weight_only_olive`` (OliVe's Outlier-Victim Pair
quantization, see ``onnxsim/passes/olive.h``). Unlike the Python side,
which builds a real ``<int8 codes>+BaseScale+OutlierScale+OutlierMask``
graph rewrite (``DequantizeLinear`` x2 + ``Cast`` + ``Where`` +
``MatMul``[+``Add``], needing opset 21), this port folds the OVP round
trip directly into a replacement float32 initializer (see that header's
own "ACCEPTED, PERMANENT DIVERGENCE" note) -- so cross-checking against
the Python port is done end-to-end through onnxruntime rather than by
comparing initializers directly. This scheme has no accumulation or
iterative-refinement step at all, so this port is expected to track the
Python port's own float64 numpy implementation closely, up to
floating-point summation-order/median-tie-breaking differences --
comparable, not required to be bit-for-bit identical, matching this
repo's established contract for every other data-free ``*_cpp`` port.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.olive import quantize_weight_only_olive

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


def _naive_single_scale_int4_nk(w_nk, block_size=_BLOCK_SIZE):
    # A naive scheme that gives every element in a block a symmetric
    # 4-bit code against ONE scale derived from the block's own absmax
    # (i.e. no outlier/victim renegotiation at all) -- OliVe's whole
    # point is that its ordinary elements should reconstruct *better*
    # than this on a block containing a real outlier, since base_scale
    # excludes the outlier from its own max computation.
    n, k = w_nk.shape
    nb = k // block_size
    blocks = w_nk.reshape(n, nb, block_size)
    scale = np.maximum(np.abs(blocks).max(axis=2) / 7.0, 1e-12)
    scale3 = scale[:, :, np.newaxis]
    code = np.clip(np.round(blocks / scale3), -7, 7)
    return (code * scale3).reshape(n, k)


def test_cpp_replaces_weight_with_same_shape_float():
    rng = np.random.default_rng(0)
    K, N = _BLOCK_SIZE * 2, 8
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.3
    model = _matmul_model(w, K, N)

    q = onnxsim.apply_olive_cpp(model)
    onnx.checker.check_model(q)
    new_w = _current_weight(q)
    assert new_w.shape == w.shape
    assert new_w.dtype == np.float32
    assert not np.array_equal(new_w, w)
    # The original initializer is left in the graph, unused -- matching
    # every sibling data-free *_cpp port's established convention.
    assert any(t.name == "W" for t in q.graph.initializer)


def test_cpp_beats_naive_single_scale_on_outlier_heavy_blocks():
    # OliVe's own point: excluding the outlier from base_scale's own max
    # computation should let the block's many ordinary elements
    # reconstruct tighter than a naive scheme whose single scale is
    # dragged wide by the outlier.
    rng = np.random.default_rng(2)
    K, N = _BLOCK_SIZE * 4, 4
    w = (rng.standard_normal((K, N)) * 0.2).astype(np.float32)
    # Inject one large outlier per (block, channel) at an even (pairable)
    # position, guaranteeing an OVP pair every block.
    for n in range(N):
        for start in range(0, K, _BLOCK_SIZE):
            w[start, n] = 20.0 if (start // _BLOCK_SIZE) % 2 == 0 else -20.0
    model = _matmul_model(w, K, N)

    q = onnxsim.apply_olive_cpp(model)
    cpp_w = _current_weight(q).astype(np.float64)
    w64 = w.astype(np.float64)

    naive_w_nk = _naive_single_scale_int4_nk(w64.T)  # [N, K]
    naive_w = naive_w_nk.T  # back to [K, N]

    # Exclude the outlier and its paired victim (position start+1, since
    # the outlier sits at the even position start) -- both schemes handle
    # those two very differently by construction -- and compare error on
    # the many ordinary elements each block also contains.
    mask = np.ones_like(w64, dtype=bool)
    for n in range(N):
        for start in range(0, K, _BLOCK_SIZE):
            mask[start, n] = False
            mask[start + 1, n] = False

    olive_err = float(np.mean((w64[mask] - cpp_w[mask]) ** 2))
    naive_err = float(np.mean((w64[mask] - naive_w[mask]) ** 2))
    assert olive_err < naive_err


def test_cpp_outlier_reconstructs_much_better_than_naive_single_scale():
    # The other half of OliVe's claim: an outlier's own dedicated
    # outlier_scale (fit to outlier magnitudes, with a wider code) should
    # reconstruct it with much lower error than a single block-wide naive
    # scale would. Using one *identical* outlier value shared by every
    # row (as an earlier version of this test did) is a degenerate case:
    # since that value IS the block's own max, it defines both OliVe's
    # outlier_scale AND the naive scheme's own single scale, so it round-
    # trips exactly (zero error) under *either* scheme -- verified
    # directly against onnxsim.olive's own _olive_quantize_blockwise on
    # that exact input before concluding this was a test-design issue,
    # not a port bug. Using a *smaller* second outlier lets OliVe's own
    # wider, outlier-magnitude-fit code actually show its advantage: the
    # naive scheme's single scale is still set by the block's largest
    # value, so a smaller (but still-outlier) value quantizes coarsely
    # against it, while OliVe's own outlier_scale is fit to the outlier
    # population and gives it much finer resolution.
    rng = np.random.default_rng(9)
    K, N = _BLOCK_SIZE * 2, 4
    w = (rng.standard_normal((K, N)) * 0.2).astype(np.float32)
    for n in range(N):
        w[0, n] = 25.0  # sets the block's own max/scale in both schemes
        w[2, n] = 12.3  # a second, smaller outlier -- the one under test

    model = _matmul_model(w, K, N)
    q = onnxsim.apply_olive_cpp(model)
    cpp_w = _current_weight(q).astype(np.float64)
    w64 = w.astype(np.float64)

    naive_w_nk = _naive_single_scale_int4_nk(w64.T)
    naive_w = naive_w_nk.T

    olive_outlier_err = float(np.mean(np.abs(w64[2, :] - cpp_w[2, :])))
    naive_outlier_err = float(np.mean(np.abs(w64[2, :] - naive_w[2, :])))
    assert olive_outlier_err < naive_outlier_err


def test_cpp_behaves_similarly_to_python_port_end_to_end():
    # The C++ port folds to a plain float32 initializer while the Python
    # port builds a real DequantizeLinear/Where node rewrite -- compare
    # them end-to-end through onnxruntime rather than by inspecting
    # initializers directly.
    rng = np.random.default_rng(3)
    K, N = _BLOCK_SIZE * 4, 8
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.4
    # A few injected outliers so both the "ordinary" and "OVP" code paths
    # actually get exercised.
    w[0, 0] = 10.0
    w[_BLOCK_SIZE, 1] = -12.0
    model = _matmul_model(w, K, N)

    py_q = quantize_weight_only_olive(model)
    cpp_q = onnxsim.apply_olive_cpp(model)
    onnx.checker.check_model(cpp_q)

    x = rng.standard_normal((4, K)).astype(np.float32)
    (py_y,) = _run(py_q, {"X": x})
    (cpp_y,) = _run(cpp_q, {"X": x})
    assert np.all(np.isfinite(cpp_y))
    assert _rel_l2(py_y, cpp_y) < 0.05


def test_cpp_gemm_transb_with_bias():
    rng = np.random.default_rng(4)
    K, N = _BLOCK_SIZE * 2, 8
    weight = rng.standard_normal((N, K)).astype(np.float32) * 0.3  # transB=1 layout
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
    q = onnxsim.apply_olive_cpp(model)
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

    result = onnxsim.apply_olive_cpp(model)
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
    result = onnxsim.apply_olive_cpp(model)
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
    result = onnxsim.apply_olive_cpp(model)
    assert result.SerializeToString() == model.SerializeToString()
