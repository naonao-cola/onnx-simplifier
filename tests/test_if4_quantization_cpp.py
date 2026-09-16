"""Tests for ``onnxsim.quantize_weight_only_if4_cpp`` -- the C++-backed
port of ``onnxsim.quantize_weight_only_if4`` (see
``onnxsim/passes/if4_quantization.h``). Like IQ4_NL, IF4 has no
accumulation or iterative-refinement step at all (see that header's own
"ACCEPTED, PERMANENT DIVERGENCE" note), so this port is expected to track
the pure-Python port unusually closely -- but these tests still check
structural/algebraic properties and comparable (not required to be
bit-for-bit identical) reconstruction error, matching this repo's own
established contract for a ``*_cpp`` port (``tests/test_iq4_nl_cpp.py``,
``tests/test_gguf_q2_k_cpp.py``).

Unlike every GGUF ``*_cpp`` port, IF4's own blocks are grouped per
(output channel, 16-element reduction-axis block) rather than over the
weight's own flattened row-major storage -- so these tests build weights
with per-(channel, block) statistics (not just per flattened-block ones)
and exercise both MatMul's (weight stored [K, N]) and Gemm-transB=1's
(weight stored [N, K]) orientation explicitly.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.if4_quantization import IF4_BLOCK_SIZE

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


def _matmul_model(w, K, N, batch="batch"):
    # w stored [K, N] -- MatMul's own native (non-transposed) layout.
    return _model(
        f"""
        g (float[{batch},{K}] X) => (float[{batch},{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(w, "W")],
    )


def _gemm_transb_model(w, K, N, batch="batch"):
    # w stored [N, K] -- Gemm(transB=1)'s own layout.
    return _model(
        f"""
        g (float[{batch},{K}] X) => (float[{batch},{N}] Y)
        {{
          Y = Gemm<transB=1>(X, W)
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
    node = next(n for n in model.graph.node if n.op_type in ("MatMul", "Gemm"))
    w_name = node.input[weight_input_index]
    w_init = next(t for t in model.graph.initializer if t.name == w_name)
    return onnx.numpy_helper.to_array(w_init)


def _mixed_weight(K, N, seed):
    # Per-(output-channel, reduction-block) scale/offset diversity, so a
    # single flattened-block scheme (or a naive single-scale-per-tensor
    # one) would clearly do worse than IF4's own per-block choice.
    rng = np.random.default_rng(seed)
    num_blocks = K // IF4_BLOCK_SIZE
    block_scales = rng.uniform(0.1, 5.0, size=(N, num_blocks))
    noise = rng.standard_normal((N, num_blocks, IF4_BLOCK_SIZE))
    w_nk = (noise * block_scales[:, :, None]).reshape(N, K)
    return w_nk.astype(np.float32)  # [N, K] -- callers transpose as needed


def test_cpp_replaces_weight_with_same_shape_float():
    K, N = IF4_BLOCK_SIZE * 4, 8
    w_nk = _mixed_weight(K, N, seed=0)
    model = _matmul_model(w_nk.T, K, N)  # MatMul wants [K, N]

    q = onnxsim.quantize_weight_only_if4_cpp(model)
    onnx.checker.check_model(q)
    new_w = _current_weight(q)
    assert new_w.shape == (K, N)
    assert new_w.dtype == np.float32
    assert not np.array_equal(new_w, w_nk.T)
    assert any(t.name == "W" for t in q.graph.initializer)


@pytest.mark.parametrize("model_fn", [_matmul_model, _gemm_transb_model])
def test_cpp_at_most_16_distinct_levels_per_channel_block(model_fn):
    K, N = IF4_BLOCK_SIZE * 3, 4
    w_nk = _mixed_weight(K, N, seed=1)
    w = w_nk if model_fn is _gemm_transb_model else w_nk.T
    model = model_fn(w, K, N)

    q = onnxsim.quantize_weight_only_if4_cpp(model)
    new_w = _current_weight(q)
    new_w_nk = new_w if model_fn is _gemm_transb_model else new_w.T

    num_blocks = K // IF4_BLOCK_SIZE
    for c in range(N):
        for b in range(num_blocks):
            block = new_w_nk[c, b * IF4_BLOCK_SIZE : (b + 1) * IF4_BLOCK_SIZE]
            assert len(np.unique(block)) <= 16


@pytest.mark.parametrize("model_fn", [_matmul_model, _gemm_transb_model])
def test_cpp_behaves_similarly_to_python_port(model_fn):
    K, N = IF4_BLOCK_SIZE * 4, 6
    w_nk = _mixed_weight(K, N, seed=4)
    w = w_nk if model_fn is _gemm_transb_model else w_nk.T
    model = model_fn(w, K, N)

    py_q = onnxsim.quantize_weight_only_if4(model)
    cpp_q = onnxsim.quantize_weight_only_if4_cpp(model)

    # The Python side dequantizes via Cast/Gather/Reshape/Mul nodes; run
    # both through onnxruntime and compare final outputs rather than
    # initializers directly, since the C++ port folds into a plain
    # replacement float32 initializer instead.
    rng = np.random.default_rng(5)
    x = rng.standard_normal((4, K)).astype(np.float32)
    (py_y,) = _run(py_q, {"X": x})
    (cpp_y,) = _run(cpp_q, {"X": x})
    w64 = w_nk.astype(np.float64)

    cpp_w_nk = _current_weight(cpp_q)
    cpp_w_nk = cpp_w_nk if model_fn is _gemm_transb_model else cpp_w_nk.T
    cpp_err = np.linalg.norm(w64 - cpp_w_nk.astype(np.float64))
    assert cpp_err < np.linalg.norm(w64) * 0.3
    assert _rel_l2(py_y, cpp_y) < 0.2


def test_cpp_beats_naive_single_scale_on_mixed_blocks():
    K, N = IF4_BLOCK_SIZE * 4, 4
    w_nk = _mixed_weight(K, N, seed=3)
    model = _matmul_model(w_nk.T, K, N)

    q = onnxsim.quantize_weight_only_if4_cpp(model)
    new_w_nk = _current_weight(q).T.astype(np.float64)
    w64 = w_nk.astype(np.float64)

    naive_out = np.empty_like(w64)
    for c in range(N):
        row = w64[c]
        scale = max(np.abs(row).max(), 1e-30) / 7.0
        codes = np.clip(np.round(row / scale), -8, 7)
        naive_out[c] = codes * scale

    if4_mse = float(np.mean((w64 - new_w_nk) ** 2))
    naive_mse = float(np.mean((w64 - naive_out) ** 2))
    assert if4_mse < naive_mse


def test_cpp_gemm_with_bias():
    K, N = IF4_BLOCK_SIZE * 2, 8
    w_nk = _mixed_weight(K, N, seed=6)
    rng = np.random.default_rng(7)
    bias = rng.standard_normal((N,)).astype(np.float32) * 0.1
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm<transB=1>(X, W, B)
        }}
        """,
        initializer=[_f32(w_nk, "W"), _f32(bias, "B")],
    )
    q = onnxsim.quantize_weight_only_if4_cpp(model)
    onnx.checker.check_model(q)

    x = rng.standard_normal((4, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.3


def test_cpp_leaves_non_block_divisible_layer_untouched():
    # K not a multiple of 16 -- if4_quantization.py's own encoder skips
    # this layer entirely (no ragged-last-block handling), and this port
    # matches that exactly rather than approximating a ragged block.
    K, N = IF4_BLOCK_SIZE + 3, 4
    rng = np.random.default_rng(9)
    w = rng.standard_normal((K, N)).astype(np.float32)
    model = _matmul_model(w, K, N)

    result = onnxsim.quantize_weight_only_if4_cpp(model)
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
    result = onnxsim.quantize_weight_only_if4_cpp(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_cpp_skips_non_2d_weight():
    rng = np.random.default_rng(10)
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
    result = onnxsim.quantize_weight_only_if4_cpp(model)
    assert result.SerializeToString() == model.SerializeToString()
