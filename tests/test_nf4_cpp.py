"""Tests for ``onnxsim.quantize_weight_only_nf4_cpp`` -- the C++-backed
port of ``onnxsim.quantize_weight_only_nf4`` (see
``onnxsim/passes/nf4.h``). Like IQ4_NL, this format has no accumulation
or iterative-refinement step at all (see that header's own "ACCEPTED,
PERMANENT DIVERGENCE" note), so this port is expected to track the
pure-Python port unusually closely -- but these tests still check
structural/algebraic properties and comparable (not required to be
bit-for-bit identical) reconstruction error, matching this repo's own
established contract for a ``*_cpp`` port (``tests/test_iq4_nl_cpp.py``).

Unlike IQ4_NL (which blocks over the weight's own flattened row-major
storage), NF4 blocks *per output channel* along the reduction axis --
see ``passes/nf4.h``'s own scope note -- so a non-block-divisible layer
is left untouched rather than zero-padded, and there is no
ragged-last-block test here (there is no ragged block at all: a
mismatched layer is simply skipped, covered by
``test_cpp_skips_non_block_divisible_weight`` below).
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.nf4 import NF4_CODEBOOK

_BLOCK_SIZE = 64  # nf4.py's own default block_size, matching passes/nf4.h

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
    # The C++ pass (like every other *_cpp weight-only port here) rewires
    # the matched node's weight input to a freshly created initializer,
    # leaving the original one dangling unused in the graph -- so the
    # *node's own current input name* is the only reliable way to find
    # the actual (post-quantization) weight, not initializer list
    # position.
    node = next(n for n in model.graph.node if n.op_type in ("MatMul", "Gemm"))
    w_name = node.input[weight_input_index]
    w_init = next(t for t in model.graph.initializer if t.name == w_name)
    return onnx.numpy_helper.to_array(w_init)


def test_cpp_nf4_replaces_weight_with_same_shape_float():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((_BLOCK_SIZE * 2, 8)).astype(np.float32)
    model = _matmul_model(w, K=_BLOCK_SIZE * 2, N=8)

    q = onnxsim.quantize_weight_only_nf4_cpp(model)
    onnx.checker.check_model(q)
    new_w = _current_weight(q)
    assert new_w.shape == w.shape
    assert new_w.dtype == np.float32
    assert not np.array_equal(new_w, w)
    # The original initializer is left in the graph, unused -- matching
    # every other *_cpp weight-only port's own established convention.
    assert any(t.name == "W" for t in q.graph.initializer)


def test_cpp_nf4_at_most_16_distinct_levels_per_channel_block():
    rng = np.random.default_rng(1)
    K, N = _BLOCK_SIZE * 3, 4
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(w, K, N)

    q = onnxsim.quantize_weight_only_nf4_cpp(model)
    new_w = _current_weight(q)  # [K, N]: reduction axis is axis 0 here

    for n in range(N):
        for start in range(0, K, _BLOCK_SIZE):
            block = new_w[start : start + _BLOCK_SIZE, n]
            assert len(np.unique(block)) <= 16


def test_cpp_nf4_reconstructed_values_match_fixed_codebook_shape():
    rng = np.random.default_rng(2)
    K, N = _BLOCK_SIZE * 2, 4
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.3
    model = _matmul_model(w, K, N)

    q = onnxsim.quantize_weight_only_nf4_cpp(model)
    new_w = _current_weight(q).astype(np.float64)

    codebook = np.asarray(NF4_CODEBOOK)
    w64 = w.astype(np.float64)
    for n in range(N):
        for start in range(0, K, _BLOCK_SIZE):
            block_in = w64[start : start + _BLOCK_SIZE, n]
            block_out = new_w[start : start + _BLOCK_SIZE, n]
            scale = max(np.abs(block_in).max(), 1e-12)
            ratios = block_out / scale
            nearest_dist = np.min(np.abs(ratios[:, None] - codebook[None, :]), axis=1)
            assert np.all(nearest_dist < 1e-4)


def test_cpp_nf4_reduces_reconstruction_error_versus_naive_uniform_int4():
    # The core empirical claim NF4 exists for, checked directly against
    # the C++ port's own actual output: a fixed non-uniform codebook
    # should reconstruct roughly-Gaussian weights more accurately than a
    # naive uniform 4-bit grid at the same block size.
    rng = np.random.default_rng(3)
    K, N = _BLOCK_SIZE * 64, 2
    w = rng.standard_normal((K, N)).astype(np.float32)
    model = _matmul_model(w, K, N)

    q = onnxsim.quantize_weight_only_nf4_cpp(model)
    new_w = _current_weight(q).astype(np.float64)
    w64 = w.astype(np.float64)

    naive_out = np.empty_like(w64)
    for n in range(N):
        for start in range(0, K, _BLOCK_SIZE):
            block = w64[start : start + _BLOCK_SIZE, n]
            scale = max(np.abs(block).max(), 1e-12) / 7.0
            codes = np.clip(np.round(block / scale), -7, 7)
            naive_out[start : start + _BLOCK_SIZE, n] = codes * scale

    nf4_mse = float(np.mean((w64 - new_w) ** 2))
    naive_mse = float(np.mean((w64 - naive_out) ** 2))
    assert nf4_mse < naive_mse


def test_cpp_nf4_behaves_similarly_to_python_port():
    # Not required to be bit-for-bit identical (see passes/nf4.h's own
    # documented divergence note), but should reach a very similar
    # reconstruction error on the same input -- this scheme has no
    # accumulation/iteration-order dependence at all, so the two ports
    # are expected to track each other unusually closely among this
    # repo's *_cpp pairs. The Python port's own output is a MatMul fed by
    # Gather/Reshape/Mul (not a plain replaced initializer), so this
    # compares the *effective* dequantized weight computed independently
    # in numpy here, not the Python graph's own initializer list.
    rng = np.random.default_rng(4)
    K, N = _BLOCK_SIZE * 4, 8
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(w, K, N)
    w64 = w.astype(np.float64)

    codebook = np.asarray(NF4_CODEBOOK)
    py_w = np.empty_like(w64)
    for n in range(N):
        for start in range(0, K, _BLOCK_SIZE):
            block = w64[start : start + _BLOCK_SIZE, n]
            scale = max(np.abs(block).max(), 1e-12)
            normalized = block / scale
            nearest = codebook[
                np.argmin(np.abs(normalized[:, None] - codebook[None, :]), axis=1)
            ]
            py_w[start : start + _BLOCK_SIZE, n] = nearest * scale

    cpp_q = onnxsim.quantize_weight_only_nf4_cpp(model)
    cpp_w = _current_weight(cpp_q).astype(np.float64)

    py_err = np.linalg.norm(py_w - w64)
    cpp_err = np.linalg.norm(cpp_w - w64)
    assert cpp_err < np.linalg.norm(w64) * 0.2
    assert cpp_err < py_err * 1.5 and py_err < cpp_err * 1.5


def test_cpp_nf4_gemm_with_bias():
    rng = np.random.default_rng(5)
    K, N = _BLOCK_SIZE * 2, 8
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    bias = rng.standard_normal((N,)).astype(np.float32) * 0.1
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm(X, W, B)
        }}
        """,
        initializer=[_f32(weight, "W"), _f32(bias, "B")],
    )
    q = onnxsim.quantize_weight_only_nf4_cpp(model)
    onnx.checker.check_model(q)

    x = rng.standard_normal((4, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.2


def test_cpp_nf4_gemm_transb_uses_channel_first_layout():
    # transB=1 stores W as [N, K] -- the reduction axis moves to axis 1,
    # exercising passes/nf4.h's own weight_transposed branch.
    rng = np.random.default_rng(8)
    K, N = _BLOCK_SIZE * 2, 6
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
    q = onnxsim.quantize_weight_only_nf4_cpp(model)
    onnx.checker.check_model(q)
    new_w = _current_weight(q)
    assert new_w.shape == weight.shape

    for n in range(N):
        for start in range(0, K, _BLOCK_SIZE):
            block = new_w[n, start : start + _BLOCK_SIZE]
            assert len(np.unique(block)) <= 16


def test_cpp_nf4_noop_when_no_matmul_gemm_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.quantize_weight_only_nf4_cpp(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_cpp_nf4_skips_non_2d_weight():
    rng = np.random.default_rng(7)
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
    result = onnxsim.quantize_weight_only_nf4_cpp(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_cpp_nf4_skips_non_block_divisible_weight():
    # K not a multiple of 64 -- passes/nf4.h's own scope note: unlike
    # gguf_legacy_quant.h's/iq4_nl.h's flattened, order-agnostic blocking,
    # NF4's per-channel blocking cannot zero-pad without changing which
    # elements share a scale group, so a non-block-divisible layer is
    # left completely untouched instead.
    rng = np.random.default_rng(9)
    K, N = _BLOCK_SIZE + 5, 4
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.4
    model = _matmul_model(w, K, N)
    result = onnxsim.quantize_weight_only_nf4_cpp(model)
    assert result.SerializeToString() == model.SerializeToString()
