"""Tests for ``onnxsim.quantize_weight_only_squeezellm_cpp`` -- the
C++-backed port of ``onnxsim.quantize_weight_only_squeezellm`` (SqueezeLLM's
sensitivity-weighted per-group codebook plus dense-and-sparse outlier
correction, see ``onnxsim/squeezellm_entry.h``). Like
``test_llm_int8_cpp.py``/``test_gptq_cpp.py``, this runs the model over
real calibration data through a real ``onnxruntime``-backed executor --
never a fake/mock executor. Unlike this repo's k-means-family ports with
a genuine RNG divergence (``test_aqlm_cpp.py``, ``test_lo_bcq_cpp.py``),
squeezellm.py's own weighted k-means fit initializes deterministically
(evenly-spaced order statistics, no random sampling at all -- see
``squeezellm_entry.h``'s own "NUMERICAL SCOPE" note), so this file checks
exact (bit-for-bit) parity against the pure-Python reference: both sides
compute the same outlier threshold, sensitivity weights, codebooks/codes
and emit the same graph, so any divergence is a bug, not an accepted
tolerance.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

from onnxsim.onnx_simplifier import quantize_weight_only_squeezellm_cpp
from onnxsim.squeezellm import quantize_weight_only_squeezellm

ort = pytest.importorskip("onnxruntime")

# This technique's own default block_size=32/bits=4 (2**bits = 16
# centroids, comfortably below block_size) is used throughout, so no
# group's own fit degenerates into "every element gets its own exact
# centroid".
_BLOCK_SIZE = 32


def _f32(array, name):
    return onnx.numpy_helper.from_array(np.asarray(array, dtype=np.float32), name)


def _model(body, initializer=(), opset=18, ir_version=10):
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


def _matmul_model(K=_BLOCK_SIZE * 3, N=6, seed=0):
    rng = np.random.default_rng(seed)
    weight = (rng.standard_normal((K, N)) * 0.5).astype(np.float32)
    # A handful of large-magnitude elements so the outlier decomposition
    # path (SparseDiff nonzero somewhere) is genuinely exercised, not just
    # the plain codebook-fit path.
    weight[0, 0] = 25.0
    weight[5, 2] = -30.0
    return _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f32(weight, "W")],
    ), weight


def _calibration(K, num_samples=24, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((num_samples, K)).astype(np.float32)
    # A couple of input channels with much larger activation magnitude,
    # so the sensitivity weight genuinely varies across the group instead
    # of being uniform (which would make the weighted fit indistinguishable
    # from a plain unweighted one).
    x[:, 1] *= 8.0
    x[:, 7] *= 5.0
    return [{"X": x}]


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)[0]


def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-9)


def _assert_exact_parity(model, calibration_data, **kwargs):
    py = quantize_weight_only_squeezellm(model, calibration_data, **kwargs)
    cpp = quantize_weight_only_squeezellm_cpp(model, calibration_data, **kwargs)
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
    return cpp


def test_cpp_matches_python_exactly():
    model, _weight = _matmul_model()
    cpp = _assert_exact_parity(model, _calibration(K=_BLOCK_SIZE * 3))
    onnx.checker.check_model(cpp)
    ops = sorted(n.op_type for n in cpp.graph.node)
    assert "GatherND" in ops
    assert ops.count("Reshape") == 1
    assert ops.count("Add") == 1
    assert ops.count("Transpose") == 1  # plain MatMul -- not already [N, K]


def test_cpp_gemm_transb_matches_python_exactly():
    # weight_transposed=True (Gemm's own transB=1): the weight is already
    # stored [N, K], so no final Transpose node is added -- exercises the
    # other branch of that conditional.
    K, N = _BLOCK_SIZE * 2, 4
    rng = np.random.default_rng(3)
    weight = (rng.standard_normal((N, K)) * 0.5).astype(np.float32)
    weight[0, 0] = 40.0
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm<transB = 1>(X, W)
        }}
        """,
        initializer=[_f32(weight, "W")],
    )
    cpp = _assert_exact_parity(model, _calibration(K=K, seed=4))
    onnx.checker.check_model(cpp)
    ops = [n.op_type for n in cpp.graph.node]
    assert "Transpose" not in ops


def test_cpp_output_stays_close_to_float():
    K, N = _BLOCK_SIZE * 3, 6
    model, weight = _matmul_model(K=K, N=N, seed=7)
    calib = _calibration(K=K, seed=8)
    cpp = quantize_weight_only_squeezellm_cpp(model, calib)
    onnx.checker.check_model(cpp)

    x = np.random.default_rng(9).standard_normal((5, K)).astype(np.float32)
    float_y = _run(model, {"X": x})
    q_y = _run(cpp, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.3


def test_cpp_noop_when_no_matmul_gemm_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = quantize_weight_only_squeezellm_cpp(
        model, [{"X": np.zeros((2, 4), np.float32)}]
    )
    assert result.SerializeToString() == model.SerializeToString()


def test_cpp_skips_non_block_divisible_k():
    # K not a multiple of block_size (32) -- squeezellm.py's own encoder
    # skips this layer entirely, and this port matches that exactly.
    K, N = _BLOCK_SIZE + 3, 4
    rng = np.random.default_rng(11)
    weight = rng.standard_normal((K, N)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f32(weight, "W")],
    )
    result = quantize_weight_only_squeezellm_cpp(model, _calibration(K=K, seed=12))
    assert result.SerializeToString() == model.SerializeToString()


def test_cpp_declines_pre_opset12():
    model, _weight = _matmul_model()
    model.opset_import[0].version = 11
    result = quantize_weight_only_squeezellm_cpp(
        model, _calibration(K=_BLOCK_SIZE * 3), block_size=_BLOCK_SIZE
    )
    assert result.SerializeToString() == model.SerializeToString()
