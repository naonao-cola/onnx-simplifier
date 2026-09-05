"""Tests for ``onnxsim.apply_quarot_cpp`` -- the C++-backed port of
``onnxsim.apply_quarot`` (see ``onnxsim/passes/quarot.h`` and
``onnxsim/passes/random_orthogonal.h``). Unlike the MXFP4/double-quantization
C++ ports, this pass draws a fresh random rotation per layer using its own
independent RNG derivation (not a numpy Generator sequenced across matches
in graph node order), so its output is expected to be *accurate*, not
bit-identical to the Python port -- these tests check structure and
numerical accuracy rather than exact equality.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.omniquant import _quantize_blockwise_int4_with_clip

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
        weight = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    return _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f32(weight, "W")],
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


def test_cpp_quarot_quantizes_matmul_with_standard_ops_only():
    model = _matmul_model(K=32, N=8, seed=0)
    q = onnxsim.apply_quarot_cpp(model, seed=0)
    onnx.checker.check_model(q)

    op_types = {n.op_type for n in q.graph.node}
    assert op_types <= {
        "MatMul",
        "Abs",
        "ReduceMax",
        "Clip",
        "Div",
        "Round",
        "Mul",
        "DequantizeLinear",
        "Add",
        "Identity",
    }
    assert all(n.domain in ("", "ai.onnx") for n in q.graph.node)


def test_cpp_quarot_rotation_is_orthogonal():
    model = _matmul_model(K=32, N=8, seed=1)
    q = onnxsim.apply_quarot_cpp(model, seed=2)
    u = next(
        onnx.numpy_helper.to_array(t)
        for t in q.graph.initializer
        if list(t.dims) == [32, 32]
    )
    identity = u.astype(np.float64) @ u.astype(np.float64).T
    assert np.allclose(identity, np.eye(32), atol=1e-4)


def test_cpp_quarot_output_stays_close_to_float_via_onnxruntime():
    model = _matmul_model(K=32, N=8, seed=3)
    q = onnxsim.apply_quarot_cpp(model, seed=3)
    onnx.checker.check_model(q)

    rng = np.random.default_rng(4)
    x = rng.standard_normal((8, 32)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.5


def test_cpp_quarot_gemm_with_bias():
    rng = np.random.default_rng(5)
    K, N = 64, 12
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
    q = onnxsim.apply_quarot_cpp(model, seed=6)
    onnx.checker.check_model(q)
    assert any(n.op_type == "Add" for n in q.graph.node)

    x = rng.standard_normal((4, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.5


def test_cpp_quarot_skips_non_block_divisible_k():
    model = _matmul_model(K=48, N=8, seed=7)  # 48 is not a multiple of 32
    q = onnxsim.apply_quarot_cpp(model, seed=0)
    assert q.SerializeToString() == model.SerializeToString()


def test_cpp_quarot_declines_pre_opset21():
    model = _matmul_model(K=32, N=8, seed=8, opset=13)
    q = onnxsim.apply_quarot_cpp(model, seed=0)
    assert q.SerializeToString() == model.SerializeToString()


def test_cpp_quarot_is_deterministic_for_a_given_seed():
    model = _matmul_model(K=32, N=8, seed=9)
    q1 = onnxsim.apply_quarot_cpp(model, seed=42)
    q2 = onnxsim.apply_quarot_cpp(model, seed=42)
    assert q1.SerializeToString() == q2.SerializeToString()


def test_cpp_quarot_different_seeds_give_different_rotations():
    model = _matmul_model(K=32, N=8, seed=10)
    q1 = onnxsim.apply_quarot_cpp(model, seed=1)
    q2 = onnxsim.apply_quarot_cpp(model, seed=2)
    assert q1.SerializeToString() != q2.SerializeToString()


def _unpack_int4(tensor):
    # Signed low-nibble-first INT4 unpacking, matching
    # TryQuantizeWeightBlockwiseInt4InPlace's / quarot.h's own packing.
    raw = tensor.raw_data
    numel = 1
    for d in tensor.dims:
        numel *= d
    out = np.zeros(numel, dtype=np.int64)
    for i in range(numel):
        byte = raw[i // 2]
        nibble = byte & 0xF if i % 2 == 0 else (byte >> 4) & 0xF
        if nibble >= 8:
            nibble -= 16
        out[i] = nibble
    return out.reshape(list(tensor.dims))


@pytest.mark.parametrize("block_size", [16, 32, 64])
def test_cpp_quarot_block_size_matches_python_quantization_math(block_size):
    # Cross-checks the C++ port's block-quantization against
    # onnxsim.omniquant's own `_quantize_blockwise_int4_with_clip` (the same
    # routine quarot.py's own apply_quarot delegates to), applied to the
    # *same* rotation matrix the C++ port produced -- this isolates the
    # block_size threading and quantization math itself from the two
    # ports' unrelated, independently-seeded RNGs (never a cross-language
    # parity goal -- see this module's own docstring), and would catch an
    # off-by-one or wrong-axis bug in how block_size is threaded through.
    K, N = 128, 4
    rng = np.random.default_rng(100 + block_size)
    weight = (rng.standard_normal((K, N)) * 0.5).astype(np.float32)
    model = _matmul_model(K=K, N=N, weight=weight)

    q = onnxsim.apply_quarot_cpp(model, seed=7, block_size=block_size)
    onnx.checker.check_model(q)

    num_blocks = K // block_size
    u = None
    codes_t = None
    scale_t = None
    for t in q.graph.initializer:
        if list(t.dims) == [K, K]:
            u = onnx.numpy_helper.to_array(t).astype(np.float64)
        elif t.data_type == onnx.TensorProto.INT4 and list(t.dims) == [K, N]:
            codes_t = t
        elif t.data_type == onnx.TensorProto.FLOAT and list(t.dims) == [num_blocks, N]:
            scale_t = t
    assert u is not None
    assert codes_t is not None
    assert scale_t is not None, f"expected a [{num_blocks}, {N}] scale initializer"

    dequant_node = next(n for n in q.graph.node if n.op_type == "DequantizeLinear")
    block_size_attr = next(
        a.i for a in dequant_node.attribute if a.name == "block_size"
    )
    assert block_size_attr == block_size

    codes_kn_cpp = _unpack_int4(codes_t)
    scale_kn_cpp = onnx.numpy_helper.to_array(scale_t)

    w_nk = weight.astype(np.float64).T  # [N, K]
    w_tilde_nk = w_nk @ u
    codes_nk_ref, scale_blocks_nk_ref = _quantize_blockwise_int4_with_clip(
        w_tilde_nk, block_size, 1.0
    )
    codes_kn_ref = codes_nk_ref.T.astype(np.int64)
    scale_kn_ref = scale_blocks_nk_ref.T.astype(np.float32)

    np.testing.assert_array_equal(codes_kn_cpp, codes_kn_ref)
    np.testing.assert_allclose(scale_kn_cpp, scale_kn_ref, rtol=1e-5, atol=1e-6)


def test_cpp_quarot_declines_non_default_block_size_when_k_not_divisible():
    model = _matmul_model(K=40, N=8, seed=11)  # 40 is not a multiple of 16
    q = onnxsim.apply_quarot_cpp(model, seed=0, block_size=16)
    assert q.SerializeToString() == model.SerializeToString()


def test_cpp_quarot_epsilon_floors_all_zero_token_scale():
    # An all-zero token drives max(|x_rotated|) to exactly 0 for that row;
    # without an epsilon floor on the Clip feeding the scale computation,
    # this would divide by zero. Both the default epsilon and a custom one
    # must keep the graph finite and reconstruct the all-zero token as
    # exactly zero (0 / any_nonzero_scale == 0).
    model = _matmul_model(K=32, N=8, seed=12)
    x = np.zeros((2, 32), dtype=np.float32)
    x[1] = np.random.default_rng(13).standard_normal(32).astype(np.float32)

    for epsilon in (1e-12, 1e-6, 1.0):
        q = onnxsim.apply_quarot_cpp(model, seed=1, epsilon=epsilon)
        onnx.checker.check_model(q)
        sess = ort.InferenceSession(
            q.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        (y,) = sess.run(None, {"X": x})
        assert np.all(np.isfinite(y))
        assert np.allclose(y[0], 0.0)


def test_cpp_quarot_epsilon_value_is_threaded_into_the_graph():
    # The Clip node's min-bound initializer should carry exactly the
    # requested epsilon (not the old hardcoded 1e-12), distinguishing it
    # from this pass's other float scalar initializers (7.0, -7.0, 7.0).
    model = _matmul_model(K=32, N=8, seed=14)
    epsilon = 0.0625  # distinct from 7.0/-7.0 and from the default 1e-12
    q = onnxsim.apply_quarot_cpp(model, seed=0, epsilon=epsilon)

    scalar_values = [
        onnx.numpy_helper.to_array(t).item()
        for t in q.graph.initializer
        if onnx.numpy_helper.to_array(t).size == 1
    ]
    assert any(np.isclose(v, epsilon) for v in scalar_values), scalar_values


def test_cpp_quarot_huge_epsilon_measurably_changes_output():
    # A large epsilon clamps every token's scale up to epsilon / 7,
    # collapsing quantization resolution -- proving epsilon is actually
    # wired into the Clip node's bound rather than silently ignored.
    model = _matmul_model(K=32, N=8, seed=15)
    rng = np.random.default_rng(16)
    x = rng.standard_normal((4, 32)).astype(np.float32)

    q_default = onnxsim.apply_quarot_cpp(model, seed=3, epsilon=1e-12)
    q_huge = onnxsim.apply_quarot_cpp(model, seed=3, epsilon=10.0)

    def run(m):
        sess = ort.InferenceSession(
            m.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        return sess.run(None, {"X": x})[0]

    y_default = run(q_default)
    y_huge = run(q_huge)
    assert np.all(np.isfinite(y_default)) and np.all(np.isfinite(y_huge))
    assert not np.allclose(y_default, y_huge)


def test_cpp_quarot_block_size_and_epsilon_defaults_match_python():
    import inspect

    sig = inspect.signature(onnxsim.apply_quarot_cpp)
    assert sig.parameters["block_size"].default == 32
    assert sig.parameters["epsilon"].default == 1e-12
