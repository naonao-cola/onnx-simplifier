"""Tests for ``onnxsim.apply_quarot_gptq_cpp`` -- the C++-backed port of
``onnxsim.quarot``'s own ``apply_quarot_gptq`` (see
``onnxsim/quarot_gptq_entry.h`` and ``onnxsim/quarot.py``).

``onnxsim.apply_quarot_gptq`` is now a thin alias for this C++ port (see
``onnxsim/quarot.py``'s own docstring), so most of the tests below
exercise ``apply_quarot_gptq_cpp`` directly -- calling
``onnxsim.apply_quarot_gptq`` would just be an extra indirection to the
same code. Like ``test_quarot_cpp.py``, this pass draws a fresh random
rotation per layer using its own independent RNG derivation (not a numpy
Generator sequenced across matches in graph node order), so a *given
seed* is not expected to reproduce onnxsim.quarot's own pre-alias
rotation -- see ``test_quarot.py``'s own
``test_quarot_gptq_rotation_no_longer_matches_plain_quarot_after_aliasing``
for that side of it (``apply_quarot`` itself was never aliased, so the
two functions' rotations permanently diverge from each other now). The
structural/numerical-accuracy tests below check this pass's own output
rather than any cross-language comparison. The one exception is
``test_cpp_quarot_gptq_matches_python_quantization_math``, which plugs
the C++ port's own rotation and captured calibration activations into
the pure-Python GPTQ column algorithm to isolate that part of the
pipeline from RNG choice entirely -- there, exact agreement is expected
and checked.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.gptq import _gptq_quantize_columns
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
    return onnx.numpy_helper.from_array(np.asarray(array, dtype=np.float32), name)


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


def _correlated_calibration(K, num_samples=32, rank=6, seed=1):
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((num_samples, rank)).astype(np.float32)
    projection = rng.standard_normal((rank, K)).astype(np.float32)
    x = latent @ projection
    x += rng.standard_normal((num_samples, K)).astype(np.float32) * 0.05
    return [{"X": x}]


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-6)


def _unpack_int4(tensor):
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


def test_cpp_quarot_gptq_quantizes_matmul_with_standard_ops_only():
    model = _matmul_model(K=32, N=8, seed=0)
    calib = _correlated_calibration(K=32, seed=2)
    q = onnxsim.apply_quarot_gptq_cpp(model, calibration_data=calib, seed=0)
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


def test_cpp_quarot_gptq_rotation_is_orthogonal():
    model = _matmul_model(K=32, N=8, seed=1)
    calib = _correlated_calibration(K=32, seed=3)
    q = onnxsim.apply_quarot_gptq_cpp(model, calibration_data=calib, seed=2)
    u = next(
        onnx.numpy_helper.to_array(t)
        for t in q.graph.initializer
        if list(t.dims) == [32, 32]
    )
    identity = u.astype(np.float64) @ u.astype(np.float64).T
    assert np.allclose(identity, np.eye(32), atol=1e-4)


def test_cpp_quarot_gptq_output_stays_close_to_float_via_onnxruntime():
    model = _matmul_model(K=32, N=8, seed=3)
    calib = _correlated_calibration(K=32, seed=4)
    q = onnxsim.apply_quarot_gptq_cpp(model, calibration_data=calib, seed=3)
    onnx.checker.check_model(q)

    rng = np.random.default_rng(5)
    x = rng.standard_normal((8, 32)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.5


def test_cpp_quarot_gptq_gemm_with_bias():
    rng = np.random.default_rng(6)
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
    calib = _correlated_calibration(K=K, seed=7)
    q = onnxsim.apply_quarot_gptq_cpp(model, calibration_data=calib, seed=6)
    onnx.checker.check_model(q)
    assert any(n.op_type == "Add" for n in q.graph.node)

    x = rng.standard_normal((4, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.5


def test_cpp_quarot_gptq_skips_non_block_divisible_k():
    model = _matmul_model(K=48, N=8, seed=8)  # 48 is not a multiple of 32
    calib = _correlated_calibration(K=48, seed=9)
    q = onnxsim.apply_quarot_gptq_cpp(model, calibration_data=calib, seed=0)
    assert q.SerializeToString() == model.SerializeToString()


def test_cpp_quarot_gptq_declines_pre_opset21():
    model = _matmul_model(K=32, N=8, seed=10, opset=13)
    calib = _correlated_calibration(K=32, seed=11)
    q = onnxsim.apply_quarot_gptq_cpp(model, calibration_data=calib, seed=0)
    assert q.SerializeToString() == model.SerializeToString()


def test_cpp_quarot_gptq_skips_layer_with_no_calibration_data():
    model = _matmul_model(K=32, N=8, seed=12)
    q = onnxsim.apply_quarot_gptq_cpp(model, calibration_data=[], seed=0)
    assert q.SerializeToString() == model.SerializeToString()


def test_cpp_quarot_gptq_is_deterministic_for_a_given_seed():
    model = _matmul_model(K=32, N=8, seed=13)
    calib = _correlated_calibration(K=32, seed=14)
    q1 = onnxsim.apply_quarot_gptq_cpp(model, calibration_data=calib, seed=42)
    q2 = onnxsim.apply_quarot_gptq_cpp(model, calibration_data=calib, seed=42)
    assert q1.SerializeToString() == q2.SerializeToString()


def test_cpp_quarot_gptq_different_seeds_give_different_rotations():
    model = _matmul_model(K=32, N=8, seed=15)
    calib = _correlated_calibration(K=32, seed=16)
    q1 = onnxsim.apply_quarot_gptq_cpp(model, calibration_data=calib, seed=1)
    q2 = onnxsim.apply_quarot_gptq_cpp(model, calibration_data=calib, seed=2)
    assert q1.SerializeToString() != q2.SerializeToString()


@pytest.mark.parametrize("block_size", [16, 32, 64])
def test_cpp_quarot_gptq_matches_python_quantization_math(block_size):
    # Cross-checks the C++ port's Hessian computation and GPTQ column
    # search against onnxsim.gptq's own `_gptq_quantize_columns` and
    # onnxsim.omniquant's own `_quantize_blockwise_int4_with_clip` (the
    # same routines quarot.py's own apply_quarot_gptq delegates to),
    # applied to the *same* rotation matrix and captured calibration
    # activations the C++ port used -- this isolates the Hessian/GPTQ math
    # itself from the two ports' unrelated, independently-seeded RNGs
    # (never a cross-language parity goal for the rotation -- see this
    # module's own docstring).
    K, N = 128, 4
    rng = np.random.default_rng(100 + block_size)
    weight = (rng.standard_normal((K, N)) * 0.5).astype(np.float32)
    model = _matmul_model(K=K, N=N, weight=weight)
    calib = _correlated_calibration(K=K, num_samples=40, rank=8, seed=200 + block_size)

    q = onnxsim.apply_quarot_gptq_cpp(
        model, calibration_data=calib, seed=7, block_size=block_size
    )
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

    codes_kn_cpp = _unpack_int4(codes_t)
    scale_kn_cpp = onnx.numpy_helper.to_array(scale_t)

    x = np.concatenate([batch["X"] for batch in calib], axis=0).astype(np.float64)
    w_nk = weight.astype(np.float64).T  # [N, K]
    w_tilde_nk = w_nk @ u
    x_rotated = x @ u
    h = x_rotated.T @ x_rotated

    _, scale_blocks_nk_ref = _quantize_blockwise_int4_with_clip(
        w_tilde_nk, block_size, 1.0
    )
    codes_nk_ref = _gptq_quantize_columns(
        w_tilde_nk, scale_blocks_nk_ref, block_size, h, 0.01, 128
    )
    codes_kn_ref = codes_nk_ref.T.astype(np.int64)
    scale_kn_ref = scale_blocks_nk_ref.T.astype(np.float32)

    np.testing.assert_array_equal(codes_kn_cpp, codes_kn_ref)
    np.testing.assert_allclose(scale_kn_cpp, scale_kn_ref, rtol=1e-5, atol=1e-6)


def test_cpp_quarot_gptq_python_alias_matches_the_cpp_port_exactly():
    # onnxsim.apply_quarot_gptq is now a thin alias for this C++ port (see
    # onnxsim/quarot.py's own docstring) -- unlike the old two-independent-
    # implementations relationship this test module's own docstring
    # describes, calling either name for the same arguments must produce
    # byte-identical output, since they're the same code underneath.
    K, N = 32, 8
    model = _matmul_model(K=K, N=N, seed=20)
    calib = _correlated_calibration(K=K, seed=21)

    py_q = onnxsim.apply_quarot_gptq(model, calibration_data=calib, seed=123)
    cpp_q = onnxsim.apply_quarot_gptq_cpp(model, calibration_data=calib, seed=123)
    assert py_q.SerializeToString() == cpp_q.SerializeToString()
