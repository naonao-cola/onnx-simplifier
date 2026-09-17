"""Tests for ``onnxsim.apply_foem_cpp`` -- the C++-backed port of
``onnxsim.apply_foem`` (FOEM's sequential, Hessian-*and*-first-order-drift-
compensated rounding, see ``onnxsim/foem.py``). Like ``test_gptq_cpp.py``
(which this extends, following ``onnxsim/foem_entry.h``'s own "GPTQ plus
one small additional term" scope), this runs the float model over real
calibration data through a real ``onnxruntime``-backed executor and checks
exact (bit-for-bit) agreement against the pure-Python reference: both sides
join the same candidates, factor the same Hessian, and apply the same
first-order-drift correction, so any divergence is a bug, not an accepted
tolerance. (The dense inverse/Cholesky use scalar double-precision kernels
rather than LAPACK; agreement is measured, not assumed -- every test below
asserts it, including under ill-conditioned calibration data.)
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.foem import apply_foem
from onnxsim.gptq import apply_gptq
from onnxsim.onnx_simplifier import apply_foem_cpp, apply_gptq_cpp

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


def _matmul_model(K=64, N=16, seed=0):
    rng = np.random.default_rng(seed)
    weight = (rng.standard_normal((K, N)) * 0.5).astype(np.float32)
    return _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f32(weight, "W")],
    )


def _correlated_calibration(K=64, num_samples=64, rank=6, seed=1):
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((num_samples, rank)).astype(np.float32)
    projection = rng.standard_normal((rank, K)).astype(np.float32)
    x = latent @ projection
    x += rng.standard_normal((num_samples, K)).astype(np.float32) * 0.05
    return [{"X": x}]


def _int4_codes(model):
    dq = next(n for n in model.graph.node if n.op_type == "DequantizeLinear")
    wq = next(t for t in model.graph.initializer if t.name == dq.input[0])
    numel = int(np.prod(list(wq.dims)))
    raw = np.frombuffer(wq.raw_data, dtype=np.uint8)
    lo = (raw & 0x0F).astype(np.int16)
    hi = ((raw >> 4) & 0x0F).astype(np.int16)
    lo = np.where(lo >= 8, lo - 16, lo)
    hi = np.where(hi >= 8, hi - 16, hi)
    codes = np.empty(numel, dtype=np.int16)
    codes[0::2] = lo[: (numel + 1) // 2]
    codes[1::2] = hi[: numel // 2]
    return codes.reshape([d for d in wq.dims])


def _assert_exact_parity(float_model, calibration_data, **kwargs):
    quant = onnxsim.quantize_weight_only_int4(float_model)
    py = apply_foem(float_model, quant, calibration_data, **kwargs)
    cpp = apply_foem_cpp(float_model, quant, calibration_data, **kwargs)
    assert np.array_equal(_int4_codes(py), _int4_codes(cpp))
    py_inits = sorted(py.graph.initializer, key=lambda t: t.name)
    cpp_inits = sorted(cpp.graph.initializer, key=lambda t: t.name)
    assert [t.name for t in py_inits] == [t.name for t in cpp_inits]
    for a, b in zip(py_inits, cpp_inits):
        assert a.data_type == b.data_type, a.name
        assert list(a.dims) == list(b.dims), a.name
        assert a.raw_data == b.raw_data, a.name
    return cpp


def test_foem_cpp_matches_python_exactly():
    _assert_exact_parity(_matmul_model(), _correlated_calibration())


def test_foem_cpp_matches_python_across_shapes_and_blocks():
    for K, N, seed, procb in [
        (128, 32, 5, 64),
        (256, 64, 11, 128),
        (96, 24, 21, 48),
        (32, 8, 23, 16),
    ]:
        model = _matmul_model(K=K, N=N, seed=seed)
        cals = _correlated_calibration(K=K, seed=seed + 100)
        _assert_exact_parity(model, cals, proc_block_size=procb)


def test_foem_cpp_matches_python_across_foem_beta():
    model = _matmul_model(K=64, N=16, seed=6)
    cals = _correlated_calibration(K=64, seed=106)
    for beta in [0.0, 0.001, 0.005, 0.02, 0.1]:
        _assert_exact_parity(model, cals, foem_beta=beta)


def test_foem_cpp_zero_beta_matches_plain_gptq():
    # foem_beta == 0.0 collapses FOEM to plain GPTQ exactly -- both
    # algorithms, on both language sides, should then agree.
    model = _matmul_model(K=64, N=16, seed=17)
    cals = _correlated_calibration(K=64, seed=117)
    quant = onnxsim.quantize_weight_only_int4(model)

    foem_cpp = apply_foem_cpp(model, quant, cals, foem_beta=0.0)
    gptq_cpp = apply_gptq_cpp(model, quant, cals)
    assert np.array_equal(_int4_codes(foem_cpp), _int4_codes(gptq_cpp))

    foem_py = apply_foem(model, quant, cals, foem_beta=0.0)
    gptq_py = apply_gptq(model, quant, cals)
    assert np.array_equal(_int4_codes(foem_py), _int4_codes(gptq_py))


def test_foem_cpp_gemm_transb():
    rng = np.random.default_rng(8)
    K, N = 96, 12
    weight = (rng.standard_normal((N, K)) * 0.5).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm<transB = 1>(X, W)
        }}
        """,
        initializer=[_f32(weight, "W")],
    )
    _assert_exact_parity(model, _correlated_calibration(K=K, seed=9))


def test_foem_cpp_ill_conditioned_calibration():
    # Dead channels plus near-duplicate channels: the Hessian is singular
    # without the dead-fix and damping, exercising exactly the
    # regularization paths most likely to diverge between LAPACK and
    # scalar kernels -- agreement must still be exact.
    K, N = 64, 16
    model = _matmul_model(K=K, N=N, seed=10)
    rng = np.random.default_rng(11)
    x = rng.standard_normal((64, K)).astype(np.float32)
    x[:, 5:10] = 0.0
    x[:, 20] = x[:, 21] + 1e-7 * rng.standard_normal(64).astype(np.float32)
    _assert_exact_parity(model, [{"X": x}], percdamp=1e-6)


def test_foem_cpp_multi_batch_and_3d_activation():
    K, N = 32, 8
    model = _matmul_model(K=K, N=N, seed=12)
    rng = np.random.default_rng(13)
    cals = [
        {"X": rng.standard_normal((8, K)).astype(np.float32)},
        {"X": rng.standard_normal((12, K)).astype(np.float32)},
    ]
    _assert_exact_parity(model, cals)

    model3d = _model(
        f"""
        g (float[batch,seq,{K}] X) => (float[batch,seq,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f32(rng.standard_normal((K, N)).astype(np.float32), "W")],
    )
    cals3d = [{"X": rng.standard_normal((2, 5, K)).astype(np.float32)}]
    cpp = _assert_exact_parity(model3d, cals3d)
    onnx.checker.check_model(cpp)


def test_foem_cpp_biased_gemm():
    rng = np.random.default_rng(14)
    K, N = 32, 8
    biased = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm(X, W, B)
        }}
        """,
        initializer=[
            _f32(rng.standard_normal((K, N)).astype(np.float32), "W"),
            _f32(rng.standard_normal((N,)).astype(np.float32), "B"),
        ],
    )
    cpp = _assert_exact_parity(biased, _correlated_calibration(K=K, seed=15))
    b_new = onnx.numpy_helper.to_array(
        next(t for t in cpp.graph.initializer if t.name == "B")
    )
    assert np.array_equal(
        b_new, onnx.numpy_helper.to_array(biased.graph.initializer[1])
    )


def test_foem_cpp_skips():
    model = _matmul_model()
    quant = onnxsim.quantize_weight_only_int4(model)
    for fn in (apply_foem, apply_foem_cpp):
        out = fn(model, quant, [])
        assert [n.op_type for n in out.graph.node] == ["DequantizeLinear", "MatMul"]


def test_foem_cpp_beats_round_to_nearest_via_onnxruntime():
    model = _matmul_model(K=64, N=16, seed=16)
    x = _correlated_calibration(K=64, num_samples=64, seed=17)[0]["X"]
    quant = onnxsim.quantize_weight_only_int4(model)
    cpp = apply_foem_cpp(model, quant, [{"X": x}])
    py = apply_foem(model, quant, [{"X": x}])
    onnx.checker.check_model(cpp)
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    q_sess = ort.InferenceSession(
        quant.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    c_sess = ort.InferenceSession(
        cpp.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    p_sess = ort.InferenceSession(
        py.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    ref = sess.run(["Y"], {"X": x})[0].astype(np.float64).ravel()
    norm = max(np.linalg.norm(ref), 1e-6)
    rtn_err = (
        np.linalg.norm(ref - q_sess.run(["Y"], {"X": x})[0].astype(np.float64).ravel())
        / norm
    )
    cpp_err = (
        np.linalg.norm(ref - c_sess.run(["Y"], {"X": x})[0].astype(np.float64).ravel())
        / norm
    )
    py_err = (
        np.linalg.norm(ref - p_sess.run(["Y"], {"X": x})[0].astype(np.float64).ravel())
        / norm
    )
    assert np.all(np.isfinite(c_sess.run(["Y"], {"X": x})[0]))
    assert cpp_err == py_err
    assert cpp_err < rtn_err
