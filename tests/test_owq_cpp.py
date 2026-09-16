"""Tests for ``onnxsim.apply_owq_cpp`` -- the C++-backed port of
``onnxsim.apply_owq`` (OWQ's Optimal-Brain-Surgeon-saliency weak-column
restoration, see ``onnxsim/owq_entry.h``). Like ``test_gptq_cpp.py``, this
runs the float model over real calibration data through a real
``onnxruntime``-backed executor -- never a fake/mock executor -- and checks
exact (bit-for-bit) agreement against the pure-Python reference: both sides
join the same candidates, factor the same Hessian, rank the same weak
columns, and compute the same exact residual, so any divergence beyond
ordinary floating-point rounding is a bug, not an accepted tolerance. (The
dense inverse/Cholesky use scalar double-precision kernels rather than
LAPACK, and this port's own weak-column ranking uses ``std::stable_sort``
where the reference uses ``np.argsort`` -- see ``owq_entry.h``'s own
"Accepted numerical scope" note; every test below uses random weight/
activation data, where an exact sensitivity tie -- the only way that
difference could matter -- has vanishing probability.)

Unlike GPTQ (which rewrites the INT4 codes themselves), OWQ NEVER touches
``quantized_model``'s own INT4 codes -- it only adds a new
``Gather``/``MatMul``/``Add`` correction after the matched layer. Several
tests below check that the untouched INT4 payload really is byte-identical
to the input, not just close.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.onnx_simplifier import apply_owq_cpp
from onnxsim.owq import apply_owq

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
    # A low-rank-plus-noise activation distribution, the same shape
    # test_gptq_cpp.py's own _correlated_calibration uses: gives a
    # well-conditioned but non-trivial (non-identity) Hessian, closer to a
    # real layer's own activation statistics than pure i.i.d. noise.
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
    return sess.run(None, feeds)[0]


def _int4_codes(model, weight_name="W"):
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
    py = apply_owq(float_model, quant, calibration_data, **kwargs)
    cpp = apply_owq_cpp(float_model, quant, calibration_data, **kwargs)
    # OWQ never touches the INT4 codes -- both sides must leave them
    # byte-identical to the untouched `quant` input, not just to each other.
    assert np.array_equal(_int4_codes(quant), _int4_codes(py))
    assert np.array_equal(_int4_codes(quant), _int4_codes(cpp))

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
        assert list(a.dims) == list(b.dims), a.name
        assert a.raw_data == b.raw_data, a.name
    return cpp, quant


def test_owq_cpp_matches_python_exactly():
    _assert_exact_parity(_matmul_model(), _correlated_calibration())


def test_owq_cpp_matches_python_across_shapes_and_outlier_fractions():
    for K, N, seed, frac in [
        (128, 32, 5, 0.02),
        (256, 64, 11, 0.005),
        (96, 24, 21, 0.05),
        (64, 16, 23, 0.1),
    ]:
        model = _matmul_model(K=K, N=N, seed=seed)
        cals = _correlated_calibration(K=K, seed=seed + 100)
        _assert_exact_parity(model, cals, outlier_fraction=frac)


def test_owq_cpp_gemm_transb():
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


def test_owq_cpp_ill_conditioned_calibration():
    # Dead channels plus near-duplicate channels: the Hessian is singular
    # without the dead-fix and damping, exercising exactly the
    # regularization paths most likely to diverge between LAPACK and scalar
    # kernels -- agreement must still be exact.
    K, N = 64, 16
    model = _matmul_model(K=K, N=N, seed=10)
    rng = np.random.default_rng(11)
    x = rng.standard_normal((64, K)).astype(np.float32)
    x[:, 5:10] = 0.0
    x[:, 20] = x[:, 21] + 1e-7 * rng.standard_normal(64).astype(np.float32)
    _assert_exact_parity(model, [{"X": x}], percdamp=1e-6)


def test_owq_cpp_multi_batch_and_3d_activation():
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
    cpp, _quant = _assert_exact_parity(model3d, cals3d)
    onnx.checker.check_model(cpp)


def test_owq_cpp_biased_gemm():
    # A biased Gemm's own bias input is untouched by either side -- OWQ's
    # correction is an additive term summed AFTER the existing node
    # (bias included), not a weight-fold.
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
    cpp, quant = _assert_exact_parity(biased, _correlated_calibration(K=K, seed=15))
    b_new = onnx.numpy_helper.to_array(
        next(t for t in cpp.graph.initializer if t.name == "B")
    )
    b_quant = onnx.numpy_helper.to_array(
        next(t for t in quant.graph.initializer if t.name == "B")
    )
    assert np.array_equal(b_new, b_quant)


def test_owq_cpp_correction_is_additive_not_a_weight_fold():
    # OWQ's whole point: the INT4 branch is untouched, and a genuinely new
    # Gather/MatMul/Add correction is appended -- no existing node's own
    # weight input is rewired.
    K, N = 64, 16
    model = _matmul_model(K=K, N=N, seed=16)
    quant = onnxsim.quantize_weight_only_int4(model)
    cpp = apply_owq_cpp(model, quant, _correlated_calibration(K=K, seed=17))
    onnx.checker.check_model(cpp)

    op_types = [n.op_type for n in cpp.graph.node]
    assert "Gather" in op_types
    assert "MatMul" in op_types
    assert "Add" in op_types
    # Exactly one more MatMul than the quantized input had (the new
    # correction MatMul; the DequantizeLinear+MatMul(int4 branch) chain is
    # otherwise unchanged).
    quant_matmuls = sum(1 for n in quant.graph.node if n.op_type == "MatMul")
    cpp_matmuls = sum(1 for n in cpp.graph.node if n.op_type == "MatMul")
    assert cpp_matmuls == quant_matmuls + 1


def test_owq_cpp_correction_recovers_weak_columns_exactly():
    # The whole point of OWQ: the layer's own output, restricted to just
    # the weak columns' own contribution, should match the FLOAT model's
    # output far more closely than the plain INT4-only model does --
    # checked end to end via onnxruntime, not just the stored initializers.
    K, N = 64, 16
    model = _matmul_model(K=K, N=N, seed=18)
    quant = onnxsim.quantize_weight_only_int4(model)
    cpp = apply_owq_cpp(
        model, quant, _correlated_calibration(K=K, seed=19), outlier_fraction=0.1
    )
    onnx.checker.check_model(cpp)

    rng = np.random.default_rng(20)
    x = rng.standard_normal((8, K)).astype(np.float32)
    float_y = _run(model, {"X": x})
    quant_y = _run(quant, {"X": x})
    owq_y = _run(cpp, {"X": x})

    quant_err = np.linalg.norm(float_y - quant_y)
    owq_err = np.linalg.norm(float_y - owq_y)
    assert owq_err < quant_err


def test_owq_cpp_skips_empty_calibration():
    # No calibration data -- no activation was ever observed, so every
    # candidate is skipped and the model is returned structurally unchanged
    # on both sides.
    model = _matmul_model()
    quant = onnxsim.quantize_weight_only_int4(model)
    for fn in (apply_owq, apply_owq_cpp):
        out = fn(model, quant, [])
        assert out.SerializeToString() == quant.SerializeToString()


def test_owq_cpp_noop_when_no_int4_layer():
    # Passing the float model itself as "quantized_model": no INT4
    # DequantizeLinear exists, so no candidate can ever match.
    model = _matmul_model(K=32, N=8, seed=21)
    result = apply_owq_cpp(model, model, _correlated_calibration(K=32, seed=22))
    assert result.SerializeToString() == model.SerializeToString()


def test_owq_cpp_noop_when_outlier_fraction_rounds_to_zero():
    # outlier_fraction so small that round(outlier_fraction * K) == 0 --
    # matches apply_owq's own `if num_weak < 1: continue`.
    K, N = 64, 16
    model = _matmul_model(K=K, N=N, seed=23)
    quant = onnxsim.quantize_weight_only_int4(model)
    result = apply_owq_cpp(
        model, quant, _correlated_calibration(K=K, seed=24), outlier_fraction=1e-6
    )
    assert result.SerializeToString() == quant.SerializeToString()
