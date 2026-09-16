"""Tests for ``onnxsim.apply_billm_cpp`` -- the C++-backed port of
``onnxsim.quantize_weight_only_billm`` (BiLLM's Hessian-guided
salient-column residual binarization, see ``onnxsim/billm_entry.h``).
Like ``test_gptq_cpp.py``, this runs the float model over real
calibration data through a real ``onnxruntime``-backed executor -- never
a fake/mock executor -- and checks tight agreement against the
pure-Python reference: both sides join the same candidates, factor the
same Hessian, rank the same salient columns, and binarize the same way,
so any divergence beyond ordinary floating-point rounding is a bug, not
an accepted tolerance. (The dense inverse/Cholesky use scalar
double-precision kernels rather than LAPACK, and this port's own salient-
column ranking uses ``std::stable_sort`` where the reference uses
``np.argsort`` -- see ``billm_entry.h``'s own "Accepted numerical scope"
note; every test below uses random weight/activation data, where an exact
sensitivity tie -- the only way that difference could matter -- has
vanishing probability.)
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

from onnxsim.billm import quantize_weight_only_billm
from onnxsim.onnx_simplifier import apply_billm_cpp

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
    # well-conditioned but non-trivial (non-identity) Hessian, closer to
    # a real layer's own activation statistics than pure i.i.d. noise.
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((num_samples, rank)).astype(np.float32)
    projection = rng.standard_normal((rank, K)).astype(np.float32)
    x = latent @ projection
    x += rng.standard_normal((num_samples, K)).astype(np.float32) * 0.05
    return [{"X": x}]


def _codes_and_scales(model, weight_name="W"):
    prefix = f"{weight_name}_billm"
    inits = {t.name: t for t in model.graph.initializer}
    code1 = next(t for n, t in inits.items() if n.startswith(prefix + "_code1"))
    code2 = next(t for n, t in inits.items() if n.startswith(prefix + "_code2"))
    scale1 = next(t for n, t in inits.items() if n.startswith(prefix + "_scale1"))
    scale2 = next(t for n, t in inits.items() if n.startswith(prefix + "_scale2"))
    return (
        onnx.numpy_helper.to_array(code1),
        onnx.numpy_helper.to_array(code2),
        onnx.numpy_helper.to_array(scale1).ravel(),
        onnx.numpy_helper.to_array(scale2).ravel(),
    )


def _assert_close_parity(model, calibration_data, weight_name="W", **kwargs):
    py = quantize_weight_only_billm(model, calibration_data, **kwargs)
    cpp = apply_billm_cpp(model, calibration_data, **kwargs)
    onnx.checker.check_model(cpp)

    py_c1, py_c2, py_s1, py_s2 = _codes_and_scales(py, weight_name)
    cpp_c1, cpp_c2, cpp_s1, cpp_s2 = _codes_and_scales(cpp, weight_name)
    assert py_c1.shape == cpp_c1.shape
    assert py_c2.shape == cpp_c2.shape
    # No RNG anywhere in this technique (closed-form Hessian + a bounded,
    # deterministic search) -- codes are expected to agree exactly for
    # random test data (see this module's own docstring on the one
    # tie-break caveat, which random data essentially never triggers).
    np.testing.assert_array_equal(cpp_c1, py_c1)
    np.testing.assert_array_equal(cpp_c2, py_c2)
    np.testing.assert_allclose(cpp_s1, py_s1, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(cpp_s2, py_s2, rtol=1e-5, atol=1e-6)
    return cpp


def test_billm_cpp_matches_python_exactly():
    _assert_close_parity(_matmul_model(), _correlated_calibration())


def test_billm_cpp_matches_python_across_shapes_and_blocks():
    for K, N, seed, block_size in [
        (128, 32, 5, 64),
        (256, 64, 11, 128),
        (96, 24, 21, 48),
        (32, 8, 23, 16),
    ]:
        model = _matmul_model(K=K, N=N, seed=seed)
        cals = _correlated_calibration(K=K, seed=seed + 100)
        _assert_close_parity(model, cals, block_size=block_size)


def test_billm_cpp_matches_python_with_narrower_salient_search():
    # max_salient_search bounds the greedy search width -- exercise a
    # value smaller than block_size - 1 so the search itself, not just
    # the block/Hessian machinery, is under test.
    K, N = 64, 16
    model = _matmul_model(K=K, N=N, seed=30)
    cals = _correlated_calibration(K=K, seed=31)
    _assert_close_parity(model, cals, block_size=32, max_salient_search=5)


def test_billm_cpp_gemm_transb():
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
    _assert_close_parity(model, _correlated_calibration(K=K, seed=9))


def test_billm_cpp_biased_gemm():
    # A biased Gemm binarizes (bias input untouched) exactly on both
    # sides.
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
    cpp = _assert_close_parity(biased, _correlated_calibration(K=K, seed=15))
    b_new = onnx.numpy_helper.to_array(
        next(t for t in cpp.graph.initializer if t.name == "B")
    )
    assert np.array_equal(
        b_new, onnx.numpy_helper.to_array(biased.graph.initializer[1])
    )


def test_billm_cpp_skips_empty_calibration():
    # No calibration data -- no activation was ever observed, so every
    # candidate is skipped and the model is returned structurally
    # unchanged on both sides.
    model = _matmul_model()
    for fn in (quantize_weight_only_billm, apply_billm_cpp):
        out = fn(model, [])
        assert [n.op_type for n in out.graph.node] == ["MatMul"]
        assert len(out.graph.initializer) == len(model.graph.initializer)


def test_billm_cpp_noop_when_no_matmul_gemm_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = apply_billm_cpp(model, [])
    assert result.SerializeToString() == model.SerializeToString()


def test_billm_cpp_reconstructs_and_runs_via_onnxruntime():
    K, N = 64, 16
    model = _matmul_model(K=K, N=N, seed=16)
    x = _correlated_calibration(K=K, num_samples=64, seed=17)[0]["X"]
    cpp = apply_billm_cpp(model, [{"X": x}])
    onnx.checker.check_model(cpp)

    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    c_sess = ort.InferenceSession(
        cpp.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    ref = sess.run(["Y"], {"X": x})[0]
    got = c_sess.run(["Y"], {"X": x})[0]
    assert np.all(np.isfinite(got))
    # ~1 bit/element average is a drastic compression -- only a loose
    # sanity bound is meaningful here (the tight cross-check against the
    # pure-Python reference above is what actually pins down correctness).
    rel_err = np.linalg.norm(got.astype(np.float64) - ref.astype(np.float64)) / max(
        np.linalg.norm(ref.astype(np.float64)), 1e-6
    )
    assert rel_err < 1.0


def test_billm_cpp_salient_columns_get_a_finer_reconstruction():
    # The whole point of BiLLM's own two-level residual scheme: a salient
    # column (scale2 != 0, i.e. it got a real residual correction) should
    # reconstruct its own original weight column with less quantization
    # error, on average, than the flat single-level binarization every
    # non-salient column gets -- checked directly against the ORIGINAL
    # float weight, via this port's own C++ output (not the Python
    # reference, which the exact-parity tests above already pin down).
    rng = np.random.default_rng(40)
    K, N = 64, 16
    weight = (rng.standard_normal((K, N)) * 0.5).astype(np.float32)
    # Make a handful of columns genuinely more Hessian-sensitive by
    # giving their own activation channel a much larger magnitude --
    # BiLLM's own sensitivity statistic is w_i^2 / diag(Hc)_ii^2, and a
    # large-activation channel drives a SMALL Hessian-inverse-Cholesky
    # diagonal there, in turn driving sensitivity up.
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f32(weight, "W")],
    )
    x = rng.standard_normal((64, K)).astype(np.float32) * 0.1
    x[:, :4] *= 50.0  # first 4 channels are strong outlier-magnitude.

    cpp = apply_billm_cpp(model, [{"X": x}], block_size=K)
    onnx.checker.check_model(cpp)
    _, _, scale1, scale2 = _codes_and_scales(cpp)
    salient_cols = np.nonzero(scale2 != 0.0)[0]
    nonsalient_cols = np.nonzero(scale2 == 0.0)[0]
    assert salient_cols.size > 0
    assert nonsalient_cols.size > 0

    w_init = next(t for t in cpp.graph.initializer if t.name == "W")
    dq = next(n for n in cpp.graph.node if n.op_type == "Add")
    mul1 = next(n for n in cpp.graph.node if n.output[0] == dq.input[0])
    mul2 = next(n for n in cpp.graph.node if n.output[0] == dq.input[1])
    cast1 = next(n for n in cpp.graph.node if n.output[0] == mul1.input[0])
    cast2 = next(n for n in cpp.graph.node if n.output[0] == mul2.input[0])
    code1 = onnx.numpy_helper.to_array(
        next(t for t in cpp.graph.initializer if t.name == cast1.input[0])
    ).astype(np.float64)
    code2 = onnx.numpy_helper.to_array(
        next(t for t in cpp.graph.initializer if t.name == cast2.input[0])
    ).astype(np.float64)
    recon = code1 * scale1[:, np.newaxis] + code2 * scale2[:, np.newaxis]
    w = onnx.numpy_helper.to_array(w_init).astype(np.float64)
    # w/recon are [K, N] (weight_transposed=False keeps the original
    # [dim0, dim1] == [K, N] storage layout), and scale1/scale2/
    # salient_cols/nonsalient_cols are all indexed along K (BiLLM's own
    # scales are per-INPUT-channel) -- so the per-column error below must
    # average over N (axis=1), not K, to stay indexed the same way.
    err_per_col = np.mean((w - recon) ** 2, axis=1)

    assert err_per_col[salient_cols].mean() < err_per_col[nonsalient_cols].mean()
