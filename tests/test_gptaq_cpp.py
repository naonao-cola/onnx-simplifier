"""Tests for ``onnxsim.apply_gptaq_cpp`` -- the C++-backed port of
``onnxsim.apply_gptaq`` (GPTAQ's asymmetric-calibration correction layered
on top of GPTQ's own sequential, Hessian-compensated rounding, see
``onnxsim/gptaq.py``). Like ``test_gptq_cpp.py``, this runs BOTH models
over real calibration data through a real ``onnxruntime``-backed executor
-- never a fake/mock executor.

``onnxsim.apply_gptaq`` now delegates to this C++ port (see
``onnxsim/gptaq.py``), so calling ``apply_gptaq`` and ``apply_gptaq_cpp``
on the same inputs is necessarily the same call chain -- the
``_assert_exact_parity`` checks below are wiring/reproducibility
regression tests (confirms delegation is actually reached and is
deterministic), not an independent numerical cross-check. The genuinely
independent check is against ``onnxsim.apply_gptq``, which remains a
fully separate pure-Python implementation: GPTAQ's own closed-form
``Shift`` term is exactly zero when there is no upstream corruption, so
``apply_gptaq_cpp`` must then agree with ``apply_gptq`` bit-for-bit, and
must reconstruct the true float output more closely than plain
``apply_gptq`` once a real upstream corruption is introduced (see
``test_gptaq.py``'s own identical comparison against the pure-Python
``apply_gptaq``, mirrored here against the C++ port directly).
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.gptaq import apply_gptaq
from onnxsim.onnx_simplifier import apply_gptaq_cpp

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


def _upstream_corrupted_model(K0=32, N1=8, corruption=None, seed=0):
    # Same construction as test_gptaq.py's own _upstream_corrupted_model:
    # a plain, deterministic `Add` simulates whatever upstream
    # quantization already did to this layer's own input, so the gap
    # between the float and quantized model's own `Y1` is an exactly
    # controlled quantity.
    rng = np.random.default_rng(seed)
    w2 = rng.standard_normal((K0, N1)).astype(np.float32) * 0.5
    corruption = (
        np.zeros(K0, dtype=np.float32)
        if corruption is None
        else corruption.astype(np.float32)
    )
    model = _model(
        f"""
        g (float[batch,{K0}] X) => (float[batch,{N1}] Y2)
        {{
          Y1 = Add(X, Corruption)
          Y2 = MatMul(Y1, W2)
        }}
        """,
        [_f32(w2, "W2"), _f32(corruption, "Corruption")],
    )
    # quantize_weight_only_int4 needs Y1's own value_info to see it is a
    # quantizable MatMul input -- see test_gptaq.py's own identical
    # comment for why shape inference is required here.
    return onnx.shape_inference.infer_shapes(model)


def _correlated_calibration(K, num_samples=64, rank=6, seed=1):
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((num_samples, rank)).astype(np.float32)
    projection = rng.standard_normal((rank, K)).astype(np.float32)
    x = latent @ projection
    x += rng.standard_normal((num_samples, K)).astype(np.float32) * 0.05
    return x


def _int4_codes(model, node_output_name="Y"):
    node = next(
        n
        for n in model.graph.node
        if n.op_type in ("MatMul", "Gemm") and n.output[0] == node_output_name
    )
    dq = next(
        n
        for n in model.graph.node
        if n.op_type == "DequantizeLinear" and n.output[0] == node.input[1]
    )
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


def _assert_exact_parity(
    float_model, quant, calibration_data, node_output_name="Y", **kwargs
):
    py = apply_gptaq(float_model, quant, calibration_data, **kwargs)
    cpp = apply_gptaq_cpp(float_model, quant, calibration_data, **kwargs)
    assert np.array_equal(
        _int4_codes(py, node_output_name), _int4_codes(cpp, node_output_name)
    )
    py_inits = sorted(py.graph.initializer, key=lambda t: t.name)
    cpp_inits = sorted(cpp.graph.initializer, key=lambda t: t.name)
    assert [t.name for t in py_inits] == [t.name for t in cpp_inits]
    for a, b in zip(py_inits, cpp_inits):
        assert a.data_type == b.data_type, a.name
        assert list(a.dims) == list(b.dims), a.name
        assert a.raw_data == b.raw_data, a.name
    return cpp


def test_gptaq_cpp_matches_python_exactly_no_upstream_corruption():
    # No upstream corruption at all (delta_x == 0 everywhere): GPTAQ
    # reduces to plain GPTQ on both sides, and the two ports must still
    # agree bit-for-bit with each other.
    model = _matmul_model(K=64, N=16, seed=0)
    x = _correlated_calibration(K=64, num_samples=64, seed=1)
    quant = onnxsim.quantize_weight_only_int4(model)
    _assert_exact_parity(model, quant, [{"X": x}])


def test_gptaq_cpp_matches_python_exactly_with_upstream_corruption():
    K0, N1 = 32, 8
    float_model = _upstream_corrupted_model(K0=K0, N1=N1, corruption=None, seed=0)
    corruption = np.random.default_rng(42).standard_normal(K0) * 0.6
    corrupted_model = _upstream_corrupted_model(
        K0=K0, N1=N1, corruption=corruption, seed=0
    )
    x = np.random.default_rng(2).standard_normal((96, K0)).astype(np.float32)
    quant = onnxsim.quantize_weight_only_int4(corrupted_model)
    _assert_exact_parity(float_model, quant, [{"X": x}], node_output_name="Y2")


def test_gptaq_cpp_matches_python_across_shapes_and_blocks():
    for K, N, seed, procb in [
        (128, 32, 5, 64),
        (96, 24, 21, 48),
        (32, 8, 23, 16),
    ]:
        model = _matmul_model(K=K, N=N, seed=seed)
        x = _correlated_calibration(K=K, seed=seed + 100)
        quant = onnxsim.quantize_weight_only_int4(model)
        _assert_exact_parity(model, quant, [{"X": x}], proc_block_size=procb)


def test_gptaq_cpp_gemm_transb():
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
    x = _correlated_calibration(K=K, seed=9)
    quant = onnxsim.quantize_weight_only_int4(model)
    _assert_exact_parity(model, quant, [{"X": x}])


def test_gptaq_cpp_multi_batch_and_3d_activation():
    K, N = 32, 8
    model = _matmul_model(K=K, N=N, seed=12)
    rng = np.random.default_rng(13)
    quant = onnxsim.quantize_weight_only_int4(model)
    cals = [
        {"X": rng.standard_normal((8, K)).astype(np.float32)},
        {"X": rng.standard_normal((12, K)).astype(np.float32)},
    ]
    _assert_exact_parity(model, quant, cals)

    model3d = _model(
        f"""
        g (float[batch,seq,{K}] X) => (float[batch,seq,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f32(rng.standard_normal((K, N)).astype(np.float32), "W")],
    )
    quant3d = onnxsim.quantize_weight_only_int4(model3d)
    cals3d = [{"X": rng.standard_normal((2, 5, K)).astype(np.float32)}]
    cpp = _assert_exact_parity(model3d, quant3d, cals3d)
    onnx.checker.check_model(cpp)


def test_gptaq_cpp_missing_calibration_input_raises():
    model = _matmul_model(K=64, N=16, seed=20)
    quant = onnxsim.quantize_weight_only_int4(model)
    # Sanity: this model actually has a quantized candidate, so the probe
    # really runs and the missing-input check is exercised (not a no-op
    # early return from an empty candidate list).
    assert any(n.op_type == "DequantizeLinear" for n in quant.graph.node)
    with pytest.raises(Exception):
        apply_gptaq_cpp(model, quant, [{"NotX": np.zeros((1, 64), dtype=np.float32)}])


def test_gptaq_cpp_noop_when_no_int4_matmul_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = apply_gptaq_cpp(
        model, model, calibration_data=[{"X": np.zeros((4, 4), dtype=np.float32)}]
    )
    assert result.SerializeToString() == model.SerializeToString()


def test_gptaq_cpp_via_top_level_export():
    # onnxsim.apply_gptaq_cpp is exported from the package root too.
    model = _matmul_model(K=32, N=8, seed=30)
    x = _correlated_calibration(K=32, num_samples=16, rank=3, seed=31)
    quant = onnxsim.quantize_weight_only_int4(model)
    out = onnxsim.apply_gptaq_cpp(model, quant, calibration_data=[{"X": x}])
    onnx.checker.check_model(out)


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-6)


def test_gptaq_cpp_matches_gptq_exactly_with_no_upstream_quantization():
    # Genuinely independent check: onnxsim.apply_gptq is NOT delegated to
    # any C++ port and remains a fully separate pure-Python
    # implementation, so this is a real cross-implementation agreement
    # check, not the tautological one _assert_exact_parity performs post-
    # delegation. With no upstream corruption, GPTAQ's own closed-form
    # Shift is exactly zero (see gptaq_entry.h's own docstring), so
    # apply_gptaq_cpp must reduce to apply_gptq bit-for-bit.
    model = _matmul_model(K=64, N=16, seed=40)
    x = _correlated_calibration(K=64, num_samples=64, seed=41)
    quant = onnxsim.quantize_weight_only_int4(model)
    gptq_out = onnxsim.apply_gptq(model, quant, calibration_data=[{"X": x}])
    gptaq_out = apply_gptaq_cpp(model, quant, calibration_data=[{"X": x}])
    assert np.array_equal(_int4_codes(gptq_out), _int4_codes(gptaq_out))


def test_gptaq_cpp_beats_gptq_once_an_upstream_layer_is_already_corrected():
    # Same aggregated-trials construction as test_gptaq.py's own identical
    # test (see that file's own comment for why single-seed comparisons
    # are cross-platform flaky), run against the C++ port directly instead
    # of the now-delegating pure-Python apply_gptaq.
    K0, N1 = 32, 8
    float_model = _upstream_corrupted_model(K0=K0, N1=N1, corruption=None, seed=0)

    total_gptq = 0.0
    total_gptaq = 0.0
    trials = [(42, 1), (7, 2), (13, 3), (99, 4), (5, 5), (123, 6), (2024, 7), (77, 8)]
    for corruption_seed, calibration_seed in trials:
        corruption = np.random.default_rng(corruption_seed).standard_normal(K0) * 0.6
        corrupted_model = _upstream_corrupted_model(
            K0=K0, N1=N1, corruption=corruption, seed=0
        )
        x = np.random.default_rng(calibration_seed).standard_normal((96, K0))
        x = x.astype(np.float32)
        calibration_data = [{"X": x}]

        quant = onnxsim.quantize_weight_only_int4(corrupted_model)
        final_gptq = onnxsim.apply_gptq(
            float_model, quant, calibration_data=calibration_data
        )
        final_gptaq = apply_gptaq_cpp(
            float_model, quant, calibration_data=calibration_data
        )
        onnx.checker.check_model(final_gptq)
        onnx.checker.check_model(final_gptaq)

        (float_y,) = _run(float_model, {"X": x})
        (gptq_y,) = _run(final_gptq, {"X": x})
        (gptaq_y,) = _run(final_gptaq, {"X": x})
        assert np.all(np.isfinite(gptaq_y))
        total_gptq += _rel_l2(float_y, gptq_y)
        total_gptaq += _rel_l2(float_y, gptaq_y)

    assert total_gptaq < total_gptq * 0.85
