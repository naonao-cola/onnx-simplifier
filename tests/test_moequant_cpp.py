"""Tests for ``onnxsim.apply_moequant_cpp`` -- the C++-backed port of
``onnxsim.apply_moequant`` (MoEQuant, see ``onnxsim/moequant.py``). EBSS's
own weighted-without-replacement subsampling does NOT reproduce numpy's own
RNG stream (this port uses its own independent RNG -- see
``onnxsim/moequant_entry.h``'s own accepted numerical scope note), so
tests here check structural/grid properties and ``ebss=False`` parity
(which removes that source of divergence) rather than bit-exact agreement
with the pure-Python reference in the ``ebss=True`` case.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

from onnxsim.moequant import apply_moequant
from onnxsim.onnx_simplifier import apply_moequant_cpp

ort = pytest.importorskip("onnxruntime")


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
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))
    return model


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _f16(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float16), name)


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _moe_inits(model):
    return {t.name: onnx.numpy_helper.to_array(t) for t in model.graph.initializer}


def _moe_router_model(
    fc1_w,
    fc2_w,
    router_w,
    router_b=None,
    fc1_b=None,
    fc3_w=None,
    activation="relu",
    k=1,
    tokens=16,
    dtype="float",
):
    num_experts, inter, hidden = fc1_w.shape
    fc1_b_arg = "FC1B" if fc1_b is not None else ""
    fc3_w_arg = "FC3W" if fc3_w is not None else ""
    router_call = "Gemm(X, RW, RB)" if router_b is not None else "Gemm(X, RW)"
    model = _model(
        f"""
        g ({dtype}[{tokens},{hidden}] X) => ({dtype}[{tokens},{hidden}] Y)
        {{
          R = {router_call}
          Y = com.microsoft.MoE <k={k}, activation_type="{activation}"> (X, R, FC1W, {fc1_b_arg}, FC2W, "", {fc3_w_arg})
        }}
        """
    )
    _cast = _f32 if dtype == "float" else _f16
    inits = [_cast(fc1_w, "FC1W"), _cast(fc2_w, "FC2W"), _cast(router_w, "RW")]
    if router_b is not None:
        inits.append(_cast(router_b, "RB"))
    if fc1_b is not None:
        inits.append(_cast(fc1_b, "FC1B"))
    if fc3_w is not None:
        inits.append(_cast(fc3_w, "FC3W"))
    model.graph.initializer.extend(inits)
    return model


def _int4_block_scale(w_nk, block_size):
    n, k = w_nk.shape
    bs = block_size if k % block_size == 0 else k
    blocks = w_nk.reshape(n, k // bs, bs)
    amax = np.max(np.abs(blocks), axis=2)
    return np.where(amax == 0.0, 1.0, amax / 7.0), bs


def _assert_on_int4_grid(original_nk, reconstructed_nk, block_size=32, atol=1e-3):
    scale, bs = _int4_block_scale(original_nk, block_size)
    n, k = original_nk.shape
    blocks = reconstructed_nk.reshape(n, k // bs, bs)
    ratio = blocks / scale[..., None]
    rounded = np.round(ratio)
    np.testing.assert_allclose(ratio, rounded, atol=atol)
    assert np.all(np.abs(rounded) <= 7 + 1e-6)


def _fixed_calibration(hidden, tokens, num_batches=4, seed=5):
    rng = np.random.default_rng(seed)
    return [
        {"X": rng.standard_normal((tokens, hidden)).astype(np.float32)}
        for _ in range(num_batches)
    ]


def test_moequant_cpp_quantizes_every_routed_expert_to_int4_grid():
    E, hidden, inter, tokens, k = 3, 8, 6, 16, 2
    rng = np.random.default_rng(1)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float32)
    router_w = (rng.standard_normal((hidden, E)) * 0.2).astype(np.float32)
    model = _moe_router_model(fc1_w, fc2_w, router_w, k=k, tokens=tokens)
    onnx.checker.check_model(model)

    calibration_data = _fixed_calibration(hidden, tokens, num_batches=6, seed=11)
    quantized = apply_moequant_cpp(model, calibration_data=calibration_data)
    onnx.checker.check_model(quantized)
    inits = _moe_inits(quantized)

    for e in range(E):
        _assert_on_int4_grid(fc1_w[e], inits["FC1W"][e])
        _assert_on_int4_grid(fc2_w[e], inits["FC2W"][e])
    assert not np.allclose(inits["FC1W"], fc1_w)
    assert not np.allclose(inits["FC2W"], fc2_w)


def test_moequant_cpp_leaves_never_routed_expert_untouched():
    E, hidden, inter, tokens = 3, 6, 5, 12
    rng = np.random.default_rng(3)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float32)
    router_w = (rng.standard_normal((hidden, E)) * 0.1).astype(np.float32)
    router_b = np.zeros(E, dtype=np.float32)
    router_b[E - 1] = -1e6
    model = _moe_router_model(
        fc1_w, fc2_w, router_w, router_b=router_b, k=1, tokens=tokens
    )
    onnx.checker.check_model(model)

    calibration_data = _fixed_calibration(hidden, tokens, num_batches=4, seed=13)
    quantized = apply_moequant_cpp(model, calibration_data=calibration_data)
    inits = _moe_inits(quantized)

    np.testing.assert_array_equal(inits["FC1W"][E - 1], fc1_w[E - 1])
    np.testing.assert_array_equal(inits["FC2W"][E - 1], fc2_w[E - 1])
    for e in range(E - 1):
        _assert_on_int4_grid(fc1_w[e], inits["FC1W"][e])
        _assert_on_int4_grid(fc2_w[e], inits["FC2W"][e])


def test_moequant_cpp_matches_python_reference_with_ebss_disabled():
    # ebss=False removes this port's own RNG divergence (see this file's
    # own module docstring) -- the AGQ-weighted Hessian and GPTQ's own
    # column update are otherwise both deterministic closed-form
    # computations, so the two should land on the SAME INT4 grid codes.
    E, hidden, inter, tokens, k = 3, 6, 5, 20, 2
    rng = np.random.default_rng(51)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float32)
    router_w = (rng.standard_normal((hidden, E)) * 0.2).astype(np.float32)
    model = _moe_router_model(fc1_w, fc2_w, router_w, k=k, tokens=tokens)

    calibration_data = _fixed_calibration(hidden, tokens, num_batches=5, seed=53)
    py_q = apply_moequant(model, calibration_data=calibration_data, ebss=False)
    cpp_q = apply_moequant_cpp(model, calibration_data=calibration_data, ebss=False)

    py_inits = _moe_inits(py_q)
    cpp_inits = _moe_inits(cpp_q)
    np.testing.assert_allclose(py_inits["FC1W"], cpp_inits["FC1W"], atol=1e-4)
    np.testing.assert_allclose(py_inits["FC2W"], cpp_inits["FC2W"], atol=1e-4)


def test_moequant_cpp_ebss_and_no_ebss_both_quantize_a_dominant_expert():
    E, hidden, inter, tokens = 3, 6, 5, 24
    rng = np.random.default_rng(21)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float32)
    router_w = (rng.standard_normal((hidden, E)) * 0.05).astype(np.float32)
    router_b = np.array([8.0, 0.0, -8.0], dtype=np.float32)
    model = _moe_router_model(
        fc1_w, fc2_w, router_w, router_b=router_b, k=1, tokens=tokens
    )
    onnx.checker.check_model(model)

    calibration_data = _fixed_calibration(hidden, tokens, num_batches=8, seed=23)

    for ebss in (True, False):
        quantized = apply_moequant_cpp(
            model, calibration_data=calibration_data, ebss=ebss, seed=1
        )
        inits = _moe_inits(quantized)
        _assert_on_int4_grid(fc1_w[0], inits["FC1W"][0])


def test_moequant_cpp_declines_fc3():
    E, hidden, inter, tokens = 2, 4, 3, 6
    rng = np.random.default_rng(31)
    fc1_w = rng.standard_normal((E, inter, hidden)).astype(np.float32)
    fc2_w = rng.standard_normal((E, hidden, inter)).astype(np.float32)
    fc3_w = rng.standard_normal((E, inter, hidden)).astype(np.float32)
    router_w = rng.standard_normal((hidden, E)).astype(np.float32)
    model = _moe_router_model(fc1_w, fc2_w, router_w, fc3_w=fc3_w, tokens=tokens)

    quantized = apply_moequant_cpp(
        model, calibration_data=_fixed_calibration(hidden, tokens)
    )
    inits = _moe_inits(quantized)
    np.testing.assert_array_equal(inits["FC1W"], fc1_w)
    np.testing.assert_array_equal(inits["FC2W"], fc2_w)


def test_moequant_cpp_declines_float16_experts():
    E, hidden, inter, tokens = 2, 4, 3, 6
    rng = np.random.default_rng(33)
    fc1_w = rng.standard_normal((E, inter, hidden)).astype(np.float32)
    fc2_w = rng.standard_normal((E, hidden, inter)).astype(np.float32)
    router_w = rng.standard_normal((hidden, E)).astype(np.float32)
    model = _moe_router_model(fc1_w, fc2_w, router_w, tokens=tokens, dtype="float16")

    calib_rng = np.random.default_rng(34)
    calibration_data = [
        {"X": calib_rng.standard_normal((tokens, hidden)).astype(np.float16)}
        for _ in range(4)
    ]
    quantized = apply_moequant_cpp(model, calibration_data=calibration_data)
    inits = _moe_inits(quantized)
    np.testing.assert_array_equal(inits["FC1W"], fc1_w.astype(np.float16))
    np.testing.assert_array_equal(inits["FC2W"], fc2_w.astype(np.float16))


def test_moequant_cpp_empty_calibration_data_is_a_no_op():
    E, hidden, inter, tokens = 2, 4, 3, 6
    rng = np.random.default_rng(37)
    fc1_w = rng.standard_normal((E, inter, hidden)).astype(np.float32)
    fc2_w = rng.standard_normal((E, hidden, inter)).astype(np.float32)
    router_w = rng.standard_normal((hidden, E)).astype(np.float32)
    model = _moe_router_model(fc1_w, fc2_w, router_w, tokens=tokens)

    quantized = apply_moequant_cpp(model, calibration_data=[])
    inits = _moe_inits(quantized)
    np.testing.assert_array_equal(inits["FC1W"], fc1_w)
    np.testing.assert_array_equal(inits["FC2W"], fc2_w)


def test_moequant_cpp_quantized_model_still_executes_on_onnxruntime():
    E, hidden, inter, tokens, k = 4, 8, 6, 20, 2
    rng = np.random.default_rng(41)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float32)
    router_w = (rng.standard_normal((hidden, E)) * 0.2).astype(np.float32)
    model = _moe_router_model(fc1_w, fc2_w, router_w, k=k, tokens=tokens)
    onnx.checker.check_model(model)

    calibration_data = _fixed_calibration(hidden, tokens, num_batches=6, seed=43)
    quantized = apply_moequant_cpp(model, calibration_data=calibration_data)
    onnx.checker.check_model(quantized)

    feed_rng = np.random.default_rng(47)
    feeds = {"X": feed_rng.standard_normal((tokens, hidden)).astype(np.float32)}
    (out_float,) = _run(model, feeds)
    (out_quant,) = _run(quantized, feeds)
    assert out_quant.shape == out_float.shape
    assert np.all(np.isfinite(out_quant))
    rel_err = np.linalg.norm(out_quant - out_float) / max(
        np.linalg.norm(out_float), 1e-6
    )
    assert rel_err < 0.5


def test_moequant_cpp_missing_calibration_input_raises():
    E, hidden, inter, tokens = 2, 4, 3, 6
    rng = np.random.default_rng(59)
    fc1_w = rng.standard_normal((E, inter, hidden)).astype(np.float32)
    fc2_w = rng.standard_normal((E, hidden, inter)).astype(np.float32)
    router_w = rng.standard_normal((hidden, E)).astype(np.float32)
    model = _moe_router_model(fc1_w, fc2_w, router_w, tokens=tokens)

    with pytest.raises((RuntimeError, ValueError)):
        apply_moequant_cpp(
            model, calibration_data=[{"WrongName": np.zeros((tokens, hidden))}]
        )
