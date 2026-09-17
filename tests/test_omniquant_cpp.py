"""Tests for ``onnxsim.apply_omniquant_cpp`` -- the C++-backed port of
OmniQuant's own grid-searched Learnable Weight Clipping (LWC) and Learnable
Equivalent Transformation (LET), see ``onnxsim/omniquant.py``.

Unlike ``test_adaround_cpp.py``/``test_tesseraq_cpp.py`` (both iterative Adam
optimizations, where cross-language floating-point agreement is only
measured, not assumed), OmniQuant's own search is a bounded, deterministic
GRID search with no RNG and no hand-rolled dense linear algebra -- so this
port is expected to, and empirically does, track the pure-Python reference
bit-for-bit (see ``onnxsim/omniquant_entry.h``'s own scope note). Because
``onnxsim.apply_omniquant`` now delegates unconditionally to this C++ port
(see ``onnxsim/omniquant.py``), the exact-parity checks below are a
wiring/reproducibility regression test as much as an independent numerical
cross-check -- the reference numpy implementation these tests compare
against was the pure-Python ``apply_omniquant`` prior to that delegation
(reconstructed here, inline, so the check stays independent of the C++ port
it verifies).
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.omniquant import _quantize_blockwise_int4_with_clip
from onnxsim.onnx_simplifier import apply_omniquant_cpp

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


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _matmul_model(K=64, N=16, weight=None, seed=0):
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
        [_f32(weight, "W")],
    )


def _outlier_calibration(K=64, num_samples=64, outlier_dims=(3, 7), seed=1):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((num_samples, K)).astype(np.float32) + 2.0
    for c in outlier_dims:
        x[:, c] *= 20.0
    return [{"X": x}]


def _reference_apply_omniquant(
    float_model,
    quantized_model,
    calibration_data,
    num_clip_steps=20,
    num_alpha_steps=20,
    min_clip_ratio=0.5,
):
    """Reimplements onnxsim.apply_omniquant's own former pure-Python
    algorithm directly here (rather than importing it, since
    onnxsim.apply_omniquant now delegates to the very C++ port under test)
    -- an independent numpy cross-check, not a reproduction of
    onnxsim/omniquant.py's own removed code.
    """
    import onnx.helper

    from onnxsim import backend
    from onnxsim.adaround import _find_int4_matmul_candidates, _node_outputs, _pack_int4
    from onnxsim.bias_correction import (
        _activation_rows,
        _add_probe_outputs,
        _all_names,
        _unique_name,
    )

    candidates = _find_int4_matmul_candidates(float_model, quantized_model)
    if not candidates:
        return quantized_model
    probe_names = sorted({c.float_node.input[0] for c in candidates})
    float_probe = _add_probe_outputs(float_model, probe_names)
    activations = {name: [] for name in probe_names}
    for batch in calibration_data:
        out = backend.run_model(float_probe, batch)
        for name in probe_names:
            activations[name].append(np.asarray(out[name], dtype=np.float64))

    clip_ratios = np.linspace(min_clip_ratio, 1.0, num_clip_steps)[::-1]
    alphas = np.linspace(0.0, 1.0, num_alpha_steps)

    rewrites = []
    for c in candidates:
        acts = _activation_rows(activations[c.float_node.input[0]])
        if not acts:
            continue
        x = np.concatenate(acts, axis=0)
        w = onnx.numpy_helper.to_array(c.w_float_init).astype(np.float64)
        dim0, dim1 = w.shape
        w_nk = w if c.weight_transposed else w.T
        if x.shape[1] != w_nk.shape[1]:
            continue
        y_float = x @ w_nk.T

        best_codes_nk, best_scale_blocks = _quantize_blockwise_int4_with_clip(
            w_nk, c.block_size, 1.0
        )
        w_hat0 = best_codes_nk * np.repeat(best_scale_blocks, c.block_size, axis=1)
        best_err = float(np.mean((y_float - x @ w_hat0.T) ** 2))
        best_clip_ratio = 1.0
        for clip_ratio in clip_ratios[1:]:
            codes_nk, scale_blocks = _quantize_blockwise_int4_with_clip(
                w_nk, c.block_size, clip_ratio
            )
            w_hat = codes_nk * np.repeat(scale_blocks, c.block_size, axis=1)
            err = float(np.mean((y_float - x @ w_hat.T) ** 2))
            if err < best_err:
                best_err = err
                best_clip_ratio = clip_ratio
                best_codes_nk, best_scale_blocks = codes_nk, scale_blocks

        shift = np.mean(x, axis=0)
        x_centered = x - shift[np.newaxis, :]
        weight_col_absmax = np.maximum(np.abs(w_nk).max(axis=0), 1e-12)
        act_col_absmax = np.maximum(np.abs(x_centered).mean(axis=0), 1e-12)
        best_channel_scale = None
        best_bias_correction = None
        for alpha in alphas[1:]:
            raw = act_col_absmax**alpha / weight_col_absmax ** (1.0 - alpha)
            channel_scale = raw / np.exp(np.mean(np.log(raw)))
            w_scaled_nk = w_nk * channel_scale[np.newaxis, :]
            codes_nk, scale_blocks = _quantize_blockwise_int4_with_clip(
                w_scaled_nk, c.block_size, best_clip_ratio
            )
            w_hat = codes_nk * np.repeat(scale_blocks, c.block_size, axis=1)
            x_transformed = x_centered / channel_scale[np.newaxis, :]
            bias_correction = w_nk @ shift
            y_hat = x_transformed @ w_hat.T + bias_correction[np.newaxis, :]
            err = float(np.mean((y_float - y_hat) ** 2))
            if err < best_err:
                best_err = err
                best_codes_nk, best_scale_blocks = codes_nk, scale_blocks
                best_channel_scale = channel_scale
                best_bias_correction = bias_correction

        codes_orig = best_codes_nk if c.weight_transposed else best_codes_nk.T
        scale_orig = best_scale_blocks if c.weight_transposed else best_scale_blocks.T
        assert codes_orig.shape == (dim0, dim1)
        rewrites.append(
            (
                c.output_name,
                c.float_node.input[0],
                c.wq_name,
                c.ws_init.name,
                codes_orig.astype(np.int8),
                scale_orig.astype(np.float32),
                best_channel_scale,
                shift if best_channel_scale is not None else None,
                best_bias_correction,
            )
        )

    if not rewrites:
        return quantized_model
    corrected = onnx.ModelProto()
    corrected.CopyFrom(quantized_model)
    codes_by_name = {r[2]: _pack_int4(r[4]) for r in rewrites}
    scale_by_name = {r[3]: r[5] for r in rewrites}
    for t in corrected.graph.initializer:
        if t.name in codes_by_name:
            t.raw_data = codes_by_name[t.name]
        if t.name in scale_by_name:
            t.CopyFrom(onnx.numpy_helper.from_array(scale_by_name[t.name], name=t.name))

    taken_names = _all_names(corrected.graph)
    q_by_output = _node_outputs(corrected.graph)
    for (
        output_name,
        act_input,
        _wq,
        _ws,
        _codes,
        _scale,
        channel_scale,
        shift,
        bias_correction,
    ) in rewrites:
        if channel_scale is None:
            continue
        qn = q_by_output[output_name]
        inv_scale = (1.0 / channel_scale).astype(np.float32)

        shift_name = _unique_name(f"{act_input}_omniquant_shift", taken_names)
        corrected.graph.initializer.append(
            onnx.numpy_helper.from_array(shift.astype(np.float32), name=shift_name)
        )
        centered_name = _unique_name(f"{act_input}_omniquant_centered", taken_names)
        sub_node = onnx.helper.make_node(
            "Sub",
            [act_input, shift_name],
            [centered_name],
            name=_unique_name(f"{act_input}_omniquant_sub", taken_names),
        )
        inv_scale_name = _unique_name(f"{act_input}_omniquant_inv_scale", taken_names)
        corrected.graph.initializer.append(
            onnx.numpy_helper.from_array(inv_scale, name=inv_scale_name)
        )
        scaled_name = _unique_name(f"{act_input}_omniquant_scaled", taken_names)
        mul_node = onnx.helper.make_node(
            "Mul",
            [centered_name, inv_scale_name],
            [scaled_name],
            name=_unique_name(f"{act_input}_omniquant_mul", taken_names),
        )
        node_idx = next(i for i, n in enumerate(corrected.graph.node) if n is qn)
        corrected.graph.node.insert(node_idx, sub_node)
        corrected.graph.node.insert(node_idx + 1, mul_node)
        qn.input[0] = scaled_name

        old_output = qn.output[0]
        base_name = _unique_name(f"{output_name}_omniquant_base", taken_names)
        qn.output[0] = base_name
        bias_name = _unique_name(f"{output_name}_omniquant_bias", taken_names)
        corrected.graph.initializer.append(
            onnx.numpy_helper.from_array(
                bias_correction.astype(np.float32), name=bias_name
            )
        )
        add_node = onnx.helper.make_node(
            "Add",
            [base_name, bias_name],
            [old_output],
            name=_unique_name(f"{output_name}_omniquant_bias_add", taken_names),
        )
        qn_idx = next(i for i, n in enumerate(corrected.graph.node) if n is qn)
        corrected.graph.node.insert(qn_idx + 1, add_node)

    return corrected


def _assert_exact_parity(float_model, quant_model, calibration_data, **kwargs):
    py = _reference_apply_omniquant(
        float_model, quant_model, calibration_data, **kwargs
    )
    cpp = apply_omniquant_cpp(float_model, quant_model, calibration_data, **kwargs)
    onnx.checker.check_model(cpp)
    py_inits = sorted(py.graph.initializer, key=lambda t: t.name)
    cpp_inits = sorted(cpp.graph.initializer, key=lambda t: t.name)
    assert [t.name for t in py_inits] == [t.name for t in cpp_inits]
    for a, b in zip(py_inits, cpp_inits):
        assert a.data_type == b.data_type, a.name
        assert list(a.dims) == list(b.dims), a.name
        assert a.raw_data == b.raw_data, a.name
    py_nodes = sorted(
        (n.op_type, tuple(n.input), tuple(n.output)) for n in py.graph.node
    )
    cpp_nodes = sorted(
        (n.op_type, tuple(n.input), tuple(n.output)) for n in cpp.graph.node
    )
    assert py_nodes == cpp_nodes
    return cpp


def test_omniquant_cpp_matches_python_exactly():
    model = _matmul_model()
    quant = onnxsim.quantize_weight_only_int4(model)
    _assert_exact_parity(model, quant, _outlier_calibration())


@pytest.mark.parametrize(
    "K,N,seed",
    [(128, 32, 3), (32, 8, 1), (96, 24, 2), (64, 16, 4)],
)
def test_omniquant_cpp_matches_python_across_shapes(K, N, seed):
    model = _matmul_model(K=K, N=N, seed=seed)
    quant = onnxsim.quantize_weight_only_int4(model)
    calib = _outlier_calibration(K=K, num_samples=48, seed=seed + 50)
    _assert_exact_parity(model, quant, calib)


def test_omniquant_cpp_gemm_transb():
    rng = np.random.default_rng(6)
    K, N = 96, 12
    weight = rng.standard_normal((N, K)).astype(np.float32) * 0.5
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm<transB = 1>(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )
    quant = onnxsim.quantize_weight_only_int4(model)
    calib = _outlier_calibration(K=K, num_samples=32, outlier_dims=(10, 50), seed=7)
    _assert_exact_parity(model, quant, calib)


def test_omniquant_cpp_gemm_transb_with_bias():
    rng = np.random.default_rng(12)
    K, N = 64, 12
    weight = rng.standard_normal((N, K)).astype(np.float32) * 0.5
    bias = rng.standard_normal(N).astype(np.float32) * 0.1
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm<transB = 1>(X, W, B)
        }}
        """,
        [_f32(weight, "W"), _f32(bias, "B")],
    )
    quant = onnxsim.quantize_weight_only_int4(model)
    calib = _outlier_calibration(K=K, num_samples=32, seed=13)
    _assert_exact_parity(model, quant, calib)


def test_omniquant_cpp_3d_activation_flattening():
    K, N = 64, 16
    rng = np.random.default_rng(0)
    model = _model(
        f"""
        g (float[batch,seq,{K}] X) => (float[batch,seq,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(rng.standard_normal((K, N)).astype(np.float32) * 0.5, "W")],
    )
    quant = onnxsim.quantize_weight_only_int4(model)
    rng2 = np.random.default_rng(401)
    x = rng2.standard_normal((8, 16, K)).astype(np.float32) + 2.0
    x[:, :, 3] *= 20.0
    x[:, :, 7] *= 20.0
    _assert_exact_parity(model, quant, [{"X": x}])


def test_omniquant_cpp_custom_grid_parameters():
    model = _matmul_model(K=64, N=16, seed=15)
    quant = onnxsim.quantize_weight_only_int4(model)
    calib = _outlier_calibration(K=64, seed=16)
    _assert_exact_parity(
        model, quant, calib, num_clip_steps=8, num_alpha_steps=6, min_clip_ratio=0.3
    )


def test_omniquant_cpp_missing_calibration_input_raises():
    model = _matmul_model()
    quant = onnxsim.quantize_weight_only_int4(model)
    with pytest.raises(Exception):
        apply_omniquant_cpp(
            model, quant, calibration_data=[{"NotX": np.zeros((1, 64), np.float32)}]
        )


def test_omniquant_cpp_noop_when_no_int4_matmul_present():
    model = _model(
        """
        g (float[batch,4] X) => (float[batch,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    quant = onnxsim.quantize_weight_only_int4(model)
    cpp = apply_omniquant_cpp(
        model, quant, calibration_data=[{"X": np.zeros((1, 4), dtype=np.float32)}]
    )
    assert cpp.SerializeToString() == quant.SerializeToString()


def test_omniquant_cpp_delegation_matches_public_api():
    # onnxsim.apply_omniquant now delegates unconditionally to this C++
    # port -- confirm both entry points give byte-identical results.
    model = _matmul_model(K=32, N=8, seed=20)
    quant = onnxsim.quantize_weight_only_int4(model)
    calib = _outlier_calibration(K=32, seed=21)
    via_public = onnxsim.apply_omniquant(model, quant, calibration_data=calib)
    via_cpp = apply_omniquant_cpp(model, quant, calibration_data=calib)
    assert via_public.SerializeToString() == via_cpp.SerializeToString()


def test_omniquant_cpp_output_stays_close_to_float():
    model = _matmul_model(K=64, N=16, seed=2)
    quant = onnxsim.quantize_weight_only_int4(model)
    calib = _outlier_calibration(K=64, num_samples=32, seed=3)
    cpp = apply_omniquant_cpp(model, quant, calibration_data=calib)
    onnx.checker.check_model(cpp)
    x = calib[0]["X"]
    (float_y,) = _run(model, {"X": x})
    (cpp_y,) = _run(cpp, {"X": x})
    assert np.all(np.isfinite(cpp_y))
    rel_err = np.linalg.norm(float_y - cpp_y) / max(np.linalg.norm(float_y), 1e-6)
    assert rel_err < 0.25
