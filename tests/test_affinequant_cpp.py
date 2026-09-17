"""Tests for ``onnxsim.apply_affinequant_cpp`` -- the C++-backed port of
AffineQuant's own OmniQuant-style grid-searched Learnable Weight Clipping
(LWC) plus block-diagonal Learnable Equivalent Transformation (LET), see
``onnxsim/affinequant.py``.

Two of this port's own three transform candidates (LWC-only, diagonal LET)
are exactly the same bounded, deterministic grid search
``test_omniquant_cpp.py`` already verifies bit-exact -- covered here the
same way. The THIRD candidate (block-affine LET) additionally uses a
hand-rolled cyclic Jacobi eigendecomposition (no LAPACK/BLAS linked into
this codebase) to fit each block's own rotation, which does NOT reproduce
``numpy.linalg.eigh``'s own eigenvector basis bit-for-bit on a block with
repeated/near-degenerate eigenvalues -- an ACCEPTED, PERMANENT DIVERGENCE
documented in ``onnxsim/affinequant_entry.h``. So this file splits its
checks: exact-byte-parity tests use an ``affine_block_size`` that does not
evenly divide ``K`` (skipping the block-affine candidate entirely, exactly
as ``onnxsim/affinequant.py``'s own docstring says a non-divisor does), and
a separate test explicitly exercises a configuration where the block-affine
candidate wins, checking structural properties (orthogonality, node
topology) and reconstruction-quality closeness rather than byte equality.

Because ``onnxsim.apply_affinequant`` now delegates unconditionally to this
C++ port (see ``onnxsim/affinequant.py``), the exact-parity checks are a
wiring/reproducibility regression test as much as an independent numerical
cross-check -- the reference implementation compared against is
reconstructed here, inline, from the technique's own former pure-Python
algorithm, independent of the C++ port under test.
"""

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.affinequant import _block_diagonal_rotation
from onnxsim.omniquant import _quantize_blockwise_int4_with_clip
from onnxsim.onnx_simplifier import apply_affinequant_cpp

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


def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-6)


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


def _correlated_calibration(K=32, num_samples=64, rank=4, seed=100):
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((num_samples, rank)).astype(np.float32)
    proj = rng.standard_normal((rank, K)).astype(np.float32)
    x = latent @ proj + rng.standard_normal((num_samples, K)).astype(np.float32) * 0.1
    return [{"X": x}]


def _reference_apply_affinequant(
    float_model,
    quantized_model,
    calibration_data,
    num_clip_steps=20,
    num_alpha_steps=20,
    min_clip_ratio=0.5,
    affine_block_size=8,
):
    """Reimplements onnxsim.apply_affinequant's own former pure-Python
    algorithm directly here (rather than importing it, since
    onnxsim.apply_affinequant now delegates to the very C++ port under
    test) -- an independent numpy cross-check.
    """
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
        k = w_nk.shape[1]
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
        bias_correction = w_nk @ shift

        best_channel_scale = None
        best_rotation = None

        weight_col_absmax = np.maximum(np.abs(w_nk).max(axis=0), 1e-12)
        act_col_absmax = np.maximum(np.abs(x_centered).mean(axis=0), 1e-12)
        for alpha in alphas[1:]:
            raw = act_col_absmax**alpha / weight_col_absmax ** (1.0 - alpha)
            channel_scale = raw / np.exp(np.mean(np.log(raw)))
            w_scaled_nk = w_nk * channel_scale[np.newaxis, :]
            codes_nk, scale_blocks = _quantize_blockwise_int4_with_clip(
                w_scaled_nk, c.block_size, best_clip_ratio
            )
            w_hat = codes_nk * np.repeat(scale_blocks, c.block_size, axis=1)
            x_transformed = x_centered / channel_scale[np.newaxis, :]
            y_hat = x_transformed @ w_hat.T + bias_correction[np.newaxis, :]
            err = float(np.mean((y_float - y_hat) ** 2))
            if err < best_err:
                best_err = err
                best_codes_nk, best_scale_blocks = codes_nk, scale_blocks
                best_channel_scale = channel_scale
                best_rotation = None

        if affine_block_size >= 1 and k % affine_block_size == 0:
            rotation = _block_diagonal_rotation(x_centered, affine_block_size)
            x_rot = x_centered @ rotation
            w_rot_nk = w_nk @ rotation
            weight_col_absmax_rot = np.maximum(np.abs(w_rot_nk).max(axis=0), 1e-12)
            act_col_absmax_rot = np.maximum(np.abs(x_rot).mean(axis=0), 1e-12)
            for alpha in alphas[1:]:
                raw = act_col_absmax_rot**alpha / weight_col_absmax_rot ** (1.0 - alpha)
                channel_scale = raw / np.exp(np.mean(np.log(raw)))
                w_hat_rot = w_rot_nk * channel_scale[np.newaxis, :]
                codes_nk, scale_blocks = _quantize_blockwise_int4_with_clip(
                    w_hat_rot, c.block_size, best_clip_ratio
                )
                w_hat = codes_nk * np.repeat(scale_blocks, c.block_size, axis=1)
                x_transformed = x_rot / channel_scale[np.newaxis, :]
                y_hat = x_transformed @ w_hat.T + bias_correction[np.newaxis, :]
                err = float(np.mean((y_float - y_hat) ** 2))
                if err < best_err:
                    best_err = err
                    best_codes_nk, best_scale_blocks = codes_nk, scale_blocks
                    best_channel_scale = channel_scale
                    best_rotation = rotation

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
                best_rotation,
                bias_correction if best_channel_scale is not None else None,
            )
        )

    if not rewrites:
        return quantized_model
    corrected = onnx.ModelProto()
    corrected.CopyFrom(quantized_model)
    codes_by_name = {r[2]: r[4] for r in rewrites}
    scale_by_name = {r[3]: r[5] for r in rewrites}
    for t in corrected.graph.initializer:
        if t.name in codes_by_name:
            t.raw_data = _pack_int4(codes_by_name[t.name])
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
        rotation,
        bias_correction,
    ) in rewrites:
        if channel_scale is None:
            continue
        qn = q_by_output[output_name]
        inv_scale = (1.0 / channel_scale).astype(np.float32)

        shift_name = _unique_name(f"{act_input}_affinequant_shift", taken_names)
        corrected.graph.initializer.append(
            onnx.numpy_helper.from_array(shift.astype(np.float32), name=shift_name)
        )
        centered_name = _unique_name(f"{act_input}_affinequant_centered", taken_names)
        sub_node = onnx.helper.make_node(
            "Sub",
            [act_input, shift_name],
            [centered_name],
            name=_unique_name(f"{act_input}_affinequant_sub", taken_names),
        )
        node_idx = next(i for i, n in enumerate(corrected.graph.node) if n is qn)
        corrected.graph.node.insert(node_idx, sub_node)
        node_idx += 1

        rotated_name = centered_name
        if rotation is not None:
            rotation_name = _unique_name(
                f"{act_input}_affinequant_rotation", taken_names
            )
            corrected.graph.initializer.append(
                onnx.numpy_helper.from_array(
                    rotation.astype(np.float32), name=rotation_name
                )
            )
            rotated_name = _unique_name(f"{act_input}_affinequant_rotated", taken_names)
            matmul_node = onnx.helper.make_node(
                "MatMul",
                [centered_name, rotation_name],
                [rotated_name],
                name=_unique_name(f"{act_input}_affinequant_matmul", taken_names),
            )
            corrected.graph.node.insert(node_idx, matmul_node)
            node_idx += 1

        inv_scale_name = _unique_name(f"{act_input}_affinequant_inv_scale", taken_names)
        corrected.graph.initializer.append(
            onnx.numpy_helper.from_array(inv_scale, name=inv_scale_name)
        )
        scaled_name = _unique_name(f"{act_input}_affinequant_scaled", taken_names)
        mul_node = onnx.helper.make_node(
            "Mul",
            [rotated_name, inv_scale_name],
            [scaled_name],
            name=_unique_name(f"{act_input}_affinequant_mul", taken_names),
        )
        corrected.graph.node.insert(node_idx, mul_node)
        qn.input[0] = scaled_name

        old_output = qn.output[0]
        base_name = _unique_name(f"{output_name}_affinequant_base", taken_names)
        qn.output[0] = base_name
        bias_name = _unique_name(f"{output_name}_affinequant_bias", taken_names)
        corrected.graph.initializer.append(
            onnx.numpy_helper.from_array(
                bias_correction.astype(np.float32), name=bias_name
            )
        )
        add_node = onnx.helper.make_node(
            "Add",
            [base_name, bias_name],
            [old_output],
            name=_unique_name(f"{output_name}_affinequant_bias_add", taken_names),
        )
        qn_idx = next(i for i, n in enumerate(corrected.graph.node) if n is qn)
        corrected.graph.node.insert(qn_idx + 1, add_node)

    return corrected


def _assert_exact_parity(float_model, quant_model, calibration_data, **kwargs):
    py = _reference_apply_affinequant(
        float_model, quant_model, calibration_data, **kwargs
    )
    cpp = apply_affinequant_cpp(float_model, quant_model, calibration_data, **kwargs)
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


# affine_block_size deliberately does not divide K in every test below
# (K=64 or 96, affine_block_size=9) so the block-affine candidate -- the
# one candidate with a documented numerical divergence -- is always
# skipped, keeping these an exact-parity check of the rest of the port
# (candidates 1 and 2, identical machinery to apply_omniquant_cpp's own).


def test_affinequant_cpp_matches_python_exactly_diagonal_only():
    model = _matmul_model(K=64, N=16, seed=0)
    quant = onnxsim.quantize_weight_only_int4(model)
    _assert_exact_parity(model, quant, _outlier_calibration(K=64), affine_block_size=9)


@pytest.mark.parametrize("K,N,seed", [(96, 24, 2), (64, 16, 4)])
def test_affinequant_cpp_matches_python_across_shapes_diagonal_only(K, N, seed):
    model = _matmul_model(K=K, N=N, seed=seed)
    quant = onnxsim.quantize_weight_only_int4(model)
    calib = _outlier_calibration(K=K, num_samples=48, seed=seed + 50)
    _assert_exact_parity(model, quant, calib, affine_block_size=9)


def test_affinequant_cpp_gemm_transb_diagonal_only():
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
    _assert_exact_parity(model, quant, calib, affine_block_size=9)


def test_affinequant_cpp_no_transform_candidate_matches_exactly():
    # Zero-mean, unstructured calibration: neither LET candidate should
    # improve on plain LWC, so the block-affine candidate's own divergent
    # rotation never even gets compared against -- exact parity holds
    # regardless of affine_block_size.
    model = _matmul_model(K=32, N=8, seed=4)
    quant = onnxsim.quantize_weight_only_int4(model)
    rng = np.random.default_rng(5)
    x = rng.standard_normal((16, 32)).astype(np.float32)
    _assert_exact_parity(model, quant, [{"X": x}], affine_block_size=8)


def test_affinequant_cpp_block_affine_candidate_close_when_it_wins():
    # A correlated-calibration scenario (matching this module's own
    # test_affinequant.py) where the block-affine candidate is known to
    # win -- verified structurally (rotation orthogonal, matching node
    # topology) and by reconstruction closeness, not byte equality, per
    # this file's own module docstring.
    K, N = 32, 8
    model = _matmul_model(K=K, N=N, seed=0)
    quant = onnxsim.quantize_weight_only_int4(model)
    calib = _correlated_calibration(K=K, num_samples=64, rank=4, seed=100)

    py = _reference_apply_affinequant(model, quant, calib, affine_block_size=8)
    cpp = apply_affinequant_cpp(model, quant, calib, affine_block_size=8)
    onnx.checker.check_model(cpp)

    py_ops = [n.op_type for n in py.graph.node]
    cpp_ops = [n.op_type for n in cpp.graph.node]
    assert py_ops == cpp_ops
    assert "MatMul" in cpp_ops, (
        "test fixture no longer exercises the block-affine (rotation) "
        "candidate -- update the seed/calibration"
    )

    rotation = next(
        onnx.numpy_helper.to_array(t)
        for t in cpp.graph.initializer
        if t.name.endswith("_affinequant_rotation")
    )
    identity = np.eye(K)
    np.testing.assert_allclose(rotation.T @ rotation, identity, atol=1e-4)

    x = calib[0]["X"]
    (float_y,) = _run(model, {"X": x})
    (py_y,) = _run(py, {"X": x})
    (cpp_y,) = _run(cpp, {"X": x})
    py_err = np.linalg.norm(float_y.astype(np.float64) - py_y.astype(np.float64))
    cpp_err = np.linalg.norm(float_y.astype(np.float64) - cpp_y.astype(np.float64))
    # Both are legitimate orthonormal bases of the same eigenspace -- close
    # reconstruction quality, not byte-identical codes.
    assert cpp_err < py_err * 1.5 + 1e-6


def test_affinequant_cpp_never_worse_than_plain_rtn():
    model = _matmul_model(K=32, N=8, seed=4)
    rng = np.random.default_rng(5)
    x = rng.standard_normal((16, 32)).astype(np.float32)
    quant = onnxsim.quantize_weight_only_int4(model)
    cpp = apply_affinequant_cpp(model, quant, [{"X": x}])

    def _dequantize_int4(m):
        dq = next(n for n in m.graph.node if n.op_type == "DequantizeLinear")
        wq = next(t for t in m.graph.initializer if t.name == dq.input[0])
        ws = next(t for t in m.graph.initializer if t.name == dq.input[1])
        block_size = next(a.i for a in dq.attribute if a.name == "block_size")
        axis = next((a.i for a in dq.attribute if a.name == "axis"), 1)
        codes = onnx.numpy_helper.to_array(wq).astype(np.float64)
        scale = onnx.numpy_helper.to_array(ws).astype(np.float64)
        scale_full = np.repeat(scale, block_size, axis=axis)
        slicer = [slice(None)] * codes.ndim
        slicer[axis] = slice(0, codes.shape[axis])
        return codes * scale_full[tuple(slicer)]

    w_float = onnx.numpy_helper.to_array(model.graph.initializer[0]).astype(np.float64)
    w_rtn = _dequantize_int4(quant)
    y_float = x.astype(np.float64) @ w_float
    rtn_err = np.linalg.norm(y_float - x.astype(np.float64) @ w_rtn)

    (cpp_y,) = _run(cpp, {"X": x})
    cpp_err = np.linalg.norm(y_float - cpp_y.astype(np.float64))
    assert cpp_err <= rtn_err + 1e-6


def test_affinequant_cpp_codes_stay_in_range():
    model = _matmul_model(K=32, N=8, seed=8)
    quant = onnxsim.quantize_weight_only_int4(model)
    calib = _outlier_calibration(K=32, num_samples=16, outlier_dims=(1,), seed=9)
    cpp = apply_affinequant_cpp(model, quant, calib)
    wq = next(t for t in cpp.graph.initializer if t.data_type == onnx.TensorProto.INT4)
    numel = int(np.prod(list(wq.dims)))
    raw = np.frombuffer(wq.raw_data, dtype=np.uint8)
    lo = (raw & 0x0F).astype(np.int8)
    hi = ((raw >> 4) & 0x0F).astype(np.int8)
    lo = np.where(lo >= 8, lo - 16, lo)
    hi = np.where(hi >= 8, hi - 16, hi)
    codes = np.empty(numel, dtype=np.int8)
    codes[0::2] = lo[: (numel + 1) // 2]
    codes[1::2] = hi[: numel // 2]
    assert np.all(codes >= -7) and np.all(codes <= 7)


def test_affinequant_cpp_missing_calibration_input_raises():
    model = _matmul_model()
    quant = onnxsim.quantize_weight_only_int4(model)
    with pytest.raises(Exception):
        apply_affinequant_cpp(
            model, quant, calibration_data=[{"NotX": np.zeros((1, 64), np.float32)}]
        )


def test_affinequant_cpp_noop_when_no_int4_matmul_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    cpp = apply_affinequant_cpp(
        model, model, calibration_data=[{"X": np.zeros((4, 4), dtype=np.float32)}]
    )
    assert cpp.SerializeToString() == model.SerializeToString()


def test_affinequant_cpp_delegation_matches_public_api():
    # onnxsim.apply_affinequant now delegates unconditionally to this C++
    # port -- confirm both entry points give byte-identical results (the
    # port is deterministic given the same inputs, whichever candidate
    # wins).
    model = _matmul_model(K=32, N=8, seed=20)
    quant = onnxsim.quantize_weight_only_int4(model)
    calib = _outlier_calibration(K=32, seed=21)
    via_public = onnxsim.apply_affinequant(model, quant, calibration_data=calib)
    via_cpp = apply_affinequant_cpp(model, quant, calibration_data=calib)
    assert via_public.SerializeToString() == via_cpp.SerializeToString()


def test_affinequant_cpp_output_stays_close_to_float():
    model = _matmul_model(K=64, N=16, seed=2)
    quant = onnxsim.quantize_weight_only_int4(model)
    calib = _outlier_calibration(K=64, num_samples=32, seed=3)
    cpp = apply_affinequant_cpp(model, quant, calibration_data=calib)
    onnx.checker.check_model(cpp)
    x = calib[0]["X"]
    (float_y,) = _run(model, {"X": x})
    (cpp_y,) = _run(cpp, {"X": x})
    assert np.all(np.isfinite(cpp_y))
    assert _rel_l2(float_y, cpp_y) < 0.25
