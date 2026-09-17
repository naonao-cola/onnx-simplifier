"""Tests for ``onnxsim.apply_brecq_cpp`` -- the C++-backed port of BRECQ's
own joint, Fisher-diagonal-weighted block reconstruction Adam loop, see
``onnxsim/brecq.py``.

Same accepted-numerical-scope class as ``test_adaround_cpp.py``: this is an
iterative Adam optimization (extended here to jointly optimize a whole
BLOCK of layers against the block's own final output, not one layer's own
output independently), not a closed-form computation, so floating-point
summation-order differences between this port's own scalar dense-matmul
kernels and numpy's own can in principle compound across iterations.
Measured empirically here, exactly like ``test_adaround_cpp.py``'s own
``_assert_agrees`` pattern, rather than assumed: every configuration below
happens to come back bit-for-bit identical to the pure-Python reference in
practice (BRECQ's own joint backward pass has no extra source of
nondeterminism beyond AdaRound's -- no RNG, no rounding-tie-sensitive
factorization), but a small mismatch tolerance is still used (never
asserted away as impossible), matching this repository's established
convention for every other Adam-loop port.

Because ``onnxsim.apply_brecq`` now delegates unconditionally to this C++
port (see ``onnxsim/brecq.py``), the reference implementation compared
against below is reconstructed here, inline, from the technique's own
former pure-Python algorithm, independent of the C++ port under test.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.onnx_simplifier import apply_brecq_cpp

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


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _residual_block_model(D=32, seed=0):
    rng = np.random.default_rng(seed)
    w1 = (rng.standard_normal((D, D)) * 0.3).astype(np.float32)
    w2 = (rng.standard_normal((D, D)) * 0.3).astype(np.float32)
    return _model(
        f"""
        g (float[batch,{D}] X) => (float[batch,{D}] Yout)
        {{
          Y1 = MatMul(X, W1)
          Y2 = MatMul(Y1, W2)
          Yout = Add(Y2, X)
        }}
        """,
        [_f32(w1, "W1"), _f32(w2, "W2")],
    )


def _quantize_chain_int4(model, weight_names):
    # Same splice-in-isolated-quantized-layers helper as test_brecq.py's
    # own (see that file's comment for why: quantize_weight_only_int4 only
    # ever quantizes a chain's first layer).
    orig_inits = {t.name: t for t in model.graph.initializer}
    quant = onnx.ModelProto()
    quant.CopyFrom(model)

    new_nodes = []
    new_initializers = [
        t for t in quant.graph.initializer if t.name not in weight_names
    ]
    for idx, node in enumerate(quant.graph.node):
        if node.op_type not in ("MatMul", "Gemm") or node.input[1] not in weight_names:
            new_nodes.append(node)
            continue
        w_name = node.input[1]
        w_init = orig_inits[w_name]
        k, n = w_init.dims[0], w_init.dims[1]
        iso = _model(
            f"""
            h (float[batch,{k}] Ain) => (float[batch,{n}] Aout)
            {{
              Aout = MatMul(Ain, {w_name})
            }}
            """,
            [w_init],
        )
        iso_quant = onnxsim.quantize_weight_only_int4(iso)
        dq_node = next(
            n for n in iso_quant.graph.node if n.op_type == "DequantizeLinear"
        )
        wq_init = next(
            t for t in iso_quant.graph.initializer if t.name == dq_node.input[0]
        )
        ws_init = next(
            t for t in iso_quant.graph.initializer if t.name == dq_node.input[1]
        )

        suffix = f"_{idx}"
        wq_renamed = onnx.TensorProto()
        wq_renamed.CopyFrom(wq_init)
        wq_renamed.name += suffix
        ws_renamed = onnx.TensorProto()
        ws_renamed.CopyFrom(ws_init)
        ws_renamed.name += suffix
        dq_out_name = f"{w_name}_dq{suffix}"

        new_dq = onnx.NodeProto()
        new_dq.CopyFrom(dq_node)
        new_dq.input[0] = wq_renamed.name
        new_dq.input[1] = ws_renamed.name
        new_dq.output[0] = dq_out_name
        new_initializers.extend([wq_renamed, ws_renamed])
        new_nodes.append(new_dq)

        new_node = onnx.NodeProto()
        new_node.CopyFrom(node)
        new_node.input[1] = dq_out_name
        new_nodes.append(new_node)

    del quant.graph.node[:]
    quant.graph.node.extend(new_nodes)
    del quant.graph.initializer[:]
    quant.graph.initializer.extend(new_initializers)
    return quant


def _correlated_calibration(D=32, num_samples=64, rank=6, seed=1):
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((num_samples, rank)).astype(np.float32)
    projection = rng.standard_normal((rank, D)).astype(np.float32)
    x = latent @ projection
    x += rng.standard_normal((num_samples, D)).astype(np.float32) * 0.05
    return x


# --- Reference reimplementation ---------------------------------------------
#
# Reconstructs onnxsim.brecq's own former pure-Python algorithm directly
# here (rather than importing it, since onnxsim.apply_brecq now delegates
# to the very C++ port under test) -- an independent numpy cross-check.

_GAMMA = -0.1
_ZETA = 1.1
_N_MIN = -7.0
_N_MAX = 7.0


def _h_and_dhdv(v):
    sig = 1.0 / (1.0 + np.exp(-v))
    raw = sig * (_ZETA - _GAMMA) + _GAMMA
    active = (raw > 0.0) & (raw < 1.0)
    h = np.clip(raw, 0.0, 1.0)
    dh_dv = np.where(active, sig * (1.0 - sig) * (_ZETA - _GAMMA), 0.0)
    return h, dh_dv


def _pack_int4(codes):
    flat = np.asarray(codes).reshape(-1).astype(np.int64)
    if flat.size % 2 != 0:
        raise ValueError("odd element count")
    lo = (flat[0::2] & 0xF).astype(np.uint8)
    hi = (flat[1::2] & 0xF).astype(np.uint8)
    return ((hi << 4) | lo).astype(np.uint8).tobytes()


def _reference_layer_arrays(float_model, quantized_model, candidate):
    from onnxsim.adaround import _find_int4_matmul_candidates

    (c,) = [
        cand
        for cand in _find_int4_matmul_candidates(float_model, quantized_model)
        if cand.output_name == candidate
    ]
    w = onnx.numpy_helper.to_array(c.w_float_init).astype(np.float64)
    scale = onnx.numpy_helper.to_array(c.ws_init).astype(np.float64)
    dim0, dim1 = w.shape
    if c.weight_transposed:
        w_nk = w
        scale_blocks = scale
    else:
        w_nk = w.T
        scale_blocks = scale.T
    scale_nk = np.repeat(scale_blocks, c.block_size, axis=1)[:, : w_nk.shape[1]]
    return w_nk, scale_nk, (dim0, dim1), c.weight_transposed, c


def _reference_optimize_block_rounding(
    chain_candidates,
    has_residual,
    x0,
    final_float,
    num_iterations,
    learning_rate,
    reg_param,
    warm_start,
    beta_range,
    fisher_eps,
):
    layers = chain_candidates
    w_nks = [w for w, _, _, _ in layers]
    scale_nks = [s for _, s, _, _ in layers]
    floor_bases = [np.floor(w / s) for w, s in zip(w_nks, scale_nks)]

    v_list = []
    for w, s, floor_base in zip(w_nks, scale_nks, floor_bases):
        frac = np.clip(w / s - floor_base, 1e-4, 1.0 - 1e-4)
        sig0 = np.clip((frac - _GAMMA) / (_ZETA - _GAMMA), 1e-4, 1.0 - 1e-4)
        v_list.append(np.log(sig0 / (1.0 - sig0)))

    var = final_float.var(axis=0)
    mean_var = var.mean()
    fisher = (
        np.ones_like(var)
        if mean_var <= fisher_eps
        else (var + fisher_eps) / (mean_var + fisher_eps)
    )

    m_list = [np.zeros_like(v) for v in v_list]
    v2_list = [np.zeros_like(v) for v in v_list]
    adam_beta1, adam_beta2, adam_eps = 0.9, 0.999, 1e-8

    warm_start_iters = int(num_iterations * warm_start)
    beta_start, beta_end = beta_range
    n_elems = x0.shape[0] * final_float.shape[1]
    num_layers = len(layers)

    for t in range(num_iterations):
        ys = [x0]
        h_list, dh_dv_list, w_hat_list, active_list = [], [], [], []
        for w_nk, scale_nk, floor_base, v in zip(w_nks, scale_nks, floor_bases, v_list):
            h, dh_dv = _h_and_dhdv(v)
            raw = floor_base + h
            w_hat = np.clip(raw, _N_MIN, _N_MAX) * scale_nk
            active = (raw > _N_MIN) & (raw < _N_MAX)
            h_list.append(h)
            dh_dv_list.append(dh_dv)
            w_hat_list.append(w_hat)
            active_list.append(active)
            ys.append(ys[-1] @ w_hat.T)

        final_hat = ys[-1] + x0 if has_residual else ys[-1]
        diff = final_hat - final_float
        grad_y = 2.0 * fisher[None, :] * diff / n_elems

        grads_v_reversed = []
        for layer_idx in range(num_layers - 1, -1, -1):
            dl_dw_hat = grad_y.T @ ys[layer_idx]
            dl_dh = dl_dw_hat * np.where(
                active_list[layer_idx], scale_nks[layer_idx], 0.0
            )
            grad_v = dl_dh * dh_dv_list[layer_idx]

            if t >= warm_start_iters:
                progress = (t - warm_start_iters) / max(
                    1, num_iterations - warm_start_iters - 1
                )
                beta = beta_start + (beta_end - beta_start) * progress
                u = 2.0 * h_list[layer_idx] - 1.0
                abs_u = np.abs(u)
                dreg_dh = (
                    -2.0 * reg_param * beta * np.sign(u) * np.power(abs_u, beta - 1.0)
                )
                grad_v = grad_v + dreg_dh * dh_dv_list[layer_idx]

            grads_v_reversed.append(grad_v)
            if layer_idx > 0:
                grad_y = grad_y @ w_hat_list[layer_idx]
        grads_v = list(reversed(grads_v_reversed))

        for layer_idx in range(num_layers):
            m_list[layer_idx] = (
                adam_beta1 * m_list[layer_idx] + (1.0 - adam_beta1) * grads_v[layer_idx]
            )
            v2_list[layer_idx] = adam_beta2 * v2_list[layer_idx] + (
                1.0 - adam_beta2
            ) * (grads_v[layer_idx] * grads_v[layer_idx])
            m_hat = m_list[layer_idx] / (1.0 - adam_beta1 ** (t + 1))
            v_hat = v2_list[layer_idx] / (1.0 - adam_beta2 ** (t + 1))
            v_list[layer_idx] = v_list[layer_idx] - learning_rate * m_hat / (
                np.sqrt(v_hat) + adam_eps
            )

    codes = []
    for floor_base, v in zip(floor_bases, v_list):
        h_final, _ = _h_and_dhdv(v)
        codes.append(np.clip(floor_base + np.round(h_final), _N_MIN, _N_MAX))
    return codes


def _reference_apply_brecq(
    float_model,
    quantized_model,
    blocks,
    calibration_data,
    num_iterations=300,
    learning_rate=0.1,
    reg_param=0.01,
    warm_start=0.2,
    beta_range=(20.0, 2.0),
    fisher_eps=1e-3,
):
    from onnxsim import backend
    from onnxsim.adaround import _find_int4_matmul_candidates, _node_outputs
    from onnxsim.bias_correction import _activation_rows, _add_probe_outputs

    candidates = _find_int4_matmul_candidates(float_model, quantized_model)
    by_input = {c.float_node.input[0]: c for c in candidates}

    def discover(block_input_name, block_output_name):
        chain = []
        seen = set()
        cur = block_input_name
        while cur != block_output_name:
            c = by_input.get(cur)
            if c is None or c.output_name in seen:
                break
            seen.add(c.output_name)
            chain.append(c)
            cur = c.output_name
        if cur == block_output_name:
            return (chain, False) if chain else None
        if not chain:
            return None
        f_by_output = _node_outputs(float_model.graph)
        add_node = f_by_output.get(block_output_name)
        if (
            add_node is not None
            and add_node.op_type == "Add"
            and len(add_node.input) == 2
            and set(add_node.input) == {cur, block_input_name}
        ):
            return chain, True
        return None

    discovered = []
    for block_input_name, block_output_name in blocks:
        found = discover(block_input_name, block_output_name)
        if found is not None:
            discovered.append((block_input_name, block_output_name, *found))
    if not discovered:
        return quantized_model

    probe_names = sorted(
        {b for b, _, _, _ in discovered} | {b for _, b, _, _ in discovered}
    )
    float_probe = _add_probe_outputs(float_model, probe_names)
    activations = {name: [] for name in probe_names}
    for batch in calibration_data:
        out = backend.run_model(float_probe, batch)
        for name in probe_names:
            activations[name].append(np.asarray(out[name], dtype=np.float64))

    optimized = {}
    for block_input_name, block_output_name, chain, has_residual in discovered:
        x0_batches, final_batches = [], []
        for xa, fa in zip(
            activations[block_input_name], activations[block_output_name]
        ):
            xr = _activation_rows([xa])
            fr = _activation_rows([fa])
            if xr and fr and xr[0].shape[0] == fr[0].shape[0]:
                x0_batches.append(xr[0])
                final_batches.append(fr[0])
        if not x0_batches:
            continue
        x0 = np.concatenate(x0_batches, axis=0)
        final_float = np.concatenate(final_batches, axis=0)

        layer_arrays = [
            _reference_layer_arrays(float_model, quantized_model, c.output_name)
            for c in chain
        ]
        if x0.shape[1] != layer_arrays[0][0].shape[1]:
            continue

        layer_codes = _reference_optimize_block_rounding(
            [(w, s, dims, t) for w, s, dims, t, _ in layer_arrays],
            has_residual,
            x0,
            final_float,
            num_iterations,
            learning_rate,
            reg_param,
            warm_start,
            beta_range,
            fisher_eps,
        )
        for (w_nk, scale_nk, (dim0, dim1), weight_transposed, c), codes_nk in zip(
            layer_arrays, layer_codes
        ):
            codes_orig = codes_nk if weight_transposed else codes_nk.T
            assert codes_orig.shape == (dim0, dim1)
            optimized[c.wq_name] = codes_orig.astype(np.int8)

    if not optimized:
        return quantized_model
    corrected = onnx.ModelProto()
    corrected.CopyFrom(quantized_model)
    for t in corrected.graph.initializer:
        if t.name in optimized:
            t.raw_data = _pack_int4(optimized[t.name])
    return corrected


def _assert_agrees(
    model, quant, blocks, calibration_data, max_mismatch_frac=0.0, **kwargs
):
    py = _reference_apply_brecq(model, quant, blocks, calibration_data, **kwargs)
    cpp = apply_brecq_cpp(model, quant, blocks, calibration_data, **kwargs)
    onnx.checker.check_model(cpp)

    py_inits = {t.name: t for t in py.graph.initializer}
    cpp_inits = {t.name: t for t in cpp.graph.initializer}
    assert set(py_inits) == set(cpp_inits)

    mismatches = 0
    total = 0
    for name, pt in py_inits.items():
        ct = cpp_inits[name]
        assert pt.data_type == ct.data_type, name
        assert list(pt.dims) == list(ct.dims), name
        if pt.data_type != onnx.TensorProto.INT4:
            assert pt.raw_data == ct.raw_data, name
            continue
        py_codes = np.frombuffer(pt.raw_data, dtype=np.uint8)
        cpp_codes = np.frombuffer(ct.raw_data, dtype=np.uint8)
        mismatches += int(np.sum(py_codes != cpp_codes))
        total += py_codes.size
    assert mismatches <= max_mismatch_frac * max(total, 1), (
        f"{mismatches}/{total} packed-byte mismatches exceeds tolerance"
    )
    return cpp


def test_brecq_cpp_matches_python_exactly():
    model = _residual_block_model(D=32, seed=0)
    quant = _quantize_chain_int4(model, {"W1", "W2"})
    x = _correlated_calibration(D=32, num_samples=64, rank=6, seed=1)
    _assert_agrees(model, quant, [("X", "Yout")], [{"X": x}])


@pytest.mark.parametrize(
    "D,seed,iters",
    [(32, 5, 800), (64, 2, 50), (32, 9, 1)],
)
def test_brecq_cpp_matches_python_across_configs(D, seed, iters):
    model = _residual_block_model(D=D, seed=seed)
    quant = _quantize_chain_int4(model, {"W1", "W2"})
    x = _correlated_calibration(D=D, num_samples=64, rank=6, seed=seed + 50)
    # A small, measured tolerance -- see this module's own docstring.
    _assert_agrees(
        model,
        quant,
        [("X", "Yout")],
        [{"X": x}],
        max_mismatch_frac=0.01,
        num_iterations=iters,
    )


def test_brecq_cpp_single_layer_chain_without_residual():
    rng = np.random.default_rng(9)
    D = 32
    w = (rng.standard_normal((D, D)) * 0.3).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{D}] X) => (float[batch,{D}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(w, "W")],
    )
    quant = onnxsim.quantize_weight_only_int4(model)
    x = _correlated_calibration(D=D, num_samples=32, rank=3, seed=10)
    _assert_agrees(model, quant, [("X", "Y")], [{"X": x}])


def test_brecq_cpp_custom_hyperparameters():
    model = _residual_block_model(D=32, seed=15)
    quant = _quantize_chain_int4(model, {"W1", "W2"})
    x = _correlated_calibration(D=32, num_samples=32, rank=4, seed=16)
    _assert_agrees(
        model,
        quant,
        [("X", "Yout")],
        [{"X": x}],
        max_mismatch_frac=0.01,
        num_iterations=100,
        learning_rate=0.3,
        reg_param=0.05,
        warm_start=0.1,
        beta_range=(10.0, 1.2),
        fisher_eps=1e-2,
    )


def test_brecq_cpp_preserves_scale_and_shape():
    model = _residual_block_model(D=32, seed=4)
    quant = _quantize_chain_int4(model, {"W1", "W2"})
    x = _correlated_calibration(D=32, num_samples=16, rank=3, seed=5)
    before_scales = {}
    for name in ("Y1", "Y2"):
        matmul_node = next(n for n in quant.graph.node if n.output[0] == name)
        dq_node = next(
            n for n in quant.graph.node if n.output[0] == matmul_node.input[1]
        )
        before_scales[name] = onnx.numpy_helper.to_array(
            next(t for t in quant.graph.initializer if t.name == dq_node.input[1])
        )

    cpp = apply_brecq_cpp(model, quant, [("X", "Yout")], [{"X": x}])
    for name in ("Y1", "Y2"):
        matmul_node = next(n for n in cpp.graph.node if n.output[0] == name)
        dq_node = next(n for n in cpp.graph.node if n.output[0] == matmul_node.input[1])
        after_scale = onnx.numpy_helper.to_array(
            next(t for t in cpp.graph.initializer if t.name == dq_node.input[1])
        )
        np.testing.assert_array_equal(before_scales[name], after_scale)
        wq = next(t for t in cpp.graph.initializer if t.name == dq_node.input[0])
        assert list(wq.dims) == [32, 32]


def test_brecq_cpp_codes_stay_in_range():
    model = _residual_block_model(D=32, seed=6)
    quant = _quantize_chain_int4(model, {"W1", "W2"})
    x = _correlated_calibration(D=32, num_samples=16, rank=2, seed=7) * 3
    cpp = apply_brecq_cpp(model, quant, [("X", "Yout")], [{"X": x}])
    checked_any = False
    for t in cpp.graph.initializer:
        if t.data_type != onnx.TensorProto.INT4:
            continue
        checked_any = True
        numel = int(np.prod(list(t.dims)))
        raw = np.frombuffer(t.raw_data, dtype=np.uint8)
        lo = (raw & 0x0F).astype(np.int8)
        hi = ((raw >> 4) & 0x0F).astype(np.int8)
        lo = np.where(lo >= 8, lo - 16, lo)
        hi = np.where(hi >= 8, hi - 16, hi)
        codes = np.empty(numel, dtype=np.int8)
        codes[0::2] = lo[: (numel + 1) // 2]
        codes[1::2] = hi[: numel // 2]
        assert np.all(codes >= -7) and np.all(codes <= 7)
    assert checked_any


def test_brecq_cpp_missing_calibration_input_raises():
    model = _residual_block_model(D=32, seed=17)
    quant = _quantize_chain_int4(model, {"W1", "W2"})
    with pytest.raises(Exception):
        apply_brecq_cpp(
            model,
            quant,
            [("X", "Yout")],
            calibration_data=[{"NotX": np.zeros((1, 32), np.float32)}],
        )


def test_brecq_cpp_noop_when_block_topology_not_discovered():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    cpp = apply_brecq_cpp(
        model,
        model,
        [("X", "Y")],
        calibration_data=[{"X": np.zeros((4, 4), dtype=np.float32)}],
    )
    assert cpp.SerializeToString() == model.SerializeToString()


def test_brecq_cpp_noop_when_no_blocks_given():
    model = _residual_block_model(D=32, seed=11)
    quant = _quantize_chain_int4(model, {"W1", "W2"})
    cpp = apply_brecq_cpp(
        model,
        quant,
        [],
        calibration_data=[{"X": np.zeros((1, 32), dtype=np.float32)}],
    )
    assert cpp.SerializeToString() == quant.SerializeToString()


def test_brecq_cpp_delegation_matches_public_api():
    # onnxsim.apply_brecq now delegates unconditionally to this C++ port
    # -- confirm both entry points give byte-identical results.
    model = _residual_block_model(D=32, seed=21)
    quant = _quantize_chain_int4(model, {"W1", "W2"})
    x = _correlated_calibration(D=32, num_samples=32, rank=4, seed=22)
    via_public = onnxsim.apply_brecq(
        model, quant, blocks=[("X", "Yout")], calibration_data=[{"X": x}]
    )
    via_cpp = apply_brecq_cpp(model, quant, [("X", "Yout")], [{"X": x}])
    assert via_public.SerializeToString() == via_cpp.SerializeToString()


def test_brecq_cpp_output_stays_close_to_float_via_onnxruntime():
    model = _residual_block_model(D=32, seed=2)
    quant = _quantize_chain_int4(model, {"W1", "W2"})
    x = _correlated_calibration(D=32, num_samples=32, rank=4, seed=3)
    cpp = apply_brecq_cpp(model, quant, [("X", "Yout")], [{"X": x}])
    onnx.checker.check_model(cpp)

    (float_y,) = _run(model, {"X": x})
    (cpp_y,) = _run(cpp, {"X": x})
    assert np.all(np.isfinite(cpp_y))
    rel = np.linalg.norm(float_y.astype(np.float64) - cpp_y.astype(np.float64)) / max(
        np.linalg.norm(float_y.astype(np.float64)), 1e-6
    )
    assert rel < 0.25
