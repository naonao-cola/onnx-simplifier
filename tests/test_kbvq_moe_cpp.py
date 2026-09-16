"""Tests for ``onnxsim.apply_kbvq_moe_cpp`` -- the C++-backed port of
``onnxsim.apply_kbvq_moe`` (KBVQ-MoE, see ``onnxsim/passes/kbvq_moe.h``).
Unlike this repo's own k-means-family ports with a genuine RNG divergence
(``tests/test_aqlm_cpp.py``, ``tests/test_lo_bcq_cpp.py``), this port has
none: its shared-KLT-basis fit is an ordinary, deterministic SVD, and its
per-expert residual codebook fit reuses ``kmeans_quantization.h``'s own
``QuantizeDequantizeKMeans`` directly -- the same deterministic,
percentile-initialized k-means ``tests/test_kmeans_quantization_cpp.py``'s
own docstring already documents as identical to the Python reference
outside one narrow edge case (fewer than ``2**bits`` distinct
percentile-derived initial centroids), which none of the random weight
tensors below hit. So, unlike the k-means-family tests just cited, this
file cross-checks numerically against ``onnxsim.apply_kbvq_moe`` directly
(not just structural/algebraic properties), in addition to the same
algorithmic-benefit demonstration (shared basis beats a matched-budget
naive per-expert codebook) ``tests/test_kbvq_moe.py`` itself uses.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim

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


def _current_expert_weights(model):
    """Reads fc1_experts_weights/fc2_experts_weights via the MoE node's own
    CURRENT input names (index 2/4), not the original "FC1W"/"FC2W" literal
    strings -- like every other weight-only ``*_cpp`` port in this repo
    (see e.g. ``tests/test_quip_sharp_cpp.py``'s own ``_current_weight``
    helper), this port rewires the matched node's own weight input to a
    freshly created, auto-named initializer, leaving the original one
    dangling unused in the graph under its old name. This is a real,
    documented divergence from ``kbvq_moe.py``'s own in-place
    ``init.CopyFrom(...)`` (same name, no new initializer, no rewiring) --
    not a bug -- so a test comparing "the current weight" between the two
    sides must resolve each one through its own node's own current input
    name, not a shared literal.
    """
    moe = next(n for n in model.graph.node if n.op_type == "MoE")
    inits = _moe_inits(model)
    return inits[moe.input[2]], inits[moe.input[4]]


def _moe_router_model(
    fc1_w,
    fc2_w,
    router_w,
    fc3_w=None,
    activation="relu",
    k=1,
    tokens=16,
    dtype="float",
):
    num_experts, inter, hidden = fc1_w.shape
    fc3_w_arg = "FC3W" if fc3_w is not None else ""
    cast = _f32 if dtype == "float" else _f16
    model = _model(
        f"""
        g ({dtype}[{tokens},{hidden}] X) => ({dtype}[{tokens},{hidden}] Y)
        {{
          R = Gemm(X, RW)
          Y = com.microsoft.MoE <k={k}, activation_type="{activation}"> (X, R, FC1W, "", FC2W, "", {fc3_w_arg})
        }}
        """
    )
    inits = [cast(fc1_w, "FC1W"), cast(fc2_w, "FC2W"), cast(router_w, "RW")]
    if fc3_w is not None:
        inits.append(cast(fc3_w, "FC3W"))
    model.graph.initializer.extend(inits)
    return model


def test_cpp_replaces_expert_weights_with_same_shape_float():
    # num_experts must comfortably exceed rank+1 (this port's own
    # hardcoded rank=4 default): centering E experts' own flattened weight
    # vectors before the KLT/SVD fit removes one degree of freedom, so the
    # centered [E, D] stack's own rank is at most E-1 -- when E-1 <= rank,
    # the shared basis captures the group's ENTIRE variance exactly
    # (confirmed identical on the pure-Python _kbvq_reconstruct reference:
    # E=5 here reconstructs to float64 epsilon), leaving nothing for the
    # per-expert residual codebook to lossily compress and making this
    # test's own "the weight changed at all" assertion fail for a genuine
    # linear-algebra reason, not a port bug. E=10 (matching this file's own
    # test_cpp_shared_basis_beats_matched_budget_per_expert_kmeans, which
    # already sized around this same trap) keeps E-1=9 comfortably above
    # rank=4.
    E, hidden, inter, tokens, k = 10, 12, 8, 20, 2
    rng = np.random.default_rng(11)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float32)
    router_w = (rng.standard_normal((hidden, E)) * 0.2).astype(np.float32)
    model = _moe_router_model(fc1_w, fc2_w, router_w, k=k, tokens=tokens)
    onnx.checker.check_model(model)

    quantized = onnxsim.apply_kbvq_moe_cpp(model)
    onnx.checker.check_model(quantized)
    new_fc1, new_fc2 = _current_expert_weights(quantized)
    assert new_fc1.shape == fc1_w.shape
    assert new_fc2.shape == fc2_w.shape
    assert new_fc1.dtype == np.float32
    assert not np.allclose(new_fc1, fc1_w)
    assert not np.allclose(new_fc2, fc2_w)
    # No new graph nodes -- weight-only, folded straight into a replacement
    # initializer (unlike kbvq_moe.py's own in-place same-name update, this
    # port rewires the MoE node's own input to a freshly created
    # initializer instead; no new op is ever added on either side).
    assert [n.op_type for n in quantized.graph.node] == [
        n.op_type for n in model.graph.node
    ]


def test_cpp_matches_python_reference_closely():
    # No RNG divergence in this port (see this module's own docstring), so
    # a genuinely tight numeric cross-check against the pure-Python
    # reference is expected here, unlike every k-means-family port with a
    # real RNG difference.
    E, hidden, inter, tokens, k = 6, 14, 10, 18, 2
    rng = np.random.default_rng(23)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.4).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.4).astype(np.float32)
    router_w = (rng.standard_normal((hidden, E)) * 0.2).astype(np.float32)
    model = _moe_router_model(fc1_w, fc2_w, router_w, k=k, tokens=tokens)
    onnx.checker.check_model(model)

    py = onnxsim.apply_kbvq_moe(model)
    cpp = onnxsim.apply_kbvq_moe_cpp(model)
    onnx.checker.check_model(cpp)

    # kbvq_moe.py updates FC1W/FC2W in place (same name); this port rewires
    # the MoE node's own input to a freshly created initializer instead
    # (see _current_expert_weights's own docstring) -- so each side's
    # current weight is resolved through its own node's own current input
    # name, not the shared literal "FC1W"/"FC2W".
    py_fc1, py_fc2 = _current_expert_weights(py)
    cpp_fc1, cpp_fc2 = _current_expert_weights(cpp)
    np.testing.assert_allclose(cpp_fc1, py_fc1, rtol=1e-3, atol=1e-5)
    np.testing.assert_allclose(cpp_fc2, py_fc2, rtol=1e-3, atol=1e-5)


def test_cpp_shared_basis_beats_matched_budget_per_expert_kmeans():
    # The whole point of KBVQ-MoE (mirroring test_kbvq_moe.py's own key
    # demonstration, end to end through the graph this time): a router
    # group with real cross-expert shared structure (every expert = a
    # shared low-rank basis's own reconstruction plus a small independent
    # residual) reconstructs with meaningfully lower error than running
    # onnxsim.kmeans_quantization's own single-codebook VQ independently
    # per expert, at a matched per-element codebook budget (this port's
    # own hardcoded bits=4 default on both sides).
    rng = np.random.default_rng(29)
    num_experts, inter, hidden, true_rank = 10, 16, 8, 2
    d = inter * hidden
    basis = rng.standard_normal((true_rank, d))
    basis /= np.linalg.norm(basis, axis=1, keepdims=True)
    coeffs = rng.standard_normal((num_experts, true_rank)) * 5.0
    mean = rng.standard_normal(d) * 0.05
    shared = mean + coeffs @ basis
    small_residual = rng.standard_normal((num_experts, d)) * 0.02
    fc1_w = (shared + small_residual).reshape(num_experts, inter, hidden).astype(
        np.float32
    )
    fc2_w = rng.standard_normal((num_experts, hidden, inter)).astype(np.float32) * 0.01
    router_w = rng.standard_normal((hidden, num_experts)).astype(np.float32) * 0.2
    model = _moe_router_model(fc1_w, fc2_w, router_w, tokens=6)
    onnx.checker.check_model(model)

    quantized = onnxsim.apply_kbvq_moe_cpp(model)
    kbvq_recon, _ = _current_expert_weights(quantized)
    kbvq_recon = kbvq_recon.astype(np.float64).reshape(num_experts, d)
    kbvq_err = float(np.mean((fc1_w.astype(np.float64).reshape(num_experts, d) - kbvq_recon) ** 2))

    # Naive baseline: quantize_weight_only_kmeans's own per-layer codebook,
    # run independently per expert (one flattened [1, D] "matrix" per
    # expert, matching this port's own bits=4 -> 16-entry codebook).
    baseline_recon = np.empty((num_experts, d), dtype=np.float64)
    for e in range(num_experts):
        row = fc1_w[e].reshape(1, d)
        expert_model = _model(
            f"""
            g (float[1,{d}] X) => (float[1,{d}] Y)
            {{
              Y = MatMul(X, W)
            }}
            """,
            [_f32(row, "W")],
        )
        q = onnxsim.apply_kmeans_quantization_cpp(expert_model)
        # Looked up via the MatMul node's own CURRENT weight input name, not
        # the literal "W" -- this port (like every weight-only *_cpp port in
        # this repo, kbvq_moe_cpp included) rewires the matched node's own
        # weight input to a freshly created, auto-named initializer, leaving
        # the original "W" dangling unused in the graph. Reading "W" directly
        # here would silently read the UNQUANTIZED original weight instead
        # (giving a trivially-zero, meaningless baseline_err below).
        matmul = next(n for n in q.graph.node if n.op_type == "MatMul")
        w_init = next(
            t for t in q.graph.initializer if t.name == matmul.input[1]
        )
        baseline_recon[e] = onnx.numpy_helper.to_array(w_init).astype(np.float64).reshape(d)
    baseline_err = float(
        np.mean((fc1_w.astype(np.float64).reshape(num_experts, d) - baseline_recon) ** 2)
    )

    assert kbvq_err < baseline_err * 0.5


def test_cpp_declines_fc3():
    # Matches pruning.py's own _match_moe_producer decline (no CPU
    # execution oracle for fc3): this port's own patternMatchPredicate
    # never matches this node, so the model is left untouched.
    E, hidden, inter, tokens = 2, 4, 3, 6
    rng = np.random.default_rng(13)
    fc1_w = rng.standard_normal((E, inter, hidden)).astype(np.float32)
    fc2_w = rng.standard_normal((E, hidden, inter)).astype(np.float32)
    fc3_w = rng.standard_normal((E, inter, hidden)).astype(np.float32)
    router_w = rng.standard_normal((hidden, E)).astype(np.float32)
    model = _moe_router_model(fc1_w, fc2_w, router_w, fc3_w=fc3_w, tokens=tokens)

    quantized = onnxsim.apply_kbvq_moe_cpp(model)
    inits = _moe_inits(quantized)
    np.testing.assert_array_equal(inits["FC1W"], fc1_w)
    np.testing.assert_array_equal(inits["FC2W"], fc2_w)


def test_cpp_declines_float16_experts():
    # The node itself still structurally matches (FLOAT16 is one of
    # _is_supported_float_dtype's three accepted dtypes), but this port's
    # own runTransform only reconstructs a FLOAT32 fc1/fc2 tensor -- a
    # FLOAT16 one is left completely untouched, per-tensor, exactly
    # matching apply_kbvq_moe's own `if init.data_type != FLOAT: continue`.
    E, hidden, inter, tokens = 2, 4, 3, 6
    rng = np.random.default_rng(17)
    fc1_w = rng.standard_normal((E, inter, hidden)).astype(np.float16)
    fc2_w = rng.standard_normal((E, hidden, inter)).astype(np.float16)
    router_w = rng.standard_normal((hidden, E)).astype(np.float16)
    model = _moe_router_model(fc1_w, fc2_w, router_w, tokens=tokens, dtype="float16")

    quantized = onnxsim.apply_kbvq_moe_cpp(model)
    inits = _moe_inits(quantized)
    np.testing.assert_array_equal(inits["FC1W"], fc1_w)
    np.testing.assert_array_equal(inits["FC2W"], fc2_w)


def test_cpp_noop_when_no_moe_node_present():
    model = _model(
        """
        g (float[4,8] X) => (float[4,8] Y)
        {
          Y = Identity(X)
        }
        """
    )
    result = onnxsim.apply_kbvq_moe_cpp(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_cpp_quantized_model_still_executes_on_onnxruntime():
    # Loose, execution-level sanity check (platform-numerics note: onnxruntime
    # is not bit-exact across CPU architectures, so this is deliberately a
    # coarse relative-error bound, separate from the tight cross-check
    # above).
    E, hidden, inter, tokens, k = 6, 10, 8, 16, 2
    rng = np.random.default_rng(19)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float32)
    router_w = (rng.standard_normal((hidden, E)) * 0.2).astype(np.float32)
    model = _moe_router_model(fc1_w, fc2_w, router_w, k=k, tokens=tokens)
    onnx.checker.check_model(model)

    quantized = onnxsim.apply_kbvq_moe_cpp(model)
    onnx.checker.check_model(quantized)

    feed_rng = np.random.default_rng(23)
    feeds = {"X": feed_rng.standard_normal((tokens, hidden)).astype(np.float32)}
    (out_float,) = _run(model, feeds)
    (out_quant,) = _run(quantized, feeds)
    assert out_quant.shape == out_float.shape
    assert np.all(np.isfinite(out_quant))
    rel_err = np.linalg.norm(out_quant - out_float) / max(
        np.linalg.norm(out_float), 1e-6
    )
    assert rel_err < 0.5
