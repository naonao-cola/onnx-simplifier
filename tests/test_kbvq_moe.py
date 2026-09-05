"""Tests for ``onnxsim.apply_kbvq_moe`` (KBVQ-MoE, see
``onnxsim/kbvq_moe.py``) -- shared-KLT-basis-plus-per-expert-residual-VQ
(simulated) quantization of a ``com.microsoft::MoE`` node's per-expert
weights, reusing ``onnxsim.pruning``'s own MoE chain matcher and
``onnxsim.kmeans_quantization``'s own Lloyd's-algorithm codebook fit.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.kbvq_moe import _kbvq_reconstruct, _klt_basis
from onnxsim.kmeans_quantization import _kmeans_1d

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
    fc3_w=None,
    activation="relu",
    k=1,
    tokens=16,
    dtype="float",
):
    num_experts, inter, hidden = fc1_w.shape
    fc3_w_arg = "FC3W" if fc3_w is not None else ""
    model = _model(
        f"""
        g ({dtype}[{tokens},{hidden}] X) => ({dtype}[{tokens},{hidden}] Y)
        {{
          R = Gemm(X, RW)
          Y = com.microsoft.MoE <k={k}, activation_type="{activation}"> (X, R, FC1W, "", FC2W, "", {fc3_w_arg})
        }}
        """
    )
    inits = [_f32(fc1_w, "FC1W"), _f32(fc2_w, "FC2W"), _f32(router_w, "RW")]
    if fc3_w is not None:
        inits.append(_f32(fc3_w, "FC3W"))
    model.graph.initializer.extend(inits)
    return model


def test_klt_basis_recovers_exact_shared_rank_one_structure():
    # Every expert is *exactly* mean + coeff_e * one shared direction, with
    # no residual at all -- rank-1 KLT should reconstruct every expert
    # exactly (up to floating-point error), since the true generative rank
    # is 1 and rank=1 is not an under-approximation here.
    rng = np.random.default_rng(0)
    direction = rng.standard_normal(20)
    direction /= np.linalg.norm(direction)
    coeffs = np.array([3.0, -1.5, 0.5, 2.0])
    mean = rng.standard_normal(20) * 0.1
    stack = mean + coeffs[:, None] * direction[None, :]

    fitted_mean, basis = _klt_basis(stack, rank=1)
    assert basis.shape == (1, 20)
    coeff = (stack - fitted_mean) @ basis.T
    reconstruction = fitted_mean + coeff @ basis
    np.testing.assert_allclose(reconstruction, stack, atol=1e-8)


def test_klt_basis_rank_zero_is_just_the_mean():
    rng = np.random.default_rng(1)
    stack = rng.standard_normal((5, 10))
    mean, basis = _klt_basis(stack, rank=0)
    np.testing.assert_allclose(mean, stack.mean(axis=0))
    assert basis.shape == (0, 10)


def test_klt_basis_rank_clamped_to_num_experts():
    # rank=100 with only 3 experts: at most 3 basis vectors exist (a set of
    # E points spans a subspace of dimension at most E-1 around their own
    # mean, and the SVD of the centered [E, D] stack has at most E nonzero
    # singular values) -- _klt_basis must clamp rather than error.
    rng = np.random.default_rng(2)
    stack = rng.standard_normal((3, 50))
    _mean, basis = _klt_basis(stack, rank=100)
    assert basis.shape[0] == 3


def test_kbvq_shared_basis_beats_matched_budget_per_expert_kmeans():
    # The whole point of KBVQ-MoE: construct a router group with real
    # cross-expert shared structure (every expert = a shared low-rank
    # basis's own reconstruction + a small independent residual) and show
    # this module's shared-basis-plus-per-expert-residual-codebook
    # reconstruction achieves meaningfully lower total reconstruction error
    # than running onnxsim.kmeans_quantization's own single-codebook VQ
    # independently per expert, at a matched per-element codebook budget
    # (same `bits` -> same 2**bits codes per weight element in both
    # schemes; KBVQ-MoE's only extra cost is the one shared basis, small
    # relative to the flattened per-expert size D used here).
    rng = np.random.default_rng(7)
    num_experts, d, true_rank = 12, 256, 2
    basis = rng.standard_normal((true_rank, d))
    basis /= np.linalg.norm(basis, axis=1, keepdims=True)
    coeffs = rng.standard_normal((num_experts, true_rank)) * 5.0
    mean = rng.standard_normal(d) * 0.05
    shared = mean + coeffs @ basis
    small_residual = rng.standard_normal((num_experts, d)) * 0.02
    stack = shared + small_residual

    bits = 3
    kbvq_reconstruction = _kbvq_reconstruct(
        stack, rank=true_rank, bits=bits, kmeans_iters=25, seed=0
    )
    kbvq_error = np.mean((stack - kbvq_reconstruction) ** 2)

    baseline_reconstruction = np.empty_like(stack)
    num_codes = 2**bits
    for e in range(num_experts):
        centroids, assignments = _kmeans_1d(stack[e], num_codes, 25, 0)
        baseline_reconstruction[e] = centroids[assignments]
    baseline_error = np.mean((stack - baseline_reconstruction) ** 2)

    assert kbvq_error < baseline_error * 0.5


def test_apply_kbvq_moe_reconstructs_experts_within_residual_codebook_range():
    # Confirms, directly against the ONNX initializers this module writes
    # (never round-tripped through onnxruntime -- see this repo's own
    # CLAUDE.md platform-numerics note), that every expert's own quantized
    # weight equals shared_e + a value drawn from that expert's own
    # dequantized residual codebook -- i.e. that apply_kbvq_moe's graph
    # output matches _kbvq_reconstruct exactly for the same parameters.
    E, hidden, inter, tokens, k = 5, 12, 8, 20, 2
    rng = np.random.default_rng(11)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float32)
    router_w = (rng.standard_normal((hidden, E)) * 0.2).astype(np.float32)
    model = _moe_router_model(fc1_w, fc2_w, router_w, k=k, tokens=tokens)
    onnx.checker.check_model(model)

    quantized = onnxsim.apply_kbvq_moe(model, rank=2, bits=3, seed=5)
    onnx.checker.check_model(quantized)
    inits = _moe_inits(quantized)

    expected_fc1 = _kbvq_reconstruct(
        fc1_w.astype(np.float64).reshape(E, -1), rank=2, bits=3, kmeans_iters=20, seed=5
    ).reshape(fc1_w.shape)
    expected_fc2 = _kbvq_reconstruct(
        fc2_w.astype(np.float64).reshape(E, -1), rank=2, bits=3, kmeans_iters=20, seed=5
    ).reshape(fc2_w.shape)
    np.testing.assert_allclose(
        inits["FC1W"], expected_fc1.astype(np.float32), rtol=1e-5
    )
    np.testing.assert_allclose(
        inits["FC2W"], expected_fc2.astype(np.float32), rtol=1e-5
    )
    assert not np.allclose(inits["FC1W"], fc1_w)
    assert not np.allclose(inits["FC2W"], fc2_w)


def test_apply_kbvq_moe_declines_fc3():
    # Matches onnxsim.pruning._match_moe_producer's own decline (no CPU
    # execution oracle for fc3 -- see onnxsim/pruning.py's own section
    # comment): _find_moe_chains never matches this node, so
    # apply_kbvq_moe has nothing to quantize and returns the model
    # untouched.
    E, hidden, inter, tokens = 2, 4, 3, 6
    rng = np.random.default_rng(13)
    fc1_w = rng.standard_normal((E, inter, hidden)).astype(np.float32)
    fc2_w = rng.standard_normal((E, hidden, inter)).astype(np.float32)
    fc3_w = rng.standard_normal((E, inter, hidden)).astype(np.float32)
    router_w = rng.standard_normal((hidden, E)).astype(np.float32)
    model = _moe_router_model(fc1_w, fc2_w, router_w, fc3_w=fc3_w, tokens=tokens)

    quantized = onnxsim.apply_kbvq_moe(model)
    inits = _moe_inits(quantized)
    np.testing.assert_array_equal(inits["FC1W"], fc1_w)
    np.testing.assert_array_equal(inits["FC2W"], fc2_w)


def test_apply_kbvq_moe_declines_float16_experts():
    E, hidden, inter, tokens = 2, 4, 3, 6
    rng = np.random.default_rng(17)
    fc1_w = rng.standard_normal((E, inter, hidden)).astype(np.float16)
    fc2_w = rng.standard_normal((E, hidden, inter)).astype(np.float16)
    router_w = rng.standard_normal((hidden, E)).astype(np.float16)
    model = _moe_router_model(fc1_w, fc2_w, router_w, tokens=tokens, dtype="float16")

    quantized = onnxsim.apply_kbvq_moe(model)
    inits = _moe_inits(quantized)
    np.testing.assert_array_equal(inits["FC1W"], fc1_w)
    np.testing.assert_array_equal(inits["FC2W"], fc2_w)


def test_apply_kbvq_moe_no_moe_node_is_a_no_op():
    model = _model(
        """
        g (float[4,8] X) => (float[4,8] Y)
        {
          Y = Identity(X)
        }
        """
    )
    quantized = onnxsim.apply_kbvq_moe(model)
    assert quantized.SerializeToString() == model.SerializeToString()


def test_apply_kbvq_moe_quantized_model_still_executes_on_onnxruntime():
    # Loose, execution-level sanity check (platform-numerics note: onnxruntime
    # is not bit-exact across CPU architectures, so this is deliberately a
    # coarse relative-error bound, separate from the exact grid check above).
    E, hidden, inter, tokens, k = 6, 10, 8, 16, 2
    rng = np.random.default_rng(19)
    fc1_w = (rng.standard_normal((E, inter, hidden)) * 0.3).astype(np.float32)
    fc2_w = (rng.standard_normal((E, hidden, inter)) * 0.3).astype(np.float32)
    router_w = (rng.standard_normal((hidden, E)) * 0.2).astype(np.float32)
    model = _moe_router_model(fc1_w, fc2_w, router_w, k=k, tokens=tokens)
    onnx.checker.check_model(model)

    quantized = onnxsim.apply_kbvq_moe(model, rank=3, bits=4)
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
