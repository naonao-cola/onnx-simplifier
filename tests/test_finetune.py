"""Tests for ``onnxsim.finetune`` -- layer-wise ridge-regression fine-tuning
that recovers accuracy lost to structured/attention-head pruning, see
``onnxsim/finetune.py``.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.finetune import (
    _find_channel_correspondence,
    _find_keep_indices,
    _ridge_fit,
)

ort = pytest.importorskip("onnxruntime")


def _model(body, initializer=(), opset=21):
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
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


def _weights(model):
    return {t.name: onnx.numpy_helper.to_array(t) for t in model.graph.initializer}


def _mlp_model(K=8, H=32, Out=4, bias=True, seed=0):
    rng = np.random.default_rng(seed)
    w1 = rng.standard_normal((K, H)).astype(np.float32)
    w2 = rng.standard_normal((H, Out)).astype(np.float32)
    initializer = [_f32(w1, "W1"), _f32(w2, "W2")]
    if bias:
        b1 = rng.standard_normal((H,)).astype(np.float32)
        gemm1 = "h = Gemm(X, W1, B1)"
        initializer.append(_f32(b1, "B1"))
    else:
        gemm1 = "h = MatMul(X, W1)"
    return _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          {gemm1}
          a = Relu(h)
          Y = MatMul(a, W2)
        }}
        """,
        initializer=initializer,
    )


def _3layer_model(K=8, H1=16, H2=16, Out=4, seed=0):
    rng = np.random.default_rng(seed)
    w1 = rng.standard_normal((K, H1)).astype(np.float32)
    w2 = rng.standard_normal((H1, H2)).astype(np.float32)
    w3 = rng.standard_normal((H2, Out)).astype(np.float32)
    return _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{Out}] Y)
        {{
          h1 = MatMul(X, W1)
          a1 = Relu(h1)
          h2 = MatMul(a1, W2)
          a2 = Relu(h2)
          Y = MatMul(a2, W3)
        }}
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2"), _f32(w3, "W3")],
    )


def _rel_err(y, y_ref):
    return np.linalg.norm(y - y_ref) / np.linalg.norm(y_ref)


def test_finetune_reduces_reconstruction_error_after_structured_pruning():
    K, H, Out = 8, 32, 4
    model = _mlp_model(K=K, H=H, Out=Out, seed=0)
    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    rng = np.random.default_rng(1)
    x_calib = rng.standard_normal((256, K)).astype(np.float32)
    calibration_data = [{"X": x_calib}]

    finetuned = onnxsim.apply_pruning_finetune(
        model, pruned, calibration_data=calibration_data, reg_param=1e-4
    )
    onnx.checker.check_model(finetuned)

    x_test = rng.standard_normal((64, K)).astype(np.float32)
    (y_orig,) = _run(model, {"X": x_test})
    (y_pruned,) = _run(pruned, {"X": x_test})
    (y_finetuned,) = _run(finetuned, {"X": x_test})

    err_pruned = _rel_err(y_pruned, y_orig)
    err_finetuned = _rel_err(y_finetuned, y_orig)
    assert err_finetuned < err_pruned


def test_finetune_shapes_are_unchanged():
    K, H, Out = 8, 32, 4
    model = _mlp_model(K=K, H=H, Out=Out, seed=2)
    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)

    rng = np.random.default_rng(3)
    calibration_data = [{"X": rng.standard_normal((64, K)).astype(np.float32)}]
    finetuned = onnxsim.apply_pruning_finetune(
        model, pruned, calibration_data=calibration_data
    )

    pruned_w = _weights(pruned)
    finetuned_w = _weights(finetuned)
    assert finetuned_w.keys() == pruned_w.keys()
    for name in pruned_w:
        assert finetuned_w[name].shape == pruned_w[name].shape


def test_finetune_leaves_output_producer_weight_unchanged():
    # The very first layer's own output channels are pruned (a producer
    # chain), but its own input channels are not -- pruning's own row
    # slice is already the exact least-squares reconstruction of the
    # target (see this module's own docstring), so fine-tuning should
    # leave it untouched.
    K, H, Out = 8, 32, 4
    model = _mlp_model(K=K, H=H, Out=Out, seed=4)
    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)

    rng = np.random.default_rng(5)
    calibration_data = [{"X": rng.standard_normal((128, K)).astype(np.float32)}]
    finetuned = onnxsim.apply_pruning_finetune(
        model, pruned, calibration_data=calibration_data, reg_param=1e-6
    )

    pruned_w = _weights(pruned)
    finetuned_w = _weights(finetuned)
    np.testing.assert_allclose(finetuned_w["W1"], pruned_w["W1"], rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(finetuned_w["B1"], pruned_w["B1"], rtol=1e-3, atol=1e-4)
    # W2 (input-pruned consumer) should actually change.
    assert not np.allclose(finetuned_w["W2"], pruned_w["W2"])


def test_finetune_declines_dual_axis_pruned_interior_layer():
    K, H1, H2, Out = 8, 16, 16, 4
    model = _3layer_model(K=K, H1=H1, H2=H2, Out=Out, seed=6)
    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)

    inits = {t.name: t for t in pruned.graph.initializer}
    # W2 is the interior layer -- pruned on both its own input (H1) and
    # output (H2) axes.
    assert list(inits["W2"].dims) != [H1, H2]

    rng = np.random.default_rng(7)
    calibration_data = [{"X": rng.standard_normal((128, K)).astype(np.float32)}]
    finetuned = onnxsim.apply_pruning_finetune(
        model, pruned, calibration_data=calibration_data
    )
    onnx.checker.check_model(finetuned)

    pruned_w = _weights(pruned)
    finetuned_w = _weights(finetuned)
    # Declined outright -- left byte-identical to the pruned model's own
    # weight, never guessed at.
    np.testing.assert_array_equal(finetuned_w["W2"], pruned_w["W2"])
    # W3 (input-pruned consumer of the last chain) should still be
    # re-fit against the shifted upstream activation.
    assert not np.allclose(finetuned_w["W3"], pruned_w["W3"])

    x_test = rng.standard_normal((16, K)).astype(np.float32)
    (y,) = _run(finetuned, {"X": x_test})
    assert np.all(np.isfinite(y))


def test_finetune_is_a_noop_when_no_matmul_or_gemm_nodes_present():
    model = _model(
        """
        g (float[batch,4] X) => (float[batch,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    pruned = onnx.ModelProto()
    pruned.CopyFrom(model)

    result = onnxsim.apply_pruning_finetune(model, pruned)
    assert result is pruned


def test_finetune_is_a_noop_when_pruned_model_equals_original():
    # No candidate is declined here, but every layer's own fit is already
    # (up to floating point) the current weight -- nothing meaningfully
    # changes.
    K, H, Out = 8, 32, 4
    model = _mlp_model(K=K, H=H, Out=Out, seed=8)
    pruned = onnx.ModelProto()
    pruned.CopyFrom(model)

    rng = np.random.default_rng(9)
    calibration_data = [{"X": rng.standard_normal((256, K)).astype(np.float32)}]
    finetuned = onnxsim.apply_pruning_finetune(
        model, pruned, calibration_data=calibration_data, reg_param=1e-8
    )

    orig_w = _weights(model)
    finetuned_w = _weights(finetuned)
    for name in orig_w:
        np.testing.assert_allclose(
            finetuned_w[name], orig_w[name], rtol=1e-2, atol=1e-3
        )


def test_finetune_accepts_file_paths(tmp_path):
    K, H, Out = 8, 32, 4
    model = _mlp_model(K=K, H=H, Out=Out, seed=10)
    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)

    orig_path = tmp_path / "orig.onnx"
    pruned_path = tmp_path / "pruned.onnx"
    onnx.save(model, str(orig_path))
    onnx.save(pruned, str(pruned_path))

    rng = np.random.default_rng(11)
    calibration_data = [{"X": rng.standard_normal((64, K)).astype(np.float32)}]
    finetuned = onnxsim.apply_pruning_finetune(
        str(orig_path), str(pruned_path), calibration_data=calibration_data
    )
    assert isinstance(finetuned, onnx.ModelProto)
    onnx.checker.check_model(finetuned)


def test_finetune_generates_calibration_data_when_omitted():
    K, H, Out = 8, 16, 4
    model = _mlp_model(K=K, H=H, Out=Out, seed=12)
    pruned = onnxsim.apply_structured_pruning(model, sparsity=0.5)
    finetuned = onnxsim.apply_pruning_finetune(model, pruned, num_samples=16, seed=0)
    onnx.checker.check_model(finetuned)


# --- unit tests for the module's own private helpers ------------------------


def test_find_keep_indices_recovers_exact_subsequence():
    orig = np.arange(30).reshape(10, 3).astype(np.float64)
    keep = np.array([1, 3, 4, 8])
    pruned = orig[keep]
    found = _find_keep_indices(orig, pruned)
    np.testing.assert_array_equal(found, keep)


def test_find_keep_indices_declines_on_mismatched_content():
    rng = np.random.default_rng(0)
    orig = rng.standard_normal((10, 3))
    pruned = rng.standard_normal((4, 3))  # unrelated rows
    assert _find_keep_indices(orig, pruned) is None


def test_find_keep_indices_declines_when_pruned_is_larger():
    orig = np.arange(12).reshape(4, 3).astype(np.float64)
    pruned = np.arange(18).reshape(6, 3).astype(np.float64)
    assert _find_keep_indices(orig, pruned) is None


def test_find_channel_correspondence_output_axis_only():
    rng = np.random.default_rng(1)
    w_orig = rng.standard_normal((10, 4))
    keep_out = np.array([0, 2, 5, 9])
    w_pruned = w_orig[keep_out]
    found_out, found_in = _find_channel_correspondence(w_orig, w_pruned)
    np.testing.assert_array_equal(found_out, keep_out)
    np.testing.assert_array_equal(found_in, np.arange(4))


def test_find_channel_correspondence_input_axis_only():
    rng = np.random.default_rng(2)
    w_orig = rng.standard_normal((5, 10))
    keep_in = np.array([1, 2, 6, 7, 8])
    w_pruned = w_orig[:, keep_in]
    found_out, found_in = _find_channel_correspondence(w_orig, w_pruned)
    np.testing.assert_array_equal(found_out, np.arange(5))
    np.testing.assert_array_equal(found_in, keep_in)


def test_find_channel_correspondence_declines_both_axes_pruned():
    rng = np.random.default_rng(3)
    w_orig = rng.standard_normal((10, 10))
    w_pruned = w_orig[np.ix_([0, 2, 4, 6], [1, 3, 5, 7])]
    found_out, found_in = _find_channel_correspondence(w_orig, w_pruned)
    assert found_out is None
    assert found_in is None


def test_ridge_fit_matches_unregularized_least_squares_oracle():
    rng = np.random.default_rng(4)
    num_samples, K, N = 200, 6, 3
    x = rng.standard_normal((num_samples, K))
    w_true = rng.standard_normal((N, K))
    b_true = rng.standard_normal(N)
    y = x @ w_true.T + b_true

    w0 = rng.standard_normal((N, K))  # far from optimal -- reg_param -> 0
    b0 = rng.standard_normal(N)
    w_fit, b_fit = _ridge_fit(x, y, w0, b0, reg_param=1e-10)

    np.testing.assert_allclose(w_fit, w_true, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(b_fit, b_true, rtol=1e-3, atol=1e-3)


def test_ridge_fit_pulls_toward_w0_with_no_calibration_signal():
    rng = np.random.default_rng(5)
    K, N = 6, 3
    x = np.zeros((4, K))  # no signal at all
    y = np.zeros((4, N))
    w0 = rng.standard_normal((N, K))
    b0 = rng.standard_normal(N)
    # x contributes nothing to the Gram matrix's own K x K weight block --
    # only the bias-augmented ones-column does -- so the regularized normal
    # equations reduce to lam * w = lam * w0 for the weight part exactly,
    # regardless of reg_param's magnitude, while the bias part still
    # competes against the (nonzero, all-ones) bias column and only
    # converges to b0 as reg_param grows large.
    w_fit, b_fit = _ridge_fit(x, y, w0, b0, reg_param=1.0)
    np.testing.assert_allclose(w_fit, w0)

    w_fit_huge, b_fit_huge = _ridge_fit(x, y, w0, b0, reg_param=1e12)
    np.testing.assert_allclose(w_fit_huge, w0)
    np.testing.assert_allclose(b_fit_huge, b0, rtol=1e-6)


def test_ridge_fit_without_bias():
    rng = np.random.default_rng(6)
    num_samples, K, N = 100, 5, 2
    x = rng.standard_normal((num_samples, K))
    w_true = rng.standard_normal((N, K))
    y = x @ w_true.T
    w0 = rng.standard_normal((N, K))
    w_fit, b_fit = _ridge_fit(x, y, w0, None, reg_param=1e-10)
    assert b_fit is None
    np.testing.assert_allclose(w_fit, w_true, rtol=1e-3, atol=1e-3)
