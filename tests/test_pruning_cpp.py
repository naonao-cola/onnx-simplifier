"""Tests for ``onnxsim.prune_magnitude_cpp`` -- the C++-backed port of
``onnxsim.apply_magnitude_pruning`` (see
``onnxsim/passes/magnitude_pruning.h``). Scope note: unlike the pure-Python
version, this port does not match ``com.microsoft::Attention``'s merged QKV
weight and offers no N:M pruning mode -- see that header's own doc comment.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim

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


def _matmul_model(K=64, N=16, seed=0):
    rng = np.random.default_rng(seed)
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    return _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f32(weight, "W")],
    )


def _single_conv_model(w, spatial=10, group=1):
    Cout, Cin_per_group, kh, kw = w.shape
    Cin = Cin_per_group * group
    out_spatial = spatial - kh + 1
    attrs = f"kernel_shape=[{kh},{kw}]"
    if group != 1:
        attrs += f", group={group}"
    return _model(
        f"""
        g (float[N,{Cin},{spatial},{spatial}] X) => (float[N,{Cout},{out_spatial},{out_spatial}] Y)
        {{
          Y = Conv<{attrs}>(X, W1)
        }}
        """,
        initializer=[_f32(w, "W1")],
    )


def _weight(model):
    # Unlike the pure-Python apply_magnitude_pruning (which mutates the
    # existing initializer in place via w_init.CopyFrom), the C++ pass
    # leaves the original initializer dangling and appends a *new*,
    # anonymously-named one for the pruned weight, and rewires the node's
    # own weight input to it (matching every other onnxsim rewrite's
    # "replace, don't mutate" convention for constants) -- so the pruned
    # weight must be found via the node's current weight input name, not
    # assumed to still be named "W" or still be initializer[0].
    node = model.graph.node[0]
    w_name = node.input[1]
    init = next(t for t in model.graph.initializer if t.name == w_name)
    return onnx.numpy_helper.to_array(init)


def test_cpp_magnitude_pruning_reaches_target_sparsity():
    model = _matmul_model(K=64, N=16)
    pruned = onnxsim.prune_magnitude_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    assert onnxsim.weight_sparsity(pruned) == pytest.approx(0.5, abs=1e-9)
    assert _weight(pruned).shape == _weight(model).shape


def test_cpp_magnitude_pruning_keeps_the_largest_entries_per_row():
    model = _matmul_model(K=64, N=16)
    pruned = onnxsim.prune_magnitude_cpp(model, sparsity=0.75)
    w = _weight(model).astype(np.float64)  # [K, N]
    w_pruned = _weight(pruned).astype(np.float64)
    for col in range(w.shape[1]):
        kept = np.flatnonzero(w_pruned[:, col] != 0)
        assert len(kept) == 16  # round(64 * 0.25)
        threshold = np.abs(w[:, col])[kept].min()
        dropped_max = np.abs(w[:, col])[np.flatnonzero(w_pruned[:, col] == 0)].max()
        assert dropped_max <= threshold


def test_cpp_magnitude_pruning_zero_sparsity_is_a_no_op():
    model = _matmul_model(K=32, N=8)
    pruned = onnxsim.prune_magnitude_cpp(model, sparsity=0.0)
    np.testing.assert_array_equal(_weight(pruned), _weight(model))


def test_cpp_magnitude_pruning_rejects_invalid_sparsity():
    model = _matmul_model(K=32, N=8)
    with pytest.raises(Exception):
        onnxsim.prune_magnitude_cpp(model, sparsity=1.0)


def test_cpp_magnitude_pruning_conv_reaches_target_sparsity():
    Cin, Cout = 4, 8  # K = Cin*3*3 = 36
    rng = np.random.default_rng(60)
    w = rng.standard_normal((Cout, Cin, 3, 3)).astype(np.float32)
    model = _single_conv_model(w)

    pruned = onnxsim.prune_magnitude_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    assert onnxsim.weight_sparsity(pruned) == pytest.approx(0.5, abs=1e-9)
    assert _weight(pruned).shape == w.shape


def test_cpp_magnitude_pruning_conv_depthwise_reaches_target_sparsity():
    C = 8
    rng = np.random.default_rng(61)
    w = rng.standard_normal((C, 1, 4, 4)).astype(np.float32)  # K=16, halved exactly
    model = _single_conv_model(w, spatial=10, group=C)

    pruned = onnxsim.prune_magnitude_cpp(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    assert onnxsim.weight_sparsity(pruned) == pytest.approx(0.5, abs=1e-9)


def test_cpp_magnitude_pruning_matches_python_reference():
    # Both ports implement the exact same per-row keep-count rule -- on the
    # same weight the *set* of surviving (nonzero) entries should match
    # exactly, even if tie-breaking among equal-magnitude entries (rare with
    # continuous random weights) could in principle differ.
    model = _matmul_model(K=64, N=16, seed=7)
    pruned_py = onnxsim.apply_magnitude_pruning(model, sparsity=0.5)
    pruned_cpp = onnxsim.prune_magnitude_cpp(model, sparsity=0.5)

    mask_py = _weight(pruned_py) != 0
    mask_cpp = _weight(pruned_cpp) != 0
    assert np.array_equal(mask_py, mask_cpp)
    np.testing.assert_array_equal(_weight(pruned_py), _weight(pruned_cpp))


def test_cpp_magnitude_pruning_output_stays_finite_and_close():
    model = _matmul_model(K=64, N=16, seed=8)
    pruned = onnxsim.prune_magnitude_cpp(model, sparsity=0.3)
    onnx.checker.check_model(pruned)

    rng = np.random.default_rng(9)
    x = rng.standard_normal((4, 64)).astype(np.float32)
    sess_f = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    sess_p = ort.InferenceSession(
        pruned.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (y_f,) = sess_f.run(None, {"X": x})
    (y_p,) = sess_p.run(None, {"X": x})
    assert np.all(np.isfinite(y_p))
    rel = np.linalg.norm(y_f - y_p) / max(np.linalg.norm(y_f), 1e-6)
    assert rel < 1.0  # 30% sparsity perturbs but shouldn't blow up the output
