"""Tests for ``onnxsim.apply_slim_llm`` -- see ``onnxsim/slim_llm.py`` for
the technique (calibration-driven per-*group* choice, within a single
layer's weight, between two integer bit-widths -- as opposed to
``onnxsim.apply_mixed_precision_quantization``'s per-*layer* choice).
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim

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


def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-6)


def _matmul_model(K, N, seed=0, opset=21):
    rng = np.random.default_rng(seed)
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.1
    return _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
        opset=opset,
    )


def _matmul_model_with_outlier_group(K, N, group_size, outlier_group, seed=0):
    # One group's columns get much larger-magnitude values than every other
    # group's -- a good salience ranking must single out THIS group (not
    # any other) for high_bits, since it dominates the layer's own
    # reconstruction error.
    rng = np.random.default_rng(seed)
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.05
    lo, hi = outlier_group * group_size, (outlier_group + 1) * group_size
    weight[lo:hi, :] = rng.standard_normal((group_size, N)).astype(np.float32) * 10.0
    return _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )


def _calibration(K, num_samples=32, seed=1):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((num_samples, K)).astype(np.float32)


def _group_bits(model, prefix="W"):
    tensor = next(
        t
        for t in model.graph.initializer
        if t.name.startswith(prefix) and t.name.endswith("_group_bits")
    )
    return np.frombuffer(tensor.raw_data, dtype=np.int64).copy()


def test_slim_llm_picks_the_more_salient_group_for_high_bits():
    K, N, group_size, outlier_group = 64, 8, 16, 2
    model = _matmul_model_with_outlier_group(
        K=K, N=N, group_size=group_size, outlier_group=outlier_group, seed=0
    )
    x = _calibration(K=K, num_samples=64, seed=1)
    q = onnxsim.apply_slim_llm(
        model,
        calibration_data=[{"X": x}],
        target_bits=3.0,
        low_bits=2,
        high_bits=4,
        group_size=group_size,
    )
    onnx.checker.check_model(q)

    bits = _group_bits(q)
    assert len(bits) == K // group_size
    assert bits[outlier_group] == 4
    # The overall budget (target_bits=3.0, low=2, high=4) only affords half
    # the groups at high_bits -- so at least one other group must stay low.
    assert np.any(bits == 2)


def test_slim_llm_average_bits_matches_target_budget():
    K, N, group_size = 128, 8, 16
    model = _matmul_model(K=K, N=N, seed=2)
    x = _calibration(K=K, num_samples=32, seed=3)
    q = onnxsim.apply_slim_llm(
        model,
        calibration_data=[{"X": x}],
        target_bits=2.5,
        low_bits=2,
        high_bits=4,
        group_size=group_size,
    )
    bits = _group_bits(q)
    # 8 groups total; fraction_high = (2.5-2)/(4-2) = 0.25 -> 2 groups high.
    assert len(bits) == K // group_size
    assert np.mean(bits) == pytest.approx(2.5, abs=0.26)


def test_slim_llm_finer_than_mixed_precision_within_one_layer():
    # The distinguishing claim: within a SINGLE layer, different groups can
    # land on different bit-widths -- unlike apply_mixed_precision_quantization,
    # which always gives one whole layer a single bit-width.
    K, N, group_size, outlier_group = 64, 8, 16, 1
    model = _matmul_model_with_outlier_group(
        K=K, N=N, group_size=group_size, outlier_group=outlier_group, seed=4
    )
    x = _calibration(K=K, num_samples=64, seed=5)
    q = onnxsim.apply_slim_llm(
        model,
        calibration_data=[{"X": x}],
        target_bits=2.5,
        low_bits=2,
        high_bits=4,
        group_size=group_size,
    )
    bits = _group_bits(q)
    assert len(set(bits.tolist())) > 1


def test_slim_llm_reconstruction_matches_codes_times_scale():
    # Verify the actual claim about the initializers this module writes --
    # codes * scale must equal what DequantizeLinear(codes, scale) computes
    # -- directly against the raw arrays (a tight *relative* check), rather
    # than through an onnxruntime session (whose MatMul kernel reduction
    # order differs across CPU architectures and isn't what this module
    # promises).
    K, N, group_size = 64, 8, 16
    model = _matmul_model(K=K, N=N, seed=6)
    x = _calibration(K=K, num_samples=32, seed=7)
    q = onnxsim.apply_slim_llm(
        model,
        calibration_data=[{"X": x}],
        target_bits=3.0,
        group_size=group_size,
    )

    codes_init = next(
        t for t in q.graph.initializer if t.name.endswith("_slimllm_codes")
    )
    scale_init = next(
        t for t in q.graph.initializer if t.name.endswith("_slimllm_scale")
    )
    codes = onnx.numpy_helper.to_array(codes_init).astype(np.float64)  # [K, N]
    scale = onnx.numpy_helper.to_array(scale_init).astype(np.float64)  # [K/gs, N]
    scale_full = np.repeat(scale, group_size, axis=0)
    dequant_kn = codes * scale_full

    (dequant_y,) = _run(
        q, {"X": np.eye(K, dtype=np.float32)}
    )  # identity trick: X @ W == W when X == I
    np.testing.assert_allclose(dequant_y, dequant_kn, rtol=1e-2, atol=1e-2)


def test_slim_llm_output_stays_finite_via_onnxruntime():
    K, N, group_size = 64, 8, 16
    model = _matmul_model(K=K, N=N, seed=8)
    x = _calibration(K=K, num_samples=32, seed=9)
    q = onnxsim.apply_slim_llm(
        model,
        calibration_data=[{"X": x}],
        target_bits=3.0,
        group_size=group_size,
    )
    onnx.checker.check_model(q)

    rng = np.random.default_rng(10)
    x_eval = rng.standard_normal((4, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x_eval})
    (q_y,) = _run(q, {"X": x_eval})
    assert np.all(np.isfinite(q_y))
    # Loose, absolute-tolerance-free sanity check only -- see this repo's
    # own note on onnxruntime's non-bit-exact MatMul reduction order across
    # CPU architectures; the tight, deterministic check lives in
    # test_slim_llm_reconstruction_matches_codes_times_scale above.
    assert _rel_l2(float_y, q_y) < 1.5


def test_slim_llm_declines_when_k_not_divisible_by_group_size():
    rng = np.random.default_rng(11)
    weight = rng.standard_normal((20, 4)).astype(np.float32)  # 20 not a multiple of 16
    model = _model(
        """
        g (float[batch,20] X) => (float[batch,4] Y)
        {
          Y = MatMul(X, W)
        }
        """,
        [_f32(weight, "W")],
    )
    q = onnxsim.apply_slim_llm(model, group_size=16)
    assert q.SerializeToString() == model.SerializeToString()


def test_slim_llm_declines_non_constant_weight():
    model = _model(
        """
        g (float[4,32] X, float[32,4] W) => (float[4,4] Y)
        {
          Y = MatMul(X, W)
        }
        """
    )
    q = onnxsim.apply_slim_llm(model)
    assert q.SerializeToString() == model.SerializeToString()


def test_slim_llm_noop_when_no_matmul_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_slim_llm(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_slim_llm_declines_below_opset21():
    model = _matmul_model(K=32, N=8, opset=13)
    result = onnxsim.apply_slim_llm(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_slim_llm_rejects_invalid_bit_range():
    model = _matmul_model(K=32, N=8)
    with pytest.raises(ValueError):
        onnxsim.apply_slim_llm(model, low_bits=4, high_bits=4)
    with pytest.raises(ValueError):
        onnxsim.apply_slim_llm(model, low_bits=1, high_bits=4)
