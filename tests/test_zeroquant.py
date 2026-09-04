"""Tests for ``onnxsim.apply_zeroquant`` -- see ``onnxsim/zeroquant.py`` for
the technique (group-wise INT8 weight quantization paired with per-token
dynamic INT8 activation quantization, executed as real ``int8 x int8``
integer matmuls via a grouped ``MatMulInteger`` pipeline).
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.zeroquant import _quantize_weight_groupwise_int8

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


def _matmul_model(K=64, N=8, weight=None, seed=0, opset=21):
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
        initializer=[_f32(weight, "W")],
        opset=opset,
    )


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-6)


def _mean_row_rel_l2(a, b):
    # Mean of each ROW's own relative L2 error, rather than one relative
    # error over the whole flattened array. On data with a wide spread of
    # per-token magnitudes (the whole point of testing per-token
    # granularity), a single flattened rel_l2 is dominated by the
    # largest-magnitude rows and can hide a baseline that reconstructs
    # small-magnitude tokens as near-total noise -- exactly the failure
    # mode a per-tensor activation scale (onnxsim.quantize_dynamic's own
    # granularity) has on such data.
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    row_norms = np.linalg.norm(a, axis=-1)
    row_errs = np.linalg.norm(a - b, axis=-1)
    return float(np.mean(row_errs / np.maximum(row_norms, 1e-6)))


def test_quantize_weight_groupwise_int8_matches_hand_computed_values():
    # K=4, block_size=2, N=1: two groups of 2 elements each, one scale per
    # group. Values chosen so the per-group max-abs (and hence the exact
    # INT8 codes) can be checked by hand, not just round-tripped.
    w = np.array([[1.0], [2.0], [-100.0], [50.0]], dtype=np.float64)
    wq, scale = _quantize_weight_groupwise_int8(w, block_size=2, epsilon=1e-12)

    assert wq.shape == (4, 1)
    assert scale.shape == (2, 1)

    group0_scale = 2.0 / 127.0  # max(|1|, |2|) / 127
    group1_scale = 100.0 / 127.0  # max(|-100|, |50|) / 127
    assert scale[0, 0] == pytest.approx(group0_scale, rel=1e-6)
    assert scale[1, 0] == pytest.approx(group1_scale, rel=1e-6)

    # Exact codes: round(value / group_scale), clipped to [-127, 127].
    expected_codes = np.array(
        [
            round(1.0 / group0_scale),
            round(2.0 / group0_scale),
            round(-100.0 / group1_scale),
            round(50.0 / group1_scale),
        ]
    )
    np.testing.assert_array_equal(wq.reshape(-1), expected_codes)


def test_zeroquant_uses_real_integer_matmul_not_simulated_roundtrip():
    model = _matmul_model(K=64, N=8, seed=0)
    q = onnxsim.apply_zeroquant(model, block_size=32)
    onnx.checker.check_model(q)

    op_types = [n.op_type for n in q.graph.node]
    # Real int8 x int8 compute -- a grouped MatMulInteger pipeline, not a
    # quantize-then-immediately-dequantize float simulation (which is what
    # onnxsim.apply_quarot/apply_duquant/apply_attention_quantization do
    # for their own per-token activation scales).
    assert op_types.count("MatMulInteger") == 2  # K=64, block_size=32 -> 2 groups
    assert "Split" in op_types
    assert "Sum" in op_types
    assert "DequantizeLinear" not in op_types
    assert "QuantizeLinear" not in op_types

    initializer_names = {t.name for t in q.graph.initializer}
    wq_groups = [n for n in initializer_names if "_wq_group" in n]
    assert len(wq_groups) == 2
    for name in wq_groups:
        t = next(t for t in q.graph.initializer if t.name == name)
        assert t.data_type == onnx.TensorProto.INT8


def test_zeroquant_output_stays_close_to_float_via_onnxruntime():
    model = _matmul_model(K=64, N=8, seed=1)
    q = onnxsim.apply_zeroquant(model, block_size=32)

    rng = np.random.default_rng(2)
    x = rng.standard_normal((8, 64)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.2


def _structured_outlier_weight(k, n, block_size, rng, small=0.01, large=8.0):
    # Every output column's reduction dimension alternates: block 0 tiny
    # magnitude, block 1 huge magnitude, block 2 tiny, ... A single
    # per-output-channel scale (onnxsim.quantize_dynamic's own weight
    # granularity) must be set by the huge blocks, crushing the tiny
    # blocks' resolution to a handful of INT8 codes; a per-group scale
    # (this module's weight granularity) resolves every block on its own
    # terms instead.
    num_groups = k // block_size
    w = rng.standard_normal((k, n)).astype(np.float64)
    for g in range(num_groups):
        magnitude = large if g % 2 == 1 else small
        w[g * block_size : (g + 1) * block_size, :] *= magnitude
    return w.astype(np.float32)


def _structured_outlier_tokens(num_tokens, k, rng, small=0.02, large=10.0):
    # Alternating tiny-magnitude / huge-magnitude rows (tokens). A single
    # per-tensor activation scale (DynamicQuantizeLinear, what
    # onnxsim.quantize_dynamic uses) is set by the huge tokens, crushing
    # the tiny tokens' resolution; a per-token scale (this module's
    # activation granularity) resolves every row on its own terms instead.
    x = rng.standard_normal((num_tokens, k)).astype(np.float64)
    for i in range(num_tokens):
        x[i] *= large if i % 2 == 1 else small
    return x.astype(np.float32)


def test_zeroquant_reconstruction_error_beats_per_tensor_activation_baseline():
    # Real reconstruction-error comparison on data with genuine per-group
    # (weight) and per-token (activation) structure to exploit: ZeroQuant's
    # fine-grained combined scheme vs. onnxsim.quantize_dynamic's coarser
    # W8A8 baseline (per-output-channel weight scale, per-TENSOR dynamic
    # activation scale -- see that module's own docstring).
    K, N, block_size = 64, 8, 32
    rng = np.random.default_rng(3)
    weight = _structured_outlier_weight(K, N, block_size, rng)
    model = _matmul_model(K=K, N=N, weight=weight, seed=0)

    zq = onnxsim.apply_zeroquant(model, block_size=block_size)
    baseline = onnxsim.quantize_dynamic(model)
    onnx.checker.check_model(zq)
    onnx.checker.check_model(baseline)

    x = _structured_outlier_tokens(16, K, np.random.default_rng(4))
    (float_y,) = _run(model, {"X": x})
    (zq_y,) = _run(zq, {"X": x})
    (baseline_y,) = _run(baseline, {"X": x})

    zq_err = _mean_row_rel_l2(float_y, zq_y)
    baseline_err = _mean_row_rel_l2(float_y, baseline_y)
    assert np.all(np.isfinite(zq_y))
    # Loose relative margin (not a fixed absolute threshold) so this stays
    # robust to platform-specific float32 MatMul reduction-order
    # differences between CI runners, while still requiring a real,
    # substantial improvement rather than a marginal one. In practice the
    # baseline's single per-tensor activation scale is dominated by this
    # data's large-magnitude tokens and reconstructs the small-magnitude
    # tokens as near-total noise (per-row error ~1.0), while ZeroQuant's
    # per-token scale keeps every token well-resolved -- a large enough gap
    # that even a generous margin leaves no ambiguity.
    assert zq_err < baseline_err * 0.3


def test_zeroquant_gemm_transb_with_bias():
    rng = np.random.default_rng(5)
    K, N, block_size = 64, 8, 32
    weight = rng.standard_normal((N, K)).astype(np.float32) * 0.5
    bias = rng.standard_normal((N,)).astype(np.float32) * 0.1
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm<transB=1>(X, W, B)
        }}
        """,
        initializer=[_f32(weight, "W"), _f32(bias, "B")],
    )
    q = onnxsim.apply_zeroquant(model, block_size=block_size)
    onnx.checker.check_model(q)
    assert "Add" in [n.op_type for n in q.graph.node]

    x = rng.standard_normal((4, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert _rel_l2(float_y, q_y) < 0.2


def test_zeroquant_3d_activation_round_trips_through_batch_and_seq_dims():
    # X has leading batch AND sequence dims (not just a flat [tokens, K]) --
    # exercises the module's generic "flatten leading dims, reshape back"
    # handling rather than only the common already-2-D case.
    K, N, block_size = 64, 8, 32
    rng = np.random.default_rng(6)
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _model(
        f"""
        g (float[batch,seq,{K}] X) => (float[batch,seq,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f32(weight, "W")],
    )
    q = onnxsim.apply_zeroquant(model, block_size=block_size)
    onnx.checker.check_model(q)

    x = rng.standard_normal((2, 5, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert q_y.shape == float_y.shape
    assert _rel_l2(float_y, q_y) < 0.2


def test_zeroquant_declines_when_k_not_divisible_by_block_size():
    model = _matmul_model(K=48, N=4, seed=7)  # 48 is not a multiple of 32
    q = onnxsim.apply_zeroquant(model, block_size=32)
    assert q.SerializeToString() == model.SerializeToString()


def test_zeroquant_declines_non_constant_weight():
    model = _model(
        """
        g (float[4,64] X, float[64,4] W) => (float[4,4] Y)
        {
          Y = MatMul(X, W)
        }
        """
    )
    q = onnxsim.apply_zeroquant(model)
    assert q.SerializeToString() == model.SerializeToString()


def test_zeroquant_noop_when_no_matmul_present():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_zeroquant(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_zeroquant_declines_below_opset18():
    model = _matmul_model(K=64, N=8, opset=17)
    result = onnxsim.apply_zeroquant(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_zeroquant_declines_non_positive_block_size():
    model = _matmul_model(K=64, N=8)
    result = onnxsim.apply_zeroquant(model, block_size=0)
    assert result.SerializeToString() == model.SerializeToString()
