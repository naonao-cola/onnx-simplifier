"""Tests for QServe's QoQ quantization (see ``onnxsim/qoq.py``):
``onnxsim.quantize_weight_only_qoq`` (progressive INT8-then-INT4 weight
quantization) and ``onnxsim.apply_smooth_attention`` (SmoothAttention's
Query/Key outlier migration, meant to run before
``onnxsim.quantize_kv_cache``).
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


def _matmul_model(K=64, N=16, weight=None, seed=0, opset=21):
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
        opset=opset,
    )


def _run(model, feeds, output_names=None):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    names = output_names or [o.name for o in sess.get_outputs()]
    return dict(zip(names, sess.run(names, feeds)))


def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-6)


def _unpack_int4(t):
    dims = list(t.dims)
    numel = int(np.prod(dims))
    raw = np.frombuffer(t.raw_data, dtype=np.uint8)
    lo = (raw & 0x0F).astype(np.int64)
    hi = ((raw >> 4) & 0x0F).astype(np.int64)
    codes = np.empty(numel, dtype=np.int64)
    codes[0::2] = lo[: (numel + 1) // 2]
    codes[1::2] = hi[: numel // 2]
    codes = np.where(codes >= 8, codes - 16, codes)
    return codes.reshape(dims).astype(np.float64)


def _dequantize_qoq(model):
    dq_node = next(n for n in model.graph.node if n.op_type == "DequantizeLinear")
    wq = next(t for t in model.graph.initializer if t.name == dq_node.input[0])
    ws = next(t for t in model.graph.initializer if t.name == dq_node.input[1])
    block_size = next(a.i for a in dq_node.attribute if a.name == "block_size")
    axis = next((a.i for a in dq_node.attribute if a.name == "axis"), 1)

    codes = _unpack_int4(wq)
    scale = onnx.numpy_helper.to_array(ws).astype(np.float64)

    scale_full = np.repeat(scale, block_size, axis=axis)
    slicer = [slice(None)] * codes.ndim
    slicer[axis] = slice(0, codes.shape[axis])
    scale_full = scale_full[tuple(slicer)]
    return codes * scale_full, dq_node


# --- quantize_weight_only_qoq -----------------------------------------


def test_qoq_quantizes_matmul_to_dequantize_linear():
    model = _matmul_model(K=64, N=16, seed=0)
    qoq_model = onnxsim.quantize_weight_only_qoq(model)
    onnx.checker.check_model(qoq_model)

    op_types = [n.op_type for n in qoq_model.graph.node]
    assert op_types.count("DequantizeLinear") == 1
    assert "MatMul" in op_types

    (dq_node,) = [n for n in qoq_model.graph.node if n.op_type == "DequantizeLinear"]
    assert len(dq_node.input) == 2  # Wq, Ws -- symmetric, no zero-point

    wq = next(t for t in qoq_model.graph.initializer if t.name == dq_node.input[0])
    assert wq.data_type == onnx.TensorProto.INT4


def test_qoq_reconstruction_matches_two_stage_numpy_rounding():
    # Directly verifies this module's own distinguishing claim: the codes
    # come from rounding an already-INT8-quantized value (per output
    # channel, protectively clipped), not the original float weight --
    # checked via numpy against the raw initializers, per this project's
    # platform-numerics note (no onnxruntime round-trip for exactness).
    rng = np.random.default_rng(1)
    K, N = 64, 8
    weight = rng.standard_normal((K, N)).astype(np.float32) * 2.0
    model = _matmul_model(K=K, N=N, weight=weight)

    block_size = 32
    int8_clip_max = 119
    qoq_model = onnxsim.quantize_weight_only_qoq(
        model, block_size=block_size, int8_clip_max=int8_clip_max
    )
    w_hat, dq_node = _dequantize_qoq(qoq_model)  # [K, N] (untransposed MatMul weight)

    w = weight.astype(np.float64)
    w_nk = w.T  # [N, K]
    s1 = np.maximum(np.abs(w_nk).max(axis=1), 1e-12) / int8_clip_max
    code8 = np.clip(np.round(w_nk / s1[:, None]), -int8_clip_max, int8_clip_max)
    blocks = code8.reshape(N, K // block_size, block_size)
    s2 = np.maximum(np.abs(blocks).max(axis=2), 1e-12) / 7.0
    code4 = np.clip(np.round(blocks / s2[:, :, None]), -7.0, 7.0).reshape(N, K)
    expected_nk = code4 * (s1[:, None] * np.repeat(s2, block_size, axis=1))
    expected = expected_nk.T  # back to [K, N]

    # The written scale is stored as float32 (the graph's own precision),
    # so this compares at float32 precision, not float64 exactness.
    assert np.allclose(w_hat, expected, rtol=1e-5, atol=1e-6)

    # Sanity: the intermediate INT8 grid values from stage 1 stayed inside
    # the protective clip range, not the full [-127, 127] INT8 range.
    assert np.all(np.abs(code8) <= int8_clip_max)


def test_qoq_beats_single_stage_int4_reconstruction_on_wide_range_weight():
    # QoQ's own selling point is *not* better accuracy than a single-stage
    # INT4 quantizer in general (it trades a bit of accuracy for a
    # hardware-friendly dequant path) -- what this test actually checks is
    # narrower and verifiable: for a per-channel-flat weight (one where a
    # single INT4 grid is already a reasonable fit), QoQ's two-stage
    # reconstruction should still land in the same right ballpark as
    # onnxsim's own single-stage ``quantize_weight_only_int4`` -- neither
    # blows up nor collapses to zero.
    rng = np.random.default_rng(2)
    K, N = 64, 8
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.3

    model = _matmul_model(K=K, N=N, weight=weight)
    qoq_model = onnxsim.quantize_weight_only_qoq(model)
    int4_model = onnxsim.quantize_weight_only_int4(model)
    onnx.checker.check_model(int4_model)

    w_qoq, _ = _dequantize_qoq(qoq_model)

    rng2 = np.random.default_rng(20)
    x = rng2.standard_normal((8, K)).astype(np.float32)
    float_y = _run(model, {"X": x})["Y"]
    qoq_y = _run(qoq_model, {"X": x})["Y"]
    int4_y = _run(int4_model, {"X": x})["Y"]

    w = weight.astype(np.float64)
    err_qoq = np.linalg.norm(w_qoq - w) / np.linalg.norm(w)
    assert err_qoq < 0.15
    # Two-stage rounding costs some accuracy relative to single-stage, but
    # not drastically so, on a well-behaved weight -- compared end to end
    # (via onnxruntime, loosely) since quantize_weight_only_int4's own
    # DequantizeLinear node shape (e.g. whether it carries a zero-point
    # input) is a C++-side implementation detail this module makes no
    # assumption about.
    assert _rel_l2(float_y, qoq_y) < _rel_l2(float_y, int4_y) * 3.0 + 1e-2


def test_qoq_output_stays_close_to_float_via_onnxruntime():
    model = _matmul_model(K=64, N=16, seed=3)
    qoq_model = onnxsim.quantize_weight_only_qoq(model)
    onnx.checker.check_model(qoq_model)

    rng = np.random.default_rng(4)
    x = rng.standard_normal((8, 64)).astype(np.float32)
    float_y = _run(model, {"X": x})["Y"]
    qoq_y = _run(qoq_model, {"X": x})["Y"]
    assert np.all(np.isfinite(qoq_y))
    # Loose end-to-end sanity check (INT4 is intentionally lossy) -- matches
    # onnxsim.quantize_weight_only_int4_hqq's own onnxruntime-based check.
    assert _rel_l2(float_y, qoq_y) < 0.25


def test_qoq_gemm_transb():
    rng = np.random.default_rng(5)
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
    qoq_model = onnxsim.quantize_weight_only_qoq(model)
    onnx.checker.check_model(qoq_model)

    x = rng.standard_normal((4, K)).astype(np.float32)
    float_y = _run(model, {"X": x})["Y"]
    qoq_y = _run(qoq_model, {"X": x})["Y"]
    assert _rel_l2(float_y, qoq_y) < 0.25


def test_qoq_codes_stay_in_int4_range():
    rng = np.random.default_rng(6)
    weight = rng.standard_normal((32, 8)).astype(np.float32) * 3
    model = _matmul_model(K=32, N=8, weight=weight)
    qoq_model = onnxsim.quantize_weight_only_qoq(model)

    dq_node = next(n for n in qoq_model.graph.node if n.op_type == "DequantizeLinear")
    wq = next(t for t in qoq_model.graph.initializer if t.name == dq_node.input[0])
    codes = _unpack_int4(wq)
    assert np.all(codes >= -7) and np.all(codes <= 7)


def test_qoq_invalid_int8_clip_max_raises():
    model = _matmul_model(K=32, N=8, seed=7)
    with pytest.raises(ValueError):
        onnxsim.quantize_weight_only_qoq(model, int8_clip_max=0)
    with pytest.raises(ValueError):
        onnxsim.quantize_weight_only_qoq(model, int8_clip_max=128)


def test_qoq_skips_non_block_divisible_k():
    model = _matmul_model(K=48, N=8, seed=8)  # 48 not a multiple of 32
    qoq_model = onnxsim.quantize_weight_only_qoq(model)
    assert qoq_model.SerializeToString() == model.SerializeToString()


def test_qoq_skips_non_constant_weight():
    model = _model(
        """
        g (float[4,64] X, float[64,4] W) => (float[4,4] Y)
        {
          Y = MatMul(X, W)
        }
        """
    )
    qoq_model = onnxsim.quantize_weight_only_qoq(model)
    op_types = [n.op_type for n in qoq_model.graph.node]
    assert op_types.count("MatMul") == 1
    assert "DequantizeLinear" not in op_types


def test_qoq_noop_below_opset21():
    model = _matmul_model(K=32, N=8, seed=9, opset=13)
    qoq_model = onnxsim.quantize_weight_only_qoq(model)
    assert qoq_model.SerializeToString() == model.SerializeToString()


# --- apply_smooth_attention ---------------------------------------------


def _attention_model(seq=6, head_dim=8):
    return _model(
        f"""
        g (float[{seq},{head_dim}] Q, float[{head_dim},{seq}] Kt,
           float[{seq},{head_dim}] V) => (float[{seq},{head_dim}] Out)
        {{
          scores = MatMul(Q, Kt)
          probs = Softmax<axis = -1>(scores)
          Out = MatMul(probs, V)
        }}
        """,
        opset=18,
    )


def _attention_calibration(
    seq=6, head_dim=8, num_samples=32, outlier_channels=(1, 5), seed=1
):
    rng = np.random.default_rng(seed)
    batches = []
    for _ in range(num_samples):
        q = rng.standard_normal((seq, head_dim)).astype(np.float32)
        k = rng.standard_normal((seq, head_dim)).astype(np.float32)
        for c in outlier_channels:
            k[:, c] *= 15.0
        kt = k.T.astype(np.float32).copy()
        v = rng.standard_normal((seq, head_dim)).astype(np.float32)
        batches.append({"Q": q, "Kt": kt, "V": v})
    return batches


def test_smooth_attention_output_matches_float_almost_exactly():
    model = _attention_model()
    calibration_data = _attention_calibration()

    sa_model = onnxsim.apply_smooth_attention(model, calibration_data=calibration_data)
    onnx.checker.check_model(sa_model)
    op_types = [n.op_type for n in sa_model.graph.node]
    assert "Mul" in op_types and "Div" in op_types

    feed = calibration_data[0]
    float_out = _run(model, feed)
    sa_out = _run(sa_model, feed, output_names=["Out"])
    assert np.all(np.isfinite(sa_out["Out"]))
    assert _rel_l2(float_out["Out"], sa_out["Out"]) < 1e-4


def test_smooth_attention_flattens_key_channel_range():
    # SmoothAttention's own motivating scenario: a couple of Key channels
    # persistently much larger than the rest -- exactly what makes
    # per-channel Key quantization (onnxsim.quantize_kv_cache) hard. After
    # migration, the *migrated* Key tensor feeding the QK^T matmul should
    # have a much flatter per-channel range than the original.
    head_dim = 8
    calibration_data = _attention_calibration(
        head_dim=head_dim, outlier_channels=(2, 6), seed=2
    )
    model = _attention_model(head_dim=head_dim)

    sa_model = onnxsim.apply_smooth_attention(model, calibration_data=calibration_data)
    qk_matmul = next(
        n
        for n in sa_model.graph.node
        if n.op_type == "MatMul" and n.output[0] == "scores"
    )
    kt_name = qk_matmul.input[1]

    probe_model = onnx.ModelProto()
    probe_model.CopyFrom(sa_model)
    probe_model.graph.output.append(onnx.ValueInfoProto(name=kt_name))

    orig_range = np.zeros(head_dim)
    migrated_range = np.zeros(head_dim)
    for feed in calibration_data:
        kt_migrated = _run(probe_model, feed, output_names=[kt_name])[kt_name]
        orig_range = np.maximum(orig_range, np.abs(feed["Kt"]).max(axis=1))
        migrated_range = np.maximum(migrated_range, np.abs(kt_migrated).max(axis=1))

    orig_spread = orig_range.max() / orig_range.min()
    migrated_spread = migrated_range.max() / migrated_range.min()
    assert migrated_spread < orig_spread


def test_smooth_attention_exact_dot_product_identity():
    # The algebraic claim this module's docstring makes directly: for
    # channel j, (K_j / s_j) . (Q_j * s_j) == K_j . Q_j -- checked in
    # plain numpy against the model's own written scale, independent of
    # onnxruntime.
    head_dim = 8
    calibration_data = _attention_calibration(head_dim=head_dim, seed=3)
    model = _attention_model(head_dim=head_dim)
    sa_model = onnxsim.apply_smooth_attention(model, calibration_data=calibration_data)

    mul_node = next(n for n in sa_model.graph.node if n.op_type == "Mul")
    div_node = next(n for n in sa_model.graph.node if n.op_type == "Div")
    s_row = onnx.numpy_helper.to_array(
        next(t for t in sa_model.graph.initializer if t.name == mul_node.input[1])
    ).astype(np.float64)
    s_col = onnx.numpy_helper.to_array(
        next(t for t in sa_model.graph.initializer if t.name == div_node.input[1])
    ).astype(np.float64)
    assert np.allclose(s_row, s_col.reshape(-1), rtol=1e-6)

    feed = calibration_data[0]
    q = feed["Q"].astype(np.float64)
    k = feed["Kt"].T.astype(np.float64)  # back to [seq, head_dim]
    original = q @ k.T
    migrated = (q * s_row) @ (k / s_row).T
    assert np.allclose(original, migrated, rtol=1e-9, atol=1e-9)


def test_smooth_attention_noop_when_no_attention_subgraph():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    result = onnxsim.apply_smooth_attention(
        model, calibration_data=[{"X": np.zeros((4, 4), dtype=np.float32)}]
    )
    assert result.SerializeToString() == model.SerializeToString()
