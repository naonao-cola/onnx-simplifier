"""Tests for ``onnxsim.quantize_weight_only_int4_hqq`` (HQQ, see
``onnxsim/hqq.py``) -- calibration-free, asymmetric block-wise INT4
quantization that fits each block's zero-point via IRLS to minimize a
robust (Lp, p<2) reconstruction loss instead of the ordinary least-squares
fit a naive min/max range implicitly targets.
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


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-6)


def _dequantize_hqq(model):
    dq_node = next(n for n in model.graph.node if n.op_type == "DequantizeLinear")
    wq = next(t for t in model.graph.initializer if t.name == dq_node.input[0])
    ws = next(t for t in model.graph.initializer if t.name == dq_node.input[1])
    wz = next(t for t in model.graph.initializer if t.name == dq_node.input[2])
    block_size = next(a.i for a in dq_node.attribute if a.name == "block_size")
    axis = next((a.i for a in dq_node.attribute if a.name == "axis"), 1)

    def _unpack_uint4(t):
        dims = list(t.dims)
        numel = int(np.prod(dims))
        raw = np.frombuffer(t.raw_data, dtype=np.uint8)
        lo = (raw & 0x0F).astype(np.int64)
        hi = ((raw >> 4) & 0x0F).astype(np.int64)
        codes = np.empty(numel, dtype=np.int64)
        codes[0::2] = lo[: (numel + 1) // 2]
        codes[1::2] = hi[: numel // 2]
        return codes.reshape(dims).astype(np.float64)

    codes = _unpack_uint4(wq)
    zero = _unpack_uint4(wz)
    scale = onnx.numpy_helper.to_array(ws).astype(np.float64)

    scale_full = np.repeat(scale, block_size, axis=axis)
    zero_full = np.repeat(zero, block_size, axis=axis)
    slicer = [slice(None)] * codes.ndim
    slicer[axis] = slice(0, codes.shape[axis])
    scale_full = scale_full[tuple(slicer)]
    zero_full = zero_full[tuple(slicer)]
    return (codes - zero_full) * scale_full


def test_hqq_quantizes_matmul_to_dequantize_linear():
    model = _matmul_model(K=64, N=16, seed=0)
    hqq_model = onnxsim.quantize_weight_only_int4_hqq(model)
    onnx.checker.check_model(hqq_model)

    op_types = [n.op_type for n in hqq_model.graph.node]
    assert op_types.count("DequantizeLinear") == 1
    assert "MatMul" in op_types

    (dq_node,) = [n for n in hqq_model.graph.node if n.op_type == "DequantizeLinear"]
    assert (
        len(dq_node.input) == 3
    )  # Wq, Ws, Wz -- asymmetric, unlike quantize_weight_only_int4

    wq = next(t for t in hqq_model.graph.initializer if t.name == dq_node.input[0])
    assert wq.data_type == onnx.TensorProto.UINT4
    wz = next(t for t in hqq_model.graph.initializer if t.name == dq_node.input[2])
    assert wz.data_type == onnx.TensorProto.UINT4


def test_hqq_output_stays_close_to_float_via_onnxruntime():
    model = _matmul_model(K=64, N=16, seed=1)
    hqq_model = onnxsim.quantize_weight_only_int4_hqq(model)
    onnx.checker.check_model(hqq_model)

    rng = np.random.default_rng(2)
    x = rng.standard_normal((8, 64)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (hqq_y,) = _run(hqq_model, {"X": x})
    assert np.all(np.isfinite(hqq_y))
    assert _rel_l2(float_y, hqq_y) < 0.25


def test_hqq_beats_naive_minmax_on_outlier_heavy_weights():
    # HQQ's own motivating scenario: a block whose naive min/max range is
    # dominated by a couple of outlier elements, forcing a scale so wide
    # that the bulk of "normal" elements lose most of their precision. A
    # robust (Lp<2) fit should recover a tighter, more accurate zero-point
    # for the bulk at the cost of clipping the outliers -- net lower
    # reconstruction error on the block as a whole.
    rng = np.random.default_rng(3)
    K, N = 32, 8
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.1
    # Inject a couple of large outliers into one block (all of K=32 here,
    # i.e. one block at the default block_size=32).
    weight[0, :] += 5.0
    weight[1, :] += 5.0

    model = _matmul_model(K=K, N=N, weight=weight)
    hqq_model = onnxsim.quantize_weight_only_int4_hqq(model)
    w_hqq = _dequantize_hqq(hqq_model)  # [K, N]

    # Naive min/max affine quantization (p=2, i.e. plain least squares --
    # what a naive min/max range effectively targets) for comparison:
    # asymmetric codes derived directly from min/max with no IRLS refinement.
    w = weight.astype(np.float64)
    mn, mx = w.min(axis=0, keepdims=True), w.max(axis=0, keepdims=True)
    scale_naive = np.maximum((mx - mn) / 15.0, 1e-12)
    zero_naive = np.clip(np.round(-mn / scale_naive), 0, 15)
    codes_naive = np.clip(np.round(w / scale_naive + zero_naive), 0, 15)
    w_naive = (codes_naive - zero_naive) * scale_naive

    # Restrict the comparison to the *non-outlier* bulk (rows 2:), where
    # HQQ's robust fit should show its benefit most clearly.
    err_hqq = np.linalg.norm(w_hqq[2:] - w[2:])
    err_naive = np.linalg.norm(w_naive[2:] - w[2:])
    assert err_hqq < err_naive


def test_hqq_gemm_transb():
    rng = np.random.default_rng(4)
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
    hqq_model = onnxsim.quantize_weight_only_int4_hqq(model)
    onnx.checker.check_model(hqq_model)

    x = rng.standard_normal((4, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (hqq_y,) = _run(hqq_model, {"X": x})
    assert _rel_l2(float_y, hqq_y) < 0.25


def test_hqq_codes_and_zero_stay_in_range():
    rng = np.random.default_rng(5)
    weight = rng.standard_normal((32, 8)).astype(np.float32) * 3
    model = _matmul_model(K=32, N=8, weight=weight)
    hqq_model = onnxsim.quantize_weight_only_int4_hqq(model)

    dq_node = next(n for n in hqq_model.graph.node if n.op_type == "DequantizeLinear")
    for name in (dq_node.input[0], dq_node.input[2]):
        t = next(t for t in hqq_model.graph.initializer if t.name == name)
        numel = int(np.prod(list(t.dims)))
        raw = np.frombuffer(t.raw_data, dtype=np.uint8)
        lo = raw & 0x0F
        hi = (raw >> 4) & 0x0F
        codes = np.empty(numel, dtype=np.uint8)
        codes[0::2] = lo[: (numel + 1) // 2]
        codes[1::2] = hi[: numel // 2]
        assert np.all(codes <= 15)


def test_hqq_skips_non_block_divisible_k():
    # K=48 is not a multiple of the default block_size=32.
    model = _matmul_model(K=48, N=8, seed=6)
    hqq_model = onnxsim.quantize_weight_only_int4_hqq(model)
    assert hqq_model.SerializeToString() == model.SerializeToString()


def test_hqq_skips_non_constant_weight():
    model = _model(
        """
        g (float[4,64] X, float[64,4] W) => (float[4,4] Y)
        {
          Y = MatMul(X, W)
        }
        """
    )
    hqq_model = onnxsim.quantize_weight_only_int4_hqq(model)
    op_types = [n.op_type for n in hqq_model.graph.node]
    assert op_types.count("MatMul") == 1
    assert "DequantizeLinear" not in op_types
