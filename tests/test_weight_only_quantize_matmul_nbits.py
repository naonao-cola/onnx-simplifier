"""Tests for ``onnxsim.quantize_weight_only_matmul_nbits`` (the
``weight_only_quantize_matmul_nbits`` C++ pass) -- weight-only INT4
quantization into ONNX Runtime's own ``com.microsoft::MatMulNBits`` contrib
op, a *vendor-specific* counterpart to
``onnxsim.quantize_weight_only_int4``'s portable standard-ONNX output (see
``passes/weight_only_quantize_matmul_nbits.h``).

Each model is built directly with ``onnx.parser`` (no torch dependency),
quantized, and then actually run through ONNX Runtime -- both before and
after quantization -- so these tests double as a minimal end-to-end
simplify/quantize/deploy check: the quantized graph must load and execute
under a real inference engine (which, being a com.microsoft contrib op,
only ONNX Runtime itself can do), and its outputs must stay close to the
float baseline. A dedicated test also decodes ``B``'s raw bit-packed bytes
directly and checks them against a from-scratch reference dequantization,
independent of ONNX Runtime -- MatMulNBits' packing format (low-nibble-
first, zero-point-8 symmetric codes) is intricate enough that "the model
runs and the output is close" alone would not catch every possible packing
mistake (e.g. one that happens to preserve aggregate accuracy).
"""

import collections

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim

# A bare ``import onnxruntime`` would fail collection (not skip the test) on
# platforms onnxruntime doesn't ship wheels for (e.g. s390x).
ort = pytest.importorskip("onnxruntime")


def _model(body, initializer=(), opset=17):
    model = parser.parse_model(
        f"""
        <
          ir_version: 8,
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


def _op_counts(model):
    return collections.Counter((n.op_type, n.domain) for n in model.graph.node)


def _assert_close(float_outputs, quant_outputs, rel_l2_tol=0.25):
    # Same lossiness class, and same tolerance rationale, as
    # test_weight_only_quantize_int4.py's identically-named helper: 16
    # levels per block, no calibration, round-to-nearest on random Gaussian
    # weights lands in the ~0.07-0.16 range on its own.
    for f, q in zip(float_outputs, quant_outputs):
        f = np.asarray(f, dtype=np.float64).ravel()
        q = np.asarray(q, dtype=np.float64).ravel()
        assert np.all(np.isfinite(q))
        rel_l2 = np.linalg.norm(f - q) / max(np.linalg.norm(f), 1e-6)
        assert rel_l2 < rel_l2_tol, f"relative L2 error too large: {rel_l2:.4f}"


def _find(model, pred):
    return next(t for t in model.graph.initializer if pred(t))


def test_quantize_matmul():
    rng = np.random.default_rng(0)
    K, N = 64, 16
    weight = _f32(rng.standard_normal((K, N)) * 0.5, "W")
    model = _model(
        f"""
        g (float[4,{K}] X) => (float[4,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[weight],
    )

    quant = onnxsim.quantize_weight_only_matmul_nbits(model)
    onnx.checker.check_model(quant)
    ops = _op_counts(quant)
    assert ops[("MatMulNBits", "com.microsoft")] == 1
    assert ("MatMul", "") not in ops
    assert any(o.domain == "com.microsoft" for o in quant.opset_import)

    b_init = _find(quant, lambda t: t.data_type == onnx.TensorProto.UINT8)
    assert list(b_init.dims) == [N, 2, 16]  # k_blocks=64/32=2, blob_size=32/2=16

    x = rng.standard_normal((4, K)).astype(np.float32)
    _assert_close(_run(model, {"X": x}), _run(quant, {"X": x}))


def test_quantize_gemm_transb_with_bias():
    # PyTorch's nn.Linear layout: weight is [out_features, in_features], i.e.
    # [N, K], exported as Gemm(X, W, B, transB=1) -- the common real-world case.
    rng = np.random.default_rng(1)
    K, N = 96, 12
    weight = _f32(rng.standard_normal((N, K)) * 0.5, "W")
    bias = _f32(rng.standard_normal(N), "B")
    model = _model(
        f"""
        g (float[3,{K}] X) => (float[3,{N}] Y)
        {{
          Y = Gemm<transB = 1>(X, W, B)
        }}
        """,
        initializer=[weight, bias],
    )

    quant = onnxsim.quantize_weight_only_matmul_nbits(model)
    onnx.checker.check_model(quant)
    (nbits_node,) = [n for n in quant.graph.node if n.op_type == "MatMulNBits"]
    # A, B, scales, zero_points (skipped), g_idx (skipped), bias.
    assert len(nbits_node.input) == 6
    assert nbits_node.input[0] == "X"
    assert nbits_node.input[1] and nbits_node.input[1] not in ("X", "B")
    assert nbits_node.input[2] and nbits_node.input[2] not in ("X", "B")
    assert nbits_node.input[3] == ""
    assert nbits_node.input[4] == ""
    assert nbits_node.input[5] == "B"

    x = rng.standard_normal((3, K)).astype(np.float32)
    _assert_close(_run(model, {"X": x}), _run(quant, {"X": x}))


def test_quantize_scale_shape_matches_block_count():
    # K=64 with the pass's block_size=32 gives k_blocks=2; MatMulNBits'
    # scales shape is (N, k_blocks) -- output channel first, unlike
    # quantize_weight_only_int4's DequantizeLinear-oriented (k_blocks, N).
    rng = np.random.default_rng(2)
    K, N = 64, 8
    weight = _f32(rng.standard_normal((K, N)) * 0.5, "W")
    model = _model(
        f"""
        g (float[1,{K}] X) => (float[1,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[weight],
    )

    quant = onnxsim.quantize_weight_only_matmul_nbits(model)
    scale_init = _find(
        quant, lambda t: t.data_type == onnx.TensorProto.FLOAT and t.name != "W"
    )
    assert list(scale_init.dims) == [N, 2]


def test_quantize_handles_k_not_divisible_by_block_size():
    # K=48 is not a multiple of the pass's block_size=32 -- unlike
    # quantize_weight_only_int4 (which declines this case entirely),
    # MatMulNBits' own k_blocks = ceil(K / block_size) already defines a
    # ragged last block, so this quantizes it instead of skipping.
    rng = np.random.default_rng(3)
    K, N = 48, 8
    weight = _f32(rng.standard_normal((K, N)) * 0.5, "W")
    model = _model(
        f"""
        g (float[1,{K}] X) => (float[1,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[weight],
    )

    quant = onnxsim.quantize_weight_only_matmul_nbits(model)
    onnx.checker.check_model(quant)
    ops = _op_counts(quant)
    assert ops[("MatMulNBits", "com.microsoft")] == 1
    b_init = _find(quant, lambda t: t.data_type == onnx.TensorProto.UINT8)
    assert list(b_init.dims) == [N, 2, 16]  # k_blocks = ceil(48/32) = 2

    x = rng.standard_normal((1, K)).astype(np.float32)
    _assert_close(_run(model, {"X": x}), _run(quant, {"X": x}))


def test_quantize_works_at_low_opset():
    # Unlike quantize_weight_only_int4 (needs opset >= 21 for ONNX's own
    # native INT4 tensor type and DequantizeLinear's block_size attribute),
    # MatMulNBits is a self-contained contrib op with no minimum standard
    # opset of its own.
    rng = np.random.default_rng(4)
    K, N = 32, 4
    weight = _f32(rng.standard_normal((K, N)) * 0.5, "W")
    model = _model(
        f"""
        g (float[1,{K}] X) => (float[1,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[weight],
        opset=11,
    )

    quant = onnxsim.quantize_weight_only_matmul_nbits(model)
    onnx.checker.check_model(quant)
    assert _op_counts(quant)[("MatMulNBits", "com.microsoft")] == 1

    x = rng.standard_normal((1, K)).astype(np.float32)
    _assert_close(_run(model, {"X": x}), _run(quant, {"X": x}))


def test_quantize_skips_non_constant_weight():
    model = _model(
        """
        g (float[4,64] X, float[64,4] W) => (float[4,4] Y)
        {
          Y = MatMul(X, W)
        }
        """
    )
    quant = onnxsim.quantize_weight_only_matmul_nbits(model)
    assert _op_counts(quant)[("MatMul", "")] == 1


def test_quantize_bit_packing_matches_reference_dequantization():
    # Independent of ONNX Runtime: decode B's raw bit-packed bytes by hand
    # (low nibble first, code - 8 per MatMulNBits' documented default zero
    # point) and check the result matches the float weight to within one
    # quantization step -- catches a packing-order or zero-point mistake
    # that happened to still produce a plausible-looking aggregate error.
    rng = np.random.default_rng(5)
    K, N = 40, 3  # a ragged last block (40 = 32 + 8)
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.3
    model = _model(
        f"""
        g (float[1,{K}] X) => (float[1,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f32(weight, "W")],
    )
    quant = onnxsim.quantize_weight_only_matmul_nbits(model)

    b_init = _find(quant, lambda t: t.data_type == onnx.TensorProto.UINT8)
    scale_init = _find(
        quant, lambda t: t.data_type == onnx.TensorProto.FLOAT and t.name != "W"
    )
    b_packed = onnx.numpy_helper.to_array(b_init)  # [N, k_blocks, blob_size]
    scales = onnx.numpy_helper.to_array(scale_init)  # [N, k_blocks]
    _, k_blocks, blob_size = b_packed.shape
    block_size = 32

    dequant = np.zeros((N, K), dtype=np.float32)
    for n in range(N):
        for kb in range(k_blocks):
            s = scales[n, kb]
            k0 = kb * block_size
            for j in range(blob_size):
                byte = int(b_packed[n, kb, j])
                lo, hi = byte & 0x0F, (byte >> 4) & 0x0F
                if k0 + 2 * j < K:
                    dequant[n, k0 + 2 * j] = (lo - 8) * s
                if k0 + 2 * j + 1 < K:
                    dequant[n, k0 + 2 * j + 1] = (hi - 8) * s

    weight_nk = weight.T  # [K, N] -> [N, K]
    # Every element's quantization error must be within half a code step
    # (the maximum possible round-to-nearest error), not just close on
    # aggregate.
    for n in range(N):
        for kb in range(k_blocks):
            k0, k1 = kb * block_size, min(K, (kb + 1) * block_size)
            step = scales[n, kb]
            err = np.abs(weight_nk[n, k0:k1] - dequant[n, k0:k1])
            assert np.all(err <= step / 2 + 1e-6)
