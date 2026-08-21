"""Tests for ``onnxsim.quantize_ternary`` (the
``dynamic_quantize_ternary_matmul`` C++ pass).

Each model is built directly with ``onnx.helper`` (no torch dependency,
unlike a full BitNet b1.58 export), quantized, and then actually run through
ONNX Runtime -- both before and after quantization -- so these tests double
as a minimal end-to-end simplify/quantize/deploy check: the quantized graph
must load and execute under a real inference engine, and its outputs must
stay close to the float baseline. Since the ternary weight encoding is
lossless (only the activation is quantized), the tolerance here is tighter
than ``quantize_dynamic``'s own tests.
"""

import collections

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper
import pytest

import onnxsim

# A bare ``import onnxruntime`` would fail collection (not skip the test) on
# platforms onnxruntime doesn't ship wheels for (e.g. s390x).
ort = pytest.importorskip("onnxruntime")


def _model(nodes, inputs, outputs, initializer, opset=13):
    graph = onnx.helper.make_graph(nodes, "g", inputs, outputs, initializer)
    # Pin a low IR version so the model loads under older onnxruntime builds
    # (which cap at IR version 11), matching test_fusion_patterns.py.
    return onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", opset)], ir_version=10
    )


def _vi(name, shape):
    return onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _ternary_weight(rng, shape, scale=0.3):
    """A [K, N] (or [N, K], caller's choice) weight of {-scale, 0, +scale}."""
    return (rng.choice([-1, 0, 1], size=shape).astype(np.float32)) * np.float32(scale)


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _op_counts(model):
    return collections.Counter(n.op_type for n in model.graph.node)


def _assert_close(float_outputs, quant_outputs, rel_l2_tol=0.05):
    # The weight encoding is lossless here (unlike quantize_dynamic's rounded
    # full-range one), so only the activation's int8 quantization contributes
    # error -- a tighter tolerance than test_dynamic_quantize_matmul.py's is
    # appropriate, but per-element bounds are still flaky at rounding
    # boundaries for the same reason documented there, so this checks
    # aggregate relative L2 error too.
    for f, q in zip(float_outputs, quant_outputs):
        f = np.asarray(f, dtype=np.float64).ravel()
        q = np.asarray(q, dtype=np.float64).ravel()
        assert np.all(np.isfinite(q))
        rel_l2 = np.linalg.norm(f - q) / max(np.linalg.norm(f), 1e-6)
        assert rel_l2 < rel_l2_tol, f"relative L2 error too large: {rel_l2:.4f}"


def test_quantize_ternary_matmul():
    rng = np.random.default_rng(0)
    K, N = 32, 16
    weight = _f32(_ternary_weight(rng, (K, N)), "W")
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    model = _model(nodes, [_vi("X", [4, K])], [_vi("Y", [4, N])], [weight])

    quant = onnxsim.quantize_ternary(model)
    onnx.checker.check_model(quant)
    ops = _op_counts(quant)
    assert ops["MatMul"] == 0
    assert ops["DynamicQuantizeLinear"] == 1
    assert ops["MatMulInteger"] == 1
    assert ops["Cast"] == 1
    assert ops["Mul"] == 2

    x = rng.standard_normal((4, K)).astype(np.float32)
    _assert_close(_run(model, {"X": x}), _run(quant, {"X": x}))


def test_quantize_ternary_gemm_transb_with_bias():
    # PyTorch's nn.Linear layout: weight is [out_features, in_features], i.e.
    # [N, K], exported as Gemm(X, W, B, transB=1) -- BitLinear's actual shape.
    rng = np.random.default_rng(1)
    K, N = 24, 12
    weight = _f32(_ternary_weight(rng, (N, K)), "W")
    bias = _f32(rng.standard_normal(N), "B")
    nodes = [onnx.helper.make_node("Gemm", ["X", "W", "B"], ["Y"], transB=1)]
    model = _model(nodes, [_vi("X", [3, K])], [_vi("Y", [3, N])], [weight, bias])

    quant = onnxsim.quantize_ternary(model)
    onnx.checker.check_model(quant)
    ops = _op_counts(quant)
    assert ops["Gemm"] == 0
    assert ops["DynamicQuantizeLinear"] == 1
    assert ops["MatMulInteger"] == 1
    # The bias is added back, unquantized, after dequantization.
    assert ops["Add"] == 1

    x = rng.standard_normal((3, K)).astype(np.float32)
    _assert_close(_run(model, {"X": x}), _run(quant, {"X": x}))


def test_quantize_ternary_weight_codes_are_lossless():
    # Unlike quantize_dynamic's rounded full-range weight encoding, the
    # ternary pass must reproduce the exact {-1, 0, 1} codes -- this is the
    # whole point of detecting the structure instead of just quantizing.
    rng = np.random.default_rng(2)
    K, N = 16, 4
    scales = np.array([0.1, 0.2, 0.05, 0.4], dtype=np.float32)
    codes = rng.choice([-1, 0, 1], size=(K, N)).astype(np.float32)
    weight = _f32(codes * scales[np.newaxis, :], "W")
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    model = _model(nodes, [_vi("X", [1, K])], [_vi("Y", [1, N])], [weight])

    quant = onnxsim.quantize_ternary(model)
    initializers = {init.name: init for init in quant.graph.initializer}
    q_init = next(
        onnx.numpy_helper.to_array(t)
        for name, t in initializers.items()
        if onnx.numpy_helper.to_array(t).dtype == np.int8
    )
    np.testing.assert_array_equal(q_init, codes.astype(np.int8))


def test_quantize_ternary_skips_non_ternary_weight():
    # Ordinary float weights (not restricted to {-s, 0, +s}) are left for
    # quantize_dynamic, not this pass.
    weight = _f32(np.random.default_rng(3).standard_normal((8, 4)), "W")
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    model = _model(nodes, [_vi("X", [4, 8])], [_vi("Y", [4, 4])], [weight])
    quant = onnxsim.quantize_ternary(model)
    assert _op_counts(quant)["MatMul"] == 1
    assert _op_counts(quant)["DynamicQuantizeLinear"] == 0


def test_quantize_ternary_skips_four_level_weight():
    # {-2, -1, 0, 1} * scale is one level too many to be ternary.
    rng = np.random.default_rng(4)
    weight = _f32(rng.choice([-2, -1, 0, 1], size=(8, 4)).astype(np.float32) * 0.1, "W")
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    model = _model(nodes, [_vi("X", [4, 8])], [_vi("Y", [4, 4])], [weight])
    quant = onnxsim.quantize_ternary(model)
    assert _op_counts(quant)["MatMul"] == 1


def test_quantize_ternary_skips_all_zero_weight():
    # An all-zero weight carries no ternary signal -- rewriting it only adds
    # nodes, so it is left as a (degenerate but harmless) float MatMul.
    weight = _f32(np.zeros((8, 4), np.float32), "W")
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    model = _model(nodes, [_vi("X", [4, 8])], [_vi("Y", [4, 4])], [weight])
    quant = onnxsim.quantize_ternary(model)
    assert _op_counts(quant)["MatMul"] == 1


def test_quantize_ternary_leaves_non_ternary_nodes_for_quantize_dynamic():
    # A model with one ternary and one ordinary MatMul: quantize_ternary only
    # touches the ternary one; quantize_dynamic (run after) picks up the rest
    # -- the two passes are meant to compose, not compete for the same node.
    rng = np.random.default_rng(5)
    K, N = 16, 8
    ternary_w = _f32(_ternary_weight(rng, (K, N)), "Wt")
    ordinary_w = _f32(rng.standard_normal((N, N)) * 0.5, "Wo")
    nodes = [
        onnx.helper.make_node("MatMul", ["X", "Wt"], ["H"]),
        onnx.helper.make_node("MatMul", ["H", "Wo"], ["Y"]),
    ]
    model = _model(
        nodes, [_vi("X", [2, K])], [_vi("Y", [2, N])], [ternary_w, ordinary_w]
    )

    ternary_only = onnxsim.quantize_ternary(model)
    assert _op_counts(ternary_only)["MatMul"] == 1  # only "Wo" left untouched

    both = onnxsim.quantize_dynamic(ternary_only)
    assert _op_counts(both)["MatMul"] == 0
    assert _op_counts(both)["MatMulInteger"] == 2

    x = rng.standard_normal((2, K)).astype(np.float32)
    _assert_close(_run(model, {"X": x}), _run(both, {"X": x}))


def test_quantize_ternary_skips_old_opset():
    # DynamicQuantizeLinear needs opset >= 11.
    rng = np.random.default_rng(6)
    weight = _f32(_ternary_weight(rng, (8, 4)), "W")
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    model = _model(nodes, [_vi("X", [4, 8])], [_vi("Y", [4, 4])], [weight], opset=10)
    quant = onnxsim.quantize_ternary(model)
    assert _op_counts(quant)["MatMul"] == 1
