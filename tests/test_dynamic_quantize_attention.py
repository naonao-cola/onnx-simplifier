"""Tests for ``onnxsim.quantize_attention_dynamic`` (the
``dynamic_quantize_attention`` C++ pass).

Unlike ``test_dynamic_quantize_matmul.py``, the input model here is an
``Attention`` (``com.microsoft``) node built directly with ``onnx.helper`` --
the shape ``fuse_attention.h`` produces (see ``onnxsim/passes/fuse_attention.h``
and ``tests/test_fuse_attention.py``), not a bare MatMul -- since this pass
expects one to already be present rather than fusing it itself. Each model is
quantized and then actually run through ONNX Runtime, both before and after,
so these tests double as a minimal end-to-end check: the quantized graph must
load and execute under a real inference engine, and its outputs must stay
close to the float baseline.
"""

import collections

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper
import pytest

import onnxsim

# A bare ``import onnxruntime`` would fail collection (not skip the test) on
# platforms onnxruntime doesn't ship wheels for; both Attention and
# QAttention are "com.microsoft" contrib ops only onnxruntime can execute.
ort = pytest.importorskip("onnxruntime")


def _vi(name, shape):
    return onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)


def _f32(array, name):
    return onnx.numpy_helper.from_array(np.asarray(array, dtype=np.float32), name)


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _op_counts(model):
    return collections.Counter(n.op_type for n in model.graph.node)


def _assert_close(float_outputs, quant_outputs, tol=0.1):
    # INT8/uint8 dynamic quantization is lossy by design -- see
    # test_dynamic_quantize_matmul.py's own _assert_close for why this checks
    # aggregate relative L2 error rather than a tight per-element band.
    for f, q in zip(float_outputs, quant_outputs):
        f = np.asarray(f, dtype=np.float64).ravel()
        q = np.asarray(q, dtype=np.float64).ravel()
        assert np.all(np.isfinite(q))
        rel_l2 = np.linalg.norm(f - q) / max(np.linalg.norm(f), 1e-6)
        assert rel_l2 < tol, f"relative L2 error too large: {rel_l2:.4f}"


def _attention_model(B=2, S=5, H=32, NH=4, VH=None, seed=0, opset=17):
    # Y = Attention(X, Wqkv, Bqkv, num_heads=NH, qkv_hidden_sizes=[H,H,VH])
    # -- exactly fuse_attention.h's own runTransform output shape.
    VH = VH or H
    rng = np.random.default_rng(seed)
    wqkv = _f32(rng.standard_normal((H, H + H + VH)) * 0.1, "wqkv")
    bqkv = _f32(rng.standard_normal(H + H + VH) * 0.1, "bqkv")
    attn = onnx.helper.make_node(
        "Attention",
        ["x", "wqkv", "bqkv"],
        ["y"],
        domain="com.microsoft",
        num_heads=NH,
        qkv_hidden_sizes=[H, H, VH],
    )
    graph = onnx.helper.make_graph(
        [attn], "g", [_vi("x", [B, S, H])], [_vi("y", [B, S, VH])], [wqkv, bqkv]
    )
    return onnx.helper.make_model(
        graph,
        opset_imports=[
            onnx.helper.make_opsetid("", opset),
            onnx.helper.make_opsetid("com.microsoft", 1),
        ],
        ir_version=10,
    )


def test_quantize_attention():
    B, S, H, NH = 2, 5, 32, 4
    model = _attention_model(B=B, S=S, H=H, NH=NH)
    quant = onnxsim.quantize_attention_dynamic(model)
    onnx.checker.check_model(quant)
    ops = _op_counts(quant)
    assert ops["Attention"] == 0
    assert ops["QAttention"] == 1
    assert ops["DynamicQuantizeLinear"] == 1

    qattn = next(n for n in quant.graph.node if n.op_type == "QAttention")
    num_heads = next(a for a in qattn.attribute if a.name == "num_heads").i
    assert num_heads == NH
    # bias is reused as-is (still the original float initializer name).
    assert qattn.input[2] == "bqkv"
    # mask_index (position 5) must be skipped as an empty input name.
    assert qattn.input[5] == ""

    rng = np.random.default_rng(42)
    x = rng.standard_normal((B, S, H)).astype(np.float32)
    _assert_close(_run(model, {"x": x}), _run(quant, {"x": x}))


def test_quantize_attention_preserves_scale_attribute():
    B, S, H, NH = 2, 5, 32, 4
    model = _attention_model(B=B, S=S, H=H, NH=NH)
    attn = model.graph.node[0]
    attn.attribute.append(onnx.helper.make_attribute("scale", 0.25))

    quant = onnxsim.quantize_attention_dynamic(model)
    onnx.checker.check_model(quant)
    qattn = next(n for n in quant.graph.node if n.op_type == "QAttention")
    scale = next(a for a in qattn.attribute if a.name == "scale").f
    assert scale == pytest.approx(0.25)

    rng = np.random.default_rng(43)
    x = rng.standard_normal((B, S, H)).astype(np.float32)
    _assert_close(_run(model, {"x": x}), _run(quant, {"x": x}))


def test_quantize_attention_declines_uneven_qkv_split():
    # V's hidden size differs from Q/K's -- Attention itself supports this
    # (qkv_hidden_sizes), but QAttention's schema assumes an even three-way
    # split, so this must decline rather than guess.
    B, S, H, NH, VH = 2, 5, 32, 4, 16
    model = _attention_model(B=B, S=S, H=H, NH=NH, VH=VH)
    quant = onnxsim.quantize_attention_dynamic(model)
    ops = _op_counts(quant)
    assert ops["Attention"] == 1
    assert ops["QAttention"] == 0

    rng = np.random.default_rng(44)
    x = rng.standard_normal((B, S, H)).astype(np.float32)
    _assert_close(_run(model, {"x": x}), _run(quant, {"x": x}), tol=1e-4)


def test_quantize_attention_declines_non_constant_weight():
    # The weight is a graph input, not a constant -- nothing to quantize
    # ahead of time.
    B, S, H, NH = 2, 5, 32, 4
    rng = np.random.default_rng(5)
    bqkv = _f32(rng.standard_normal(H * 3) * 0.1, "bqkv")
    attn = onnx.helper.make_node(
        "Attention",
        ["x", "wqkv", "bqkv"],
        ["y"],
        domain="com.microsoft",
        num_heads=NH,
        qkv_hidden_sizes=[H, H, H],
    )
    graph = onnx.helper.make_graph(
        [attn],
        "g",
        [_vi("x", [B, S, H]), _vi("wqkv", [H, H * 3])],
        [_vi("y", [B, S, H])],
        [bqkv],
    )
    model = onnx.helper.make_model(
        graph,
        opset_imports=[
            onnx.helper.make_opsetid("", 17),
            onnx.helper.make_opsetid("com.microsoft", 1),
        ],
        ir_version=10,
    )
    quant = onnxsim.quantize_attention_dynamic(model)
    ops = _op_counts(quant)
    assert ops["Attention"] == 1
    assert ops["QAttention"] == 0


def test_quantize_attention_declines_old_opset():
    # DynamicQuantizeLinear needs opset >= 11.
    B, S, H, NH = 2, 5, 32, 4
    model = _attention_model(B=B, S=S, H=H, NH=NH, opset=10)
    quant = onnxsim.quantize_attention_dynamic(model)
    ops = _op_counts(quant)
    assert ops["Attention"] == 1
    assert ops["QAttention"] == 0
