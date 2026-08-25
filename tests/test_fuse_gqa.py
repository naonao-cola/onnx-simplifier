"""Tests for the ``fuse_gqa`` C++ pass (``onnxsim/passes/fuse_gqa.h``) --
pattern-matches a causal grouped-query/multi-query attention block (fewer K/V
heads than Q heads, broadcast via HuggingFace's standard ``repeat_kv``, plus
an additive causal mask) into a single ONNX Runtime "com.microsoft" contrib
op, ``GroupQueryAttention``. Like ``fuse_attention``, this is a default-on
graph-shape fusion that always runs as part of plain ``onnxsim.simplify()``.

Every model here is built directly with ``onnx.helper`` (no torch dependency)
to mirror what a real traced ``repeat_kv``-based GQA export produces -- see
``fuse_gqa.h``'s own top-of-file comment for the exact node shape this
targets and, importantly, why it only fires when the additive mask is a
*provable constant* matching the causal pattern exactly (``GroupQueryAttention``
always applies causal masking internally and unconditionally -- confirmed
during development by comparing its output against manual bidirectional vs.
causal references on the same random inputs).
"""

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper
import pytest

import onnxsim

# A bare ``import onnxruntime`` would fail collection (not skip the test) on
# platforms onnxruntime doesn't ship wheels for; the fused output is a
# "com.microsoft" contrib op that only onnxruntime can execute.
ort = pytest.importorskip("onnxruntime")


def _vi(name, shape):
    return onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)


def _f32(array, name):
    return onnx.numpy_helper.from_array(np.asarray(array, dtype=np.float32), name)


def _i64(array, name):
    return onnx.numpy_helper.from_array(np.asarray(array, dtype=np.int64), name)


def _causal_mask(seq_len):
    mask = np.zeros((1, 1, seq_len, seq_len), dtype=np.float32)
    mask[0, 0][np.triu_indices(seq_len, k=1)] = -3.0e38
    return mask


def _gqa_model(B=2, S=6, NH=8, NKV=2, Dh=16, mask=None, mask_is_input=False):
    # Builds Y = Linear(ctx) where ctx is a causal GQA/MQA self-attention
    # context: separate Q/K/V nn.Linear-style (bias-free) projections,
    # head-split, K/V's repeat_kv broadcast up to Q's head count, scaled
    # dot-product, an additive mask, softmax, weighted sum -- see
    # fuse_gqa.h's own top comment for the exact shape this mirrors.
    n_rep = NH // NKV
    H = NH * Dh
    HKV = NKV * Dh
    rng = np.random.default_rng(0)

    inits = [
        _f32(rng.standard_normal((H, H)) * 0.1, "wq"),
        _f32(rng.standard_normal((H, HKV)) * 0.1, "wk"),
        _f32(rng.standard_normal((H, HKV)) * 0.1, "wv"),
        _f32(rng.standard_normal((H, H)) * 0.1, "wo"),
        _i64([B, S, NH, Dh], "shape_q"),
        _i64([B, S, NKV, Dh], "shape_kv"),
        _i64([2], "unsq_axes"),
        _i64([B, NKV, n_rep, S, Dh], "expand_shape"),
        _i64([B, NH, S, Dh], "merge_shape"),
        _f32(np.array(float(Dh) ** 0.5), "sqrt_dh"),
        _i64([B, S, H], "shape_ctx"),
    ]

    def repeat_kv_nodes(raw_name, prefix):
        return [
            onnx.helper.make_node(
                "Unsqueeze", [raw_name, "unsq_axes"], [f"{prefix}_unsq"]
            ),
            onnx.helper.make_node(
                "Expand", [f"{prefix}_unsq", "expand_shape"], [f"{prefix}_exp"]
            ),
            onnx.helper.make_node(
                "Reshape", [f"{prefix}_exp", "merge_shape"], [f"{prefix}_rep"]
            ),
        ]

    nodes = [
        onnx.helper.make_node("MatMul", ["x", "wq"], ["q_mm"]),
        onnx.helper.make_node("Reshape", ["q_mm", "shape_q"], ["q_r"]),
        onnx.helper.make_node("Transpose", ["q_r"], ["q_t"], perm=[0, 2, 1, 3]),
        onnx.helper.make_node("MatMul", ["x", "wk"], ["k_mm"]),
        onnx.helper.make_node("Reshape", ["k_mm", "shape_kv"], ["k_r"]),
        onnx.helper.make_node("Transpose", ["k_r"], ["k_raw"], perm=[0, 2, 1, 3]),
        *repeat_kv_nodes("k_raw", "k"),
        onnx.helper.make_node("Transpose", ["k_rep"], ["k_t"], perm=[0, 1, 3, 2]),
        onnx.helper.make_node("MatMul", ["x", "wv"], ["v_mm"]),
        onnx.helper.make_node("Reshape", ["v_mm", "shape_kv"], ["v_r"]),
        onnx.helper.make_node("Transpose", ["v_r"], ["v_raw"], perm=[0, 2, 1, 3]),
        *repeat_kv_nodes("v_raw", "v"),
        onnx.helper.make_node("MatMul", ["q_t", "k_t"], ["qk"]),
        onnx.helper.make_node("Div", ["qk", "sqrt_dh"], ["scores"]),
    ]

    graph_inputs = [_vi("x", [B, S, H])]
    if mask is None:
        softmax_input = "scores"
    elif mask_is_input:
        nodes.append(onnx.helper.make_node("Add", ["scores", "mask"], ["masked"]))
        graph_inputs.append(_vi("mask", [1, 1, S, S]))
        softmax_input = "masked"
    else:
        inits.append(_f32(mask, "mask"))
        nodes.append(onnx.helper.make_node("Add", ["scores", "mask"], ["masked"]))
        softmax_input = "masked"

    nodes += [
        onnx.helper.make_node("Softmax", [softmax_input], ["probs"], axis=-1),
        onnx.helper.make_node("MatMul", ["probs", "v_rep"], ["ctx0"]),
        onnx.helper.make_node("Transpose", ["ctx0"], ["ctx1"], perm=[0, 2, 1, 3]),
        onnx.helper.make_node("Reshape", ["ctx1", "shape_ctx"], ["ctx2"]),
        onnx.helper.make_node("MatMul", ["ctx2", "wo"], ["y"]),
    ]

    graph = onnx.helper.make_graph(
        nodes, "g", graph_inputs, [_vi("y", [B, S, H])], inits
    )
    return onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 17)], ir_version=10
    )


def _op_counts(model):
    import collections

    return collections.Counter(n.op_type for n in model.graph.node)


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _assert_close(float_outputs, fused_outputs):
    for f, q in zip(float_outputs, fused_outputs):
        f = np.asarray(f, dtype=np.float64).ravel()
        q = np.asarray(q, dtype=np.float64).ravel()
        assert np.all(np.isfinite(q))
        rel_l2 = np.linalg.norm(f - q) / max(np.linalg.norm(f), 1e-6)
        assert rel_l2 < 1e-4, f"relative L2 error too large: {rel_l2:.6f}"


def test_fuse_gqa_basic():
    B, S, NH, NKV, Dh = 2, 6, 8, 2, 16
    model = _gqa_model(B=B, S=S, NH=NH, NKV=NKV, Dh=Dh, mask=_causal_mask(S))
    simplified, ok = onnxsim.simplify(model)
    assert ok
    ops = _op_counts(simplified)
    assert ops["GroupQueryAttention"] == 1
    assert ops["Softmax"] == 0
    gqa = next(n for n in simplified.graph.node if n.op_type == "GroupQueryAttention")
    num_heads = next(a for a in gqa.attribute if a.name == "num_heads").i
    kv_num_heads = next(a for a in gqa.attribute if a.name == "kv_num_heads").i
    assert num_heads == NH
    assert kv_num_heads == NKV
    domains = {o.domain for o in simplified.opset_import}
    assert "com.microsoft" in domains
    onnx.checker.check_model(simplified)

    rng = np.random.default_rng(42)
    x = rng.standard_normal((B, S, NH * Dh)).astype(np.float32)
    _assert_close(_run(model, {"x": x}), _run(simplified, {"x": x}))


def test_fuse_gqa_multi_query():
    # MQA: a single shared K/V head (NKV=1) is the extreme case of GQA.
    B, S, NH, NKV, Dh = 2, 5, 4, 1, 8
    model = _gqa_model(B=B, S=S, NH=NH, NKV=NKV, Dh=Dh, mask=_causal_mask(S))
    simplified, ok = onnxsim.simplify(model)
    assert ok
    ops = _op_counts(simplified)
    assert ops["GroupQueryAttention"] == 1
    onnx.checker.check_model(simplified)

    rng = np.random.default_rng(7)
    x = rng.standard_normal((B, S, NH * Dh)).astype(np.float32)
    _assert_close(_run(model, {"x": x}), _run(simplified, {"x": x}))


def test_fuse_gqa_declines_without_mask():
    # Bidirectional (no additive mask at all): GroupQueryAttention always
    # applies causal masking internally with no way to disable it, so this
    # must decline rather than silently turn a bidirectional block causal.
    B, S, NH, NKV, Dh = 2, 6, 8, 2, 16
    model = _gqa_model(B=B, S=S, NH=NH, NKV=NKV, Dh=Dh, mask=None)
    simplified, ok = onnxsim.simplify(model)
    assert ok
    assert _op_counts(simplified)["GroupQueryAttention"] == 0

    rng = np.random.default_rng(1)
    x = rng.standard_normal((B, S, NH * Dh)).astype(np.float32)
    _assert_close(_run(model, {"x": x}), _run(simplified, {"x": x}))


def test_fuse_gqa_declines_non_causal_mask():
    # A mask that's present but numerically not the standard causal pattern
    # (all zeros here -- i.e. no actual masking) must not be assumed causal.
    B, S, NH, NKV, Dh = 2, 6, 8, 2, 16
    model = _gqa_model(
        B=B, S=S, NH=NH, NKV=NKV, Dh=Dh, mask=np.zeros((1, 1, S, S), dtype=np.float32)
    )
    simplified, ok = onnxsim.simplify(model)
    assert ok
    assert _op_counts(simplified)["GroupQueryAttention"] == 0

    rng = np.random.default_rng(2)
    x = rng.standard_normal((B, S, NH * Dh)).astype(np.float32)
    _assert_close(_run(model, {"x": x}), _run(simplified, {"x": x}))


def test_fuse_gqa_declines_runtime_mask():
    # A mask that's present, causal-shaped, and would numerically pass
    # VerifyCausalMaskConstant -- but is a runtime graph *input*, not a
    # compile-time constant. Real GQA exports almost always pass their mask
    # this way; this pass deliberately declines rather than trust an
    # un-provable runtime tensor is exactly causal-shaped (see fuse_gqa.h's
    # own top comment for why).
    B, S, NH, NKV, Dh = 2, 6, 8, 2, 16
    model = _gqa_model(
        B=B, S=S, NH=NH, NKV=NKV, Dh=Dh, mask=_causal_mask(S), mask_is_input=True
    )
    simplified, ok = onnxsim.simplify(model)
    assert ok
    assert _op_counts(simplified)["GroupQueryAttention"] == 0

    rng = np.random.default_rng(3)
    x = rng.standard_normal((B, S, NH * Dh)).astype(np.float32)
    mask = _causal_mask(S)
    _assert_close(
        _run(model, {"x": x, "mask": mask}), _run(simplified, {"x": x, "mask": mask})
    )
