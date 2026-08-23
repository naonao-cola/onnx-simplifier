"""Tests for the ``fuse_attention`` C++ pass
(``onnxsim/passes/fuse_attention.h``) -- pattern-matches a "hand-written"
multi-head self-attention subgraph (separate Q/K/V ``nn.Linear``-style
projections + reshape/transpose to heads + scaled dot-product + softmax +
weighted sum) into a single ONNX Runtime "com.microsoft" contrib op,
``Attention``. Unlike the ``quantize_*`` passes elsewhere in this test suite,
this is a default-on graph-shape fusion (registered the same way
``fuse_rms_norm``/``fuse_gelu``/``fuse_layer_norm`` are, see
``custom_optimizer_passes.cpp``) -- it always runs as part of plain
``onnxsim.simplify()``, with no separate Python entry point or CLI flag.

Every model here is built directly with ``onnx.helper`` (no torch dependency)
to mirror exactly what a real PyTorch export of a hand-rolled (eager, not
``nn.MultiheadAttention``) attention module produces -- see
``onnxsim/passes/fuse_attention.h``'s own file comment for the node-by-node
shape this targets, which was derived by tracing real PyTorch exports.
"""

import collections

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper
import pytest

import onnxsim

# A bare ``import onnxruntime`` would fail collection (not skip the test) on
# platforms onnxruntime doesn't ship wheels for (e.g. s390x); the fused
# output is a "com.microsoft" contrib op that only onnxruntime can execute.
ort = pytest.importorskip("onnxruntime")


def _vi(name, shape):
    return onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _i64(array, name):
    return onnx.numpy_helper.from_array(np.asarray(array, dtype=np.int64), name)


def _linear_nodes(rng, x_name, in_dim, out_dim, prefix, bias):
    # MatMul(x, W)[+ Add(., B)] -- the shape a plain ``nn.Linear`` exported via
    # ``MatMul`` (not ``Gemm``) produces; matches fuse_attention.h's
    # MatchAttentionProjection Add-branch.
    w = _f32(rng.standard_normal((in_dim, out_dim)) * 0.3, f"{prefix}_w")
    nodes = [onnx.helper.make_node("MatMul", [x_name, w.name], [f"{prefix}_mm"])]
    inits = [w]
    out_name = f"{prefix}_mm"
    if bias:
        b = _f32(rng.standard_normal(out_dim) * 0.1, f"{prefix}_b")
        nodes.append(
            onnx.helper.make_node("Add", [b.name, out_name], [f"{prefix}_out"])
        )
        inits.append(b)
        out_name = f"{prefix}_out"
    return nodes, inits, out_name


def _head_split_nodes(x_name, shape_name, perm, prefix):
    reshape_out = f"{prefix}_reshape"
    transpose_out = f"{prefix}_transpose"
    nodes = [
        onnx.helper.make_node("Reshape", [x_name, shape_name], [reshape_out]),
        onnx.helper.make_node("Transpose", [reshape_out], [transpose_out], perm=perm),
    ]
    return nodes, transpose_out


def _attention_model(
    B=2,
    S=5,
    H=32,
    NH=4,
    VH=None,
    bias=True,
    scale_op="Div",
    kv_source="x",
    seed=0,
    opset=17,
):
    # Builds: Y = Linear(ctx, Wout[, Bout]) where ctx is the fused-away
    # self-attention context -- see fuse_attention.h's own top-of-file
    # comment for the exact node shape this mirrors.
    rng = np.random.default_rng(seed)
    VH = VH or H
    Dh, Dv = H // NH, VH // NH
    inits = []
    nodes = []

    shape_qk = _i64([B, S, NH, Dh], "shape_qk")
    shape_v = _i64([B, S, NH, Dv], "shape_v")
    inits += [shape_qk, shape_v]

    q_nodes, q_inits, q_out = _linear_nodes(rng, "x", H, H, "q", bias)
    k_nodes, k_inits, k_out = _linear_nodes(rng, kv_source, H, H, "k", bias)
    v_nodes, v_inits, v_out = _linear_nodes(rng, kv_source, H, VH, "v", bias)
    nodes += q_nodes + k_nodes + v_nodes
    inits += q_inits + k_inits + v_inits

    qh_nodes, q_t = _head_split_nodes(q_out, "shape_qk", [0, 2, 1, 3], "q")
    kh_nodes, k_t = _head_split_nodes(k_out, "shape_qk", [0, 2, 3, 1], "k")
    vh_nodes, v_t = _head_split_nodes(v_out, "shape_v", [0, 2, 1, 3], "v")
    nodes += qh_nodes + kh_nodes + vh_nodes

    nodes.append(onnx.helper.make_node("MatMul", [q_t, k_t], ["qk"]))
    if scale_op == "Div":
        divisor = _f32(np.array(float(Dh) ** 0.5), "divisor")
        inits.append(divisor)
        nodes.append(onnx.helper.make_node("Div", ["qk", divisor.name], ["scores"]))
    else:
        mult = _f32(np.array(float(Dh) ** -0.5), "mult")
        inits.append(mult)
        nodes.append(onnx.helper.make_node("Mul", ["qk", mult.name], ["scores"]))
    nodes.append(onnx.helper.make_node("Softmax", ["scores"], ["attn"], axis=-1))
    nodes.append(onnx.helper.make_node("MatMul", ["attn", v_t], ["ctx0"]))
    nodes.append(
        onnx.helper.make_node("Transpose", ["ctx0"], ["ctx1"], perm=[0, 2, 1, 3])
    )
    shape_ctx = _i64([B, S, NH * Dv], "shape_ctx")
    inits.append(shape_ctx)
    nodes.append(onnx.helper.make_node("Reshape", ["ctx1", "shape_ctx"], ["ctx2"]))

    out_nodes, out_inits, out_name = _linear_nodes(rng, "ctx2", VH, H, "out", bias)
    nodes += out_nodes
    inits += out_inits
    nodes.append(onnx.helper.make_node("Identity", [out_name], ["y"]))

    graph_inputs = [_vi("x", [B, S, H])]
    if kv_source != "x":
        graph_inputs.append(_vi(kv_source, [B, S, H]))
    graph = onnx.helper.make_graph(
        nodes, "g", graph_inputs, [_vi("y", [B, S, H])], inits
    )
    return onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", opset)], ir_version=10
    )


def _op_counts(model):
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


def test_fuse_attention_basic():
    B, S, H = 2, 5, 32
    model = _attention_model(B=B, S=S, H=H, bias=True, scale_op="Div")
    simplified, ok = onnxsim.simplify(model)
    assert ok
    ops = _op_counts(simplified)
    assert ops["Attention"] == 1
    # The whole Q/K/V/softmax/context subgraph collapses into the one
    # Attention node; only the (untouched) output projection survives from
    # the original 6 MatMuls -- as either MatMul or Gemm, depending on
    # whether fuse_matmul_add_bias_into_gemm_batched also fires on it.
    assert ops["MatMul"] + ops["Gemm"] == 1
    assert ops["Softmax"] == 0
    assert ops["Transpose"] == 0
    domains = {o.domain for o in simplified.opset_import}
    assert "com.microsoft" in domains
    onnx.checker.check_model(simplified)

    rng = np.random.default_rng(42)
    x = rng.standard_normal((B, S, H)).astype(np.float32)
    _assert_close(_run(model, {"x": x}), _run(simplified, {"x": x}))


def test_fuse_attention_no_bias_synthesizes_zero_bias():
    # Q/K/V/out all bias-free: the fusion must still emit a (zero-filled)
    # bias input rather than omitting it -- see fuse_attention.h's own
    # comment on why (at least one real ONNX Runtime CPU build segfaults on
    # an Attention node with only 2 inputs).
    B, S, H = 2, 5, 24
    model = _attention_model(B=B, S=S, H=H, NH=4, bias=False, scale_op="Div")
    simplified, ok = onnxsim.simplify(model)
    assert ok
    ops = _op_counts(simplified)
    assert ops["Attention"] == 1
    attn = next(n for n in simplified.graph.node if n.op_type == "Attention")
    assert len(attn.input) == 3
    bias_init = next(t for t in simplified.graph.initializer if t.name == attn.input[2])
    assert np.all(onnx.numpy_helper.to_array(bias_init) == 0.0)
    onnx.checker.check_model(simplified)

    rng = np.random.default_rng(1)
    x = rng.standard_normal((B, S, H)).astype(np.float32)
    _assert_close(_run(model, {"x": x}), _run(simplified, {"x": x}))


def test_fuse_attention_mul_scale():
    # `scores * (1/sqrt(head_size))` instead of `scores / sqrt(head_size)`.
    B, S, H = 2, 4, 16
    model = _attention_model(B=B, S=S, H=H, NH=2, bias=True, scale_op="Mul")
    simplified, ok = onnxsim.simplify(model)
    assert ok
    assert _op_counts(simplified)["Attention"] == 1
    onnx.checker.check_model(simplified)

    rng = np.random.default_rng(2)
    x = rng.standard_normal((B, S, H)).astype(np.float32)
    _assert_close(_run(model, {"x": x}), _run(simplified, {"x": x}))


def test_fuse_attention_different_v_hidden_size():
    # V's projection hidden size may differ from Q/K's, per Attention's own
    # documented qkv_hidden_sizes semantics.
    B, S, H, VH = 2, 5, 32, 16
    model = _attention_model(B=B, S=S, H=H, NH=4, VH=VH, bias=True)
    simplified, ok = onnxsim.simplify(model)
    assert ok
    ops = _op_counts(simplified)
    assert ops["Attention"] == 1
    attn = next(n for n in simplified.graph.node if n.op_type == "Attention")
    qkv_sizes = next(a for a in attn.attribute if a.name == "qkv_hidden_sizes").ints
    assert list(qkv_sizes) == [H, H, VH]
    onnx.checker.check_model(simplified)

    rng = np.random.default_rng(3)
    x = rng.standard_normal((B, S, H)).astype(np.float32)
    _assert_close(_run(model, {"x": x}), _run(simplified, {"x": x}))


def test_fuse_attention_declines_cross_attention():
    # Q reads a different source than K/V: fuse_attention.h only handles
    # self-attention and must decline, leaving the graph numerically correct
    # (just unfused) rather than misfiring.
    B, S, H = 2, 5, 32
    model = _attention_model(B=B, S=S, H=H, kv_source="mem")
    simplified, ok = onnxsim.simplify(model)
    assert ok
    assert _op_counts(simplified)["Attention"] == 0

    rng = np.random.default_rng(4)
    x = rng.standard_normal((B, S, H)).astype(np.float32)
    mem = rng.standard_normal((B, S, H)).astype(np.float32)
    _assert_close(
        _run(model, {"x": x, "mem": mem}), _run(simplified, {"x": x, "mem": mem})
    )
