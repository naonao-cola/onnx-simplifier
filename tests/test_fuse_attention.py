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
import math

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


# --------------------------------------------------------------------------- #
# scaled_dot_product_attention's own decomposition -- MatchScaledQKMatMul's
# "pre-scaled" shape: `scores = MatMul(Mul(Q_t, c), Mul(K_t, c))`, i.e. the
# combined scale split as sqrt(scale) onto each operand *before* the dot
# product, rather than applied to its result once. This is what
# `torch.nn.functional.scaled_dot_product_attention`'s ONNX export
# decomposes into (both PyTorch's legacy TorchScript and dynamo exporters),
# once its own dynamic Shape/Slice/Cast/Sqrt/... scale computation is
# constant-folded -- verified directly against real torch.onnx.export output
# during development; these tests reproduce the resulting shape directly via
# onnx.helper so they need no torch dependency.
# --------------------------------------------------------------------------- #
def _sdpa_prescaled_model(
    B=2, S=5, H=32, NH=4, scale=None, k_scale=None, ctx_rank3=True, seed=10
):
    # Builds Y = Linear(ctx) where ctx is a self-attention context computed
    # via the pre-scaled-QK shape above, optionally with an asymmetric
    # (different) scale on the K side (`k_scale`) to exercise the decline
    # path, and optionally with `ctx_rank3=False` to reproduce the *other*
    # real-export wrinkle this pass handles: the context reshape collapsed
    # directly to a 2-D [B*S, H] target (merged with the output projection's
    # own Gemm-input flatten by an earlier fixed-point iteration) instead of
    # the "natural" 3-D [B,S,H] one.
    Dh = H // NH
    scale = (Dh**-0.5) if scale is None else scale
    c = math.sqrt(scale)
    k_c = c if k_scale is None else math.sqrt(k_scale)
    rng = np.random.default_rng(seed)
    inits = []
    nodes = []

    shape_qkv = _i64([B, S, NH, Dh], "shape_qkv")
    inits.append(shape_qkv)

    q_nodes, q_inits, q_out = _linear_nodes(rng, "x", H, H, "q", True)
    k_nodes, k_inits, k_out = _linear_nodes(rng, "x", H, H, "k", True)
    v_nodes, v_inits, v_out = _linear_nodes(rng, "x", H, H, "v", True)
    nodes += q_nodes + k_nodes + v_nodes
    inits += q_inits + k_inits + v_inits

    qh_nodes, q_t = _head_split_nodes(q_out, "shape_qkv", [0, 2, 1, 3], "q")
    kh_nodes, k_t = _head_split_nodes(k_out, "shape_qkv", [0, 2, 3, 1], "k")
    vh_nodes, v_t = _head_split_nodes(v_out, "shape_qkv", [0, 2, 1, 3], "v")
    nodes += qh_nodes + kh_nodes + vh_nodes

    c_init = _f32(np.array(c), "q_scale_c")
    k_c_init = _f32(np.array(k_c), "k_scale_c")
    inits += [c_init, k_c_init]
    nodes.append(onnx.helper.make_node("Mul", [q_t, c_init.name], ["q_scaled"]))
    nodes.append(onnx.helper.make_node("Mul", [k_t, k_c_init.name], ["k_scaled"]))
    nodes.append(onnx.helper.make_node("MatMul", ["q_scaled", "k_scaled"], ["scores"]))
    nodes.append(onnx.helper.make_node("Softmax", ["scores"], ["attn"], axis=-1))
    nodes.append(onnx.helper.make_node("MatMul", ["attn", v_t], ["ctx0"]))
    nodes.append(
        onnx.helper.make_node("Transpose", ["ctx0"], ["ctx1"], perm=[0, 2, 1, 3])
    )
    ctx_shape = [B, S, H] if ctx_rank3 else [B * S, H]
    shape_ctx = _i64(ctx_shape, "shape_ctx")
    inits.append(shape_ctx)
    nodes.append(onnx.helper.make_node("Reshape", ["ctx1", "shape_ctx"], ["ctx2"]))

    if ctx_rank3:
        out_nodes, out_inits, out_name = _linear_nodes(rng, "ctx2", H, H, "out", True)
        nodes += out_nodes
        inits += out_inits
    else:
        # Mirrors the real SDPA-export trace exactly: ctx2 is already 2-D, so
        # the output projection is a plain Gemm reading it directly (no
        # separate flattening Reshape -- ctx2's own target already merged
        # that step in).
        w = _f32(rng.standard_normal((H, H)) * 0.3, "out_w")
        b = _f32(rng.standard_normal(H) * 0.1, "out_b")
        inits += [w, b]
        nodes.append(
            onnx.helper.make_node("Gemm", ["ctx2", w.name, b.name], ["out_out"])
        )
        out_name = "out_out"
    out_shape = [B, S, H] if ctx_rank3 else [B * S, H]
    nodes.append(onnx.helper.make_node("Identity", [out_name], ["y"]))

    graph = onnx.helper.make_graph(
        nodes, "g", [_vi("x", [B, S, H])], [_vi("y", out_shape)], inits
    )
    return onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 17)], ir_version=10
    )


def test_fuse_attention_sdpa_prescaled_pattern():
    B, S, H, NH = 2, 5, 32, 4
    Dh = H // NH
    scale = Dh**-0.5
    model = _sdpa_prescaled_model(B=B, S=S, H=H, NH=NH, scale=scale)
    simplified, ok = onnxsim.simplify(model)
    assert ok
    ops = _op_counts(simplified)
    assert ops["Attention"] == 1
    assert ops["Softmax"] == 0
    attn = next(n for n in simplified.graph.node if n.op_type == "Attention")
    got_scale = next(a for a in attn.attribute if a.name == "scale").f
    assert math.isclose(got_scale, scale, rel_tol=1e-5)
    onnx.checker.check_model(simplified)

    rng = np.random.default_rng(42)
    x = rng.standard_normal((B, S, H)).astype(np.float32)
    _assert_close(_run(model, {"x": x}), _run(simplified, {"x": x}))


def test_fuse_attention_declines_asymmetric_prescale():
    # Q and K scaled by *different* constants before the dot product: not
    # the sqrt(scale)-on-each-side decomposition MatchScaledQKMatMul expects,
    # so this must decline (stay numerically correct, just unfused) rather
    # than fusing with a wrong (e.g. geometric-mean-only-by-accident) scale.
    B, S, H, NH = 2, 5, 32, 4
    Dh = H // NH
    scale = Dh**-0.5
    model = _sdpa_prescaled_model(
        B=B, S=S, H=H, NH=NH, scale=scale, k_scale=scale * 4.0
    )
    simplified, ok = onnxsim.simplify(model)
    assert ok
    assert _op_counts(simplified)["Attention"] == 0

    rng = np.random.default_rng(43)
    x = rng.standard_normal((B, S, H)).astype(np.float32)
    _assert_close(_run(model, {"x": x}), _run(simplified, {"x": x}))


def test_fuse_attention_handles_context_reshape_collapsed_to_2d():
    # The context reshape's own target is already the *final* 2-D
    # [B*S, H] shape (an earlier fixed-point iteration merged "reshape ctx
    # to 3-D" with "flatten for the output projection's Gemm" into one
    # Reshape) rather than the "natural" 3-D [B,S,H] one -- exactly what a
    # real scaled_dot_product_attention export's own Gemm-conversion
    # produces. Fusing must still succeed and reuse that 2-D target for
    # Attention's own (always rank-3) output, rather than assuming it is
    # always 3-D and mismatching the output projection's Gemm.
    B, S, H, NH = 2, 5, 32, 4
    Dh = H // NH
    scale = Dh**-0.5
    model = _sdpa_prescaled_model(B=B, S=S, H=H, NH=NH, scale=scale, ctx_rank3=False)
    simplified, ok = onnxsim.simplify(model)
    assert ok
    ops = _op_counts(simplified)
    assert ops["Attention"] == 1
    # A Reshape must sit between Attention's rank-3 output and the (rank-2
    # input) output-projection Gemm.
    assert ops["Reshape"] >= 1
    onnx.checker.check_model(simplified)

    rng = np.random.default_rng(44)
    x = rng.standard_normal((B, S, H)).astype(np.float32)
    _assert_close(_run(model, {"x": x}), _run(simplified, {"x": x}))


def test_fuse_attention_recovers_input_flattened_by_gemm_conversion():
    # Simulates the state a real scaled_dot_product_attention export ends up
    # in after onnx-optimizer's own fuse_matmul_add_into_gemm has already
    # flattened Q/K/V's shared 3-D input to 2-D (Gemm requires 2-D operands)
    # in an earlier fixed-point iteration -- see
    # RecoverRank3AttentionInput's own doc comment in fuse_attention.h for
    # why this reliably happens for a real SDPA export (its own scale
    # computation needs constant folding to resolve first, so
    # Gemm-conversion -- which has no such prerequisite -- reliably wins
    # that race). Fusing must still succeed, using the recovered original
    # rank-3 `x` as Attention's own X input rather than the rank-2 flattened
    # value, or declining outright rather than emitting a shape-invalid
    # Attention node.
    B, S, H, NH = 2, 5, 32, 4
    Dh = H // NH
    rng = np.random.default_rng(20)
    inits = []
    nodes = []

    shape_flat = _i64([B * S, H], "shape_flat")
    inits.append(shape_flat)
    nodes.append(onnx.helper.make_node("Reshape", ["x", "shape_flat"], ["x_flat"]))

    def gemm_linear(prefix, in_name, in_dim, out_dim):
        w = _f32(rng.standard_normal((in_dim, out_dim)) * 0.3, f"{prefix}_w")
        b = _f32(rng.standard_normal(out_dim) * 0.1, f"{prefix}_b")
        nodes.append(
            onnx.helper.make_node("Gemm", [in_name, w.name, b.name], [f"{prefix}_out"])
        )
        inits.extend([w, b])
        return f"{prefix}_out"

    q_out = gemm_linear("q", "x_flat", H, H)
    k_out = gemm_linear("k", "x_flat", H, H)
    v_out = gemm_linear("v", "x_flat", H, H)

    shape_qkv = _i64([B, S, NH, Dh], "shape_qkv")
    inits.append(shape_qkv)
    qh_nodes, q_t = _head_split_nodes(q_out, "shape_qkv", [0, 2, 1, 3], "q")
    kh_nodes, k_t = _head_split_nodes(k_out, "shape_qkv", [0, 2, 3, 1], "k")
    vh_nodes, v_t = _head_split_nodes(v_out, "shape_qkv", [0, 2, 1, 3], "v")
    nodes += qh_nodes + kh_nodes + vh_nodes

    divisor = _f32(np.array(float(Dh) ** 0.5), "divisor")
    inits.append(divisor)
    nodes.append(onnx.helper.make_node("MatMul", [q_t, k_t], ["qk"]))
    nodes.append(onnx.helper.make_node("Div", ["qk", divisor.name], ["scores"]))
    nodes.append(onnx.helper.make_node("Softmax", ["scores"], ["attn"], axis=-1))
    nodes.append(onnx.helper.make_node("MatMul", ["attn", v_t], ["ctx0"]))
    nodes.append(
        onnx.helper.make_node("Transpose", ["ctx0"], ["ctx1"], perm=[0, 2, 1, 3])
    )
    shape_ctx = _i64([B, S, H], "shape_ctx")
    inits.append(shape_ctx)
    nodes.append(onnx.helper.make_node("Reshape", ["ctx1", "shape_ctx"], ["ctx2"]))
    nodes.append(onnx.helper.make_node("Identity", ["ctx2"], ["y"]))

    graph = onnx.helper.make_graph(
        nodes, "g", [_vi("x", [B, S, H])], [_vi("y", [B, S, H])], inits
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 17)], ir_version=10
    )

    simplified, ok = onnxsim.simplify(model)
    assert ok
    ops = _op_counts(simplified)
    assert ops["Attention"] == 1
    attn = next(n for n in simplified.graph.node if n.op_type == "Attention")
    # Attention's own X input must be the recovered rank-3 `x`, not the
    # rank-2 flattened value Q/K/V's Gemms needed.
    assert attn.input[0] == "x"
    onnx.checker.check_model(simplified)

    rng2 = np.random.default_rng(21)
    x = rng2.standard_normal((B, S, H)).astype(np.float32)
    _assert_close(_run(model, {"x": x}), _run(simplified, {"x": x}))
