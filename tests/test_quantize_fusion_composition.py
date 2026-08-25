"""Confirms ``quantize_dynamic`` (and, by the same reasoning, the other
MatMul/Gemm-pattern-matching quantize_* passes) composes correctly with
``simplify()``'s attention-family fusions -- ``fuse_gqa`` (``GroupQueryAttention``)
and ``fuse_rope`` (``RotaryEmbedding``).

Neither fused op takes a weight input the way ``Attention`` does (see
``dynamic_quantize_attention.h``'s own ``QAttention`` counterpart):
``GroupQueryAttention``'s query/key/value are pre-projected activations, and
``RotaryEmbedding`` has no learned parameters at all. So there is no
"QGroupQueryAttention"/"QRotaryEmbedding" to build -- the natural
quantization surface is the surrounding Q/K/V/O ``MatMul`` projections
``quantize_dynamic`` already pattern-matches on independently of what other
ops surround them. This module locks that composition in with a regression
test per fusion, rather than leaving it as an unverified assumption: each
model is fused via ``simplify()``, quantized via ``quantize_dynamic``, and
checked that the fused attention-family op survives untouched while its
neighboring projections are quantized, with the end-to-end numeric result
still close to the unquantized baseline.
"""

import collections

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper
import pytest

import onnxsim

# A bare ``import onnxruntime`` would fail collection (not skip the test) on
# platforms onnxruntime doesn't ship wheels for; GroupQueryAttention is a
# "com.microsoft" contrib op only onnxruntime can execute.
ort = pytest.importorskip("onnxruntime")


def _vi(name, shape):
    return onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)


def _f32(array, name):
    return onnx.numpy_helper.from_array(np.asarray(array, dtype=np.float32), name)


def _i64(array, name):
    return onnx.numpy_helper.from_array(np.asarray(array, dtype=np.int64), name)


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


def _causal_mask(seq_len):
    mask = np.zeros((1, 1, seq_len, seq_len), dtype=np.float32)
    mask[0, 0][np.triu_indices(seq_len, k=1)] = -3.0e38
    return mask


def _gqa_model(B=2, S=6, NH=8, NKV=2, Dh=16, seed=0):
    # Same shape tests/test_fuse_gqa.py's own _gqa_model builds -- see
    # fuse_gqa.h's top comment for the exact node-by-node pattern this
    # mirrors.
    n_rep = NH // NKV
    H = NH * Dh
    HKV = NKV * Dh
    rng = np.random.default_rng(seed)

    inits = [
        _f32(rng.standard_normal((H, H)) * 0.1, "wq"),
        _f32(rng.standard_normal((H, HKV)) * 0.1, "wk"),
        _f32(rng.standard_normal((H, HKV)) * 0.1, "wv"),
        _f32(rng.standard_normal((H, H)) * 0.1, "wo"),
        _f32(_causal_mask(S), "mask"),
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
        onnx.helper.make_node("Add", ["scores", "mask"], ["masked"]),
        onnx.helper.make_node("Softmax", ["masked"], ["probs"], axis=-1),
        onnx.helper.make_node("MatMul", ["probs", "v_rep"], ["ctx0"]),
        onnx.helper.make_node("Transpose", ["ctx0"], ["ctx1"], perm=[0, 2, 1, 3]),
        onnx.helper.make_node("Reshape", ["ctx1", "shape_ctx"], ["ctx2"]),
        onnx.helper.make_node("MatMul", ["ctx2", "wo"], ["y"]),
    ]
    graph = onnx.helper.make_graph(
        nodes, "g", [_vi("x", [B, S, H])], [_vi("y", [B, S, H])], inits
    )
    return onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 17)], ir_version=10
    )


def test_quantize_dynamic_composes_with_fuse_gqa():
    B, S, NH, NKV, Dh = 2, 6, 8, 2, 16
    H = NH * Dh
    model = _gqa_model(B=B, S=S, NH=NH, NKV=NKV, Dh=Dh)

    simplified, ok = onnxsim.simplify(model)
    assert ok
    assert _op_counts(simplified)["GroupQueryAttention"] == 1

    quantized = onnxsim.quantize_dynamic(simplified)
    onnx.checker.check_model(quantized)
    ops = _op_counts(quantized)
    # GroupQueryAttention has no weight input, so it must survive untouched
    # -- only the surrounding Q/K/V/O projection MatMuls are quantized.
    assert ops["GroupQueryAttention"] == 1
    assert ops["MatMul"] == 0
    assert ops["DynamicQuantizeLinear"] == 4
    assert ops["MatMulInteger"] == 4

    rng = np.random.default_rng(9)
    x = rng.standard_normal((B, S, H)).astype(np.float32)
    _assert_close(_run(simplified, {"x": x}), _run(quantized, {"x": x}))


def _rope_model(B=2, NH=4, S=6, Dh=8, seed=0):
    # Same shape tests/test_fusion_patterns.py's own _rope_model builds --
    # see fuse_rope.h's top comment for the exact node-by-node pattern this
    # mirrors.
    half = Dh // 2
    H = NH * Dh
    rng = np.random.default_rng(seed)
    inits = [
        _f32(rng.standard_normal((H, H)) * 0.1, "wq"),
        _f32(rng.standard_normal((H, H)) * 0.1, "wk"),
        _i64([B, S, NH, Dh], "shape_qk"),
        _i64([0], "slice_start0"),
        _i64([half], f"slice_end{half}"),
        _i64([half], f"slice_start{half}"),
        _i64([np.iinfo(np.int64).max], "slice_end_max"),
        _i64([-1], "slice_axism1"),
        _i64([1], "unsq_axis1"),
    ]

    def rope_apply_nodes(x_name, prefix):
        return [
            onnx.helper.make_node("Mul", [x_name, "cos_bcast"], [f"{prefix}_a"]),
            onnx.helper.make_node(
                "Slice",
                [x_name, "slice_start0", f"slice_end{half}", "slice_axism1"],
                [f"{prefix}_x1"],
            ),
            onnx.helper.make_node(
                "Slice",
                [x_name, f"slice_start{half}", "slice_end_max", "slice_axism1"],
                [f"{prefix}_x2"],
            ),
            onnx.helper.make_node("Neg", [f"{prefix}_x2"], [f"{prefix}_neg_x2"]),
            onnx.helper.make_node(
                "Concat",
                [f"{prefix}_neg_x2", f"{prefix}_x1"],
                [f"{prefix}_rotated"],
                axis=-1,
            ),
            onnx.helper.make_node(
                "Mul", [f"{prefix}_rotated", "sin_bcast"], [f"{prefix}_b"]
            ),
            onnx.helper.make_node(
                "Add", [f"{prefix}_a", f"{prefix}_b"], [f"{prefix}_embed"]
            ),
        ]

    nodes = [
        onnx.helper.make_node("MatMul", ["x", "wq"], ["q_mm"]),
        onnx.helper.make_node("Reshape", ["q_mm", "shape_qk"], ["q_r"]),
        onnx.helper.make_node("Transpose", ["q_r"], ["q"], perm=[0, 2, 1, 3]),
        onnx.helper.make_node("MatMul", ["x", "wk"], ["k_mm"]),
        onnx.helper.make_node("Reshape", ["k_mm", "shape_qk"], ["k_r"]),
        onnx.helper.make_node("Transpose", ["k_r"], ["k"], perm=[0, 2, 1, 3]),
        onnx.helper.make_node("Concat", ["angle", "angle"], ["emb"], axis=-1),
        onnx.helper.make_node("Cos", ["emb"], ["cos_full"]),
        onnx.helper.make_node("Sin", ["emb"], ["sin_full"]),
        onnx.helper.make_node("Unsqueeze", ["cos_full", "unsq_axis1"], ["cos_bcast"]),
        onnx.helper.make_node("Unsqueeze", ["sin_full", "unsq_axis1"], ["sin_bcast"]),
    ]
    nodes += rope_apply_nodes("q", "q")
    nodes += rope_apply_nodes("k", "k")
    nodes.append(onnx.helper.make_node("Identity", ["q_embed"], ["y_q"]))
    nodes.append(onnx.helper.make_node("Identity", ["k_embed"], ["y_k"]))

    graph = onnx.helper.make_graph(
        nodes,
        "g",
        [_vi("x", [B, S, H]), _vi("angle", [B, S, half])],
        [_vi("y_q", [B, NH, S, Dh]), _vi("y_k", [B, NH, S, Dh])],
        inits,
    )
    return onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 23)], ir_version=11
    )


def test_quantize_dynamic_composes_with_fuse_rope():
    B, NH, S, Dh = 2, 4, 6, 8
    half = Dh // 2
    H = NH * Dh
    model = _rope_model(B=B, NH=NH, S=S, Dh=Dh)

    simplified, ok = onnxsim.simplify(model)
    assert ok
    assert _op_counts(simplified)["RotaryEmbedding"] == 2

    quantized = onnxsim.quantize_dynamic(simplified)
    onnx.checker.check_model(quantized)
    ops = _op_counts(quantized)
    # RotaryEmbedding has no weight input at all, so it must survive
    # untouched -- only the Q/K projection MatMuls are quantized.
    assert ops["RotaryEmbedding"] == 2
    assert ops["MatMul"] == 0
    assert ops["DynamicQuantizeLinear"] == 2
    assert ops["MatMulInteger"] == 2

    rng = np.random.default_rng(11)
    x = rng.standard_normal((B, S, H)).astype(np.float32)
    angle = rng.standard_normal((B, S, half)).astype(np.float32)
    feeds = {"x": x, "angle": angle}
    _assert_close(_run(simplified, feeds), _run(quantized, feeds))


# --------------------------------------------------------------------------- #
# The models above are hand-built with ``onnx.helper`` (this module's own
# established convention, and every other pass-isolated test file's), which
# mirrors what a real trace produces but isn't one. These two tests trace a
# real ``torch.onnx.export`` model for each fusion instead, to confirm the
# composition on the genuine article -- requested during review of the
# hand-built-only version of this file.
# --------------------------------------------------------------------------- #
torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402  (after the torch importorskip guard)


def _repeat_kv(x, n_rep):
    B, NKV, S, D = x.shape
    if n_rep == 1:
        return x
    x = x[:, :, None, :, :].expand(B, NKV, n_rep, S, D)
    return x.reshape(B, NKV * n_rep, S, D)


class _TorchGQAAttention(nn.Module):
    def __init__(self, hidden=64, num_heads=8, num_kv_heads=2, seq_len=6):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden // num_heads
        self.n_rep = num_heads // num_kv_heads
        self.q_proj = nn.Linear(hidden, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden, bias=False)
        # A *baked* (buffer, not runtime-input) causal mask -- fuse_gqa only
        # fires when the additive mask is a provable constant matching the
        # causal pattern exactly (GroupQueryAttention applies causal masking
        # internally and unconditionally; see fuse_gqa.h's own top comment).
        mask = torch.zeros(1, 1, seq_len, seq_len)
        mask.masked_fill_(
            torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool(), -3.0e38
        )
        self.register_buffer("mask", mask, persistent=True)

    def forward(self, x):
        B, S, H = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        k = _repeat_kv(k, self.n_rep)
        v = _repeat_kv(v, self.n_rep)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        scores = scores + self.mask
        probs = torch.softmax(scores, dim=-1)
        ctx = (
            torch.matmul(probs, v)
            .transpose(1, 2)
            .reshape(B, S, self.num_heads * self.head_dim)
        )
        return self.o_proj(ctx)


def test_quantize_dynamic_composes_with_fuse_gqa_real_torch_export(tmp_path):
    B, S, H = 2, 6, 64
    model_module = _TorchGQAAttention(hidden=H, seq_len=S).eval()
    x = torch.randn(B, S, H)
    onnx_path = str(tmp_path / "gqa.onnx")
    torch.onnx.export(
        model_module,
        (x,),
        onnx_path,
        input_names=["x"],
        output_names=["y"],
        dynamo=False,
        opset_version=17,
    )
    exported = onnx.load(onnx_path)

    simplified, ok = onnxsim.simplify(exported)
    assert ok
    assert _op_counts(simplified)["GroupQueryAttention"] == 1

    quantized = onnxsim.quantize_dynamic(simplified)
    onnx.checker.check_model(quantized)
    ops = _op_counts(quantized)
    assert ops["GroupQueryAttention"] == 1
    assert ops["MatMul"] == 0

    x_np = x.numpy()
    _assert_close(_run(exported, {"x": x_np}), _run(simplified, {"x": x_np}), tol=1e-4)
    _assert_close(_run(simplified, {"x": x_np}), _run(quantized, {"x": x_np}))


def _torch_rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class _TorchRoPEProjection(nn.Module):
    def __init__(self, hidden=64, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, position_ids):
        B, S, H = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        angle = position_ids[:, :, None].float() * self.inv_freq[None, None, :]
        emb = torch.cat((angle, angle), dim=-1)
        cos, sin = emb.cos()[:, None], emb.sin()[:, None]
        q_embed = q * cos + _torch_rotate_half(q) * sin
        k_embed = k * cos + _torch_rotate_half(k) * sin
        return q_embed, k_embed


def test_quantize_dynamic_composes_with_fuse_rope_real_torch_export(tmp_path):
    from onnx import version_converter

    B, S, H = 2, 6, 64
    model_module = _TorchRoPEProjection(hidden=H).eval()
    x = torch.randn(B, S, H)
    position_ids = torch.arange(S)[None].expand(B, S).contiguous()
    onnx_path = str(tmp_path / "rope.onnx")
    torch.onnx.export(
        model_module,
        (x, position_ids),
        onnx_path,
        input_names=["x", "position_ids"],
        output_names=["q_embed", "k_embed"],
        dynamo=False,
        opset_version=17,
    )
    exported = onnx.load(onnx_path)
    # RotaryEmbedding needs opset >= 23 (see fuse_rope.h); the legacy
    # exporter's own ops are all opset<=17-compatible, so upgrading the
    # *declared* opset via onnx's own version_converter is safe here.
    exported23 = version_converter.convert_version(exported, 23)
    exported23.ir_version = 11
    onnx.checker.check_model(exported23)

    simplified, ok = onnxsim.simplify(exported23)
    assert ok
    assert _op_counts(simplified)["RotaryEmbedding"] == 2

    quantized = onnxsim.quantize_dynamic(simplified)
    onnx.checker.check_model(quantized)
    ops = _op_counts(quantized)
    assert ops["RotaryEmbedding"] == 2
    assert ops["MatMul"] == 0

    feeds = {"x": x.numpy(), "position_ids": position_ids.numpy().astype(np.int64)}
    _assert_close(_run(exported23, feeds), _run(simplified, feeds), tol=1e-4)
    _assert_close(_run(simplified, feeds), _run(quantized, feeds))
