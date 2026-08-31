"""Tests for onnxsim.gguf_reconstruct's standalone gpt-oss attention block
(``_gpt_oss_attention_block``, ``_gpt_oss_yarn_cos_sin``,
``_gpt_oss_is_sliding_layer``, ``_gpt_oss_attn_mask``) -- see that module's
own "gpt-oss attention block" section header for exactly what these were
verified against in a real llama.cpp checkout, and for the corrections to
the original brief this implementation was scoped from.

None of these functions are reachable from
``onnxsim.reconstruct_gguf_graph`` yet (``gpt-oss`` is not in
``_SUPPORTED_ARCHITECTURES`` -- this is deliberately a standalone,
not-yet-wired-in deliverable), so this drives them directly: build a tiny
one-layer graph with ``_Builder`` the same way ``_reconstruct_llama_family``
would for a real layer, hydrate its placeholder weights from a small
hand-written real GGUF v3 file (mirroring test_gguf_reconstruct.py's own
byte-level rigor for ``_write_gguf``), run it through
``onnx.reference.ReferenceEvaluator``, and compare against an INDEPENDENT
from-scratch numpy implementation of gpt-oss's YaRN RoPE, sliding-window
masking, and attention-sink softmax -- re-derived directly from the
ggml/llama.cpp formulas (``ggml_rope_yarn_corr_dims``, ``rope_yarn``,
``ggml_compute_forward_soft_max_f32``'s sink branch,
``llama_hparams::is_masked_swa``), not by calling the onnxsim helpers under
test.
"""

import math
import struct

import numpy as np
import onnx
import onnx.helper
from onnx.reference import ReferenceEvaluator

from onnxsim.gguf_reconstruct import (
    _Builder,
    _gpt_oss_attention_block,
    _gpt_oss_attn_mask,
    _gpt_oss_is_sliding_layer,
    _gpt_oss_yarn_cos_sin,
)
from onnxsim.onnx_simplifier import import_gguf_weights

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
GGML_TYPE_F32 = 0

_OPSET = 17
_IR_VERSION = 8


def _string_bytes(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _align_up(n, align=32):
    rem = n % align
    return n if rem == 0 else n + (align - rem)


def _write_gguf(path, weights):
    """A real GGUF v3 file containing exactly ``weights`` (name -> numpy
    array, ONNX-shape order) and NO metadata kv pairs -- this test drives
    ``_gpt_oss_attention_block`` directly rather than through
    ``reconstruct_gguf_graph``, so only ``import_gguf_weights``'s
    tensor-data section is exercised, not any general.architecture/hparam
    reading. Otherwise byte-for-byte the same encoding as
    test_gguf_reconstruct.py's own ``_write_gguf`` (see that file's
    comments for why the ``ne`` reversal / little-endian raw bytes /
    32-byte alignment are done this way)."""
    infos = b""
    data_chunks = []
    offset = 0
    for name, arr in weights.items():
        ne = list(reversed(arr.shape))
        raw = arr.astype("<f4").tobytes()
        infos += _string_bytes(name)
        infos += struct.pack("<I", len(ne))
        for d in ne:
            infos += struct.pack("<Q", d)
        infos += struct.pack("<I", GGML_TYPE_F32)
        infos += struct.pack("<Q", offset)
        data_chunks.append((offset, raw))
        offset = _align_up(offset + len(raw))

    header = struct.pack("<IIQQ", GGUF_MAGIC, GGUF_VERSION, len(weights), 0)
    header_end = len(header) + len(infos)
    data_section_start = _align_up(header_end)

    with open(path, "wb") as f:
        f.write(header)
        f.write(infos)
        f.write(b"\x00" * (data_section_start - header_end))
        pos = data_section_start
        for rel_offset, raw in data_chunks:
            abs_offset = data_section_start + rel_offset
            f.write(b"\x00" * (abs_offset - pos))
            f.write(raw)
            pos = abs_offset + len(raw)


# A tiny config deliberately chosen so head_dim is NOT n_embd // n_head
# (n_embd/n_head = 8/4 = 2, but head_dim = 6) -- exercising the exact
# structural difference _gpt_oss_attention_block's own docstring (point 1)
# says every _SUPPORTED_ARCHITECTURES entry today gets wrong for gpt-oss.
N_EMBD = 8
N_HEAD = 4
N_HEAD_KV = 2
HEAD_DIM = 6
FREQ_BASE = 100.0
YARN_FACTOR = 4.0
YARN_ORIG_CTX = 8.0
YARN_BETA_FAST = 8.0
YARN_BETA_SLOW = 1.0
SLIDING_WINDOW = 3
SWA_PERIOD = 2
EPS = 1e-5
BATCH = 1
SEQ = 6


def _make_weights(seed):
    rng = np.random.default_rng(seed)

    def rand(*shape):
        return rng.standard_normal(shape).astype(np.float32) * 0.1

    n_embd_q = N_HEAD * HEAD_DIM
    n_embd_kv = N_HEAD_KV * HEAD_DIM
    return {
        "blk.0.attn_norm.weight": rand(N_EMBD) + 1.0,
        "blk.0.attn_q.weight": rand(n_embd_q, N_EMBD),
        "blk.0.attn_q.bias": rand(n_embd_q),
        "blk.0.attn_k.weight": rand(n_embd_kv, N_EMBD),
        "blk.0.attn_k.bias": rand(n_embd_kv),
        "blk.0.attn_v.weight": rand(n_embd_kv, N_EMBD),
        "blk.0.attn_v.bias": rand(n_embd_kv),
        "blk.0.attn_output.weight": rand(N_EMBD, n_embd_q),
        "blk.0.attn_output.bias": rand(N_EMBD),
        "blk.0.attn_sinks.weight": rand(N_HEAD),
    }


def _build_one_layer_model(weights, layer_idx):
    """Builds a runnable one-layer ONNX model around
    ``_gpt_oss_attention_block``/``_gpt_oss_yarn_cos_sin`` -- the graph
    STRUCTURE comes entirely from the onnxsim code under test (this
    function only supplies inputs/outputs and a placeholder-declaring
    closure pair, the same role ``_reconstruct_llama_family``'s own
    ``declare``/``declare_optional`` closures play there)."""
    b = _Builder()

    def declare(name, expected_shape):
        assert list(weights[name].shape) == expected_shape, (
            name,
            weights[name].shape,
            expected_shape,
        )
        b.placeholder_weight(name, expected_shape, onnx.TensorProto.FLOAT)
        return name

    def declare_optional(name, expected_shape):
        return declare(name, expected_shape) if name in weights else None

    hidden = "hidden_states"
    position_ids = "position_ids"

    cos_b, sin_b = _gpt_oss_yarn_cos_sin(
        b,
        position_ids,
        HEAD_DIM,
        FREQ_BASE,
        YARN_FACTOR,
        YARN_ORIG_CTX,
        YARN_BETA_FAST,
        YARN_BETA_SLOW,
        "rope",
    )
    out = _gpt_oss_attention_block(
        b,
        hidden,
        "blk.0",
        layer_idx,
        N_EMBD,
        N_HEAD,
        N_HEAD_KV,
        HEAD_DIM,
        cos_b,
        sin_b,
        SLIDING_WINDOW,
        SWA_PERIOD,
        BATCH,
        SEQ,
        EPS,
        declare,
        declare_optional,
    )

    graph = onnx.helper.make_graph(
        b.nodes,
        "gpt_oss_attn_test",
        [
            onnx.helper.make_tensor_value_info(
                hidden, onnx.TensorProto.FLOAT, [BATCH, SEQ, N_EMBD]
            ),
            onnx.helper.make_tensor_value_info(
                position_ids, onnx.TensorProto.INT64, [BATCH, SEQ]
            ),
        ],
        [
            onnx.helper.make_tensor_value_info(
                out, onnx.TensorProto.FLOAT, [BATCH, SEQ, N_EMBD]
            )
        ],
        initializer=b.initializers,
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", _OPSET)]
    )
    model.ir_version = _IR_VERSION
    return model


def _ref_yarn_cos_sin(position_ids, head_dim):
    """Independent from-scratch re-derivation of gpt-oss's YaRN RoPE,
    directly from ggml/src/ggml.c's ``ggml_rope_yarn_corr_dims`` and
    ggml/src/ggml-cpu/ops.cpp's ``rope_yarn`` -- written without looking at
    ``_gpt_oss_yarn_cos_sin``'s own code, only at the same llama.cpp source
    it cites."""
    half = head_dim // 2
    j = np.arange(half, dtype=np.float64)
    inv_freq = 1.0 / (FREQ_BASE ** (2.0 * j / head_dim))

    def corr_dim(n_rot):
        return (
            head_dim
            * math.log(YARN_ORIG_CTX / (n_rot * 2.0 * math.pi))
            / (2.0 * math.log(FREQ_BASE))
        )

    low = max(0.0, math.floor(corr_dim(YARN_BETA_FAST)))
    high = min(float(head_dim - 1), math.ceil(corr_dim(YARN_BETA_SLOW)))
    ramp = 1.0 - np.clip((j - low) / max(1e-3, high - low), 0.0, 1.0)

    freq_scale = 1.0 / YARN_FACTOR
    theta_extrap = position_ids[..., None].astype(np.float64) * inv_freq[None, None, :]
    theta_interp = freq_scale * theta_extrap
    theta = theta_interp * (1 - ramp) + theta_extrap * ramp
    mscale = 1.0 if YARN_FACTOR <= 1.0 else 0.1 * math.log(YARN_FACTOR) + 1.0

    emb = np.concatenate([theta, theta], axis=-1)
    cos = (np.cos(emb) * mscale).astype(np.float32)[:, None, :, :]
    sin = (np.sin(emb) * mscale).astype(np.float32)[:, None, :, :]
    return cos, sin


def _ref_rotate_half(x):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([-x2, x1], axis=-1)


def _ref_attn_mask(sliding):
    """Independent re-derivation of llama_hparams::is_masked_swa's
    LLAMA_SWA_TYPE_STANDARD rule, written directly from that source
    (masked when key-to-query distance p1-p0 >= n_swa), not from
    ``_gpt_oss_attn_mask``'s code."""
    mask = np.zeros((SEQ, SEQ), dtype=np.float32)
    for p1 in range(SEQ):  # query position
        for p0 in range(SEQ):  # key position
            masked = p0 > p1  # causal
            if sliding and (p1 - p0) >= SLIDING_WINDOW:
                masked = True
            if masked:
                mask[p1, p0] = -1e9
    return mask


def _ref_forward(weights, input_hidden, position_ids, layer_idx):
    """Independent from-scratch numpy forward pass for one gpt-oss
    attention block -- RMSNorm, QKV projection, YaRN RoPE, GQA scores,
    sliding-window-or-causal mask, attention-sink softmax
    (``softmax_i = exp(s_i-m)/(sum_j exp(s_j-m) + exp(sink-m))``,
    ``m = max(max(s), sink)`` -- read directly off
    ggml_compute_forward_soft_max_f32's sink branch, not off
    ``_gpt_oss_attention_block``'s code), output projection, residual add.
    """
    resid = input_hidden
    var = np.mean(input_hidden * input_hidden, axis=-1, keepdims=True)
    h = input_hidden / np.sqrt(var + EPS) * weights["blk.0.attn_norm.weight"]

    q = h @ weights["blk.0.attn_q.weight"].T + weights["blk.0.attn_q.bias"]
    k = h @ weights["blk.0.attn_k.weight"].T + weights["blk.0.attn_k.bias"]
    v = h @ weights["blk.0.attn_v.weight"].T + weights["blk.0.attn_v.bias"]

    q = q.reshape(BATCH, SEQ, N_HEAD, HEAD_DIM).transpose(0, 2, 1, 3)
    k = k.reshape(BATCH, SEQ, N_HEAD_KV, HEAD_DIM).transpose(0, 2, 1, 3)
    v = v.reshape(BATCH, SEQ, N_HEAD_KV, HEAD_DIM).transpose(0, 2, 1, 3)

    cos, sin = _ref_yarn_cos_sin(position_ids, HEAD_DIM)
    q = q * cos + _ref_rotate_half(q) * sin
    k = k * cos + _ref_rotate_half(k) * sin

    n_rep = N_HEAD // N_HEAD_KV
    q5 = q.reshape(BATCH, N_HEAD_KV, n_rep, SEQ, HEAD_DIM)
    k5 = k[:, :, None, :, :]
    v5 = v[:, :, None, :, :]

    scores = q5 @ np.swapaxes(k5, -1, -2) / math.sqrt(HEAD_DIM)

    sliding = _gpt_oss_is_sliding_layer(layer_idx, SWA_PERIOD)
    mask = _ref_attn_mask(sliding)
    scores = scores + mask

    sinks = weights["blk.0.attn_sinks.weight"].reshape(N_HEAD_KV, n_rep)
    sinks_b = sinks[None, :, :, None, None]

    row_max = np.max(scores, axis=-1, keepdims=True)
    m = np.maximum(row_max, sinks_b)
    exp_s = np.exp(scores - m)
    sum_exp = np.sum(exp_s, axis=-1, keepdims=True)
    sink_exp = np.exp(sinks_b - m)
    attn = exp_s / (sum_exp + sink_exp)

    out = attn @ v5  # [B, n_head_kv, n_rep, S, head_dim]
    # Merge (n_head_kv, n_rep) -> n_head (row-major -- the same convention
    # _gpt_oss_attention_block's q5/k5/v5 broadcasting split relies on),
    # then move the head axis after S and flatten heads*head_dim -- matches
    # the onnx graph's reshape(->n_head,S,head_dim)/transpose/reshape
    # sequence exactly (merging is associative for row-major reshapes, so
    # doing it in one transpose+reshape here is equivalent).
    out = out.transpose(0, 3, 1, 2, 4).reshape(BATCH, SEQ, N_HEAD * HEAD_DIM)
    out = (
        out @ weights["blk.0.attn_output.weight"].T + weights["blk.0.attn_output.bias"]
    )

    return resid + out


def test_gpt_oss_sliding_layer_pattern_matches_config():
    # openai/gpt-oss-20b's own config.json (fetched from the HF Hub) lists
    # layer_types = ["sliding_attention", "full_attention", ...] x12 for its
    # 24 layers -- i.e. even layer indices are local/SWA, odd are
    # full/global, with the default period-2 pattern
    # (src/models/openai-moe.cpp never writes a
    # sliding_window_pattern override into a real gpt-oss GGUF).
    expected = [i % 2 == 0 for i in range(24)]
    actual = [_gpt_oss_is_sliding_layer(i, swa_period=2) for i in range(24)]
    assert actual == expected


def test_gpt_oss_attn_mask_is_banded_for_sliding_layers():
    mask = _gpt_oss_attn_mask(seq_len=6, sliding=True, n_swa=3)
    # query pos 5 (0-indexed) may attend to key positions 3,4,5 only
    # (n_swa=3 most recent positions, causal) -- positions 0,1,2 masked.
    allowed = mask[5] == 0.0
    assert list(allowed) == [False, False, False, True, True, True]
    # every row still respects causality (no attending to the future)
    for q in range(6):
        assert np.all(mask[q, q + 1 :] == -1e9)


def test_gpt_oss_attn_mask_is_plain_causal_for_full_layers():
    mask = _gpt_oss_attn_mask(seq_len=6, sliding=False, n_swa=3)
    expected = np.triu(np.full((6, 6), -1e9, dtype=np.float32), k=1)
    np.testing.assert_array_equal(mask, expected)


def test_gpt_oss_yarn_degenerates_to_plain_rope_at_factor_one():
    # yarn_factor<=1.0 should be indistinguishable from plain (non-YaRN)
    # RoPE: freq_scale==1 makes the interpolated/extrapolated angle
    # identical regardless of the ramp, and mscale's scale<=1 branch is
    # exactly 1.0 -- see _gpt_oss_yarn_cos_sin's own docstring.
    b = _Builder()
    cos_name, sin_name = _gpt_oss_yarn_cos_sin(
        b, "position_ids", HEAD_DIM, FREQ_BASE, 1.0, YARN_ORIG_CTX, 8.0, 1.0, "rope"
    )
    graph = onnx.helper.make_graph(
        b.nodes,
        "yarn_degenerate_test",
        [
            onnx.helper.make_tensor_value_info(
                "position_ids", onnx.TensorProto.INT64, [1, SEQ]
            )
        ],
        [
            onnx.helper.make_tensor_value_info(cos_name, onnx.TensorProto.FLOAT, None),
            onnx.helper.make_tensor_value_info(sin_name, onnx.TensorProto.FLOAT, None),
        ],
        initializer=b.initializers,
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", _OPSET)]
    )
    model.ir_version = _IR_VERSION

    position_ids = np.arange(SEQ, dtype=np.int64)[None, :]
    sess = ReferenceEvaluator(model)
    cos, sin = sess.run(None, {"position_ids": position_ids})

    half = HEAD_DIM // 2
    inv_freq = 1.0 / (FREQ_BASE ** (np.arange(0, HEAD_DIM, 2) / HEAD_DIM))
    freqs = position_ids[..., None].astype(np.float64) * inv_freq[None, None, :]
    emb = np.concatenate([freqs, freqs], axis=-1)
    expected_cos = np.cos(emb).astype(np.float32)[:, None, :, :]
    expected_sin = np.sin(emb).astype(np.float32)[:, None, :, :]

    np.testing.assert_allclose(cos, expected_cos, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(sin, expected_sin, atol=1e-5, rtol=1e-5)
    assert half == 3  # sanity: HEAD_DIM is even, as RoPE requires


def test_gpt_oss_attention_block_matches_independent_reference_sliding_layer(
    tmp_path,
):
    _check_layer_against_reference(tmp_path, layer_idx=0)  # even -> sliding


def test_gpt_oss_attention_block_matches_independent_reference_full_layer(tmp_path):
    _check_layer_against_reference(tmp_path, layer_idx=1)  # odd -> full/global


def _check_layer_against_reference(tmp_path, layer_idx):
    weights = _make_weights(seed=layer_idx + 1)
    model = _build_one_layer_model(weights, layer_idx)

    gguf_path = str(tmp_path / f"tiny_gpt_oss_layer{layer_idx}.gguf")
    _write_gguf(gguf_path, weights)
    model, skipped = import_gguf_weights(model, gguf_path)
    assert skipped == []
    onnx.checker.check_model(model)

    rng = np.random.default_rng(100 + layer_idx)
    hidden = rng.standard_normal((BATCH, SEQ, N_EMBD)).astype(np.float32) * 0.5
    position_ids = np.arange(SEQ, dtype=np.int64)[None, :].repeat(BATCH, axis=0)

    sess = ReferenceEvaluator(model)
    (onnx_out,) = sess.run(
        None, {"hidden_states": hidden, "position_ids": position_ids}
    )

    ref_out = _ref_forward(weights, hidden, position_ids, layer_idx)

    assert onnx_out.shape == (BATCH, SEQ, N_EMBD)
    np.testing.assert_allclose(onnx_out, ref_out, atol=1e-4, rtol=1e-4)


def test_gpt_oss_attention_block_uses_independent_head_dim_not_embd_over_head_count():
    # n_embd // n_head == 2 here, but HEAD_DIM == 6 -- confirming
    # _gpt_oss_attention_block declares Q/K/V/O shapes from head_dim
    # directly rather than (incorrectly) deriving it as n_embd // n_head,
    # which every _SUPPORTED_ARCHITECTURES entry does today but gpt-oss's
    # own checkpoint shapes do not support (see the module docstring's
    # point 1: gpt-oss-20b itself has n_embd/n_head=45 != head_dim=64).
    assert N_EMBD // N_HEAD != HEAD_DIM

    weights = _make_weights(seed=42)
    model = _build_one_layer_model(weights, layer_idx=0)
    by_name = {i.name: i for i in model.graph.initializer}
    n_embd_q = N_HEAD * HEAD_DIM
    n_embd_kv = N_HEAD_KV * HEAD_DIM
    assert list(by_name["blk.0.attn_q.weight"].dims) == [n_embd_q, N_EMBD]
    assert list(by_name["blk.0.attn_k.weight"].dims) == [n_embd_kv, N_EMBD]
    assert list(by_name["blk.0.attn_output.weight"].dims) == [N_EMBD, n_embd_q]
    assert list(by_name["blk.0.attn_sinks.weight"].dims) == [N_HEAD]
