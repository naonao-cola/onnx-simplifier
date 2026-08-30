"""Tests for ``onnxsim.reconstruct_gguf_graph`` -- building an ONNX graph
*and* hydrating its weights directly from a GGUF checkpoint's own declared
architecture (see onnxsim/gguf_reconstruct.py's module docstring for the
design).

The core claim under test: the graph this builds computes the same function
a Llama-family transformer described by the same hyperparameters does. That
is checked by an *independent* from-scratch numpy implementation of the
same tiny transformer (not a reuse of anything in gguf_reconstruct.py), run
against the identical hand-written GGUF checkpoint and compared to the
ONNX graph's own output via ``onnx.reference.ReferenceEvaluator`` --
mirroring test_import_gguf_weights.py's own "compute the expected result
independently, don't reuse the decoder under test" rigor.
"""

import struct

import numpy as np
import onnx
import onnx.utils
import pytest
from onnx.reference import ReferenceEvaluator

import onnxsim
from onnxsim.gguf_reconstruct import UnsupportedArchitectureError

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3

GGUF_METADATA_VALUE_TYPE_UINT32 = 4
GGUF_METADATA_VALUE_TYPE_FLOAT32 = 6
GGUF_METADATA_VALUE_TYPE_STRING = 8

GGML_TYPE_F32 = 0


def _string_bytes(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _kv_string(key, value):
    return (
        _string_bytes(key)
        + struct.pack("<I", GGUF_METADATA_VALUE_TYPE_STRING)
        + _string_bytes(value)
    )


def _kv_uint32(key, value):
    return (
        _string_bytes(key)
        + struct.pack("<I", GGUF_METADATA_VALUE_TYPE_UINT32)
        + struct.pack("<I", value)
    )


def _kv_float32(key, value):
    return (
        _string_bytes(key)
        + struct.pack("<I", GGUF_METADATA_VALUE_TYPE_FLOAT32)
        + struct.pack("<f", value)
    )


def _align_up(n, align=32):
    rem = n % align
    return n if rem == 0 else n + (align - rem)


def _write_gguf(path, kv_chunks, weights):
    """``weights``: dict of name -> numpy array, ONNX-shape order (already
    the order read_gguf_metadata reports -- see its own reversed-``ne``
    note). Every tensor is written as plain F32, matching GGML's
    innermost-dimension-first ``ne`` (the reverse of the array's own
    shape) over the SAME contiguous row-major bytes, exactly like
    tensor_pool_gguf.cpp's LoadGGUF/SaveGGUF reversal."""
    infos = b""
    data_chunks = []
    offset = 0
    for name, arr in weights.items():
        ne = list(reversed(arr.shape))
        # GGUF tensor data is little-endian regardless of host byte order
        # (same convention as ONNX's raw_data -- see
        # onnxsim/passes/endian_read.h's doc comment and
        # test_onnx_safetensors_input.py's mirror of this note).
        # `arr.astype(np.float32).tobytes()` would serialize in numpy's
        # native/host order instead, which is only correct by coincidence on
        # little-endian hosts.
        raw = arr.astype("<f4").tobytes()
        infos += _string_bytes(name)
        infos += struct.pack("<I", len(ne))
        for d in ne:
            infos += struct.pack("<Q", d)
        infos += struct.pack("<I", GGML_TYPE_F32)
        infos += struct.pack("<Q", offset)
        data_chunks.append((offset, raw))
        offset = _align_up(offset + len(raw))

    header = struct.pack(
        "<IIQQ", GGUF_MAGIC, GGUF_VERSION, len(weights), len(kv_chunks)
    )
    body = b"".join(kv_chunks)
    header_end = len(header) + len(body) + len(infos)
    data_section_start = _align_up(header_end)

    with open(path, "wb") as f:
        f.write(header)
        f.write(body)
        f.write(infos)
        f.write(b"\x00" * (data_section_start - header_end))
        pos = data_section_start
        for rel_offset, raw in data_chunks:
            abs_offset = data_section_start + rel_offset
            f.write(b"\x00" * (abs_offset - pos))
            f.write(raw)
            pos = abs_offset + len(raw)


def _rmsnorm(x, w, eps):
    var = np.mean(x * x, axis=-1, keepdims=True)
    return x / np.sqrt(var + eps) * w


def _rotate_half(x):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([-x2, x1], axis=-1)


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _build_tiny_llama_checkpoint(
    tmp_path, arch, n_head, n_head_kv, tie_embeddings, seed=0
):
    """A hand-built, real GGUF v3 file for a tiny (2-layer) Llama-family
    checkpoint, plus the config dict and weight dict needed to compute an
    independent numpy reference forward pass for it."""
    rng = np.random.default_rng(seed)
    n_embd = 8
    head_dim = n_embd // n_head
    n_layer = 2
    n_ff = 16
    vocab = 12
    eps = 1e-5
    freq_base = 10000.0

    def rand(*shape):
        return rng.standard_normal(shape).astype(np.float32) * 0.1

    weights = {"token_embd.weight": rand(vocab, n_embd)}
    for i in range(n_layer):
        p = f"blk.{i}"
        weights[f"{p}.attn_norm.weight"] = rand(n_embd) + 1.0
        weights[f"{p}.attn_q.weight"] = rand(n_embd, n_embd)
        weights[f"{p}.attn_q.bias"] = rand(n_embd)
        weights[f"{p}.attn_k.weight"] = rand(n_head_kv * head_dim, n_embd)
        weights[f"{p}.attn_k.bias"] = rand(n_head_kv * head_dim)
        weights[f"{p}.attn_v.weight"] = rand(n_head_kv * head_dim, n_embd)
        weights[f"{p}.attn_v.bias"] = rand(n_head_kv * head_dim)
        weights[f"{p}.attn_output.weight"] = rand(n_embd, n_embd)  # no bias
        weights[f"{p}.ffn_norm.weight"] = rand(n_embd) + 1.0
        weights[f"{p}.ffn_gate.weight"] = rand(n_ff, n_embd)
        weights[f"{p}.ffn_up.weight"] = rand(n_ff, n_embd)
        weights[f"{p}.ffn_down.weight"] = rand(n_embd, n_ff)  # no bias
    weights["output_norm.weight"] = rand(n_embd) + 1.0
    if not tie_embeddings:
        weights["output.weight"] = rand(vocab, n_embd)

    kv_chunks = [
        _kv_string("general.architecture", arch),
        _kv_uint32(f"{arch}.block_count", n_layer),
        _kv_uint32(f"{arch}.embedding_length", n_embd),
        _kv_uint32(f"{arch}.feed_forward_length", n_ff),
        _kv_uint32(f"{arch}.attention.head_count", n_head),
        _kv_uint32(f"{arch}.attention.head_count_kv", n_head_kv),
        _kv_float32(f"{arch}.attention.layer_norm_rms_epsilon", eps),
        _kv_float32(f"{arch}.rope.freq_base", freq_base),
    ]
    path = str(tmp_path / "tiny.gguf")
    _write_gguf(path, kv_chunks, weights)

    config = dict(
        n_embd=n_embd,
        head_dim=head_dim,
        n_layer=n_layer,
        n_ff=n_ff,
        vocab=vocab,
        eps=eps,
        freq_base=freq_base,
        n_head=n_head,
        n_head_kv=n_head_kv,
        tie_embeddings=tie_embeddings,
    )
    return path, weights, config


def _reference_forward(weights, config, input_ids, position_ids):
    """Independent from-scratch numpy implementation of the same tiny
    Llama-family transformer -- deliberately not sharing any code with
    onnxsim/gguf_reconstruct.py, so agreement between the two is a real
    correctness check, not a tautology."""
    n_embd, head_dim = config["n_embd"], config["head_dim"]
    n_head, n_head_kv = config["n_head"], config["n_head_kv"]
    n_layer, eps, freq_base = config["n_layer"], config["eps"], config["freq_base"]
    seq = input_ids.shape[1]

    x = weights["token_embd.weight"][input_ids]

    inv_freq = 1.0 / (freq_base ** (np.arange(0, head_dim, 2) / head_dim))
    freqs = position_ids[..., None].astype(np.float64) * inv_freq[None, None, :]
    emb = np.concatenate([freqs, freqs], axis=-1)
    cos = np.cos(emb).astype(np.float32)[:, None, :, :]
    sin = np.sin(emb).astype(np.float32)[:, None, :, :]

    n_rep = n_head // n_head_kv
    causal_mask = np.triu(np.full((seq, seq), -1e9, dtype=np.float32), k=1)
    batch = x.shape[0]

    for i in range(n_layer):
        p = f"blk.{i}"
        resid = x
        h = _rmsnorm(x, weights[f"{p}.attn_norm.weight"], eps)

        q = h @ weights[f"{p}.attn_q.weight"].T + weights[f"{p}.attn_q.bias"]
        k = h @ weights[f"{p}.attn_k.weight"].T + weights[f"{p}.attn_k.bias"]
        v = h @ weights[f"{p}.attn_v.weight"].T + weights[f"{p}.attn_v.bias"]

        q = q.reshape(batch, seq, n_head, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, seq, n_head_kv, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, seq, n_head_kv, head_dim).transpose(0, 2, 1, 3)

        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin

        k_rep = np.repeat(k, n_rep, axis=1)
        v_rep = np.repeat(v, n_rep, axis=1)

        scores = q @ k_rep.transpose(0, 1, 3, 2) / np.sqrt(head_dim)
        scores = scores + causal_mask
        attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn = attn / attn.sum(axis=-1, keepdims=True)
        out = attn @ v_rep

        out = out.transpose(0, 2, 1, 3).reshape(batch, seq, n_embd)
        out = out @ weights[f"{p}.attn_output.weight"].T
        x = resid + out

        resid = x
        h = _rmsnorm(x, weights[f"{p}.ffn_norm.weight"], eps)
        gate = h @ weights[f"{p}.ffn_gate.weight"].T
        up = h @ weights[f"{p}.ffn_up.weight"].T
        act = _silu(gate) * up
        down = act @ weights[f"{p}.ffn_down.weight"].T
        x = resid + down

    x = _rmsnorm(x, weights["output_norm.weight"], eps)
    lm_head = (
        weights["output.weight"]
        if not config["tie_embeddings"]
        else weights["token_embd.weight"]
    )
    return x @ lm_head.T


def _build_tiny_mixtral_checkpoint(tmp_path, n_expert, n_expert_used, seed=2):
    """A hand-built, real GGUF v3 file for a tiny (2-layer) Mixtral-style
    MoE checkpoint. llama.cpp gives this the same general.architecture
    ("llama") as a dense Llama checkpoint -- expert_count > 0 is what
    actually distinguishes it (see gguf_reconstruct.py's module docstring)
    -- so this otherwise mirrors _build_tiny_llama_checkpoint exactly,
    replacing only the per-layer FFN tensors with MoE ones."""
    rng = np.random.default_rng(seed)
    n_embd, n_head, n_head_kv = 8, 2, 2
    head_dim = n_embd // n_head
    n_layer = 2
    n_ff = 6
    vocab = 12
    eps = 1e-5
    freq_base = 10000.0

    def rand(*shape):
        return rng.standard_normal(shape).astype(np.float32) * 0.1

    weights = {"token_embd.weight": rand(vocab, n_embd)}
    for i in range(n_layer):
        p = f"blk.{i}"
        weights[f"{p}.attn_norm.weight"] = rand(n_embd) + 1.0
        weights[f"{p}.attn_q.weight"] = rand(n_embd, n_embd)
        weights[f"{p}.attn_k.weight"] = rand(n_head_kv * head_dim, n_embd)
        weights[f"{p}.attn_v.weight"] = rand(n_head_kv * head_dim, n_embd)
        weights[f"{p}.attn_output.weight"] = rand(n_embd, n_embd)
        weights[f"{p}.ffn_norm.weight"] = rand(n_embd) + 1.0
        weights[f"{p}.ffn_gate_inp.weight"] = rand(n_expert, n_embd)
        weights[f"{p}.ffn_gate_exps.weight"] = rand(n_expert, n_ff, n_embd)
        weights[f"{p}.ffn_up_exps.weight"] = rand(n_expert, n_ff, n_embd)
        weights[f"{p}.ffn_down_exps.weight"] = rand(n_expert, n_embd, n_ff)
    weights["output_norm.weight"] = rand(n_embd) + 1.0
    weights["output.weight"] = rand(vocab, n_embd)

    kv_chunks = [
        _kv_string("general.architecture", "llama"),
        _kv_uint32("llama.block_count", n_layer),
        _kv_uint32("llama.embedding_length", n_embd),
        _kv_uint32("llama.feed_forward_length", n_ff),
        _kv_uint32("llama.attention.head_count", n_head),
        _kv_uint32("llama.attention.head_count_kv", n_head_kv),
        _kv_float32("llama.attention.layer_norm_rms_epsilon", eps),
        _kv_float32("llama.rope.freq_base", freq_base),
        _kv_uint32("llama.expert_count", n_expert),
        _kv_uint32("llama.expert_used_count", n_expert_used),
    ]
    path = str(tmp_path / "tiny_mixtral.gguf")
    _write_gguf(path, kv_chunks, weights)

    config = dict(
        n_embd=n_embd,
        head_dim=head_dim,
        n_layer=n_layer,
        n_ff=n_ff,
        vocab=vocab,
        eps=eps,
        freq_base=freq_base,
        n_head=n_head,
        n_head_kv=n_head_kv,
        tie_embeddings=False,
        n_expert=n_expert,
        n_expert_used=n_expert_used,
    )
    return path, weights, config


def _reference_layer0_router_logits(weights, config, input_ids, position_ids):
    """Independent from-scratch numpy computation of just block 0's router
    logits (flattened to ``[batch*seq, n_expert]``, matching
    ``com.microsoft.MoE``'s own ``router_probs`` input shape) for the tiny
    Mixtral-style checkpoint -- as far as this file's independent-reference
    rigor can go without a CPU-runnable oracle for the MoE node itself (see
    the test using this for why). Shares the same attention/RoPE/RMSNorm
    math as ``_reference_forward`` (already independent of
    gguf_reconstruct.py), so this exercises everything upstream of the
    actual expert-routing/FFN math -- which is exhaustively validated
    against real onnxruntime elsewhere (generate_moe_function_templates.py,
    contrib_schemas_moe_test.cpp)."""
    n_embd, head_dim = config["n_embd"], config["head_dim"]
    n_head, n_head_kv = config["n_head"], config["n_head_kv"]
    eps, freq_base = config["eps"], config["freq_base"]
    seq = input_ids.shape[1]

    x = weights["token_embd.weight"][input_ids]

    inv_freq = 1.0 / (freq_base ** (np.arange(0, head_dim, 2) / head_dim))
    freqs = position_ids[..., None].astype(np.float64) * inv_freq[None, None, :]
    emb = np.concatenate([freqs, freqs], axis=-1)
    cos = np.cos(emb).astype(np.float32)[:, None, :, :]
    sin = np.sin(emb).astype(np.float32)[:, None, :, :]

    n_rep = n_head // n_head_kv
    causal_mask = np.triu(np.full((seq, seq), -1e9, dtype=np.float32), k=1)
    batch = x.shape[0]

    p = "blk.0"
    resid = x
    h = _rmsnorm(x, weights[f"{p}.attn_norm.weight"], eps)

    q = h @ weights[f"{p}.attn_q.weight"].T
    k = h @ weights[f"{p}.attn_k.weight"].T
    v = h @ weights[f"{p}.attn_v.weight"].T

    q = q.reshape(batch, seq, n_head, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(batch, seq, n_head_kv, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(batch, seq, n_head_kv, head_dim).transpose(0, 2, 1, 3)

    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin

    k_rep = np.repeat(k, n_rep, axis=1)
    v_rep = np.repeat(v, n_rep, axis=1)

    scores = q @ k_rep.transpose(0, 1, 3, 2) / np.sqrt(head_dim)
    scores = scores + causal_mask
    attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
    attn = attn / attn.sum(axis=-1, keepdims=True)
    out = attn @ v_rep

    out = out.transpose(0, 2, 1, 3).reshape(batch, seq, n_embd)
    out = out @ weights[f"{p}.attn_output.weight"].T
    x = resid + out

    h = _rmsnorm(x, weights[f"{p}.ffn_norm.weight"], eps)
    logits = h @ weights[f"{p}.ffn_gate_inp.weight"].T  # [batch, seq, n_expert]
    return logits.reshape(-1, logits.shape[-1])


def test_reconstructed_moe_graph_wires_up_the_moe_node_correctly(tmp_path):
    # ONNX Runtime's CPU MoE kernel rejects fc3 unconditionally (see
    # contrib_schemas_moe_test.cpp's top comment / PR #921), so unlike
    # test_reconstructed_graph_matches_independent_reference above, the
    # reconstructed graph can't be run end to end here. This instead checks
    # everything that CAN be checked without a CPU oracle: the MoE node's
    # own attributes/input wiring exactly match what llama.cpp's
    # build_moe_ffn does for this call shape (see gguf_reconstruct.py's
    # _moe_ffn docstring for the source references), onnx.checker and
    # onnxsim.simplify's shape inference accept the resulting graph, and
    # block 0's own router logits -- everything upstream of the actual
    # expert routing/FFN math, which is validated exhaustively elsewhere --
    # match a from-scratch numpy reference exactly, by requesting that
    # intermediate tensor as an output (onnxruntime then has no need to
    # execute any MoE node, since none is an ancestor of a block-0 tensor).
    ort = pytest.importorskip("onnxruntime")
    n_expert, n_expert_used = 4, 2
    path, weights, config = _build_tiny_mixtral_checkpoint(
        tmp_path, n_expert, n_expert_used
    )
    batch, seq = 1, 5

    model, skipped = onnxsim.reconstruct_gguf_graph(path, batch_size=batch, seq_len=seq)
    assert skipped == []
    onnx.checker.check_model(model)

    moe_nodes = [n for n in model.graph.node if n.op_type == "MoE"]
    assert len(moe_nodes) == config["n_layer"]
    node = moe_nodes[0]
    assert node.domain == "com.microsoft"
    attrs = {a.name: a for a in node.attribute}
    assert attrs["k"].i == n_expert_used
    assert attrs["activation_type"].s == b"silu"
    assert attrs["normalize_routing_weights"].i == 1
    # input, router_probs, fc1(gate)_w, fc1_bias(absent), fc2(down)_w,
    # fc2_bias(absent), fc3(up)_w -- see BuildMoEFunctionBody's own comment
    # in contrib_schemas.cpp for this naming (gate=fc1, up=fc3, down=fc2).
    assert list(node.input[2:]) == [
        "blk.0.ffn_gate_exps.weight",
        "",
        "blk.0.ffn_down_exps.weight",
        "",
        "blk.0.ffn_up_exps.weight",
    ]

    by_name = {i.name: i for i in model.graph.initializer}
    n_ff, n_embd = config["n_ff"], config["n_embd"]
    assert list(by_name["blk.0.ffn_gate_exps.weight"].dims) == [n_expert, n_ff, n_embd]
    assert list(by_name["blk.0.ffn_down_exps.weight"].dims) == [n_expert, n_embd, n_ff]
    assert list(by_name["blk.0.ffn_up_exps.weight"].dims) == [n_expert, n_ff, n_embd]

    simplified, ok = onnxsim.simplify(
        model, check_n=0, perform_optimization=False, skip_constant_folding=True
    )
    assert ok
    out_shape = simplified.graph.output[0].type.tensor_type.shape
    assert [d.dim_value for d in out_shape.dim] == [batch, seq, config["vocab"]]

    router_flat_name = next(
        n.output[0]
        for n in model.graph.node
        if n.op_type == "Reshape" and n.output[0].startswith("blk.0.moe_router_flat")
    )
    # onnxruntime's SequentialExecutor computes every node in the session's
    # execution plan regardless of which declared output a given Run() call
    # actually asks for -- appending router_flat_name as an extra output
    # alongside the real one still runs (and trips over) every MoE node.
    # onnx.utils.Extractor instead builds a real subgraph containing only
    # the nodes reachable from this one output, which excludes every MoE
    # node entirely (none is upstream of block 0's own router logits). It
    # requires the extraction point to already have a value_info entry.
    extractable = onnx.ModelProto()
    extractable.CopyFrom(model)
    extractable.graph.value_info.append(
        onnx.helper.make_tensor_value_info(
            router_flat_name, onnx.TensorProto.FLOAT, [None, n_expert]
        )
    )
    probe_model = onnx.utils.Extractor(extractable).extract_model(
        ["input_ids", "position_ids"], [router_flat_name]
    )

    rng = np.random.default_rng(3)
    input_ids = rng.integers(0, config["vocab"], size=(batch, seq)).astype(np.int64)
    position_ids = np.arange(seq, dtype=np.int64)[None, :].repeat(batch, axis=0)

    sess = ort.InferenceSession(
        probe_model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (router_logits,) = sess.run(
        [router_flat_name], {"input_ids": input_ids, "position_ids": position_ids}
    )
    ref_logits = _reference_layer0_router_logits(
        weights, config, input_ids, position_ids
    )
    np.testing.assert_allclose(router_logits, ref_logits, atol=1e-4, rtol=1e-4)


def test_moe_expert_used_count_exceeding_expert_count_raises(tmp_path):
    # A checkpoint with expert_used_count > expert_count is nonsensical
    # (can't pick more experts than exist) and must be rejected up front,
    # the same way test_tensor_shape_mismatch_raises rejects a corrupt
    # dense checkpoint -- built directly (not via
    # _build_tiny_mixtral_checkpoint, which always writes a consistent
    # pair) so only this one field is wrong.
    _, weights, _ = _build_tiny_mixtral_checkpoint(
        tmp_path, n_expert=2, n_expert_used=2
    )
    bad_path = str(tmp_path / "bad_expert_used_count.gguf")
    kv_chunks = [
        _kv_string("general.architecture", "llama"),
        _kv_uint32("llama.block_count", 2),
        _kv_uint32("llama.embedding_length", 8),
        _kv_uint32("llama.feed_forward_length", 6),
        _kv_uint32("llama.attention.head_count", 2),
        _kv_uint32("llama.attention.head_count_kv", 2),
        _kv_uint32("llama.expert_count", 2),
        _kv_uint32("llama.expert_used_count", 3),  # > expert_count
    ]
    _write_gguf(bad_path, kv_chunks, weights)
    with pytest.raises(UnsupportedArchitectureError, match="expert_used_count"):
        onnxsim.reconstruct_gguf_graph(bad_path)


@pytest.mark.parametrize("arch", ["llama", "qwen2", "mistral"])
@pytest.mark.parametrize("n_head,n_head_kv", [(2, 2), (2, 1)], ids=["mha", "gqa"])
@pytest.mark.parametrize("tie_embeddings", [False, True], ids=["untied", "tied"])
def test_reconstructed_graph_matches_independent_reference(
    tmp_path, arch, n_head, n_head_kv, tie_embeddings
):
    path, weights, config = _build_tiny_llama_checkpoint(
        tmp_path, arch, n_head, n_head_kv, tie_embeddings
    )
    batch, seq = 1, 5

    model, skipped = onnxsim.reconstruct_gguf_graph(path, batch_size=batch, seq_len=seq)
    assert skipped == []
    onnx.checker.check_model(model)

    rng = np.random.default_rng(1)
    input_ids = rng.integers(0, config["vocab"], size=(batch, seq)).astype(np.int64)
    position_ids = np.arange(seq, dtype=np.int64)[None, :].repeat(batch, axis=0)

    sess = ReferenceEvaluator(model)
    (onnx_logits,) = sess.run(
        None, {"input_ids": input_ids, "position_ids": position_ids}
    )
    ref_logits = _reference_forward(weights, config, input_ids, position_ids)

    assert onnx_logits.shape == (batch, seq, config["vocab"])
    np.testing.assert_allclose(onnx_logits, ref_logits, atol=1e-3, rtol=1e-3)


def test_simplify_folds_weight_transposes(tmp_path):
    """A weight's Transpose (see gguf_reconstruct.py's _linear) is constant
    -- once import_gguf_weights hydrates it, onnxsim.simplify should fold
    it away, leaving only the genuine runtime (q/k/v/out) transposes."""
    path, _, _ = _build_tiny_llama_checkpoint(
        tmp_path, "llama", 2, 1, tie_embeddings=False
    )
    model, _ = onnxsim.reconstruct_gguf_graph(path, batch_size=1, seq_len=5)
    before = sum(1 for n in model.graph.node if n.op_type == "Transpose")

    simplified, check_ok = onnxsim.simplify(model)

    assert check_ok
    after = sum(1 for n in simplified.graph.node if n.op_type == "Transpose")
    assert after < before


def test_unrecognized_architecture_raises(tmp_path):
    path = str(tmp_path / "gpt2.gguf")
    _write_gguf(path, [_kv_string("general.architecture", "gpt2")], {})
    with pytest.raises(UnsupportedArchitectureError, match="gpt2"):
        onnxsim.reconstruct_gguf_graph(path)


def test_missing_required_tensor_raises(tmp_path):
    path = str(tmp_path / "no_tensors.gguf")
    kv_chunks = [
        _kv_string("general.architecture", "llama"),
        _kv_uint32("llama.block_count", 1),
        _kv_uint32("llama.embedding_length", 8),
        _kv_uint32("llama.feed_forward_length", 16),
        _kv_uint32("llama.attention.head_count", 2),
    ]
    _write_gguf(path, kv_chunks, {})
    with pytest.raises(UnsupportedArchitectureError, match="token_embd.weight"):
        onnxsim.reconstruct_gguf_graph(path)


def test_tensor_shape_mismatch_raises(tmp_path):
    """token_embd.weight's declared embedding dim (7) disagrees with
    llama.embedding_length (8) -- a corrupt/inconsistent checkpoint must be
    rejected up front, not silently built into a graph with wrong shapes."""
    path = str(tmp_path / "bad_shape.gguf")
    kv_chunks = [
        _kv_string("general.architecture", "llama"),
        _kv_uint32("llama.block_count", 1),
        _kv_uint32("llama.embedding_length", 8),
        _kv_uint32("llama.feed_forward_length", 16),
        _kv_uint32("llama.attention.head_count", 2),
    ]
    _write_gguf(
        path, kv_chunks, {"token_embd.weight": np.zeros((12, 7), dtype=np.float32)}
    )
    with pytest.raises(UnsupportedArchitectureError, match="embedding_length"):
        onnxsim.reconstruct_gguf_graph(path)
