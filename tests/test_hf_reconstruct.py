"""Tests for ``onnxsim.reconstruct_hf_graph`` -- building an ONNX graph
*and* hydrating its weights directly from a HuggingFace checkpoint
directory's own declared architecture (see onnxsim/hf_reconstruct.py's
module docstring for the design).

Same rigor as test_gguf_reconstruct.py: the core claim under test is that
the graph this builds computes the same function a Llama-family
transformer described by the same hyperparameters does, checked against an
*independent* from-scratch numpy implementation (not a reuse of anything in
hf_reconstruct.py itself), run against the identical hand-written
safetensors checkpoint and compared to the ONNX graph's own output via
``onnx.reference.ReferenceEvaluator``.
"""

import json
import struct

import numpy as np
import onnx
import pytest
from onnx.reference import ReferenceEvaluator

import onnxsim
from onnxsim.gguf_reconstruct import UnsupportedArchitectureError


def _write_safetensors(path, tensors):
    """A real, valid ``.safetensors`` file: an 8-byte little-endian
    header-length prefix, a JSON header (name -> dtype/shape/byte-offset),
    then the raw tensor bytes back to back -- the exact format
    hf_reconstruct.py's own ``_read_safetensors_header``/``_read_tensor``
    parse, written independently here rather than via the ``safetensors``
    package (kept out of this project's test dependencies)."""
    header = {}
    offset = 0
    blobs = []
    for name, arr in tensors.items():
        nbytes = arr.nbytes
        header[name] = {
            "dtype": "F32",
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
        blobs.append(arr.astype("<f4").tobytes())
    header_bytes = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for b in blobs:
            f.write(b)


def _rmsnorm(x, w, eps):
    var = np.mean(x * x, axis=-1, keepdims=True)
    return x / np.sqrt(var + eps) * w


def _rotate_half(x):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([-x2, x1], axis=-1)


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _build_tiny_hf_checkpoint(
    tmp_path,
    model_type,
    n_head,
    n_head_kv,
    tie_word_embeddings,
    use_qk_norm=False,
    head_dim_override=None,
    seed=0,
):
    """A tiny (2-layer) HF-shaped checkpoint directory (config.json +
    model.safetensors), plus the weight dict and config needed to compute
    an independent numpy reference forward pass for it."""
    rng = np.random.default_rng(seed)
    n_embd = 8
    head_dim = head_dim_override or (n_embd // n_head)
    n_layer = 2
    n_ff = 16
    vocab = 12
    eps = 1e-5
    freq_base = 10000.0
    n_embd_q = n_head * head_dim
    n_embd_kv = n_head_kv * head_dim

    def rand(*shape):
        return rng.standard_normal(shape).astype(np.float32) * 0.1

    weights = {"model.embed_tokens.weight": rand(vocab, n_embd)}
    for i in range(n_layer):
        p = f"model.layers.{i}"
        weights[f"{p}.input_layernorm.weight"] = rand(n_embd) + 1.0
        weights[f"{p}.self_attn.q_proj.weight"] = rand(n_embd_q, n_embd)
        weights[f"{p}.self_attn.k_proj.weight"] = rand(n_embd_kv, n_embd)
        weights[f"{p}.self_attn.v_proj.weight"] = rand(n_embd_kv, n_embd)
        weights[f"{p}.self_attn.o_proj.weight"] = rand(n_embd, n_embd_q)
        if use_qk_norm:
            weights[f"{p}.self_attn.q_norm.weight"] = rand(head_dim) + 1.0
            weights[f"{p}.self_attn.k_norm.weight"] = rand(head_dim) + 1.0
        weights[f"{p}.post_attention_layernorm.weight"] = rand(n_embd) + 1.0
        weights[f"{p}.mlp.gate_proj.weight"] = rand(n_ff, n_embd)
        weights[f"{p}.mlp.up_proj.weight"] = rand(n_ff, n_embd)
        weights[f"{p}.mlp.down_proj.weight"] = rand(n_embd, n_ff)
    weights["model.norm.weight"] = rand(n_embd) + 1.0
    if not tie_word_embeddings:
        weights["lm_head.weight"] = rand(vocab, n_embd)

    hf_dir = tmp_path / "tiny_hf"
    hf_dir.mkdir()
    config = {
        "model_type": model_type,
        "hidden_size": n_embd,
        "num_hidden_layers": n_layer,
        "intermediate_size": n_ff,
        "num_attention_heads": n_head,
        "num_key_value_heads": n_head_kv,
        "rms_norm_eps": eps,
        "rope_theta": freq_base,
        "vocab_size": vocab,
        "tie_word_embeddings": tie_word_embeddings,
        "attention_bias": False,
    }
    if head_dim_override is not None:
        config["head_dim"] = head_dim_override
    with open(hf_dir / "config.json", "w") as f:
        json.dump(config, f)
    _write_safetensors(str(hf_dir / "model.safetensors"), weights)

    ref_config = dict(
        n_embd=n_embd,
        head_dim=head_dim,
        n_layer=n_layer,
        vocab=vocab,
        eps=eps,
        freq_base=freq_base,
        n_head=n_head,
        n_head_kv=n_head_kv,
        tie_embeddings=tie_word_embeddings,
        use_qk_norm=use_qk_norm,
    )
    return str(hf_dir), weights, ref_config


def _reference_forward(weights, config, input_ids, position_ids):
    """Independent from-scratch numpy implementation of the same tiny
    Llama-family (optionally Qwen3-style QK-normed) transformer --
    deliberately not sharing any code with onnxsim/hf_reconstruct.py."""
    head_dim = config["head_dim"]
    n_head, n_head_kv = config["n_head"], config["n_head_kv"]
    n_layer, eps, freq_base = config["n_layer"], config["eps"], config["freq_base"]
    seq = input_ids.shape[1]
    batch = input_ids.shape[0]

    x = weights["model.embed_tokens.weight"][input_ids]

    inv_freq = 1.0 / (freq_base ** (np.arange(0, head_dim, 2) / head_dim))
    freqs = position_ids[..., None].astype(np.float64) * inv_freq[None, None, :]
    emb = np.concatenate([freqs, freqs], axis=-1)
    cos = np.cos(emb).astype(np.float32)[:, None, :, :]
    sin = np.sin(emb).astype(np.float32)[:, None, :, :]

    n_rep = n_head // n_head_kv
    causal_mask = np.triu(np.full((seq, seq), -1e9, dtype=np.float32), k=1)

    for i in range(n_layer):
        p = f"model.layers.{i}"
        resid = x
        h = _rmsnorm(x, weights[f"{p}.input_layernorm.weight"], eps)

        q = h @ weights[f"{p}.self_attn.q_proj.weight"].T
        k = h @ weights[f"{p}.self_attn.k_proj.weight"].T
        v = h @ weights[f"{p}.self_attn.v_proj.weight"].T

        q = q.reshape(batch, seq, n_head, head_dim)
        k = k.reshape(batch, seq, n_head_kv, head_dim)
        v = v.reshape(batch, seq, n_head_kv, head_dim)

        if config["use_qk_norm"]:
            q = _rmsnorm(q, weights[f"{p}.self_attn.q_norm.weight"], eps)
            k = _rmsnorm(k, weights[f"{p}.self_attn.k_norm.weight"], eps)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin

        k_rep = np.repeat(k, n_rep, axis=1)
        v_rep = np.repeat(v, n_rep, axis=1)

        scores = q @ k_rep.transpose(0, 1, 3, 2) / np.sqrt(head_dim)
        scores = scores + causal_mask
        attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn = attn / attn.sum(axis=-1, keepdims=True)
        out = attn @ v_rep

        out = out.transpose(0, 2, 1, 3).reshape(batch, seq, n_head * head_dim)
        out = out @ weights[f"{p}.self_attn.o_proj.weight"].T
        x = resid + out

        resid = x
        h = _rmsnorm(x, weights[f"{p}.post_attention_layernorm.weight"], eps)
        gate = h @ weights[f"{p}.mlp.gate_proj.weight"].T
        up = h @ weights[f"{p}.mlp.up_proj.weight"].T
        act = _silu(gate) * up
        down = act @ weights[f"{p}.mlp.down_proj.weight"].T
        x = resid + down

    x = _rmsnorm(x, weights["model.norm.weight"], eps)
    lm_head = (
        weights["model.embed_tokens.weight"]
        if config["tie_embeddings"]
        else weights["lm_head.weight"]
    )
    return x @ lm_head.T


@pytest.mark.parametrize("model_type", ["llama", "mistral", "qwen2"])
@pytest.mark.parametrize("n_head,n_head_kv", [(2, 2), (2, 1)], ids=["mha", "gqa"])
@pytest.mark.parametrize("tie_word_embeddings", [False, True], ids=["untied", "tied"])
def test_reconstructed_graph_matches_independent_reference(
    tmp_path, model_type, n_head, n_head_kv, tie_word_embeddings
):
    hf_dir, weights, config = _build_tiny_hf_checkpoint(
        tmp_path, model_type, n_head, n_head_kv, tie_word_embeddings
    )
    batch, seq = 1, 5

    model = onnxsim.reconstruct_hf_graph(hf_dir, batch_size=batch, seq_len=seq)
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


@pytest.mark.parametrize("n_head,n_head_kv", [(4, 4), (4, 2)], ids=["mha", "gqa"])
def test_qwen3_qk_norm_and_head_dim_override_match_reference(
    tmp_path, n_head, n_head_kv
):
    """Qwen3's two real differences from the shared Llama-family shape,
    confirmed against the real ``transformers`` ``modeling_qwen3.py``
    source (see hf_reconstruct.py's module docstring): per-head QK-RMSNorm,
    and ``head_dim`` decoupled from ``hidden_size // num_attention_heads``
    (here: hidden_size=8, num_attention_heads=4 would imply head_dim=2, but
    the config explicitly overrides it to 6 -- mirroring the real
    Qwen/Qwen3-0.6B checkpoint, whose head_dim=128 similarly isn't
    hidden_size // num_attention_heads)."""
    hf_dir, weights, config = _build_tiny_hf_checkpoint(
        tmp_path,
        "qwen3",
        n_head,
        n_head_kv,
        tie_word_embeddings=False,
        use_qk_norm=True,
        head_dim_override=6,
    )
    batch, seq = 1, 5

    model = onnxsim.reconstruct_hf_graph(hf_dir, batch_size=batch, seq_len=seq)
    onnx.checker.check_model(model)

    rng = np.random.default_rng(1)
    input_ids = rng.integers(0, config["vocab"], size=(batch, seq)).astype(np.int64)
    position_ids = np.arange(seq, dtype=np.int64)[None, :].repeat(batch, axis=0)

    sess = ReferenceEvaluator(model)
    (onnx_logits,) = sess.run(
        None, {"input_ids": input_ids, "position_ids": position_ids}
    )
    ref_logits = _reference_forward(weights, config, input_ids, position_ids)
    np.testing.assert_allclose(onnx_logits, ref_logits, atol=1e-3, rtol=1e-3)


def test_bf16_weight_matches_float32_reference_after_cast(tmp_path):
    """Confirmed real bug this guards against (see hf_reconstruct.py's
    module docstring): a real Qwen/Qwen3-0.6B checkpoint stores every
    weight as BF16, and mixing a preserved-BF16 initializer into this
    module's FLOAT32-only op-building helpers produced a mixed-dtype node
    onnxruntime rejected outright. The fix casts each non-FLOAT32 weight to
    FLOAT32 in the graph (declare()) rather than upcasting the stored
    bytes (which would double the model's size -- also confirmed to matter
    for a real ~1.5GB checkpoint, see the same docstring). This test
    represents one weight as BF16 (rounding a hand-picked exact-in-BF16
    value) and checks the reconstructed graph's output exactly matches an
    all-FLOAT32 checkpoint with the same (BF16-representable) values --
    i.e. the Cast is wired in correctly, not merely present.
    """
    hf_dir, weights, config = _build_tiny_hf_checkpoint(
        tmp_path, "llama", 2, 2, tie_word_embeddings=False
    )
    # Round layer-0's input_layernorm weight to values exactly representable
    # in BF16 (truncate the low 16 mantissa bits), so the BF16 round-trip
    # introduces no precision loss relative to the FLOAT32 reference this
    # is compared against.
    name = "model.layers.0.input_layernorm.weight"
    f32 = weights[name].astype("<f4")
    bf16_bits = (f32.view("<u4") >> 16).astype("<u2")
    # NumPy's binary operators (here, `<< 16`) return a *native*-byte-order
    # result even when their operand is explicitly little-endian-tagged (the
    # `.astype("<u2")` two lines up survives this because `.astype()` itself
    # is byte-order-aware and re-encodes; a bare arithmetic op is not) -- so
    # on a big-endian host, `(bf16_bits.astype("<u4") << 16)` is a *native*
    # (big-endian) uint32 holding the correct bit pattern, and a bare
    # `.view("<f4")` on top of that reinterprets those big-endian bytes as
    # little-endian, producing a denormal near-zero garbage value instead of
    # the intended widened float. Confirmed on real s390x hardware (this test
    # passes on little-endian, where native and "<u4" already coincide, which
    # is why it went unnoticed there). The extra `.astype("<u4")` re-encodes
    # the value into genuinely little-endian bytes before the final `.view()`
    # reinterprets them -- a no-op copy on a little-endian host, a real
    # byte-order fix on a big-endian one.
    weights[name] = (bf16_bits.astype("<u4") << 16).astype("<u4").view("<f4")

    header = {}
    offset = 0
    blobs = []
    for tensor_name, arr in weights.items():
        if tensor_name == name:
            nbytes = arr.size * 2
            header[tensor_name] = {
                "dtype": "BF16",
                "shape": list(arr.shape),
                "data_offsets": [offset, offset + nbytes],
            }
            blobs.append(bf16_bits.tobytes())
        else:
            nbytes = arr.nbytes
            header[tensor_name] = {
                "dtype": "F32",
                "shape": list(arr.shape),
                "data_offsets": [offset, offset + nbytes],
            }
            blobs.append(arr.astype("<f4").tobytes())
        offset += nbytes
    header_bytes = json.dumps(header).encode("utf-8")
    with open(f"{hf_dir}/model.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for b in blobs:
            f.write(b)

    batch, seq = 1, 5
    model = onnxsim.reconstruct_hf_graph(hf_dir, batch_size=batch, seq_len=seq)
    onnx.checker.check_model(model)

    rng = np.random.default_rng(1)
    input_ids = rng.integers(0, config["vocab"], size=(batch, seq)).astype(np.int64)
    position_ids = np.arange(seq, dtype=np.int64)[None, :].repeat(batch, axis=0)

    sess = ReferenceEvaluator(model)
    (onnx_logits,) = sess.run(
        None, {"input_ids": input_ids, "position_ids": position_ids}
    )
    ref_logits = _reference_forward(weights, config, input_ids, position_ids)
    np.testing.assert_allclose(onnx_logits, ref_logits, atol=1e-4, rtol=1e-4)


def test_sharded_checkpoint_loads_correctly(tmp_path):
    """A checkpoint split across model-NNNNN-of-MMMMM.safetensors shards,
    indexed by model.safetensors.index.json -- the layout real large HF
    checkpoints ship as (a single model.safetensors, used everywhere else
    in this file, only fits small checkpoints)."""
    hf_dir, weights, config = _build_tiny_hf_checkpoint(
        tmp_path, "llama", 2, 2, tie_word_embeddings=False
    )
    import os

    single_path = f"{hf_dir}/model.safetensors"
    names = list(weights.keys())
    half = len(names) // 2
    shard_names = {
        "model-00001-of-00002.safetensors": names[:half],
        "model-00002-of-00002.safetensors": names[half:],
    }
    weight_map = {}
    for shard, shard_tensor_names in shard_names.items():
        _write_safetensors(
            f"{hf_dir}/{shard}", {n: weights[n] for n in shard_tensor_names}
        )
        for n in shard_tensor_names:
            weight_map[n] = shard
    with open(f"{hf_dir}/model.safetensors.index.json", "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f)
    os.remove(single_path)

    batch, seq = 1, 5
    model = onnxsim.reconstruct_hf_graph(hf_dir, batch_size=batch, seq_len=seq)
    onnx.checker.check_model(model)

    rng = np.random.default_rng(1)
    input_ids = rng.integers(0, config["vocab"], size=(batch, seq)).astype(np.int64)
    position_ids = np.arange(seq, dtype=np.int64)[None, :].repeat(batch, axis=0)
    sess = ReferenceEvaluator(model)
    (onnx_logits,) = sess.run(
        None, {"input_ids": input_ids, "position_ids": position_ids}
    )
    ref_logits = _reference_forward(weights, config, input_ids, position_ids)
    np.testing.assert_allclose(onnx_logits, ref_logits, atol=1e-3, rtol=1e-3)


def test_unrecognized_model_type_raises(tmp_path):
    hf_dir = tmp_path / "gpt2"
    hf_dir.mkdir()
    with open(hf_dir / "config.json", "w") as f:
        json.dump({"model_type": "gpt2"}, f)
    _write_safetensors(str(hf_dir / "model.safetensors"), {})
    with pytest.raises(UnsupportedArchitectureError, match="gpt2"):
        onnxsim.reconstruct_hf_graph(str(hf_dir))


def test_missing_required_tensor_raises(tmp_path):
    hf_dir = tmp_path / "no_tensors"
    hf_dir.mkdir()
    config = {
        "model_type": "llama",
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "vocab_size": 12,
    }
    with open(hf_dir / "config.json", "w") as f:
        json.dump(config, f)
    _write_safetensors(str(hf_dir / "model.safetensors"), {})
    with pytest.raises(UnsupportedArchitectureError, match="embed_tokens"):
        onnxsim.reconstruct_hf_graph(str(hf_dir))


def test_moe_config_raises():
    """HF's per-expert tensor layout (mlp.experts.{i}.*) differs from
    GGUF's fused-expert tensors this module's reused _moe_ffn expects, and
    hasn't been adapted -- see hf_reconstruct.py's module docstring."""
    with pytest.raises(UnsupportedArchitectureError, match="num_local_experts"):
        onnxsim.hf_reconstruct._reconstruct_llama_family_hf(
            {
                "model_type": "qwen3",
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "intermediate_size": 16,
                "num_attention_heads": 2,
                "vocab_size": 12,
                "num_local_experts": 4,
            },
            {},
            batch_size=1,
            seq_len=1,
        )
