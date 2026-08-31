"""Integration test for ``onnxsim.reconstruct_gguf_graph``'s gpt-oss
dispatch: wires ``_gpt_oss_attention_block`` and ``_gpt_oss_moe_ffn``
together (via ``_reconstruct_gpt_oss``) into a full multi-layer graph and
checks the *plumbing* -- tensor names, shapes, attribute values, and the
per-layer sliding/full mask alternation -- rather than re-deriving either
block's own math, which ``tests/test_gpt_oss_attention.py`` and
``tests/test_gpt_oss_moe_reconstruct.py`` already validate independently
against real onnxruntime/``onnx.reference.ReferenceEvaluator`` runs.

Deliberately exercises head_dim independence from
embedding_length/head_count (see ``_gpt_oss_attention_block``'s own
docstring, point 1): this checkpoint's ``head_dim`` does not equal
``embedding_length // head_count``, the one thing that would silently break
if the integration reused ``_reconstruct_llama_family``'s shape derivation
instead of reading ``attention.key_length`` directly.
"""

import struct

import numpy as np
import onnx
import pytest

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
    """Same minimal, byte-accurate GGUF v3 writer as
    ``tests/test_gguf_reconstruct.py``'s own ``_write_gguf`` (duplicated
    per this repo's convention of not importing fixtures across test
    files)."""
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


def _build_tiny_gpt_oss_checkpoint(tmp_path, seed=11):
    """A hand-built, real GGUF v3 file for a tiny (2-layer) gpt-oss-style
    checkpoint -- every tensor gpt-oss's own ``load_arch_tensors`` requires
    (see ``_gpt_oss_attention_block``/``_gpt_oss_moe_ffn``'s own docstrings
    for the exact llama.cpp source each tensor name/shape was verified
    against), including the required per-tensor biases and the per-layer
    ``attn_sinks``/``post_attention_norm`` tensors neither building block
    reads itself (the norm is this module's own integration responsibility
    -- see ``_reconstruct_gpt_oss``'s docstring).

    ``head_dim`` (6) is deliberately NOT ``n_embd // n_head`` (8 // 2 = 4)
    -- the one detail that would silently break if the integration reused
    ``_reconstruct_llama_family``'s shape derivation instead of reading
    ``attention.key_length`` directly.
    """
    rng = np.random.default_rng(seed)
    n_embd, n_head, n_head_kv, head_dim = 8, 2, 1, 6
    n_embd_q = n_head * head_dim
    n_embd_kv = n_head_kv * head_dim
    n_layer = 2
    n_ff = 5
    n_expert, n_expert_used = 4, 2
    vocab = 10
    eps = 1e-5
    freq_base = 10000.0
    sliding_window = 2
    yarn_factor = 2.0

    def rand(*shape):
        return rng.standard_normal(shape).astype(np.float32) * 0.1

    weights = {"token_embd.weight": rand(vocab, n_embd)}
    for i in range(n_layer):
        p = f"blk.{i}"
        weights[f"{p}.attn_norm.weight"] = rand(n_embd) + 1.0
        weights[f"{p}.attn_q.weight"] = rand(n_embd_q, n_embd)
        weights[f"{p}.attn_q.bias"] = rand(n_embd_q)
        weights[f"{p}.attn_k.weight"] = rand(n_embd_kv, n_embd)
        weights[f"{p}.attn_k.bias"] = rand(n_embd_kv)
        weights[f"{p}.attn_v.weight"] = rand(n_embd_kv, n_embd)
        weights[f"{p}.attn_v.bias"] = rand(n_embd_kv)
        weights[f"{p}.attn_output.weight"] = rand(n_embd, n_embd_q)
        weights[f"{p}.attn_output.bias"] = rand(n_embd)
        weights[f"{p}.attn_sinks.weight"] = rand(n_head)
        weights[f"{p}.post_attention_norm.weight"] = rand(n_embd) + 1.0
        weights[f"{p}.ffn_gate_inp.weight"] = rand(n_expert, n_embd)
        weights[f"{p}.ffn_gate_inp.bias"] = rand(n_expert)
        weights[f"{p}.ffn_gate_exps.weight"] = rand(n_expert, n_ff, n_embd)
        weights[f"{p}.ffn_gate_exps.bias"] = rand(n_expert, n_ff)
        weights[f"{p}.ffn_up_exps.weight"] = rand(n_expert, n_ff, n_embd)
        weights[f"{p}.ffn_up_exps.bias"] = rand(n_expert, n_ff)
        weights[f"{p}.ffn_down_exps.weight"] = rand(n_expert, n_embd, n_ff)
        weights[f"{p}.ffn_down_exps.bias"] = rand(n_expert, n_embd)
    weights["output_norm.weight"] = rand(n_embd) + 1.0
    weights["output.weight"] = rand(vocab, n_embd)

    kv_chunks = [
        _kv_string("general.architecture", "gpt-oss"),
        _kv_uint32("gpt-oss.block_count", n_layer),
        _kv_uint32("gpt-oss.embedding_length", n_embd),
        _kv_uint32("gpt-oss.expert_feed_forward_length", n_ff),
        _kv_uint32("gpt-oss.attention.head_count", n_head),
        _kv_uint32("gpt-oss.attention.head_count_kv", n_head_kv),
        _kv_uint32("gpt-oss.attention.key_length", head_dim),
        _kv_uint32("gpt-oss.attention.value_length", head_dim),
        _kv_float32("gpt-oss.attention.layer_norm_rms_epsilon", eps),
        _kv_float32("gpt-oss.rope.freq_base", freq_base),
        _kv_uint32("gpt-oss.attention.sliding_window", sliding_window),
        _kv_uint32("gpt-oss.expert_count", n_expert),
        _kv_uint32("gpt-oss.expert_used_count", n_expert_used),
        _kv_string("gpt-oss.rope.scaling.type", "yarn"),
        _kv_float32("gpt-oss.rope.scaling.factor", yarn_factor),
        _kv_float32("gpt-oss.rope.scaling.original_context_length", 4096.0),
        _kv_float32("gpt-oss.rope.scaling.yarn_beta_fast", 32.0),
        _kv_float32("gpt-oss.rope.scaling.yarn_beta_slow", 1.0),
    ]
    path = str(tmp_path / "tiny_gpt_oss.gguf")
    _write_gguf(path, kv_chunks, weights)

    config = dict(
        n_embd=n_embd,
        n_head=n_head,
        n_head_kv=n_head_kv,
        head_dim=head_dim,
        n_layer=n_layer,
        n_ff=n_ff,
        n_expert=n_expert,
        n_expert_used=n_expert_used,
        vocab=vocab,
        sliding_window=sliding_window,
    )
    return path, weights, config


def test_reconstruct_gpt_oss_wires_up_the_full_graph_correctly(tmp_path):
    path, weights, config = _build_tiny_gpt_oss_checkpoint(tmp_path)
    n_embd = config["n_embd"]
    n_head = config["n_head"]
    head_dim = config["head_dim"]
    n_embd_q = n_head * head_dim
    n_layer = config["n_layer"]
    n_expert_used = config["n_expert_used"]
    vocab = config["vocab"]
    batch, seq = 1, 4

    model, skipped = onnxsim.reconstruct_gguf_graph(path, batch_size=batch, seq_len=seq)
    assert skipped == []
    onnx.checker.check_model(model)

    out = model.graph.output[0]
    out_dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
    assert out_dims == [batch, seq, vocab]

    by_name = {i.name: i for i in model.graph.initializer}

    # head_dim independence: attn_q/attn_output use n_head*head_dim (12),
    # not n_embd (8) -- the detail that would silently break if this
    # integration reused _reconstruct_llama_family's shape derivation.
    assert n_embd_q != n_embd
    assert list(by_name["blk.0.attn_q.weight"].dims) == [n_embd_q, n_embd]
    assert list(by_name["blk.0.attn_output.weight"].dims) == [n_embd, n_embd_q]

    # The pre-FFN norm is named "post_attention_norm" for this architecture,
    # NOT "ffn_norm" (see _reconstruct_gpt_oss's own docstring) -- both
    # tensors must actually be hydrated (real values, not the placeholder's
    # zero-fill) since the checkpoint provides post_attention_norm and
    # reconstruct_gguf_graph calls import_gguf_weights internally.
    assert "blk.0.post_attention_norm.weight" in by_name
    assert "blk.0.ffn_norm.weight" not in by_name
    got_norm = onnx.numpy_helper.to_array(by_name["blk.0.post_attention_norm.weight"])
    np.testing.assert_allclose(
        got_norm, weights["blk.0.post_attention_norm.weight"], rtol=1e-6
    )

    # attn_sinks is present and hydrated with the real per-head values.
    got_sinks = onnx.numpy_helper.to_array(by_name["blk.0.attn_sinks.weight"])
    np.testing.assert_allclose(got_sinks, weights["blk.0.attn_sinks.weight"], rtol=1e-6)

    # Exactly one MoE node per layer, each with gpt-oss's swiglu attributes.
    moe_nodes = [n for n in model.graph.node if n.op_type == "MoE"]
    assert len(moe_nodes) == n_layer
    for node in moe_nodes:
        assert node.domain == "com.microsoft"
        attrs = {a.name: a for a in node.attribute}
        assert attrs["k"].i == n_expert_used
        assert attrs["activation_type"].s == b"swiglu"
        assert attrs["swiglu_fusion"].i == 1
        assert attrs["activation_alpha"].f == pytest.approx(1.702)
        assert attrs["activation_beta"].f == pytest.approx(1.0)
        assert attrs["swiglu_limit"].f == pytest.approx(7.0)
        assert attrs["normalize_routing_weights"].i == 1

    # fc1_w is a computed (Reshape) output, not itself an initializer, until
    # a later onnxsim.simplify() constant-folds it -- check it's actually
    # produced by the gate/up interleave fusion (named with the
    # _interleave_gate_up prefix _gpt_oss_moe_ffn passes) via its name and
    # producing node's op_type, not by looking it up as an initializer.
    fc1_w_name = moe_nodes[0].input[2]
    assert fc1_w_name.startswith("blk.0.moe_fc1_w")
    producer = next(n for n in model.graph.node if fc1_w_name in n.output)
    assert producer.op_type == "Reshape"

    # Per-layer sliding/full mask alternation (swa_period defaults to 2,
    # so layer 0 is sliding and layer 1 is full -- see
    # _gpt_oss_is_sliding_layer's own docstring): each layer's additive
    # attention-mask CONSTANT is a distinct initializer (not shared across
    # layers, since only every-other layer is banded), so this checks the
    # actual embedded mask values rather than trusting the naming alone.
    # Every _Builder-emitted name gets a unique numeric suffix (see
    # _Builder._name), so look these up by prefix rather than exact name.
    def _initializer_by_prefix(prefix: str) -> onnx.TensorProto:
        matches = [t for name, t in by_name.items() if name.startswith(prefix)]
        assert len(matches) == 1, (prefix, list(by_name))
        return matches[0]

    sliding_window = config["sliding_window"]
    mask0 = onnx.numpy_helper.to_array(_initializer_by_prefix("blk.0.attn_mask"))
    mask1 = onnx.numpy_helper.to_array(_initializer_by_prefix("blk.1.attn_mask"))
    assert mask0.shape == (seq, seq)
    # Sliding layer: query position 3 can see key positions 2,3 only
    # (window width 2) -- key position 0/1 must be masked even though a
    # plain causal mask alone would allow them.
    assert mask0[3, 1] < -1.0 and mask0[3, 2] == 0.0 and mask0[3, 3] == 0.0
    # Full layer: query position 3 can see every earlier key position.
    assert mask1[3, 0] == 0.0 and mask1[3, 1] == 0.0
    assert not np.array_equal(mask0, mask1)
    assert sliding_window == 2  # sanity: matches the checkpoint's own metadata

    # onnxsim.simplify() itself accepts the reconstructed graph (shape
    # inference succeeds end to end) -- MoE has no CPU kernel, so this
    # can't run through onnxruntime, matching the Mixtral MoE test's own
    # scope (see test_gguf_reconstruct.py).
    simplified, ok = onnxsim.simplify(
        model, check_n=0, perform_optimization=False, skip_constant_folding=True
    )
    assert ok
    out_shape = simplified.graph.output[0].type.tensor_type.shape
    assert [d.dim_value for d in out_shape.dim] == [batch, seq, vocab]


def test_reconstruct_gpt_oss_requires_sliding_window_metadata(tmp_path):
    path, _weights, _config = _build_tiny_gpt_oss_checkpoint(tmp_path)
    # Rewrite without the (required, per llama.cpp's own 2-arg get_key)
    # sliding_window key by dropping it from a fresh build -- simplest way
    # to construct the negative case is to omit it directly.
    rng = np.random.default_rng(0)
    n_embd, n_head, n_head_kv, head_dim = 4, 1, 1, 4
    weights = {
        "token_embd.weight": rng.standard_normal((3, n_embd)).astype(np.float32),
    }
    kv_chunks = [
        _kv_string("general.architecture", "gpt-oss"),
        _kv_uint32("gpt-oss.block_count", 1),
        _kv_uint32("gpt-oss.embedding_length", n_embd),
        _kv_uint32("gpt-oss.expert_feed_forward_length", 4),
        _kv_uint32("gpt-oss.attention.head_count", n_head),
        _kv_uint32("gpt-oss.attention.head_count_kv", n_head_kv),
        _kv_uint32("gpt-oss.attention.key_length", head_dim),
        _kv_uint32("gpt-oss.expert_count", 2),
        _kv_uint32("gpt-oss.expert_used_count", 1),
        # attention.sliding_window deliberately omitted.
    ]
    path2 = str(tmp_path / "missing_swa.gguf")
    _write_gguf(path2, kv_chunks, weights)

    with pytest.raises(UnsupportedArchitectureError, match="sliding_window"):
        onnxsim.reconstruct_gguf_graph(path2)


def test_reconstruct_gpt_oss_requires_head_dim_metadata(tmp_path):
    rng = np.random.default_rng(0)
    n_embd = 4
    weights = {
        "token_embd.weight": rng.standard_normal((3, n_embd)).astype(np.float32),
    }
    kv_chunks = [
        _kv_string("general.architecture", "gpt-oss"),
        _kv_uint32("gpt-oss.block_count", 1),
        _kv_uint32("gpt-oss.embedding_length", n_embd),
        _kv_uint32("gpt-oss.expert_feed_forward_length", 4),
        _kv_uint32("gpt-oss.attention.head_count", 1),
        _kv_uint32("gpt-oss.attention.head_count_kv", 1),
        # attention.key_length/value_length deliberately omitted.
        _kv_uint32("gpt-oss.attention.sliding_window", 2),
        _kv_uint32("gpt-oss.expert_count", 2),
        _kv_uint32("gpt-oss.expert_used_count", 1),
    ]
    path = str(tmp_path / "missing_head_dim.gguf")
    _write_gguf(path, kv_chunks, weights)

    with pytest.raises(UnsupportedArchitectureError, match="key_length"):
        onnxsim.reconstruct_gguf_graph(path)
