"""Real-world companion to ``tests/test_speculative_decoding_drafter.py`` and
``tests/test_speculative_decoding_drafter_torch_export.py``: those two build
a synthetic, EAGLE3-shaped graph (by hand, and via a real ``torch.onnx.export``
respectively) at a tiny 32-dim size kept as literal test code per this repo's
convention (see CLAUDE.md). This file instead downloads and exercises an
actual **published** EAGLE3 drafter checkpoint from the Hugging Face Hub --
``AngelSlim/Qwen3-1.7B_eagle3`` (paired with the real Qwen3-1.7B target
model) -- the smallest real EAGLE3 checkpoint found on the Hub at the time
this was written, and the one this whole test family was originally
developed and validated against.

The architecture (``_RealEagle3Drafter`` below) is a faithful, from-scratch
PyTorch re-implementation of the ``LlamaForCausalLMEagle3`` reference
(sglang's ``srt/models/llama_eagle3.py``, adapted from SafeAILab/EAGLE),
built from this checkpoint's own real tensor names/shapes rather than
guessed -- see this test's own weight-name mapping in
``_load_real_checkpoint`` for exactly what a real checkpoint contains. It
takes the target model's token embeddings and concatenated auxiliary hidden
states as plain tensor inputs (not the ~3.4GB Qwen3-1.7B target model
itself): that mirrors how EAGLE3 is actually served (SGLang/vLLM run it as
a companion model fed the target's hidden states), and keeps this test to
downloading only the ~270MB drafter.

Same heavy/optional dependencies and skip conventions as
``test_export_transformers.py``: torch, onnxscript (torch's dynamo ONNX
exporter needs it), and onnxruntime are not normal test dependencies, so
this file skips unless they're already importable, and skips (rather than
fails) on a network error downloading the real checkpoint from the Hub. To
run it locally::

    pip install torch onnxscript onnxruntime
    pip install --force-reinstall --no-deps .   # the onnxsim under test
    pytest tests/test_speculative_decoding_drafter_hub_checkpoint.py -v -s
"""

import json
import urllib.request

import numpy as np
import onnx
import pytest

import onnxsim

torch = pytest.importorskip("torch")
pytest.importorskip("onnxscript")
ort = pytest.importorskip("onnxruntime")

HF_REPO = "AngelSlim/Qwen3-1.7B_eagle3"
_DOWNLOAD_TIMEOUT_S = 120


def _download(tmp_path):
    config_path = tmp_path / "config.json"
    weights_path = tmp_path / "pytorch_model.bin"
    for name, dest in [
        ("config.json", config_path),
        ("pytorch_model.bin", weights_path),
    ]:
        url = f"https://huggingface.co/{HF_REPO}/resolve/main/{name}"
        with urllib.request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT_S) as resp:
            dest.write_bytes(resp.read())
    return config_path, weights_path


class _RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x, residual=None):
        if residual is not None:
            x = x + residual
            residual = x
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x * self.weight
        return x if residual is None else (x, residual)


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class _RealEagle3Drafter(torch.nn.Module):
    """Faithful re-implementation of ``LlamaForCausalLMEagle3``'s single
    layer (see this module's docstring), parameterized from a real
    checkpoint's own ``config.json`` rather than hardcoded constants.
    """

    def __init__(self, cfg):
        super().__init__()
        H = cfg["hidden_size"]
        self.head_dim = cfg["head_dim"]
        self.num_heads = cfg["num_attention_heads"]
        self.num_kv_heads = cfg["num_key_value_heads"]
        self.rope_theta = cfg["rope_theta"]
        eps = cfg["rms_norm_eps"]
        num_aux = 3

        self.fc = torch.nn.Linear(H * num_aux, H, bias=False)
        self.hidden_norm = _RMSNorm(H, eps)
        self.input_layernorm = _RMSNorm(H, eps)
        self.q_proj = torch.nn.Linear(2 * H, self.num_heads * self.head_dim, bias=False)
        self.k_proj = torch.nn.Linear(
            2 * H, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = torch.nn.Linear(
            2 * H, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = torch.nn.Linear(self.num_heads * self.head_dim, H, bias=False)
        self.post_attention_layernorm = _RMSNorm(H, eps)
        self.gate_proj = torch.nn.Linear(H, cfg["intermediate_size"], bias=False)
        self.up_proj = torch.nn.Linear(H, cfg["intermediate_size"], bias=False)
        self.down_proj = torch.nn.Linear(cfg["intermediate_size"], H, bias=False)
        self.norm = _RMSNorm(H, eps)
        self.lm_head = torch.nn.Linear(H, cfg["draft_vocab_size"], bias=False)

    def _rope(self, positions, device, dtype):
        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32)
                / self.head_dim
            )
        )
        freqs = positions.float()[..., None] * inv_freq[None, None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[:, None, :, :].to(dtype), emb.sin()[:, None, :, :].to(dtype)

    def forward(self, input_embeds, aux_hidden_states, positions):
        cos, sin = self._rope(positions, input_embeds.device, input_embeds.dtype)

        fc_out = self.fc(aux_hidden_states)
        residual = fc_out
        hidden_states = self.hidden_norm(fc_out)
        embeds = self.input_layernorm(input_embeds)
        qkv_in = torch.cat([embeds, hidden_states], dim=-1)

        bsz, seqlen, _ = qkv_in.shape
        q = (
            self.q_proj(qkv_in)
            .view(bsz, seqlen, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(qkv_in)
            .view(bsz, seqlen, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(qkv_in)
            .view(bsz, seqlen, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        rep = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).reshape(bsz, seqlen, self.num_heads * self.head_dim)
        attn_out = self.o_proj(attn)

        hidden_states, residual = self.post_attention_layernorm(attn_out, residual)
        mlp_out = self.down_proj(
            torch.nn.functional.silu(self.gate_proj(hidden_states))
            * self.up_proj(hidden_states)
        )

        hidden_states, _ = self.norm(mlp_out, residual)
        return self.lm_head(hidden_states)


def _load_real_checkpoint(model, state_dict):
    # The real checkpoint's own tensor names (see this repo's development
    # notes / sglang's llama_eagle3.py load_weights): a single "midlayer"
    # holding the drafter's one transformer layer, plus top-level fc/norm/
    # lm_head. d2t/t2d (draft<->target vocab remap tables) are a serving-time
    # concern, irrelevant to the graph this test exports.
    mapping = {
        "midlayer.self_attn.q_proj.weight": "q_proj.weight",
        "midlayer.self_attn.k_proj.weight": "k_proj.weight",
        "midlayer.self_attn.v_proj.weight": "v_proj.weight",
        "midlayer.self_attn.o_proj.weight": "o_proj.weight",
        "midlayer.mlp.gate_proj.weight": "gate_proj.weight",
        "midlayer.mlp.up_proj.weight": "up_proj.weight",
        "midlayer.mlp.down_proj.weight": "down_proj.weight",
        "midlayer.hidden_norm.weight": "hidden_norm.weight",
        "midlayer.input_layernorm.weight": "input_layernorm.weight",
        "midlayer.post_attention_layernorm.weight": "post_attention_layernorm.weight",
        "norm.weight": "norm.weight",
        "fc.weight": "fc.weight",
        "lm_head.weight": "lm_head.weight",
    }
    own = dict(model.named_parameters())
    missing = [k for k in mapping if k not in state_dict]
    if missing:
        raise KeyError(f"real checkpoint is missing expected tensors: {missing}")
    with torch.no_grad():
        for ckpt_name, own_name in mapping.items():
            own[own_name].copy_(state_dict[ckpt_name].float())


@pytest.fixture(scope="module")
def real_eagle3_drafter(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("eagle3_hub_checkpoint")
    try:
        config_path, weights_path = _download(tmp_path)
        cfg = json.loads(config_path.read_text())
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    except Exception as e:  # network/hub errors surface as a variety of types
        pytest.skip(f"Could not download {HF_REPO} from Hugging Face Hub: {e}")

    model = _RealEagle3Drafter(cfg)
    _load_real_checkpoint(model, state_dict)
    model.eval()
    return model, cfg


def _feeds(hidden_size, batch=1, seq=6, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "input_embeds": rng.standard_normal((batch, seq, hidden_size)).astype(
            np.float32
        ),
        "aux_hidden_states": rng.standard_normal((batch, seq, hidden_size * 3)).astype(
            np.float32
        ),
        "positions": np.arange(seq, dtype=np.int64)[None, :],
    }


def _run(onnx_model, feeds):
    sess = ort.InferenceSession(
        onnx_model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(["draft_logits"], feeds)[0]


@pytest.fixture(scope="module")
def exported_onnx_model(real_eagle3_drafter, tmp_path_factory):
    # torch.onnx.export's dynamo tracing takes 20-60s on this real,
    # 2048-hidden-dim GQA model -- module-scoped so the three tests below
    # share one export instead of paying that cost three times over.
    model, cfg = real_eagle3_drafter
    H = cfg["hidden_size"]
    example = (
        torch.randn(1, 6, H),
        torch.randn(1, 6, H * 3),
        torch.arange(6, dtype=torch.long).unsqueeze(0),
    )
    onnx_path = str(tmp_path_factory.mktemp("export") / "real_eagle3_drafter.onnx")
    torch.onnx.export(
        model,
        example,
        onnx_path,
        input_names=["input_embeds", "aux_hidden_states", "positions"],
        output_names=["draft_logits"],
        dynamic_axes={
            "input_embeds": {0: "batch", 1: "seq"},
            "aux_hidden_states": {0: "batch", 1: "seq"},
            "positions": {0: "batch", 1: "seq"},
            "draft_logits": {0: "batch", 1: "seq"},
        },
        opset_version=21,
    )
    return onnx.load(onnx_path)


def test_real_checkpoint_exports_and_matches_eager(
    real_eagle3_drafter, exported_onnx_model
):
    model, cfg = real_eagle3_drafter
    feeds = _feeds(cfg["hidden_size"])
    with torch.no_grad():
        eager_out = model(
            torch.from_numpy(feeds["input_embeds"]),
            torch.from_numpy(feeds["aux_hidden_states"]),
            torch.from_numpy(feeds["positions"]),
        ).numpy()

    onnx_out = _run(exported_onnx_model, feeds)
    assert onnx_out.shape == (1, 6, cfg["draft_vocab_size"])
    np.testing.assert_allclose(eager_out, onnx_out, atol=1e-3, rtol=1e-3)


def test_real_checkpoint_simplifies_cleanly(real_eagle3_drafter, exported_onnx_model):
    _, cfg = real_eagle3_drafter
    feeds = _feeds(cfg["hidden_size"])
    baseline = _run(exported_onnx_model, feeds)

    simplified, check_ok = onnxsim.simplify(exported_onnx_model)
    onnx.checker.check_model(simplified)
    assert check_ok

    simplified_out = _run(simplified, feeds)
    np.testing.assert_allclose(baseline, simplified_out, atol=1e-3, rtol=1e-3)


def test_real_checkpoint_quantizes_and_prunes(real_eagle3_drafter, exported_onnx_model):
    _, cfg = real_eagle3_drafter
    simplified, _ = onnxsim.simplify(exported_onnx_model)

    quantized = onnxsim.quantize_weight_only_int4(simplified)
    onnx.checker.check_model(quantized)
    dq_nodes = [n for n in quantized.graph.node if n.op_type == "DequantizeLinear"]
    # Unlike the synthetic companion tests' single-head attention (where
    # Q/K/V all project to the same width and simplify() fuses them into one
    # wider matmul), this is real GQA -- 16 query heads, 8 KV heads -- so
    # q_proj (2048-wide) and k_proj/v_proj (1024-wide each) have different
    # output shapes and are never fusion candidates. All 9 weighted layers
    # (fc, q, k, v, o, gate, up, down, lm_head) stay distinct and quantized.
    assert len(dq_nodes) == 9

    pruned = onnxsim.apply_magnitude_pruning(simplified, sparsity=0.5)
    onnx.checker.check_model(pruned)
    assert onnxsim.weight_sparsity(pruned) == pytest.approx(0.5, abs=1e-6)

    feeds = _feeds(cfg["hidden_size"])
    for m in (quantized, pruned):
        out = _run(m, feeds)
        assert np.all(np.isfinite(out))
        assert out.shape == (1, 6, cfg["draft_vocab_size"])
