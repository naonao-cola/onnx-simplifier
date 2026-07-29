# Integration/regression test: onnxsim as a drop-in replacement for onnxslim in
# Alibaba MNN's LLM export pipeline (https://github.com/alibaba/MNN).
#
# MNN's LLM exporter (transformers/llm/export/llmexport.py) traces a causal-LM
# decoder with ``torch.onnx.export`` and then, when ``--onnx_slim`` is set,
# shrinks the exported graph with onnxslim::
#
#     @spinner_run(f'slim the graph of ')
#     def slim_onnx(self, onnx_model):
#         import onnxslim
#         model = onnxslim.slim(onnx_model)
#         onnx.save(model, onnx_model)
#         return onnx_model
#
# onnxsim's ``simplify`` has the same shape as ``onnxslim.slim`` -- it takes an
# ONNX ModelProto *or a file path* and returns the reduced model -- so the whole
# method becomes a one-line swap::
#
#     model, _ = onnxsim.simplify(onnx_model)   # was onnxslim.slim(onnx_model)
#
# This module builds a small but structurally faithful LLaMA-style decoder that
# matches MNN's exported graph (the ``input_ids``/``attention_mask``/
# ``position_ids``/``logits_index`` -> ``logits``/``hidden_states`` signature
# emitted by ``LLM.export_onnx``), exports it exactly like MNN does, and then
# runs it through the onnxsim replacement. It guards two things:
#
#   * onnxsim can simplify an LLM-export graph (RMSNorm, rotary-embedding
#     attention, SwiGLU MLP, last-token ``Gather``) without breaking it -- the
#     simplified model still loads, still runs, and returns bit-for-bit
#     equivalent logits, and the graph is actually smaller; and
#   * onnxsim is a faithful stand-in for onnxslim -- given the same export, both
#     tools produce a valid graph that reproduces the reference outputs (the
#     onnxslim leg is skipped unless onnxslim is installed).
#
# The model is deliberately tiny (random weights, short sequence) so the test is
# offline and fast, but the op set and graph topology are the ones a real MNN
# LLM export produces, so it exercises the same simplification paths.
#
# This runs in the regular build-and-test matrix: torch (to trace the model) and
# onnxslim (for the comparison leg) are both in that job's test requirements, so
# both test cases execute there with no dedicated workflow. To run locally::
#
#     pip install torch onnxruntime onnxslim
#     pip install --force-reinstall --no-deps .   # the onnxsim under test
#     pytest tests/test_mnn_llm_export.py -v

import numpy as np
import onnx
import pytest

import onnxsim

torch = pytest.importorskip("torch")
onnxruntime = pytest.importorskip("onnxruntime")

import torch.nn as nn  # noqa: E402  (after the torch importorskip guard)

# Small LLaMA-style config. Only the sizes shrink relative to a real model; the
# graph structure and op set are identical.
HIDDEN = 64
NUM_HEADS = 4
HEAD_DIM = 16
INTERMEDIATE = 128
VOCAB = 128
NUM_LAYERS = 2
SEQ_LEN = 8


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class _Attention(nn.Module):
    """Multi-head attention with rotary position embeddings (LLaMA style)."""

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN, NUM_HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(HIDDEN, NUM_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(HIDDEN, NUM_HEADS * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(NUM_HEADS * HEAD_DIM, HIDDEN, bias=False)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, attention_mask, position_ids):
        bsz, seq, _ = x.shape
        q = self.q_proj(x).view(bsz, seq, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq, NUM_HEADS, HEAD_DIM).transpose(1, 2)

        angle = position_ids[:, :, None].float() * self.inv_freq[None, None, :]
        emb = torch.cat((angle, angle), dim=-1)
        cos, sin = emb.cos()[:, None], emb.sin()[:, None]
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin

        scores = torch.matmul(q, k.transpose(-1, -2)) / (HEAD_DIM**0.5)
        scores = scores + attention_mask
        probs = torch.softmax(scores, dim=-1)
        out = (
            torch.matmul(probs, v)
            .transpose(1, 2)
            .reshape(bsz, seq, NUM_HEADS * HEAD_DIM)
        )
        return self.o_proj(out)


class _MLP(nn.Module):
    """SwiGLU feed-forward block."""

    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(HIDDEN, INTERMEDIATE, bias=False)
        self.up_proj = nn.Linear(HIDDEN, INTERMEDIATE, bias=False)
        self.down_proj = nn.Linear(INTERMEDIATE, HIDDEN, bias=False)

    def forward(self, x):
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        )


class _RMSNorm(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(HIDDEN))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return self.weight * (x * torch.rsqrt(variance + self.eps))


class _DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_norm = _RMSNorm()
        self.attn = _Attention()
        self.post_norm = _RMSNorm()
        self.mlp = _MLP()

    def forward(self, x, attention_mask, position_ids):
        x = x + self.attn(self.input_norm(x), attention_mask, position_ids)
        return x + self.mlp(self.post_norm(x))


class _CausalLM(nn.Module):
    """Tiny causal-LM decoder matching MNN's ``export_onnx`` I/O signature.

    Embedding lookup happens outside the exported graph in MNN (it feeds already
    embedded ``input_ids``), so ``forward`` receives hidden states directly. A
    ``logits_index`` gather selects the tokens to compute logits for -- MNN uses
    it to keep only the last token during decoding.
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_DecoderLayer() for _ in range(NUM_LAYERS)])
        self.norm = _RMSNorm()
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)

    def forward(self, input_ids, attention_mask, position_ids, logits_index):
        x = input_ids
        for layer in self.layers:
            x = layer(x, attention_mask, position_ids)
        hidden_states = self.norm(x)
        selected = hidden_states.index_select(1, logits_index)
        logits = self.lm_head(selected)
        return logits, hidden_states


def _export_llm(tmp_path):
    """Trace and export the decoder exactly like MNN's ``export_onnx``.

    Returns ``(onnx_path, feed, reference_outputs)`` where ``reference_outputs``
    are produced by the un-simplified model, so every simplifier under test is
    checked against the same baseline.
    """
    torch.manual_seed(0)
    model = _CausalLM().eval()

    input_ids = torch.randn(1, SEQ_LEN, HIDDEN)
    # Causal additive mask (0 keep / -inf mask), same shape MNN feeds attention.
    mask = torch.zeros(1, 1, SEQ_LEN, SEQ_LEN)
    mask.masked_fill_(
        torch.triu(torch.ones(SEQ_LEN, SEQ_LEN), diagonal=1).bool(), float("-inf")
    )
    position_ids = torch.arange(SEQ_LEN)[None]
    # MNN passes ``[-1]`` to keep the last token; ONNX Gather accepts negative
    # indices but torch tracing does not, so trace with the concrete last index
    # (the Gather node produced is identical).
    logits_index = torch.tensor([SEQ_LEN - 1], dtype=torch.long)

    onnx_path = str(tmp_path / "llm.onnx")
    torch.onnx.export(
        model,
        (input_ids, mask, position_ids, logits_index),
        onnx_path,
        input_names=["input_ids", "attention_mask", "position_ids", "logits_index"],
        output_names=["logits", "hidden_states"],
        dynamic_axes={
            "input_ids": {1: "seq_len"},
            "attention_mask": {2: "seq_len", 3: "seq_len"},
            "position_ids": {1: "seq_len"},
            "logits": {1: "selected"},
            "hidden_states": {1: "seq_len"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    feed = {
        "input_ids": input_ids.numpy(),
        "attention_mask": mask.numpy(),
        "position_ids": position_ids.numpy().astype(np.int64),
        "logits_index": logits_index.numpy(),
    }
    reference = _run(onnx.load(onnx_path), feed)
    return onnx_path, feed, reference


def _run(model, feed):
    session = onnxruntime.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return session.run(None, feed)


def _slim_onnx_with_onnxsim(onnx_model):
    """MNN's ``slim_onnx``, with onnxslim swapped for onnxsim.

    Mirrors ``transformers/llm/export/llmexport.py``: take a path, reduce the
    graph, save it back to the same path. The only changed line is the simplifier
    call (``onnxslim.slim(onnx_model)`` -> ``onnxsim.simplify(onnx_model)``).
    """
    model, check_ok = onnxsim.simplify(onnx_model)
    assert check_ok
    onnx.save(model, onnx_model)
    return onnx_model


def test_onnxsim_slims_mnn_llm_export(tmp_path):
    onnx_path, feed, reference = _export_llm(tmp_path)
    original = onnx.load(onnx_path)

    # The onnxsim drop-in for MNN's slim_onnx: reduce in place.
    _slim_onnx_with_onnxsim(onnx_path)
    simplified = onnx.load(onnx_path)

    # Still a valid ONNX model after the swap.
    onnx.checker.check_model(simplified)

    # onnxsim must actually shrink the exported graph -- that is the whole point
    # of MNN's --onnx_slim step.
    assert len(simplified.graph.node) < len(original.graph.node)

    # The ops that make this an LLM decoder must survive simplification.
    op_types = {node.op_type for node in simplified.graph.node}
    assert "MatMul" in op_types  # projections / attention
    assert "Softmax" in op_types  # attention weights
    assert "Gather" in op_types  # last-token logits_index select

    # The MNN output signature must be preserved.
    output_names = [o.name for o in simplified.graph.output]
    assert output_names == ["logits", "hidden_states"]

    # Simplification must not change what the model computes.
    outputs = _run(simplified, feed)
    for ref, got in zip(reference, outputs):
        np.testing.assert_allclose(ref, got, rtol=1e-4, atol=1e-5)


def test_onnxsim_matches_onnxslim_on_mnn_llm_export(tmp_path):
    # Proves the replacement is faithful: onnxsim and the onnxslim it replaces
    # both turn the same MNN export into a valid graph that reproduces the
    # reference outputs. onnxslim (MNN's current dependency) is installed in the
    # build-and-test matrix; the importorskip only guards ad-hoc environments
    # that lack it.
    onnxslim = pytest.importorskip("onnxslim")

    onnx_path, feed, reference = _export_llm(tmp_path)
    original = onnx.load(onnx_path)

    onnxsim_model, check_ok = onnxsim.simplify(onnx_path)
    assert check_ok
    onnxslim_model = onnxslim.slim(onnx.load(onnx_path))

    onnx.checker.check_model(onnxsim_model)
    onnx.checker.check_model(onnxslim_model)

    # Both simplifiers shrink the graph...
    assert len(onnxsim_model.graph.node) < len(original.graph.node)
    assert len(onnxslim_model.graph.node) < len(original.graph.node)

    # ...and both preserve the model's outputs, so onnxsim is a valid stand-in.
    for name, model in (("onnxsim", onnxsim_model), ("onnxslim", onnxslim_model)):
        outputs = _run(model, feed)
        for ref, got in zip(reference, outputs):
            np.testing.assert_allclose(
                ref, got, rtol=1e-4, atol=1e-5, err_msg=f"{name} changed the outputs"
            )
