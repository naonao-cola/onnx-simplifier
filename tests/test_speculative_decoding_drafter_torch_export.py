"""End-to-end companion to ``tests/test_speculative_decoding_drafter.py``:
that file's graph is hand-built via ``onnx.parser`` text (per this repo's own
convention, see CLAUDE.md); this one builds the *same* architecture as a real
PyTorch ``nn.Module`` and puts it through ``torch.onnx.export`` for real,
before feeding the result to the same onnxsim passes. Hand-written ONNX text
can't accidentally reproduce exporter-specific quirks -- this closes that
gap. Two real ones showed up doing this against an actual published EAGLE3
checkpoint (AngelSlim/Qwen3-1.7B_eagle3) during development and are pinned
here as regressions:

  - the dynamo-based ``torch.onnx.export`` path requires **opset 21**
    for :func:`onnxsim.quantize_weight_only_int4` to do anything at all --
    at opset < 21 every node is silently left unquantized (`quantize_weight_
    only_int4`'s own docstring: "an opset older than 21 ... left untouched"),
    with no error or warning from either torch or onnxsim;
  - unlike the parser-built companion test's raw graph -- which has no
    ``value_info`` for its intermediates, so :func:`onnxsim.quantize_weight_
    only_int4` (which explicitly skips shape inference) only catches the
    first layer in a chain -- torch's dynamo exporter fully annotates every
    intermediate tensor's shape already, so quantization works directly on
    the raw export without a ``simplify()`` pass first. Worth confirming
    explicitly rather than assuming one hand-built-graph finding transfers
    to a real exporter's output. ``simplify()`` is still worth running
    first regardless: it fuses the sibling Q/K/V projections (all three
    read the same ``qkv_in`` tensor) into one wider matmul, taking the
    weighted-layer count from 9 down to 7 before quantization ever runs.

Same heavy/optional dependencies and skip convention as
``test_export_transformers.py``: torch and onnxscript (torch's dynamo ONNX
exporter needs it, see torch/onnx/_internal/exporter/_core.py) are not
normal test dependencies, so this file skips unless they're already
importable. To run it locally::

    pip install torch onnxscript onnxruntime
    pip install --force-reinstall --no-deps .   # the onnxsim under test
    pytest tests/test_speculative_decoding_drafter_torch_export.py -v
"""

import numpy as np
import onnx
import pytest

import onnxsim

torch = pytest.importorskip("torch")
pytest.importorskip("onnxscript")
ort = pytest.importorskip("onnxruntime")

H = 32  # drafter hidden size
AUX = 3 * H  # concatenated target hidden states (3 aux layers, EAGLE3 default)
QKV_IN = 2 * H  # doubled: concat(embeds, fc(aux)) before Q/K/V
INTER = 64  # MLP intermediate size
DRAFT_VOCAB = 16  # deliberately smaller than any real target vocab


class _TinyEagle3Drafter(torch.nn.Module):
    """Same architecture, same shapes, as this module's parser-built sibling
    (``tests/test_speculative_decoding_drafter.py``'s ``_eagle3_drafter_model``)
    -- RoPE/GQA/causal masking deliberately left out for the same reason: the
    point here is exercising the drafter-specific fc-fuse + doubled-QKV-input
    + reduced-vocab-head shape through a real exporter, not re-testing
    ordinary decoder mechanics.
    """

    def __init__(self):
        super().__init__()
        gen = torch.Generator().manual_seed(0)

        def w(k, n):
            return torch.nn.Parameter(torch.randn(k, n, generator=gen) / k**0.5)

        self.w_fc = w(AUX, H)
        self.w_q = w(QKV_IN, H)
        self.w_k = w(QKV_IN, H)
        self.w_v = w(QKV_IN, H)
        self.w_o = w(H, H)
        self.w_gate = w(H, INTER)
        self.w_up = w(H, INTER)
        self.w_down = w(INTER, H)
        self.w_lm_head = w(H, DRAFT_VOCAB)

    def forward(self, input_embeds, aux_hidden_states):
        hidden = aux_hidden_states @ self.w_fc
        qkv_in = torch.cat([input_embeds, hidden], dim=-1)
        q = qkv_in @ self.w_q
        k = qkv_in @ self.w_k
        v = qkv_in @ self.w_v
        scores = q @ k.transpose(-2, -1)
        probs = torch.softmax(scores, dim=-1)
        attn = probs @ v
        h2 = attn @ self.w_o + hidden
        gate = h2 @ self.w_gate
        up = h2 @ self.w_up
        silu = gate * torch.sigmoid(gate)
        mlp_out = (silu * up) @ self.w_down
        h3 = mlp_out + h2
        return h3 @ self.w_lm_head


def _feeds(batch=2, seq=5, seed=1):
    rng = np.random.default_rng(seed)
    return {
        "input_embeds": rng.standard_normal((batch, seq, H)).astype(np.float32),
        "aux_hidden_states": rng.standard_normal((batch, seq, AUX)).astype(np.float32),
    }


def _export(tmp_path, opset_version=21, name="drafter.onnx"):
    torch.manual_seed(0)
    model = _TinyEagle3Drafter().eval()
    example = (torch.randn(2, 5, H), torch.randn(2, 5, AUX))
    onnx_path = str(tmp_path / name)
    torch.onnx.export(
        model,
        example,
        onnx_path,
        input_names=["input_embeds", "aux_hidden_states"],
        output_names=["draft_logits"],
        dynamic_axes={
            "input_embeds": {0: "batch", 1: "seq"},
            "aux_hidden_states": {0: "batch", 1: "seq"},
            "draft_logits": {0: "batch", 1: "seq"},
        },
        opset_version=opset_version,
    )
    return model, onnx_path


def _run(onnx_path_or_model, feeds):
    payload = (
        onnx_path_or_model
        if isinstance(onnx_path_or_model, str)
        else onnx_path_or_model.SerializeToString()
    )
    sess = ort.InferenceSession(payload, providers=["CPUExecutionProvider"])
    return sess.run(["draft_logits"], feeds)[0]


def test_torch_export_matches_eager(tmp_path):
    model, onnx_path = _export(tmp_path)
    feeds = _feeds()

    with torch.no_grad():
        eager_out = model(
            torch.from_numpy(feeds["input_embeds"]),
            torch.from_numpy(feeds["aux_hidden_states"]),
        ).numpy()

    onnx_out = _run(onnx_path, feeds)
    np.testing.assert_allclose(eager_out, onnx_out, atol=1e-4, rtol=1e-3)


def test_simplify_preserves_torch_exported_output(tmp_path):
    _, onnx_path = _export(tmp_path)
    feeds = _feeds()
    baseline = _run(onnx_path, feeds)

    simplified, check_ok = onnxsim.simplify(onnx.load(onnx_path))
    onnx.checker.check_model(simplified)
    assert check_ok

    simplified_out = _run(simplified, feeds)
    np.testing.assert_allclose(baseline, simplified_out, atol=1e-4, rtol=1e-3)


def test_quantize_weight_only_int4_needs_opset_21(tmp_path):
    # Regression for the real, silent trap found exporting an actual EAGLE3
    # checkpoint: at opset 18 the pass matches nothing at all, with no error.
    _, onnx_path_old = _export(tmp_path, opset_version=18, name="old.onnx")
    quant_old = onnxsim.quantize_weight_only_int4(
        onnxsim.simplify(onnx.load(onnx_path_old))[0]
    )
    assert not any(n.op_type == "DequantizeLinear" for n in quant_old.graph.node)

    _, onnx_path_new = _export(tmp_path, opset_version=21, name="new.onnx")
    quant_new = onnxsim.quantize_weight_only_int4(
        onnxsim.simplify(onnx.load(onnx_path_new))[0]
    )
    dq_nodes = [n for n in quant_new.graph.node if n.op_type == "DequantizeLinear"]
    # fc, qkv (simplify fuses q/k/v), o, gate, up, down, lm_head
    assert len(dq_nodes) == 7


def test_quantize_weight_only_int4_works_without_simplify_first(tmp_path):
    # The parser-built companion test (test_speculative_decoding_drafter.py)
    # shows quantize_weight_only_int4 catching only the first layer of a
    # hand-built graph with no value_info for its intermediates -- it
    # explicitly skips shape inference itself. It would be easy to assume
    # that finding just carries over to any multi-layer graph; it does not:
    # torch's dynamo exporter already annotates every intermediate tensor's
    # shape, so all 9 weighted layers (fc, q, k, v, o, gate, up, down,
    # lm_head -- unfused, since simplify() never ran to merge q/k/v) get
    # quantized directly on the raw export.
    _, onnx_path = _export(tmp_path)
    raw_quant = onnxsim.quantize_weight_only_int4(onnx.load(onnx_path))
    assert sum(n.op_type == "DequantizeLinear" for n in raw_quant.graph.node) == 9

    # simplify() is still worth running first regardless: it additionally
    # fuses the sibling Q/K/V matmuls (all three read qkv_in) into one wider
    # matmul, so the same 9 layers collapse to 7 distinct nodes to quantize.
    simplified_quant = onnxsim.quantize_weight_only_int4(
        onnxsim.simplify(onnx.load(onnx_path))[0]
    )
    simplified_dq = sum(
        n.op_type == "DequantizeLinear" for n in simplified_quant.graph.node
    )
    assert simplified_dq == 7


def test_quantize_and_prune_torch_exported_drafter_stays_finite(tmp_path):
    _, onnx_path = _export(tmp_path)
    simplified, _ = onnxsim.simplify(onnx.load(onnx_path))

    quantized = onnxsim.quantize_weight_only_int4(simplified)
    onnx.checker.check_model(quantized)
    pruned = onnxsim.apply_magnitude_pruning(simplified, sparsity=0.5)
    onnx.checker.check_model(pruned)
    assert onnxsim.weight_sparsity(pruned) == pytest.approx(0.5, abs=1e-6)

    feeds = _feeds()
    for m in (quantized, pruned):
        out = _run(m, feeds)
        assert np.all(np.isfinite(out))
        assert out.shape == (2, 5, DRAFT_VOCAB)
