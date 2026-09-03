"""Tests that onnxsim's generic (op-pattern-based) quantization and pruning
passes -- :func:`onnxsim.quantize_weight_only_int4` and
:func:`onnxsim.apply_magnitude_pruning` -- work unmodified on a speculative
decoding "drafter" model, e.g. an EAGLE3 (Li et al., 2025) draft model.

onnxsim has no architecture-specific handling for these models -- there is
none to have, since these passes only ever look for a constant-weight
MatMul/Gemm/Conv, and a drafter's linear layers are exactly that. This is a
regression test for that claim, built from the real structural fingerprint
of a published EAGLE3 checkpoint (AngelSlim/Qwen3-1.7B_eagle3 on the
Hugging Face Hub; see also sglang's llama_eagle3.py, the reference
implementation for the ``LlamaForCausalLMEagle3`` architecture): unlike a
plain decoder layer, a drafter's attention block reads target-model context
rather than only its own tokens, so:

  - the drafter's own token embedding (``input_embeds``) is concatenated
    with the target model's hidden states (fused down from several
    concatenated target layers by an ``fc`` projection) *before* the
    QKV projections -- so those projections' input dimension is double the
    hidden size, not equal to it (``qkv_in`` is ``2H`` below);
  - the output head projects to a *draft* vocabulary
    (``draft_vocab_size``), which is deliberately smaller than the target
    model's real vocabulary, rather than a same-sized LM head.

This model intentionally skips RoPE/GQA/causal masking (ordinary decoder
mechanics already covered by ``tests/test_fuse_attention.py`` and
``tests/test_fuse_gqa.py``) to isolate exactly those two drafter-specific
traits, at a size (32-dim hidden, so every quantizable layer's reduction
size is a clean multiple of the INT4 pass's 32-element block) small enough
to keep as literal text per this repo's convention (see CLAUDE.md) while
still exercising the same op patterns -- weighted MatMul, plus two
activation-only MatMuls for the attention score/value products that must be
left unquantized -- found in the real checkpoint's exported graph.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim

ort = pytest.importorskip("onnxruntime")

H = 32  # drafter hidden size
AUX = 3 * H  # concatenated target hidden states (3 aux layers, EAGLE3 default)
QKV_IN = 2 * H  # doubled: concat(embeds, fc(aux)) before Q/K/V
INTER = 64  # MLP intermediate size
DRAFT_VOCAB = 16  # deliberately smaller than any real target vocab


def _model(body, initializer=(), opset=21, ir_version=10):
    model = parser.parse_model(
        f"""
        <
          ir_version: {ir_version},
          opset_import: ["": {opset}]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _weight(rng, k, n, name):
    return _f32(rng.standard_normal((k, n)) * (1.0 / np.sqrt(k)), name)


def _eagle3_drafter_model(seed=0):
    # Mirrors LlamaForCausalLMEagle3's single-layer forward: fc-fuse the
    # target's aux hidden states, concat with the drafter's own token
    # embedding to double the QKV input width, self-attend, MLP, then a
    # reduced-vocabulary LM head -- see this module's docstring.
    rng = np.random.default_rng(seed)
    weights = [
        _weight(rng, AUX, H, "W_fc"),
        _weight(rng, QKV_IN, H, "W_q"),
        _weight(rng, QKV_IN, H, "W_k"),
        _weight(rng, QKV_IN, H, "W_v"),
        _weight(rng, H, H, "W_o"),
        _weight(rng, H, INTER, "W_gate"),
        _weight(rng, H, INTER, "W_up"),
        _weight(rng, INTER, H, "W_down"),
        _weight(rng, H, DRAFT_VOCAB, "W_lm_head"),
    ]
    model = _model(
        f"""
        eagle3_drafter (float[batch,seq,{H}] input_embeds, float[batch,seq,{AUX}] aux_hidden_states)
            => (float[batch,seq,{DRAFT_VOCAB}] draft_logits)
        {{
          hidden = MatMul(aux_hidden_states, W_fc)
          qkv_in = Concat <axis = -1> (input_embeds, hidden)
          q = MatMul(qkv_in, W_q)
          k = MatMul(qkv_in, W_k)
          v = MatMul(qkv_in, W_v)
          kt = Transpose <perm = [0, 2, 1]> (k)
          scores = MatMul(q, kt)
          probs = Softmax <axis = -1> (scores)
          attn = MatMul(probs, v)
          attn_out = MatMul(attn, W_o)
          h2 = Add(attn_out, hidden)
          gate = MatMul(h2, W_gate)
          up = MatMul(h2, W_up)
          gate_sigmoid = Sigmoid(gate)
          silu = Mul(gate, gate_sigmoid)
          mlp_hidden = Mul(silu, up)
          mlp_out = MatMul(mlp_hidden, W_down)
          h3 = Add(mlp_out, h2)
          draft_logits = MatMul(h3, W_lm_head)
        }}
        """,
        weights,
    )
    onnx.checker.check_model(model)
    return model


def _feeds(batch=2, seq=5, seed=1):
    rng = np.random.default_rng(seed)
    return {
        "input_embeds": rng.standard_normal((batch, seq, H)).astype(np.float32),
        "aux_hidden_states": rng.standard_normal((batch, seq, AUX)).astype(np.float32),
    }


def test_drafter_fingerprint_qkv_takes_doubled_input():
    # The one thing that distinguishes this from an ordinary decoder layer's
    # self-attention: Q/K/V read a *2H*-wide concat(embeds, target features),
    # not a plain H-wide hidden state.
    model = _eagle3_drafter_model()
    w_q = next(t for t in model.graph.initializer if t.name == "W_q")
    assert w_q.dims[0] == QKV_IN == 2 * H


def test_simplify_preserves_output():
    model = _eagle3_drafter_model()
    feeds = _feeds()
    (float_y,) = _run(model, feeds)

    simplified, check_ok = onnxsim.simplify(model)
    onnx.checker.check_model(simplified)
    assert check_ok

    (simplified_y,) = _run(simplified, feeds)
    np.testing.assert_allclose(float_y, simplified_y, atol=1e-5, rtol=1e-5)


def test_quantize_weight_only_int4_hits_every_weighted_matmul_only():
    # quantize_weight_only_int4 deliberately skips shape inference (see its
    # own docstring), so it only sees a *later* MatMul's reduction size once
    # something has annotated the intermediate tensors' shapes -- simplify()
    # is the documented way to do that. A drafter is a chain of several
    # such layers (fc -> q/k/v/o -> gate/up/down -> lm_head), so skipping
    # this step silently leaves every layer but the first (whose input
    # shape is already known from the graph signature) unquantized.
    #
    # simplify() also fuses Q/K/V here: all three read the same qkv_in
    # tensor, so the simplifier merges them into one wider MatMul + Split
    # (a real, separately-useful optimization for a drafter specifically,
    # since Q/K/V projections sharing one input is exactly this
    # architecture's shape) -- collapsing 9 weighted layers to 7 distinct
    # MatMul nodes before quantization ever runs.
    model, _ = onnxsim.simplify(_eagle3_drafter_model())
    quant = onnxsim.quantize_weight_only_int4(model)
    onnx.checker.check_model(quant)

    dq_nodes = [n for n in quant.graph.node if n.op_type == "DequantizeLinear"]
    # 7 weighted layers post-fusion: fc, qkv (fused), o, gate, up, down,
    # lm_head. The two activation@activation MatMuls (q@k^T and probs@v)
    # must NOT be touched -- they have no constant weight operand to
    # quantize.
    assert len(dq_nodes) == 7
    for dq in dq_nodes:
        block_size = next(a.i for a in dq.attribute if a.name == "block_size")
        assert block_size == 32

    feeds = _feeds()
    (quant_y,) = _run(quant, feeds)
    assert np.all(np.isfinite(quant_y))
    assert quant_y.shape == (2, 5, DRAFT_VOCAB)


def test_quantize_weight_only_int4_without_simplify_first_only_hits_first_layer():
    # The pitfall the previous test's comment describes, made explicit: on
    # the *raw* parser-built graph (no value_info for intermediate tensors,
    # since only the graph's own input/output signature is typed), only
    # W_fc's MatMul -- whose input shape is known directly from the graph
    # signature -- gets quantized. The other 8 chained layers are silently
    # left alone. Wiring quantization into a real multi-layer model means
    # simplify()-ing (or otherwise shape-inferring) it first, always.
    quant = onnxsim.quantize_weight_only_int4(_eagle3_drafter_model())
    dq_nodes = [n for n in quant.graph.node if n.op_type == "DequantizeLinear"]
    assert len(dq_nodes) == 1


def test_magnitude_pruning_reaches_target_sparsity_and_still_runs():
    model = _eagle3_drafter_model()
    pruned = onnxsim.apply_magnitude_pruning(model, sparsity=0.5)
    onnx.checker.check_model(pruned)
    assert onnxsim.weight_sparsity(pruned) == pytest.approx(0.5, abs=1e-6)

    feeds = _feeds()
    (pruned_y,) = _run(pruned, feeds)
    assert np.all(np.isfinite(pruned_y))
    assert pruned_y.shape == (2, 5, DRAFT_VOCAB)


def test_quantize_then_prune_compose():
    # The two passes are independent graph rewrites over the same op
    # patterns, so nothing about running one should preclude the other --
    # this is the "wire quantization/pruning up to a real drafter" claim in
    # its simplest composable form.
    model, _ = onnxsim.simplify(_eagle3_drafter_model())
    quant = onnxsim.quantize_weight_only_int4(model)
    simplified, _ = onnxsim.simplify(quant, skip_fuse_bn=True)
    pruned = onnxsim.apply_magnitude_pruning(simplified, sparsity=0.3)
    onnx.checker.check_model(pruned)

    feeds = _feeds()
    (out,) = _run(pruned, feeds)
    assert np.all(np.isfinite(out))
    assert out.shape == (2, 5, DRAFT_VOCAB)
