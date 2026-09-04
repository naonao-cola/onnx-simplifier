"""Real-model counterpart to tests/test_jax_export_integration.py: exports two
actual Hugging Face Transformers *Flax* model implementations -- a Vision
Transformer (`FlaxViTModel`) and a GPT-2 causal decoder (`FlaxGPT2LMHeadModel`)
-- via jax2onnx, then feeds the result through onnxsim.simplify(). Where that
other file exercises small, hand-written functions/modules targeting specific
lowering patterns, this one exercises complete, real transformer
architectures: patch embedding, multi-head self-attention (bidirectional for
ViT, causal for GPT-2), GELU-activated MLP blocks, and LayerNormalization,
all wired together the way transformers' own modeling code wires them --
matching the "build a small instance of the real model class with random
weights, no checkpoint download" convention used by e.g.
tests/test_rfdetr.py.

Both models use a tiny/random-weight config (2 layers, small hidden size),
so this stays fast and fully offline while exercising the same graph shapes
production configs produce.

Hugging Face Transformers removed its Flax (and TensorFlow) model
implementations entirely as of transformers 5.0 -- ``FlaxViTModel`` and
``FlaxGPT2LMHeadModel`` no longer exist on that or later versions. This
module therefore needs ``transformers < 5`` specifically (not just
"transformers"), on top of jax2onnx's own dependencies, so it skips cleanly
whenever the installed transformers is too new (or missing) rather than
failing on an AttributeError. To run locally::

    pip install jax2onnx "transformers<5"
    pip install --force-reinstall --no-deps .   # the onnxsim under test
    pytest tests/test_jax_real_model_integration.py -v

A note on why this doesn't also cover FlaxBertModel: as of transformers
4.49.0, exporting it through jax2onnx 0.16.1 hits a jax2onnx bug -- one of
the attention block's internal broadcast tensors is lowered to an ONNX
Einsum input typed int32 against a float32 softmax-output operand, which
ONNX Runtime rejects at load time. That is a jax2onnx/transformers
interaction issue, not something onnxsim can be expected to compensate for,
so it is left out here rather than adding a test pinned to a known-broken
upstream lowering.
"""

import numpy as np
import onnx
import pytest

import onnxsim

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
jax2onnx = pytest.importorskip("jax2onnx")
onnxruntime = pytest.importorskip("onnxruntime")
transformers = pytest.importorskip("transformers")

FlaxViTModel = getattr(transformers, "FlaxViTModel", None)
FlaxGPT2LMHeadModel = getattr(transformers, "FlaxGPT2LMHeadModel", None)
if FlaxViTModel is None or FlaxGPT2LMHeadModel is None:
    pytest.skip(
        "transformers >= 5 dropped its Flax model implementations "
        "(FlaxViTModel/FlaxGPT2LMHeadModel); install transformers < 5 to "
        "run this module",
        allow_module_level=True,
    )

to_onnx = jax2onnx.to_onnx


def _simplify(model):
    sim_model, check_ok = onnxsim.simplify(model, check_n=3)
    assert check_ok
    onnx.checker.check_model(sim_model)
    return sim_model


def _run(model, feeds):
    sess = onnxruntime.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    out_names = [o.name for o in sess.get_outputs()]
    return dict(zip(out_names, sess.run(out_names, feeds)))


def test_jax_export_vit_model():
    config = transformers.ViTConfig(
        image_size=32,
        patch_size=8,
        num_channels=3,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
    )
    model = FlaxViTModel(config, seed=0)

    def f(pixel_values):
        return model(pixel_values=pixel_values).last_hidden_state

    x = np.random.RandomState(0).randn(2, 3, 32, 32).astype(np.float32)
    onnx_model = to_onnx(f, [jnp.asarray(x)])
    sim_model = _simplify(onnx_model)

    input_name = onnx_model.graph.input[0].name
    output_name = onnx_model.graph.output[0].name
    expected = np.asarray(f(jnp.asarray(x)))
    orig_out = _run(onnx_model, {input_name: x})[output_name]
    sim_out = _run(sim_model, {input_name: x})[output_name]
    np.testing.assert_allclose(orig_out, expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(sim_out, expected, rtol=1e-3, atol=1e-4)


def test_jax_export_gpt2_causal_lm():
    config = transformers.GPT2Config(
        vocab_size=99, n_positions=64, n_embd=32, n_layer=2, n_head=4
    )
    model = FlaxGPT2LMHeadModel(config, seed=0)

    def f(input_ids):
        return model(input_ids=input_ids).logits

    input_ids = np.random.RandomState(1).randint(0, 99, size=(2, 8)).astype(np.int32)
    onnx_model = to_onnx(f, [jnp.asarray(input_ids)])
    sim_model = _simplify(onnx_model)

    input_name = onnx_model.graph.input[0].name
    output_name = onnx_model.graph.output[0].name
    expected = np.asarray(f(jnp.asarray(input_ids)))
    orig_out = _run(onnx_model, {input_name: input_ids})[output_name]
    sim_out = _run(sim_model, {input_name: input_ids})[output_name]
    np.testing.assert_allclose(orig_out, expected, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(sim_out, expected, rtol=1e-3, atol=1e-4)
