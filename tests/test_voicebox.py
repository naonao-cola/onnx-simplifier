# Integration/regression test: Voicebox
# (https://arxiv.org/abs/2306.15687, Meta's conditional flow-matching TTS
# model) exports simplified by onnxsim.
#
# Meta never released Voicebox weights or code. The only installable
# implementation is the unofficial ``voicebox-pytorch`` package
# (https://github.com/lucidrains/voicebox-pytorch), which this test targets.
# It exercises the ``VoiceBox`` module -- the conditional flow-matching
# network that is the actual neural core of the TTS pipeline (predicting a
# vector field given the current noisy audio, a diffusion time, and text/audio
# conditioning) -- covering:
#
#   * depthwise Conv1d positional embedding (``ConvPositionEmbed``),
#   * RMSNorm/AdaptiveRMSNorm (time-conditioned normalization, ReduceL2-based),
#   * multi-head self-attention with rotary embeddings (Sin/Cos, Softmax,
#     Einsum-based QK^T/attn@V), and
#   * the learned sinusoidal timestep embedding used to condition on the flow
#     time.
#
# The model is built with random weights (no checkpoint -- none exists) at a
# small size: only tensor sizes shrink, the graph structure and op set are
# identical to a full-size Voicebox, so regression coverage is unchanged
# while the test stays fast and offline.
#
# ``voicebox-pytorch`` is NOT a normal test dependency and is not installed in
# CI. Its ``audiolm-pytorch`` dependency unconditionally requires ``fairseq``,
# which fails to build under modern pip/setuptools (a long-standing,
# unresolved upstream issue -- fairseq has been unmaintained since 2022). A
# plain ``pip install voicebox-pytorch`` therefore fails on a fresh
# environment; this test ``importorskip``s the package and only runs where a
# working install already exists (e.g. via a local fairseq workaround). To run
# it locally once such an environment is set up::
#
#     pip install voicebox-pytorch onnxruntime
#     pip install --force-reinstall --no-deps .   # the onnxsim under test
#     pytest tests/test_voicebox.py -v

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("voicebox_pytorch")

import onnxruntime  # noqa: E402

# onnxsim.test_utils imports torch at module load, so it must follow the
# importorskip guard above (hence the E402 exemption).
from onnxsim.test_utils import export_simplify_and_check_by_python_api  # noqa: E402

DIM = 64
SEQ_LEN = 32
BATCH = 1
NUM_COND_TOKENS = 50


class _VoiceBoxExportWrapper(torch.nn.Module):
    """Wraps VoiceBox.forward's keyword-only args into a positional signature
    torch.onnx.export can trace, with a fixed cond_drop_prob=0 (classifier-free
    guidance dropout is a training-time trick and must be off for export)."""

    def __init__(self, voicebox):
        super().__init__()
        self.voicebox = voicebox

    def forward(self, x, times, cond_token_ids, cond):
        return self.voicebox(
            x,
            times=times,
            cond_token_ids=cond_token_ids,
            cond=cond,
            cond_drop_prob=0.0,
        )


def test_voicebox_export_simplify():
    from voicebox_pytorch import VoiceBox

    torch.manual_seed(0)
    model = VoiceBox(
        num_cond_tokens=NUM_COND_TOKENS,
        dim=DIM,
        dim_cond_emb=DIM,
        depth=2,
        dim_head=16,
        heads=4,
        ff_mult=2,
        conv_pos_embed_kernel_size=7,
        attn_flash=False,
        num_register_tokens=2,
        condition_on_text=True,
    ).eval()
    wrapper = _VoiceBoxExportWrapper(model).eval()

    x = torch.randn(BATCH, SEQ_LEN, DIM)
    times = torch.rand(BATCH)
    cond_token_ids = torch.randint(0, NUM_COND_TOKENS, (BATCH, SEQ_LEN))
    cond = torch.randn(BATCH, SEQ_LEN, DIM)

    opt = export_simplify_and_check_by_python_api(
        wrapper,
        (x, times, cond_token_ids, cond),
        export_kwargs={
            "opset_version": 17,
            "do_constant_folding": True,
            "input_names": ["x", "times", "cond_token_ids", "cond"],
            "output_names": ["output"],
        },
    )

    # The simplified graph must still carry the ops that make Voicebox
    # Voicebox: the depthwise conv positional embedding, RMSNorm (via
    # ReduceL2), and attention (Softmax). If a future onnxsim change silently
    # drops/miscompiles these, the numerical check above would already fail,
    # but assert their presence too so the test documents what it protects.
    op_types = {node.op_type for node in opt.graph.node}
    assert "Conv" in op_types
    assert "Softmax" in op_types
    assert "ReduceL2" in op_types

    # The simplified model must run and produce the expected output signature:
    # a predicted vector field with the same shape as the input audio latents.
    session = onnxruntime.InferenceSession(opt.SerializeToString())
    outputs = session.run(
        None,
        {
            "x": x.numpy().astype("float32"),
            "times": times.numpy().astype("float32"),
            "cond_token_ids": cond_token_ids.numpy().astype("int64"),
            "cond": cond.numpy().astype("float32"),
        },
    )
    assert [o.name for o in session.get_outputs()] == ["output"]
    assert outputs[0].shape == (BATCH, SEQ_LEN, DIM)
