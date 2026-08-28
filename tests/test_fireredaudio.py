# Integration/regression test: onnxsim against FireRedTeam's FireRedAudio
# (https://huggingface.co/FireRedTeam/FireRedAudio,
# https://github.com/FireRedTeam/FireRedAudio, arXiv:2608.24168), a
# general-purpose audio language model built on a shared 9B-parameter LLM
# backbone with "decoupled continuous representations": an Audio Encoder
# pathway for understanding (ASR, audio QA) and a RedAE pathway for
# generation (zero-shot/instruct TTS, semantic/acoustic speech editing, voice
# design), sharing one backbone.
#
# Unlike the other model-integration tests in this repo, FireRedAudio has no
# published ONNX export to target (contrast tests/test_pocket_tts.py,
# tests/test_voicevox.py), and its checkpoint is far too large to export
# whole: ~21GB across 5 safetensors shards for the 9B backbone + encoders,
# plus another ~8.4GB RedAE_decoder/model.pt for the generation vocoder. So
# this test targets two small, self-contained, real transformer submodules
# from FireRedAudio's own generation pathway instead -- both defined in
# fireredaudio/flow/ and used by RedDiT.generate()'s Euler-solver sampling
# loop, neither requiring the 9B backbone LLM, the RedAE decoder, or any
# CUDA-only kernel (flash-attn/causal-conv1d/mamba_ssm) to construct or run:
#
#   * RedPatchEncoder (fireredaudio.flow.patch_encoder) -- a small DiT-style
#     transformer that aggregates `patch_size` consecutive RedAE VAE latents
#     into one patch embedding conditioning the backbone LLM.
#   * RedDiT (fireredaudio.flow.estimator), specifically its
#     `_forward_estimator` method -- the AdaLN-conditioned, attention +
#     causal-Conv1d flow-matching vector-field network that
#     `RedDiT.generate`'s Euler loop calls once per sampling step.
#
# Both are exercised with FireRedAudio's real pretrained weights (not random
# init -- unlike tests/test_voicebox.py, real weights are available here) and
# guard onnxsim against a graph shape this repo's other audio-model tests
# don't cover: AdaLN time-conditioning (chunked Sigmoid/SiLU-gated modulation
# feeding every block) layered on top of RoPE self-attention, interleaved
# with a depthwise-style Conv1d branch per DiT block -- and, between the two
# modules, two independently-instantiated RotaryEmbedding modules and
# TimestepEmbedder's sinusoidal (Sin/Cos) embedding, all deep constant-folding
# candidates.
#
# Efficient, targeted download: FireRedAudio's real weights for both modules
# ("dit.*" and "patch_encoder.*" in the state dict) live entirely inside one
# shard, FireRedAudio/model-00005-of-00005.safetensors (~2.9GB; the 9B
# backbone LLM occupies the other 4 shards plus most of this one). safetensors
# stores each top-level parameter group contiguously, so rather than
# downloading the whole shard, this test parses the safetensors header (a
# small JSON directory of tensor name -> byte offset/shape/dtype) and issues
# one HTTP Range request per group covering just its byte span: ~641MB for
# "dit.", ~212MB for "patch_encoder." -- under a third of the full shard.
#
# fireredaudio is not published on PyPI, and its GitHub repo root isn't
# pip-installable as-is: it has both fireredaudio/ and assets/ as top-level
# directories, which trips setuptools' flat-layout auto-discovery
# ("Multiple top-level packages discovered in a flat-layout") on a plain
# `pip install`. So this test's setup is "clone the repo and put it on
# PYTHONPATH" rather than "pip install fireredaudio"; see the dedicated
# .github/workflows/fireredaudio.yml for the exact steps. Locally::
#
#     git clone --depth 1 https://github.com/FireRedTeam/FireRedAudio /tmp/fireredaudio_src
#     pip install torch transformers==5.8.0 einops onnxruntime
#     pip install --force-reinstall --no-deps .   # the onnxsim under test
#     PYTHONPATH=/tmp/fireredaudio_src ONNXSIM_RUN_FIREREDAUDIO_TESTS=1 \
#         pytest tests/test_fireredaudio.py -v
#
# One more thing worth knowing if using this test as a reference:
# fireredaudio.flow.modules.Attention.forward hardcodes
# `with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):` around its
# scaled_dot_product_attention call. EFFICIENT_ATTENTION is a CUDA-only
# backend in PyTorch and raises "No viable backend for
# scaled_dot_product_attention" unconditionally on CPU -- which is all
# GitHub-hosted CI runners have. The backend only selects which kernel
# PyTorch's eager runtime uses; it has no effect on the graph
# torch.onnx.export produces (aten::scaled_dot_product_attention is lowered
# to its mathematical decomposition regardless of which backend traced it),
# so this test monkeypatches sdpa_kernel to a no-op context manager for the
# duration of export -- safe, and it changes nothing about the exported ONNX
# graph.
#
# Downloads ~850MB total over the network and is opt-in via
# ONNXSIM_RUN_FIREREDAUDIO_TESTS=1, set by the dedicated weekly
# .github/workflows/fireredaudio.yml.

import contextlib
import json
import os
import struct
import tempfile
import time
import urllib.error
import urllib.request

import pytest

import onnxsim

# A bare ``import onnxruntime`` would fail collection (not skip the test) on
# platforms onnxruntime doesn't ship wheels for, before the
# ONNXSIM_RUN_FIREREDAUDIO_TESTS skipif below even gets a chance to apply.
onnxruntime = pytest.importorskip("onnxruntime")
torch = pytest.importorskip("torch")
# fireredaudio is not a normal test dependency (see module docstring: not on
# PyPI, needs PYTHONPATH rather than pip install) and is not installed in CI
# except by the dedicated fireredaudio.yml workflow.
pytest.importorskip("fireredaudio")

pytestmark = pytest.mark.skipif(
    os.environ.get("ONNXSIM_RUN_FIREREDAUDIO_TESTS") != "1",
    reason="Set ONNXSIM_RUN_FIREREDAUDIO_TESTS=1 to run (downloads ~850MB of real "
    "pretrained weights from Hugging Face; needs fireredaudio on PYTHONPATH).",
)

# onnxsim.test_utils imports torch at module load, so it must follow the
# importorskip guards above (hence the E402 exemptions below).
import fireredaudio.flow.modules as _fireredaudio_modules  # noqa: E402
from fireredaudio.flow.estimator import RedDiT, RedDiTConfig  # noqa: E402
from fireredaudio.flow.patch_encoder import (  # noqa: E402
    RedPatchEncoder,
    RedPatchEncoderConfig,
)

from onnxsim.test_utils import export_simplify_and_check_by_python_api  # noqa: E402

_REPO = "FireRedTeam/FireRedAudio"
_CONFIG_URL = f"https://huggingface.co/{_REPO}/resolve/main/FireRedAudio/config.json"
_SHARD_URL = (
    f"https://huggingface.co/{_REPO}/resolve/main/FireRedAudio/"
    "model-00005-of-00005.safetensors"
)

_DOWNLOAD_ATTEMPTS = 5

_SAFETENSORS_DTYPE_TO_TORCH = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


def _get_with_retries(url: str, *, headers=None) -> bytes:
    last_error: Exception = RuntimeError("unreachable")
    for attempt in range(_DOWNLOAD_ATTEMPTS):
        try:
            request = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(request) as response:
                return response.read()
        except (urllib.error.URLError, OSError) as e:
            last_error = e
            if attempt + 1 < _DOWNLOAD_ATTEMPTS:
                time.sleep(3 * (attempt + 1))
    pytest.skip(
        f"Could not download {url} after {_DOWNLOAD_ATTEMPTS} attempts: {last_error}"
    )


def _fetch_prefixed_state_dict(prefix: str, cache_dir: str) -> dict:
    """Fetches only the ``prefix``-prefixed tensors (e.g. ``"dit."``) of
    FireRedAudio's model-00005-of-00005.safetensors shard, via targeted HTTP
    Range requests -- see the module docstring for why (avoids the other
    ~2GB of that shard, mostly the 9B backbone LLM's weights, which this test
    never loads)."""
    cache_fn = os.path.join(cache_dir, prefix.rstrip(".") + ".pt")
    if os.path.exists(cache_fn):
        return torch.load(cache_fn, weights_only=True)

    header_len_bytes = _get_with_retries(_SHARD_URL, headers={"Range": "bytes=0-7"})
    header_len = struct.unpack("<Q", header_len_bytes)[0]
    header_bytes = _get_with_retries(
        _SHARD_URL, headers={"Range": f"bytes=8-{8 + header_len - 1}"}
    )
    header = json.loads(header_bytes)
    header.pop("__metadata__", None)

    infos = {k: v for k, v in header.items() if k.startswith(prefix)}
    if not infos:
        pytest.skip(f"No tensors with prefix {prefix!r} in the safetensors header")
    data_start = 8 + header_len
    group_lo = min(info["data_offsets"][0] for info in infos.values())
    group_hi = max(info["data_offsets"][1] for info in infos.values())
    span = _get_with_retries(
        _SHARD_URL,
        headers={"Range": f"bytes={data_start + group_lo}-{data_start + group_hi - 1}"},
    )

    state_dict = {}
    for key, info in infos.items():
        t_lo, t_hi = info["data_offsets"]
        t_lo -= group_lo
        t_hi -= group_lo
        dtype = _SAFETENSORS_DTYPE_TO_TORCH[info["dtype"]]
        tensor = torch.frombuffer(bytearray(span[t_lo:t_hi]), dtype=dtype).reshape(
            info["shape"]
        )
        state_dict[key[len(prefix) :]] = tensor.float()

    torch.save(state_dict, cache_fn)
    return state_dict


@pytest.fixture(scope="module")
def firered_audio_config():
    return json.loads(_get_with_retries(_CONFIG_URL))


@pytest.fixture(scope="module")
def firered_audio_cache_dir(tmp_path_factory):
    return str(tmp_path_factory.mktemp("firered_audio_weights"))


@pytest.fixture(autouse=True)
def _force_default_sdpa_backend(monkeypatch):
    # See module docstring: fireredaudio.flow.modules.Attention.forward
    # hardcodes the CUDA-only EFFICIENT_ATTENTION backend, which raises on
    # CPU. No-opping the context manager is safe for export/tracing purposes.
    monkeypatch.setattr(
        _fireredaudio_modules,
        "sdpa_kernel",
        lambda backend: contextlib.nullcontext(),
    )


def test_firered_audio_patch_encoder_simplify(
    firered_audio_config, firered_audio_cache_dir
):
    config = RedPatchEncoderConfig(**firered_audio_config["patch_encoder_config"])
    model = RedPatchEncoder(config).eval()

    state_dict = _fetch_prefixed_state_dict("patch_encoder.", firered_audio_cache_dir)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    assert not missing, f"missing keys loading real RedPatchEncoder weights: {missing}"
    assert not unexpected, (
        f"unexpected keys loading real RedPatchEncoder weights: {unexpected}"
    )

    torch.manual_seed(0)
    inputs_embeds_vae = torch.randn(1, 4 * config.patch_size, config.vae_dim)

    with tempfile.TemporaryDirectory() as tmpdir:
        export_fn = os.path.join(tmpdir, "patch_encoder.onnx")
        torch.onnx.export(
            model,
            (inputs_embeds_vae,),
            export_fn,
            opset_version=17,
            do_constant_folding=True,
            input_names=["inputs_embeds_vae"],
            output_names=["hidden_states"],
            dynamo=False,
        )
        loaded, _pool = onnxsim.load_model(export_fn)
        nodes_before = len(loaded.graph.node)

    opt = export_simplify_and_check_by_python_api(
        model,
        (inputs_embeds_vae,),
        export_kwargs={
            "opset_version": 17,
            "do_constant_folding": True,
            "input_names": ["inputs_embeds_vae"],
            "output_names": ["hidden_states"],
        },
    )

    # The simplified graph must still carry what makes RedPatchEncoder a
    # transformer over rotary-embedded patches: multi-head attention
    # (Softmax). If a future onnxsim change silently drops/miscompiles it,
    # the numerical check inside export_simplify_and_check_by_python_api
    # would already fail, but assert its presence too so the test documents
    # what it protects.
    op_types = {node.op_type for node in opt.graph.node}
    assert "Softmax" in op_types
    # Unlike RedDiT's estimator (test below), whose timestep embedding
    # depends on the runtime `t` input and must stay dynamic, this rotary
    # embedding's Sin/Cos depend only on the patch's fixed length
    # (1 + patch_size), not on any runtime input -- a correctly-simplifying
    # onnxsim constant-folds them away entirely, along with the rest of that
    # position-index arithmetic, which should show up as a real node-count
    # drop.
    assert not ({"Sin", "Cos"} & op_types)
    assert len(opt.graph.node) < nodes_before

    session = onnxruntime.InferenceSession(opt.SerializeToString())
    outputs = session.run(None, {"inputs_embeds_vae": inputs_embeds_vae.numpy()})
    assert [o.name for o in session.get_outputs()] == ["hidden_states"]
    assert outputs[0].shape == (1, 4, config.out_dim)


def test_firered_audio_dit_estimator_simplify(
    firered_audio_config, firered_audio_cache_dir
):
    config = RedDiTConfig(**firered_audio_config["dit_config"])
    dit = RedDiT(config).eval()

    state_dict = _fetch_prefixed_state_dict("dit.", firered_audio_cache_dir)
    missing, unexpected = dit.load_state_dict(state_dict, strict=False)
    assert not missing, f"missing keys loading real RedDiT weights: {missing}"
    assert not unexpected, f"unexpected keys loading real RedDiT weights: {unexpected}"

    # RedDiT has no top-level forward(); _forward_estimator is the actual
    # flow-matching vector-field network that RedDiT.generate()'s Euler loop
    # calls once per sampling step. Wrap it so torch.onnx.export can trace a
    # plain positional forward().
    class _EstimatorWrapper(torch.nn.Module):
        def __init__(self, dit):
            super().__init__()
            self.dit = dit

        def forward(self, x, t):
            return self.dit._forward_estimator(x, t)

    wrapper = _EstimatorWrapper(dit).eval()

    torch.manual_seed(0)
    # One history window (history_patches patches) plus the current noisy
    # patch being denoised, matching what RedDiT.generate() feeds
    # _forward_estimator each step.
    seq_len = config.history_patches * config.patch_size + config.patch_size
    x = torch.randn(1, seq_len, config.vae_channels + config.hidden_size)
    t = torch.rand(1, 1, 1)

    opt = export_simplify_and_check_by_python_api(
        wrapper,
        (x, t),
        export_kwargs={
            "opset_version": 17,
            "do_constant_folding": True,
            "input_names": ["x", "t"],
            "output_names": ["v"],
        },
    )

    # The simplified graph must still carry what makes RedDiT's estimator
    # what it is: AdaLN-conditioned attention (Softmax) plus each DiTBlock's
    # Conv1d branch, and both RoPE and the timestep embedder's sinusoidal
    # embedding (Sin/Cos).
    op_types = {node.op_type for node in opt.graph.node}
    assert "Conv" in op_types
    assert "Softmax" in op_types
    assert {"Sin", "Cos"} <= op_types

    session = onnxruntime.InferenceSession(opt.SerializeToString())
    outputs = session.run(None, {"x": x.numpy(), "t": t.numpy()})
    assert [o.name for o in session.get_outputs()] == ["v"]
    assert outputs[0].shape == (1, seq_len, config.vae_channels)
