# Integration/regression test: onnxsim against real ONNX exports from the
# Audio8 org on Hugging Face (https://huggingface.co/Audio8), a set of
# permissively-licensed TTS/ASR audio models shipped with ONNX Runtime
# deployment packages. This guards against onnxsim regressions on the kind of
# LLM-style, KV-cache-carrying, quantized graphs those packages ship:
# ``com.microsoft`` weight-only INT4 ops (``MatMulNBits``,
# ``GatherBlockQuantized``), ORT dynamic-quantization QOperator ops
# (``DynamicQuantizeLinear``/``MatMulInteger``), fp16 Conv/ConvTranspose
# codec stacks, and autoregressive decoder graphs with 8-16 parallel KV-cache
# input/output pairs.
#
# One real onnxsim bug was found and fixed via this model set:
# ``fast_ar_int4.onnx``'s attention blocks simplify to a model that fails to
# load in onnxruntime with "Invalid tensor data type 0". Root cause: onnx's
# Graph-native shape inference (``onnx::InferShapesOnGraph``) left a handful
# of Reshape-output ``value_info`` entries with a shape but no resolved
# element type (``elem_type == UNDEFINED``); ``onnx.checker.check_model``
# tolerates that malformed TypeProto, but onnxruntime's model loader does
# not. See ``DropIncompleteValueInfo`` in onnxsim.cpp, which strips such
# entries (value_info is optional annotation; dropping an incomplete entry
# cannot change model semantics) -- this test's ``fast_ar_int4.onnx`` case
# guards against that regressing.
#
# Downloads real model weights (10-290MB per case, ~450MB total across all 6)
# from Hugging Face and, for the larger TTS models, takes over a minute per
# model under check_n=3 -- too slow/network-dependent for every PR. It is
# opt-in via ONNXSIM_RUN_AUDIO8_TESTS=1, set by the dedicated weekly
# .github/workflows/audio8.yml. To run it locally::
#
#     pip install onnxruntime
#     pip install --force-reinstall --no-deps .   # the onnxsim under test
#     ONNXSIM_RUN_AUDIO8_TESTS=1 pytest tests/test_audio8.py -v

import os
import time
import urllib.error
import urllib.request

import onnx
import pytest

import onnxsim

# A bare ``import onnxruntime`` would fail collection (not skip the test) on
# platforms onnxruntime doesn't ship wheels for, before the
# ONNXSIM_RUN_AUDIO8_TESTS skipif below even gets a chance to apply.
onnxruntime = pytest.importorskip("onnxruntime")

pytestmark = pytest.mark.skipif(
    os.environ.get("ONNXSIM_RUN_AUDIO8_TESTS") != "1",
    reason="Set ONNXSIM_RUN_AUDIO8_TESTS=1 to run (downloads real models "
    "from Hugging Face; the TTS cases take over a minute each).",
)

_HF_BASE = "https://huggingface.co"


# Pinned to a specific commit (rather than "main") for reproducibility, so an
# upstream model update can't turn this test flaky -- bump deliberately if a
# case ever needs to track a new upload.
#
# name -> (repo, revision, path, has_external_data, test_input_shapes,
#           input_fill, check_tol)
#
# ``test_input_shapes`` fixes the one truly dynamic dim each model has;
# dims left out rely on onnxsim's "dynamic dim[0] -> assume batch size 1"
# default (matches every other input in these graphs, which are otherwise
# fully static once batch is pinned). ``check_tol`` overrides onnxsim's
# default (rtol=1e-4, atol=1e-5), which assumes fp32: codec_decoder_fp16.onnx
# computes entirely in float16 (~3 decimal digits of precision), and its
# near-zero waveform samples fail the fp32-tuned default by a few 1e-4 even
# though the model is simplified correctly -- None keeps the default.
CASES = {
    # Audio8 TTS Preview 0.6B, DualAR architecture: weight-only INT4
    # transformer step (MatMulNBits/GatherBlockQuantized) with 4 parallel
    # KV-cache pairs and a bool "use_slow_hidden" gate input.
    "fast_ar_int4.onnx": (
        "Audio8/Audio8-TTS-Preview-0.6B-ONNX-INT4",
        "818569c6b832118ad68d61bbd873abe250fcd68a",
        "fast_ar_int4.onnx",
        True,
        None,
        "random",
        None,
    ),
    # Same package's fp16 neural audio codec decoder: Conv/ConvTranspose
    # stack with LayerNormalization, Sin/Cos rotary embeddings, and a
    # dynamic "frames" dim onnxsim must be told to fix for check_n.
    "codec_decoder_fp16.onnx": (
        "Audio8/Audio8-TTS-Preview-0.6B-ONNX-INT4",
        "818569c6b832118ad68d61bbd873abe250fcd68a",
        "codec_decoder_fp16.onnx",
        True,
        {"codes": [1, 10, 50]},
        "random",
        (1e-2, 2e-3),
    ),
    # Audio8 ASR 0.1B, decode step: a fully statically-shaped (fixed
    # 512-token KV cache) INT4 transformer step -- no dynamic dims at all.
    "lm_cache_decode_int4.onnx": (
        "Audio8/Audio8-ASR-0.1B-onnx-runtime",
        "5b6d058a54853700223dd23cb4fe466b86c8fece",
        "model_bundle/lm_cache_decode_int4.onnx",
        True,
        None,
        "random",
        None,
    ),
    # Same package's prefill step: variable-length "sequence" dim across
    # inputs_embeds/cache_position that onnxsim must propagate consistently.
    "lm_cache_prefill_int4.onnx": (
        "Audio8/Audio8-ASR-0.1B-onnx-runtime",
        "5b6d058a54853700223dd23cb4fe466b86c8fece",
        "model_bundle/lm_cache_prefill_int4.onnx",
        True,
        {"inputs_embeds": [1, 5, 512], "cache_position": [5]},
        "random",
        None,
    ),
    # ark-asr-0.6b's audio-feature adapter MLP: ORT dynamic quantization
    # (DynamicQuantizeLinear/MatMulInteger), already minimal going in -- this
    # case asserts onnxsim is a safe no-op rather than that it finds work.
    "ark_asr_audio_encoder_adapter_int8.onnx": (
        "Audio8/ark-asr-0.6b-int8-onnx",
        "ced3e6c0cc45eda7f718b885e9fd9562ed3ec94d",
        "audio_encoder_adapter_int8.onnx",
        False,
        {"merged_audio_features": [1, 50, 5120]},
        "random",
        None,
    ),
    # GPA-v1.5's sibling adapter model: same architecture/op set as the
    # ark-asr case above from a related export pipeline, different weights.
    "gpa_audio_encoder_adapter_int8.onnx": (
        "Audio8/GPA-v1.5-onnx-runtime",
        "b9f463b5a29461257d45bb734463edf41a297d86",
        "model/audio_encoder_adapter_int8.onnx",
        False,
        {"merged_audio_features": [1, 50, 5120]},
        "random",
        None,
    ),
}


@pytest.fixture(scope="module")
def audio8_model_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("audio8_models")


_DOWNLOAD_ATTEMPTS = 5


def _download_one(url: str, dest: str) -> None:
    if os.path.exists(dest):
        return
    tmp_dest = f"{dest}.part"
    last_error: Exception = RuntimeError("unreachable")
    for attempt in range(_DOWNLOAD_ATTEMPTS):
        try:
            # Some of these files are 100+MB; a plain GET occasionally comes
            # back truncated on a flaky connection. Download to a temp name
            # and only rename into place on a fully successful transfer, so a
            # partial file is never mistaken for a complete one on retry or by
            # a later test run.
            urllib.request.urlretrieve(url, tmp_dest)
            os.replace(tmp_dest, dest)
            return
        except (urllib.error.URLError, OSError) as e:
            last_error = e
            if os.path.exists(tmp_dest):
                os.remove(tmp_dest)
            if attempt + 1 < _DOWNLOAD_ATTEMPTS:
                time.sleep(3 * (attempt + 1))
    pytest.skip(f"Could not download {url} after {_DOWNLOAD_ATTEMPTS} attempts: {last_error}")


def _download(filename: str, dest_dir) -> str:
    repo, revision, path, has_external_data, _, _, _ = CASES[filename]
    base_url = f"{_HF_BASE}/{repo}/resolve/{revision}/{path}"
    dest = str(dest_dir / filename)
    _download_one(base_url, dest)
    if has_external_data:
        # External-data tensors are referenced by filename relative to the
        # .onnx file, so the sidecar must land next to it under the same name
        # onnx.load() expects: "<file>.onnx.data".
        _download_one(f"{base_url}.data", f"{dest}.data")
    return dest


@pytest.mark.parametrize("filename", sorted(CASES.keys()))
def test_audio8_model_simplify(audio8_model_dir, filename):
    _, _, _, _, test_input_shapes, input_fill, check_tol = CASES[filename]
    path = _download(filename, audio8_model_dir)

    nodes_before = len(onnx.load(path, load_external_data=False).graph.node)

    check_rtol, check_atol = check_tol or (1e-4, 1e-5)
    model_opt, check_ok = onnxsim.simplify(
        path,
        check_n=3,
        test_input_shapes=test_input_shapes,
        input_fill=input_fill,
        check_rtol=check_rtol,
        check_atol=check_atol,
    )
    assert check_ok, f"{filename}: onnxsim numerical check failed"
    assert len(model_opt.graph.node) <= nodes_before, (
        f"{filename}: simplification must not increase node count"
    )

    # The simplified model must still load in onnxruntime. This is the
    # regression this test set caught: a malformed value_info (elem_type ==
    # UNDEFINED) that onnx.checker.check_model tolerates but onnxruntime's
    # loader rejects outright (see DropIncompleteValueInfo in onnxsim.cpp).
    onnxruntime.InferenceSession(model_opt.SerializeToString())
