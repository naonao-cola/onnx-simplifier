# Integration/regression test: onnxsim against Kyutai's Pocket TTS
# (https://github.com/kyutai-labs/pocket-tts, "a TTS that fits in your CPU
# (and pocket)"), a small streaming-state text-to-speech model. The upstream
# checkpoints are gated, so this test targets the community ONNX export
# published at https://huggingface.co/KevinAHM/pocket-tts-onnx (exported by
# https://github.com/KevinAHM/pocket-tts-onnx-export from the real
# ``kyutai/pocket-tts`` weights), a public, non-gated mirror of the same
# graphs -- no HF auth needed to download.
#
# The bundle covers pocket-tts's whole pipeline and, between the five graphs,
# exercises the streaming/stateful patterns onnxsim needs to handle
# correctly:
#
#   * ``text_conditioner.onnx``  -- a single Gather-based token embedding.
#   * ``mimi_encoder.onnx``      -- audio -> latent Conv/Transformer encoder.
#   * ``flow_lm_flow.onnx``      -- the stateless flow-matching (Euler) step.
#   * ``mimi_decoder.onnx``      -- latent -> audio streaming decoder: a
#     Mod/ScatterND ring-buffer KV cache plus dozens of small Conv "previous
#     frame" states, several of them ``bool`` flags.
#   * ``flow_lm_main.onnx``      -- the transformer backbone: 18 KV-cache
#     state tensors per call, including a genuinely **static, zero-length**
#     dimension (``state_1``'s shape is ``(0,)``) used as an "empty so far"
#     sentinel -- not a symbolic/dynamic dim defaulting to 0, an exported
#     dim actually fixed at 0.
#
# That last detail is exactly what this test set was written to guard
# against: ``onnxsim.generate_random_calibration_data`` (via its internal
# ``_input_specs``) used to promote *any* non-positive ``dim_value`` to 1,
# conflating "genuinely fixed to 0" with "unset/symbolic" -- so calibrating
# or measuring accuracy drop on ``flow_lm_main.onnx``/``mimi_decoder.onnx``
# built a shape ONNX Runtime rejected outright, and that exception was
# silently swallowed by ``recommend_quantization``'s candidate loop, which
# reported a generic "no candidate quantization scheme both applied to this
# model and shrank it" instead of the real calibration failure. Fixed in
# ``onnxsim/calibration.py`` by checking ``dim.HasField("dim_value")``
# instead of ``dim.dim_value > 0`` (matching the pattern ``model_info.py``
# already used for the same ambiguity); see
# ``test_generate_random_calibration_data_keeps_static_zero_dim`` in
# ``test_static_quantize_matmul.py`` for the minimal unit-level regression
# test, and ``test_pocket_tts_flow_lm_main_quantize_weight_only`` below for
# end-to-end coverage on the real model that surfaced it.
#
# Two more things worth knowing if you're using this test set as a
# reference for quantizing similar exports:
#
#   * onnxsim's weight quantizers only recognize a MatMul/Gemm's weight
#     input when it is *directly* a graph initializer. The legacy
#     TorchScript ``torch.onnx.export`` path used by pocket-tts's exporter
#     emits each ``nn.Linear`` as ``MatMul(x, Transpose(weight))`` --
#     quantizing the raw export directly finds almost nothing to quantize.
#     Running ``onnxsim.simplify()`` first folds ``Transpose(initializer)``
#     into a plain pre-transposed initializer, which is what actually
#     exposes the real Linear weights to every ``quantize_*`` function.
#     Always simplify before quantizing.
#   * ``quantize_weight_only_int4`` needs ONNX opset 21 for the native INT4
#     tensor type and silently no-ops on pocket-tts's opset-14 export;
#     ``onnxsim.simplify(model, target_opset_version=21)`` upgrades it in
#     place before quantizing.
#
# Downloads real model weights (16-290MB per case, ~460MB total across all
# 5) over the network and, for flow_lm_main.onnx, takes well over a minute
# under check_n=3 -- too slow/network-dependent for every PR. It is opt-in
# via ONNXSIM_RUN_POCKET_TTS_TESTS=1, set by the dedicated weekly
# .github/workflows/pocket_tts.yml. To run it locally::
#
#     pip install onnxruntime
#     pip install --force-reinstall --no-deps .   # the onnxsim under test
#     ONNXSIM_RUN_POCKET_TTS_TESTS=1 pytest tests/test_pocket_tts.py -v

import os
import time
import urllib.error
import urllib.request

import numpy as np
import onnx
import pytest

# A bare ``import onnxruntime`` would fail collection (not skip the test) on
# platforms onnxruntime doesn't ship wheels for, before the
# ONNXSIM_RUN_POCKET_TTS_TESTS skipif below even gets a chance to apply.
onnxruntime = pytest.importorskip("onnxruntime")

pytestmark = pytest.mark.skipif(
    os.environ.get("ONNXSIM_RUN_POCKET_TTS_TESTS") != "1",
    reason="Set ONNXSIM_RUN_POCKET_TTS_TESTS=1 to run (downloads real models "
    "from Hugging Face; flow_lm_main.onnx takes well over a minute).",
)

import onnxsim  # noqa: E402

_HF_BASE = "https://huggingface.co/KevinAHM/pocket-tts-onnx/resolve/main/onnx"

# filename -> (test_input_shapes, input_fill). Dims left out rely on
# onnxsim's "dynamic dim[0] -> assume batch size 1" default -- true of every
# input here except text_conditioner's token_ids and mimi_encoder's audio,
# whose dynamic length sits at position 1/2 instead.
_SIMPLE_CASES = {
    "text_conditioner.onnx": ({"token_ids": [1, 20]}, "zeros"),  # valid vocab index 0
    "mimi_encoder.onnx": ({"audio": [1, 1, 24000]}, "random"),
    "flow_lm_flow.onnx": ({}, "random"),
}

# mimi_decoder.onnx and flow_lm_main.onnx carry dozens of stateful KV-cache
# inputs of mixed dtype (float/int64/bool) and, for flow_lm_main.onnx, a
# genuinely static zero-length dimension -- exactly the case
# generate_random_calibration_data's own input-shape inference now handles
# correctly (see the module docstring). Building their check_n input through
# it, rather than onnxsim's CLI-level input_fill, both sidesteps hand-rolling
# per-state shapes/dtypes and doubles as coverage for that fix on the real
# model that surfaced it.
_STATEFUL_CASES = ["mimi_decoder.onnx", "flow_lm_main.onnx"]


@pytest.fixture(scope="module")
def pocket_tts_model_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("pocket_tts_onnx")


_DOWNLOAD_ATTEMPTS = 5


def _download(filename: str, dest_dir) -> str:
    dest = str(dest_dir / filename)
    if os.path.exists(dest):
        return dest
    url = f"{_HF_BASE}/{filename}"
    tmp_dest = f"{dest}.part"
    last_error: Exception = RuntimeError("unreachable")
    for attempt in range(_DOWNLOAD_ATTEMPTS):
        try:
            # flow_lm_main.onnx alone is ~290MB; a plain GET occasionally comes
            # back truncated on a flaky connection. Download to a temp name and
            # only rename into place on a fully successful transfer, so a
            # partial file is never mistaken for a complete one on retry or by
            # a later test run.
            urllib.request.urlretrieve(url, tmp_dest)
            os.replace(tmp_dest, dest)
            return dest
        except (urllib.error.URLError, OSError) as e:
            last_error = e
            if os.path.exists(tmp_dest):
                os.remove(tmp_dest)
            if attempt + 1 < _DOWNLOAD_ATTEMPTS:
                time.sleep(3 * (attempt + 1))
    pytest.skip(
        f"Could not download {filename} from Hugging Face after "
        f"{_DOWNLOAD_ATTEMPTS} attempts: {last_error}"
    )


@pytest.mark.parametrize("filename", sorted(_SIMPLE_CASES.keys()))
def test_pocket_tts_model_simplify(pocket_tts_model_dir, filename):
    test_input_shapes, input_fill = _SIMPLE_CASES[filename]
    path = _download(filename, pocket_tts_model_dir)

    nodes_before = len(onnx.load(path, load_external_data=False).graph.node)

    model_opt, check_ok = onnxsim.simplify(
        path,
        check_n=3,
        test_input_shapes=test_input_shapes or None,
        input_fill=input_fill,
    )
    assert check_ok, f"{filename}: onnxsim numerical check failed"
    assert len(model_opt.graph.node) <= nodes_before, (
        f"{filename}: simplification must not increase node count"
    )

    # The simplified model must still load and run in onnxruntime.
    onnxruntime.InferenceSession(model_opt.SerializeToString())


@pytest.mark.parametrize("filename", _STATEFUL_CASES)
def test_pocket_tts_stateful_model_simplify(pocket_tts_model_dir, filename):
    path = _download(filename, pocket_tts_model_dir)
    original = onnx.load(path, load_external_data=False)
    nodes_before = len(original.graph.node)

    # A single realistic batch, built the same way onnxsim's own calibration
    # helpers do -- reused here (rather than hand-rolling ~20-60 per-state
    # shapes/dtypes) both for convenience and because it exercises the exact
    # zero-length-dim shape inference this test module exists to guard.
    input_data = onnxsim.generate_random_calibration_data(
        original, num_samples=1, seed=0
    )[0]

    # check_n=1: input_data is one fixed batch reused for every check_n
    # trial, so repeating it further would just re-run the same comparison.
    model_opt, check_ok = onnxsim.simplify(path, check_n=1, input_data=input_data)
    assert check_ok, f"{filename}: onnxsim numerical check failed"
    assert len(model_opt.graph.node) < nodes_before, (
        f"{filename}: simplification must reduce node count "
        f"(these graphs carry a lot of foldable RoPE/mask/state-index "
        f"arithmetic)"
    )

    session = onnxruntime.InferenceSession(model_opt.SerializeToString())
    outputs = session.run(None, {k: v for k, v in input_data.items()})
    assert len(outputs) == len(original.graph.output)


def test_pocket_tts_flow_lm_main_quantize_weight_only(pocket_tts_model_dir):
    # Regression coverage, on the real model that surfaced it, for the
    # generate_random_calibration_data zero-dim fix this test module's
    # docstring describes: measure_accuracy_drop (called here through
    # quantize_weight_only + measure_accuracy_drop directly, the same pair
    # recommend_quantization uses internally) must not raise on
    # flow_lm_main.onnx's genuinely-zero-length "state_1" input.
    #
    # Quantizing the raw export directly finds almost no Linear weights (see
    # module docstring: torch.onnx's legacy exporter hides them behind a
    # Transpose): simplify first to fold Transpose(initializer) into a plain
    # initializer, exposing flow_lm_main's real ~226MB of Linear weights to
    # the quantizer.
    path = _download("flow_lm_main.onnx", pocket_tts_model_dir)
    simplified, _ = onnxsim.simplify(path, check_n=0)  # check_n=0: no numerical check, just the fold

    float_bytes = sum(
        len(t.raw_data) if t.HasField("raw_data") else len(t.SerializeToString())
        for t in simplified.graph.initializer
    )

    quantized = onnxsim.quantize_weight_only(simplified)  # int8, per-channel

    report = onnxsim.measure_accuracy_drop(simplified, quantized, num_samples=4, seed=0)
    assert report.all_finite, "quantized flow_lm_main.onnx produced non-finite output"
    # ~0.03 observed; keep a wide margin -- this asserts "quantization is in
    # the ballpark it should be", not a tight regression pin on the exact
    # float value, since that would make the test flaky against unrelated
    # onnxsim graph-optimization changes that shift constant-folding order.
    assert report.worst_relative_l2 < 0.15

    quantized_simplified, _ = onnxsim.simplify(quantized, check_n=0)
    quant_bytes = sum(
        len(t.raw_data) if t.HasField("raw_data") else len(t.SerializeToString())
        for t in quantized_simplified.graph.initializer
    )
    # Re-simplifying after quantization is required to prune the now-unused
    # float32 initializers quantize_weight_only leaves behind (it swaps each
    # MatMul's weight input for a DequantizeLinear, but doesn't itself drop
    # the now-orphaned original initializer) -- comparing sizes measured
    # before/after that cleanup is a routine papercut, not a bug in either
    # function individually.
    assert quant_bytes < float_bytes * 0.5

    onnxruntime.InferenceSession(quantized_simplified.SerializeToString())
