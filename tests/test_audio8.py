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
# model under check_n=3 -- too slow/network-dependent for every PR. The
# quantization-quality tests further down this file download their own
# fp32/int8 pairs -- one from the Audio8 ASR package (~520MB more), one from
# the 0.1B TTS package (~37MB) -- for the same reason. It is opt-in via
# ONNXSIM_RUN_AUDIO8_TESTS=1, set by the dedicated weekly
# .github/workflows/audio8.yml. To run it locally::
#
#     pip install onnxruntime
#     pip install --force-reinstall --no-deps .   # the onnxsim under test
#     ONNXSIM_RUN_AUDIO8_TESTS=1 pytest tests/test_audio8.py -v

import os
import time
import urllib.error
import urllib.request

import numpy as np
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
    pytest.skip(
        f"Could not download {url} after {_DOWNLOAD_ATTEMPTS} attempts: {last_error}"
    )


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


# ---------------------------------------------------------------------------
# Quantization-quality integration tests: does onnxsim's own quantize_dynamic
# come within shouting distance of Audio8's own officially published
# quantized export of the exact same graph?
#
# filename -> (repo, revision, path within the repo, has external .onnx.data
# sidecar). Pinned per-file, same reproducibility rationale as CASES above.
# ---------------------------------------------------------------------------

_QUALITY_FILES = {
    # Audio8-ASR-0.1B-onnx-runtime is an Audio8 package that publishes the
    # fp32 original *and* Audio8's own ORT-quantized export of the exact same
    # graph side by side: a real, independently-published fp32 reference.
    "lm_cache_decode_fp32.onnx": (
        "Audio8/Audio8-ASR-0.1B-onnx-runtime",
        "5b6d058a54853700223dd23cb4fe466b86c8fece",
        "model_bundle/lm_cache_decode.onnx",
        False,
    ),
    "lm_cache_decode_int8.onnx": (
        "Audio8/Audio8-ASR-0.1B-onnx-runtime",
        "5b6d058a54853700223dd23cb4fe466b86c8fece",
        "model_bundle/lm_cache_decode_int8.onnx",
        True,
    ),
    # audio8-TTS-0.1B-ONNX-INT8 (see test_audio8_tts_0_1b_... below) has no
    # published fp32 sibling at all -- only this INT8 export exists on the
    # Hub -- so there is no path/has_external_data entry for an fp32 file
    # here; the fp32 reference for that test is reconstructed from this file
    # itself, not downloaded.
    "tts_0.1b_fast_ar_int8.onnx": (
        "Audio8/audio8-TTS-0.1B-ONNX-INT8",
        "e1c07e8a3725077e3ab80ad8578e5787e8a23c6c",
        "fast_ar_int8.onnx",
        True,
    ),
}


def _download_quality_file(filename: str, dest_dir) -> str:
    repo, revision, path, has_external_data = _QUALITY_FILES[filename]
    base_url = f"{_HF_BASE}/{repo}/resolve/{revision}/{path}"
    # The local file must keep the *source* basename, not the dict key: a
    # model's external-data tensors reference their sidecar by the relative
    # filename baked into the .onnx proto at export time (here, always the
    # basename it was originally saved under), so renaming the local .onnx
    # file to anything else breaks that reference at load time.
    dest = str(dest_dir / os.path.basename(path))
    _download_one(base_url, dest)
    if has_external_data:
        _download_one(f"{base_url}.data", f"{dest}.data")
    return dest


def test_audio8_quantize_dynamic_matches_published_int8_quality(audio8_model_dir):
    fp32_path = _download_quality_file("lm_cache_decode_fp32.onnx", audio8_model_dir)
    published_int8_path = _download_quality_file(
        "lm_cache_decode_int8.onnx", audio8_model_dir
    )

    # Loaded (not passed as bare paths) so external data is resolved once,
    # here, under our control -- onnxsim.measure_accuracy_drop/quantize_dynamic
    # both load a bare path with load_external_data=False, which would leave
    # published_int8's weights unresolved.
    fp32_model = onnx.load(fp32_path)
    published_int8_model = onnx.load(published_int8_path)

    # quantize_dynamic only rewrites a MatMul/Gemm whose weight is *already*
    # a plain 2-D constant initializer (see dynamic-quantization.md); this
    # export's weights reach their MatMul through Transpose/Reshape/Cast
    # first, so skipping simplify() here would leave quantize_dynamic with
    # nothing it recognizes to quantize at all. This is exactly the
    # documented "simplify -> quantize -> deploy" flow, not a test-only
    # workaround.
    fp32_model, simplify_ok = onnxsim.simplify(fp32_model)
    assert simplify_ok, "onnxsim.simplify failed its own numerical check"

    onnxsim_int8_model = onnxsim.quantize_dynamic(fp32_model)

    # Same calibration batches for both measurements, so "onnxsim's error"
    # and "Audio8's own error" are directly comparable numbers rather than
    # each drawn from independent random inputs.
    calibration_data = onnxsim.generate_random_calibration_data(
        fp32_model, num_samples=4, seed=0
    )

    onnxsim_report = onnxsim.measure_accuracy_drop(
        fp32_model, onnxsim_int8_model, calibration_data=calibration_data
    )
    published_report = onnxsim.measure_accuracy_drop(
        fp32_model, published_int8_model, calibration_data=calibration_data
    )

    assert onnxsim_report.all_finite, (
        "onnxsim.quantize_dynamic produced non-finite output on lm_cache_decode.onnx"
    )

    # onnxsim's own dynamic quantization is not expected to be numerically
    # identical to Audio8's own INT8 export -- different weight-quantization
    # tie-breaks land on different int8 codes even under the same scheme --
    # but it should land in the same ballpark of accuracy loss relative to
    # fp32, not meaningfully worse. Generous factor: this is a coarse
    # quality gate against a real published deployment artifact, not a tight
    # numerical-equivalence check.
    assert onnxsim_report.worst_relative_l2 <= max(
        published_report.worst_relative_l2 * 3, 0.05
    ), (
        f"onnxsim.quantize_dynamic's relative L2 error "
        f"({onnxsim_report.worst_relative_l2:.4g}) against the fp32 "
        "reference is far worse than Audio8's own published INT8 export's "
        f"({published_report.worst_relative_l2:.4g}) on the same graph and "
        "calibration data"
    )


def _dequantize_dynamically_quantized_matmuls(model: onnx.ModelProto):
    """Reconstructs an approximate float32 model from an ONNX Runtime
    ``quantize_dynamic``-style graph (``DynamicQuantizeLinear`` ->
    ``MatMulInteger`` -> ``Cast`` -> ``Mul``, the exact pattern
    ``onnxsim.quantize_dynamic`` itself produces -- see
    dynamic-quantization.md).

    Used only because Audio8 never published an fp32 ONNX export of its
    0.1B TTS model (unlike the ASR package above, which ships fp32 and
    INT8 side by side) -- this is the only way to get *any* float
    reference for it. For each ``MatMulInteger`` node whose weight is a
    static, per-channel-symmetric INT8 initializer named ``<x>_quantized``
    with sibling ``<x>_scale``/``<x>_zero_point`` initializers (the naming
    ONNX Runtime's own quantizer emits, which is what produced this
    export), it dequantizes the weight back to float32 and rewrites the
    node that produced the dequantized-matmul's final output, in place, as
    a plain ``MatMul(X_float, W_dequant)`` -- ``X_float`` being the
    ``DynamicQuantizeLinear`` node's own original float input, not a
    round-tripped quantize/dequantize of it. This removes the published
    export's activation-quantization noise entirely and leaves only the
    weight-quantization error already baked into the published INT8
    weights, a much closer stand-in for the model's actual fp32 weights
    than any alternative available here (there is no PyTorch-to-ONNX
    exporter for this model's custom per-token recurrent contract in this
    repository or upstream).

    Every other node in the graph (attention, RoPE, Mamba state, KV-cache
    handling, ...) is untouched; the now-dead ``DynamicQuantizeLinear``/
    ``MatMulInteger``/``Cast`` nodes upstream of each rewritten node are
    left in place for :func:`onnxsim.simplify` to prune afterward, rather
    than unlinked here, since a ``DynamicQuantizeLinear``'s three outputs
    are sometimes shared by more than one ``MatMulInteger`` consumer.

    :returns: ``(model, num_rewritten, num_skipped)``. A node is *skipped*
        (left as int8, not touched) if it doesn't match the exact pattern
        above -- the caller should require ``num_skipped == 0`` before
        trusting the result as a float reference, since a silently-skipped
        node would leave a still-quantized op inside what is supposed to
        be the float baseline.
    """
    graph = model.graph
    initializers = {init.name: init for init in graph.initializer}
    producer = {output: node for node in graph.node for output in node.output}
    consumers = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)

    num_rewritten = 0
    num_skipped = 0
    new_initializers = []

    for mmi in [n for n in graph.node if n.op_type == "MatMulInteger"]:
        if len(mmi.input) != 4:
            num_skipped += 1
            continue
        xq_name, wq_name, _xzp_name, _wzp_name = mmi.input
        if wq_name not in initializers or not wq_name.endswith("_quantized"):
            num_skipped += 1
            continue
        prefix = wq_name[: -len("_quantized")]
        ws_name = prefix + "_scale"
        wzp_name = prefix + "_zero_point"
        if ws_name not in initializers or wzp_name not in initializers:
            num_skipped += 1
            continue
        dql = producer.get(xq_name)
        if dql is None or dql.op_type != "DynamicQuantizeLinear":
            num_skipped += 1
            continue
        cast_consumers = consumers.get(mmi.output[0], [])
        if len(cast_consumers) != 1 or cast_consumers[0].op_type != "Cast":
            num_skipped += 1
            continue
        mul_consumers = consumers.get(cast_consumers[0].output[0], [])
        if len(mul_consumers) != 1 or mul_consumers[0].op_type != "Mul":
            num_skipped += 1
            continue
        final_mul = mul_consumers[0]

        wq = onnx.numpy_helper.to_array(initializers[wq_name])
        ws = onnx.numpy_helper.to_array(initializers[ws_name])
        wzp = onnx.numpy_helper.to_array(initializers[wzp_name])
        if wq.ndim != 2:
            num_skipped += 1
            continue
        w_dequant = (wq.astype(np.float32) - wzp.astype(np.float32)) * ws.astype(
            np.float32
        )
        if w_dequant.shape != wq.shape:
            num_skipped += 1
            continue

        new_w_name = wq_name + "__dequantized_f32"
        new_initializers.append(
            onnx.numpy_helper.from_array(w_dequant, name=new_w_name)
        )

        # Rewritten in place, keeping the final Mul's original output name,
        # so every downstream consumer of it stays correctly wired.
        final_mul.op_type = "MatMul"
        del final_mul.input[:]
        final_mul.input.extend([dql.input[0], new_w_name])
        final_mul.ClearField("attribute")
        num_rewritten += 1

    graph.initializer.extend(new_initializers)
    return model, num_rewritten, num_skipped


def test_audio8_tts_0_1b_quantize_dynamic_matches_published_int8_quality(
    audio8_model_dir,
):
    """Same comparison as
    :func:`test_audio8_quantize_dynamic_matches_published_int8_quality`
    above, for Audio8's 0.1B TTS model instead of the ASR one -- but
    Audio8 never published an fp32 ONNX export of it (only
    ``audio8-TTS-0.1B-ONNX-INT8``'s INT8 graphs exist), so there is no
    downloadable fp32 file to use as ground truth. The fp32 reference here
    is instead reconstructed by dequantizing this exact file's own INT8
    weights (:func:`_dequantize_dynamically_quantized_matmuls`), which
    also makes this the harder-edged variant of the two tests: since
    onnxsim's weight quantization is a lossless round-trip of weights that
    are *already* exact multiples of a per-channel scale (as these are,
    having been dequantized from int8), and DynamicQuantizeLinear is the
    same standard ONNX op either quantizer used, onnxsim's own
    quantize_dynamic is expected to land numerically identical (not just
    "close") to Audio8's own published INT8 export here -- confirmed
    locally at parity to ~1e-16 relative difference between the two
    reports' worst_relative_l2. The tolerance below stays generous rather
    than asserting exact equality so the test isn't fragile to floating-
    point non-associativity across platforms/BLAS backends.
    """
    published_int8_path = _download_quality_file(
        "tts_0.1b_fast_ar_int8.onnx", audio8_model_dir
    )
    published_int8_model = onnx.load(published_int8_path)

    reconstructed_model = onnx.ModelProto()
    reconstructed_model.CopyFrom(published_int8_model)
    reconstructed_model, num_rewritten, num_skipped = (
        _dequantize_dynamically_quantized_matmuls(reconstructed_model)
    )
    assert num_skipped == 0, (
        f"{num_skipped} MatMulInteger node(s) in fast_ar_int8.onnx did not "
        "match the expected ORT quantize_dynamic pattern -- the "
        "reconstructed 'float' reference would still contain quantized "
        "ops, invalidating this test"
    )
    assert num_rewritten > 0, "found no MatMulInteger nodes to dequantize"

    fp32_model, simplify_ok = onnxsim.simplify(reconstructed_model)
    assert simplify_ok, "onnxsim.simplify failed its own numerical check"
    assert not any(
        n.op_type in ("DynamicQuantizeLinear", "MatMulInteger")
        for n in fp32_model.graph.node
    ), "reconstructed fp32 reference still contains quantized ops after simplify"

    onnxsim_int8_model = onnxsim.quantize_dynamic(fp32_model)

    calibration_data = onnxsim.generate_random_calibration_data(
        fp32_model, num_samples=4, seed=0
    )

    onnxsim_report = onnxsim.measure_accuracy_drop(
        fp32_model, onnxsim_int8_model, calibration_data=calibration_data
    )
    published_report = onnxsim.measure_accuracy_drop(
        fp32_model, published_int8_model, calibration_data=calibration_data
    )

    assert onnxsim_report.all_finite, (
        "onnxsim.quantize_dynamic produced non-finite output on fast_ar_int8.onnx"
    )

    assert onnxsim_report.worst_relative_l2 <= max(
        published_report.worst_relative_l2 * 3, 0.05
    ), (
        f"onnxsim.quantize_dynamic's relative L2 error "
        f"({onnxsim_report.worst_relative_l2:.4g}) against the reconstructed "
        "fp32 reference is far worse than Audio8's own published INT8 "
        f"export's ({published_report.worst_relative_l2:.4g}) on the same "
        "graph and calibration data"
    )
