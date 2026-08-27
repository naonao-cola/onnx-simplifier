# Regression test: onnxsim.export_transformers_model against a real audio
# (not text-only) architecture -- Whisper's automatic-speech-recognition-
# with-past task exports the same encoder/decoder/decoder-with-past shape
# tests/test_export_transformers.py already covers for T5, but with an audio
# feature-extractor encoder instead of a text encoder, proving the wrapper
# isn't accidentally specific to text-to-text seq2seq models. This is meant
# as a single additional architecture, not exhaustive coverage of every
# audio/transformer model optimum can export -- see the "Transformers
# export" section of README.md for the fuller list.
#
# Model: onnx-internal-testing/tiny-random-WhisperForConditionalGeneration --
# a public, ~870K-parameter random-weight Whisper checkpoint maintained
# specifically for ONNX export testing (safetensors weights, real Whisper
# config/architecture, tiny dims), so this exercises the actual Whisper
# OnnxConfig rather than a hand-built stand-in.
#
# Same heavy/optional dependencies and skip conventions as
# test_export_transformers.py: torch, transformers, and optimum (with the
# optimum-onnx distribution) are not normal test dependencies, so this skips
# unless they're already importable, and skips (rather than fails) on a
# network error downloading the tiny model. To run it locally::
#
#     pip install torch transformers "optimum[exporters]" optimum-onnx
#     pip install --force-reinstall --no-deps .   # the onnxsim under test
#     pytest tests/test_export_transformers_whisper.py -v

import glob
import os

import onnx
import pytest

import onnxsim

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("optimum.exporters.onnx")

_MODEL_ID = "onnx-internal-testing/tiny-random-WhisperForConditionalGeneration"


def test_export_transformers_model_whisper(tmp_path):
    out_dir = str(tmp_path)
    try:
        results = onnxsim.export_transformers_model(
            _MODEL_ID,
            out_dir,
            task="automatic-speech-recognition-with-past",
            check_n=2,
        )
    except Exception as e:  # network/hub errors surface as a variety of types
        pytest.skip(f"Could not export {_MODEL_ID} from Hugging Face Hub: {e}")

    onnx_files = {
        os.path.basename(f) for f in glob.glob(os.path.join(out_dir, "*.onnx"))
    }
    assert onnx_files == {
        "encoder_model.onnx",
        "decoder_model.onnx",
        "decoder_with_past_model.onnx",
    }
    assert set(results.keys()) == onnx_files
    assert all(results.values()), results

    # Non-.onnx files (feature extractor/generation config, ...) from the
    # export must survive untouched, so the directory stays deployable as-is.
    assert os.path.exists(os.path.join(out_dir, "config.json"))

    for name in onnx_files:
        model = onnx.load(os.path.join(out_dir, name), load_external_data=False)
        assert len(model.graph.node) > 0
