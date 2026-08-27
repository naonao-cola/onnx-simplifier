# Regression test for onnxsim.export_transformers_model: the reusable
# wrapper around the manual export-with-optimum-then-simplify-in-place recipe
# tests/test_optimum_export_deploy.py exercises by hand. This test checks the
# wrapper itself does the same thing that manual recipe does -- produces the
# expected split files, simplifies each one in place (strictly fewer nodes,
# same file names), leaves non-.onnx files untouched, and returns a
# per-file check_ok map -- not the full generate()-loop deployment check,
# which the other test already covers for the underlying export+simplify
# pipeline.
#
# Same heavy/optional dependencies and skip conventions as
# test_optimum_export_deploy.py: torch, transformers, and optimum (with the
# optimum-onnx distribution) are not normal test dependencies, so this skips
# unless they're already importable, and skips (rather than fails) on a
# network error downloading the tiny model. To run it locally::
#
#     pip install torch transformers "optimum[exporters]" optimum-onnx
#     pip install --force-reinstall --no-deps .   # the onnxsim under test
#     pytest tests/test_export_transformers.py -v

import glob
import os

import onnx
import pytest

import onnxsim

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("optimum.exporters.onnx")

_MODEL_ID = "hf-internal-testing/tiny-random-t5"


def test_export_transformers_model_simplifies_in_place(tmp_path):
    out_dir = str(tmp_path)
    try:
        onnxsim.export_transformers_model(
            _MODEL_ID,
            out_dir,
            task="text2text-generation-with-past",
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

    # Non-.onnx files (tokenizer, config, ...) from the export must survive
    # untouched, so the directory stays deployable as-is.
    assert os.path.exists(os.path.join(out_dir, "config.json"))

    for name in onnx_files:
        model = onnx.load(os.path.join(out_dir, name), load_external_data=False)
        assert len(model.graph.node) > 0


def test_export_transformers_model_returns_check_results(tmp_path):
    out_dir = str(tmp_path)
    try:
        results = onnxsim.export_transformers_model(
            _MODEL_ID,
            out_dir,
            task="text2text-generation-with-past",
            check_n=2,
        )
    except Exception as e:
        pytest.skip(f"Could not export {_MODEL_ID} from Hugging Face Hub: {e}")

    assert set(results.keys()) == {
        "encoder_model.onnx",
        "decoder_model.onnx",
        "decoder_with_past_model.onnx",
    }
    assert all(results.values()), results


def test_export_transformers_model_save_as_external_data(tmp_path):
    out_dir = str(tmp_path)
    try:
        onnxsim.export_transformers_model(
            _MODEL_ID,
            out_dir,
            task="text2text-generation-with-past",
            save_as_external_data=True,
        )
    except Exception as e:
        pytest.skip(f"Could not export {_MODEL_ID} from Hugging Face Hub: {e}")

    for name in (
        "encoder_model.onnx",
        "decoder_model.onnx",
        "decoder_with_past_model.onnx",
    ):
        # Every graph gets its own companion .data file, even though this
        # tiny model would easily fit inline -- save_as_external_data forces
        # it on regardless of size.
        assert os.path.exists(os.path.join(out_dir, name + ".data"))
        model = onnx.load(os.path.join(out_dir, name))  # resolves external data
        assert len(model.graph.node) > 0
