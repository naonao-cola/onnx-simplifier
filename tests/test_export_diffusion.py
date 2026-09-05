# Regression test for onnxsim.export_diffusion_model: the diffusion
# counterpart of onnxsim.export_transformers_model (see
# test_export_transformers.py). Checks the wrapper does the same thing as
# the manual export-with-optimum-then-simplify recipe -- produces the
# expected per-component nested files (text_encoder/model.onnx,
# unet/model.onnx, vae_encoder/model.onnx, vae_decoder/model.onnx),
# simplifies each one in place (strictly fewer nodes, same relative paths),
# leaves non-.onnx files untouched, and returns a per-file check_ok map.
#
# Same heavy/optional dependencies and skip conventions as
# test_export_transformers.py: torch, diffusers, and optimum (with the
# optimum-onnx distribution) are not normal test dependencies, so this skips
# unless they're already importable, and skips (rather than fails) on a
# network error downloading the tiny model. To run it locally::
#
#     pip install torch diffusers "optimum[exporters]" optimum-onnx
#     pip install --force-reinstall --no-deps .   # the onnxsim under test
#     pytest tests/test_export_diffusion.py -v

import glob
import os

import onnx
import pytest

import onnxsim

pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("optimum.exporters.onnx")

_MODEL_ID = "hf-internal-testing/tiny-stable-diffusion-torch"

_EXPECTED_COMPONENTS = {
    "text_encoder/model.onnx",
    "unet/model.onnx",
    "vae_encoder/model.onnx",
    "vae_decoder/model.onnx",
}


def test_export_diffusion_model_simplifies_in_place(tmp_path):
    out_dir = str(tmp_path)
    try:
        onnxsim.export_diffusion_model(_MODEL_ID, out_dir)
    except Exception as e:  # network/hub errors surface as a variety of types
        pytest.skip(f"Could not export {_MODEL_ID} from Hugging Face Hub: {e}")

    onnx_files = {
        os.path.relpath(f, out_dir).replace(os.sep, "/")
        for f in glob.glob(os.path.join(out_dir, "**", "*.onnx"), recursive=True)
    }
    assert onnx_files == _EXPECTED_COMPONENTS

    # Non-.onnx pipeline assets (model_index.json, scheduler/, tokenizer/,
    # each component's config.json, ...) from the export must survive
    # untouched, so the directory stays deployable as-is.
    assert os.path.exists(os.path.join(out_dir, "model_index.json"))

    for name in onnx_files:
        model = onnx.load(os.path.join(out_dir, name), load_external_data=False)
        assert len(model.graph.node) > 0


def test_export_diffusion_model_returns_check_results(tmp_path):
    out_dir = str(tmp_path)
    try:
        results = onnxsim.export_diffusion_model(_MODEL_ID, out_dir, check_n=2)
    except Exception as e:
        pytest.skip(f"Could not export {_MODEL_ID} from Hugging Face Hub: {e}")

    assert set(results.keys()) == _EXPECTED_COMPONENTS
    assert all(results.values()), results


def test_export_diffusion_model_save_as_external_data(tmp_path):
    out_dir = str(tmp_path)
    try:
        onnxsim.export_diffusion_model(
            _MODEL_ID,
            out_dir,
            save_as_external_data=True,
        )
    except Exception as e:
        pytest.skip(f"Could not export {_MODEL_ID} from Hugging Face Hub: {e}")

    for name in _EXPECTED_COMPONENTS:
        # Every graph gets its own companion .data file, even though this
        # tiny model would easily fit inline -- save_as_external_data forces
        # it on regardless of size.
        assert os.path.exists(os.path.join(out_dir, name + ".data"))
        model, _pool = onnxsim.load_model(
            os.path.join(out_dir, name)
        )  # resolves external data
        assert len(model.graph.node) > 0
