# Integration/regression test: onnxsim against a real Hugging Face Optimum
# ONNX export, run through the actual deployment flow (multi-file
# encoder/decoder-with-past autoregressive generation via
# ``optimum.onnxruntime.ORTModelForSeq2SeqLM``), not just a single
# ``session.run()`` call.
#
# ``optimum-onnx``'s ``main_export`` is the standard way encoder-decoder
# models (T5, Whisper, BART, ...) get exported to ONNX for deployment: it
# produces a *directory* of files -- ``encoder_model.onnx``,
# ``decoder_model.onnx`` (no KV-cache inputs, used for the first decode
# step), ``decoder_with_past_model.onnx`` (takes/returns KV-cache tensors,
# used for every step after) -- plus the tokenizer/config files, all
# consumed together by ``ORTModelForSeq2SeqLM.from_pretrained``. This is the
# split-model shape described in this repo's own notes on why autoregressive
# audio/LLM exports come out as several ``.onnx`` files instead of one (see
# ``tests/test_audio8.py``'s docstring for the KV-cache-heavy real-world
# case). This test exercises ``onnxsim.export_transformers_model()`` -- the
# reusable wrapper around exactly this export-then-simplify-in-place recipe,
# see ``onnxsim/transformers_export.py`` -- over that multi-file shape, then
# re-runs the *whole* generate() loop -- encoder once, decoder-with-past
# repeatedly, KV-cache fed back each step -- against the simplified files and
# checks the generated token ids are unchanged. ``exported_dir`` below is a
# separate, unsimplified export used only as the "before" baseline (node
# counts and generation output to compare against): export_transformers_model
# always simplifies, so it cannot itself produce that baseline.
#
# Exported with ``no_post_process=True``: by default, recent
# ``optimum-onnx`` merges ``decoder_model``/``decoder_with_past_model`` into
# a single ``decoder_model_merged.onnx`` with a top-level ``If`` node
# switching between the two branches. As of this writing, simplifying that
# merged file (onnxsim does descend into ``If`` branch subgraphs by default)
# produces a model that fails at runtime with an ONNX Runtime broadcast
# error inside the cross-attention block -- see the follow-up note left in
# git history/PR discussion for this test file. ``no_post_process=True``
# keeps the plain (no control-flow) three-file split, which simplifies and
# reloads correctly; ``ORTModelForSeq2SeqLM`` falls back to it automatically
# when no merged file is present.
#
# The model is ``hf-internal-testing/tiny-random-t5`` (random weights, tiny
# dims) -- small/fast to download and export, but it is a real encoder-decoder
# architecture exported by real Optimum code, not a hand-built stand-in, so
# this exercises the actual export + deploy code path rather than a model
# shaped like it.
#
# ``torch``, ``transformers``, and ``optimum`` (with the ``optimum-onnx``
# distribution installed for ``optimum.exporters.onnx``/
# ``optimum.onnxruntime``) are NOT normal test dependencies -- they are heavy
# and, as of writing, ``optimum-onnx`` is a very new split-out of what used to
# be ``optimum``'s own ``onnxruntime`` extra, so pinning it into the regular
# matrix is left for a follow-up. This test skips unless they're already
# importable, and skips (rather than fails) on a network error downloading
# the tiny model, matching ``tests/test_audio8.py``'s convention for
# network-dependent tests. To run it locally::
#
#     pip install torch transformers "optimum[exporters]" optimum-onnx onnxruntime
#     pip install --force-reinstall --no-deps .   # the onnxsim under test
#     pytest tests/test_optimum_export_deploy.py -v

import glob
import os

import onnx
import pytest

import onnxsim

pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("optimum.exporters.onnx")
pytest.importorskip("optimum.onnxruntime")
onnxruntime = pytest.importorskip("onnxruntime")

from optimum.exporters.onnx import main_export  # noqa: E402
from optimum.onnxruntime import ORTModelForSeq2SeqLM  # noqa: E402

_MODEL_ID = "hf-internal-testing/tiny-random-t5"
_PROMPT = "translate English to German: Hello world"


@pytest.fixture(scope="module")
def exported_dir(tmp_path_factory):
    out_dir = str(tmp_path_factory.mktemp("optimum_export"))
    try:
        main_export(
            _MODEL_ID,
            output=out_dir,
            task="text2text-generation-with-past",
            no_post_process=True,
        )
    except Exception as e:  # network/hub errors surface as a variety of types
        pytest.skip(f"Could not export {_MODEL_ID} from Hugging Face Hub: {e}")
    return out_dir


def test_optimum_export_produces_split_files(exported_dir):
    # The shape this test exists to guard: three plain (no control-flow)
    # graphs, not one, for exactly the encoder-once /
    # decode-step-run-in-a-host-loop reasons discussed for autoregressive
    # models generally.
    names = {
        os.path.basename(f) for f in glob.glob(os.path.join(exported_dir, "*.onnx"))
    }
    assert names == {
        "encoder_model.onnx",
        "decoder_model.onnx",
        "decoder_with_past_model.onnx",
    }


def test_optimum_export_simplify_and_deploy(exported_dir, tmp_path):
    tokenizer = transformers.AutoTokenizer.from_pretrained(exported_dir)
    inputs = tokenizer(_PROMPT, return_tensors="pt")

    baseline_model = ORTModelForSeq2SeqLM.from_pretrained(exported_dir)
    baseline_ids = baseline_model.generate(**inputs, max_new_tokens=8, do_sample=False)

    nodes_before = {
        os.path.basename(f): len(onnx.load(f, load_external_data=False).graph.node)
        for f in glob.glob(os.path.join(exported_dir, "*.onnx"))
    }

    # export_transformers_model does its own fresh export (same model/task,
    # so structurally identical to exported_dir's) straight into sim_dir,
    # simplifying every graph in place and writing the tokenizer/config files
    # itself -- no manual glob-and-copy loop needed.
    sim_dir = str(tmp_path / "simplified")
    results = onnxsim.export_transformers_model(
        _MODEL_ID,
        sim_dir,
        task="text2text-generation-with-past",
    )
    assert set(results.keys()) == set(nodes_before.keys())
    assert all(results.values()), f"onnxsim numerical check failed: {results}"

    nodes_after = {
        name: len(
            onnx.load(os.path.join(sim_dir, name), load_external_data=False).graph.node
        )
        for name in nodes_before
    }
    for name in nodes_before:
        assert nodes_after[name] <= nodes_before[name], (
            f"{name}: simplification must not increase node count"
        )

    # Plain Optimum export (no --optimize, no --slim) does no fusion/folding
    # beyond torch.onnx.export's own constant folding -- see this test
    # module's docstring and the "does export do any optimization" question
    # it answers. There should be real simplification left on the table for
    # onnxsim to find.
    assert sum(nodes_after.values()) < sum(nodes_before.values())

    # The actual "deploy flow": encoder runs once, decoder_with_past_model
    # runs once per generated token with KV-cache fed back each step, all
    # driven by ORTModelForSeq2SeqLM.generate() exactly as a real deployment
    # would call it. Simplification must not change the generated sequence.
    sim_model_ort = ORTModelForSeq2SeqLM.from_pretrained(sim_dir)
    sim_ids = sim_model_ort.generate(**inputs, max_new_tokens=8, do_sample=False)
    assert sim_ids.tolist() == baseline_ids.tolist()
