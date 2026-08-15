# Integration/regression test: onnxsim against VOICEVOX
# (https://github.com/VOICEVOX/voicevox_core), a widely-used Japanese TTS
# engine whose inference pipeline is entirely ONNX-based (see
# crates/voicevox_core/src/core/infer/ in voicevox_core -- it runs models
# through the `ort` crate). This test guards against onnxsim regressions on
# VOICEVOX-style graphs by simplifying the real, real-weight ONNX models
# shipped as voicevox_core's ``model/sample.vvm`` dev fixture.
#
# ``sample.vvm`` -- not the production ``.vvm`` releases -- is used
# deliberately: production voices (github.com/VOICEVOX/voicevox_vvm) ship
# their model weights in a proprietary obfuscated container ("vv-bin", only
# loadable by VOICEVOX's own onnxruntime fork) specifically so they cannot be
# extracted; voicevox_core's TERMS.txt for that data forbids reverse
# engineering. ``sample.vvm`` is voicevox_core's own dev/build fixture and
# ships its models as plain, directly loadable ``.onnx`` files instead.
#
# The 9 models between them cover VOICEVOX's whole pipeline (see the module
# docstring intro above): phoneme-duration and pitch predictors, the
# talk/singing decoders and vocoders, and the singing-teacher models -- which
# is also what surfaced two real onnxsim bugs this test now guards against:
#
#   * predict_sing_f0.onnx / predict_sing_volume.onnx use RandomNormalLike
#     (flow-matching / diffusion-style sampling). onnxsim's check_n
#     correctness check now pins Random* ops to a fixed seed internally so
#     the comparison is meaningful (see onnxsim/model_checking.py) --
#     without that, every check_n trial reported a false failure regardless
#     of whether simplification changed anything.
#   * predict_sing_f0.onnx / sf_decode.onnx have deep, highly symbolic
#     MACs/FLOPs formulas (1000+ nodes, data-dependent output shapes) that
#     used to crash the CLI's summary table with a sympy RecursionError (see
#     onnxsim/model_info.py's ``_factor_or_str``).
#
# Downloads real model files from voicevox_core over the network and, for the
# larger decoder/vocoder models, takes several minutes per model under
# check_n=3 -- too slow/network-dependent for every PR. It is opt-in via
# ONNXSIM_RUN_VOICEVOX_TESTS=1, set by the dedicated weekly
# .github/workflows/voicevox.yml. To run it locally::
#
#     pip install onnxruntime
#     pip install --force-reinstall --no-deps .   # the onnxsim under test
#     ONNXSIM_RUN_VOICEVOX_TESTS=1 pytest tests/test_voicevox.py -v

import os
import time
import urllib.error
import urllib.request

import onnx
import pytest

import onnxsim

# A bare ``import onnxruntime`` would fail collection (not skip the test) on
# platforms onnxruntime doesn't ship wheels for (e.g. s390x), before the
# ONNXSIM_RUN_VOICEVOX_TESTS skipif below even gets a chance to apply.
onnxruntime = pytest.importorskip("onnxruntime")

pytestmark = pytest.mark.skipif(
    os.environ.get("ONNXSIM_RUN_VOICEVOX_TESTS") != "1",
    reason="Set ONNXSIM_RUN_VOICEVOX_TESTS=1 to run (downloads real models "
    "from the network; the decoder/vocoder cases take minutes each).",
)

# Pinned to a voicevox_core commit for reproducibility. Bump deliberately
# (e.g. if the sample models are ever regenerated) rather than tracking a
# branch, so a spurious upstream change can't turn this test flaky.
_VOICEVOX_CORE_COMMIT = "8f4e299c10b181cc5058d32b1a492999b5c2b2c4"
_RAW_BASE = (
    "https://raw.githubusercontent.com/VOICEVOX/voicevox_core/"
    f"{_VOICEVOX_CORE_COMMIT}/model/sample.vvm"
)

# filename -> (overwrite_input_shapes, input_fill). Dims left out rely on
# onnxsim's "dynamic dim[0] -> assume batch size 1" default; the singing
# models' dynamic dim sits at position 1, so those need every input spelled
# out. predict_intonation.onnx needs input_fill="ones": its accent/boundary
# lists are 0/1 flags, and onnxsim's default random-fill casts uniform [0, 1)
# to int64 (i.e. always 0), which makes an internal split-by-boundary op see
# no boundaries and produce a degenerate empty sequence.
CASES = {
    "predict_duration.onnx": ({}, "random"),
    "predict_intonation.onnx": ({}, "ones"),
    "decode.onnx": (
        {"f0": [5, 1], "phoneme": [5, 45], "speaker_id": [1]},
        "random",
    ),
    "predict_spectrogram.onnx": (
        {"f0": [5, 1], "phoneme": [5, 45], "speaker_id": [1]},
        "random",
    ),
    "vocoder.onnx": ({"spec": [5, 80]}, "random"),
    "predict_sing_consonant_length.onnx": (
        {
            "consonants": [1, 5],
            "vowels": [1, 5],
            "note_durations": [1, 5],
            "speaker_id": [1],
        },
        "random",
    ),
    "predict_sing_f0.onnx": (
        {"phonemes": [1, 5], "notes": [1, 5], "speaker_id": [1]},
        "random",
    ),
    "predict_sing_volume.onnx": (
        {
            "phonemes": [1, 5],
            "notes": [1, 5],
            "frame_f0s": [1, 5],
            "speaker_id": [1],
        },
        "random",
    ),
    "sf_decode.onnx": (
        {
            "frame_phonemes": [1, 5],
            "frame_f0s": [1, 5],
            "frame_volumes": [1, 5],
            "speaker_id": [1],
        },
        "random",
    ),
}


@pytest.fixture(scope="module")
def voicevox_model_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("voicevox_sample_vvm")


_DOWNLOAD_ATTEMPTS = 5


def _download(filename: str, dest_dir) -> str:
    dest = str(dest_dir / filename)
    if os.path.exists(dest):
        return dest
    url = f"{_RAW_BASE}/{filename}"
    tmp_dest = f"{dest}.part"
    last_error: Exception = RuntimeError("unreachable")
    for attempt in range(_DOWNLOAD_ATTEMPTS):
        try:
            # Some of these files are 50+MB; a plain GET occasionally comes back
            # truncated (observed as urllib's "retrieval incomplete" on a flaky
            # connection). Download to a temp name and only rename into place on
            # a fully successful transfer, so a partial file is never mistaken
            # for a complete one on retry or by a later test run.
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
        f"Could not download {filename} from voicevox_core after "
        f"{_DOWNLOAD_ATTEMPTS} attempts: {last_error}"
    )


@pytest.mark.parametrize("filename", sorted(CASES.keys()))
def test_voicevox_model_simplify(voicevox_model_dir, filename):
    overwrite_input_shapes, input_fill = CASES[filename]
    path = _download(filename, voicevox_model_dir)

    nodes_before = len(onnx.load(path, load_external_data=False).graph.node)

    model_opt, check_ok = onnxsim.simplify(
        path,
        check_n=3,
        overwrite_input_shapes=overwrite_input_shapes or None,
        input_fill=input_fill,
    )
    assert check_ok, f"{filename}: onnxsim numerical check failed"
    assert len(model_opt.graph.node) <= nodes_before, (
        f"{filename}: simplification must not increase node count"
    )

    # The simplified model must still load in onnxruntime.
    onnxruntime.InferenceSession(model_opt.SerializeToString())
