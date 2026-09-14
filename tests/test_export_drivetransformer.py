# Regression test for onnxsim.export_drivetransformer_model: the
# DriveTransformer counterpart of onnxsim.export_detectron_model /
# onnxsim.export_sam2_model. Checks the wrapper produces a valid, strictly-
# simplified ONNX file and returns the check_ok result, using a randomly-
# initialized model (no checkpoint=, so no network call) to keep this
# runnable offline.
#
# torch and a real DriveTransformer checkout are not normal test
# dependencies -- DriveTransformer isn't pip-installable, and its own
# "mmcv"/"mmdet"/"mmdet3d" are its own bundled, merged fork, unrelated to
# the real PyPI 'mmcv' package (see onnxsim.drivetransformer_export's module
# docstring) -- so this test is only meaningful from inside an actual
# DriveTransformer checkout, and skips everywhere else: on the real PyPI
# 'mmcv' (which has no 'mmcv.models', so the importorskip below still
# skips), and unless DRIVETRANSFORMER_CONFIG points at a real
# 'drivetransformer_*.py' mmdet3d_plugin config from such a checkout. To run
# it locally, from inside a DriveTransformer checkout::
#
#     pip install torch && pip install -v -e .   # DriveTransformer itself
#     pip install --force-reinstall --no-deps /path/to/onnxsim  # the onnxsim under test
#     DRIVETRANSFORMER_CONFIG=adzoo/drivetransformer/configs/drivetransformer/drivetransformer_large.py \
#         pytest /path/to/onnxsim/tests/test_export_drivetransformer.py -v

import os

import onnx
import pytest

import onnxsim

pytest.importorskip("torch")
pytest.importorskip("mmcv.models")

_CONFIG = os.environ.get("DRIVETRANSFORMER_CONFIG")
if not _CONFIG:
    pytest.skip(
        "set DRIVETRANSFORMER_CONFIG to a DriveTransformer mmdet3d_plugin "
        "config path to run this test",
        allow_module_level=True,
    )


def test_export_drivetransformer_model_simplifies(tmp_path):
    out_path = str(tmp_path / "model.onnx")

    check_ok = onnxsim.export_drivetransformer_model(_CONFIG, out_path)

    assert check_ok is True
    model = onnx.load(out_path, load_external_data=False)
    assert len(model.graph.node) > 0


def test_export_drivetransformer_model_returns_check_results(tmp_path):
    out_path = str(tmp_path / "model.onnx")

    check_ok = onnxsim.export_drivetransformer_model(_CONFIG, out_path, check_n=2)

    assert check_ok is True


def test_export_drivetransformer_model_save_as_external_data(tmp_path):
    out_path = str(tmp_path / "model.onnx")

    onnxsim.export_drivetransformer_model(_CONFIG, out_path, save_as_external_data=True)

    assert os.path.exists(out_path + ".data")
    model, _pool = onnxsim.load_model(out_path)
    assert len(model.graph.node) > 0
