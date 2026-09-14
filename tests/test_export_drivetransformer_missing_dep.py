# onnxsim.export_drivetransformer_model's optional-dependency error message,
# checked with 'mmcv' forced unimportable regardless of whether it is
# actually installed in this environment -- unlike a real
# test_export_drivetransformer.py (which would need torch plus a real
# DriveTransformer checkout to exercise the export itself, and so isn't
# included here: DriveTransformer is not pip-installable and needs its own
# bundled mmcv/mmdet/mmdet3d fork, not the real PyPI packages of those
# names, see onnxsim.drivetransformer_export's module docstring), this
# test's whole point is exercising the *absence* path, so it must not be
# skipped just because 'mmcv' (the real PyPI package, unrelated to
# DriveTransformer's own fork) happens to be present.

import builtins

import pytest

import onnxsim


def test_export_drivetransformer_model_needs_mmcv(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "mmcv" or name.startswith("mmcv."):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match=r"onnxsim\[drivetransformer\]"):
        onnxsim.export_drivetransformer_model(
            "some-config.py", "/tmp/does-not-matter.onnx"
        )
