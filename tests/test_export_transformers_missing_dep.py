# onnxsim.export_transformers_model's optional-dependency error message,
# checked with 'optimum' forced unimportable regardless of whether it is
# actually installed in this environment -- unlike test_export_transformers.py
# (which needs torch/transformers/optimum for real to exercise the export
# itself), this test's whole point is exercising the *absence* path, so it
# must not be skipped just because those heavy packages happen to be present.

import builtins

import pytest

import onnxsim


def test_export_transformers_model_needs_optimum(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("optimum"):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match=r"onnxsim\[transformers\]"):
        onnxsim.export_transformers_model("some-model", "/tmp/does-not-matter")
