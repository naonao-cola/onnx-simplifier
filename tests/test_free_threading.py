"""Free-threading (no-GIL) regression tests.

These only do something on a free-threaded CPython build (the ``cp*t`` wheels,
e.g. cp314t). On a regular GIL build they are skipped.
"""
import subprocess
import sys
import sysconfig

import pytest

FREE_THREADED = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


@pytest.mark.skipif(
    not FREE_THREADED,
    reason="only meaningful on a free-threaded (Py_GIL_DISABLED) build",
)
def test_extension_keeps_gil_disabled():
    """Importing the compiled extension must not re-enable the GIL.

    The nanobind module is built with ``FREE_THREADED`` (see CMakeLists.txt), so
    it is tagged ``Py_MOD_GIL_NOT_USED`` and a free-threaded interpreter keeps
    the GIL disabled on import. Without that declaration CPython silently
    re-enables the GIL and emits a RuntimeWarning, defeating the free-threaded
    wheel. This guards against that regression.

    Run in a fresh subprocess importing *only* the extension: other test
    dependencies (torch, numpy) imported in-process might not declare
    free-threading support and would re-enable the GIL themselves, masking what
    we are checking here.
    """
    code = (
        "import sys, onnxsim.onnxsim_cpp2py_export\n"
        "assert sys._is_gil_enabled() is False, 'importing onnxsim re-enabled the GIL'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
