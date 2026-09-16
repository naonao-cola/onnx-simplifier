"""Real TIDL model-import/compile check -- an actual compiler run, not a
heuristic.

Unlike `tests/test_edgeai_tidl_compat.py`, this module is backed by TI's
own `onnxruntime_tidl` (`TIDLCompilationProvider`) and `tidl_tools`
binaries, run in x86 "PC emulation"/compile-only mode -- see
`scripts/edgeai/real_compile.py`'s docstring for exactly what that does and
does not confirm (the compile/import stage is real; there is no on-device
inference here).

Skipped unless both `TIDL_PYTHON` (a Python 3.10 venv with a working
`onnxruntime_tidl`) and `TIDL_TOOLS_PATH` (an extracted `tidl_tools`
directory) are set and actually work -- see `scripts/edgeai/README.md`'s
"Running a real compile" section for how to set them up. The regular
`edgeai-integration.yml` CI does not set these by default for every PR (see
that workflow for which job does).
"""

import os
import sys

import pytest

_EDGEAI_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "edgeai"
)
_AXERA_DIR = os.path.join(os.path.dirname(_EDGEAI_DIR), "axera")
for _dir in (_EDGEAI_DIR, _AXERA_DIR):
    if _dir not in sys.path:
        sys.path.insert(0, _dir)

import real_compile as rc  # noqa: E402
from _local_import import fresh  # noqa: E402

models = fresh("models", _EDGEAI_DIR)

_TIDL_PYTHON = rc.find_tidl_python()
_TIDL_TOOLS_PATH = rc.tidl_tools_path()

pytestmark = pytest.mark.skipif(
    _TIDL_PYTHON is None or _TIDL_TOOLS_PATH is None,
    reason=(
        "needs TIDL_PYTHON (a Python 3.10 venv with a working "
        "onnxruntime_tidl) and TIDL_TOOLS_PATH (an extracted tidl_tools "
        "directory) -- see scripts/edgeai/README.md"
    ),
)


def _compile(model, tmp_path, name):
    import onnx

    model = onnx.shape_inference.infer_shapes(model)
    model_path = str(tmp_path / f"{name}.onnx")
    onnx.save(model, model_path)
    artifacts_dir = str(tmp_path / f"{name}_artifacts")
    return rc.compile_offload_summary(
        _TIDL_PYTHON, model_path, _TIDL_TOOLS_PATH, artifacts_dir
    )


@pytest.mark.parametrize(
    "name", ["conv_bn_relu", "foldable_shape_reshape", "matmul_bias_tanh"]
)
def test_simplify_does_not_regress_real_tidl_offload(name, tmp_path):
    """The concrete claim `test_edgeai_tidl_compat.py`'s static heuristic
    can only approximate: onnxsim's simplification must not reduce how much
    of the graph the *real* TIDL compiler offloads to C7x, on an actual
    compile run.
    """
    from onnxsim import simplify

    model = models.build(name)
    simplified, check_ok = simplify(model)
    assert check_ok

    orig = _compile(model, tmp_path, f"{name}_orig")
    simp = _compile(simplified, tmp_path, f"{name}_simplified")

    assert orig["returncode"] == 0, orig["stdout"][-2000:]
    assert simp["returncode"] == 0, simp["stdout"][-2000:]
    assert simp["c7x_nodes"] >= orig["c7x_nodes"], (orig, simp)
    assert simp["cpu_nodes"] <= orig["cpu_nodes"], (orig, simp)
