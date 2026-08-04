"""Tests for emitting ONNX-dialect MLIR via the onnx-mlir compiler.

Unlike torch-mlir (a pure-Python importer, covered by ``test_mlir_export.py``),
onnx-mlir emits MLIR by shelling out to its compiler binary. That binary is not
pip-installable and not part of onnxsim's test requirements, so:

* The backend-independent checks (target is registered, capability probe, error
  when the binary is missing) run everywhere -- they need only onnxsim + onnx.
* The shell-out plumbing (argument construction, reading back the ``.onnx.mlir``
  file) is covered with a tiny **fake** ``onnx-mlir`` script, so it runs in the
  normal test matrix without a real onnx-mlir. Skipped on Windows, where the
  fake shebang script is not executable.
* The real end-to-end conversion runs only when an actual onnx-mlir binary can
  be located (``ONNX_MLIR`` / ``ONNX_MLIR_HOME`` / PATH).
"""

import os
import stat
import sys
import textwrap

import onnx
import pytest
from onnx import TensorProto, helper

import onnxsim
from onnxsim import mlir_export


def _relu_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])
    node = helper.make_node("Relu", ["x"], ["y"])
    graph = helper.make_graph([node], "relu", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    return model


def _write_fake_onnx_mlir(path) -> str:
    """Write an executable stand-in for onnx-mlir and return its path.

    It parses ``-o <base>`` exactly like the real binary and writes
    ``<base>.onnx.mlir`` so the read-back path in ``convert_to_onnx_mlir`` is
    exercised end to end.
    """
    script = textwrap.dedent(
        f"""\
        #!{sys.executable}
        import sys
        out = None
        args = sys.argv[1:]
        i = 0
        while i < len(args):
            if args[i] == "-o":
                out = args[i + 1]
                i += 2
            else:
                i += 1
        with open(out + ".onnx.mlir", "w") as f:
            f.write('module {{\\n  func.func @main_graph() {{\\n'
                    '    onnx.Return\\n  }}\\n}}\\n')
        """
    )
    path.write_text(script)
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IRUSR)
    return str(path)


def test_onnx_is_a_supported_target():
    assert "onnx" in mlir_export.SUPPORTED_TARGETS


def test_has_onnx_mlir_returns_bool():
    assert isinstance(mlir_export.has_onnx_mlir(), bool)


@pytest.mark.skipif(
    mlir_export.has_onnx_mlir(),
    reason="onnx-mlir is available; the missing-binary path cannot be exercised",
)
def test_missing_binary_raises_runtime_error():
    with pytest.raises(RuntimeError, match="onnx-mlir is required"):
        onnxsim.export_mlir(_relu_model(), target="onnx")
    # An explicit, non-existent path must fail the same way.
    with pytest.raises(RuntimeError, match="onnx-mlir is required"):
        onnxsim.export_mlir(
            _relu_model(), target="onnx", onnx_mlir="/no/such/onnx-mlir"
        )


@pytest.mark.skipif(os.name == "nt", reason="fake shebang script is not executable")
def test_shell_out_with_fake_binary(tmp_path):
    fake = _write_fake_onnx_mlir(tmp_path / "onnx-mlir")
    assert mlir_export.has_onnx_mlir(fake) is True

    text = onnxsim.export_mlir(_relu_model(), target="onnx", onnx_mlir=fake)
    assert "func.func" in text
    assert "onnx.Return" in text

    # Writing to a file returns the same text it wrote.
    out = tmp_path / "model.mlir"
    written = onnxsim.export_mlir(
        _relu_model(), str(out), target="onnx", onnx_mlir=fake
    )
    assert out.read_text() == written


@pytest.mark.skipif(os.name == "nt", reason="fake shebang script is not executable")
def test_failed_compilation_raises_with_stderr(tmp_path):
    failing = tmp_path / "onnx-mlir"
    failing.write_text(
        f"#!{sys.executable}\nimport sys\nsys.stderr.write('boom\\n')\nsys.exit(2)\n"
    )
    failing.chmod(failing.stat().st_mode | stat.S_IEXEC | stat.S_IRUSR)
    with pytest.raises(RuntimeError, match="onnx-mlir failed to emit MLIR"):
        onnxsim.export_mlir(_relu_model(), target="onnx", onnx_mlir=str(failing))


@pytest.mark.skipif(
    not mlir_export.has_onnx_mlir(),
    reason="a real onnx-mlir binary is required",
)
def test_end_to_end_with_real_onnx_mlir():
    model = _relu_model()
    simplified, ok = onnxsim.simplify(model)
    assert ok
    text = onnxsim.export_mlir(simplified, target="onnx")
    # onnx-mlir wraps the graph in a func and prefixes ops with the onnx dialect.
    assert "func.func" in text
    assert "onnx." in text
