"""Emit a (simplified) ONNX model as MLIR.

onnxsim's job stops at a cleaned-up ``onnx.ModelProto``. Compiler stacks built on
MLIR want that graph as MLIR instead. This module bridges the two, with two
backends selected by ``target``:

* ``"torch"`` -- **Torch-dialect** MLIR via `torch-mlir
  <https://github.com/llvm/torch-mlir>`_'s pure-Python ONNX importer
  (``torch_mlir.extras.onnx_importer``), the same importer that backs torch-mlir's
  ``torch-mlir-import-onnx`` / IREE's ``iree-import-onnx`` tools.
* ``"onnx"`` -- **ONNX-dialect** MLIR via `onnx-mlir
  <https://github.com/onnx/onnx-mlir>`_. onnx-mlir has no in-process Python
  importer, so this backend shells out to the ``onnx-mlir`` compiler binary
  (``--EmitONNXIR``) and reads back the ``.onnx.mlir`` it writes.

Feeding a *simplified* model to either backend is the point: constant folding and
the optimizer passes collapse the shape-manipulation subgraphs the importers would
otherwise translate op by op, so the emitted MLIR is smaller and closer to what
the downstream compiler actually needs.

Both backends are **optional**, mirroring how onnxruntime is optional for constant
folding (see ``backend.py``): nothing here is imported/located at
``import onnxsim`` time, only when ``--emit-mlir`` / the ``export_mlir`` API run.
A missing backend raises a ``RuntimeError`` with an install hint.
"""

import os
import shutil
import subprocess
import tempfile
from typing import Optional, Sequence

import onnx

# MLIR targets this module can emit. Kept as named constants so callers can
# compare against them and a future backend has an obvious place to slot in.
TORCH_TARGET = "torch"
ONNX_TARGET = "onnx"
SUPPORTED_TARGETS = (TORCH_TARGET, ONNX_TARGET)

# onnx-mlir emit flag used by default. ``--EmitONNXIR`` ingests ONNX and emits the
# corresponding ONNX dialect (with shape inference / decomposition), which is the
# useful "ONNX dialect" representation; ``--EmitONNXBasic`` is the raw import.
ONNX_MLIR_DEFAULT_EMIT = "EmitONNXIR"

_TORCH_MLIR_INSTALL_HINT = (
    "torch-mlir is required to emit Torch-dialect MLIR but is not installed. "
    "Install it with `pip install torch-mlir` (prebuilt wheels are listed at "
    "https://github.com/llvm/torch-mlir)."
)

_ONNX_MLIR_INSTALL_HINT = (
    "onnx-mlir is required to emit ONNX-dialect MLIR but its compiler binary "
    "could not be found. Build or install onnx-mlir "
    "(https://github.com/onnx/onnx-mlir), then either put `onnx-mlir` on your "
    "PATH, set the ONNX_MLIR_HOME environment variable to its install prefix "
    "(the binary is expected at $ONNX_MLIR_HOME/bin/onnx-mlir), set ONNX_MLIR to "
    "the binary path, or pass the path explicitly via the `onnx_mlir` argument "
    "(CLI: --onnx-mlir)."
)


def has_torch_mlir() -> bool:
    """Whether torch-mlir's ONNX importer is importable in this environment."""
    try:
        from torch_mlir.extras import onnx_importer  # noqa: F401
    except ImportError:
        return False
    return True


def _find_onnx_mlir(onnx_mlir: Optional[str] = None) -> Optional[str]:
    """Locate the ``onnx-mlir`` compiler binary, or return ``None``.

    Resolution order: an explicit path/name, then ``$ONNX_MLIR`` (full binary
    path), then ``$ONNX_MLIR_HOME/bin/onnx-mlir``, then ``PATH``.
    """

    def _usable(path: Optional[str]) -> Optional[str]:
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path
        return None

    if onnx_mlir:
        # Accept either a direct path or a name to look up on PATH.
        return _usable(onnx_mlir) or shutil.which(onnx_mlir)

    from_env = _usable(os.environ.get("ONNX_MLIR"))
    if from_env:
        return from_env

    home = os.environ.get("ONNX_MLIR_HOME")
    if home:
        from_home = _usable(os.path.join(home, "bin", "onnx-mlir"))
        if from_home:
            return from_home

    return shutil.which("onnx-mlir")


def has_onnx_mlir(onnx_mlir: Optional[str] = None) -> bool:
    """Whether the onnx-mlir compiler binary can be located (see ``_find_onnx_mlir``)."""
    return _find_onnx_mlir(onnx_mlir) is not None


def _maybe_convert_opset(
    model: onnx.ModelProto, opset_version: Optional[int]
) -> onnx.ModelProto:
    """Return ``model`` upgraded/downgraded to ``opset_version`` (or unchanged)."""
    if opset_version is None:
        return model
    from onnx import version_converter

    return version_converter.convert_version(model, opset_version)


def convert_to_torch_mlir(
    model: onnx.ModelProto,
    *,
    opset_version: Optional[int] = None,
    run_shape_inference: bool = True,
    data_prop: bool = True,
    verify: bool = True,
) -> str:
    """Convert an ONNX model to Torch-dialect MLIR and return it as text.

    Parameters
    ----------
    model:
        The ONNX model to import. Typically the output of :func:`onnxsim.simplify`.
        The proto is not mutated; opset conversion / shape inference operate on
        copies when they run.
    opset_version:
        If given, upgrade (or downgrade) the model to this ONNX opset with
        ``onnx.version_converter`` before importing. torch-mlir's op coverage
        targets recent opsets, so bumping an old model can unlock more of the
        importer. ``None`` (the default) imports the model at its current opset.
    run_shape_inference:
        Run ``onnx.shape_inference.infer_shapes`` before importing. A simplified
        model is normally already shape-inferred, but the importer produces
        better-typed MLIR when value shapes are present, so this is on by default
        and treated as best-effort (failures are ignored rather than fatal).
    data_prop:
        Enable ONNX data propagation during that shape-inference pass, which
        recovers some shapes that plain inference misses.
    verify:
        Verify the produced MLIR module before returning. Turning this off lets
        you inspect structurally-invalid output for debugging.

    Returns
    -------
    str
        The module's MLIR assembly.

    Raises
    ------
    RuntimeError
        If torch-mlir is not installed.
    """
    try:
        from torch_mlir.dialects import torch as torch_d
        from torch_mlir.extras import onnx_importer
        from torch_mlir.ir import Context
    except ImportError as exc:
        raise RuntimeError(_TORCH_MLIR_INSTALL_HINT) from exc

    model_proto = _maybe_convert_opset(model, opset_version)

    if run_shape_inference:
        try:
            model_proto = onnx.shape_inference.infer_shapes(
                model_proto, data_prop=data_prop
            )
        except Exception:
            # Best-effort only: the importer can still run on a model whose
            # shapes onnxsim already inferred, so a re-inference hiccup (e.g. a
            # custom op ONNX cannot type) must not block MLIR emission.
            pass

    # Mirror torch-mlir's own ``torch-mlir-import-onnx`` tool: create a context
    # with the Torch dialect registered, build the module skeleton, then let the
    # NodeImporter walk the main graph.
    config = onnx_importer.Config()
    context = Context()
    torch_d.register_dialect(context)
    model_info = onnx_importer.ModelInfo(model_proto, config=config)
    module = model_info.create_module(context=context)
    module_op = module.operation
    importer = onnx_importer.NodeImporter.define_function(
        model_info.main_graph, module_op
    )
    importer.import_all()
    if verify:
        module_op.verify()
    return module_op.get_asm(assume_verified=verify)


def _read_onnx_mlir_output(out_base: str, tmpdir: str) -> str:
    """Read the ``.mlir`` file onnx-mlir wrote for output base ``out_base``."""
    # ``--EmitONNXIR``/``--EmitONNXBasic`` write ``<out_base>.onnx.mlir``; other
    # IR-emitting flags may write ``<out_base>.mlir``. Try both, then fall back to
    # any ``.mlir`` produced in the scratch dir.
    for candidate in (out_base + ".onnx.mlir", out_base + ".mlir"):
        if os.path.isfile(candidate):
            with open(candidate, encoding="utf-8") as f:
                return f.read()

    import glob

    matches = sorted(glob.glob(os.path.join(tmpdir, "*.mlir")))
    if matches:
        with open(matches[0], encoding="utf-8") as f:
            return f.read()

    raise RuntimeError(
        "onnx-mlir returned success but produced no .mlir file. The chosen emit "
        "flag may not emit MLIR text (use an --EmitONNX* flag)."
    )


def convert_to_onnx_mlir(
    model: onnx.ModelProto,
    *,
    onnx_mlir: Optional[str] = None,
    emit: str = ONNX_MLIR_DEFAULT_EMIT,
    opset_version: Optional[int] = None,
    extra_args: Optional[Sequence[str]] = None,
    timeout: Optional[float] = None,
) -> str:
    """Convert an ONNX model to ONNX-dialect MLIR via the onnx-mlir compiler.

    onnx-mlir has no in-process Python importer, so this writes ``model`` to a
    temporary ``.onnx`` file, runs the ``onnx-mlir`` binary with an emit flag, and
    reads back the ``.onnx.mlir`` it produces.

    Parameters
    ----------
    model:
        The ONNX model to convert (usually the output of :func:`onnxsim.simplify`).
    onnx_mlir:
        Path to (or name of) the ``onnx-mlir`` binary. When omitted it is located
        via ``$ONNX_MLIR``, ``$ONNX_MLIR_HOME/bin/onnx-mlir``, then ``PATH``.
    emit:
        The onnx-mlir emit flag, with or without leading dashes. Defaults to
        ``"EmitONNXIR"``; ``"EmitONNXBasic"`` emits the raw import without shape
        inference. Only ONNX-dialect-emitting flags produce readable MLIR.
    opset_version:
        If given, convert the model to this opset with ``onnx.version_converter``
        before handing it to onnx-mlir.
    extra_args:
        Additional command-line arguments passed to ``onnx-mlir`` (e.g.
        ``["-O3"]``).
    timeout:
        Optional timeout (seconds) for the onnx-mlir subprocess.

    Returns
    -------
    str
        The emitted ONNX-dialect MLIR assembly.

    Raises
    ------
    RuntimeError
        If the onnx-mlir binary cannot be found or the compilation fails.
    """
    exe = _find_onnx_mlir(onnx_mlir)
    if exe is None:
        raise RuntimeError(_ONNX_MLIR_INSTALL_HINT)

    model_proto = _maybe_convert_opset(model, opset_version)
    emit_flag = "--" + emit.lstrip("-")

    tmpdir = tempfile.mkdtemp(prefix="onnxsim_mlir_")
    try:
        in_path = os.path.join(tmpdir, "model.onnx")
        try:
            onnx.save(model_proto, in_path)
        except ValueError:
            # >2GB models exceed the protobuf single-file limit; spill weights to
            # a side file next to the model so onnx-mlir can still load it.
            onnx.save(
                model_proto,
                in_path,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location="model.onnx.data",
            )

        out_base = os.path.join(tmpdir, "out")
        cmd = [exe, emit_flag, in_path, "-o", out_base]
        if extra_args:
            cmd.extend(extra_args)

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except FileNotFoundError as exc:
            raise RuntimeError(_ONNX_MLIR_INSTALL_HINT) from exc

        if proc.returncode != 0:
            raise RuntimeError(
                f"onnx-mlir failed to emit MLIR (exit code {proc.returncode}).\n"
                f"Command: {' '.join(cmd)}\n"
                f"stderr:\n{proc.stderr.strip()}"
            )

        return _read_onnx_mlir_output(out_base, tmpdir)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def export_mlir(
    model: onnx.ModelProto,
    output_path: Optional[str] = None,
    *,
    target: str = TORCH_TARGET,
    **kwargs,
) -> str:
    """Emit ``model`` as MLIR text, optionally writing it to ``output_path``.

    This is the public entry point used by the ``onnxsim --emit-mlir`` CLI and is
    re-exported as ``onnxsim.export_mlir``. It returns the MLIR text regardless of
    whether ``output_path`` is given, so it is equally usable in-memory.

    Parameters
    ----------
    model:
        The ONNX model to convert (usually the output of :func:`onnxsim.simplify`).
    output_path:
        If given, the MLIR text is written to this path (UTF-8). If ``None``, the
        text is only returned.
    target:
        Which MLIR dialect to emit: ``"torch"`` (Torch dialect, via torch-mlir) or
        ``"onnx"`` (ONNX dialect, via the onnx-mlir compiler). Any other value
        raises ``ValueError``.

    Other keyword arguments are forwarded to the selected backend --
    :func:`convert_to_torch_mlir` for ``"torch"`` and :func:`convert_to_onnx_mlir`
    for ``"onnx"``.

    Returns
    -------
    str
        The emitted MLIR assembly.

    Raises
    ------
    ValueError
        If ``target`` is not a supported MLIR target.
    RuntimeError
        If the selected target's backend (torch-mlir / onnx-mlir) is unavailable.
    """
    if target == TORCH_TARGET:
        text = convert_to_torch_mlir(model, **kwargs)
    elif target == ONNX_TARGET:
        text = convert_to_onnx_mlir(model, **kwargs)
    else:
        raise ValueError(
            f"Unsupported MLIR target {target!r}; supported targets are "
            f"{', '.join(repr(t) for t in SUPPORTED_TARGETS)}."
        )

    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)

    return text
