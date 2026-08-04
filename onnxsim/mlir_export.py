"""Emit a (simplified) ONNX model as MLIR.

onnxsim's job stops at a cleaned-up ``onnx.ModelProto``. Compiler stacks built on
MLIR -- `torch-mlir <https://github.com/llvm/torch-mlir>`_ and, downstream of it,
IREE -- want that graph as MLIR instead. This module bridges the two: it converts
an ``onnx.ModelProto`` into **Torch-dialect** MLIR text using torch-mlir's
pure-Python ONNX importer (``torch_mlir.extras.onnx_importer``), the same importer
that backs torch-mlir's own ``torch-mlir-import-onnx`` / IREE's ``iree-import-onnx``
command-line tools.

Feeding a *simplified* model to the importer is the point: constant folding and
the optimizer passes collapse the shape-manipulation subgraphs that torch-mlir
would otherwise have to import op by op, so the emitted MLIR is smaller and closer
to what the downstream compiler actually needs.

torch-mlir is an **optional** dependency, mirroring how onnxruntime is optional for
constant folding (see ``backend.py``). It is imported lazily inside the functions
below, so ``import onnxsim`` never requires it; only ``--emit-mlir`` / the
``export_mlir`` API do. Install it with ``pip install torch-mlir`` (prebuilt
wheels: https://github.com/llvm/torch-mlir).

Only the Torch dialect is supported today. ``target`` is accepted as a parameter
so an ONNX-dialect (onnx-mlir) backend can be added later without breaking the
signature.
"""

from typing import Optional

import onnx

# The only MLIR target implemented so far. Kept as a named constant (rather than
# hard-coded strings) so a future onnx-mlir backend has an obvious place to slot
# in and callers can compare against it.
TORCH_TARGET = "torch"
SUPPORTED_TARGETS = (TORCH_TARGET,)

_TORCH_MLIR_INSTALL_HINT = (
    "torch-mlir is required to emit MLIR but is not installed. Install it with "
    "`pip install torch-mlir` (prebuilt wheels are listed at "
    "https://github.com/llvm/torch-mlir)."
)


def has_torch_mlir() -> bool:
    """Whether torch-mlir's ONNX importer is importable in this environment."""
    try:
        from torch_mlir.extras import onnx_importer  # noqa: F401
    except ImportError:
        return False
    return True


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

    model_proto = model

    if opset_version is not None:
        from onnx import version_converter

        model_proto = version_converter.convert_version(model_proto, opset_version)

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


def export_mlir(
    model: onnx.ModelProto,
    output_path: Optional[str] = None,
    *,
    target: str = TORCH_TARGET,
    opset_version: Optional[int] = None,
    run_shape_inference: bool = True,
    data_prop: bool = True,
    verify: bool = True,
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
        Which MLIR dialect to emit. Only ``"torch"`` (Torch dialect, via
        torch-mlir) is supported today; any other value raises ``ValueError``.

    Other keyword arguments are forwarded to :func:`convert_to_torch_mlir`.

    Returns
    -------
    str
        The emitted MLIR assembly.

    Raises
    ------
    ValueError
        If ``target`` is not a supported MLIR target.
    RuntimeError
        If the selected target's backend (torch-mlir) is not installed.
    """
    if target not in SUPPORTED_TARGETS:
        raise ValueError(
            f"Unsupported MLIR target {target!r}; supported targets are "
            f"{', '.join(repr(t) for t in SUPPORTED_TARGETS)}."
        )

    text = convert_to_torch_mlir(
        model,
        opset_version=opset_version,
        run_shape_inference=run_shape_inference,
        data_prop=data_prop,
        verify=verify,
    )

    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)

    return text
