import onnx

__all__ = [
    "generate_xnnpack_c",
    "export_xnnpack_c",
]


def generate_xnnpack_c(model: onnx.ModelProto, function_prefix: str = "model") -> str:
    """Generate a standalone C source file reconstructing ``model`` as an
    XNNPACK (https://github.com/google/XNNPACK) Subgraph, for embedding into a
    target that can carry libxnnpack but not onnxsim/onnx/protobuf at runtime.

    Delegates entirely to the C++ core (``onnxsim/xnnpack_codegen.h``/``.cpp``)
    -- see that header's module comment for the full v1 scope (supported ops,
    fp32-only, and critically the NHWC layout convention every 4-D tensor is
    emitted under, which differs from ONNX's own NCHW). ``model`` is not
    modified; shape inference runs internally regardless of whether ``model``
    already carries shapes, since generated code must bake in concrete sizes.

    The returned string is a complete ``.c`` file exposing three functions
    named from ``function_prefix`` (default ``"model"``): ``model_create``,
    ``model_run``, and ``model_destroy`` (plus a ``model_model_t`` struct) --
    see the comment at the top of the generated file itself for their exact
    signatures and the model's input/output order. Compile it against
    XNNPACK's own headers/library; nothing else is required.

    Raises the same errors :func:`onnxsim._C._generate_xnnpack_c` raises: a
    ``ValueError``-wrapped message for an invalid ``function_prefix``, or a
    ``RuntimeError`` naming the offending node/tensor for anything outside
    this generator's v1 scope (an unsupported op, a non-fp32 tensor, a shape
    that does not resolve to a concrete size, or a layout-unsafe Reshape).
    """
    from onnxsim import onnxsim_cpp2py_export as _C

    return _C._generate_xnnpack_c(model.SerializeToString(), function_prefix)


def export_xnnpack_c(
    model: onnx.ModelProto, path: str, function_prefix: str = "model"
) -> None:
    """Like :func:`generate_xnnpack_c`, but writes the generated C source
    directly to ``path`` instead of returning it as a string.
    """
    source = generate_xnnpack_c(model, function_prefix)
    with open(path, "w", encoding="utf-8") as f:
        f.write(source)
