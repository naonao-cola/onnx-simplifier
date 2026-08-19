from onnxsim.onnx_simplifier import (
    import_onnx_schemas,
    main,
    quantize_dynamic,
    simplify,
)

from .version import version as __version__

__all__ = [
    "simplify",
    "quantize_dynamic",
    "main",
    "import_onnx_schemas",
    "__version__",
]
