from onnxsim.onnx_simplifier import import_onnx_schemas, main, simplify

from .version import version as __version__

__all__ = ["simplify", "main", "import_onnx_schemas", "__version__"]
