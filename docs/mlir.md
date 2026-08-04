# Exporting to MLIR (torch-mlir)

onnxsim's output is a cleaned-up `onnx.ModelProto`. MLIR-based compiler stacks —
[torch-mlir](https://github.com/llvm/torch-mlir) and, downstream of it,
[IREE](https://iree.dev/) — want that graph as MLIR instead. `onnxsim.export_mlir`
(and the `--emit-mlir` CLI flag) bridges the two by converting the model into
**Torch-dialect** MLIR.

This document records how the bridge works and why it is shaped the way it is.

## Why simplify first

torch-mlir's ONNX importer translates the graph op by op. A model exported from a
framework is typically full of shape-manipulation subgraphs (`Shape` → `Gather`
→ `Unsqueeze` → `Concat` → `Reshape`, dynamic-axis arithmetic, and so on) that
exist only to recompute constants the exporter could not fold. Importing those
verbatim produces bulky MLIR that the downstream compiler then has to fold again.

Running onnxsim first — constant folding plus the optimizer passes — collapses
those subgraphs into constants before the importer ever sees them, so the emitted
MLIR is smaller and closer to the actual computation. Emitting MLIR from *the
simplified model* is therefore the whole point, not an afterthought.

## How it works

`onnxsim/mlir_export.py` is a thin, pure-Python wrapper around torch-mlir's
importer. It mirrors torch-mlir's own `torch-mlir-import-onnx` /
`iree-import-onnx` command-line tools:

1. Optionally upgrade the model to a requested opset with
   `onnx.version_converter` (torch-mlir's op coverage targets recent opsets).
2. Best-effort re-run `onnx.shape_inference.infer_shapes` (with data
   propagation) so value shapes are present — the importer produces
   better-typed MLIR when they are. A simplified model is normally already
   shape-inferred, so a failure here is ignored rather than fatal.
3. Create an MLIR `Context` with the Torch dialect registered, build the module
   skeleton via `onnx_importer.ModelInfo(...).create_module(...)`, and let
   `onnx_importer.NodeImporter.define_function(...).import_all()` walk the main
   graph.
4. Verify the module and return its assembly (`Operation.get_asm`).

## torch-mlir is optional

torch-mlir ships large prebuilt wheels pinned to specific LLVM/PyTorch versions.
Like onnxruntime (used for constant folding, see [`onnxsim/backend.py`](../onnxsim/backend.py)),
it is an **optional** dependency: it is imported lazily inside `mlir_export.py`,
so `import onnxsim` never requires it. Only `export_mlir` / `--emit-mlir` do, and
they raise a clear `RuntimeError` with an install hint when it is missing.

Install it with:

```
pip install torch-mlir
```

Prebuilt wheels are listed at <https://github.com/llvm/torch-mlir>.

## Usage

CLI:

```
# writes simplified.onnx and simplified.mlir
onnxsim input.onnx simplified.onnx --emit-mlir

# choose the MLIR path explicitly
onnxsim input.onnx simplified.onnx --emit-mlir model.mlir
```

Python:

```python
import onnx
import onnxsim

model = onnx.load("input.onnx")
model_simp, ok = onnxsim.simplify(model)
assert ok

mlir_text = onnxsim.export_mlir(model_simp)        # return the MLIR text
onnxsim.export_mlir(model_simp, "model.mlir")      # and/or write it to a file
```

## Testing

`tests/test_mlir_export.py` covers the conversion end to end. Because torch-mlir
is not part of onnxsim's test requirements, the module skips itself when
torch-mlir is not installed (the same `pytest.importorskip` arrangement as
`tests/test_modelopt_integration.py`). The dedicated `mlir-integration` CI
workflow installs torch-mlir and runs these tests.

## Future work: onnx-mlir

Only the Torch dialect is implemented today. `export_mlir` takes a `target`
argument (currently only `"torch"`) so an ONNX-dialect backend built on
[onnx-mlir](https://github.com/onnx/onnx-mlir) can be added later without
changing the public signature.
