# Exporting to MLIR (torch-mlir / onnx-mlir)

onnxsim's output is a cleaned-up `onnx.ModelProto`. MLIR-based compiler stacks —
[torch-mlir](https://github.com/llvm/torch-mlir), [IREE](https://iree.dev/)
downstream of it, and [onnx-mlir](https://github.com/onnx/onnx-mlir) — want that
graph as MLIR instead. `onnxsim.export_mlir` (and the `--emit-mlir` CLI flag)
bridges the two, emitting one of two dialects selected by `target` /
`--mlir-target`:

- `torch` (default) — **Torch dialect**, via torch-mlir's pure-Python ONNX
  importer.
- `onnx` — **ONNX dialect**, via the onnx-mlir compiler binary.

This document records how the bridge works and why it is shaped the way it is.

## Why simplify first

Compiler importers translate the graph op by op. A model exported from a
framework is typically full of shape-manipulation subgraphs (`Shape` → `Gather`
→ `Unsqueeze` → `Concat` → `Reshape`, dynamic-axis arithmetic, and so on) that
exist only to recompute constants the exporter could not fold. Importing those
verbatim produces bulky MLIR that the downstream compiler then has to fold again.

Running onnxsim first — constant folding plus the optimizer passes — collapses
those subgraphs into constants before the importer ever sees them, so the emitted
MLIR is smaller and closer to the actual computation. Emitting MLIR from *the
simplified model* is therefore the whole point, not an afterthought.

## Both backends are optional

Like onnxruntime (used for constant folding, see
[`onnxsim/backend.py`](../onnxsim/backend.py)), neither MLIR backend is a hard
dependency. Nothing is imported or located at `import onnxsim` time — only
`export_mlir` / `--emit-mlir` reach for a backend, and a missing one raises a
clear `RuntimeError` with an install hint. The implementation lives in
[`onnxsim/mlir_export.py`](../onnxsim/mlir_export.py).

## torch-mlir (Torch dialect)

`convert_to_torch_mlir` is a thin, pure-Python wrapper around torch-mlir's
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

Install it with `pip install torch-mlir` (prebuilt wheels are listed at
<https://github.com/llvm/torch-mlir>).

## onnx-mlir (ONNX dialect)

onnx-mlir has no in-process Python importer; the ONNX → ONNX-dialect conversion
lives in the compiler's C++ frontend. So `convert_to_onnx_mlir` shells out to the
`onnx-mlir` binary:

1. Optionally convert the model's opset.
2. Write the model to a temporary `.onnx` file (spilling weights to a side file
   for models above the 2 GB protobuf limit).
3. Run `onnx-mlir --EmitONNXIR <model.onnx> -o <out>`, which writes
   `<out>.onnx.mlir`. (`--EmitONNXBasic` emits the raw import without shape
   inference; the emit flag is configurable via `emit`.)
4. Read the produced `.onnx.mlir` back and return it. A non-zero exit is
   surfaced as a `RuntimeError` carrying onnx-mlir's stderr.

The binary is located, in order, from an explicit `onnx_mlir` argument
(`--onnx-mlir`), the `ONNX_MLIR` environment variable (full binary path),
`$ONNX_MLIR_HOME/bin/onnx-mlir`, then `PATH`. Build or install onnx-mlir from
<https://github.com/onnx/onnx-mlir>.

## Usage

CLI:

```
# Torch dialect (default), written next to the output model as simplified.mlir
onnxsim input.onnx simplified.onnx --emit-mlir

# Torch dialect, explicit path
onnxsim input.onnx simplified.onnx --emit-mlir model.mlir

# ONNX dialect via onnx-mlir (located on PATH / ONNX_MLIR_HOME / ONNX_MLIR)
onnxsim input.onnx simplified.onnx --emit-mlir --mlir-target onnx

# ONNX dialect, pointing at the binary explicitly
onnxsim input.onnx simplified.onnx --emit-mlir model.mlir \
  --mlir-target onnx --onnx-mlir /path/to/onnx-mlir
```

Python:

```python
import onnx
import onnxsim

model = onnx.load("input.onnx")
model_simp, ok = onnxsim.simplify(model)
assert ok

# Torch dialect
mlir_text = onnxsim.export_mlir(model_simp)          # return the MLIR text
onnxsim.export_mlir(model_simp, "model.mlir")        # and/or write it to a file

# ONNX dialect
onnx_mlir_text = onnxsim.export_mlir(model_simp, target="onnx")
```

## Testing

- `tests/test_mlir_export.py` covers the torch-mlir path end to end. Because
  torch-mlir is not part of onnxsim's test requirements, the module skips itself
  when torch-mlir is not installed (`pytest.importorskip`, the same arrangement
  as `tests/test_modelopt_integration.py`). The dedicated `mlir-integration` CI
  workflow installs torch-mlir and runs it.
- `tests/test_onnx_mlir_export.py` covers the onnx-mlir path. The backend-neutral
  checks and the shell-out plumbing (via a tiny fake `onnx-mlir` script) run in
  the normal test matrix without a real onnx-mlir; the true end-to-end test runs
  only when an `onnx-mlir` binary can be located.
