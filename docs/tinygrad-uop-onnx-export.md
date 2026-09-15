# Representing a tinygrad UOp graph in ONNX (experimental, inspection-only)

**Status: experimental, inspection-only.** `onnxsim.tinygrad_uop_export`
exports a real tinygrad `UOp` graph -- the same per-kernel `Ops.SINK`-rooted
AST `onnxsim.webgpu_tinygrad_codegen` renders to WGSL (see
`docs/webgpu-kernel-dispatch.md`) -- as an ONNX `ModelProto`, under a
private, non-standard domain (`"tinygrad.uop"`). The result is **never
runnable** by any ONNX runtime: no runtime implements `ALU`/`RANGE`/
`BUFFER`/`REDUCE`/... as tensor ops. This exists purely so a `UOp` graph can
be *inspected* through ordinary ONNX tooling instead of tinygrad's own
`VIZ=1` debug server.

## Why

tinygrad already has its own UOp graph inspection story: `VIZ=1` (or
`TRACK_MATCH_STATS=2`) records every graph-rewrite step as a `RewriteTrace`
dataclass and pickles it to a temp file (`tinygrad/viz/serve.py`), then
serves a bundled d3.js/dagre web UI that converts a `UOp` to a JSON
node/edge structure (`uop_to_json`) on the fly for rendering.
`onnxsim.tinygrad_uop_export` trades that for two things:

- **A stable, safe container.** A pickle is a live Python object graph
  frozen to disk -- unpickling runs arbitrary code, and the format silently
  breaks across tinygrad versions whenever a pickled class's shape changes
  (`UOp`, `ShapeTracker`, `KernelInfo`, ...). A `.onnx` file is plain
  protobuf: safe to open from anywhere, and forward/backward tolerant of
  unknown fields the way protobuf always is.
- **Free generic tooling.** Netron (and any other protobuf/ONNX-aware
  viewer) renders a custom, unrecognized domain's nodes and edges
  generically -- readable graph structure with no bespoke UI to build or
  maintain.

## What it does

`uop_to_onnx_model(root)` walks `root.toposort()` (the same traversal
`onnxsim.webgpu_tinygrad_codegen._lower_tensor_program` itself uses) and
emits one `NodeProto` per `UOp`:

- `op_type` is the `Ops` enum member's own name (`"ADD"`, `"RANGE"`,
  `"REDUCE"`, `"SINK"`, ...).
- `domain` is `"tinygrad.uop"` (`onnxsim.tinygrad_uop_export.DOMAIN`).
- Node/output names are `u<i>`, where `i` is that `UOp`'s position in the
  toposort ordering -- deterministic given a fixed graph, unlike Python's
  own `id()`, so two exports of the same graph produce byte-identical
  files.
- Inputs are that `UOp`'s own `.src`, by position.
- A `dtype` attribute always carries `str(u.dtype)`.
- An `arg` attribute (`arg_i`/`arg_f`/`arg_s`/`arg_repr`, whichever fits)
  carries `u.arg` when the `UOp` has one -- see below.

## What's lossy, and why that's fine here

`UOp.arg`'s type varies wildly by op: a plain `int` for `DEFINE_GLOBAL`, an
`Ops` enum member paired with an axis id for `REDUCE`, a `ParamArg`/
`KernelInfo` dataclass for `PARAM`/`SINK`, a `ShapeTracker`/`View` for
movement ops on a pre-schedule graph, and so on. ONNX's own `AttributeProto`
only has native slots for scalars/strings/tensors/graphs and lists thereof
-- there is no faithful, lossless encoding for an arbitrary Python object.
`_encode_arg` encodes a plain `int`/`float`/`bool`/`str` natively
(`arg_i`/`arg_f`/`arg_s`) and falls back to `repr()`, stored as a plain
string attribute (`arg_repr`), for everything else. This is a deliberately
lossy escape hatch: the goal is faithful *inspection* (a human or a generic
viewer can read every node's real dtype and a reasonable rendering of its
arg), not a lossless round trip back to a real `UOp` -- nothing here
attempts one, and there is no corresponding importer.

## Scope

Exports exactly the `UOp` DAG reachable from one root -- typically a single
per-kernel AST (one `Ops.CALL`'s own `src[0]` from a scheduled program), not
a whole multi-kernel schedule (`schedule_linear()`'s own result, which
threads several such ASTs together via `Ops.CALL` nodes). Exporting a whole
program is just calling this once per kernel AST; stitching the resulting
graphs into one file isn't done here.

## Testing

`tests/test_tinygrad_uop_export.py` builds a real per-kernel AST from a
small `Conv2D` (tagged for the `WEBGPU` device, never actually opened --
same reasoning as `onnxsim.webgpu_tinygrad_codegen`'s own tests) and checks:

- `onnx.checker.check_model` accepts the exported model.
- Node count matches the `UOp` toposort exactly, and every node's
  `op_type`/inputs/`dtype` attribute match the corresponding `UOp`
  structurally (not just "some graph got produced").
- The graph's single output name is the root `UOp`'s own synthetic name.
- The custom domain is declared in `opset_import`.
- A `CONST` node's plain int/float `arg` is encoded natively (`arg_i`/
  `arg_f`), not via the `arg_repr` fallback -- confirming the fallback is
  only reached for genuinely irregular `arg` types.

Skipped when `tinygrad` isn't installed (`pytest.importorskip("tinygrad")`),
matching `tests/test_webgpu_tinygrad_codegen.py`'s own convention.
