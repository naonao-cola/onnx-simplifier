# Bfloat16 conversion (`quantize_bf16`)

## What this is

`onnxsim.quantize_bf16` is a single, self-contained C++ whole-graph
transform (`onnxsim/passes/quantize_bf16.h`) that converts every float32
weight -- and, by default, every internal activation -- in a model to
bfloat16 ("brain float 16"). It is the same kind of calibration-free
"quantization" as `quantize_fp16`: bfloat16 is still an IEEE-754-style
floating-point format, not an integer scheme, so there is **no scale, no
zero-point, and no calibration data of any kind** -- every float32 value is
simply rounded to its nearest representable bfloat16 value.

```
Before:
  Y = MatMul(Relu(MatMul(X, W1)), W2)    # W1, W2 constant, float32

After (keep_io_types=True, the default):
  Xbf16 = Cast(X, to=BFLOAT16)
  Y16   = MatMul(Relu(MatMul(Xbf16, W1bf16)), W2bf16)   # W1bf16/W2bf16: bfloat16
  Y     = Cast(Y16, to=FLOAT)
```

Like `quantize_fp16`, this is a **whole-graph** transform, not a narrow
per-node pattern match: it converts every constant float32 tensor it finds
(a true graph initializer, or a `Constant` node's embedded value), and, when
`keep_io_types` is true, inserts a boundary `Cast` right after each float32
graph input and right before each float32 graph output, so the model's
external interface (input/output names and types) is unchanged. With
`keep_io_types=False`, graph inputs/outputs are redeclared bfloat16 directly
instead (no casts; callers must then feed/read bfloat16 tensors themselves).

## bfloat16 vs. float16

Both are 16-bit IEEE-754-style formats, but they split their bits
differently, which changes their tradeoffs:

| | Sign | Exponent | Mantissa | Range | Precision |
|---|---|---|---|---|---|
| float32 | 1 | 8 | 23 | huge | high |
| bfloat16 | 1 | 8 | 7 | same as float32 | low (~2-3 decimal digits) |
| float16 | 1 | 5 | 10 | +-65504 | higher than bfloat16 (~3-4 decimal digits) |

Because bfloat16 keeps float32's full 8-bit exponent, converting to it is
**structurally simpler** than converting to float16: no subnormal-number
handling and no clamping is needed anywhere in `FloatToBFloat16Bits` --
every finite float32 value maps to a finite bfloat16 value (and `+-Inf`
stays `+-Inf`, unlike `quantize_fp16`, which must clamp out-of-range values
to avoid manufacturing new infinities). The cost is bfloat16's much coarser
mantissa (7 bits vs. float16's 10), so it is less numerically precise
value-for-value than float16 -- the two formats sit at different points on
the range-vs-precision tradeoff, not one strictly better than the other.

No node's `op_type` or attributes are touched, and there is no per-op
bfloat16-support check, for the same reasons `quantize_fp16` has none (see
`docs/fp16-quantization.md`): this pass never traces type propagation
itself, it just makes every value along the way bfloat16-typed and lets each
op's own dtype-propagation behavior carry that through.

## A real runtime-support caveat

As of onnxruntime 1.29, its `CPUExecutionProvider` has **no bfloat16 compute
kernels** for the vast majority of ops (`MatMul`, `Relu`, `Add`, etc. all
raise `NOT_IMPLEMENTED` when handed bfloat16 tensors) -- only `Cast` and a
handful of type-agnostic ops like `Identity` currently have one. This is a
limitation of that specific runtime/provider combination, not of this pass
or of ONNX's own bfloat16 support: other runtimes and providers (e.g. many
accelerator/GPU execution providers, or newer CPU-kernel coverage in a later
onnxruntime release) may support bfloat16 compute far more broadly. Treat
`quantize_bf16` as most useful today for reducing on-disk/in-memory model
size and for deployment targets that do have real bfloat16 kernel coverage,
and verify your specific target runtime's bfloat16 op support before relying
on it for actual inference speedups.

## Scope

Handled:
- Every float32 initializer and every float32-valued `Constant` node's
  embedded value in the top-level graph, regardless of which op consumes
  it.
- Every float32 graph input/output (boundary `Cast` inserted, or redeclared
  directly, per `keep_io_types`).

Left untouched (safe no-op, that piece passes through as-is):
- Nodes inside control-flow subgraphs (`If`/`Loop`/`Scan` bodies) -- only
  the top-level graph is converted.
- An initializer whose name is also a graph input (the rarely-used ONNX
  "optional input with a default value" convention) -- left alone entirely,
  same reasoning as `quantize_fp16`.
- Non-float32 tensors are never touched, regardless of `keep_io_types`.

Like every other onnxsim quantization pass, this does not run shape
inference, constant folding, or any other simplification pass -- it applies
exactly this one rewrite, once, to a copy of the model (which is left
untouched) and returns the result. The old float32 initializer a converted
weight replaces is left orphaned in the model rather than pruned; call
`onnxsim.simplify()` afterward (or before, or both) to clean the graph up
and drop it. If the model already carries `value_info` for its interior
activations (e.g. because `simplify()` ran on it first, as the recommended
flow below does), this pass clears the now-stale float32 declarations those
tensors no longer have -- rather than leaving a *wrong* type in the exported
model, which ONNX Runtime's own load-time type-checking rejects outright,
unlike a merely absent value_info entry, which it infers fresh.

## End-to-end: simplify -> quantize -> deploy

### CLI

```bash
onnxsim model.onnx model.bf16.onnx --bf16-quantize
```

### Python API

```python
import onnx
import onnxsim

model = onnx.load("model.onnx")
model, check_ok = onnxsim.simplify(model)
assert check_ok

model = onnxsim.quantize_bf16(model)  # keep_io_types=True by default
onnx.save(model, "model.bf16.onnx")
```

Pass `keep_io_types=False` to redeclare the model's own inputs/outputs as
bfloat16 too (no boundary casts), at the cost of callers needing to feed and
read bfloat16 tensors themselves -- e.g. via the `ml_dtypes` package's
`ml_dtypes.bfloat16` numpy extension dtype, which `onnx.numpy_helper` uses
for `BFLOAT16` tensors:

```python
model = onnxsim.quantize_bf16(model, keep_io_types=False)
```

`tests/test_quantize_bf16.py` checks the produced graphs against
`onnx.checker`, verifies the converted weight values by decoding the
produced bfloat16 initializers back to float32 (via `ml_dtypes.bfloat16`)
and comparing against the original weights, and executes one quantized
graph through `onnxruntime.InferenceSession` end-to-end (using `Identity` as
the compute op, per the runtime-support caveat above).

## Relationship to onnxsim's other quantization methods

See `docs/fp16-quantization.md`'s comparison table for float16 and the
INT8/INT4 schemes. bfloat16 sits alongside float16 as a calibration-free
floating-point option, trading float16's higher per-value precision for
float32-equivalent dynamic range and structurally simpler conversion.
