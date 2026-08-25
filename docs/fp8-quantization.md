# 8-bit floating-point conversion (`quantize_fp8`)

## What this is

`onnxsim.quantize_fp8` is a single, self-contained C++ whole-graph transform
(`onnxsim/passes/quantize_fp8.h`) that converts every float32 weight -- and,
by default, every internal activation -- in a model to an 8-bit
floating-point format. It is the same kind of calibration-free
"quantization" as `quantize_fp16`/`quantize_bf16`: every format offered here
is still an IEEE-754-style floating-point format, not an integer scheme, so
there is **no scale, no zero-point, and no calibration data of any kind**.

```
Before:
  Y = MatMul(Relu(MatMul(X, W1)), W2)    # W1, W2 constant, float32

After (format="e4m3", keep_io_types=True, both defaults):
  Xf8 = Cast(X, to=FLOAT8E4M3FN)
  Y8  = MatMul(Relu(MatMul(Xf8, W1f8)), W2f8)   # W1f8/W2f8: FLOAT8E4M3FN
  Y   = Cast(Y8, to=FLOAT)
```

Like `quantize_fp16`/`quantize_bf16`, this is a **whole-graph** transform,
not a narrow per-node pattern match: it converts every constant float32
tensor it finds (a true graph initializer, or a `Constant` node's embedded
value), and, when `keep_io_types` is true, inserts a boundary `Cast` right
after each float32 graph input and right before each float32 graph output.
With `keep_io_types=False`, graph inputs/outputs are redeclared in the
target format directly instead (no casts; callers must then feed/read
tensors in that format themselves).

## Two target formats

`format` selects which 8-bit floating-point layout to convert to (see
[onnx's own float8 spec](https://github.com/onnx/onnx/blob/main/docs/docsgen/source/technical/float8.md)
for the full bit-level detail):

| | `"e4m3"` (default) | `"e5m2"` |
|---|---|---|
| Layout | 1 sign, 4 exponent, 3 mantissa bits | 1 sign, 5 exponent, 2 mantissa bits |
| Exponent bias | 7 | 15 |
| Max finite magnitude | 448 | 57344 |
| Infinity | none (all-ones exponent is entirely NaN) | yes |
| Typical use | weights | gradients (dynamic range closer to float16) |

Only these two formats -- the ones broadly implemented by NVIDIA/Intel/ARM
hardware and the ones the introducing papers (Micikevicius et al. 2022,
Noune et al. 2022) focus on -- are offered. Their `FNUZ` cousins
(`E4M3FNUZ`/`E5M2FNUZ`: AMD/GraphCore-oriented, no negative zero, a
different exponent bias, and a NaN encoding that collides with their
most-negative representable value) are not implemented here.

## Out-of-range values are saturated, not turned into infinity/NaN

A float32 value whose magnitude exceeds the target format's max finite
value -- including `+-Inf` itself -- is **clamped** to that max finite value
rather than mapped to the format's own infinity/NaN encoding. This is the
same design choice `quantize_fp16` makes for its own out-of-range values, and
matches the "with saturation" column of onnx's float8 conversion-semantics
table. A value that is already `NaN` maps to the format's canonical NaN
encoding (not to a clamped finite value).

This is worth calling out because it is a deliberate difference from some
other tools' default float8 casts (including plain NumPy-style
`array.astype(a_float8_dtype)` via the `ml_dtypes` package, which follows
the *non*-saturating column instead: out-of-range magnitudes become NaN
under E4M3FN, since it has no infinity encoding at all, or `+-Inf` under
E5M2). `quantize_fp8` always saturates, so a single outlier weight can never
turn a previously-finite value into `NaN`/`Inf` purely as a side effect of
this conversion.

## Rounding: ties-to-even, not ties-away-from-zero

`quantize_fp16`/`quantize_bf16` round ties away from zero -- a deliberate
simplification acceptable there because float16/bfloat16's much finer
mantissa (10 and 7 bits) makes an exact tie between two representable values
vanishingly rare on real weight/activation data. Float8's mantissa is only
2-3 bits wide, so exact ties are common enough in practice that
`quantize_fp8` implements true round-to-nearest, **ties-to-even** (RNE)
instead, matching float8's documented standard.

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

**Needs opset >= 19** -- the first opset where ONNX's `Cast` op supports
float8 types.

## A real runtime-support caveat

Float8 compute kernel coverage across current runtimes is narrower than
even bfloat16's (see `docs/bf16-quantization.md`'s own caveat about
onnxruntime 1.29's CPUExecutionProvider). Treat `quantize_fp8` as most
useful today for reducing on-disk/in-memory model size and for deployment
targets that do have real float8 kernel coverage (e.g. some GPU execution
providers targeting recent accelerator hardware), and verify your specific
target runtime's float8 op support before relying on it for actual
inference speedups.

## End-to-end: simplify -> quantize -> deploy

### CLI

```bash
onnxsim model.onnx model.fp8.onnx --fp8-quantize --fp8-format e4m3
```

### Python API

```python
import onnx
import onnxsim

model = onnx.load("model.onnx")
model, check_ok = onnxsim.simplify(model)
assert check_ok

model = onnxsim.quantize_fp8(model)  # format="e4m3", keep_io_types=True by default
onnx.save(model, "model.fp8.onnx")
```

Pass `format="e5m2"` for the wider-range, lower-precision format, and/or
`keep_io_types=False` to redeclare the model's own inputs/outputs in the
target format too (no boundary casts) -- e.g. via the `ml_dtypes` package's
`ml_dtypes.float8_e4m3fn`/`ml_dtypes.float8_e5m2` numpy extension dtypes,
which `onnx.numpy_helper` uses for these TensorProto types:

```python
model = onnxsim.quantize_fp8(model, format="e5m2", keep_io_types=False)
```

`tests/test_quantize_fp8.py` checks the produced graphs against
`onnx.checker`, verifies the converted weight values two ways -- by
decoding the produced float8 initializers back to float32 and comparing
against `ml_dtypes`'s own (independent) implementation for in-range values,
and by exact known tie values to confirm ties-to-even rounding -- and
executes one quantized graph through `onnxruntime.InferenceSession`
end-to-end (using `Identity` as the compute op, per the runtime-support
caveat above).

## Relationship to onnxsim's other quantization methods

See `docs/fp16-quantization.md`'s comparison table for float16 and the
INT8/INT4 schemes, and `docs/bf16-quantization.md` for bfloat16. Float8
sits alongside them as a calibration-free floating-point option with the
smallest footprint of any of onnxsim's floating-point quantization methods
(half of float16/bfloat16's), at the cost of the least numeric precision of
any format onnxsim offers, integer schemes included.
