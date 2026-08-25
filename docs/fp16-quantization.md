# Float16 conversion (`quantize_fp16`)

## What this is

`onnxsim.quantize_fp16` is a single, self-contained C++ whole-graph
transform (`onnxsim/passes/quantize_fp16.h`) that converts every float32
weight -- and, by default, every internal activation -- in a model to
float16. It is a fundamentally different kind of "quantization" from every
other `quantize_*` function onnxsim ships: float16 is still an IEEE 754
floating-point format (5 exponent bits / 10 mantissa bits, versus float32's
8/23), not an integer scheme, so there is **no scale, no zero-point, and no
calibration data of any kind** -- every float32 value is simply rounded to
its nearest representable float16 value.

```
Before:
  Y = MatMul(Relu(MatMul(X, W1)), W2)    # W1, W2 constant, float32

After (keep_io_types=True, the default):
  Xf16 = Cast(X, to=FLOAT16)
  Y16  = MatMul(Relu(MatMul(Xf16, W1f16)), W2f16)   # W1f16/W2f16: float16
  Y    = Cast(Y16, to=FLOAT)
```

Unlike every INT8/INT4 pass here (each a narrow per-node pattern match),
this is a **whole-graph** transform: it doesn't look for `MatMul`/`Gemm`/
`Conv` specifically. It converts every constant float32 tensor it finds (a
true graph initializer, or a `Constant` node's embedded value -- both are
found the same way), and, when `keep_io_types` is true, inserts a boundary
`Cast` right after each float32 graph input and right before each float32
graph output, so the model's external interface (input/output names and
types) is unchanged -- only its internal weights and compute switch to
float16. With `keep_io_types=False`, graph inputs/outputs are redeclared
float16 directly instead (no casts; callers must then feed/read float16
tensors themselves).

No node's `op_type` or attributes are touched, and there is no per-op
float16-support check. An ordinary feedforward graph ends up computing
end-to-end in float16 purely as a side effect of every value along the way
now being float16-typed -- this pass never has to trace type propagation
itself, since almost every ONNX op propagates its input dtype to its output
dtype. The corollary: **a model containing an op with no float16 kernel in
the runtime it's deployed on will fail at execution time, not at conversion
time here** -- the same limitation every other float32-to-float16 model
converter (e.g. `onnxconverter-common`'s `convert_float_to_float16`) has.

## Out-of-range values are clamped, not rounded to infinity

Float16's largest finite magnitude is 65504. A float32 value beyond
`+-65504` (including `+-Inf` itself) is clamped to `+-65504` rather than
rounded to a float16 infinity, so this pass never silently introduces a new
`Inf`/`NaN` into the graph's numeric data on account of a single outlier
weight. A value that is already `NaN` stays `NaN`.

## Scope

Handled:
- Every float32 initializer and every float32-valued `Constant` node's
  embedded value in the top-level graph, regardless of which op consumes
  it -- unlike onnxsim's other quantization passes, this is not limited to
  `MatMul`/`Gemm`/`Conv`.
- Every float32 graph input/output (boundary `Cast` inserted, or redeclared
  directly, per `keep_io_types`).

Left untouched (safe no-op, that piece passes through as-is):
- Nodes inside control-flow subgraphs (`If`/`Loop`/`Scan` bodies) -- only
  the top-level graph is converted.
- An initializer whose name is also a graph input (the rarely-used ONNX
  "optional input with a default value" convention) -- left alone entirely,
  both to avoid the ambiguity of which type it should end up as and because
  it is vanishingly rare in practice.
- Non-float32 tensors (already-integer weights, `int64` `Shape` outputs,
  etc.) are never touched, regardless of `keep_io_types`.

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
onnxsim model.onnx model.fp16.onnx --fp16-quantize
```

### Python API

```python
import onnx
import onnxruntime as ort
import onnxsim

model = onnx.load("model.onnx")
model, check_ok = onnxsim.simplify(model)
assert check_ok

model = onnxsim.quantize_fp16(model)  # keep_io_types=True by default
onnx.save(model, "model.fp16.onnx")

sess = ort.InferenceSession("model.fp16.onnx", providers=["CPUExecutionProvider"])
outputs = sess.run(None, {"input": your_input_array})  # still float32 in/out
```

Pass `keep_io_types=False` to redeclare the model's own inputs/outputs as
float16 too (no boundary casts), at the cost of callers needing to feed and
read float16 tensors themselves:

```python
model = onnxsim.quantize_fp16(model, keep_io_types=False)
```

`tests/test_quantize_fp16.py` runs this simplify -> quantize -> deploy
sequence on small multi-op models (not a single isolated `MatMul`, since this
pass is a whole-graph transform), executing both the float and converted
graphs through `onnxruntime.InferenceSession` -- including a
`keep_io_types=False` run that feeds/reads `float16` numpy arrays directly,
and a dedicated case checking that an out-of-range weight is clamped rather
than turning into `Inf`/`NaN`.

## Relationship to onnxsim's other quantization methods

Float16 sits in a different part of the precision/size/accuracy tradeoff
than every INT8/INT4 scheme onnxsim ships:

| | Format | Calibration data | Typical size vs. float32 | Typical accuracy cost |
|---|---|---|---|---|
| `quantize_fp16` | float16 (floating point) | none | ~2x smaller | very low -- same exponent range as float32, just less mantissa precision |
| `quantize_weight_only` | INT8 (weights only) | none | ~4x smaller (weights) | low -- per-channel symmetric INT8 |
| `quantize_dynamic`/`quantize_static` | INT8/uint8 | none / required | ~4x smaller | moderate -- INT8's ~1/127 relative step |
| `quantize_weight_only_int4` | INT4 (weights only) | none | ~8x smaller (weights) | higher -- INT4's much coarser step, offset by block-wise scales |

Float16 is a good first thing to try when a model needs to be smaller or
faster but INT8/INT4's larger accuracy cost (or the deployment target's lack
of an integer-quantized-kernel runtime) is a concern: it halves storage with
much less numerical risk than any integer scheme, at the cost of a smaller
size reduction. It can also be combined with the weight-only INT8/INT4
passes' *activations* staying in float16 rather than float32 implicitly,
simply by running `quantize_fp16` on a model that has already gone through
one of them -- though note the weight-only passes' own `DequantizeLinear`
output stays float32 unless you explicitly convert the whole graph
afterward with `quantize_fp16`.
