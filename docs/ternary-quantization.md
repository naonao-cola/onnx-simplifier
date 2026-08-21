# Ternary weight quantization (`quantize_ternary`)

## What this is

`onnxsim.quantize_ternary` is a single, self-contained C++ graph rewrite
(`onnxsim/passes/dynamic_quantize_ternary_matmul.h`) that detects MatMul/
"vanilla" Gemm nodes whose constant weight is *structurally ternary* --
every element of every output column is one of `{-s, 0, +s}` for that
column's own scale `s` -- and dynamically quantizes them.

This is the weight representation [BitNet
b1.58](https://github.com/microsoft/BitNet) and similar ternary-weight
models use internally (the "absmean quantizer": each `BitLinear` layer's
weight is one of `{-s, 0, +s}`). A generic ONNX export still stores such a
weight as a dense float32 initializer -- 16x larger than it needs to be, and
run on the generic float `MatMul` kernel instead of an integer one.

## Relationship to `quantize_dynamic`

This is not a new quantization scheme -- it is `quantize_dynamic`'s exact
rewrite (see `docs/dynamic-quantization.md`), applied only where the weight
turns out to already be ternary, with one difference: the weight's INT8
encoding is derived *structurally* rather than by rounding the weight's full
dynamic range to `[-127, 127]`.

```
Before:
  Y = MatMul(X, W)              # W constant, [K, N], float32, ternary

After (identical shape to quantize_dynamic):
  Xq, Xs, Xzp = DynamicQuantizeLinear(X)      # uint8, computed at runtime
  Wq          = <int8, values in {-1, 0, 1}>  # LOSSLESS -- not rounded
  Ws          = <float32, one scale per column of W>
  Acc         = MatMulInteger(Xq, Wq, Xzp)    # int32
  Y           = Cast<float>(Acc) * (Xs * Ws)  # dequantize
```

Because `Wq`'s codes are an exact structural decomposition (`round(W /
scale)` always lands exactly on `{-1, 0, 1}` within `rtol`, by construction
of what "ternary" means here), the only quantization error this rewrite
introduces is in the activation -- `quantize_dynamic`'s rounded weight
encoding contributes none of the error for a ternary weight, since a ternary
weight's full range already *is* `{-1, 0, 1}` scaled, so the two encodings
happen to coincide numerically. What differs is *detection*: this pass only
fires on weights it can prove are ternary, and leaves everything else --
including ordinary dense float32 weights -- for `quantize_dynamic` to handle
if you want them quantized too. **The two passes compose**: run
`quantize_ternary` first to catch the ternary layers precisely, then
`quantize_dynamic` to catch the rest (a BitNet-family model's LM head, for
example, is typically full-precision and stays a float `MatMul` either way).

```python
model = onnxsim.quantize_ternary(model)   # ternary BitLinear projections
model = onnxsim.quantize_dynamic(model)   # anything left (e.g. the LM head)
```

## Scope

Handled:
- `MatMul(X, W)` / `Gemm(X, W[, B], transA=0, alpha=1, beta=1)` with `W` a
  constant 2-D float32 tensor whose every column is within `rtol` (default
  `1e-5`, relative to the column's own scale) of `{-s, 0, +s}`.
- Opsets >= 11 (`DynamicQuantizeLinear` was introduced in opset 11).

Left untouched (safe no-op, node passes through as-is):
- Any weight that is not structurally ternary -- including an ordinary
  dense float32 weight, a 4-level weight (`{-2s, -s, 0, s}` is one level too
  many), or a weight where even a single element falls outside `rtol` of its
  column's `{-s, 0, +s}`.
- An entirely-zero weight (carries no ternary signal; rewriting it only adds
  nodes for no benefit).
- Non-constant or non-2-D weights, non-default Gemm attributes, non-float32
  activations, or an opset older than 11.

## End-to-end: simplify -> quantize -> deploy

### CLI

```bash
onnxsim bitnet.onnx bitnet.quant.onnx --ternary-quantize --dynamic-quantize
```

`onnxsim`'s ordinary simplification runs first -- for a BitNet export this
matters more than usual: `nn.Linear`'s weight is `[out_features,
in_features]`, so an unfused exporter may emit `Transpose(weight) -> MatMul`
rather than a plain initializer, which structural ternary detection cannot
see through on its own. Constant folding (part of ordinary simplification)
is what exposes the plain `[K, N]`/`[N, K]` initializer this pass needs.

### Python API

```python
import onnx
import onnxruntime as ort
import onnxsim

model = onnx.load("bitnet.onnx")
model, check_ok = onnxsim.simplify(model)
assert check_ok

model = onnxsim.quantize_ternary(model)   # ternary BitLinear projections
model = onnxsim.quantize_dynamic(model)   # anything left non-ternary
onnx.save(model, "bitnet.quant.onnx")

sess = ort.InferenceSession("bitnet.quant.onnx", providers=["CPUExecutionProvider"])
outputs = sess.run(None, {"input_ids": your_input_array})
```

`tests/test_ternary_quantize.py` covers structural detection (including the
non-ternary/four-level/all-zero rejection cases), the lossless weight-code
property, composition with `quantize_dynamic`, and running both the float
and quantized graphs through `onnxruntime.InferenceSession`.

## Why not `com.microsoft::MatMulNBits`?

ONNX Runtime ships a contrib op, `MatMulNBits`, that packs low-bit weights
(down to 2 bits for a ternary weight, a further ~4x saving on top of what
this pass's int8 codes get) and has an optimized int8-compute kernel for
them. An earlier iteration of this converter targeted that op directly.

This pass deliberately does not: every other onnxsim quantization pass
(`quantize_dynamic`, `quantize_static`, `quantize_weight_only`) emits only
standard ONNX operators, so the result loads on any conformant runtime, not
just onnxruntime builds new enough to carry a given contrib op's kernel (2-bit
`MatMulNBits` support is itself recent and not present in every onnxruntime
build). A `com.microsoft`-domain rewrite is exactly the kind of
runtime/vendor-specific output onnxsim's broader quantization roadmap treats
as a distinct, opt-in "target profile" rather than the default -- see the
project roadmap for that as tracked future work, alongside similar profiles
for TensorRT- and QNN-oriented QDQ conventions.
