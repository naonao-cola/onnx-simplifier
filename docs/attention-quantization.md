# Attention computation quantization (`apply_attention_quantization`)

## What this is

`onnxsim.apply_attention_quantization` quantizes the attention
computation itself -- not a weight, and not the KV-cache -- for the
common decomposed attention subgraph:

```
Before:
  scores  = MatMul(Q, Kt)                  -- Kt: K, transposed
  scaled  = Mul(scores, scale)  [optional]  -- e.g. 1/sqrt(head_dim)
  masked  = Add(scaled, mask)   [optional]  -- e.g. causal mask
  probs   = Softmax(masked, axis=-1)
  out     = MatMul(probs, V)

After:
  Qdq, Kdq = per-token INT8 round-trip of Q, Kt        -- data-free
  scores   = MatMul(Qdq, Kdq)                          -- unchanged shape/attrs
  ...scale/mask/softmax unchanged...
  probsdq  = fixed-scale (1/255) UINT8 round-trip of probs
  Vdq      = per-token INT8 round-trip of V             -- data-free
  out      = MatMul(probsdq, Vdq)
```

## Where this comes from

Every other quantizer in onnxsim targets either a weight-bearing
MatMul/Gemm layer (`quantize_weight_only_int4` and everything built on
it -- `apply_spinquant`, `apply_quarot`, `apply_duquant`, ...) or the
KV-cache tensors specifically (`onnxsim.quantize_kv_cache`). Nothing
quantizes the attention *computation* itself: the `QK^T` score matmul, or
the `softmax(QK^T)@V` value-weighted sum. Both are pure
activation-to-activation matmuls -- no constant weight at all -- so none
of onnxsim's weight-quantization machinery applies to them.

Three tensors get quantized to INT8, each via the technique already
best-suited to what it actually is -- needing **no calibration data for
any of them**:

- **Q and K**: data-free, per-token dynamic INT8, the same pattern
  `apply_quarot`/`apply_duquant` already use for their own activation
  quantization (`scale = max(|x|, axis=-1) / 127`, computed fresh at
  graph-run time).
- **V**: the same per-token dynamic INT8 scheme.
- **The Softmax output** (the attention probabilities): unlike every
  other activation in onnxsim, a Softmax output's range isn't merely
  typical -- it's *guaranteed* to lie in `[0, 1]` for any input at all
  (the "activation-range provenance" fact `onnxsim.precision_estimator`'s
  own docstring already names, point 4, but never previously used to
  actually quantize anything). That makes it the one activation in this
  whole package quantizable with a **fixed, non-data-dependent** scale --
  UINT8 with `scale = 1/255`, `zero_point = 0` -- no calibration run, no
  runtime scale computation at all, just an ordinary round-to-nearest
  against a constant.

The score matmul itself and the Softmax normalization are left running
in float, exactly as `apply_smoothquant`/`apply_awq` leave their own
internal reductions in float -- only the tensors *crossing* a matmul
boundary are quantized. Like `apply_quarot`/`apply_duquant`'s own
activation handling, the quantize-dequantize round trip is simulated in
float32 (no true INT8 tensor type is used) since there is no real integer
MatMul kernel available in an ordinary ONNX graph to benefit from an
actual cast -- the point is to measure and preserve the precision loss a
real INT8 attention kernel would introduce, not to save graph-level
storage the way weight quantization does.

## Scope

Handled:

- The subgraph `MatMul(Q, Kt) -> [Mul or Div] -> [Add] -> Softmax ->
  MatMul(probs, V)`, where the optional scale (`Mul`/`Div`) and mask
  (`Add`) nodes may each be present or absent, in either combination.
- Opset >= 18 (the per-token Q/K/V scale's `ReduceMax` needs its
  `axes`-as-input form, matching `quantize_kv_cache`'s own Value-style
  gate).

Left untouched (safe no-op, subgraph passes through as-is):

- A Softmax node whose input doesn't trace back to a MatMul within two
  hops, or whose output isn't consumed as a MatMul's first input.
- A model with no matching subgraph, or an opset older than 18.

## Usage

```python
import onnx
import onnxsim

model = onnx.load("model.onnx")
quantized = onnxsim.apply_attention_quantization(model)
onnx.save(quantized, "model.attnq.onnx")
```

Needs no calibration data at all: every scale in this module is either
computed fresh from the actual data at graph-run time (Q, K, V) or fixed
by the Softmax output's own guaranteed range (the attention
probabilities).
