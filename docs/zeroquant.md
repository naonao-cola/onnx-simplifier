# ZeroQuant: group-wise weight + per-token activation W8A8 (`apply_zeroquant`)

## What this is

`onnxsim.apply_zeroquant` quantizes both operands of a MatMul/vanilla-Gemm
layer to INT8, at two specific granularities applied *together*:

- **Weight**: symmetric INT8, one scale per `(block_size`-wide K-group,
  output channel)`.
- **Activation**: symmetric INT8, one scale per token (per row), computed
  fresh at graph-run time from that token's own values -- no calibration
  data.

Both quantized operands are then contracted with a real, grouped
`MatMulInteger` pipeline (genuine `int8 x int8` integer compute, not a
quantize-then-immediately-dequantize float simulation):

```
Before:
  Y = MatMul(X, W) [+ bias]      -- W constant, [K, N], float32

After (conceptually -- see "Why grouped MatMulInteger" below for the
actual node-level construction):
  Xq = round_to_nearest_int8_per_token(X)          -- computed at runtime
  Wq, Ws = per_group_symmetric_int8(W)             -- computed once, here
  Acc = sum over K-groups g of MatMulInteger(Xq[:, group g], Wq[group g])
  Y = Acc * Xscale * Ws [+ bias]                    -- dequantize
```

## Where this comes from

[ZeroQuant](https://arxiv.org/abs/2206.01861) (Yao, Aminabadi, Zhang, Wu,
Li, He, 2022) targets exactly this combination as the hardware-friendly
sweet spot for W8A8 transformer inference. **Both halves of this scheme
already existed in onnxsim before this module, in isolation**:

- Group-wise INT8 weight quantization is exactly
  `onnxsim.quantize_weight_only_int8_block` (see
  [int8-block-quantization.md](int8-block-quantization.md)). This module
  does not reimplement that pass's weight math; it reuses the identical
  granularity.
- Per-token dynamic INT8 activation quantization (`scale = max(|x|,
  axis=-1) / 127`, computed at graph-run time) is a pattern this repo
  already uses repeatedly -- `onnxsim.apply_quarot`, `onnxsim.apply_duquant`,
  `onnxsim.apply_attention_quantization`, and
  `onnxsim.quantize_kv_cache`'s Value-style rewrite. But every one of those
  existing uses immediately dequantizes back to float32 right after
  quantizing -- a round-trip that simulates the *precision loss* of
  quantizing (useful for those modules' own INT4 weight/activation
  schemes, which keep the actual matmul running in float), never feeding
  the quantized activation into a true integer matmul. And onnxsim's one
  existing *integer*-executing activation path,
  `onnxsim.quantize_dynamic` (`onnxsim/passes/dynamic_quantize_matmul.h`),
  uses standard ONNX `DynamicQuantizeLinear` -- which computes **one scale
  for the entire input tensor**, not one per row/token, despite "dynamic"
  in the name.

So genuine per-token dynamic quantization feeding a *real* integer matmul
did not exist anywhere in onnxsim. ZeroQuant's real, non-redundant
contribution here is pairing the two existing granularities and executing
them as real integer compute -- not either piece alone.

The paper's other contribution, **layer-by-layer knowledge distillation
(LKD)** -- a training loop that compensates deeper-layer quantization error
by distilling each quantized layer against its own original-precision
output -- is **out of scope**. onnxsim's whole architecture is stateless
graph-rewriting on an existing ONNX protobuf; LKD needs a training loop
over a framework-native model with gradient computation, the same
quantization-aware-training boundary
[nncf-comparison-future-work.md](nncf-comparison-future-work.md)'s own
"Explicitly out of scope" section draws. LKD is not reproduced here, nor
anywhere else in onnxsim.

## Why grouped `MatMulInteger`, and why the activation is symmetric

Standard ONNX's `MatMulInteger` schema documents a per-row zero point on
`A` and a per-column zero point on `B` -- which would appear to be exactly
what a per-token activation scale and a per-group weight scale need, in
one call. Two separate obstacles rule that out:

1. A single `MatMulInteger` call always contracts the *entire* `K`
   dimension into one integer accumulator, so a weight scale that varies
   partway through `K` (this module's whole point) cannot be applied after
   the fact -- the different groups' products are already summed together
   by the time the op returns. This module therefore slices both the
   quantized activation and the quantized weight into `block_size`-wide
   groups along `K`, runs one real `MatMulInteger` per group, and combines
   the groups' dequantized partial sums in float32 afterward. Each group's
   own accumulation is small enough (`block_size` terms) that int32
   overflow is a non-issue in practice (`_MAX_SAFE_GROUP_SIZE` is
   ~66,311), unlike `onnxsim.quantize_dynamic`'s own accumulator-overflow
   guard on the full, ungrouped reduction depth.
2. Empirically (checked directly against `onnxruntime` while building this
   module, not assumed from the spec text alone): ONNX Runtime's own CPU
   `MatMulInteger` kernel **rejects a genuine per-row `a_zero_point`** at
   run time (`IsScalarOr1ElementVector(a_zero_point) was false`) even
   though the ONNX operator *schema* documents that shape as valid -- the
   schema's per-row zero point is, in practice, unimplemented on the one
   execution provider onnxsim's own tests run against. So this module
   quantizes the activation **symmetrically** (zero point always exactly
   `0`) instead of the asymmetric, `DynamicQuantizeLinear`-style scheme its
   own per-token *scale* formula would otherwise suggest -- the same
   symmetric convention every other per-token dynamic scale in this repo
   already uses. With zero point fixed at a compile-time constant `0`
   (never a per-row tensor), `a_zero_point`/`b_zero_point` are omitted
   entirely (their documented default), sidestepping the unimplemented
   shape completely. Only the *scale* varies per token; that multiply
   happens entirely outside `MatMulInteger` (in this module's own `Mul`
   node against the dequantized float accumulator), so per-token
   granularity is fully preserved -- it is only the zero point that had to
   give.

## Scope

Handled:

- `MatMul(X, W)` with `W` a constant 2-D float32 tensor whose reduction
  dimension `K` is divisible by `block_size`, or `Gemm(X, W[, B])` with
  `transA=0`, `alpha=1`, `beta=1` (when `B` is present) under the same
  weight constraint. `transB` may be 0 or 1.
- `X` of any rank >= 1 -- every leading (batch/sequence) dimension is
  flattened into the per-token row dimension at graph-run time via a
  `Shape`/`Reshape` pair, then restored on the output.
- Opsets >= 18 (this module's per-token activation scale needs
  `ReduceMax`'s `axes`-as-input form, and equal-sized `Split` needs its
  `num_outputs` attribute -- both opset 18).

Left untouched (safe no-op, node passes through as-is):

- Non-constant or non-2-D weights.
- A reduction dimension `K` not evenly divisible by `block_size` -- a
  ragged last group is left to a future extension rather than
  approximated, matching `quantize_weight_only_int8_block`'s own scope
  choice.
- `block_size` that is not a positive divisor of `K`, or exceeds
  `_MAX_SAFE_GROUP_SIZE`.
- A model with no matching layer, or an opset older than 18.

## Usage

```python
import onnx
import onnxsim

model = onnx.load("model.onnx")
model, check_ok = onnxsim.simplify(model)
assert check_ok

quantized = onnxsim.apply_zeroquant(model, block_size=32)
onnx.save(quantized, "model.zeroquant.onnx")

import onnxruntime as ort
sess = ort.InferenceSession("model.zeroquant.onnx", providers=["CPUExecutionProvider"])
outputs = sess.run(None, {"input": your_input_array})
```

Needs no calibration data at all: the weight's per-group scales come from
the weight's own static values, and the activation's per-token scale is
computed fresh at graph-run time from that token's own values.

## Relationship to onnxsim's other W8A8/weight-only schemes

| | Weight scale granularity | Activation scale granularity | Real integer compute |
|---|---|---|---|
| `quantize_weight_only` | per output channel | untouched (float) | no |
| `quantize_weight_only_int8_block` | per (K-group, channel) | untouched (float) | no |
| `quantize_dynamic` | per output channel | per tensor (`DynamicQuantizeLinear`) | yes |
| `apply_zeroquant` | per (K-group, channel) | per token | yes |

`apply_zeroquant` is the only scheme in this table combining group-wise
weight granularity with per-token activation granularity while still
executing as real integer `MatMulInteger` compute -- the specific
combination the ZeroQuant paper identifies as its own contribution.
