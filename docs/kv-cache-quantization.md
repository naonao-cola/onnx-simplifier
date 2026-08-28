# KV-cache quantization (`quantize_kv_cache`)

## What this is

`onnxsim.quantize_kv_cache` finds every autoregressive decoder KV-cache
stream in a model -- a graph input (`past_key`/`past_key_values.{i}.key`,
...) concatenated with this step's freshly computed key or value along the
sequence axis, feeding a graph output (`present_key`/`present.{i}.key`,
...) that a caller feeds back in as next step's `past_*` input, exactly the
shape `tools/onnx-deploy`'s own `KvCachePipeline` and
`tests/test_symexpr_kv_cache_consistency.py`'s toy model both use -- and
quantizes it to INT8, symmetric, in one of two ways depending on the
stream:

- **Key-style** (the default): one scale per channel (the head-dim axis),
  calibrated once from representative data and shared by every cached
  token for that stream's whole lifetime.
- **Value-style** (matched by name, or via `value_output_names`): a fresh,
  data-free scale per *token*, computed from that token's own values the
  instant it's produced, carried forward as a second growing KV-cache
  stream alongside the codes.

Every other quantizer in onnxsim compresses a **weight**: something
computed once, offline, before the model ever runs. A KV cache is the
opposite -- an **activation** that keeps growing for the entire lifetime of
one generation, one new key/value vector appended per decode step. That is
exactly why it is worth quantizing at all: it is the part of an LLM's
memory footprint that scales with sequence length, unlike the weights.

```
Key-style, before:
  past_key: graph input, float32 [..., seq_past, head_dim]
  new_key:  float32 [..., seq_new, head_dim]         -- this step's own K/V
  present_key = Concat(past_key, new_key, axis=seq)  -- graph output,
                and consumed by the attention math (QK^T / softmax@V)

Key-style, after:
  past_key: graph input, INT8 [..., seq_past, head_dim]    -- dtype changed
  key_scale: initializer, float32 [head_dim]                 -- per-channel
  key_zero_point: initializer, INT8 [head_dim], all zero     -- symmetric
  new_key_q = QuantizeLinear(new_key, key_scale, key_zero_point, axis=-1)
  present_key = Concat(past_key, new_key_q, axis=seq)   -- INT8 graph output
  present_key_f = DequantizeLinear(present_key, key_scale, key_zero_point,
                                    axis=-1)             -- float32
  <every other consumer of the old float present_key now reads present_key_f>
```

Concatenating `past_key` (already int8) with `new_key_q` (freshly quantized
with the *same* per-channel scale) along the sequence axis is lossless with
respect to what was already stored -- the scale never changes step to step,
so there is no compounding requantization error the way there would be if
the whole growing cache were dequantized and requantized with a fresh scale
every step. Only this step's new tokens are ever quantized, so the cost per
decode step stays constant as the sequence grows, and the graph's own
`present_*` output is genuinely compressed (roughly 4x smaller than
float32) the whole way through a caller's decode loop -- not just an
internal round-trip that still stores float32 everywhere.

Value-style streams (matched by a `present` output name containing
`".value"`, or listed explicitly via `value_output_names`) get a different
rewrite -- no calibration, but a second parallel scale stream:

```
Value-style, after:
  past_value: graph input, INT8 [..., seq_past, head_dim]
  past_value_scale: graph input, float32 [..., seq_past, 1]   -- NEW input,
    one scale per already-cached token
  new_scale = max(reduce_max(abs(new_value), axis=head_dim), eps) / 127
    -- one scale per new token, computed fresh from that token's own
    values, no calibration data involved
  new_value_q = cast(clip(round(new_value / new_scale), -128, 127), INT8)
  present_value = Concat(past_value, new_value_q, axis=seq)          -- INT8
  present_value_scale = Concat(past_value_scale, new_scale, axis=seq) -- NEW
    output, float32, grows in lockstep with present_value
  present_value_f = cast(present_value, float32) * present_value_scale
  <every other consumer of the old float present_value now reads present_value_f>
```

Past tokens' scales are never revised once set (the same "no compounding
requantization error" property as Key-style above) -- only this step's new
token is ever quantized, at a scale tailored to it specifically. The new
`past_value_scale`/`present_value_scale` pair is picked up by
`KvCachePipeline`'s existing `present.`/`past_key_values.`
string-substitution convention automatically, with no C++ changes needed
(it's float32, a dtype `detail::BorrowView` already handled).

## Where this comes from

Two published techniques quantize the KV cache well: **KIVI** (Liu et al.,
ICML 2024, <https://arxiv.org/abs/2402.02750>) and **KVQuant** (Hooper et
al., NeurIPS 2024, <https://arxiv.org/abs/2401.18079>). Both share the same
core empirical finding: Key activations have a handful of channels with
persistently large magnitude across the *whole* sequence, so quantizing Key
**per channel** (one scale shared by every cached token) preserves far more
accuracy than quantizing it per token. `quantize_kv_cache`'s Key-style
rewrite reproduces that part of both papers.

KIVI's other empirical finding -- Value activations *don't* have that
persistent-channel structure, so a **per-token** scale preserves more
accuracy there instead -- is reproduced by `quantize_kv_cache`'s
Value-style rewrite (matched automatically by name, or via
`value_output_names`; needs opset 18, see Scope below).

What it does **not** reproduce:

- **KIVI's residual-window bookkeeping** (the most recent `R` tokens kept
  in full precision, only finalized into low-bit once they age out of that
  window). Deciding which tokens have "aged out" and need finalizing is
  cross-step, host-side state -- not something one exported ONNX graph can
  express on its own. It belongs in
  `tools/onnx-deploy/include/onnx_deploy/kv_cache_pipeline.h` (which
  already owns exactly this kind of cross-step cache state across decode
  steps) as a follow-up, not here.
- **KVQuant's non-uniform codebook and dense-and-sparse outlier isolation**
  -- both add real complexity (a fitted, non-uniform datatype; per-vector
  outlier separation) for a further accuracy gain past plain per-channel
  INT8; not implemented here, a natural next step if calibrated INT8 alone
  turns out not to be enough for a given model.

## Scope

Handled:

- A `Concat(past, new, axis=seq)` node whose `past` operand is a float32
  graph input consumed *only* by that Concat, and whose own output is
  directly a graph output. No assumption is made about tensor names -- this
  matches `past_key`/`present_key` as well as `optimum-onnx`'s own
  `past_key_values.{i}.key`/`present.{i}.key` convention.
- Opsets >= 13 (`QuantizeLinear`/`DequantizeLinear`'s per-channel `axis`
  needs opset 13) for Key-style streams; opsets >= 18
  (`ReduceMax`'s `axes`-as-input form) for Value-style streams -- each
  `Reduce*` op moved its `axes` attribute to an input on its own schedule,
  not all at opset 13 the way `ReduceSum` did.

Left untouched (safe no-op, node passes through as-is):

- A `past` operand with any other consumer besides the Concat (declining
  rather than silently breaking that other use).
- A Concat whose axis *is* the last (channel) axis -- no distinct axis is
  left to quantize per-channel on.
- A model with no matching Concat pattern, or an opset older than 13.
- A stream matched as Value-style, when the model's opset is below 18 --
  left completely untouched (not silently downgraded to Key-style).

## Usage

```python
import onnx
import onnxsim

model = onnx.load("decoder_with_past_model.onnx")
# Key-style (calibrated, per-channel) by default; any matched stream whose
# present output name contains ".value" automatically gets Value-style
# (data-free, per-token) treatment instead -- see value_output_names to
# name streams explicitly.
quantized = onnxsim.quantize_kv_cache(model, num_samples=32)
onnx.save(quantized, "decoder_with_past_model.kv_int8.onnx")
```

`calibration_data` defaults to `onnxsim.generate_random_calibration_data`
(random input, no external dependency); pass real representative batches
(e.g. via `onnxsim.load_huggingface_calibration_data`) for a tighter
calibrated scale, since a per-channel scale that under-covers a real
model's activation range clips outliers on exactly the channels KIVI/
KVQuant's own finding says matter most.

A model quantized this way needs a caller that actually stores its
`past_*`/`present_*` tensors as INT8 across decode steps to see any real
memory benefit -- `tools/onnx-deploy`'s `KvCachePipeline`
(`include/onnx_deploy/kv_cache_pipeline.h`) supports this:
`detail::BorrowView` handles INT8 tensors (alongside the original fp32/
int64) and threads them through `Generate()`'s decode loop exactly like any
other cache dtype, verified end-to-end by
`tools/onnx-deploy/scripts/make_toy_int8_kv_decoder.py` and
`.github/workflows/onnx-deploy.yml` against a real, growing INT8 cache
across many steps and two different ONNX Runtime releases. Note that a real
multi-file `optimum-onnx`-style export needs `decoder_model.onnx` (the
"no past" first step) quantized consistently with `decoder_with_past_model.onnx`
for the same cache stream too, since both files share the same
`present.*`/`past_key_values.*` dtype contract; `quantize_kv_cache` only
matches graphs with a `Concat(past, new)` pattern (present in the "with
past" file, absent in the first-step file, which has no past to concat),
so quantizing both files of a real pipeline consistently is left to the
caller for now.

`tests/test_kv_cache_quantization.py` verifies this end-to-end with a
genuine two-step round trip (an empty starting cache, then feeding the
first step's own INT8 `present_key` output back in as the second step's
`past_key` input -- exactly what a real pipeline does), comparing the
dequantized cache tensor against the float model's own output.
