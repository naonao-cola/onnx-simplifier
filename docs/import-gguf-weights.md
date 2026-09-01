# Importing GGUF weight values (`import_gguf_weights`)

## What this is

`onnxsim.import_gguf_weights(model, gguf_path)` hydrates an *existing* ONNX
graph's initializers, by name, from any GGUF file -- including a plain
third-party checkpoint like a Hugging Face GGUF export (e.g. Unsloth's Qwen3
GGUFs), which has no ONNX graph inside it at all, just weight tensors and
llama.cpp architecture metadata.

This is different in kind from `onnxsim.import_gguf`: that function requires
a GGUF file `onnxsim.export_gguf` itself produced (an embedded `model.onnx`
blob alongside the tensors -- the same self-describing-archive trick
`export_safetensors`/`import_safetensors` use), and reconstructs the whole
graph from it. A real quantized-LLM `.gguf` has no such embedded graph, so
`import_gguf` cannot open one. `import_gguf_weights` needs no embedded
model: bring your own graph for the same architecture (e.g. exported by
another tool), with initializers named to match the checkpoint's own tensor
names, and this fills in their values.

```python
import onnx
import onnxsim

model = onnx.load("qwen3_architecture.onnx")  # from some other exporter
model, skipped = onnxsim.import_gguf_weights(model, "model.gguf")
onnx.save(model, "qwen3_hydrated.onnx")
```

`skipped` lists GGUF tensors present in the file that a same-named
initializer in `model` could not actually be hydrated from -- either the
GGUF tensor's quantized format has no decoder here (see "Scope" below), or
the file's tensor decodes to a different byte count than `model`'s
initializer declares (dtype x dims), e.g. a placeholder built for the wrong
shape. Either way the initializer keeps its original value rather than
being overwritten with bytes that don't fit its declared shape. This does
**not** include tensors simply absent from `model`'s initializers, which
are silently left alone.

## GGML "K-quant", legacy quant, and MXFP4 support

Real quantized checkpoints store most of their weights as one of GGML's
block-quantized formats, not plain float. This function decodes the
**K-quant** family -- `Q4_K`, `Q5_K`, `Q6_K` (256-element super-blocks, each
with its own packed 6-bit per-sub-block scale/min pair) and `Q8_0`
(32-element blocks, one fp16 scale each) -- which is what Unsloth's `*_K_M`/
`*_K_S`/`Q8_0` GGUF exports actually use for the bulk of their tensors; the
**legacy** family -- `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1` (plain 32-element blocks,
no super-block scale/min table) -- which llama.cpp's own mixed-precision
quantizers still pick for particular tensor roles (embeddings, attention
projections) even in an otherwise K-quant checkpoint; and **MXFP4**
(32-element blocks, one shared power-of-two E8M0 exponent byte and 16 bytes
of packed 4-bit e2m1-style codes) -- the OCP Microscaling FP4 format
official gpt-oss GGUF releases use natively for their MoE expert weights.
Every block layout and dequantization formula is transcribed directly from
GGML's own reference implementation
(https://github.com/ggml-org/ggml -- `ggml-common.h`'s block structs,
`ggml-quants.c`'s `dequantize_row_q*`/`dequantize_row_mxfp4` functions,
`ggml-impl.h`'s `ggml_e8m0_to_fp32_half`) and cross-checked against an
independent from-scratch re-implementation over full random blocks before
being committed (see `onnxsim/ggml_kquant.h`'s, `onnxsim/
ggml_legacy_quant.h`'s, and `onnxsim/ggml_mxfp4.h`'s file comments).

The legacy family's real-world importance was confirmed empirically, not
assumed: fetching the real header of several official
[`unsloth/gpt-oss-20b-GGUF`](https://huggingface.co/unsloth/gpt-oss-20b-GGUF)
quantizations (via an HTTP range request for just the header bytes, no need
to download the multi-gigabyte weight data) showed that most of the
popular, size-optimized ones -- `Q4_K_M`, `Q4_K_S`, `Q5_K_M`, `Q4_0`,
`UD-Q4_K_XL` -- mix in one or more legacy-family tensors for `token_embd`/
`attn_q`/`attn_k`/`attn_v`, and would fail to import at all without this.
Only the largest variants (`F16`, `Q8_0`, `Q6_K`, `UD-Q8_K_XL`) used
exclusively K-quant/MXFP4/raw types before this family was added. The
smallest variants (`Q2_K`, `Q3_K_M`) still fail: they additionally need
`Q3_K` and an `IQ*`/`Q2_K`-family type, which remain out of scope (see
"Scope" below).

A matched K-quant, legacy-quant, or MXFP4 tensor's initializer has its
`data_type` forced to `FLOAT` regardless of what `model` previously
declared for it -- the decoded values are only meaningful as float32.

## Scope

Handled:
- `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0` (K-quant), `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`
  (legacy), `MXFP4` -- decoded to float32.
- Any raw (already-unquantized) GGML type (`F32`, `F16`, `BF16`, `F64`,
  `I8`/`I16`/`I32`/`I64`) -- copied through unchanged, same as `import_gguf`.

Not handled (reported in `skipped`, left untouched):
- `Q8_1`, `Q2_K`/`Q3_K`/`Q8_K`, `NVFP4`, and every `IQ*` importance-matrix
  variant -- these have real, different block layouts this decoder does not
  implement.

Only an initializer whose *name* matches a GGUF tensor is ever touched;
`import_gguf_weights` never adds new initializers or otherwise changes the
graph's structure.

## Mixture-of-experts (MoE) checkpoints

llama.cpp's own GGUF convention for a Mixtral-style MoE model's per-expert
feed-forward weights lines up with `com.microsoft.MoE`'s Mixtral-style
(`activation_type="silu"` + separate `fc3`) layout without any extra work:
the tensors are named `blk.N.ffn_gate_exps.weight` /
`blk.N.ffn_up_exps.weight` / `blk.N.ffn_down_exps.weight` (gate=
`fc1_experts_weights`, up=`fc3_experts_weights`, down=`fc2_experts_weights`
-- see `contrib_schemas.cpp`'s `BuildMoEFunctionBody` comment for that
naming) and `blk.N.ffn_gate_inp.weight` for the router. Their GGML shapes,
reversed by this function's existing (rank-agnostic) `ne[]`-order rule,
land exactly on `com.microsoft.MoE`'s own `fc1_experts_weights`/
`fc2_experts_weights`/`fc3_experts_weights` shapes -- no additional
transpose needed, because GGML's per-expert matrix already uses the same
`[in, out]` convention as an ordinary 2D linear weight (which this function
already round-trips correctly), just with an extra leading expert axis.
K-quant/MXFP4 blocks apply to the flattened element stream regardless of
tensor rank, so the existing decoder needs no MoE-specific change either --
see `tests/test_import_gguf_weights.py`'s
`test_import_gguf_weights_hydrates_a_moe_node_with_llama_cpp_names` for an
end-to-end round trip through a real `com.microsoft.MoE` node.

**gpt-oss is different**: it keeps gate/up as the same two separate
`ffn_gate_exps`/`ffn_up_exps` tensors (verified directly against
`llama.cpp/src/models/openai-moe.cpp`'s `load_arch_tensors` -- there is no
single fused tensor anywhere in a real gpt-oss GGUF file), but each one is
individually decodable by this function exactly like Mixtral's (MXFP4 or
K-quant, doesn't matter). What's genuinely different is which
`com.microsoft.MoE` reference decomposition applies: gpt-oss needs the
`activation_type="swiglu"`/`swiglu_fusion=1` one (added alongside the MXFP4
decoder above), which expects ONE interleaved fc1 tensor, not two separate
ones. `import_gguf_weights` alone -- simple 1:1 name-matching -- cannot
build that interleaved tensor; `onnxsim.reconstruct_gguf_graph`'s `gpt-oss`
architecture template (see the next section) does this as ordinary ONNX
graph ops (`Reshape`/`Concat`/`Reshape`) at graph-construction time, which
`onnxsim.simplify()` later constant-folds away for free. A caller building
their own gpt-oss graph from scratch (rather than going through
`reconstruct_gguf_graph`) would need to do the same interleave themselves
before calling `import_gguf_weights` -- this function still has no
built-in way to fuse two named tensors into one on a caller's behalf.

Some newer llama.cpp architectures fuse the gate and up projections into
one `ffn_gate_up_exps` tensor *in the GGUF file itself* instead of two
separate ones -- a genuinely different, still-unaddressed case (there is no
1:1 initializer for `import_gguf_weights` to hydrate that into at all, and
no matching GGUF tensor for it to report as `skipped` either, since the
absence is simply invisible from a caller's own two-separate-initializer
graph). gpt-oss is not an example of this -- see above.

## Why not reconstruct the whole model from the GGUF file?

For an arbitrary architecture, this is out of scope: a `.gguf` LLM
checkpoint's metadata (layer count, hidden size, attention/RoPE
configuration, ...) describes a llama.cpp-runtime model, not an ONNX
computation graph, and there is no generic way to recover an `Add`/`MatMul`/
`Attention` node structure from it for architectures nobody has written a
template for. `import_gguf_weights` solves the narrower, always-applicable
problem of getting a checkpoint's *weight values* into a graph you already
have, regardless of architecture.

`onnxsim.reconstruct_gguf_graph` (see `onnxsim/gguf_reconstruct.py`) takes
the other approach for a small, curated set of recognized architectures: it
builds the graph itself from the checkpoint's declared hyperparameters,
then calls `import_gguf_weights` internally to hydrate it.

- The Llama family (`llama`, `qwen2`, `mistral`, which share the same
  RMSNorm/RoPE/GQA block shape) -- either a plain SwiGLU FFN, or, when
  `expert_count > 0` (a Mixtral-style checkpoint), the Mixtral-style
  `com.microsoft.MoE` node described above.
- `gpt-oss` -- a genuinely different architecture (YaRN-scaled RoPE,
  alternating sliding-window/full attention with attention sinks, and an
  always-MoE FFN using the `swiglu_fusion=1` decomposition, gate/up fused
  from the checkpoint's two separate tensors as described above). See
  `onnxsim/gguf_reconstruct.py`'s module docstring and
  `_reconstruct_gpt_oss`'s own docstring for exactly which llama.cpp source
  every detail (tensor names, hardcoded activation constants, the
  gating-function equivalence, YaRN's exact formula) was verified against.

For any OTHER architecture (including one whose GGUF file fuses gate/up
into a single `ffn_gate_up_exps` tensor -- the still-unaddressed case noted
above), the "MoE checkpoints" section above is about `import_gguf_weights`
alone, for a MoE graph you already built or exported some other way.

## Tests

`tests/test_import_gguf_weights.py` writes real, byte-accurate GGUF v3
files containing hand-encoded `Q8_0`/`Q4_K` (K-quant), `Q4_0`/`Q4_1`/`Q5_0`/
`Q5_1` (legacy, see `test_import_q4_0_weights` and friends), and MXFP4
blocks with known values (computing each expected float independently, not
by reusing the C++ decoder under test) and checks `import_gguf_weights`'
decoded result against them, plus coverage for multi-dimensional shapes
(GGML's innermost-dimension-first `ne[]` vs ONNX's outermost-first shape),
the unsupported/unmatched skip list, raw-dtype passthrough, a real 3D
expert-tensor round trip under llama.cpp's own MoE naming convention (see
the "Mixture-of-experts" section above), and a shape-mismatched tensor
ending up in `skipped` with its initializer's original value intact.
