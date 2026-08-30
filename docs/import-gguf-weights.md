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

## GGML "K-quant" support

Real quantized checkpoints store most of their weights as one of GGML's
block-quantized formats, not plain float. This function decodes the
**K-quant** family -- `Q4_K`, `Q5_K`, `Q6_K` (256-element super-blocks, each
with its own packed 6-bit per-sub-block scale/min pair) and `Q8_0`
(32-element blocks, one fp16 scale each) -- which is what Unsloth's `*_K_M`/
`*_K_S`/`Q8_0` GGUF exports actually use for the bulk of their tensors. Every
block layout and dequantization formula is transcribed directly from GGML's
own reference implementation
(https://github.com/ggml-org/ggml -- `ggml-common.h`'s block structs,
`ggml-quants.c`'s `dequantize_row_q*` functions) and cross-checked against an
independent from-scratch re-implementation over full random blocks before
being committed (see `onnxsim/ggml_kquant.h`'s file comment).

A matched K-quant tensor's initializer has its `data_type` forced to
`FLOAT` regardless of what `model` previously declared for it -- the
decoded values are only meaningful as float32.

## Scope

Handled:
- `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0` -- decoded to float32.
- Any raw (already-unquantized) GGML type (`F32`, `F16`, `BF16`, `F64`,
  `I8`/`I16`/`I32`/`I64`) -- copied through unchanged, same as `import_gguf`.

Not handled (reported in `skipped`, left untouched):
- The legacy `Q4_0`/`Q4_1`/`Q5_0`/`Q5_1`/`Q8_1` family, `Q2_K`/`Q3_K`/`Q8_K`,
  and every `IQ*` importance-matrix variant -- these have real, different
  block layouts this decoder does not implement.

Only an initializer whose *name* matches a GGUF tensor is ever touched;
`import_gguf_weights` never adds new initializers or otherwise changes the
graph's structure.

## Mixture-of-experts (MoE) checkpoints

llama.cpp's own GGUF convention for a MoE model's per-expert feed-forward
weights lines up with `com.microsoft.MoE`'s layout without any extra work,
for the three families checked (Mixtral, Qwen3-MoE, gpt-oss): the tensors
are named `blk.N.ffn_gate_exps.weight` / `blk.N.ffn_up_exps.weight` /
`blk.N.ffn_down_exps.weight` (gate=`fc1_experts_weights`, up=
`fc3_experts_weights`, down=`fc2_experts_weights` -- see
`contrib_schemas.cpp`'s `BuildMoEFunctionBody` comment for that naming) and
`blk.N.ffn_gate_inp.weight` for the router. Their GGML shapes, reversed by
this function's existing (rank-agnostic) `ne[]`-order rule, land exactly on
`com.microsoft.MoE`'s own `fc1_experts_weights`/`fc2_experts_weights`/
`fc3_experts_weights` shapes -- no additional transpose needed, because
GGML's per-expert matrix already uses the same `[in, out]` convention as an
ordinary 2D linear weight (which this function already round-trips
correctly), just with an extra leading expert axis. K-quant blocks apply to
the flattened element stream regardless of tensor rank, so the existing
decoder needs no MoE-specific change either -- see
`tests/test_import_gguf_weights.py`'s
`test_import_gguf_weights_hydrates_a_moe_node_with_llama_cpp_names` for an
end-to-end round trip through a real `com.microsoft.MoE` node.

Two real gaps remain, both reported via `skipped` (or simply absent from
the model, in the second case) rather than silently mishandled:
- Official gpt-oss GGUF releases quantize expert weights as **MXFP4**, not
  one of the K-quant formats this decoder implements.
- Some newer llama.cpp architectures fuse the gate and up projections into
  one `ffn_gate_up_exps` tensor instead of two separate ones -- there is no
  1:1 initializer to hydrate that into (the same fused-layout gap
  `swiglu_fusion` has in the reference decomposition itself).

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
the other approach for a small, curated set of recognized architectures --
currently the Llama family (`llama`, `qwen2`, `mistral`, which share the
same RMSNorm/RoPE/GQA/SwiGLU block shape): it builds the graph itself from
the checkpoint's declared hyperparameters, then calls
`import_gguf_weights` internally to hydrate it. No MoE architecture has a
template there yet -- the "MoE checkpoints" section above is about
`import_gguf_weights` alone, for a MoE graph you already built or exported
some other way.

## Tests

`tests/test_import_gguf_weights.py` writes real, byte-accurate GGUF v3
files containing hand-encoded `Q8_0`/`Q4_K` blocks with known values
(computing each expected float independently, not by reusing the C++
decoder under test) and checks `import_gguf_weights`' decoded result against
them, plus coverage for multi-dimensional shapes (GGML's
innermost-dimension-first `ne[]` vs ONNX's outermost-first shape), the
unsupported/unmatched skip list, raw-dtype passthrough, a real 3D
expert-tensor round trip under llama.cpp's own MoE naming convention (see
the "Mixture-of-experts" section above), and a shape-mismatched tensor
ending up in `skipped` with its initializer's original value intact.
