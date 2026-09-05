# Emitting standalone XNNPACK C code (`onnxsim.generate_xnnpack_c`)

[XNNPACK](https://github.com/google/XNNPACK) is a library of optimized
neural-network operator kernels (Arm/x86/WebAssembly), not a runtime that
loads `.onnx` files — a program links `libxnnpack` directly and calls its
[Subgraph API](https://github.com/google/XNNPACK/blob/master/include/xnnpack.h)
to build the computation itself. `onnxsim.generate_xnnpack_c` bridges that
gap: it turns an `onnx.ModelProto` into one self-contained `.c` file that
makes exactly those calls, for embedding into a target that can carry
`libxnnpack` but not onnxsim/onnx/protobuf at runtime.

This is a different thing from onnxsim's other, older XNNPACK integration
(`onnxsim/onnx_to_xnnpack_subgraph.h`, [dlpack-executor.md](dlpack-executor.md)):
that one builds a *live* `xnn_subgraph_t` in-process, as one of onnxsim's own
constant-folding backends. This one emits *text* — and, because it only needs
to name XNNPACK's C API correctly rather than call it, has no XNNPACK build
dependency itself and works in every ordinary `pip install onnxsim`.

```python
import onnx
import onnxsim

model = onnx.load("model.onnx")
onnxsim.export_xnnpack_c(model, "model.c", function_prefix="model")
# or: source = onnxsim.generate_xnnpack_c(model, "model")
```

The generated file exposes three functions and one struct, named from
`function_prefix`:

```c
typedef struct { xnn_subgraph_t subgraph; xnn_runtime_t runtime; } model_model_t;
int  model_create(model_model_t* model);
int  model_run(model_model_t* model, const float* const* inputs, float* const* outputs);
void model_destroy(model_model_t* model);
```

`inputs`/`outputs` are positional, in the same order as the source model's
own `graph.input`/`graph.output`. Compile the file against XNNPACK's headers
and library; nothing else is required. See the comment at the top of the
generated file itself for the exact input/output list.

## Supported ops (v1)

`Add`, `Sub`, `Mul`, `Div`, `Relu`, `Sigmoid`, `Gemm`, `MatMul`, `Reshape`
(only where layout-safe — see below), `Conv` (regular, grouped, and
depthwise), `GlobalAveragePool`. fp32 only, no quantization. Anything else
raises `RuntimeError` naming the unsupported op. Every tensor in the graph
must resolve to a concrete shape — the same bar
[`plan_activation_memory`](activation-memory-planning.md) holds tensors to —
since generated code bakes shapes in as literals; a dynamic dimension raises
`RuntimeError` too.

## The NHWC layout convention

XNNPACK's convolution Nodes hard-require NHWC (batch, height, width,
channels) activations and an OHWI-ish filter layout — confirmed against the
pinned XNNPACK commit (`cmake/build_xnnpack.cmake`'s
`ONNXSIM_XNNPACK_GIT_TAG`) from both its header's doc comments and its own
`test/subgraph/{convolution-2d,depthwise-convolution-2d}.cc` reference
implementations. ONNX is NCHW/OIHW by convention. Rather than insert a real,
data-moving Transpose Node at every NCHW/NHWC boundary — XNNPACK's Subgraph
API has no generic N-D transpose Node, only `xnn_define_static_reshape`'s
flat reinterpretation, which is not the same operation — this generator:

1. Emits **every** rank-4 tensor's XNNPACK Value already in NHWC. A Conv's
   own input/output Values, any purely-elementwise activation immediately
   upstream/downstream of one (`Add`/`Sub`/`Mul`/`Div`/`Relu`/`Sigmoid` don't
   care what order their axes are in — they only need producer and consumer
   to agree, which permuting *every* rank-4 tensor the same way guarantees),
   and any rank-4 *constant* feeding one of those ops (e.g. a per-channel
   bias reshaped to `[1, C, 1, 1]`, so it still broadcasts correctly against
   an NHWC activation) — none of these ever need a real data-moving
   transpose at all.
2. Permutes `Conv`/depthwise-`Conv` filter data (and any other rank-4
   constant) exactly once, at generation time — a free, one-time transpose
   of already-known constant bytes, emitted directly as the permuted C array
   literal. This is not a runtime cost.
3. Requires the model's own graph inputs/outputs to be supplied/read already
   in NHWC order by the generated code's caller — which is also not a real
   cost in the intended use case: a `cv::Mat` holding an interleaved image is
   *already* row-major HWC (see [Feeding a `cv::Mat` in](#feeding-a-cvmat-in)
   below), so the common "image in, tensor out" pipeline needs no conversion
   at that boundary either.

This is airtight as long as no op actually depends on axis *order* rather
than per-element values — true for everything above except `Reshape`, which
reinterprets a flat/row-major byte sequence and therefore only produces the
same result under NHWC physical order as ONNX's own (NCHW-assuming)
semantics intended when neither side has a real spatial extent to reorder:
rank != 4, or rank == 4 with H == W == 1 (immediately after a
`GlobalAveragePool`, e.g. the standard "backbone → classifier head" pattern:
`Conv` stack → `GlobalAveragePool` → `Reshape`/`Flatten` → `Gemm`).
Flattening a genuinely multi-pixel spatial map (H or W > 1) is rejected with
an explanatory `RuntimeError` instead of silently producing wrong numbers —
route it through `GlobalAveragePool` first, or keep it out of scope for now.

`GlobalAveragePool` itself is represented directly as its already-2-D `[N,
C]` result (reducing the NHWC input's H/W axes via `xnn_define_static_reduce`
with `xnn_reduce_mean`), rather than ONNX's own `[N, C, 1, 1]` — so a
`Reshape`/`Flatten` immediately after it becomes a same-shape no-op this
generator recognizes as trivially layout-safe (per the H == W == 1 rule
above) and folds into a redundant-but-harmless `xnn_define_static_reshape`.

### Feeding a `cv::Mat` in

[`onnxsim/xnnpack_cv_mat.hpp`](../onnxsim/xnnpack_cv_mat.hpp) is a template
you copy into your own C++ project (it is not compiled by onnxsim's own
build — onnxsim has no OpenCV dependency otherwise, and this header would
force one). It converts a `cv::Mat` to/from the tightly-packed NHWC float32
buffer a generated model's `..._run` expects, handling the one real cost a
non-continuous `cv::Mat` (a padded-row allocation, or an ROI/sub-`Mat` view)
still has: a genuine per-row copy, the same "materializing copy" data
movement this document's NHWC section describes for a transpose, just for a
row-strided source instead of a transposed one. See that header's own module
comment for the details.

## Scope (v1)

- **fp32 only.** No quantization (`onnx_to_xnnpack_subgraph.h`'s int8/uint8
  support is out of scope here).
- **A curated op list**, not general ONNX coverage — see above. Notably no
  sliding-window pooling (`AveragePool`/`MaxPool`, only the global variant),
  no `Concat`/`Split`/`Transpose`, and `Gemm`'s `alpha != 1`/`transA != 0` are
  unsupported, mirroring `onnx_to_xnnpack_subgraph.h`'s own `Gemm` limits.
- **No control flow, no dynamic shapes.** Every tensor must resolve to a
  concrete size.
- **A generator, not a verifier.** Nothing here executes the generated code
  or the source model to cross-check them; validate a generated file the
  normal way — compile it, run it, and diff its output against the source
  model run through `onnx.reference.ReferenceEvaluator` or ONNX Runtime.
