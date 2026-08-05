#pragma once

#include <onnx/onnx_pb.h>

#include <memory>
#include <optional>
#include <vector>

#include "dlpack/dlpack.h"

// RAII owner for a DLManagedTensor: releasing it invokes the tensor's own
// DLPack deleter exactly once (per the DLPack contract), which frees whatever
// the producer attached -- a borrowed-buffer no-op, an Ort::Value, a host
// allocation, etc. Move-only.
struct DLManagedTensorDeleter {
  void operator()(DLManagedTensor* t) const {
    if (t != nullptr && t->deleter != nullptr) {
      t->deleter(t);
    }
  }
};
using DLManagedTensorPtr =
    std::unique_ptr<DLManagedTensor, DLManagedTensorDeleter>;

// The constant-folding executor boundary. onnxsim runs each fold group by
// building a throwaway sub-model and asking an executor to evaluate it. Tensors
// cross this boundary as DLPack DLManagedTensors rather than onnx::TensorProto,
// so an executor can borrow onnxsim's buffers (and hand its results back)
// without a protobuf serialize/parse round trip. This is also the seam an
// embedder implements to plug in its own ONNX runtime (see the C ABI executor
// callback in capi/onnxsim_c_api.h, and docs/dlpack-executor.md).
struct ModelExecutor {
  virtual ~ModelExecutor() = default;

  // Evaluate `model`, whose graph inputs are fed by `inputs` (positional, i.e.
  // inputs[i] feeds model.graph().input(i)), and return one tensor per graph
  // output (positional, matching model.graph().output()).
  //
  // Ownership: `inputs` are BORROWED for the duration of the call -- the
  // executor must not retain them past return. Each returned DLManagedTensorPtr
  // is freshly owned by the caller. Tensors are CPU, contiguous, and in host
  // byte order (raw_data's little-endian layout is converted at the DLPack
  // boundary -- see dlpack_bridge.h).
  //
  // public for pybind11 / nanobind trampolines
  virtual std::vector<DLManagedTensorPtr> Run(
      const onnx::ModelProto& model,
      const std::vector<const DLManagedTensor*>& inputs) const = 0;
};

// A user-supplied whole-graph rewriter. When one is passed to ``Simplify`` it
// is run inside the simplification fixed point, letting Python code (for
// example an
// ``onnxscript.rewriter`` rule set) rewrite the model between the optimizer and
// constant-folding rounds so a rewrite can unlock further simplification and
// vice versa. Passing ``nullptr`` (the default) leaves simplification behaviour
// exactly as before.
struct GraphRewriter {
  virtual ~GraphRewriter() = default;

  // Rewrite ``model`` in place. Returns ``true`` if the model was changed and
  // ``false`` if the rewriter left it untouched -- in the latter case ``model``
  // is not modified, so callers can skip re-copying it. Being able to report
  // "nothing changed" lets a rewriter that matched no rule (for example an
  // ``onnxscript.rewriter`` rule set whose patterns did not fire) avoid parsing
  // and copying a fresh, identical ModelProto back on every fixed-point round.
  // public it for pybind11
  virtual bool _Run(onnx::ModelProto& model) const = 0;
};

void InitEnv();

#ifndef NO_BUILTIN_ORT
// Returns the built-in model executor backed by ONNX Runtime. Only available
// when onnxsim is built with the built-in ONNX Runtime.
std::shared_ptr<const ModelExecutor> GetBuiltinModelExecutor();
#endif

// ``target_opset_version``, when set, converts the model to that opset version
// of the default ONNX domain (using onnx's version converter) before
// simplifying, so the simplifier can clean up any redundant nodes the
// conversion introduces. std::nullopt leaves the opset version unchanged.
// ``initializers_as_constants`` (default true) controls whether graph
// initializers are treated as constant tensors during simplification. With the
// default, initializers are constants: constant folding materializes nodes that
// depend only on them, and the onnx optimizer's value-baking passes (e.g.
// fuse_bn_into_conv) may fold them. When set to false, initializers are treated
// as non-constant, so nodes rooted only at initializers are left in the graph
// and their weights survive simplification as tunable tensors; ``Constant``
// nodes are still treated as constants either way.
// ``include_inline_functions`` (default false) inlines the model's local
// (model-defined) functions into the main graph before simplifying, via onnx's
// inliner. This flattens function calls into plain ops so the optimizer, shape
// inference and constant folding can see through them; schema-defined
// (built-in) functions are left alone. With the default the model's functions
// are left untouched.
onnx::ModelProto Simplify(
    const ModelExecutor& executor, const onnx::ModelProto& model,
    std::optional<std::vector<std::string>> skip_optimizers,
    bool constant_folding, bool shape_inference, size_t tensor_size_threshold,
    std::optional<int> target_opset_version = std::nullopt,
    const GraphRewriter* rewriter = nullptr,
    bool initializers_as_constants = true,
    bool include_inline_functions = false);

// Debugging helpers: run a *single* one of the transforms that ``Simplify``
// otherwise drives to a fixed point, once, on a copy of ``model``, and return
// the result. They let a caller inspect the isolated effect of a step (e.g. the
// WASM converter's "run a single feature" panel) instead of the whole
// fixed-point simplification. The input model is never mutated.
//
// ``InferShapesOnce`` runs ONNX shape inference (populates value_info / output
// types). ``PropagateDataOnce`` runs onnxsim's partial-shape / data-propagation
// pass, which rewrites nodes whose output value became statically known into
// ``Constant`` nodes. ``FoldConstantOnce`` runs the same partial-shape pass and
// then one constant-folding round through ``executor`` (so it needs a model
// executor, exactly like ``Simplify``); ``tensor_size_threshold`` caps the size
// of tensors that folding may materialize, matching ``Simplify``'s parameter.
onnx::ModelProto InferShapesOnce(const onnx::ModelProto& model);
onnx::ModelProto PropagateDataOnce(const onnx::ModelProto& model);
onnx::ModelProto FoldConstantOnce(const ModelExecutor& executor,
                                  const onnx::ModelProto& model,
                                  size_t tensor_size_threshold,
                                  bool initializers_as_constants = true);

void SimplifyPath(const ModelExecutor& executor, const std::string& in_path,
                  const std::string& out_path,
                  std::optional<std::vector<std::string>> skip_optimizers,
                  bool constant_folding, bool shape_inference,
                  size_t tensor_size_threshold,
                  std::optional<int> target_opset_version = std::nullopt,
                  const GraphRewriter* rewriter = nullptr,
                  bool initializers_as_constants = true,
                  bool include_inline_functions = false);
