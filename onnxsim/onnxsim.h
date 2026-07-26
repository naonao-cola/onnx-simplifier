#pragma once

#include <memory>
#include <optional>
#include <vector>

#include <onnx/onnx_pb.h>

struct ModelExecutor {
  virtual ~ModelExecutor() = default;

  // public it for pybind11
  virtual std::vector<onnx::TensorProto> _Run(
      const onnx::ModelProto& model,
      const std::vector<onnx::TensorProto>& inputs) const = 0;
};

// A user-supplied whole-graph rewriter. When one is passed to ``Simplify`` it is
// run inside the simplification fixed point, letting Python code (for example an
// ``onnxscript.rewriter`` rule set) rewrite the model between the optimizer and
// constant-folding rounds so a rewrite can unlock further simplification and
// vice versa. Passing ``nullptr`` (the default) leaves simplification behaviour
// exactly as before.
struct GraphRewriter {
  virtual ~GraphRewriter() = default;

  // public it for pybind11
  virtual onnx::ModelProto _Run(const onnx::ModelProto& model) const = 0;
};

void InitEnv();

#ifndef NO_BUILTIN_ORT
// Returns the built-in model executor backed by ONNX Runtime. Only available
// when onnxsim is built with the built-in ONNX Runtime.
std::shared_ptr<const ModelExecutor> GetBuiltinModelExecutor();
#endif

onnx::ModelProto Simplify(
    const ModelExecutor& executor, const onnx::ModelProto& model,
    std::optional<std::vector<std::string>> skip_optimizers,
    bool constant_folding, bool shape_inference, size_t tensor_size_threshold,
    const GraphRewriter* rewriter = nullptr);

void SimplifyPath(const ModelExecutor& executor, const std::string& in_path,
                  const std::string& out_path,
                  std::optional<std::vector<std::string>> skip_optimizers,
                  bool constant_folding, bool shape_inference,
                  size_t tensor_size_threshold,
                  const GraphRewriter* rewriter = nullptr);
