#include "pruning_entry.h"

#include <onnx/onnx_pb.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "custom_optimizer_passes.h"
#include "model_prep.h"
#include "onnxoptimizer/optimize.h"
#include "passes/magnitude_pruning.h"

onnx::ModelProto PruneMagnitude(const onnx::ModelProto& model, double sparsity) {
  if (!(sparsity >= 0.0 && sparsity < 1.0)) {
    throw std::invalid_argument("PruneMagnitude: sparsity must be in [0, 1), got " +
                                std::to_string(sparsity));
  }
  PrepareSchemasForDebug(model);
  // Registers magnitude_pruning_matmul/_conv (idempotent) into
  // onnxoptimizer's registry so OptimizeFixed can find them by name below.
  onnxsim::RegisterCustomOptimizerPasses();
  // magnitude_pruning_matmul/_conv read this the same way quantize_fp16
  // reads QuantizeFp16KeepIoTypes() -- OptimizeFixed's pass-name list has no
  // way to carry a parameter directly.
  ONNX_NAMESPACE::optimization::onnxsim_passes::MagnitudePruningSparsity() =
      sparsity;
  return onnx::optimization::OptimizeFixed(
      model, std::vector<std::string>{"magnitude_pruning_matmul",
                                      "magnitude_pruning_conv"});
}
