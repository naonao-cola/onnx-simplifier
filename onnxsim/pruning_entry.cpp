#include "pruning_entry.h"

#include <onnx/onnx_pb.h>

#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "custom_optimizer_passes.h"
#include "model_prep.h"
#include "onnxoptimizer/optimize.h"
#include "passes/magnitude_pruning.h"

onnx::ModelProto PruneMagnitude(const onnx::ModelProto& model, double sparsity,
                                const std::optional<int64_t>& n,
                                const std::optional<int64_t>& m,
                                bool global_sparsity) {
  // Mirrors pruning.py's own `_validate_pattern` exactly: n/m must be given
  // together (N:M mode) or not at all, and `sparsity` is only validated when
  // n/m are absent (it is simply unused, unvalidated, in N:M mode).
  if (n.has_value() != m.has_value()) {
    throw std::invalid_argument(
        "PruneMagnitude: n and m must be given together (N:M pruning) or "
        "not at all");
  }
  if (n.has_value()) {
    if (!(*n > 0 && *n <= *m)) {
      throw std::invalid_argument(
          "PruneMagnitude: require 0 < n <= m, got n=" + std::to_string(*n) +
          ", m=" + std::to_string(*m));
    }
  } else if (!(sparsity >= 0.0 && sparsity < 1.0)) {
    throw std::invalid_argument(
        "PruneMagnitude: sparsity must be in [0, 1), got " +
        std::to_string(sparsity));
  }
  if (global_sparsity && n.has_value()) {
    throw std::invalid_argument(
        "PruneMagnitude: global_sparsity is not supported together with "
        "N:M pruning (n/m) -- see PruneMagnitude's own declaration comment");
  }

  PrepareSchemasForDebug(model);
  // Registers magnitude_pruning_matmul/_conv/_attention/_global (idempotent)
  // into onnxoptimizer's registry so OptimizeFixed can find them by name
  // below.
  onnxsim::RegisterCustomOptimizerPasses();
  // These read the statics below the same way quantize_fp16 reads
  // QuantizeFp16KeepIoTypes() -- OptimizeFixed's pass-name list has no way to
  // carry a parameter directly.
  namespace mp = ONNX_NAMESPACE::optimization::onnxsim_passes;
  mp::MagnitudePruningSparsity() = sparsity;
  mp::MagnitudePruningN() = n.value_or(-1);
  mp::MagnitudePruningM() = m.value_or(-1);

  if (global_sparsity) {
    return onnx::optimization::OptimizeFixed(
        model, std::vector<std::string>{"magnitude_pruning_global"});
  }
  return onnx::optimization::OptimizeFixed(
      model, std::vector<std::string>{"magnitude_pruning_matmul",
                                      "magnitude_pruning_conv",
                                      "magnitude_pruning_attention"});
}
