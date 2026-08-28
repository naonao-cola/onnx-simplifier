#include "cross_layer_equalization_entry.h"

#include <onnx/onnx_pb.h>

#include <vector>

#include "custom_optimizer_passes.h"
#include "model_prep.h"
#include "onnxoptimizer/optimize.h"

onnx::ModelProto CrossLayerEqualize(const onnx::ModelProto& model) {
  PrepareSchemasForDebug(model);
  // Registers cross_layer_equalization (idempotent) into onnxoptimizer's
  // registry so OptimizeFixed can find it by name below.
  onnxsim::RegisterCustomOptimizerPasses();
  return onnx::optimization::OptimizeFixed(
      model, std::vector<std::string>{"cross_layer_equalization"});
}
