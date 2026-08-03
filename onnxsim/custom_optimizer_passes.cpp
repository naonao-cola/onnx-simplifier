/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "custom_optimizer_passes.h"

#include <memory>
#include <mutex>
#include <string>

#include "onnxoptimizer/optimize.h"
#include "passes/eliminate_reshape_around_elementwise.h"
#include "passes/fuse_consecutive_mul.h"
#include "passes/fuse_matmul_add_bias_into_gemm_batched.h"
#include "passes/fuse_mul_into_conv.h"

namespace onnxsim {

namespace {

// Register onnxsim's pass T into onnxoptimizer's global registry, replacing any
// existing pass of the same name. onnxoptimizer's own registerPass<T> always
// appends to pass_names, so reusing it to overwrite a name would list -- and
// therefore run -- that pass twice. We touch the registry's public members
// directly instead: overwrite the map entry, and add the name only when it is
// new. This is what lets onnxsim ship its own version of an onnxoptimizer pass
// (e.g. a bug-fixed one) and have it win over the built-in of the same name.
template <typename T>
void RegisterOrReplace(ONNX_NAMESPACE::optimization::GlobalPassRegistry& reg) {
  auto pass = std::make_shared<T>();
  const std::string name = pass->getPassName();
  if (reg.passes.find(name) == reg.passes.end()) {
    reg.pass_names.emplace_back(name);
  }
  reg.passes[name] = std::move(pass);
}

}  // namespace

void RegisterCustomOptimizerPasses() {
  static std::once_flag flag;
  std::call_once(flag, [] {
    namespace opt = ONNX_NAMESPACE::optimization;
    // Inject onnxsim's passes into onnxoptimizer's existing global registry,
    // overwriting any built-in of the same name. call_once keeps this a
    // one-time write, matching the registry's static, read-only-after-init use.
    auto& registry = opt::Optimizer::passes;
    RegisterOrReplace<opt::EliminateReshapeAroundElementwise>(registry);
    RegisterOrReplace<opt::FuseConsecutiveMul>(registry);
    RegisterOrReplace<opt::FuseMatMulAddBiasIntoGemmBatched>(registry);
    RegisterOrReplace<opt::FuseMulIntoConv>(registry);
  });
}

}  // namespace onnxsim
