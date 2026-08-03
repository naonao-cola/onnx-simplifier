/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "custom_optimizer_passes.h"

#include <mutex>

#include "onnxoptimizer/optimize.h"
#include "passes/eliminate_reshape_around_elementwise.h"
#include "passes/fuse_consecutive_mul.h"
#include "passes/fuse_matmul_add_bias_into_gemm_batched.h"
#include "passes/fuse_mul_into_conv.h"

namespace onnxsim {

void RegisterCustomOptimizerPasses() {
  static std::once_flag flag;
  std::call_once(flag, [] {
    namespace opt = ONNX_NAMESPACE::optimization;
    // Add onnxsim's passes into onnxoptimizer's existing global registry
    // directly (registerPass<T> constructs and registers one instance), so no
    // change to onnxoptimizer itself is required. call_once keeps this a
    // one-time write, matching the registry's static, read-only-after-init use.
    auto& registry = opt::Optimizer::passes;
    registry.registerPass<opt::EliminateReshapeAroundElementwise>();
    registry.registerPass<opt::FuseConsecutiveMul>();
    registry.registerPass<opt::FuseMatMulAddBiasIntoGemmBatched>();
    registry.registerPass<opt::FuseMulIntoConv>();
  });
}

}  // namespace onnxsim
