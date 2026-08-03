/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "custom_optimizer_passes.h"

#include <memory>
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
    opt::RegisterExternalPass(
        std::make_shared<opt::EliminateReshapeAroundElementwise>());
    opt::RegisterExternalPass(std::make_shared<opt::FuseConsecutiveMul>());
    opt::RegisterExternalPass(
        std::make_shared<opt::FuseMatMulAddBiasIntoGemmBatched>());
    opt::RegisterExternalPass(std::make_shared<opt::FuseMulIntoConv>());
  });
}

}  // namespace onnxsim
