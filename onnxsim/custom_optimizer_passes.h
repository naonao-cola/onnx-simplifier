/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace onnxsim {

// Register onnxsim's own optimizer passes into onnxoptimizer's global pass
// registry so they can be selected by name through Optimize/OptimizeFixed and
// appear in GetAvailablePasses / GetFuseAndEliminationPass, exactly like the
// passes that ship with onnxoptimizer.
//
// These four passes are onnxsim-specific graph rewrites that used to live in
// onnxsim's onnxoptimizer fork. They are defined under onnxsim/passes/ and
// injected directly into onnxoptimizer's existing global pass registry
// (onnx::optimization::Optimizer::passes), so onnxoptimizer needs no change:
//
//   - fuse_mul_into_conv
//   - fuse_consecutive_mul
//   - fuse_matmul_add_bias_into_gemm_batched
//   - eliminate_reshape_around_elementwise
//
// The registration runs at most once per process and is safe to call from any
// simplification entry point. It MUST run before the pass list is built (a
// couple of these passes are auto-included via GetFuseAndEliminationPass) and
// before OptimizeFixed is invoked, so the simplification entry points call it
// up front.
void RegisterCustomOptimizerPasses();

}  // namespace onnxsim
