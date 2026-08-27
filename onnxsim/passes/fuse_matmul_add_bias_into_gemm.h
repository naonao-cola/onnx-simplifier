// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

// Before:
//   Z = MatMul(X, Y)
//   A = Z + Bias
// After:
//   A = Gemm(X, Y, Bias)
//
// the pass can handle the case when:
//   case 1: Bias is 1D tensor and Bias.dim[0] == Z.dim[1]
//   case 2: Bias is 2D tensor and Bias.dim[0] == Z.dim[0] or 1
//           and Bias.dim[1] = Z.dim[1]
//
// This is onnxsim's patched version of onnx-optimizer's built-in pass of the
// same name (RegisterOrReplace overwrites the registry entry -- see
// custom_optimizer_passes.cpp). The only difference from upstream is the
// added FLOAT-only guard below, active for the default
// GemmFusionBackend::kOrtCpu (see gemm_fusion_backend.h): ONNX Runtime's CPU
// execution provider has no fast Gemm kernel for FLOAT16 (it falls back to a
// naive/reference path), while its MatMul kernel does -- fusing MatMul+Add
// into Gemm for a FLOAT16 operand was measured making that op ~70x *slower*
// to actually run (a K=1024, N=4096 case: ~110ms for MatMul+Add vs. ~7.8s
// for the fused Gemm), silently turning this "optimization" into a severe
// regression for exactly the kind of fp16 codec/vocoder graphs the Audio8
// model set covers (see tests/test_audio8.py's codec_decoder_fp16.onnx
// case). Bailing out here leaves the original MatMul+Add in place for
// FLOAT16 (and any other non-FLOAT type) under that default, matching
// upstream's behavior for the FLOAT32 case (always a speedup on ORT CPU) and
// for every dtype when GemmFusionBackend::kUnrestricted is selected -- a
// deployment target other than ORT CPU may have a perfectly good FP16 Gemm
// kernel and benefit from this fusion the same way FLOAT32 does.

#include <numeric>

#include "gemm_fusion_backend.h"
#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {
// onnxsim's own passes live in this nested namespace so their class
// names never collide (ODR) with the same-named passes compiled into
// onnxoptimizer; RegisterOrReplace still keys them by getPassName().
namespace onnxsim_passes {

struct FuseMatMulAddBiasIntoGemm final : public PredicateBasedPass {
  explicit FuseMatMulAddBiasIntoGemm()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "fuse_matmul_add_bias_into_gemm";
  }
  bool patternMatchPredicate(Node* node) override {
    if (!CheckKind(node, kAdd, 0, kMatMul)) {
      return false;
    }
    if (onnxsim::GetGemmFusionBackend() !=
        onnxsim::GemmFusionBackend::kOrtCpu) {
      return true;
    }
    // See the file comment: ONNX Runtime's CPU Gemm kernel has no fast path
    // for FLOAT16 (or any non-FLOAT32 type), so fusing into Gemm there is a
    // severe runtime regression rather than an optimization.
    return node->inputs()[0]->node()->inputs()[0]->elemType() ==
           TensorProto_DataType_FLOAT;
  }
  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    // due to current broadcasting's constraint, MatMul has to be the first
    // operand
    destroy_current = NodeDestroyType::DestroyZero;
    auto orig_matmul = n->inputs()[0];
    auto orig_bias = n->inputs()[1];

    // check if MatMul is only used by Add
    if (orig_matmul->uses().size() > 1) {
      return false;
    }
    auto x_shape = orig_matmul->node()->inputs()[0]->sizes();
    auto y_shape = orig_matmul->node()->inputs()[1]->sizes();
    int64_t z_N = -1;
    int64_t z_M = -1;
    // try to get feature N from x_shape
    if (static_cast<int64_t>(x_shape.size()) == 2 && x_shape[0].is_int) {
      z_N = x_shape[0].dim;
    } else {
      return false;
    }
    // try to get feature M from y_shape
    if (static_cast<int64_t>(y_shape.size()) == 2 && y_shape[1].is_int) {
      z_M = y_shape[1].dim;
    } else {
      return false;
    }
    // check if bias_shape is compatible
    auto bias_shape = orig_bias->sizes();
    auto bias_dim = static_cast<int64_t>(bias_shape.size());
    int64_t bias_N = -1;
    int64_t bias_M = -1;
    if (bias_dim == 1 && bias_shape[0].is_int) {
      bias_N = 1;
      bias_M = bias_shape[0].dim;
    } else if (bias_dim == 2 && bias_shape[0].is_int && bias_shape[1].is_int) {
      bias_N = bias_shape[0].dim;
      bias_M = bias_shape[1].dim;
    } else {
      return false;
    }
    if ((bias_N != z_N && bias_N != 1) || bias_M != z_M) {
      return false;
    }
    // proceed to fuse MatMul and Add into Gemm
    Node* gemm =
        graph.create(kGemm, orig_matmul->node()->inputs(), n->outputs().size());
    gemm->addInput(n->inputs()[1]);
    for (int i = 0; i < static_cast<int64_t>(gemm->outputs().size()); ++i) {
      gemm->outputs()[i]->copyMetadata(n->outputs()[i]);
    }
    gemm->f_(kalpha, 1.0);
    gemm->f_(kbeta, 1.0);
    gemm->i_(ktransA, 0);
    gemm->i_(ktransB, 0);
    gemm->insertBefore(n);
    const bool replacing_success = tryReplacingAllUsesWith(n, gemm);
    if (!replacing_success) {
      return false;
    }
    // only destroy MatMul here and DCE will take care of the Add
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
