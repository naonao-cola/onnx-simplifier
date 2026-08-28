#pragma once

// Which runtime onnxsim should assume will execute the *simplified* model,
// for the one pass heuristic whose right answer depends on that runtime's
// actual performance characteristics rather than pure graph semantics:
// fuse_matmul_add_bias_into_gemm(_batched)'s MatMul+Add -> Gemm fusion (see
// those passes' own file comments). ONNX Runtime's CPU execution provider
// has no fast Gemm kernel for FLOAT16 -- it falls back to a naive path, made
// the fusion a measured ~70x slowdown rather than a speedup -- but that is a
// property of ORT's CPU EP specifically, not of the ONNX graph: a different
// runtime, or ORT on a different execution provider, may have a perfectly
// good FP16 Gemm kernel and benefit from the same fusion FLOAT32 already
// does everywhere.

#include <string>

namespace onnxsim {

enum class GemmFusionBackend {
  // Default: assume ONNX Runtime's CPU execution provider, and restrict
  // MatMul+Add -> Gemm fusion to FLOAT32 operands accordingly.
  kOrtCpu,
  // No restriction: fuse regardless of operand dtype, matching upstream
  // onnx-optimizer's original (backend-agnostic) behavior. Pick this when
  // the simplified model will run somewhere ORT CPU's FP16 Gemm slowness
  // does not apply.
  kUnrestricted,
};

// Process-global, read by fuse_matmul_add_bias_into_gemm(_batched)'s
// patternMatchPredicate on every match attempt -- not thread-safe against a
// concurrent Simplify() call requesting a different backend, matching this
// codebase's other cross-cutting pass toggles (e.g. onnx-optimizer's
// SetPassPhaseProfilingEnabled). SimplifyImpl sets it once, from the
// ONNXSIM_GEMM_FUSION_BACKEND environment variable, at the start of every
// Simplify()/SimplifyConsumeInput() call, defaulting to kOrtCpu when unset --
// so it never carries state over from an unrelated prior call in the same
// process.
void SetGemmFusionBackend(GemmFusionBackend backend);
GemmFusionBackend GetGemmFusionBackend();

// Parses the binding-facing string spelling ("ort_cpu" / "unrestricted").
// Throws std::invalid_argument on an unrecognized name.
GemmFusionBackend ParseGemmFusionBackend(const std::string& name);

}  // namespace onnxsim
