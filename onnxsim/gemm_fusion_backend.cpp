#include "gemm_fusion_backend.h"

#include <atomic>
#include <stdexcept>

namespace onnxsim {

namespace {
std::atomic<GemmFusionBackend> g_gemm_fusion_backend{
    GemmFusionBackend::kOrtCpu};
}  // namespace

void SetGemmFusionBackend(GemmFusionBackend backend) {
  g_gemm_fusion_backend.store(backend, std::memory_order_relaxed);
}

GemmFusionBackend GetGemmFusionBackend() {
  return g_gemm_fusion_backend.load(std::memory_order_relaxed);
}

GemmFusionBackend ParseGemmFusionBackend(const std::string& name) {
  if (name == "ort_cpu") {
    return GemmFusionBackend::kOrtCpu;
  }
  if (name == "unrestricted") {
    return GemmFusionBackend::kUnrestricted;
  }
  throw std::invalid_argument("unknown gemm_fusion_backend '" + name +
                              "' (expected 'ort_cpu' or 'unrestricted')");
}

}  // namespace onnxsim
