/*
 * SPDX-License-Identifier: Apache-2.0
 */
#define ONNXSIM_C_API_BUILD
#include "onnxsim_c_api.h"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <optional>
#include <string>
#include <vector>

#include "onnx/proto_utils.h"
#include "onnxoptimizer/optimize.h"
#include "onnxsim.h"

#ifdef NO_BUILTIN_ORT
#error "The onnxsim C API requires the built-in ONNX Runtime (ONNXSIM_BUILTIN_ORT=ON)."
#endif

namespace {

// Duplicate a std::string into a malloc'd, NUL-terminated C string so the
// caller can free it with std::free (via onnxsim_free_string). Returns nullptr
// on allocation failure.
char* DupCString(const std::string& s) {
  char* p = static_cast<char*>(std::malloc(s.size() + 1));
  if (p != nullptr) {
    std::memcpy(p, s.data(), s.size());
    p[s.size()] = '\0';
  }
  return p;
}

void SetError(char** out_error, const std::string& message) {
  if (out_error != nullptr) {
    *out_error = DupCString(message);
  }
}

// Translate the (pointer array, is_null) FFI encoding into the optional vector
// the C++ core expects.
std::optional<std::vector<std::string>> BuildSkipOptimizers(
    const char* const* skip_optimizers, size_t num_skip_optimizers,
    int skip_optimizers_is_null) {
  if (skip_optimizers_is_null != 0) {
    return std::nullopt;
  }
  std::vector<std::string> passes;
  passes.reserve(num_skip_optimizers);
  for (size_t i = 0; i < num_skip_optimizers; ++i) {
    const char* name = skip_optimizers != nullptr ? skip_optimizers[i] : nullptr;
    passes.emplace_back(name != nullptr ? name : "");
  }
  return passes;
}

// A target opset version of <= 0 means "leave the opset version unchanged".
std::optional<int> BuildTargetOpsetVersion(int target_opset_version) {
  if (target_opset_version <= 0) {
    return std::nullopt;
  }
  return target_opset_version;
}

}  // namespace

extern "C" {

OnnxsimStatus onnxsim_simplify(const void* model_data, size_t model_size,
                               const char* const* skip_optimizers,
                               size_t num_skip_optimizers,
                               int skip_optimizers_is_null, int constant_folding,
                               int shape_inference, size_t tensor_size_threshold,
                               int target_opset_version, void** out_data,
                               size_t* out_size, char** out_error) {
  if (out_data != nullptr) {
    *out_data = nullptr;
  }
  if (out_size != nullptr) {
    *out_size = 0;
  }
  if (out_error != nullptr) {
    *out_error = nullptr;
  }
  if (model_data == nullptr || out_data == nullptr || out_size == nullptr) {
    SetError(out_error, "onnxsim_simplify: required argument is NULL");
    return ONNXSIM_ERROR;
  }
  try {
    // Force env initialization to register opsets, matching the Python binding.
    InitEnv();
    onnx::ModelProto model;
    if (!ParseProtoFromBytes(&model, static_cast<const char*>(model_data),
                             model_size)) {
      SetError(out_error, "failed to parse ONNX ModelProto from input bytes");
      return ONNXSIM_ERROR;
    }
    onnx::ModelProto result = Simplify(
        *GetBuiltinModelExecutor(), model,
        BuildSkipOptimizers(skip_optimizers, num_skip_optimizers,
                            skip_optimizers_is_null),
        constant_folding != 0, shape_inference != 0, tensor_size_threshold,
        BuildTargetOpsetVersion(target_opset_version));

    std::string out;
    if (!result.SerializeToString(&out)) {
      SetError(out_error, "failed to serialize the simplified ModelProto");
      return ONNXSIM_ERROR;
    }
    // malloc(0) may return nullptr; hand back a distinct non-null pointer so the
    // caller can always free it and can rely on out_data != NULL on success.
    void* buffer = std::malloc(out.size() != 0 ? out.size() : 1);
    if (buffer == nullptr) {
      SetError(out_error, "out of memory while copying the simplified model");
      return ONNXSIM_ERROR;
    }
    std::memcpy(buffer, out.data(), out.size());
    *out_data = buffer;
    *out_size = out.size();
    return ONNXSIM_OK;
  } catch (const std::exception& e) {
    SetError(out_error, e.what());
    return ONNXSIM_ERROR;
  } catch (...) {
    SetError(out_error, "unknown error while simplifying the model");
    return ONNXSIM_ERROR;
  }
}

OnnxsimStatus onnxsim_simplify_path(const char* in_path, const char* out_path,
                                    const char* const* skip_optimizers,
                                    size_t num_skip_optimizers,
                                    int skip_optimizers_is_null,
                                    int constant_folding, int shape_inference,
                                    size_t tensor_size_threshold,
                                    int target_opset_version,
                                    char** out_error) {
  if (out_error != nullptr) {
    *out_error = nullptr;
  }
  if (in_path == nullptr || out_path == nullptr) {
    SetError(out_error, "onnxsim_simplify_path: in_path/out_path is NULL");
    return ONNXSIM_ERROR;
  }
  try {
    InitEnv();
    SimplifyPath(*GetBuiltinModelExecutor(), in_path, out_path,
                 BuildSkipOptimizers(skip_optimizers, num_skip_optimizers,
                                    skip_optimizers_is_null),
                 constant_folding != 0, shape_inference != 0,
                 tensor_size_threshold,
                 BuildTargetOpsetVersion(target_opset_version));
    return ONNXSIM_OK;
  } catch (const std::exception& e) {
    SetError(out_error, e.what());
    return ONNXSIM_ERROR;
  } catch (...) {
    SetError(out_error, "unknown error while simplifying the model");
    return ONNXSIM_ERROR;
  }
}

char* onnxsim_list_optimizers(void) {
  try {
    std::string joined;
    for (const auto& pass : onnx::optimization::GetFuseAndEliminationPass()) {
      if (!joined.empty()) {
        joined.push_back('\n');
      }
      joined.append(pass);
    }
    return DupCString(joined);
  } catch (...) {
    return nullptr;
  }
}

void onnxsim_free_buffer(void* data) { std::free(data); }

void onnxsim_free_string(char* data) { std::free(data); }

}  // extern "C"
