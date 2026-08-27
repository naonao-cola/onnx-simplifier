// SPDX-License-Identifier: Apache-2.0
//
// Implements onnx_deploy_c_api.h: the swappable-libort C ABI over
// onnx_deploy::KvCachePipeline. See that header for the contract and
// ../../README.md for the design. Built with zero link-time dependency on
// libonnxruntime -- see onnx_deploy_load_ort below.

#define ONNX_DEPLOY_C_API_BUILD
#include "onnx_deploy/onnx_deploy_c_api.h"

#include "onnx_deploy/kv_cache_pipeline.h"

#include <cstring>
#include <exception>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

char* DupCString(const std::string& s) {
  char* out = static_cast<char*>(std::malloc(s.size() + 1));
  if (!out) return nullptr;
  std::memcpy(out, s.data(), s.size());
  out[s.size()] = '\0';
  return out;
}

void SetError(char** out_error, const std::string& msg) {
  if (out_error) *out_error = DupCString(msg);
}

// Resolves libonnxruntime's OrtGetApiBase export at runtime and wires the
// result into Ort::* (see kv_cache_pipeline.h's ORT_API_MANUAL_INIT usage
// contract). This is the whole "swappable libort" mechanism: no symbol in
// this file, or in kv_cache_pipeline.h, is resolved against libonnxruntime
// at link time -- only header type/inline-wrapper definitions are used, so
// nothing here requires -lonnxruntime at all.
using OrtGetApiBaseFn = const OrtApiBase* (*)();

}  // namespace

extern "C" OnnxDeployStatus onnx_deploy_load_ort(const char* libort_path, char** out_error) {
  try {
    OrtGetApiBaseFn get_api_base = nullptr;
#if defined(_WIN32)
    HMODULE handle = ::LoadLibraryA(libort_path);
    if (!handle) {
      SetError(out_error, std::string("LoadLibrary failed for ") + libort_path);
      return ONNX_DEPLOY_ERROR;
    }
    get_api_base = reinterpret_cast<OrtGetApiBaseFn>(::GetProcAddress(handle, "OrtGetApiBase"));
#else
    void* handle = dlopen(libort_path, RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
      SetError(out_error, std::string("dlopen failed for ") + libort_path + ": " + dlerror());
      return ONNX_DEPLOY_ERROR;
    }
    get_api_base = reinterpret_cast<OrtGetApiBaseFn>(dlsym(handle, "OrtGetApiBase"));
#endif
    if (!get_api_base) {
      SetError(out_error, std::string(libort_path) + " has no OrtGetApiBase export -- not a libonnxruntime build?");
      return ONNX_DEPLOY_ERROR;
    }
    const OrtApiBase* api_base = get_api_base();
    if (!api_base) {
      SetError(out_error, "OrtGetApiBase() returned NULL");
      return ONNX_DEPLOY_ERROR;
    }
    const OrtApi* api = api_base->GetApi(ORT_API_VERSION);
    if (!api) {
      SetError(out_error,
               "OrtApiBase::GetApi(ORT_API_VERSION=" + std::to_string(ORT_API_VERSION) +
                   ") returned NULL -- this header's ORT_API_VERSION is newer than what libort_path implements");
      return ONNX_DEPLOY_ERROR;
    }
    Ort::InitApi(api);
    return ONNX_DEPLOY_OK;
  } catch (const std::exception& e) {
    SetError(out_error, e.what());
    return ONNX_DEPLOY_ERROR;
  } catch (...) {
    SetError(out_error, "unknown error in onnx_deploy_load_ort");
    return ONNX_DEPLOY_ERROR;
  }
}

struct OnnxDeployPipeline {
  Ort::Env env;
  onnx_deploy::KvCachePipeline pipeline;
  OnnxDeployPipeline(const std::string& model_dir, const onnx_deploy::PipelineOptions& pipeline_options)
      : env(ORT_LOGGING_LEVEL_WARNING, "onnx-deploy"), pipeline(env, model_dir, pipeline_options) {}
};

extern "C" OnnxDeployPipeline* onnx_deploy_create(const char* model_dir, char** out_error) {
  return onnx_deploy_create_ex(model_dir, "cpu", 0, out_error);
}

extern "C" OnnxDeployPipeline* onnx_deploy_create_ex(const char* model_dir, const char* execution_provider,
                                                      int cuda_device_id, char** out_error) {
  try {
    onnx_deploy::PipelineOptions pipeline_options;
    std::string ep = execution_provider ? execution_provider : "cpu";
    if (ep == "cpu") {
      pipeline_options.execution_provider = onnx_deploy::PipelineOptions::ExecutionProvider::kCpu;
    } else if (ep == "cuda") {
      pipeline_options.execution_provider = onnx_deploy::PipelineOptions::ExecutionProvider::kCuda;
      pipeline_options.cuda_device_id = cuda_device_id;
    } else if (ep == "webgpu") {
      pipeline_options.execution_provider = onnx_deploy::PipelineOptions::ExecutionProvider::kWebGpu;
    } else {
      SetError(out_error,
               "onnx_deploy_create_ex: unknown execution_provider '" + ep + "' (expected \"cpu\", \"cuda\", or \"webgpu\")");
      return nullptr;
    }
    return new OnnxDeployPipeline(model_dir ? model_dir : "", pipeline_options);
  } catch (const std::exception& e) {
    SetError(out_error, e.what());
    return nullptr;
  } catch (...) {
    SetError(out_error, "unknown error in onnx_deploy_create_ex");
    return nullptr;
  }
}

extern "C" void onnx_deploy_destroy(OnnxDeployPipeline* pipeline) { delete pipeline; }

extern "C" int onnx_deploy_is_seq2seq(const OnnxDeployPipeline* pipeline) {
  return (pipeline && pipeline->pipeline.is_seq2seq()) ? 1 : 0;
}

extern "C" OnnxDeployStatus onnx_deploy_generate(OnnxDeployPipeline* pipeline, const int64_t* input_ids,
                                                  size_t num_input_ids, int64_t max_new_tokens, int64_t eos_token_id,
                                                  int64_t decoder_start_token_id, int64_t** out_ids,
                                                  size_t* out_count, char** out_error) {
  if (!pipeline || !input_ids || !out_ids || !out_count) {
    SetError(out_error, "onnx_deploy_generate: null argument");
    return ONNX_DEPLOY_ERROR;
  }
  try {
    onnx_deploy::GenerationConfig config;
    config.max_new_tokens = max_new_tokens;
    config.eos_token_id = eos_token_id;
    config.decoder_start_token_id = decoder_start_token_id;

    std::vector<int64_t> ids(input_ids, input_ids + num_input_ids);
    std::vector<int64_t> generated = pipeline->pipeline.Generate(ids, config);

    int64_t* buf = generated.empty() ? nullptr : static_cast<int64_t*>(std::malloc(generated.size() * sizeof(int64_t)));
    if (!generated.empty() && !buf) {
      SetError(out_error, "onnx_deploy_generate: allocation failure");
      return ONNX_DEPLOY_ERROR;
    }
    if (buf) std::memcpy(buf, generated.data(), generated.size() * sizeof(int64_t));
    *out_ids = buf;
    *out_count = generated.size();
    return ONNX_DEPLOY_OK;
  } catch (const std::exception& e) {
    SetError(out_error, e.what());
    return ONNX_DEPLOY_ERROR;
  } catch (...) {
    SetError(out_error, "unknown error in onnx_deploy_generate");
    return ONNX_DEPLOY_ERROR;
  }
}

extern "C" void onnx_deploy_free_ids(int64_t* ids) { std::free(ids); }

extern "C" void onnx_deploy_free_string(char* data) { std::free(data); }
