// SPDX-License-Identifier: Apache-2.0
//
// Compiled nanobind binding over the onnx_deploy C ABI (onnx_deploy_c_api.h)
// -- not over kv_cache_pipeline.h directly, so this file has no ORT header
// dependency at all; it only needs the plain-C onnx_deploy_c_api.h and links
// against the already-built onnx_deploy_c shared library. See ../README.md
// for why this exists: a compiled, opaque alternative to
// optimum.onnxruntime's pure-Python encoder/decoder/KV-cache generate() loop.

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "onnx_deploy/onnx_deploy_c_api.h"

namespace nb = nanobind;

namespace {

[[noreturn]] void ThrowFromError(const char* what, char* err) {
  std::string msg = std::string(what) + ": " + (err ? err : "(no message)");
  onnx_deploy_free_string(err);
  throw std::runtime_error(msg);
}

void LoadOrt(const std::string& libort_path) {
  char* err = nullptr;
  if (onnx_deploy_load_ort(libort_path.c_str(), &err) != ONNX_DEPLOY_OK) ThrowFromError("load_ort", err);
}

class Pipeline {
 public:
  explicit Pipeline(const std::string& model_dir) {
    char* err = nullptr;
    handle_ = onnx_deploy_create(model_dir.c_str(), &err);
    if (!handle_) ThrowFromError("Pipeline", err);
  }
  ~Pipeline() { onnx_deploy_destroy(handle_); }
  Pipeline(const Pipeline&) = delete;
  Pipeline& operator=(const Pipeline&) = delete;

  bool is_seq2seq() const { return onnx_deploy_is_seq2seq(handle_) != 0; }

  std::vector<int64_t> generate(const std::vector<int64_t>& input_ids, int64_t max_new_tokens, int64_t eos_token_id,
                                 int64_t decoder_start_token_id) {
    int64_t* out_ids = nullptr;
    size_t out_count = 0;
    char* err = nullptr;
    OnnxDeployStatus status = onnx_deploy_generate(handle_, input_ids.data(), input_ids.size(), max_new_tokens,
                                                    eos_token_id, decoder_start_token_id, &out_ids, &out_count, &err);
    if (status != ONNX_DEPLOY_OK) ThrowFromError("generate", err);
    std::vector<int64_t> result(out_ids, out_ids + out_count);
    onnx_deploy_free_ids(out_ids);
    return result;
  }

 private:
  OnnxDeployPipeline* handle_ = nullptr;
};

}  // namespace

NB_MODULE(onnx_deploy_py, m) {
  m.doc() =
      "Compiled Python binding over onnx_deploy's C ABI: the same "
      "swappable-libort encoder/decoder KV-cache generation pipeline the "
      "onnx-deploy CLI uses, callable from Python without going through "
      "optimum.onnxruntime's pure-Python generate() loop.";

  m.def("load_ort", &LoadOrt, nb::arg("libort_path"),
        "Load libonnxruntime from libort_path (via dlopen/LoadLibrary) and wire it "
        "up for every Pipeline in this process. Call once, before constructing any "
        "Pipeline.");

  nb::class_<Pipeline>(m, "Pipeline")
      .def(nb::init<const std::string&>(), nb::arg("model_dir"),
           "Loads an optimum-onnx export directory (see ../README.md for the expected "
           "file shape). load_ort() must have already succeeded.")
      .def_prop_ro("is_seq2seq", &Pipeline::is_seq2seq)
      .def("generate", &Pipeline::generate, nb::arg("input_ids"), nb::arg("max_new_tokens") = 32,
           nb::arg("eos_token_id") = -1, nb::arg("decoder_start_token_id") = 0,
           "Greedy-decode up to max_new_tokens ids (batch size 1), stopping early if a "
           "generated id equals eos_token_id (-1 disables early stop). Returns the newly "
           "generated ids, not including the prompt.");
}
