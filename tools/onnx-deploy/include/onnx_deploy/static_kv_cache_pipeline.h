// onnx_deploy/static_kv_cache_pipeline.h
//
// Generic C++ glue for the fixed-buffer ("static") KV-cache ONNX export
// shape `onnxsim.export_causal_lm_static_cache` produces (see that
// function's own docstring in onnxsim/transformers_export.py for the full
// rationale) --
//   prefill.onnx   (variable prompt length, empty cache)
//   decode.onnx    (one new token, cache partially filled)
// -- decoder-only causal LMs only (no encoder/cross-attention -- that's
// KvCachePipeline's job, in kv_cache_pipeline.h, for optimum-onnx's own
// growing-cache export shape).
//
// This is a DIFFERENT pipeline than KvCachePipeline, not an extension of
// it, because the two export shapes need genuinely different drivers:
// KvCachePipeline's `present.*` output for step N becomes the *entire*
// `past_key_values.*` input for step N+1 (a full tensor handoff, since the
// tensor itself grows every step -- see that header's own comment). Here,
// past_key.{i}/past_value.{i} are fixed-size (batch, kv_heads, max_cache_len,
// head_dim) buffers allocated ONCE per Generate() call and reused in place
// for every step -- prefill and every decode step read AND write the exact
// same buffers, via `Ort::IoBinding` binding the identical `Ort::Value` as
// both the "past_key.{i}" input and the "present_key.{i}" output. That
// tells ONNX Runtime to write each step's result directly into the buffer
// we already own, instead of allocating a fresh (batch, kv_heads,
// max_cache_len, head_dim) tensor and copying it back -- eliminating the
// per-step reallocation-and-copy KvCachePipeline's growing-cache design
// requires (see that header's own docstring). Whether ONNX Runtime's own
// ScatterND kernel avoids an *internal* copy too when its `data` input and
// output alias the same buffer is up to ORT's own kernel/memory-planner
// implementation, not something this code can force further -- what this
// pipeline guarantees is that the *caller-visible* buffer never moves and
// is never reallocated across a generation.
//
// Usage contract: identical to kv_cache_pipeline.h's own -- built against
// ONNX Runtime's C++ API with ORT_API_MANUAL_INIT, so this header does not
// require linking against libonnxruntime; the embedder calls
// `Ort::InitApi(api)` exactly once before constructing anything here,
// exactly as kv_cache_pipeline.h documents. This header reuses that one's
// `PipelineOptions` and `onnx_deploy::detail` helpers rather than
// duplicating them -- include kv_cache_pipeline.h first (or let this header
// pull it in, since it includes it itself below).
#pragma once

#include "onnx_deploy/kv_cache_pipeline.h"

#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace onnx_deploy {

struct StaticKvCacheGenerationConfig {
  int64_t eos_token_id = -1;  // -1 disables early stop
  int64_t max_new_tokens = 32;
};

// Loads and drives one export_causal_lm_static_cache() export directory's
// worth of sessions (prefill.onnx + decode.onnx), batch size 1.
class StaticKvCachePipeline {
 public:
  // model_dir must contain prefill.onnx and decode.onnx (see this header's
  // own top comment). max_cache_len must match the value
  // export_causal_lm_static_cache() was called with -- it sizes every KV
  // buffer this pipeline allocates; there is no way to recover it from the
  // ONNX files themselves, since past_key.0's own declared shape already
  // has that dimension baked in as a fixed size (which this constructor
  // does cross-check against the caller-supplied value, so a mismatch
  // fails fast here rather than at the first out-of-range Run()).
  StaticKvCachePipeline(Ort::Env& env, const std::string& model_dir, int64_t max_cache_len,
                        const PipelineOptions& pipeline_options = {})
      : env_(env), max_cache_len_(max_cache_len) {
    namespace fs = std::filesystem;
    auto load = [&](const std::string& filename) -> std::unique_ptr<Ort::Session> {
      fs::path p = fs::path(model_dir) / filename;
      if (!fs::exists(p)) return nullptr;
      Ort::SessionOptions options = detail::BuildSessionOptions(pipeline_options);
      return std::make_unique<Ort::Session>(env_, detail::ToOrtPath(p.string()).c_str(), options);
    };

    prefill_session_ = load("prefill.onnx");
    decode_session_ = load("decode.onnx");
    if (!prefill_session_ || !decode_session_) {
      throw std::runtime_error(model_dir +
                                ": expected prefill.onnx and decode.onnx (the "
                                "export_causal_lm_static_cache export shape -- "
                                "see static_kv_cache_pipeline.h)");
    }

    prefill_input_names_ = detail::InputNames(*prefill_session_, allocator_);
    prefill_output_names_ = detail::OutputNames(*prefill_session_, allocator_);
    decode_input_names_ = detail::InputNames(*decode_session_, allocator_);
    decode_output_names_ = detail::OutputNames(*decode_session_, allocator_);

    // Derive layer count / kv head count / head dim from decode.onnx's own
    // declared past_key.{i}/past_value.{i} input shapes -- purely by
    // counting and inspecting names, the same "no architecture-specific
    // knowledge baked in" idiom KvCachePipeline's own HarvestPresentIntoCache
    // uses (see that header). Every "past_key.N" name present names one
    // layer; its shape (1, kv_heads, max_cache_len, head_dim) gives
    // everything else.
    num_layers_ = 0;
    for (const auto& name : decode_input_names_) {
      if (name.rfind("past_key.", 0) == 0) ++num_layers_;
    }
    if (num_layers_ == 0) {
      throw std::runtime_error(model_dir + ": decode.onnx has no past_key.* inputs");
    }
    Ort::TypeInfo type_info = decode_session_->GetInputTypeInfo(IndexOfInput(decode_input_names_, "past_key.0"));
    auto shape = type_info.GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() != 4) {
      throw std::runtime_error(model_dir + ": past_key.0 is not rank 4");
    }
    num_kv_heads_ = shape[1];
    head_dim_ = shape[3];
    if (shape[2] > 0 && shape[2] != max_cache_len) {
      throw std::runtime_error(model_dir + ": past_key.0's declared cache length (" +
                                std::to_string(shape[2]) + ") doesn't match max_cache_len (" +
                                std::to_string(max_cache_len) + ")");
    }
  }

  // Greedy decode, batch size 1. `input_ids` is the full prompt. Returns
  // the newly generated token ids (not including the prompt).
  //
  // Precondition: input_ids.size() + config.max_new_tokens <= max_cache_len
  // -- there is no bounds checking here (a violation silently wraps around
  // via the underlying TensorScatter/ScatterND overwrite semantics rather
  // than erroring; see onnxsim's rewrite_tensor_scatter_to_scatter_elements
  // pass and its own doc comment on why `Mod` is applied unconditionally).
  std::vector<int64_t> Generate(const std::vector<int64_t>& input_ids, const StaticKvCacheGenerationConfig& config) {
    const int64_t prompt_len = static_cast<int64_t>(input_ids.size());

    // Allocate every layer's key/value buffer ONCE, zero-initialized, and
    // keep it alive (and at the same address) for this whole Generate()
    // call -- this is the buffer every session.Run() below reads from AND
    // writes back into, via IoBinding, rather than a fresh tensor per step.
    std::vector<int64_t> kv_shape = {1, num_kv_heads_, max_cache_len_, head_dim_};
    const size_t kv_elems = static_cast<size_t>(num_kv_heads_ * max_cache_len_ * head_dim_);
    std::vector<std::vector<float>> key_buffers(static_cast<size_t>(num_layers_));
    std::vector<std::vector<float>> value_buffers(static_cast<size_t>(num_layers_));
    std::vector<Ort::Value> key_values;
    std::vector<Ort::Value> value_values;
    key_values.reserve(static_cast<size_t>(num_layers_));
    value_values.reserve(static_cast<size_t>(num_layers_));
    for (int64_t i = 0; i < num_layers_; ++i) {
      key_buffers[static_cast<size_t>(i)].assign(kv_elems, 0.0f);
      value_buffers[static_cast<size_t>(i)].assign(kv_elems, 0.0f);
      key_values.push_back(Ort::Value::CreateTensor<float>(mem_info_, key_buffers[static_cast<size_t>(i)].data(),
                                                            kv_elems, kv_shape.data(), kv_shape.size()));
      value_values.push_back(Ort::Value::CreateTensor<float>(mem_info_, value_buffers[static_cast<size_t>(i)].data(),
                                                              kv_elems, kv_shape.data(), kv_shape.size()));
    }

    std::vector<int64_t> generated;
    Ort::Value logits = RunStep(*prefill_session_, prefill_input_names_, prefill_output_names_, input_ids,
                                /*cache_start=*/0, key_values, value_values);
    int64_t next_token = ArgmaxLastToken(logits);
    generated.push_back(next_token);
    if (config.eos_token_id >= 0 && next_token == config.eos_token_id) return generated;

    for (int64_t step = 1; step < config.max_new_tokens; ++step) {
      const int64_t cache_start = prompt_len + step - 1;
      logits = RunStep(*decode_session_, decode_input_names_, decode_output_names_, {next_token}, cache_start,
                       key_values, value_values);
      next_token = ArgmaxLastToken(logits);
      generated.push_back(next_token);
      if (config.eos_token_id >= 0 && next_token == config.eos_token_id) break;
    }
    return generated;
  }

 private:
  static size_t IndexOfInput(const std::vector<std::string>& names, const std::string& name) {
    for (size_t i = 0; i < names.size(); ++i)
      if (names[i] == name) return i;
    throw std::runtime_error("expected input '" + name + "' not found");
  }

  // Runs one prefill or decode step, binding every past_key.{i}/
  // past_value.{i} input and present_key.{i}/present_value.{i} output to
  // the SAME persistent buffer (via Ort::IoBinding), and returns "logits"
  // (the only output that isn't a KV-cache buffer, so the only one that
  // still needs a normal, freshly-allocated Ort::Value).
  Ort::Value RunStep(Ort::Session& session, const std::vector<std::string>& in_names,
                      const std::vector<std::string>& out_names, const std::vector<int64_t>& step_input_ids,
                      int64_t cache_start, std::vector<Ort::Value>& key_values,
                      std::vector<Ort::Value>& value_values) {
    const int64_t seq_len = static_cast<int64_t>(step_input_ids.size());
    std::vector<int64_t> ids_copy = step_input_ids;
    std::vector<int64_t> ids_shape = {1, seq_len};
    std::vector<int64_t> cache_position(static_cast<size_t>(seq_len));
    for (int64_t i = 0; i < seq_len; ++i) cache_position[static_cast<size_t>(i)] = cache_start + i;
    std::vector<int64_t> cache_position_shape = {seq_len};

    // Full-buffer-length mask, 1s for real content written so far (through
    // this step's own new tokens), 0s for the not-yet-written tail -- the
    // same convention transformers.StaticCache's own get_mask_sizes()
    // expects (see onnxsim/transformers_export.py's own docstring).
    std::vector<int64_t> attention_mask(static_cast<size_t>(max_cache_len_), 0);
    for (int64_t i = 0; i < cache_start + seq_len && i < max_cache_len_; ++i)
      attention_mask[static_cast<size_t>(i)] = 1;
    std::vector<int64_t> attention_mask_shape = {1, max_cache_len_};

    Ort::IoBinding binding(session);
    for (const auto& name : in_names) {
      if (name == "input_ids") {
        binding.BindInput(name.c_str(), Ort::Value::CreateTensor<int64_t>(mem_info_, ids_copy.data(),
                                                                          ids_copy.size(), ids_shape.data(),
                                                                          ids_shape.size()));
      } else if (name == "cache_position") {
        binding.BindInput(name.c_str(), Ort::Value::CreateTensor<int64_t>(mem_info_, cache_position.data(),
                                                                          cache_position.size(),
                                                                          cache_position_shape.data(),
                                                                          cache_position_shape.size()));
      } else if (name == "attention_mask") {
        binding.BindInput(name.c_str(), Ort::Value::CreateTensor<int64_t>(mem_info_, attention_mask.data(),
                                                                          attention_mask.size(),
                                                                          attention_mask_shape.data(),
                                                                          attention_mask_shape.size()));
      } else if (name.rfind("past_key.", 0) == 0) {
        int64_t layer = std::stoll(name.substr(std::string("past_key.").size()));
        binding.BindInput(name.c_str(), key_values[static_cast<size_t>(layer)]);
      } else if (name.rfind("past_value.", 0) == 0) {
        int64_t layer = std::stoll(name.substr(std::string("past_value.").size()));
        binding.BindInput(name.c_str(), value_values[static_cast<size_t>(layer)]);
      } else {
        throw std::runtime_error("unrecognized static-cache decoder input: " + name +
                                  " (extend StaticKvCachePipeline for this export)");
      }
    }
    for (const auto& name : out_names) {
      if (name.rfind("present_key.", 0) == 0) {
        int64_t layer = std::stoll(name.substr(std::string("present_key.").size()));
        binding.BindOutput(name.c_str(), key_values[static_cast<size_t>(layer)]);
      } else if (name.rfind("present_value.", 0) == 0) {
        int64_t layer = std::stoll(name.substr(std::string("present_value.").size()));
        binding.BindOutput(name.c_str(), value_values[static_cast<size_t>(layer)]);
      } else if (name == "logits") {
        binding.BindOutput(name.c_str(), mem_info_);  // let ORT allocate this one
      } else {
        throw std::runtime_error("unrecognized static-cache decoder output: " + name +
                                  " (extend StaticKvCachePipeline for this export)");
      }
    }

    session.Run(Ort::RunOptions{nullptr}, binding);
    std::vector<Ort::Value> outputs = binding.GetOutputValues();
    std::vector<std::string> bound_out_names = binding.GetOutputNames();
    for (size_t i = 0; i < bound_out_names.size(); ++i)
      if (bound_out_names[i] == "logits") return std::move(outputs[i]);
    throw std::runtime_error("static-cache decoder graph has no 'logits' output");
  }

  // Greedy argmax over the last position's logits. Same technique as
  // KvCachePipeline's own ArgmaxLastToken -- batch size 1 only.
  int64_t ArgmaxLastToken(const Ort::Value& logits) const {
    auto info = logits.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> shape = info.GetShape();  // [batch, seq, vocab]
    int64_t vocab = shape.back();
    int64_t seq = shape.size() >= 2 ? shape[shape.size() - 2] : 1;
    const float* data = logits.GetTensorData<float>();
    const float* last_row = data + (seq - 1) * vocab;
    int64_t best = 0;
    float best_val = -std::numeric_limits<float>::infinity();
    for (int64_t v = 0; v < vocab; ++v) {
      if (last_row[v] > best_val) {
        best_val = last_row[v];
        best = v;
      }
    }
    return best;
  }

  Ort::Env& env_;
  int64_t max_cache_len_;
  int64_t num_layers_ = 0;
  int64_t num_kv_heads_ = 0;
  int64_t head_dim_ = 0;
  std::unique_ptr<Ort::Session> prefill_session_;
  std::unique_ptr<Ort::Session> decode_session_;
  Ort::AllocatorWithDefaultOptions allocator_;
  Ort::MemoryInfo mem_info_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  std::vector<std::string> prefill_input_names_, prefill_output_names_;
  std::vector<std::string> decode_input_names_, decode_output_names_;
};

}  // namespace onnx_deploy
