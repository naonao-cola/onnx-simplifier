// onnx_deploy_wasm.cpp
//
// WASM port of onnx_deploy::KvCachePipeline's algorithm (see
// ../../include/onnx_deploy/kv_cache_pipeline.h and ../../README.md), driven
// through JS/onnxruntime-web instead of a native Ort::Session, so "swappable
// libort" in the browser means the JS host picking which onnxruntime-web
// build/version/execution-provider backs Module.onnxDeployRunSession --
// there is no ONNX Runtime C/C++ dependency in this file or its CMakeLists.txt
// at all.
//
// This reuses the SAME two design decisions as the native pipeline --
// present.* -> past_key_values.* renamed purely by string substitution, and
// a cache entry not re-output by a call stays valid for later calls -- but
// against a much simpler tensor representation (WasmTensor, data always as
// std::vector<double>) instead of Ort::Value, since there is no native ORT
// here to hand real typed buffers to. int64 tensor values (token ids, small
// counts) round-trip through `double` exactly for anything under 2^53; this
// is fine for what this pipeline actually carries (ids, KV-cache tensors of
// modest size) and is called out here rather than silently assumed.
//
// The JS/C++ boundary (see ../test/ort_web_runtime.mjs for the reference
// implementation, and Emscripten's ASYNCIFY docs for the val::await()
// mechanism this relies on):
//   Module.onnxDeployCreateSession(modelBytes: Uint8Array)
//     -> Promise<{handle: number, inputNames: string[], outputNames: string[]}>
//   Module.onnxDeployRunSession(handle: number, inputs: TensorObj[], outputNames: string[])
//     -> Promise<TensorObj[]>   // one entry per outputNames, same order
//   where TensorObj = {name: string, dtype: "int64"|"float32", shape: number[], data: number[]}
//
// Scope note: unlike the native/Python layers (a KvCachePipeline object you
// can call .generate() on repeatedly), this exposes ONE async entry point,
// generate(), that creates sessions, runs the whole decode loop, and returns
// -- proving persistent sessions held across many Asyncify-awaited calls
// works (this repo's only prior Asyncify bridge, JsModelExecutor, is
// single-call), without also building a stateful class-lifetime API on top.
// A reusable Pipeline class wrapping multiple generate() calls without
// re-creating sessions each time is a natural follow-up, not done here.

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using emscripten::val;

namespace {

struct WasmTensor {
  std::string dtype;         // "int64" or "float32"
  std::vector<double> shape;
  std::vector<double> data;  // row-major, flattened
};

val TensorToVal(const std::string& name, const WasmTensor& t) {
  val obj = val::object();
  obj.set("name", name);
  obj.set("dtype", t.dtype);
  obj.set("shape", val::array(t.shape.begin(), t.shape.end()));
  obj.set("data", val::array(t.data.begin(), t.data.end()));
  return obj;
}

WasmTensor ValToTensor(const val& obj) {
  WasmTensor t;
  t.dtype = obj["dtype"].as<std::string>();
  t.shape = emscripten::vecFromJSArray<double>(obj["shape"]);
  t.data = emscripten::vecFromJSArray<double>(obj["data"]);
  return t;
}

WasmTensor MakeIdsTensor(const std::vector<int64_t>& ids) {
  WasmTensor t;
  t.dtype = "int64";
  t.shape = {1, static_cast<double>(ids.size())};
  t.data.assign(ids.begin(), ids.end());
  return t;
}

WasmTensor MakeMaskTensor(size_t len) {
  WasmTensor t;
  t.dtype = "int64";
  t.shape = {1, static_cast<double>(len)};
  t.data.assign(len, 1.0);
  return t;
}

struct Session {
  double handle = -1;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
};

// Awaits Module.onnxDeployCreateSession(bytes) and records the returned
// handle + the graph's actual input/output names (mirrors
// kv_cache_pipeline.h's detail::InputNames/OutputNames, which read them off
// a native Ort::Session -- here they come back from the JS-side
// ort.InferenceSession instead).
Session CreateSession(const val& bytes) {
  val result = val::module_property("onnxDeployCreateSession")(bytes).await();
  Session s;
  s.handle = result["handle"].as<double>();
  s.input_names = emscripten::vecFromJSArray<std::string>(result["inputNames"]);
  s.output_names = emscripten::vecFromJSArray<std::string>(result["outputNames"]);
  return s;
}

// Awaits Module.onnxDeployRunSession(handle, inputs, outputNames), returning
// one WasmTensor per requested output name, same order.
std::vector<WasmTensor> RunSession(const Session& session, const std::map<std::string, WasmTensor>& named_inputs) {
  val inputs_arr = val::array();
  for (const auto& name : session.input_names) {
    auto it = named_inputs.find(name);
    if (it == named_inputs.end()) throw std::runtime_error("RunSession: no value supplied for input " + name);
    inputs_arr.call<void>("push", TensorToVal(name, it->second));
  }
  val output_names_arr = val::array(session.output_names.begin(), session.output_names.end());

  val result = val::module_property("onnxDeployRunSession")(session.handle, inputs_arr, output_names_arr).await();
  std::vector<WasmTensor> outputs;
  outputs.reserve(session.output_names.size());
  for (size_t i = 0; i < session.output_names.size(); ++i) outputs.push_back(ValToTensor(result[i]));
  return outputs;
}

// Builds the named-input map for one decoder Run() call, exactly mirroring
// kv_cache_pipeline.h's RunDecoderStep dispatch (same input-name
// conventions), then harvests present.* into `cache` the same way
// HarvestPresentIntoCache does. Returns the logits tensor.
WasmTensor RunDecoderStep(const Session& session, const std::vector<int64_t>& step_input_ids,
                           const WasmTensor* encoder_hidden_states, size_t encoder_seq_len, size_t total_len,
                           std::map<std::string, WasmTensor>& cache) {
  std::map<std::string, WasmTensor> named_inputs;
  for (const auto& name : session.input_names) {
    if (name == "input_ids") {
      named_inputs[name] = MakeIdsTensor(step_input_ids);
    } else if (name == "attention_mask") {
      named_inputs[name] = MakeMaskTensor(total_len);
    } else if (name == "encoder_attention_mask") {
      named_inputs[name] = MakeMaskTensor(encoder_seq_len);
    } else if (name == "encoder_hidden_states") {
      if (!encoder_hidden_states) throw std::runtime_error("decoder graph wants encoder_hidden_states but no encoder ran");
      named_inputs[name] = *encoder_hidden_states;
    } else if (name.rfind("past_key_values.", 0) == 0) {
      auto it = cache.find(name);
      if (it == cache.end()) throw std::runtime_error("missing cache entry for " + name);
      named_inputs[name] = it->second;  // WasmTensor copies its (small) data vector -- no move-only constraint here
    } else {
      throw std::runtime_error("unrecognized decoder input: " + name);
    }
  }

  std::vector<WasmTensor> outputs = RunSession(session, named_inputs);

  static const std::string kPresentPrefix = "present.";
  static const std::string kPastPrefix = "past_key_values.";
  const WasmTensor* logits = nullptr;
  for (size_t i = 0; i < session.output_names.size(); ++i) {
    const std::string& name = session.output_names[i];
    if (name == "logits") {
      logits = &outputs[i];
    } else if (name.rfind(kPresentPrefix, 0) == 0) {
      cache[kPastPrefix + name.substr(kPresentPrefix.size())] = outputs[i];
    }
  }
  if (!logits) throw std::runtime_error("decoder graph has no 'logits' output");
  return *logits;
}

int64_t ArgmaxLastToken(const WasmTensor& logits) {
  size_t vocab = static_cast<size_t>(logits.shape.back());
  size_t seq = logits.shape.size() >= 2 ? static_cast<size_t>(logits.shape[logits.shape.size() - 2]) : 1;
  size_t offset = (seq - 1) * vocab;
  size_t best = 0;
  double best_val = -1e300;
  for (size_t v = 0; v < vocab; ++v) {
    double x = logits.data[offset + v];
    if (x > best_val) {
      best_val = x;
      best = v;
    }
  }
  return static_cast<int64_t>(best);
}

// Mirrors KvCachePipeline::Generate. `encoder_bytes` may be val::undefined()/
// val::null() for a decoder-only (causal LM) pipeline.
val Generate(val encoder_bytes, val decoder_bytes, val decoder_past_bytes, val input_ids_val, double max_new_tokens,
             double eos_token_id, double decoder_start_token_id) {
  bool is_seq2seq = !(encoder_bytes.isUndefined() || encoder_bytes.isNull());
  std::vector<double> input_ids_d = emscripten::vecFromJSArray<double>(input_ids_val);
  std::vector<int64_t> input_ids(input_ids_d.begin(), input_ids_d.end());

  Session decoder_session = CreateSession(decoder_bytes);
  Session decoder_past_session = CreateSession(decoder_past_bytes);

  WasmTensor encoder_hidden_states;
  size_t encoder_seq_len = 0;
  bool have_encoder_hidden_states = false;
  if (is_seq2seq) {
    Session encoder_session = CreateSession(encoder_bytes);
    encoder_seq_len = input_ids.size();
    std::map<std::string, WasmTensor> enc_inputs;
    for (const auto& name : encoder_session.input_names) {
      if (name == "input_ids") enc_inputs[name] = MakeIdsTensor(input_ids);
      else if (name == "attention_mask") enc_inputs[name] = MakeMaskTensor(input_ids.size());
      else throw std::runtime_error("unrecognized encoder input: " + name);
    }
    std::vector<WasmTensor> enc_outputs = RunSession(encoder_session, enc_inputs);
    encoder_hidden_states = enc_outputs.front();  // last_hidden_state is the encoder's only output
    have_encoder_hidden_states = true;
  }

  std::map<std::string, WasmTensor> cache;
  std::vector<int64_t> decoder_tokens =
      is_seq2seq ? std::vector<int64_t>{static_cast<int64_t>(decoder_start_token_id)} : input_ids;
  std::vector<int64_t> generated;
  bool use_past = false;
  int64_t eos = static_cast<int64_t>(eos_token_id);

  for (int64_t step = 0; step < static_cast<int64_t>(max_new_tokens); ++step) {
    std::vector<int64_t> step_input = use_past ? std::vector<int64_t>{decoder_tokens.back()} : decoder_tokens;
    size_t total_len = decoder_tokens.size();

    const Session& session = use_past ? decoder_past_session : decoder_session;
    WasmTensor logits = RunDecoderStep(session, step_input, have_encoder_hidden_states ? &encoder_hidden_states : nullptr,
                                        encoder_seq_len, total_len, cache);
    int64_t next_token = ArgmaxLastToken(logits);
    generated.push_back(next_token);
    decoder_tokens.push_back(next_token);
    use_past = true;

    if (eos_token_id >= 0 && next_token == eos) break;
  }

  return val::array(generated.begin(), generated.end());
}

}  // namespace

EMSCRIPTEN_BINDINGS(onnx_deploy_wasm) { emscripten::function("generate", &Generate); }
