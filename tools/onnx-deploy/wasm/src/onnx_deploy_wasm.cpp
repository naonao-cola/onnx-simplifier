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

// Module.onnxDeployCreateSession/onnxDeployRunSession never reject their
// returned Promise (see ort_web_runtime.mjs's errorResult) -- they resolve
// to `{ __onnxDeployError: message }` on failure instead, because a
// rejection surfacing through val::await() has been observed to crash the
// process instead of reliably becoming a catchable C++ exception. Every
// .await() result is checked with this immediately afterward, so a JS-side
// failure becomes a normal, synchronously-thrown C++ exception -- ordinary
// propagation from there, no Asyncify unwind involved.
void ThrowIfJsError(const val& result) {
  val error = result["__onnxDeployError"];
  if (!error.isUndefined()) throw std::runtime_error(error.as<std::string>());
}

// Awaits Module.onnxDeployCreateSession(bytes, executionProviders) and
// records the returned handle + the graph's actual input/output names
// (mirrors kv_cache_pipeline.h's detail::InputNames/OutputNames, which read
// them off a native Ort::Session -- here they come back from the JS-side
// ort.InferenceSession instead). `execution_providers` is a JS array of
// onnxruntime-web EP name strings (e.g. ["webgpu"]); pass val::array() (or
// leave it default) for onnxruntime-web's own default (the wasm/CPU
// backend). See ort_web_runtime.mjs's onnxDeployCreateSession for the other
// half of this contract, including why a missing WebGPU host fails cleanly
// here rather than crashing.
Session CreateSession(const val& bytes, const val& execution_providers) {
  val result = val::module_property("onnxDeployCreateSession")(bytes, execution_providers).await();
  ThrowIfJsError(result);
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
  ThrowIfJsError(result);
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

// ---------------------------------------------------------------------
// Fixed-buffer ("static") KV cache pipeline -- WASM port of
// onnx_deploy::StaticKvCachePipeline's algorithm (see
// ../../include/onnx_deploy/static_kv_cache_pipeline.h), for the
// onnxsim.export_causal_lm_static_cache() export shape (prefill.onnx +
// decode.onnx, decoder-only causal LMs only). Unlike the native pipeline,
// this can't bind an output back onto the same buffer it read as an input
// (Ort::IoBinding's buffer-aliasing trick) -- every RunSession call crosses
// the JS/wasm Asyncify boundary through a serialized WasmTensor, which is
// always a fresh copy on both sides -- so there is no in-place-write
// optimization to port here; this is a functionally-equivalent, not
// performance-equivalent, port (still avoids the *growing*-buffer
// reallocation-and-copy the Layer 1 WASM port has, since every buffer here
// stays exactly max_cache_len long, but each step still copies that whole
// fixed-size buffer across the JS boundary, same as any other tensor).
//
// Also unlike the native pipeline (which reads num_kv_heads/head_dim off
// decode.onnx's own declared input shape via Ort::Session::GetInputTypeInfo),
// onnxDeployCreateSession's contract here only reports input/output *names*,
// not shapes (see ort_web_runtime.mjs) -- extending it to report shapes too
// is a reasonable follow-up, not done here. generateStatic() instead takes
// num_kv_heads/head_dim as explicit parameters, the same "caller already
// knows the architecture" pattern max_cache_len itself already uses.

WasmTensor MakeCachePositionTensor(int64_t start, size_t len) {
  WasmTensor t;
  t.dtype = "int64";
  t.shape = {static_cast<double>(len)};
  t.data.resize(len);
  for (size_t i = 0; i < len; ++i) t.data[i] = static_cast<double>(start) + static_cast<double>(i);
  return t;
}

// Full-buffer-length mask, 1s for real content written so far, 0s for the
// not-yet-written tail -- same convention transformers.StaticCache's own
// get_mask_sizes() expects (see onnxsim/transformers_export.py's own
// docstring, and static_kv_cache_pipeline.h's RunStep, which this mirrors).
WasmTensor MakeStaticMaskTensor(int64_t max_cache_len, int64_t valid_len) {
  WasmTensor t;
  t.dtype = "int64";
  t.shape = {1, static_cast<double>(max_cache_len)};
  t.data.assign(static_cast<size_t>(max_cache_len), 0.0);
  for (int64_t i = 0; i < valid_len && i < max_cache_len; ++i) t.data[static_cast<size_t>(i)] = 1.0;
  return t;
}

WasmTensor ZeroKvTensor(int64_t num_kv_heads, int64_t max_cache_len, int64_t head_dim) {
  WasmTensor t;
  t.dtype = "float32";
  t.shape = {1, static_cast<double>(num_kv_heads), static_cast<double>(max_cache_len), static_cast<double>(head_dim)};
  t.data.assign(static_cast<size_t>(num_kv_heads * max_cache_len * head_dim), 0.0);
  return t;
}

// Runs one prefill or decode step against `kv` (keyed "past_key.{i}"/
// "past_value.{i}", updated in place from this call's present_key.{i}/
// present_value.{i} outputs -- see this section's own header comment on why
// that's a copy here, unlike the native pipeline's true buffer aliasing).
// Returns "logits".
WasmTensor RunStaticStep(const Session& session, const std::vector<int64_t>& step_input_ids, int64_t cache_start,
                          int64_t max_cache_len, std::map<std::string, WasmTensor>& kv) {
  std::map<std::string, WasmTensor> named_inputs;
  for (const auto& name : session.input_names) {
    if (name == "input_ids") {
      named_inputs[name] = MakeIdsTensor(step_input_ids);
    } else if (name == "cache_position") {
      named_inputs[name] = MakeCachePositionTensor(cache_start, step_input_ids.size());
    } else if (name == "attention_mask") {
      named_inputs[name] = MakeStaticMaskTensor(max_cache_len, cache_start + static_cast<int64_t>(step_input_ids.size()));
    } else if (name.rfind("past_key.", 0) == 0 || name.rfind("past_value.", 0) == 0) {
      auto it = kv.find(name);
      if (it == kv.end()) throw std::runtime_error("missing cache entry for " + name);
      named_inputs[name] = it->second;
    } else {
      throw std::runtime_error("unrecognized static-cache decoder input: " + name);
    }
  }

  std::vector<WasmTensor> outputs = RunSession(session, named_inputs);

  static const std::string kPresentKeyPrefix = "present_key.";
  static const std::string kPresentValuePrefix = "present_value.";
  const WasmTensor* logits = nullptr;
  for (size_t i = 0; i < session.output_names.size(); ++i) {
    const std::string& name = session.output_names[i];
    if (name == "logits") {
      logits = &outputs[i];
    } else if (name.rfind(kPresentKeyPrefix, 0) == 0) {
      kv["past_key." + name.substr(kPresentKeyPrefix.size())] = outputs[i];
    } else if (name.rfind(kPresentValuePrefix, 0) == 0) {
      kv["past_value." + name.substr(kPresentValuePrefix.size())] = outputs[i];
    }
  }
  if (!logits) throw std::runtime_error("static-cache decoder graph has no 'logits' output");
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
// val::null() for a decoder-only (causal LM) pipeline. `execution_providers`
// is a JS array of onnxruntime-web EP name strings applied to every session
// (e.g. ["webgpu"]; an empty array uses onnxruntime-web's own default). See
// CreateSession above and ort_web_runtime.mjs's onnxDeployCreateSession for
// what this actually does and how it fails when the requested EP isn't
// available on the host (e.g. no navigator.gpu for "webgpu").
val GenerateImpl(val encoder_bytes, val decoder_bytes, val decoder_past_bytes, val input_ids_val,
                  double max_new_tokens, double eos_token_id, double decoder_start_token_id,
                  val execution_providers) {
  bool is_seq2seq = !(encoder_bytes.isUndefined() || encoder_bytes.isNull());
  std::vector<double> input_ids_d = emscripten::vecFromJSArray<double>(input_ids_val);
  std::vector<int64_t> input_ids(input_ids_d.begin(), input_ids_d.end());

  Session decoder_session = CreateSession(decoder_bytes, execution_providers);
  Session decoder_past_session = CreateSession(decoder_past_bytes, execution_providers);

  WasmTensor encoder_hidden_states;
  size_t encoder_seq_len = 0;
  bool have_encoder_hidden_states = false;
  if (is_seq2seq) {
    Session encoder_session = CreateSession(encoder_bytes, execution_providers);
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

// A C++ exception thrown after an Asyncify unwind (i.e. anywhere past the
// first val::await()) does NOT reliably surface as a rejected
// Module.generate() Promise on its own -- observed in practice as an
// uncaught exception that crashes the whole process instead (e.g.
// requesting the "webgpu" EP with no navigator.gpu available). Catching
// here and explicitly returning a rejected JS Promise works around it: a
// `val` that is itself a thenable, returned from an Asyncify-wrapped
// exported function, is adopted by the outer Promise Emscripten's runtime
// already builds around this call -- standard Promise resolution
// semantics, not a wasm/embind-specific trick -- so the rejection reaches
// the actual caller of Module.generate() normally. See
// ../test/run_test.mjs's webgpu-unavailable case, which is exactly what
// this fixes.
val Generate(val encoder_bytes, val decoder_bytes, val decoder_past_bytes, val input_ids_val, double max_new_tokens,
             double eos_token_id, double decoder_start_token_id, val execution_providers) {
  try {
    return GenerateImpl(encoder_bytes, decoder_bytes, decoder_past_bytes, input_ids_val, max_new_tokens,
                         eos_token_id, decoder_start_token_id, execution_providers);
  } catch (const std::exception& e) {
    return val::global("Promise").call<val>("reject", val(std::string(e.what())));
  } catch (...) {
    return val::global("Promise").call<val>("reject", val(std::string("unknown error in onnx_deploy_wasm generate()")));
  }
}

// Mirrors StaticKvCachePipeline::Generate. `execution_providers` is the
// same JS array of onnxruntime-web EP name strings Generate() above takes.
val GenerateStaticImpl(val prefill_bytes, val decode_bytes, val input_ids_val, double max_new_tokens,
                        double eos_token_id, double max_cache_len, double num_kv_heads, double head_dim,
                        val execution_providers) {
  std::vector<double> input_ids_d = emscripten::vecFromJSArray<double>(input_ids_val);
  std::vector<int64_t> input_ids(input_ids_d.begin(), input_ids_d.end());
  int64_t max_cache = static_cast<int64_t>(max_cache_len);
  int64_t kv_heads = static_cast<int64_t>(num_kv_heads);
  int64_t dim = static_cast<int64_t>(head_dim);
  int64_t eos = static_cast<int64_t>(eos_token_id);

  Session prefill_session = CreateSession(prefill_bytes, execution_providers);
  Session decode_session = CreateSession(decode_bytes, execution_providers);

  int64_t num_layers = 0;
  for (const auto& name : decode_session.input_names)
    if (name.rfind("past_key.", 0) == 0) ++num_layers;
  if (num_layers == 0) throw std::runtime_error("decode session has no past_key.* inputs");

  std::map<std::string, WasmTensor> kv;
  for (int64_t i = 0; i < num_layers; ++i) {
    kv["past_key." + std::to_string(i)] = ZeroKvTensor(kv_heads, max_cache, dim);
    kv["past_value." + std::to_string(i)] = ZeroKvTensor(kv_heads, max_cache, dim);
  }

  std::vector<int64_t> generated;
  WasmTensor logits = RunStaticStep(prefill_session, input_ids, /*cache_start=*/0, max_cache, kv);
  int64_t next_token = ArgmaxLastToken(logits);
  generated.push_back(next_token);
  int64_t write_pos = static_cast<int64_t>(input_ids.size());

  if (!(eos_token_id >= 0 && next_token == eos)) {
    for (int64_t step = 1; step < static_cast<int64_t>(max_new_tokens); ++step) {
      logits = RunStaticStep(decode_session, {next_token}, write_pos, max_cache, kv);
      next_token = ArgmaxLastToken(logits);
      generated.push_back(next_token);
      write_pos += 1;
      if (eos_token_id >= 0 && next_token == eos) break;
    }
  }

  return val::array(generated.begin(), generated.end());
}

// Same crash-avoidance wrapping as Generate above -- see its own comment.
val GenerateStatic(val prefill_bytes, val decode_bytes, val input_ids_val, double max_new_tokens,
                    double eos_token_id, double max_cache_len, double num_kv_heads, double head_dim,
                    val execution_providers) {
  try {
    return GenerateStaticImpl(prefill_bytes, decode_bytes, input_ids_val, max_new_tokens, eos_token_id, max_cache_len,
                              num_kv_heads, head_dim, execution_providers);
  } catch (const std::exception& e) {
    return val::global("Promise").call<val>("reject", val(std::string(e.what())));
  } catch (...) {
    return val::global("Promise")
        .call<val>("reject", val(std::string("unknown error in onnx_deploy_wasm generateStatic()")));
  }
}

}  // namespace

EMSCRIPTEN_BINDINGS(onnx_deploy_wasm) {
  emscripten::function("generate", &Generate);
  emscripten::function("generateStatic", &GenerateStatic);
}
