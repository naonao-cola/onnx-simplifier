#include "js_model_executor.h"

#if defined(__EMSCRIPTEN__) && defined(ONNXSIM_WASM_ORT_WEB)

#include <emscripten/val.h>

#include <bit>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

// wasm is always little-endian; the (de)serialization below memcpy's typed data
// to/from raw little-endian bytes, so bail loudly if that ever stops holding.
static_assert(std::endian::native == std::endian::little,
              "the onnxruntime-web executor assumes a little-endian target");

namespace {

using emscripten::val;

// The page registers its onnxruntime-web runner on the Emscripten Module under
// this name (see ort_executor.mjs / worker.js). Signature, in JS:
//   async (modelBytes: Uint8Array, inputs: InTensor[]) => { [name]: OutTensor }
// where InTensor  = { name, dataType, dims: number[], data: Uint8Array }
//   and OutTensor = { dataType, dims: number[], data: Uint8Array }.
// `dataType` is the ONNX TensorProto.DataType enum value; `data` is the tensor
// content as raw little-endian bytes of the element type.
constexpr const char* kRunnerProp = "onnxsimOrtWebRun";

// Copy `s` into a fresh JS-owned Uint8Array. Constructing `new Uint8Array(view)`
// from a view over the wasm heap copies the bytes into a new ArrayBuffer, so
// the result stays valid across the Asyncify suspend in _Run -- while suspended
// the wasm heap can grow (ALLOW_MEMORY_GROWTH=1) and any view still pointing at
// the old heap would be detached.
val StringToJsU8Copy(const std::string& s) {
  val view = val(emscripten::typed_memory_view(
      s.size(), reinterpret_cast<const uint8_t*>(s.data())));
  return val::global("Uint8Array").new_(view);
}

// Copy a JS Uint8Array back into a std::string. Called after the await, so the
// destination view is created against the current (post-suspend) heap.
std::string JsU8ToString(const val& u8) {
  const size_t len = u8["length"].as<size_t>();
  std::string out;
  out.resize(len);
  val dest = val(emscripten::typed_memory_view(
      len, reinterpret_cast<uint8_t*>(out.data())));
  dest.call<void>("set", u8);
  return out;
}

// Serialize a TensorProto's element data to raw little-endian bytes. Mirrors the
// dtype coverage of the built-in CppModelExecutor's TensorProtoToTensor so the
// two executors accept the same models. onnx packs the sub-32-bit integer and
// bool types into int32_data, so narrow each element to its true width.
std::string TensorProtoRawBytes(const onnx::TensorProto& t) {
  if (t.has_raw_data()) {
    return t.raw_data();
  }
  std::string raw;
#define CASE_DTYPE(onnx_dtype, storage_dtype, cpp_type)                  \
  case onnx::TensorProto::onnx_dtype: {                                  \
    std::vector<cpp_type> vec;                                           \
    vec.reserve(t.storage_dtype##_data_size());                         \
    for (const auto& x : t.storage_dtype##_data()) {                    \
      vec.push_back(static_cast<cpp_type>(x));                           \
    }                                                                    \
    raw.assign(reinterpret_cast<const char*>(vec.data()),               \
               vec.size() * sizeof(cpp_type));                          \
    break;                                                              \
  }
  switch (t.data_type()) {
    CASE_DTYPE(FLOAT, float, float)
    CASE_DTYPE(DOUBLE, double, double)
    CASE_DTYPE(INT64, int64, int64_t)
    CASE_DTYPE(UINT64, uint64, uint64_t)
    CASE_DTYPE(INT32, int32, int32_t)
    CASE_DTYPE(UINT8, int32, uint8_t)
    CASE_DTYPE(INT8, int32, int8_t)
    CASE_DTYPE(UINT16, int32, uint16_t)
    CASE_DTYPE(INT16, int32, int16_t)
    CASE_DTYPE(BOOL, int32, int8_t)
    default:
      throw std::invalid_argument("onnxruntime-web executor: unsupported input "
                                  "dtype " +
                                  std::to_string(t.data_type()));
  }
#undef CASE_DTYPE
  return raw;
}

struct JsModelExecutor : public ModelExecutor {
  std::vector<onnx::TensorProto> _Run(
      const onnx::ModelProto& model,
      const std::vector<onnx::TensorProto>& inputs) const override {
    // Reach the runner the page registered on the Module.
    val runner = val::module_property(kRunnerProp);
    if (runner.isUndefined() || runner.isNull()) {
      throw std::runtime_error(
          "onnxruntime-web executor: Module." + std::string(kRunnerProp) +
          " is not set. The hosting page must register an onnxruntime-web "
          "runner (see ort_executor.mjs) before constant folding runs.");
    }

    // Marshal the sub-model and its feeds into JS-owned buffers *before* the
    // await, so nothing points into the wasm heap while it is suspended.
    const std::string model_str = model.SerializeAsString();
    val js_model = StringToJsU8Copy(model_str);

    val js_inputs = val::array();
    for (const auto& inp : inputs) {
      const std::string raw = TensorProtoRawBytes(inp);
      val obj = val::object();
      obj.set("name", inp.name());
      obj.set("dataType", static_cast<int>(inp.data_type()));
      val dims = val::array();
      for (int i = 0; i < inp.dims_size(); i++) {
        dims.call<void>("push", static_cast<double>(inp.dims(i)));
      }
      obj.set("dims", dims);
      obj.set("data", StringToJsU8Copy(raw));
      js_inputs.call<void>("push", obj);
    }

    // Run onnxruntime-web and block on its Promise. val::await() unwinds the
    // wasm stack via Asyncify and resumes here once the Promise settles; a
    // rejected Promise surfaces as a C++ exception.
    val result = runner(js_model, js_inputs).await();

    // RunOps names the returned tensors positionally, in graph-output order, so
    // return them in exactly that order (their own names are overwritten).
    std::vector<onnx::TensorProto> outputs;
    outputs.reserve(model.graph().output_size());
    for (const auto& out_vi : model.graph().output()) {
      val t = result[out_vi.name()];
      if (t.isUndefined() || t.isNull()) {
        throw std::runtime_error(
            "onnxruntime-web executor: runner did not return output '" +
            out_vi.name() + "'");
      }
      onnx::TensorProto tp;
      tp.set_data_type(
          static_cast<onnx::TensorProto::DataType>(t["dataType"].as<int>()));
      val dims = t["dims"];
      const size_t ndim = dims["length"].as<size_t>();
      for (size_t i = 0; i < ndim; i++) {
        tp.add_dims(static_cast<int64_t>(dims[i].as<double>()));
      }
      tp.set_raw_data(JsU8ToString(t["data"]));
      outputs.push_back(std::move(tp));
    }
    return outputs;
  }
};

}  // namespace

std::shared_ptr<const ModelExecutor> GetJsModelExecutor() {
  static std::shared_ptr<const ModelExecutor> executor =
      std::make_shared<JsModelExecutor>();
  return executor;
}

#endif  // __EMSCRIPTEN__ && ONNXSIM_WASM_ORT_WEB
