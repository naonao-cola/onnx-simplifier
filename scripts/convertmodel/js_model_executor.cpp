#include "js_model_executor.h"

#if defined(__EMSCRIPTEN__) && defined(ONNXSIM_WASM_ORT_WEB)

#include <emscripten/val.h>

#include <bit>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "dlpack_bridge.h"

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

// Copy `len` bytes at `data` into a fresh JS-owned Uint8Array. Constructing
// `new Uint8Array(view)` from a view over the wasm heap copies the bytes into a
// new ArrayBuffer, so the result stays valid across the Asyncify suspend in
// Run() -- while suspended the wasm heap can grow (ALLOW_MEMORY_GROWTH=1) and
// any view still pointing at the old heap would be detached.
val RawToJsU8Copy(const uint8_t* data, size_t len) {
  val view = val(emscripten::typed_memory_view(len, data));
  return val::global("Uint8Array").new_(view);
}

val StringToJsU8Copy(const std::string& s) {
  return RawToJsU8Copy(reinterpret_cast<const uint8_t*>(s.data()), s.size());
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

// A ModelExecutor that evaluates each constant-folding sub-model with
// onnxruntime-web. Tensors cross the executor boundary as DLPack
// DLManagedTensors (see onnxsim.h / dlpack_bridge.h): inputs are borrowed for
// the call, and outputs are returned as freshly owned managed tensors. Across
// the wasm<->JS boundary the payload still has to be copied into JS-owned
// buffers (onnxsim's wasm heap and onnxruntime-web's are separate memories), but
// nothing is serialized to TensorProto -- the contract is ONNX dtype enums plus
// raw little-endian bytes, matching the built-in CppModelExecutor's dtype set.
struct JsModelExecutor : public ModelExecutor {
  std::vector<DLManagedTensorPtr> Run(
      const onnx::ModelProto& model,
      const std::vector<const DLManagedTensor*>& inputs) const override {
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

    // Inputs are positional w.r.t. model.graph().input(); recover each feed's
    // name from there (DLPack tensors carry no name), matching how the built-in
    // executor maps feeds to graph inputs.
    val js_inputs = val::array();
    for (size_t i = 0; i < inputs.size(); i++) {
      const DLTensor& t = inputs[i]->dl_tensor;
      int32_t onnx_dtype;
      if (!onnxsim::dlpack::TryDLToOnnx(t.dtype, &onnx_dtype)) {
        throw std::invalid_argument(
            "onnxruntime-web executor: unsupported input dtype (DLPack code=" +
            std::to_string(t.dtype.code) +
            ", bits=" + std::to_string(t.dtype.bits) + ")");
      }
      val obj = val::object();
      obj.set("name", i < static_cast<size_t>(model.graph().input_size())
                          ? model.graph().input(static_cast<int>(i)).name()
                          : std::string());
      obj.set("dataType", onnx_dtype);
      val dims = val::array();
      for (int32_t d = 0; d < t.ndim; d++) {
        dims.call<void>("push", static_cast<double>(t.shape[d]));
      }
      obj.set("dims", dims);
      const size_t nbytes =
          static_cast<size_t>(onnxsim::dlpack::NumElements(t.shape, t.ndim)) *
          onnxsim::dlpack::SizeOf(t.dtype);
      const uint8_t* base = static_cast<const uint8_t*>(t.data) + t.byte_offset;
      obj.set("data", RawToJsU8Copy(base, nbytes));
      js_inputs.call<void>("push", obj);
    }

    // Run onnxruntime-web and block on its Promise. val::await() unwinds the
    // wasm stack via Asyncify and resumes here once the Promise settles; a
    // rejected Promise surfaces as a C++ exception.
    val result = runner(js_model, js_inputs).await();

    // RunOps names the returned tensors positionally, in graph-output order, so
    // return them in exactly that order. Each output is rebuilt as a TensorProto
    // (dtype + dims + raw bytes) and handed to the DLPack boundary as an owning
    // managed tensor.
    std::vector<DLManagedTensorPtr> outputs;
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
      for (size_t d = 0; d < ndim; d++) {
        tp.add_dims(static_cast<int64_t>(dims[d].as<double>()));
      }
      tp.set_raw_data(JsU8ToString(t["data"]));
      outputs.emplace_back(
          onnxsim::dlpack::FromTensorProtoOwning(std::move(tp)));
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
