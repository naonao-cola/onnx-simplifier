#include <jni.h>
#include <onnxruntime_cxx_api.h>
#include <nnapi_provider_factory.h>
#include <android/NeuralNetworks.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
std::string ToString(JNIEnv* env, jstring value) {
  const char* chars = env->GetStringUTFChars(value, nullptr);
  std::string result(chars);
  env->ReleaseStringUTFChars(value, chars);
  return result;
}

std::string NnapiDevices() {
  uint32_t count = 0;
  if (ANeuralNetworks_getDeviceCount(&count) != ANEURALNETWORKS_NO_ERROR) {
    return "NNAPI devices unavailable";
  }
  std::string result;
  for (uint32_t i = 0; i < count; ++i) {
    ANeuralNetworksDevice* device = nullptr;
    const char* name = "unknown";
    int32_t type = -1;
    if (ANeuralNetworks_getDevice(i, &device) == ANEURALNETWORKS_NO_ERROR &&
        ANeuralNetworksDevice_getName(device, &name) == ANEURALNETWORKS_NO_ERROR &&
        ANeuralNetworksDevice_getType(device, &type) == ANEURALNETWORKS_NO_ERROR) {
      const char* type_name = type == ANEURALNETWORKS_DEVICE_GPU ? "GPU" :
                              type == ANEURALNETWORKS_DEVICE_ACCELERATOR ? "accelerator" :
                              type == ANEURALNETWORKS_DEVICE_CPU ? "CPU" : "other";
      result += std::string(name) + "[" + type_name + "] ";
    }
  }
  return result.empty() ? "no NNAPI devices" : result;
}

std::string RunReluOnQtiDsp(const std::array<float, 4>& input,
                            std::array<float, 4>& output) {
  const auto check = [](int status, const char* operation) {
    if (status != ANEURALNETWORKS_NO_ERROR) {
      throw std::runtime_error(std::string(operation) + " failed: " +
                               std::to_string(status));
    }
  };

  uint32_t count = 0;
  check(ANeuralNetworks_getDeviceCount(&count), "NNAPI device enumeration");
  ANeuralNetworksDevice* dsp = nullptr;
  for (uint32_t i = 0; i < count; ++i) {
    ANeuralNetworksDevice* device = nullptr;
    const char* name = nullptr;
    if (ANeuralNetworks_getDevice(i, &device) == ANEURALNETWORKS_NO_ERROR &&
        ANeuralNetworksDevice_getName(device, &name) == ANEURALNETWORKS_NO_ERROR &&
        name != nullptr && std::string(name) == "qti-dsp") {
      dsp = device;
      break;
    }
  }
  if (dsp == nullptr) throw std::runtime_error("NNAPI device qti-dsp is unavailable");

  ANeuralNetworksModel* raw_model = nullptr;
  check(ANeuralNetworksModel_create(&raw_model), "NNAPI model creation");
  std::unique_ptr<ANeuralNetworksModel, decltype(&ANeuralNetworksModel_free)> model(
      raw_model, ANeuralNetworksModel_free);
  const uint32_t dimensions[] = {4};
  const ANeuralNetworksOperandType tensor_type{
      ANEURALNETWORKS_TENSOR_FLOAT32, 1, dimensions, 0.0f, 0};
  check(ANeuralNetworksModel_addOperand(model.get(), &tensor_type), "add input operand");
  check(ANeuralNetworksModel_addOperand(model.get(), &tensor_type), "add output operand");
  const uint32_t operation_inputs[] = {0};
  const uint32_t operation_outputs[] = {1};
  const uint32_t model_inputs[] = {0};
  const uint32_t model_outputs[] = {1};
  check(ANeuralNetworksModel_addOperation(model.get(), ANEURALNETWORKS_RELU, 1,
                                           operation_inputs, 1, operation_outputs),
        "add RELU operation");
  check(ANeuralNetworksModel_identifyInputsAndOutputs(model.get(), 1,
                                                       model_inputs, 1,
                                                       model_outputs),
        "identify model inputs and outputs");
  check(ANeuralNetworksModel_relaxComputationFloat32toFloat16(model.get(), true),
        "enable float16 relaxation");
  check(ANeuralNetworksModel_finish(model.get()), "finish NNAPI model");

  bool supported = false;
  const ANeuralNetworksDevice* devices[] = {dsp};
  check(ANeuralNetworksModel_getSupportedOperationsForDevices(model.get(), devices, 1,
                                                               &supported),
        "query qti-dsp operation support");
  if (!supported) throw std::runtime_error("qti-dsp does not support float32 RELU");

  ANeuralNetworksCompilation* raw_compilation = nullptr;
  check(ANeuralNetworksCompilation_createForDevices(model.get(), devices, 1,
                                                     &raw_compilation),
        "compile explicitly for qti-dsp");
  std::unique_ptr<ANeuralNetworksCompilation,
                  decltype(&ANeuralNetworksCompilation_free)> compilation(
      raw_compilation, ANeuralNetworksCompilation_free);
  check(ANeuralNetworksCompilation_finish(compilation.get()),
        "finish qti-dsp compilation");

  ANeuralNetworksExecution* raw_execution = nullptr;
  check(ANeuralNetworksExecution_create(compilation.get(), &raw_execution),
        "create qti-dsp execution");
  std::unique_ptr<ANeuralNetworksExecution, decltype(&ANeuralNetworksExecution_free)>
      execution(raw_execution, ANeuralNetworksExecution_free);
  check(ANeuralNetworksExecution_setInput(execution.get(), 0, nullptr, input.data(),
                                           input.size() * sizeof(float)),
        "bind qti-dsp input");
  check(ANeuralNetworksExecution_setOutput(execution.get(), 0, nullptr, output.data(),
                                            output.size() * sizeof(float)),
        "bind qti-dsp output");
  check(ANeuralNetworksExecution_compute(execution.get()), "execute qti-dsp RELU");
  return "qti-dsp[explicit]";
}
}  // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_org_onnxsim_androidtest_MainActivity_runModel(JNIEnv* env, jclass,
                                                    jstring original_path,
                                                    jstring simplified_path,
                                                    jstring input_path,
                                                    jstring output_path,
                                                    jstring target_value,
                                                    jstring qnn_library_path) {
  std::string device_diagnostics;
  try {
    const auto original_model = ToString(env, original_path);
    const auto simplified_model = ToString(env, simplified_path);
    const auto input_file_path = ToString(env, input_path);
    const auto output_file_path = ToString(env, output_path);
    const auto target = ToString(env, target_value);
    const auto qnn_library = ToString(env, qnn_library_path);

    std::ifstream input_file(input_file_path, std::ios::binary);
    if (!input_file) throw std::runtime_error("cannot open input tensor");
    std::vector<char> raw_input((std::istreambuf_iterator<char>(input_file)), {});
    if (raw_input.size() < 4 * sizeof(float) || raw_input.size() % sizeof(float) != 0) {
      throw std::runtime_error("input must contain float32 values (at least four)");
    }
    std::vector<float> input(raw_input.size() / sizeof(float));
    std::memcpy(input.data(), raw_input.data(), raw_input.size());

    if (target == "nnapi-dsp-direct") {
      std::array<float, 4> dsp_input{};
      std::copy_n(input.begin(), dsp_input.size(), dsp_input.begin());
      std::array<float, 4> output{};
      device_diagnostics = RunReluOnQtiDsp(dsp_input, output);
      std::ofstream output_file(output_file_path, std::ios::binary);
      output_file.write(reinterpret_cast<const char*>(output.data()), sizeof(output));
      if (!output_file) throw std::runtime_error("could not write DSP output tensor");
      for (size_t i = 0; i < dsp_input.size(); ++i) {
        const float expected = dsp_input[i] < 0.0f ? 0.0f : dsp_input[i];
        if (std::abs(output[i] - expected) > 1e-2f) {
          throw std::runtime_error("qti-dsp RELU output differs from reference");
        }
      }
      return env->NewStringUTF(("PASS nnapi-dsp-direct RELU reference " +
                                device_diagnostics).c_str());
    }

    Ort::Env ort_env(ORT_LOGGING_LEVEL_WARNING, "onnxsim-android-app-test");
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    if (target == "qnn-htp" || target == "qnn-gpu") {
      ort_env.RegisterExecutionProviderLibrary("QNNExecutionProvider", qnn_library);
      std::vector<Ort::ConstEpDevice> qnn_devices;
      const auto wanted_type = target == "qnn-htp" ? OrtHardwareDeviceType_NPU
                                                   : OrtHardwareDeviceType_GPU;
      for (const auto& device : ort_env.GetEpDevices()) {
        if (device.EpName() == std::string("QNNExecutionProvider")) {
          device_diagnostics += std::string(device.EpName()) + "/" +
                                std::to_string(static_cast<int>(device.Device().Type())) + " ";
          if (device.Device().Type() == wanted_type) qnn_devices.push_back(device);
        }
      }
      if (qnn_devices.empty()) {
        throw std::runtime_error("QNN EP exposed no device matching backend; devices: " +
                                 device_diagnostics);
      }
      options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
      std::unordered_map<std::string, std::string> qnn_options{
          {"backend_type", target == "qnn-htp" ? "htp" : "gpu"}};
      if (target == "qnn-htp") {
        qnn_options["enable_htp_fp16_precision"] = "1";
        qnn_options["offload_graph_io_quantization"] = "0";
      }
      options.AppendExecutionProvider_V2(ort_env, qnn_devices, qnn_options);
    } else if (target == "nnapi-no-cpu") {
      device_diagnostics = NnapiDevices();
      options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
      Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Nnapi(
          options.GetUnowned(), NNAPI_FLAG_USE_FP16 | NNAPI_FLAG_CPU_DISABLED));
    } else if (target != "cpu") {
      throw std::runtime_error("unknown target: " + target);
    }

    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const auto run_one = [&](const std::string& model, Ort::SessionOptions& run_options) {
      Ort::Session session(ort_env, model.c_str(), run_options);
      if (session.GetInputCount() != 1 || session.GetOutputCount() != 1) {
        throw std::runtime_error("only single-input, single-output models are supported");
      }
      Ort::AllocatorWithDefaultOptions allocator;
      auto input_name = session.GetInputNameAllocated(0, allocator);
      auto output_name = session.GetOutputNameAllocated(0, allocator);
      auto input_info = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
      auto shape = input_info.GetShape();
      for (auto& dimension : shape) {
        if (dimension < 0) dimension = 1;
      }
      size_t expected_input_count = 1;
      for (const auto dimension : shape) expected_input_count *= static_cast<size_t>(dimension);
      if (input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
          expected_input_count != input.size()) {
        throw std::runtime_error("model input must be fixed-shape float32 matching input.f32");
      }
      auto tensor = Ort::Value::CreateTensor<float>(memory, input.data(), input.size(),
                                                     shape.data(), shape.size());
      const char* input_names[] = {input_name.get()};
      const char* output_names[] = {output_name.get()};
      auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, &tensor, 1,
                                 output_names, 1);
      if (outputs.size() != 1 || !outputs[0].IsTensor() ||
          outputs[0].GetTensorTypeAndShapeInfo().GetElementType() !=
              ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw std::runtime_error("model must return one float32 tensor");
      }
      auto output_info = outputs[0].GetTensorTypeAndShapeInfo();
      const float* data = outputs[0].GetTensorData<float>();
      return std::vector<float>(data, data + output_info.GetElementCount());
    };
    Ort::SessionOptions cpu_options;
    cpu_options.SetIntraOpNumThreads(1);
    const auto original_cpu_values = run_one(original_model, cpu_options);
    const auto simplified_cpu_values = run_one(simplified_model, cpu_options);
    const auto original_values = run_one(original_model, options);
    const auto values = run_one(simplified_model, options);
    const auto compare_outputs = [](const std::vector<float>& actual,
                                    const std::vector<float>& expected,
                                    const char* description) {
      if (actual.size() != expected.size() || actual.empty()) {
        throw std::runtime_error(std::string(description) + " has an unexpected output size");
      }
      float max_abs_error = 0.0f;
      size_t actual_top = 0;
      size_t expected_top = 0;
      for (size_t i = 0; i < actual.size(); ++i) {
        max_abs_error = std::max(max_abs_error, std::abs(actual[i] - expected[i]));
        if (actual[i] > actual[actual_top]) actual_top = i;
        if (expected[i] > expected[expected_top]) expected_top = i;
      }
      if (max_abs_error > 0.05f || actual_top != expected_top) {
        throw std::runtime_error(std::string(description) + " differs from CPU reference (max_abs=" +
                                 std::to_string(max_abs_error) + ", top1=" +
                                 std::to_string(actual_top) + "/" +
                                 std::to_string(expected_top) + ")");
      }
      return max_abs_error;
    };
    compare_outputs(simplified_cpu_values, original_cpu_values, "simplified CPU output");
    const float original_error = compare_outputs(original_values, original_cpu_values,
                                                 "original hardware output");
    const float simplified_error = compare_outputs(values, simplified_cpu_values,
                                                   "simplified hardware output");
    std::ofstream output_file(output_file_path, std::ios::binary);
    output_file.write(reinterpret_cast<const char*>(values.data()), sizeof(float) * values.size());
    if (!output_file) throw std::runtime_error("could not write model output");
    return env->NewStringUTF(("PASS " + target + " original/simplified vs CPU top1; max_abs=" +
                              std::to_string(std::max(original_error, simplified_error)) + " " +
                              device_diagnostics).c_str());
  } catch (const Ort::Exception& error) {
    return env->NewStringUTF(("FAIL " + ToString(env, target_value) + ": " + error.what() +
                              "; QNN devices=" + device_diagnostics).c_str());
  } catch (const std::exception& error) {
    return env->NewStringUTF(("FAIL " + ToString(env, target_value) + ": " + error.what() +
                              "; QNN devices=" + device_diagnostics).c_str());
  }
}
