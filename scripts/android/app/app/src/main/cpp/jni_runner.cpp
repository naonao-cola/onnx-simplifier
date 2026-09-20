#include <jni.h>
#include <onnxruntime_cxx_api.h>
#include <nnapi_provider_factory.h>
#include <android/NeuralNetworks.h>

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
    if (raw_input.size() != 4 * sizeof(float)) {
      throw std::runtime_error("invalid float32 input tensor");
    }
    std::array<float, 4> input{};
    std::memcpy(input.data(), raw_input.data(), sizeof(input));

    if (target == "nnapi-dsp-direct") {
      std::array<float, 4> output{};
      device_diagnostics = RunReluOnQtiDsp(input, output);
      std::ofstream output_file(output_file_path, std::ios::binary);
      output_file.write(reinterpret_cast<const char*>(output.data()), sizeof(output));
      if (!output_file) throw std::runtime_error("could not write DSP output tensor");
      for (size_t i = 0; i < input.size(); ++i) {
        const float expected = input[i] < 0.0f ? 0.0f : input[i];
        if (std::abs(output[i] - expected) > 1e-5f) {
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

    const std::array<int64_t, 2> shape{1, 4};
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const char* input_names[] = {"X"};
    const char* output_names[] = {"Y"};
    const auto run_one = [&](const std::string& model) {
      Ort::Session session(ort_env, model.c_str(), options);
      auto tensor = Ort::Value::CreateTensor<float>(memory, input.data(), input.size(),
                                                     shape.data(), shape.size());
      auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, &tensor, 1,
                                 output_names, 1);
      if (outputs.size() != 1 || !outputs[0].IsTensor() ||
          outputs[0].GetTensorTypeAndShapeInfo().GetElementCount() != input.size()) {
        throw std::runtime_error("unexpected model output");
      }
      const float* data = outputs[0].GetTensorData<float>();
      return std::vector<float>(data, data + input.size());
    };
    const auto original_values = run_one(original_model);
    const auto values = run_one(simplified_model);
    std::ofstream output_file(output_file_path, std::ios::binary);
    output_file.write(reinterpret_cast<const char*>(values.data()), sizeof(float) * input.size());
    if (!output_file) throw std::runtime_error("could not write model output");
    for (size_t i = 0; i < input.size(); ++i) {
      const float expected = input[i] < 0.0f ? 0.0f : input[i];
      if (std::abs(values[i] - expected) > 1e-5f ||
          std::abs(original_values[i] - values[i]) > 1e-5f) {
        throw std::runtime_error("original/simplified output differs from expected Relu values");
      }
    }
    return env->NewStringUTF(("PASS " + target + " original/simplified reference " +
                              device_diagnostics).c_str());
  } catch (const Ort::Exception& error) {
    return env->NewStringUTF(("FAIL " + ToString(env, target_value) + ": " + error.what() +
                              "; QNN devices=" + device_diagnostics).c_str());
  } catch (const std::exception& error) {
    return env->NewStringUTF(("FAIL " + ToString(env, target_value) + ": " + error.what() +
                              "; QNN devices=" + device_diagnostics).c_str());
  }
}
